from collections.abc import Callable
from typing import Literal

import torch
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cs336_alignment.checkpoint import get_response_log_probs, tokenize_prompt_and_output


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    rewards = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths, strict=True)
    ]

    raw_rewards = torch.tensor([item["reward"] for item in rewards])
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_format_reward": torch.tensor(
            [item["format_reward"] for item in rewards]
        ).mean().item(),
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    if group_size <= 0 or raw_rewards.numel() % group_size:
        raise ValueError("raw_rewards must divide evenly into positive-size groups.")

    grouped_rewards = raw_rewards.reshape(-1, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    group_stds = grouped_rewards.std(dim=1, keepdim=True)
    if baseline == "mean":
        advantages = grouped_rewards - group_means
    elif baseline == "none":
        advantages = grouped_rewards
    else:
        raise ValueError(f"Unsupported baseline: {baseline}")

    if advantage_normalizer == "std":
        advantages = advantages / (group_stds + advantage_eps)
    elif advantage_normalizer != "none":
        raise ValueError(f"Unsupported advantage normalizer: {advantage_normalizer}")
    advantages = advantages.reshape(-1)
    return advantages, {
        "mean_reward": raw_rewards.mean().item(),
        "mean_group_std": group_stds.mean().item(),
    }


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if importance_reweighting_method != "none":
        raise NotImplementedError("Only on-policy loss without reweighting is supported.")
    del old_log_probs, cliprange, response_mask
    return -raw_rewards_or_advantages.reshape(-1, 1) * policy_log_probs, {}


# A microbatch is a memory-sized subset of rollout responses and may contain multiple GRPO groups.
def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    masked_losses = per_token_policy_gradient_loss * mask
    if loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError("normalization_constant is required for constant normalization.")
        return masked_losses.sum() / normalization_constant
    sequence_losses = masked_losses.sum(dim=1) / mask.sum(dim=1)
    return sequence_losses.mean()


def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    if gradient_accumulation_steps <= 0 or len(rollout_responses) % gradient_accumulation_steps:
        raise ValueError("rollout batch size must divide evenly into positive microbatches.")

    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )
    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )
    microbatch_size = len(rollout_responses) // gradient_accumulation_steps
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    microbatch_losses = []
    token_entropies = []

    for start in range(0, len(rollout_responses), microbatch_size):
        end = start + microbatch_size
        tokenized = tokenize_prompt_and_output(
            repeated_prompts[start:end], rollout_responses[start:end], tokenizer
        )
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)
        log_prob_output = get_response_log_probs(
            model, input_ids, labels, return_token_entropy=True
        )
        per_token_loss, _ = compute_policy_gradient_loss(
            advantages[start:end].to(device),
            log_prob_output["log_probs"],
            importance_reweighting_method,
            old_log_probs,
            cliprange,
        )
        microbatch_loss = aggregate_loss_across_microbatch(
            per_token_loss, response_mask, loss_normalization, normalization_constant
        )
        (microbatch_loss / gradient_accumulation_steps).backward()
        microbatch_losses.append(microbatch_loss.detach())
        token_entropies.append(
            (log_prob_output["token_entropy"] * response_mask).sum()
            / response_mask.sum()
        )

    if max_grad_norm is None:
        grad_norm = torch.linalg.vector_norm(
            torch.stack([param.grad.norm() for param in model.parameters() if param.grad is not None])
        )
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss = torch.stack(microbatch_losses).mean()
    return loss, {
        **reward_metadata,
        **advantage_metadata,
        "loss": loss,
        "grad_norm": grad_norm,
        "mean_token_entropy": torch.stack(token_entropies).mean(),
    }
