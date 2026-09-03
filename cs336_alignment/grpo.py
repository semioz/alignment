from collections.abc import Callable
from typing import Literal

import torch

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
    if baseline != "mean" or advantage_normalizer != "std":
        raise NotImplementedError("Only mean baseline with std normalization is supported.")
    if group_size <= 0 or raw_rewards.numel() % group_size:
        raise ValueError("raw_rewards must divide evenly into positive-size groups.")

    grouped_rewards = raw_rewards.reshape(-1, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    group_stds = grouped_rewards.std(dim=1, keepdim=True)
    advantages = ((grouped_rewards - group_means) / (group_stds + advantage_eps)).reshape(-1)
    return advantages, {
        "mean_reward": raw_rewards.mean().item(),
        "mean_group_std": group_stds.mean().item(),
    }