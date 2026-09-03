from collections.abc import Callable

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

    return (raw_rewards, metadata)