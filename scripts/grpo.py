#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from cs336_alignment.checkpoint import (
    get_model_and_tokenizer,
    get_response_log_probs,
    tokenize_prompt_and_output,
)
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.grpo import grpo_train_step
from cs336_alignment.prompting_eval import prompt_filename, render_prompt
from cs336_alignment.vllm_utils import VLLMServer
from scripts.evaluate_prompting import DEFAULT_MODEL_ID, ROOT, load_examples


def repeat_rollout_inputs(
    examples: list[dict[str, str]], group_size: int
) -> tuple[list[str], list[str]]:
    return (
        [example["question"] for example in examples for _ in range(group_size)],
        [example["ground_truth"] for example in examples for _ in range(group_size)],
    )


def make_sampling_params(
    prompt_name: str, temperature: float, max_tokens: int
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "n": 1,
    }
    if prompt_name != "question_only":
        params["stop"] = ["</answer>"]
        params["include_stop_str_in_output"] = True
    return params


def get_old_response_log_probs(
    policy: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    responses: list[str],
) -> torch.Tensor:
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    device = next(policy.parameters()).device
    with torch.no_grad():
        return get_response_log_probs(
            policy,
            tokenized["input_ids"].to(device),
            tokenized["labels"].to(device),
            return_token_entropy=False,
        )["log_probs"].cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-dataset", type=Path, default=ROOT / "data/gsm8k/train.jsonl")
    parser.add_argument("--val-dataset", type=Path, default=ROOT / "data/gsm8k/test.jsonl")
    parser.add_argument("--prompts-dir", type=Path, default=ROOT / "cs336_alignment/prompts")
    parser.add_argument(
        "--prompt-name",
        choices=("question_only", "r1_zero", "r1_zero_three_shot"),
        default="r1_zero",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/grpo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--baseline", choices=("mean", "none"), default="mean")
    parser.add_argument(
        "--advantage-normalizer", choices=("std", "none", "mean"), default="std"
    )
    parser.add_argument(
        "--loss-normalization", choices=("sequence", "constant"), default="sequence"
    )
    parser.add_argument("--normalization-constant", type=int)
    parser.add_argument(
        "--importance-reweighting-method",
        choices=("none", "noclip", "grpo"),
        default="none",
    )
    parser.add_argument("--cliprange", type=float)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--rollout-log-interval", type=int, default=40)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--wandb-project", default="cs336-grpo-gsm8k")
    parser.add_argument("--wandb-mode", choices=("online", "disabled"), default="online")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rollout_batch_size != args.train_batch_size:
        raise ValueError("On-policy GRPO requires equal rollout and train batch sizes.")
    if args.rollout_batch_size % args.group_size:
        raise ValueError("rollout_batch_size must be divisible by group_size.")
    if args.train_batch_size % args.gradient_accumulation_steps:
        raise ValueError("train_batch_size must divide evenly into gradient accumulation steps.")
    if args.loss_normalization == "constant" and args.normalization_constant is None:
        args.normalization_constant = args.train_batch_size * args.sampling_max_tokens
    if args.importance_reweighting_method == "grpo" and args.cliprange is None:
        raise ValueError("cliprange is required for GRPO importance reweighting.")


def evaluate(
    server: VLLMServer,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    sampling_params: dict[str, Any],
    batch_size: int,
) -> dict[str, float]:
    completions = server.generate_completions(prompts, sampling_params, batch_size=batch_size)
    rewards = [
        reward_fn(completion.text, ground_truth)
        for completion, ground_truth in zip(completions, ground_truths, strict=True)
    ]
    return {
        "val/reward": sum(reward["reward"] for reward in rewards) / len(rewards),
        "val/format_reward": sum(reward["format_reward"] for reward in rewards) / len(rewards),
        "val/response_length": sum(len(completion.token_ids) for completion in completions)
        / len(completions),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    train_examples = load_examples(args.train_dataset, args.n_train_examples)
    val_examples = load_examples(args.val_dataset, args.n_val_examples)
    reward_fn = question_only_reward_fn if args.prompt_name == "question_only" else r1_zero_reward_fn
    prompt_template = (args.prompts_dir / prompt_filename(args.prompt_name)).read_text()
    val_questions = [example["question"] for example in val_examples]
    val_prompts = [render_prompt(prompt_template, question) for question in val_questions]
    val_ground_truths = [example["ground_truth"] for example in val_examples]
    sampling_params = make_sampling_params(
        args.prompt_name, args.sampling_temperature, args.sampling_max_tokens
    )
    policy, tokenizer = get_model_and_tokenizer(args.model_id, args.policy_device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )
    import wandb

    run = wandb.init(project=args.wandb_project, mode=args.wandb_mode, config=vars(args))
    server = VLLMServer(model_id=args.model_id, gpu=args.vllm_gpu, port=args.port, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"

    server.start()
    try:
        server.init_weight_sync(args.policy_device)
        for step in range(args.num_rollout_steps):
            batch = random.sample(train_examples, args.rollout_batch_size // args.group_size)
            questions, ground_truths = repeat_rollout_inputs(batch, args.group_size)
            rollout_prompts = [render_prompt(prompt_template, question) for question in questions]
            server.sync_policy_weights(policy)
            completions = server.generate_completions(
                rollout_prompts, sampling_params, batch_size=args.rollout_batch_size
            )
            responses = [completion.text for completion in completions]
            old_log_probs = None
            if args.importance_reweighting_method != "none":
                policy.eval()
                old_log_probs = get_old_response_log_probs(
                    policy, tokenizer, rollout_prompts, responses
                )
            policy.train()
            loss, train_metrics = grpo_train_step(
                policy,
                tokenizer,
                optimizer,
                args.gradient_accumulation_steps,
                args.max_grad_norm,
                reward_fn,
                rollout_prompts,
                responses,
                ground_truths,
                args.group_size,
                baseline=args.baseline,
                advantage_normalizer=args.advantage_normalizer,
                loss_normalization=args.loss_normalization,
                normalization_constant=args.normalization_constant,
                importance_reweighting_method=args.importance_reweighting_method,
                old_log_probs=old_log_probs,
                cliprange=args.cliprange,
            )
            metrics: dict[str, Any] = {
                "step": step,
                **{
                    key: value.item() if isinstance(value, torch.Tensor) else value
                    for key, value in train_metrics.items()
                },
            }
            if step % args.eval_interval == 0:
                server.sync_policy_weights(policy)
                policy.eval()
                metrics.update(
                    evaluate(
                        server,
                        val_prompts,
                        val_ground_truths,
                        reward_fn,
                        sampling_params,
                        args.rollout_batch_size,
                    )
                )
            if step % args.rollout_log_interval == 0:
                metrics["train/rollouts"] = [
                    {"prompt": prompt, "response": response}
                    for prompt, response in zip(rollout_prompts[:8], responses[:8], strict=True)
                ]
            logging.info("step %d: %s", step, metrics)
            run.log(metrics, step=step)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")

        policy.save_pretrained(args.output_dir / "checkpoint")
        tokenizer.save_pretrained(args.output_dir / "checkpoint")
    finally:
        server.stop()
        run.finish()


if __name__ == "__main__":
    main()
