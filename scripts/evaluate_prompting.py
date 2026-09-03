#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.prompting_eval import (
    category_for_reward,
    extract_gsm8k_answer,
    prompt_filename,
    render_prompt,
    summarize_results,
)
from cs336_alignment.vllm_utils import VLLMServer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "allenai/OLMo-2-0425-1B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/gsm8k/test.jsonl")
    parser.add_argument("--prompts-dir", type=Path, default=ROOT / "cs336_alignment/prompts")
    parser.add_argument("--output", type=Path, default=ROOT / "prompting_baselines.json")
    parser.add_argument("--gpu", type=int, default=0, help="GPU used by vLLM.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only this many examples.")
    parser.add_argument("--manual-review-limit", type=int, default=10)
    parser.add_argument(
        "--launch-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start a vLLM server, or use one already running at --port.",
    )
    return parser.parse_args()


def load_examples(dataset_path: Path, limit: int | None) -> list[dict[str, str]]:
    examples = []
    with dataset_path.open() as handle:
        for line in handle:
            raw_example = json.loads(line)
            examples.append(
                {
                    "question": raw_example["question"],
                    "ground_truth": extract_gsm8k_answer(raw_example),
                }
            )
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No examples found in {dataset_path}.")
    return examples


def evaluate_prompt(
    *,
    server: VLLMServer,
    prompt_name: str,
    template: str,
    reward_fn: Callable[[str, str], dict[str, float]],
    examples: list[dict[str, str]],
    sampling_params: dict[str, Any],
    batch_size: int,
    manual_review_limit: int,
) -> dict[str, Any]:
    prompts = [render_prompt(template, example["question"]) for example in examples]
    completions = server.generate_completions(prompts, sampling_params, batch_size=batch_size)
    if len(completions) != len(examples):
        raise RuntimeError(f"Expected {len(examples)} completions, received {len(completions)}.")

    results = []
    for example, prompt, completion in zip(examples, prompts, completions, strict=True):
        reward = reward_fn(completion.text, example["ground_truth"])
        results.append(
            {
                "question": example["question"],
                "ground_truth": example["ground_truth"],
                "prompt": prompt,
                "response": completion.text,
                "finish_reason": completion.finish_reason,
                "reward": reward,
                "category": category_for_reward(reward),
            }
        )

    return {
        "prompt": prompt_name,
        "sampling_params": sampling_params,
        "summary": summarize_results(results, manual_review_limit),
        "results": results,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    examples = load_examples(args.dataset, args.limit)

    prompts = [
        ("question_only", question_only_reward_fn, None),
        ("r1_zero", r1_zero_reward_fn, ["</answer>"]),
        ("r1_zero_three_shot", r1_zero_reward_fn, ["</answer>"]),
    ]
    sampling_params: dict[str, Any] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "n": 1,
        "seed": args.seed,
    }

    server = VLLMServer(
        model_id=args.model_id,
        gpu=args.gpu,
        port=args.port,
        seed=args.seed,
        launch_server=args.launch_server,
    )
    server.start()
    try:
        evaluations = {}
        for prompt_name, reward_fn, stop in prompts:
            logging.info("Evaluating %s on %d GSM8K examples.", prompt_name, len(examples))
            params = sampling_params.copy()
            if stop is not None:
                params["stop"] = stop
                params["include_stop_str_in_output"] = True
            template = (args.prompts_dir / prompt_filename(prompt_name)).read_text()
            evaluation = evaluate_prompt(
                server=server,
                prompt_name=prompt_name,
                template=template,
                reward_fn=reward_fn,
                examples=examples,
                sampling_params=params,
                batch_size=args.batch_size,
                manual_review_limit=args.manual_review_limit,
            )
            evaluations[prompt_name] = evaluation
            print(f"\n{prompt_name}: {evaluation['summary']['counts']}")

        output = {
            "model_id": args.model_id,
            "dataset": str(args.dataset),
            "seed": args.seed,
            "evaluations": evaluations,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(f"\nSaved per-example results and manual-review indices to {args.output}")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
