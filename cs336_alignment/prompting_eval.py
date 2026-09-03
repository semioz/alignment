"""Reusable helpers for the Assignment 5 prompting-baseline evaluation."""

from __future__ import annotations

from collections import Counter
from typing import Any


CATEGORY_NAMES = ("correct", "formatted_incorrect", "unformatted")


def prompt_filename(prompt_name: str) -> str:
    """Return the repository filename for a named prompting baseline."""
    if prompt_name == "r1_zero_three_shot":
        return "r1_zero_three_shot_gsm8k.prompt"
    return f"{prompt_name}.prompt"


def extract_gsm8k_answer(example: dict[str, Any]) -> str:
    """Return the final GSM8K answer, excluding the supplied rationale."""
    try:
        return example["answer"].rsplit("####", 1)[1].strip()
    except (KeyError, IndexError) as error:
        raise ValueError("GSM8K example must have an answer containing '####'.") from error


def render_prompt(template: str, question: str) -> str:
    """Insert a GSM8K question into one of the provided prompt templates."""
    return template.format(question=question)


def category_for_reward(reward: dict[str, float]) -> str:
    """Map grader output to the three categories requested in the handout."""
    if reward["format_reward"] == 1.0 and reward["answer_reward"] == 1.0:
        return "correct"
    if reward["format_reward"] == 1.0 and reward["answer_reward"] == 0.0:
        return "formatted_incorrect"
    if reward["format_reward"] == 0.0 and reward["answer_reward"] == 0.0:
        return "unformatted"
    raise ValueError(f"Unexpected reward combination: {reward!r}")


def summarize_results(
    results: list[dict[str, Any]], manual_review_limit: int = 10
) -> dict[str, Any]:
    """Count evaluation categories and identify examples for manual inspection."""
    counts = Counter(result["category"] for result in results)
    manual_review_indices = {
        category: [
            index
            for index, result in enumerate(results)
            if result["category"] == category
        ][:manual_review_limit]
        for category in ("formatted_incorrect", "unformatted")
    }
    return {
        "n_examples": len(results),
        "counts": {category: counts[category] for category in CATEGORY_NAMES},
        "manual_review_indices": manual_review_indices,
    }
