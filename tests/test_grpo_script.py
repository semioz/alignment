from __future__ import annotations

import sys

from scripts.grpo import make_sampling_params, parse_args, validate_args


def test_training_sampling_uses_independent_rollouts() -> None:
    params = make_sampling_params("r1_zero", temperature=1.0, max_tokens=512)

    assert params["stop"] == ["</answer>"]
    assert "seed" not in params


def test_off_policy_cli_accepts_grpo_reweighting(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grpo.py",
            "--importance-reweighting-method",
            "grpo",
            "--cliprange",
            "0.1",
        ],
    )

    args = parse_args()
    validate_args(args)

    assert args.importance_reweighting_method == "grpo"
    assert args.cliprange == 0.1


def test_constant_loss_cli_uses_drgrpo_options(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grpo.py",
            "--baseline",
            "mean",
            "--advantage-normalizer",
            "none",
            "--loss-normalization",
            "constant",
        ],
    )

    args = parse_args()
    validate_args(args)

    assert args.baseline == "mean"
    assert args.advantage_normalizer == "none"
    assert args.loss_normalization == "constant"
    assert args.normalization_constant == 256 * 512
