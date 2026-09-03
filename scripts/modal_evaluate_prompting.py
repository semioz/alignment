#!/usr/bin/env python3
"""Run the GSM8K prompting baseline on a Modal GPU.

Usage:
    modal run scripts/modal_evaluate_prompting.py
    modal volume get prompting-baselines-results prompting_baselines.json ./prompting_baselines.json
"""

from __future__ import annotations

import subprocess
import sys

import modal

from cs336_alignment.modal_utils import GPU, REMOTE_ROOT, app, image


RESULTS_VOLUME_NAME = "prompting-baselines-results"
RESULTS_MOUNT_PATH = "/mnt/prompting-baselines-results"
RESULTS_FILENAME = "prompting_baselines.json"
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


def build_command(args: list[str]) -> list[str]:
    return [
        sys.executable,
        "-u",
        "scripts/evaluate_prompting.py",
        "--gpu",
        "0",
        "--output",
        f"{RESULTS_MOUNT_PATH}/{RESULTS_FILENAME}",
        *args,
    ]


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    max_containers=1,
    volumes={RESULTS_MOUNT_PATH: results_volume},
)
def run_evaluation(args: list[str]) -> str:
    command = build_command(args)
    print(" ".join(command), flush=True)
    try:
        subprocess.run(command, check=True, cwd=REMOTE_ROOT)
    finally:
        results_volume.commit()
    return f"{RESULTS_VOLUME_NAME}/{RESULTS_FILENAME}"


@app.local_entrypoint()
def main(*args: str) -> None:
    print(f"Saved results to {run_evaluation.remote(list(args))}")
