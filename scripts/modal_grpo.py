#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys

import modal

from cs336_alignment.modal_utils import GPU, REMOTE_ROOT, app, image, wandb_secret


RESULTS_VOLUME_NAME = "grpo-results"
RESULTS_MOUNT_PATH = "/mnt/grpo-results"
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


def build_command(args: list[str]) -> list[str]:
    return [
        sys.executable,
        "-u",
        "scripts/grpo.py",
        "--policy-device",
        "cuda:0",
        "--vllm-gpu",
        "1",
        "--output-dir",
        f"{RESULTS_MOUNT_PATH}/run",
        *args,
    ]


@app.function(
    image=image,
    gpu=GPU,
    timeout=8 * 60 * 60,
    max_containers=1,
    volumes={RESULTS_MOUNT_PATH: results_volume},
    secrets=[wandb_secret],
)
def run_grpo(args: list[str]) -> str:
    command = build_command(args)
    print(" ".join(command), flush=True)
    try:
        subprocess.run(command, check=True, cwd=REMOTE_ROOT)
    finally:
        results_volume.commit()
    return RESULTS_VOLUME_NAME


@app.local_entrypoint()
def main(*args: str) -> None:
    print(f"Saved results to Modal Volume: {run_grpo.remote(list(args))}")
