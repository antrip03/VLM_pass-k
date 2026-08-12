"""
Portable (Modal-independent) launcher for the real Phase 1 veRL Dr. GRPO run,
for execution inside the Docker container built from configs/gcp/Dockerfile.verl
on a GCP VM. Mirrors the real, already-validated invocation pattern from
scripts/run_verl_dry_run_on_modal.py's run_dry_run() function (same
HF_HOME/subprocess/main_ppo pattern, confirmed working on live Modal
infrastructure) - just swapped from Modal Volumes/dry_run_config to plain
paths/phase1_config, since this doesn't run inside a Modal function.

Data prep (prepare_data.py) is invoked here too, not as a separate step -
it's lightweight (pandas only, no GPU) and idempotent (skips if the Parquet
files already exist), so folding it in avoids an extra manual step without
meaningfully lengthening a real training run.

Usage (inside the container, repo bind-mounted at /workspace):
    HF_HOME=/data/hf_cache python3 scripts/run_phase1_on_gcp.py \
        --data-dir /data --checkpoint-dir /data/checkpoints
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src as _src  # resolve the REAL repo root, not a guessed one

from src.training.phase1_config import build_args
from src.training.prepare_data import write_parquet

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Where train/test Parquet + HF cache live")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--total-training-steps", type=int, default=500)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_file = data_dir / "gsm8k_train.parquet"
    val_file = data_dir / "gsm8k_test.parquet"

    if not train_file.exists():
        print(f"Preparing training data -> {train_file}")
        write_parquet("train", train_file)
    if not val_file.exists():
        print(f"Preparing validation data -> {val_file}")
        write_parquet("test", val_file)

    src_root = Path(_src.__file__).resolve().parent
    reward_fn_path = src_root / "training" / "reward_fn.py"

    verl_args = build_args(
        model_path=MODEL_ID,
        train_file=str(train_file),
        val_file=str(val_file),
        reward_fn_path=str(reward_fn_path),
        checkpoint_dir=args.checkpoint_dir,
        total_training_steps=args.total_training_steps,
    )

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(data_dir / "hf_cache"))

    print(f"Launching real Phase 1 training: {args.total_training_steps} steps, "
          f"checkpoint_dir={args.checkpoint_dir}")
    result = subprocess.run(["python3", "-m", "verl.trainer.main_ppo", *verl_args], env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
