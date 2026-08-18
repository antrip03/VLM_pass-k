#!/usr/bin/env bash
# Runs ON the GCP VM - launches the real veRL Dr. GRPO training run inside
# the image built by setup_env.sh. Repo code is bind-mounted (not baked into
# the image), so a `git pull` in the repo dir picks up without rebuilding.
#
# STRONGLY RECOMMENDED FIRST RUN: a short calibration pass at the REAL
# production batch shape (S=128, not the tiny dry-run shape) before
# committing the full 500-step / ~$41 budget - this validates real veRL
# throughput/VRAM (with use_fused_kernels + vLLM rollout, NOT the raw-HF
# proxy PLAN.md's estimate is based on) cheaply first:
#   TOTAL_STEPS=5 bash scripts/gcp/run_training.sh
# Then, once that's confirmed sane (no OOM, reasonable per-step time in the
# logs), run the real thing:
#   TOTAL_STEPS=500 AUTO_SHUTDOWN=true bash scripts/gcp/run_training.sh

set -euo pipefail

REPO_DIR="$HOME/VLM_pass-k"
DATA_DIR="/mnt/disks/data"
TOTAL_STEPS="${TOTAL_STEPS:-500}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-false}"

# WANDB_API_KEY is read from THIS shell's environment (export it, or run
# `wandb login`, directly on this VM before calling this script) and passed
# through to the container - never hardcoded here or anywhere in the repo.
#
# WANDB_MODE=disabled when no key is set (2026-08-15, real fix - a
# previous version of this comment claimed training "runs fine either
# way" without verifying it): wandb does NOT gracefully skip when it has
# no key in a non-interactive container - it either errors ("API key not
# configured") or hangs on an interactive login prompt nothing can
# answer. WANDB_MODE=disabled is the real, correct way to make it no-op
# cleanly instead of risking the whole run failing/hanging on this alone.
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARGS=(-e "WANDB_API_KEY=${WANDB_API_KEY}")
else
  echo "WANDB_API_KEY not set in this shell - training will run WITHOUT wandb logging (console only)."
  echo "Run 'export WANDB_API_KEY=...' or 'wandb login' on this VM first if you want it."
  WANDB_ARGS=(-e "WANDB_MODE=disabled")
fi

echo "Launching training: $TOTAL_STEPS steps. Data/checkpoints: $DATA_DIR"

set +e
sudo docker run --rm --gpus all \
  -v "$REPO_DIR":/workspace \
  -v "$DATA_DIR":/data \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  "${WANDB_ARGS[@]}" \
  vlm-pass-k-verl:latest \
  python3 scripts/run_phase1_on_gcp.py \
    --data-dir /data \
    --checkpoint-dir /data/checkpoints \
    --total-training-steps "$TOTAL_STEPS"
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -eq 0 ]; then
  echo "Training finished successfully (exit 0). Checkpoints in $DATA_DIR/checkpoints."
else
  echo "Training exited with code $EXIT_CODE - check the output above before assuming a real run failure"
  echo "(vs. e.g. an OOM that calibration should have caught first)."
fi

if [ "$AUTO_SHUTDOWN" = "true" ]; then
  echo "AUTO_SHUTDOWN=true - stopping this VM in 60s (Ctrl+C to cancel)."
  echo "This pauses compute billing; the boot disk keeps a small residual cost"
  echo "until you delete the instance."
  sleep 60
  sudo shutdown -h now
fi

exit $EXIT_CODE
