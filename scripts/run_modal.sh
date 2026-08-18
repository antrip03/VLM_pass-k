#!/usr/bin/env bash
# THE single command to start the real Phase 1 run on Modal, once
# `modal setup` has been done. Unlike scripts/run_phase1_on_modal.py's own
# wandb handling (which treats wandb as optional and falls back to
# console-only), THIS wrapper treats wandb as REQUIRED and REAL-VERIFIED -
# it will refuse to start the run at all if a working key isn't available,
# rather than silently starting an unlogged run.
#
# Usage:
#   export WANDB_API_KEY=your-key   (or put it in a local .env file instead)
#   bash scripts/run_modal.sh
#
# Optional: TOTAL_STEPS=50 bash scripts/run_modal.sh   (quick check instead
# of the real 500-step run)

set -euo pipefail

if [ -f .env ]; then
  echo "Loading .env (local only, never committed - see .gitignore)"
  set -a
  source .env
  set +a
fi

TOTAL_STEPS="${TOTAL_STEPS:-500}"

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY is not set."
  echo "wandb logging is required for this project - this script will not start an unlogged run."
  echo ""
  echo "Fix: export WANDB_API_KEY=your-key-here"
  echo "  (or: echo \"WANDB_API_KEY=your-key-here\" > .env)"
  echo "Get a key from https://wandb.ai -> Settings -> API keys."
  exit 1
fi

if ! command -v wandb &>/dev/null; then
  echo "wandb CLI not found locally - installing (pip install wandb)..."
  pip install -q wandb
fi

echo "Verifying the wandb key actually authenticates (real server-side check, not just checking it's non-empty)..."
if ! wandb login "$WANDB_API_KEY" --verify; then
  echo ""
  echo "ERROR: wandb key verification FAILED - this key does not actually authenticate."
  echo "Get a real key from https://wandb.ai -> Settings -> API keys and try again."
  echo "Stopping here - not starting Modal training with a broken wandb key."
  exit 1
fi
echo "wandb key verified OK - live training curves will work."

if ! command -v modal &>/dev/null; then
  echo "ERROR: modal CLI not found. Run: pip install modal && modal setup"
  exit 1
fi

echo ""
echo "Starting the real Phase 1 run on Modal (total_training_steps=$TOTAL_STEPS)..."
modal run scripts/run_phase1_on_modal.py --total-training-steps "$TOTAL_STEPS"
