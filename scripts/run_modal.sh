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

# Windows consoles default Python's stdout to a codepage (e.g. cp1252) that
# can't encode the Unicode characters (checkmarks, etc.) Modal's CLI prints -
# without this, `modal run` crashes on its own status output before it even
# gets to dispatching a single function call. Harmless on Linux/macOS.
export PYTHONUTF8=1

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

if [ -z "${HF_TOKEN:-}" ]; then
  echo ""
  echo "WARNING: HF_TOKEN is not set - checkpoints will only live on this Modal account's"
  echo "Volume. If you might switch Modal accounts/workspaces mid-run, that Volume will NOT"
  echo "be reachable from the new account. Set HF_TOKEN (a write-scoped token from"
  echo "https://huggingface.co/settings/tokens, e.g. in .env) to mirror checkpoints to a"
  echo "private HF Hub repo as a cross-account backup. Continuing without it..."
fi

echo ""
echo "Deploying the training app to Modal (persistent - not tied to this session, unlike"
echo "a plain 'modal run', which was observed live to die on a local network drop even"
echo "with --detach)..."
modal deploy scripts/run_phase1_on_modal.py

echo ""
echo "Triggering the run via a single fire-and-forget call (total_training_steps=$TOTAL_STEPS)."
echo "Safe to close this terminal/laptop as soon as this returns..."
TOTAL_STEPS="$TOTAL_STEPS" python3 scripts/trigger_phase1_training.py
