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

if [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_UPLOAD_REPO:-}" ]; then
  echo "HF_TOKEN + HF_UPLOAD_REPO set - checkpoints will also back up to huggingface.co/$HF_UPLOAD_REPO (private)."
else
  echo "HF_TOKEN/HF_UPLOAD_REPO not set - checkpoints stay on the Modal Volume only (optional extra backup, not required)."
fi

if ! command -v modal &>/dev/null; then
  echo "ERROR: modal CLI not found. Run: pip install modal && modal setup"
  exit 1
fi

# Config sanity check, added 2026-08-19 after TWO real Modal runs crashed
# identically on a known FSDP bug that had already been fixed in the repo -
# the wasted compute came from not being able to SEE which code was actually
# running. This makes that observable before anything is spent, and hard-
# fails on the known-bad value instead of discovering it 20 minutes in.
echo ""
echo "=== Config sanity check (confirms which code you're actually about to run) ==="
echo "  branch: $(git branch --show-current 2>/dev/null || echo '?')"
echo "  commit: $(git log -1 --format='%h %s' 2>/dev/null || echo 'not a git checkout')"

# python3 on Linux/macOS, python on some setups (e.g. Windows/conda) - don't
# assume either exists under one name, that would fail the check for an
# irrelevant reason and block a legitimate run.
# Tests that the candidate can actually EXECUTE, not just that the name
# resolves on PATH - Windows ships a "python3" App Store shim that exists
# but fails when run, which silently broke this check during testing.
PY_BIN=""
for candidate in python3 python; do
  if "$candidate" -c "" &>/dev/null; then PY_BIN="$candidate"; break; fi
done
if [ -z "$PY_BIN" ]; then
  echo "ERROR: no python3/python found on PATH - cannot verify the config."
  exit 1
fi

set +e
CONFIG_CHECK="$("$PY_BIN" - <<'PY'
import sys
try:
    from src.training.phase1_config import build_args
except Exception as e:
    print(f"IMPORT_FAILED: {e}")
    sys.exit(2)
args = build_args(model_path="x", train_file="x", val_file="x",
                  reward_fn_path="x", checkpoint_dir="x")
watch = ("fsdp_config.param_offload", "fsdp_config.optimizer_offload",
         "rollout.gpu_memory_utilization", "fsdp_config.use_orig_params")
for a in args:
    if any(k in a for k in watch):
        print(a)
PY
)"
CONFIG_RC=$?
set -e

if [ $CONFIG_RC -ne 0 ]; then
  echo "$CONFIG_CHECK"
  echo "ERROR: could not resolve the training config."
  echo "Are you in the repo root, on the Guneesh branch? (cd into the cloned repo first)"
  exit 1
fi
echo "$CONFIG_CHECK" | sed 's/^/  /'

if echo "$CONFIG_CHECK" | grep -q "actor.fsdp_config.param_offload=True"; then
  echo ""
  echo "ERROR: actor param_offload=True detected - this is the KNOWN-BROKEN setting that"
  echo "crashed previous runs with:"
  echo "    AssertionError: as_params=True type(prim_param)=<class 'torch.Tensor'>"
  echo "You are running STALE code from before that fix."
  echo ""
  echo "This repo's DEFAULT branch (main) does NOT contain this project's code at all -"
  echo "a plain 'git clone' gets you the wrong branch. Get the right one:"
  echo "    git checkout Guneesh && git pull origin Guneesh"
  echo "Then re-run. Expected: actor.fsdp_config.param_offload=False"
  exit 1
fi
echo "  OK - actor param_offload is disabled (the FSDP crash fix IS present in this code)."

echo ""
echo "Starting the real Phase 1 run on Modal (total_training_steps=$TOTAL_STEPS)..."
modal run scripts/run_phase1_on_modal.py --total-training-steps "$TOTAL_STEPS"
