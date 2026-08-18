"""
Real Phase 1 veRL Dr. GRPO run on Modal - the Modal-based counterpart to
scripts/run_phase1_on_gcp.py, for whoever runs their share of training on
Modal rather than a raw GCP VM (PLAN.md's Phase 1 design; see
HANDOFF.md for the full split-run plan between two people/environments).

Reuses configs/modal_app_verl.py's already-validated image (same real
veRL environment, same fixes, proven working on live Modal infrastructure
2026-08-10-11) and the same invocation pattern as
scripts/run_verl_dry_run_on_modal.py (prepare_data -> run_training,
subprocess into verl.trainer.main_ppo) - just swapped from the tiny
dry_run_config to the real phase1_config, with the FULL GSM8K train split
(not a 50-example slice), a configurable total_training_steps (so this
can be invoked repeatedly across a multi-person split run), wandb via a
Modal Secret (never a hardcoded key - see HANDOFF.md for setup), and a
dedicated checkpoint Volume separate from the dry-run one so a real run's
checkpoints never get mixed up with old plumbing-check artifacts.

Checkpoint persistence: this Volume (grpo-vlm-phase1-checkpoints) is where
trainer.resume_mode=auto looks for the latest checkpoint on every
invocation - re-running this script with the same total_training_steps
(or a higher one) will automatically resume from wherever the last
invocation left off, matching the same resume mechanism validated for
the GCP path. To hand a checkpoint off to someone running on a DIFFERENT
environment (e.g. GCP), see HANDOFF.md's export/transfer steps - this
script alone does not move checkpoints off Modal.

Usage:
    modal run scripts/run_phase1_on_modal.py --total-training-steps 250
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modal

from configs.modal_app_verl import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

DATA_DIR = "/data"
CHECKPOINT_DIR = "/checkpoints"
data_volume = modal.Volume.from_name("grpo-vlm-phase1-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("grpo-vlm-phase1-checkpoints", create_if_missing=True)

# Real run, not a plumbing check - full GSM8K train split, not a slice.
TRAIN_FILE = f"{DATA_DIR}/gsm8k_train.parquet"
VAL_FILE = f"{DATA_DIR}/gsm8k_test.parquet"

TRAINING_TIMEOUT_SECONDS = 20 * 60 * 60  # 20hr ceiling per invocation - generous for "one person's share"


@app.function(image=image, volumes={DATA_DIR: data_volume}, timeout=10 * 60)
def prepare_data() -> dict:
    """Idempotent - skips re-writing if the Parquet files already exist on the Volume."""
    import os

    from src.training.prepare_data import write_parquet

    result = {}
    if not os.path.exists(TRAIN_FILE):
        write_parquet("train", TRAIN_FILE)
        result["train"] = "written"
    else:
        result["train"] = "already present"
    if not os.path.exists(VAL_FILE):
        write_parquet("test", VAL_FILE)
        result["val"] = "written"
    else:
        result["val"] = "already present"
    data_volume.commit()
    return result


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={MODEL_CACHE_DIR: model_cache, DATA_DIR: data_volume, CHECKPOINT_DIR: checkpoint_volume},
    timeout=TRAINING_TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_training(total_training_steps: int) -> dict:
    """
    secrets=[modal.Secret.from_name("wandb-secret")]: requires a Modal
    Secret named "wandb-secret" containing WANDB_API_KEY, created via the
    Modal dashboard (Secrets -> Create new secret) BEFORE running this -
    never pass the key as a CLI arg or hardcode it here. See HANDOFF.md.
    """
    import os
    import subprocess

    import src as _src

    from src.training.phase1_config import build_args

    src_root = Path(_src.__file__).resolve().parent
    reward_fn_path = src_root / "training" / "reward_fn.py"

    args = build_args(
        model_path=MODEL_ID,
        train_file=TRAIN_FILE,
        val_file=VAL_FILE,
        reward_fn_path=str(reward_fn_path),
        checkpoint_dir=CHECKPOINT_DIR,
        total_training_steps=total_training_steps,
    )

    env = os.environ.copy()
    env["HF_HOME"] = MODEL_CACHE_DIR
    # WANDB_API_KEY already present in env via the Modal Secret above.

    print(f"Launching training: total_training_steps={total_training_steps} "
          f"(resume_mode=auto - will pick up from the latest checkpoint in "
          f"{CHECKPOINT_DIR} if one exists from a prior invocation)")

    # Real gap found and fixed 2026-08-15: subprocess.run(..., timeout=...)
    # raises TimeoutExpired on timeout - previously uncaught, meaning
    # checkpoint_volume.commit() below never ran and the function crashed
    # with a raw traceback instead of a clear result. Modal Volumes DO
    # auto-commit in the background every few seconds and on container
    # shutdown (confirmed via modal.com/docs/guide/volumes), so checkpoint
    # progress itself was likely NOT actually at risk even before this fix
    # - but relying on that implicitly rather than committing explicitly,
    # and crashing instead of reporting clearly, was still worth fixing:
    # your teammate should see "hit the timeout, re-run to resume", not an
    # unexplained crash that looks like something is broken.
    try:
        result = subprocess.run(
            ["python3", "-m", "verl.trainer.main_ppo", *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=TRAINING_TIMEOUT_SECONDS - 300,
        )
        returncode = result.returncode
        stdout_tail = result.stdout[-8000:]
        stderr_tail = result.stderr[-8000:]
        timed_out = False
    except subprocess.TimeoutExpired as e:
        returncode = -1
        stdout_tail = (e.stdout or "")[-8000:] if e.stdout else ""
        stderr_tail = (e.stderr or "")[-8000:] if e.stderr else ""
        timed_out = True
    finally:
        # Explicit commit regardless of how the subprocess ended (success,
        # failure, or timeout) - belt-and-suspenders on top of Modal's own
        # background/shutdown auto-commit, not a substitute for it.
        checkpoint_volume.commit()

    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


@app.local_entrypoint()
def main(total_training_steps: int = 500):
    print("=== Step 1: prepare_data (idempotent) ===")
    data_result = prepare_data.remote()
    print(f"  {data_result}")

    print(f"\n=== Step 2: run_training (total_training_steps={total_training_steps}) ===")
    train_result = run_training.remote(total_training_steps)
    print(f"  returncode: {train_result['returncode']}")
    print("  --- stdout tail ---")
    print(train_result["stdout_tail"])
    print("  --- stderr tail ---")
    print(train_result["stderr_tail"])

    if train_result["returncode"] == 0:
        print("\nTraining run exited cleanly (exit 0).")
        print("Checkpoints are on the 'grpo-vlm-phase1-checkpoints' Modal Volume.")
        print("See HANDOFF.md for how to export/transfer them if handing off to another environment.")
    elif train_result.get("timed_out"):
        print(f"\nHit this invocation's {TRAINING_TIMEOUT_SECONDS // 3600}hr timeout - this is expected for a")
        print("long run, not a failure. Checkpoints up to the last save are safely committed to the Volume.")
        print("Just re-run the exact same command again - resume_mode=auto will pick up from the last")
        print("checkpoint automatically, not restart from step 0.")
    else:
        print(f"\nTraining exited with code {train_result['returncode']} - check the output above.")
