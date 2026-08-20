"""
Fire-and-forget trigger for the deployed Phase 1 app - the robust
counterpart to run_phase1_on_modal.py's own `modal run` local_entrypoint.

Why this exists: `modal run --detach` still ties the run to this local
process for the initial call - if the local client's connection drops
mid-call (observed live, 2026-08-18: a local DNS failure produced
"Received a cancellation signal... Successfully canceled input", killing
an already-running detached App), the remote run dies with it, detached
or not. Sending a single `.spawn()` call to an already-`modal deploy`ed
app removes that dependency almost entirely: the app isn't tied to any
client's session, and the trigger itself is one short request rather
than a connection held open for the run's duration - so it's safe to
close this laptop right after this script exits.

Usage (after `modal deploy scripts/run_phase1_on_modal.py`):
    python3 scripts/trigger_phase1_training.py
    TOTAL_STEPS=50 python3 scripts/trigger_phase1_training.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modal

from configs.modal_app_verl import APP_NAME


def main() -> None:
    total_training_steps = int(os.environ.get("TOTAL_STEPS", "500"))

    prepare_data = modal.Function.from_name(APP_NAME, "prepare_data")
    run_training = modal.Function.from_name(APP_NAME, "run_training")

    print("=== Step 1: prepare_data (idempotent) ===")
    # Blocking .get() here is fine - prepare_data is short (idempotent
    # skip if already run) and not the multi-hour job we need decoupled.
    data_result = prepare_data.spawn().get()
    print(f"  {data_result}")

    wandb_key = os.environ.get("WANDB_API_KEY")
    secrets = {}
    if wandb_key:
        print("WANDB_API_KEY found locally - attaching it to this run.")
        secrets["WANDB_API_KEY"] = wandb_key
    else:
        print("WANDB_API_KEY not set locally - training will run WITHOUT wandb logging (console only).")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and os.environ.get("SKIP_HF_SYNC"):
        # Escape hatch added after a host-RAM OOM killed a run at step
        # 467/500: the in-process mirror uploads multi-GB checkpoints
        # repeatedly over many hours and is the prime suspect for gradual
        # memory growth. Set SKIP_HF_SYNC=1 to run without it (checkpoints
        # still persist on the Modal Volume; only the cross-account backup
        # is skipped).
        print("SKIP_HF_SYNC set - NOT attaching HF_TOKEN; checkpoints will live on the Modal Volume only.")
        hf_token = None
    if hf_token:
        print("HF_TOKEN found locally - attaching it so checkpoints get mirrored to a private HF Hub repo.")
        secrets["HF_TOKEN"] = hf_token
    else:
        print("HF_TOKEN not set locally - checkpoints will only live on the Modal Volume.")

    if os.environ.get("SKIP_VAL"):
        print("SKIP_VAL set - skipping veRL's full-test-set validation pass before training.")
        secrets["SKIP_VAL"] = "1"

    training_fn = run_training.with_options(secrets=[modal.Secret.from_dict(secrets)]) if secrets else run_training

    print(f"\n=== Step 2: spawning run_training (total_training_steps={total_training_steps}) ===")
    call = training_fn.spawn(total_training_steps, wandb_mode_disabled=not bool(wandb_key))
    print(f"Spawned. call_id={call.object_id}")
    print("This is now running independently on the deployed app - not tied to this process.")
    print("Safe to close this terminal/laptop now. Track progress via wandb, the Modal dashboard,")
    print(f"or later with: modal.functions.FunctionCall.from_id('{call.object_id}').get()")


if __name__ == "__main__":
    main()
