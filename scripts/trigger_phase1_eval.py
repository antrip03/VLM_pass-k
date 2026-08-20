"""
Fire-and-forget trigger for the DEPLOYED Phase 1 eval app.

Exists for the same reason scripts/trigger_phase1_training.py does, and
because the lesson had to be learned twice: `modal run` creates an
EPHEMERAL app that is torn down as soon as the local entrypoint returns,
which cancels anything spawned from inside it ("Stopping app - local
entrypoint completed", then 0 tasks). Spawning against an already
`modal deploy`ed app has no such lifecycle tie.

Usage:
    modal deploy scripts/run_phase1_eval_on_modal.py
    python3 scripts/trigger_phase1_eval.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modal

from configs.modal_app_verl import APP_NAME


def main() -> None:
    n = int(os.environ.get("EVAL_N", "128"))
    problems = int(os.environ.get("EVAL_PROBLEMS", "50"))
    image_chunk = int(os.environ.get("EVAL_IMAGE_CHUNK", "32"))

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN is required - the RL checkpoint lives on the HF Hub mirror.")

    eval_fn = modal.Function.from_name(APP_NAME, "eval_one_model")
    fn = eval_fn.with_options(secrets=[modal.Secret.from_dict({"HF_TOKEN": hf_token})])

    print(f"Spawning base + RL evals in parallel (n={n}, problems={problems}, image_chunk={image_chunk})")
    calls = {kind: fn.spawn(kind, n, problems, image_chunk) for kind in ("base", "rl")}
    for kind, call in calls.items():
        print(f"  {kind}: call_id={call.object_id}")
    print("\nRunning on the deployed app - not tied to this process. Safe to disconnect.")
    print("When both finish:  modal run scripts/run_phase1_eval_on_modal.py::analyze")


if __name__ == "__main__":
    main()
