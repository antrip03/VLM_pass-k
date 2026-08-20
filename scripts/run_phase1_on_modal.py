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
can be invoked repeatedly across a multi-person split run), and a
dedicated checkpoint Volume separate from the dry-run one so a real run's
checkpoints never get mixed up with old plumbing-check artifacts.

wandb (2026-08-15, fixed from an earlier, worse design): NOT a
pre-created Modal Secret requirement anymore - that made wandb a hard
block on running at all (the function would fail before training even
started if a "wandb-secret" Modal Secret hadn't been created via the
dashboard first). Now reads WANDB_API_KEY from whoever's *local* shell
runs `modal run` and passes it through as a dynamically-constructed
per-call secret (modal.Secret.from_dict, via Function.with_options -
real, current Modal API, not guessed) if present. If WANDB_API_KEY isn't
set locally, WANDB_MODE=disabled is passed instead - verified this
actually matters: wandb does NOT gracefully no-op without a key in a
non-interactive environment, it either errors ("API key not configured")
or hangs on an interactive login prompt nothing can answer inside a
container. WANDB_MODE=disabled is the real, correct way to make it skip
cleanly, not just omitting the key and hoping.

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
)
def run_training(total_training_steps: int, wandb_mode_disabled: bool) -> dict:
    """
    No secrets= here - wandb is attached dynamically per-call from main()
    below via with_options(), not required at decoration time. See the
    module docstring for why (a fixed Modal Secret made wandb a hard
    block on running at all).
    """
    import os
    import subprocess
    import threading

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
        checkpoint_every=25,
        # Short runs are smoke tests (does the first training step survive?)
        # - skip veRL's ~8min full-test-set baseline validation for those,
        # keep it for real runs where the baseline is actual data. SKIP_VAL
        # additionally forces it off for short RESUME runs, where the
        # validation pass can cost more than the handful of remaining
        # training steps (our own T/D/E harness evaluates checkpoints
        # offline anyway - see phase1_config.py's module docstring).
        val_before_train=total_training_steps > 10 and not os.environ.get("SKIP_VAL"),
    )

    env = os.environ.copy()
    env["HF_HOME"] = MODEL_CACHE_DIR
    if wandb_mode_disabled:
        # No WANDB_API_KEY was available locally when this was launched -
        # verified this is necessary, not just harmless-to-omit: wandb
        # does NOT gracefully skip without a key in a non-interactive
        # environment, it errors or hangs on a login prompt nothing can
        # answer. WANDB_MODE=disabled makes it genuinely no-op instead.
        env["WANDB_MODE"] = "disabled"
    # else: WANDB_API_KEY is already present in env via the dynamically
    # attached per-call secret (see main() below).

    print(f"Launching training: total_training_steps={total_training_steps} "
          f"(resume_mode=auto - will pick up from the latest checkpoint in "
          f"{CHECKPOINT_DIR} if one exists from a prior invocation)")

    # HF Hub checkpoint mirror: only active if HF_TOKEN was attached (see
    # main() below). Covers the case a Modal Volume alone can't - resuming
    # after switching to a DIFFERENT Modal account/workspace, since Volumes
    # don't carry over between accounts. Pull-before/push-during, both
    # best-effort (see src/training/hf_checkpoint_sync.py docstring).
    hf_watch_stop = None
    hf_watch_thread = None
    if os.environ.get("HF_TOKEN"):
        from src.training.hf_checkpoint_sync import ensure_repo, pull_latest_checkpoint, watch_and_push

        try:
            rid = ensure_repo()
            print(f"[hf_checkpoint_sync] Mirroring checkpoints to Hub repo {rid}.")
            pull_latest_checkpoint(CHECKPOINT_DIR)
        except Exception as e:
            print(f"[hf_checkpoint_sync] WARNING: setup failed, continuing without Hub mirror: {e}")
        else:
            hf_watch_stop = threading.Event()
            hf_watch_thread = threading.Thread(
                target=watch_and_push, args=(CHECKPOINT_DIR, hf_watch_stop), daemon=True
            )
            hf_watch_thread.start()
    else:
        print("[hf_checkpoint_sync] HF_TOKEN not set - checkpoints will NOT be mirrored off Modal; "
              "a Modal account/workspace switch would lose them.")

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
    # Streamed line-by-line (2026-08-19, replacing subprocess.run with
    # capture_output=True): the old version buffered ALL of veRL's output
    # until the process exited, so a multi-hour run showed literally
    # nothing in `modal app logs` until it was over - no way to tell
    # "loading the model" from "training step 300" from "hung". Popen with
    # stderr merged into stdout lets every line reach Modal's logs live,
    # while a bounded deque still keeps the tail for the return value
    # (so nothing that used to be reported is lost).
    import collections
    import re
    import time

    deadline = time.time() + (TRAINING_TIMEOUT_SECONDS - 300)
    tail = collections.deque(maxlen=500)
    step_re = re.compile(r"training/global_step:(\d+)")
    run_start = time.time()
    prev_step, prev_step_time = 0, run_start
    recent_step_secs: collections.deque = collections.deque(maxlen=20)
    timed_out = False

    def _hms(seconds: float) -> str:
        seconds = int(max(seconds, 0))
        return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"

    proc = subprocess.Popen(
        ["python3", "-m", "verl.trainer.main_ppo", *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    print(f"[progress] training subprocess started (pid {proc.pid}). "
          f"Target: {total_training_steps} steps, checkpoint every 25.", flush=True)
    try:
        for line in proc.stdout:
            line = line.rstrip()
            tail.append(line)
            print(line, flush=True)

            match = step_re.search(line)
            if match:
                step = int(match.group(1))
                if step > prev_step:
                    now = time.time()
                    recent_step_secs.append((now - prev_step_time) / (step - prev_step))
                    avg = sum(recent_step_secs) / len(recent_step_secs)
                    remaining = (total_training_steps - step) * avg
                    print(
                        f"[progress] step {step}/{total_training_steps}"
                        f" | elapsed {_hms(now - run_start)}"
                        f" | {avg:.1f}s/step (last {len(recent_step_secs)})"
                        f" | ETA {_hms(remaining)}"
                        f" | done ~{time.strftime('%H:%M UTC', time.gmtime(now + remaining))}",
                        flush=True,
                    )
                    prev_step, prev_step_time = step, now

            # Deadline is checked per line rather than by a hard timer:
            # veRL is chatty enough that this fires promptly in practice,
            # and Modal's own function timeout remains the hard backstop.
            if time.time() > deadline:
                print("[progress] per-invocation deadline reached - stopping so checkpoints "
                      "commit cleanly. Re-run to resume from the last checkpoint.", flush=True)
                proc.kill()
                timed_out = True
                break

        proc.wait(timeout=120)
        returncode = -1 if timed_out else proc.returncode
        text_tail = "\n".join(tail)
        stdout_tail = text_tail[-8000:]
        stderr_tail = text_tail[-3000:]  # stderr is merged into stdout now

        # Exit code alone is NOT trustworthy here (real case 2026-08-19: a
        # host-RAM OOM SIGKILLed a DataLoader worker at step 467/500, Ray
        # swallowed the worker failure, and the driver still exited 0 - a
        # crashed run reporting success). Treat "did not reach the target
        # step" as failure regardless of exit code.
        if returncode == 0 and not timed_out and prev_step < total_training_steps:
            print(f"[progress] WARNING: process exited 0 but only reached step {prev_step}/"
                  f"{total_training_steps} - treating as FAILURE, not success.", flush=True)
            returncode = 1
    finally:
        # Explicit commit regardless of how the subprocess ended (success,
        # failure, or timeout) - belt-and-suspenders on top of Modal's own
        # background/shutdown auto-commit, not a substitute for it.
        checkpoint_volume.commit()
        if hf_watch_stop is not None:
            hf_watch_stop.set()
            hf_watch_thread.join(timeout=600)

    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "last_step": prev_step,
        "total_training_steps": total_training_steps,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


@app.local_entrypoint()
def main(total_training_steps: int = 500):
    import os

    print("=== Step 1: prepare_data (idempotent) ===")
    data_result = prepare_data.remote()
    print(f"  {data_result}")

    wandb_key = os.environ.get("WANDB_API_KEY")
    secrets = {}
    if wandb_key:
        print("WANDB_API_KEY found locally - attaching it to this run.")
        secrets["WANDB_API_KEY"] = wandb_key
    else:
        print("WANDB_API_KEY not set locally - training will run WITHOUT wandb logging (console only).")
        print("export WANDB_API_KEY=your-key before running this if you want live curves.")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("HF_TOKEN found locally - attaching it so checkpoints get mirrored to a private HF Hub repo "
              "(survives a Modal account/workspace switch, which the Volume alone does not).")
        secrets["HF_TOKEN"] = hf_token
    else:
        print("HF_TOKEN not set locally - checkpoints will only live on the Modal Volume. If you might switch "
              "Modal accounts mid-run, export HF_TOKEN (a write-scoped token from huggingface.co/settings/tokens) "
              "so checkpoints get mirrored to the Hub too.")

    training_fn = run_training.with_options(secrets=[modal.Secret.from_dict(secrets)]) if secrets else run_training

    print(f"\n=== Step 2: run_training (total_training_steps={total_training_steps}) ===")
    train_result = training_fn.remote(total_training_steps, wandb_mode_disabled=not bool(wandb_key))
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
