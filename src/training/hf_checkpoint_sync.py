"""
Cross-account checkpoint backup: mirrors the Modal Volume's checkpoint
directory to a private HF Hub model repo, so training can resume even if
the Modal *account/workspace* itself changes mid-run - a real gap the
Volume alone can't cover, since Volumes are scoped to one workspace and
aren't visible from a different account (unlike an API-token rotation
within the same account, which the Volume survives fine on its own).

Best-effort by design: every push/pull is wrapped so a network hiccup or
Hub outage logs a warning and gets retried on the next poll, never raises
into the training subprocess.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

DEFAULT_REPO_NAME = "grpo-vlm-phase1-checkpoints"


def _token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")
    return token


def repo_id() -> str:
    override = os.environ.get("HF_CHECKPOINT_REPO_ID")
    if override:
        return override
    from huggingface_hub import HfApi

    username = HfApi(token=_token()).whoami()["name"]
    return f"{username}/{DEFAULT_REPO_NAME}"


def ensure_repo() -> str:
    from huggingface_hub import HfApi

    rid = repo_id()
    HfApi(token=_token()).create_repo(rid, private=True, repo_type="model", exist_ok=True)
    return rid


def pull_latest_checkpoint(checkpoint_dir: str) -> None:
    """Only pulls if checkpoint_dir has no checkpoint yet - i.e. a fresh
    Volume on a newly-switched-to account. Never overwrites local progress."""
    local = Path(checkpoint_dir)
    if any(local.glob("global_step_*")):
        print(f"[hf_checkpoint_sync] {checkpoint_dir} already has checkpoints - skipping Hub pull.")
        return

    from huggingface_hub import HfApi, snapshot_download

    rid = repo_id()
    try:
        files = HfApi(token=_token()).list_repo_files(rid, repo_type="model")
    except Exception as e:
        print(f"[hf_checkpoint_sync] Could not list {rid} (likely doesn't exist yet): {e}")
        return
    if not any("global_step_" in f for f in files):
        print(f"[hf_checkpoint_sync] {rid} has no checkpoints yet - starting fresh.")
        return

    print(f"[hf_checkpoint_sync] Pulling existing checkpoints from {rid} into {checkpoint_dir} ...")
    snapshot_download(rid, repo_type="model", local_dir=checkpoint_dir, token=_token())
    print("[hf_checkpoint_sync] Pull complete.")


def push_new_checkpoints(checkpoint_dir: str, already_pushed: set[str]) -> None:
    """Uploads any global_step_* folder not yet in already_pushed. Mutates
    already_pushed in place, only on confirmed success, so a failed
    upload is retried next call instead of being silently skipped."""
    from huggingface_hub import HfApi

    local = Path(checkpoint_dir)
    step_dirs = sorted(p for p in local.glob("global_step_*") if p.is_dir())
    pending = [p for p in step_dirs if p.name not in already_pushed]
    if not pending:
        return

    rid = repo_id()
    api = HfApi(token=_token())
    for step_dir in pending:
        try:
            api.upload_folder(
                repo_id=rid,
                repo_type="model",
                folder_path=str(step_dir),
                path_in_repo=step_dir.name,
                commit_message=f"checkpoint {step_dir.name}",
            )
            already_pushed.add(step_dir.name)
            print(f"[hf_checkpoint_sync] Pushed {step_dir.name} to {rid}.")
        except Exception as e:
            print(f"[hf_checkpoint_sync] WARNING: push of {step_dir.name} failed, will retry next poll: {e}")


def watch_and_push(checkpoint_dir: str, stop_event: threading.Event, poll_interval: int = 120) -> None:
    already_pushed: set[str] = set()
    while not stop_event.is_set():
        push_new_checkpoints(checkpoint_dir, already_pushed)
        stop_event.wait(poll_interval)
    # Final sweep in case a checkpoint landed between the last poll and stop.
    push_new_checkpoints(checkpoint_dir, already_pushed)
