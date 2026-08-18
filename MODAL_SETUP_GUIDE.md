# Modal Setup Guide — Phase 1 Training

Real Phase 1 training run (`src/training/phase1_config.py`) on Modal, on
your own Modal account. This is the Modal-based counterpart to
`GCP_SETUP_GUIDE.md` — same training config, same checkpointing/resume
behavior, different infrastructure. Use this one if you're running on
Modal; use `GCP_SETUP_GUIDE.md` if you're running on a raw GCP VM. If
you're specifically doing a split run between two people across both
environments, see `HANDOFF.md` instead, which covers the handoff/transfer
mechanics on top of what's here.

Script used throughout: `scripts/run_phase1_on_modal.py`.

---

## 1. One-time account setup

1. **Create a Modal account** at [modal.com](https://modal.com) if you don't have one — free to sign up, GPU usage is billed separately (see Cost, below).
2. **Install and authenticate the CLI**, on whichever machine you'll run `modal run` from (your laptop is fine — the actual training runs on Modal's cloud, not your machine):
   ```bash
   pip install modal
   modal setup
   ```
   This opens a browser to link the CLI to your account.

## 2. Get the code
```bash
git clone --branch Guneesh https://github.com/antrip03/VLM_pass-k.git
cd VLM_pass-k
```

## 3. wandb setup (optional, for live training curves)
```bash
export WANDB_API_KEY=your-key-here
```
Get a key from [wandb.ai](https://wandb.ai) → Settings → API keys. **Only ever paste this into your own terminal** — never into a chat, file, or anything that gets committed. This is read from your *local* shell (wherever you run `modal run` from) and passed through securely per-run — no Modal Secret needs to be pre-created in the dashboard.

If you skip this, training still runs correctly — it just logs to console only, with `WANDB_MODE=disabled` passed through automatically so nothing hangs or errors waiting for a login it can't get.

## 4. Run it
```bash
modal run scripts/run_phase1_on_modal.py --total-training-steps 500
```
This prepares the full GSM8K train split (idempotent — skips if already done), then launches real training on an A100-40GB. `--total-training-steps 500` is the real Phase 1 target; you can pass a smaller number (e.g. `50`) first as a quick sanity check if you want to see it run end-to-end before committing the full budget.

## 5. Checkpointing and resume — what's actually protected

- Checkpoints save automatically every 100 steps to a dedicated, persistent Modal Volume (`grpo-vlm-phase1-checkpoints`) — separate from any earlier plumbing-check volumes, so nothing gets mixed up.
- `trainer.resume_mode=auto` means **re-running the exact same command** looks for and continues from the latest checkpoint on that Volume automatically — it does not restart from step 0.
- If the run hits its 20-hour per-invocation ceiling (a real possibility for a long run — Modal functions can't run forever in one call), you'll see a clear message telling you this is expected and to just re-run the same command — not a crash, not lost progress.
- **Not yet verified with a real live test**: whether `resume_mode=auto` actually resumes correctly has never been tested end-to-end on Modal specifically. Before trusting a long unattended run, do a cheap real check first:
  ```bash
  modal run scripts/run_phase1_on_modal.py --total-training-steps 10
  # let it finish, then run again with the same or higher target:
  modal run scripts/run_phase1_on_modal.py --total-training-steps 10
  ```
  The second invocation should be noticeably faster and should not redo all 10 steps from scratch — check the printed output near the start for evidence it picked up from a saved checkpoint rather than starting over.

## 6. Checking progress
If wandb is set up: go to wandb.ai, project `grpo-vlm-modality-shift`, experiment `qwen2_5_vl_3b_phase1`. Also worth watching (under `rollout_corr/` in the same dashboard): `rollout_corr/kl` and `rollout_corr/log_ppl_abs_diff` — these track a real, documented mismatch between Modal's rollout engine (vLLM) and the training engine that can destabilize RL training if large. A mitigation (`disable_cascade_attn`) is already on in the config; these metrics let you confirm it's actually working rather than just trusting it.

Without wandb, watch the console output from `modal run` directly, or check the [Modal dashboard](https://modal.com/apps) for the running app's logs.

## 7. Cost
Modal's A100-40GB rate is roughly **~$2/hr** (check Modal's own current pricing page for the exact live rate — third-party trackers gave inconsistent numbers when checked, not trusted here). At the real ~11.1hr/500-step estimate from `PLAN.md` §9.2 (itself still based on a raw-HF proxy, not real-Modal-measured throughput), that's roughly **~$22** for training — cheaper than the GCP on-demand A100 estimate (~$41), though neither number is confirmed against real veRL throughput yet. Treat both as priors, not commitments.

## 8. If something goes wrong
- **`modal run` fails immediately with an auth error**: re-run `modal setup`.
- **Training crashes early (not a timeout)**: check the printed stdout/stderr tail from the failed run — this is genuinely a new environment combination (first real execution of this exact production config on Modal), so a first-run issue wouldn't be surprising given this project's track record elsewhere. Report the actual error rather than guessing at a fix.
- **Unsure if it's actually making progress**: check the Modal dashboard's live logs for the running function, or wandb if configured — the console output prints real per-step information, not just silence until it's done.
