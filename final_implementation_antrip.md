# Phase 1 Training — Final Implementation Guide (Modal)

Everything needed to run the real Phase 1 training for this project on
Modal, cleanly, in one command after authentication. This is the
definitive reference for that — for the full research design and its
justification, see `PLAN.md`; for the engineering story of how this was
built (11 real bugs found and fixed along the way), see
`IMPLEMENTATION_LOG.md`. This file is about running it correctly, right now.

---

## What this trains

Real RLVR (Dr. GRPO) fine-tuning of `Qwen2.5-VL-3B-Instruct` on GSM8K,
text-only — testing whether a reasoning gain from text-only RL survives
being tested on images afterward. Training itself never touches images.

**Every hyperparameter below is already decided, justified, and locked in
`src/training/phase1_config.py`** — this is not a config you need to
tune, just run:

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Algorithm | Dr. GRPO (group size 8, 16 prompts/step, 128 sequences/step) |
| Steps | 500 (checkpoint every 100) |
| LoRA | rank 32, alpha 32 |
| Learning rate | 1e-6, no KL penalty |
| Seed | 0 |
| vLLM fix | `disable_cascade_attn=True` (real training-inference mismatch fix) |
| Logging | console + wandb, `rollout_corr/*` mismatch metrics enabled |
| Resume | `trainer.resume_mode=auto` |

---

## Prerequisites (one-time)

1. **A Modal account** — [modal.com](https://modal.com), free to sign up.
2. **A wandb account and API key** — [wandb.ai](https://wandb.ai) → Settings → API keys. **This project requires wandb logging to actually work — it is not optional.** The run script below will refuse to start training if a valid key isn't available, rather than silently running without logging.
3. **Local tools**:
   ```bash
   pip install modal wandb
   modal setup
   ```
   `modal setup` opens a browser to link the CLI to your Modal account.

## Get the code
```bash
git clone --branch Guneesh https://github.com/antrip03/VLM_pass-k.git
cd VLM_pass-k
```

## Set your wandb key (once)
```bash
echo "WANDB_API_KEY=your-real-key-here" > .env
```
`.env` is git-ignored — it never gets committed, never leaves your machine. The run script sources it automatically every time, so this is a one-time step, not something to repeat.

**Never paste a real key into a chat, ticket, or anything that gets committed.** If a key is ever exposed somewhere it shouldn't be, treat it as compromised and generate a new one — that applies regardless of intent, since a leaked key gets abused by third parties, not just whoever exposed it.

---

## The single command

```bash
bash scripts/run_modal.sh
```

That's it. This script:
1. Loads your wandb key from `.env`.
2. **Actually verifies the key works** — a real server-side check (`wandb login --verify`), not just confirming the string is non-empty. If it doesn't authenticate, the script stops here with a clear error and does **not** spend any Modal compute.
3. Launches the real 500-step training run on an A100-40GB via `scripts/run_phase1_on_modal.py`.

If you want a quick sanity check first instead of committing to the full run:
```bash
TOTAL_STEPS=50 bash scripts/run_modal.sh
```

---

## Checkpointing and resume — what's actually protected

- Checkpoints save automatically every 100 steps to a dedicated, persistent Modal Volume (`grpo-vlm-phase1-checkpoints`).
- Re-running `bash scripts/run_modal.sh` **continues from the latest checkpoint automatically** — it does not restart from step 0. `trainer.resume_mode=auto` handles this; you don't pass any special flags.
- If a run hits its 20-hour per-invocation ceiling (Modal functions can't run forever in one call), you'll see a clear "hit the timeout, this is expected, re-run to resume" message, not a crash.
- **Honest gap**: resume has never been tested with a real live interruption on Modal specifically. Before trusting a long unattended run, verify it yourself once:
  ```bash
  TOTAL_STEPS=10 bash scripts/run_modal.sh
  # let it finish, then:
  TOTAL_STEPS=10 bash scripts/run_modal.sh
  ```
  The second run should be noticeably faster and should mention resuming from a saved checkpoint in its output, not silently redo all 10 steps.

---

## Monitoring

Go to [wandb.ai](https://wandb.ai), project `grpo-vlm-modality-shift`, experiment `qwen2_5_vl_3b_phase1`. Watch:
- **Reward** — should trend upward over the run, not stay flat or collapse.
- **`rollout_corr/kl`** and **`rollout_corr/log_ppl_abs_diff`** — track a real, documented mismatch between the rollout engine (vLLM) and the training engine that can silently destabilize RL training. A fix (`disable_cascade_attn`) is already on; these metrics let you confirm it's actually working rather than just trusting it. If `rollout_corr/kl` looks unusually large or keeps climbing, that's worth flagging, not letting the run continue for hours on it.

You can also watch the raw console output from `modal run`, or check the [Modal dashboard](https://modal.com/apps) for the running app's live logs.

---

## Cost

Modal's A100-40GB rate is roughly **~$2/hr** (verify against Modal's own current pricing page — third-party trackers gave inconsistent numbers when checked). At the ~11.1hr/500-step estimate in `PLAN.md` §9.2 (itself still based on a raw-HF proxy, not real-Modal-measured throughput), that's roughly **~$22** for training. Treat this as a prior, not a commitment — the first real run on Modal is also the first real measurement of actual throughput here.

---

## If something goes wrong

- **`bash scripts/run_modal.sh` says wandb verification failed**: your key is wrong, expired, or mistyped. Get a fresh one from wandb.ai and update `.env`.
- **`modal run` fails with an auth error**: re-run `modal setup`.
- **Training crashes (not the wandb check, not a timeout)**: this is the first real execution of this exact production config on Modal — a first-run issue wouldn't be surprising given how this project has gone on every new environment so far (11 real bugs found and fixed getting the GCP/Modal paths working in the first place). Capture the actual printed error and report it rather than guessing at a fix.
- **Unsure if it's making real progress**: check wandb or the Modal dashboard's live logs — the console prints real per-step information throughout, not just silence until done.

---

## What this does NOT cover

This guide is training only. Evaluation (the T/D/E sampling harness, pass@k, bootstrap CIs that actually answer this project's research question) is a separate phase, already built (`src/inference/`, `src/metrics/`) but not part of this run. Come back to `PLAN.md` §3 Phase 1 items 7–9 once training completes.
