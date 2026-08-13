# Phase 1 Training — Status, Plan, and Handoff Instructions

Written 2026-08-13. This is the single reference for: exactly where this
project stands, exactly how Phase 1 training will be run, and exactly how
to hand a partially-completed run from one person/environment to another.
For the full research design and its justification, see `PLAN.md`; for
the engineering story of how Steps 1-7 were built, see
`IMPLEMENTATION_LOG.md`. This file is about **execution**, not design.

---

## 1. Exact current status

**Decided and locked** (all in `src/training/phase1_config.py` + `PLAN.md`):
- Model: `Qwen/Qwen2.5-VL-3B-Instruct`, 1 seed (`data.seed=0`)
- Algorithm: Dr. GRPO, text-only training (GSM8K), never images
- Batch shape: group size 8, 16 prompts/step → S=128 sequences/step (`ppo_mini_batch_size=128`, confirmed global not per-GPU)
- Steps: 500 target, checkpoint every 100 steps
- LoRA: rank 32, alpha 32 (real RLVR ablation evidence, not the original unverified veRL-example default of 64)
- `lr=1e-6`, `use_kl_loss=False` (matches Dr. GRPO's real recipe, not veRL's generic example defaults)
- Eval: 50 problems (budget-constrained, below real paper precedent — see `PLAN.md` §9.3), n=128 samples/problem, T/D/E conditions
- `trainer.resume_mode=auto`, `trainer.logger=[console,wandb]`
- No QLoRA (doesn't target our actual VRAM bottleneck — see prior discussion; not applicable regardless)

**Not yet done, real and unverified**:
- The actual training run itself (this document exists to get it started)
- A real kill-and-restart test confirming `resume_mode=auto` actually works (mandatory before trusting any handoff — see §5)
- Real veRL throughput/VRAM at production batch shape (our time estimates are still raw-HF-proxy-based)

**Infrastructure that exists**:
- GCP path: `scripts/gcp/*.sh`, `configs/gcp/Dockerfile.verl`, `scripts/run_phase1_on_gcp.py` — see `GCP_SETUP_GUIDE.md`
- Modal path: `scripts/run_phase1_on_modal.py` (new, written alongside this document) — reuses the already-validated Modal veRL environment from `configs/modal_app_verl.py`

---

## 2. The plan

Your friend runs the first portion of training on **Modal**, then hands off the checkpoint to you, and you complete the remaining steps on **GCP**. Both sides target the same `total_training_steps=500` — nobody needs to coordinate an exact split point. Whoever's turn it is just runs until they want to stop, checkpoints save automatically every 100 steps, and the next person resumes from the latest one.

**Both of you must independently verify `resume_mode=auto` actually works in your own environment before relying on it for the real handoff** (§5.1 for Modal, `GCP_SETUP_GUIDE.md` step 3b for GCP) — Modal and GCP are different infrastructure paths; one working doesn't guarantee the other does.

---

## 3. Friend's setup (Modal)

### 3.1 Prerequisites
- A Modal account ([modal.com](https://modal.com)), `modal` CLI installed (`pip install modal`) and authenticated (`modal setup`).
- Clone the repo, `Guneesh` branch:
  ```bash
  git clone --branch Guneesh https://github.com/antrip03/VLM_pass-k.git
  cd VLM_pass-k
  ```

### 3.2 wandb setup — via a Modal Secret, never pasted anywhere
Go to your Modal dashboard → **Secrets** → **Create new secret** → choose the **Weights & Biases** template (or a custom secret) → name it exactly `wandb-secret` → set the key `WANDB_API_KEY` to your real wandb API key. This is entered directly into Modal's own secret storage, never through any chat, file, or terminal history. `scripts/run_phase1_on_modal.py` references this secret by name (`modal.Secret.from_name("wandb-secret")`) — if it doesn't exist under that exact name, the run will fail with a clear "secret not found" error, not silently skip wandb.

### 3.3 Launch training
```bash
modal run scripts/run_phase1_on_modal.py --total-training-steps 500
```
This prepares the full real GSM8K train split (not a tiny slice — idempotent, only writes once), then launches the real training run on an A100-40GB. Checkpoints save every 100 steps to the `grpo-vlm-phase1-checkpoints` Modal Volume automatically — this happens regardless of when you stop.

### 3.4 Stopping / when to hand off
Just interrupt it (Ctrl+C on the `modal run` command, or let it finish if you're doing the whole thing) whenever you've done your share. There's no need to hit an exact step count — the next checkpoint save (every 100 steps) is your real handoff point. If you stop between saves, whatever happened since the last save is lost (same as any other crash) — not catastrophic, just wasted compute for that partial stretch.

### 3.5 Export the checkpoint for handoff
```bash
modal volume get grpo-vlm-phase1-checkpoints / ./phase1_checkpoint_handoff
```
This downloads the **entire** checkpoint directory tree (all `global_step_N/` folders, not just the latest) to your local machine — transfer the whole thing, don't cherry-pick, to avoid missing files the resume logic needs.

Then get `./phase1_checkpoint_handoff` to your friend. **Recommended: zip it and upload to Google Drive**, share the link — lowest-friction option that doesn't require your friend to have any Modal or GCP access on your end:
```bash
zip -r phase1_checkpoint_handoff.zip phase1_checkpoint_handoff/
```
(Real size unknown until you actually check — could be a few GB depending on what `checkpoint.save_contents=[model,hf_model]` actually persists. Check the folder size before uploading so you know what you're dealing with.)

---

## 4. Your setup (GCP) — full detail in `GCP_SETUP_GUIDE.md`

Follow `GCP_SETUP_GUIDE.md` through Step 2 (VM created, environment set up, wandb logged in with **your own** key — same rule, never share/reuse a key that passed through any chat). Then:

### 4.1 Import the handed-off checkpoint
On your GCP VM:
```bash
pip install gdown   # if not already present
gdown "https://drive.google.com/uc?id=YOUR_FILE_ID" -O phase1_checkpoint_handoff.zip
unzip phase1_checkpoint_handoff.zip -d /mnt/disks/data/checkpoints_import
# Merge/move its contents into the real checkpoint dir the training script expects:
mv /mnt/disks/data/checkpoints_import/phase1_checkpoint_handoff/* /mnt/disks/data/checkpoints/
```
(Adjust paths to match whatever your friend's zip actually contains — check with `unzip -l` first if unsure.)

### 4.2 Continue training
```bash
TOTAL_STEPS=500 AUTO_SHUTDOWN=true bash scripts/gcp/run_training.sh
```
Same `total_training_steps=500` as your friend used — `resume_mode=auto` should detect the imported checkpoint and continue from wherever it left off, not restart from step 0.

**Verify this actually happened** — check the console log output near the start of the run for veRL explicitly stating which step it resumed from. If it says step 0, the checkpoint wasn't found/read correctly — stop and re-check the import path before letting it run for hours.

---

## 5. Checkpoint safety — every guardrail, in one place

Since this run is split across two people/environments, here's every real protection currently in place, and what each one does and doesn't cover:

| Guardrail | What it protects against | What it does NOT cover |
|---|---|---|
| `trainer.save_freq=100` | Losing more than 100 steps of progress to any crash, on either side | Progress since the last save, always |
| `trainer.resume_mode=auto` | Manually re-running training after a stop/crash silently restarting from step 0 | **Unverified in practice** — must be tested (§5.1) before trusting it |
| Modal Volume (`grpo-vlm-phase1-checkpoints`) | Checkpoints surviving between separate `modal run` invocations on the friend's side | Nothing outside Modal — must be explicitly exported (§3.5) to reach you |
| GCP boot disk persistence | Checkpoints surviving a GCP VM *stop* (not delete) on your side | Disk deletion — never delete the VM/disk before confirming the run is actually complete |
| `data.seed=0` | Data shuffling order being reproducible | NOT full bit-for-bit run reproducibility (LoRA init, vLLM sampling have their own randomness — see `phase1_config.py` docstring) |

### 5.1 Mandatory: test resume on BOTH sides before the real handoff
**Friend, on Modal**, before doing your real run:
```bash
modal run scripts/run_phase1_on_modal.py --total-training-steps 10
# let it run past at least one save point if save_freq allows, or just re-invoke:
modal run scripts/run_phase1_on_modal.py --total-training-steps 10
# second invocation should resume, not restart - check console output
```
**You, on GCP**: `GCP_SETUP_GUIDE.md` step 3b (same idea, already written there).

Do not skip either side's test. A crash or handoff mistake discovered 15 hours into a real run is a much worse time to learn resume doesn't work than now.

---

## 6. wandb notes
Both of you log to the same project (`trainer.project_name=grpo-vlm-modality-shift`) and experiment name (`qwen2_5_vl_3b_phase1`), but **each separate invocation of `main_ppo` will likely create its own wandb run** (no explicit run-ID continuity is wired in) — expect to see two runs in the wandb UI (friend's portion, then yours), not one continuous curve. This is cosmetic only; it doesn't affect training correctness, since the actual continuity (model weights, optimizer state, step count) lives in the checkpoint, not in wandb. If you want a single visual curve later, you can manually note the step ranges when comparing the two runs.

---

## 7. After training completes
Both halves done, 500 steps reached — next is evaluation (Steps 6/7's T/D/E sampling harness + bootstrap CI, already built and validated), not covered in this document. Come back to `PLAN.md` §3 Phase 1 items 7-9 and the existing `src/inference/`, `src/metrics/` modules for that.
