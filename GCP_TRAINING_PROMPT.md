# Prompt for Claude Code (VS Code) — Execute Phase 1 Training on GCP

Copy everything below into your Claude Code session in VS Code.

---

You are driving a real, billed GCP VM to run the Phase 1 training run for a
research project (VLM_pass-k — testing whether RLVR's reasoning gain on a
VLM survives a text→image modality shift). Everything needed is already
built and merged into the repo's `Guneesh` branch; your job is EXECUTION,
not design — follow the sequence below, verify each step's real output
before moving to the next, and stop to report back (don't guess or
assume success) if anything looks wrong.

## Target infrastructure
- GCP project: `steering-505317`
- Zone: `asia-east1-b`
- Instance: `instance-20260812-172906`
- Repo: `https://github.com/antrip03/VLM_pass-k.git`, branch `Guneesh`

If you're connecting via `gcloud compute ssh`, use:
```bash
gcloud compute ssh --zone "asia-east1-b" "instance-20260812-172906" --project "steering-505317"
```
If instead you're already running natively ON the VM (e.g. VS Code Remote-SSH
directly into it), skip straight to step 0 below.

## Step 0 — Verify what you're actually working with (do this first, don't assume)
```bash
gcloud compute instances describe instance-20260812-172906 --zone=asia-east1-b --project=steering-505317 --format="value(machineType)"
nvidia-smi
```
This tells you whether this is an A100 (`a2-highgpu-1g`) or L4 (`g2-standard-8`/`g2-standard-24`) instance,
and confirms the GPU is actually visible. This matters because:
- **A100**: we have real measured throughput/VRAM data (raw-HF proxy, not yet real-veRL-verified) — see `PLAN.md` §9.2.
- **L4**: we have ZERO real data for the 3B model at this VRAM tier (24GB raw, ~22GB usable). Our own repo's docstring
  (`scripts/vram_smoke_test_2000cap_a10g.py`) already shows a real finding that the smallest known-working training
  micro-batch on A100 (29.32GB peak) *exceeds* a 24GB-class GPU's usable capacity — on the raw-HF proxy, without
  the fused kernel real veRL uses. This is exactly why calibration (step 3 below) is not optional if this is L4.

Report back what machine type and GPU you actually have before proceeding.

## Step 1 — Environment setup (idempotent, safe to re-run if it fails partway)
```bash
curl -fsSL https://raw.githubusercontent.com/antrip03/VLM_pass-k/Guneesh/scripts/gcp/setup_env.sh -o setup_env.sh
bash setup_env.sh
```
This installs Docker + nvidia-container-toolkit (explicitly, not assumed pre-installed), **verifies real GPU
passthrough** with an actual `nvidia-smi` container run, clones the repo to `~/VLM_pass-k`, and builds the training
image (`configs/gcp/Dockerfile.verl` — real, already-debugged veRL environment, same one validated on Modal
earlier in this project; base image pull + verl install takes several minutes the first time).

**Verify it actually passed** — look for the `nvidia-smi` output inside the Docker-run GPU check succeeding, and
the final "Setup complete" message. Do not proceed on a partial/failed run.

## Step 2 — wandb login
```bash
export WANDB_API_KEY=<the user's rotated wandb key — ask the user for this directly, do not guess or reuse any
key that may already be visible elsewhere; a real key was accidentally pasted into an earlier chat session and
was told to be rotated — if you're not certain the key you have is the rotated one, ask before using it>
```
Training logs to `console + wandb` (`trainer.logger=[console,wandb]`, already wired into
`src/training/phase1_config.py`). Training still runs fine without this — it just won't authenticate the wandb
logger — so don't block on this if the user isn't ready with a key yet, just flag it and continue.

## Step 3 — Calibration (mandatory, do not skip to the real run without this)
Two separate real things need verification, neither has ever been tested for real:

**3a. Real throughput/VRAM at production batch shape** (S=128 sequences/step — our $67/~18.3hr Phase 1 estimate is
still based on a raw-HF proxy test, not real veRL with `use_fused_kernels` + vLLM rollout):
```bash
cd ~/VLM_pass-k
TOTAL_STEPS=5 bash scripts/gcp/run_training.sh
```
Check: no OOM, a sane per-step time, and that the reward signal in the logs is actually moving (not stuck at
exactly 0 or exactly 1 for every sample — that would indicate a broken reward function or extraction bug, not
just slow training). Report the actual numbers back, don't just say "it worked."

**3b. Checkpoint resume actually works** (`trainer.resume_mode=auto` is wired in, but never verified live):
```bash
TOTAL_STEPS=10 bash scripts/gcp/run_training.sh &
sleep 60
kill %1
TOTAL_STEPS=10 bash scripts/gcp/run_training.sh
```
The second invocation should resume from a saved checkpoint, not restart from step 0 — check the log output
explicitly confirms this (veRL prints which step it's resuming from). If it silently restarts from 0, STOP and
report this back — do not proceed to the real run with an unverified resume path, especially if this VM is Spot
(check step 0's machine-type output / `gcloud compute instances describe ... --format="value(scheduling.provisioningModel)"` —
if it says `SPOT`, preemption is a real, expected possibility over an 11-30 hour run, not a hypothetical).

## Step 4 — The real run (only after 3a and 3b both pass)
```bash
tmux new -s training
TOTAL_STEPS=500 AUTO_SHUTDOWN=true bash scripts/gcp/run_training.sh
# Ctrl+B then D to detach; reattach later with: tmux attach -t training
```
`AUTO_SHUTDOWN=true` stops the VM automatically when training exits (pass or fail) — this matters because a raw
GCP VM bills continuously until stopped, unlike ephemeral per-job compute. Checkpoints save every 100 steps to
`/mnt/disks/data/checkpoints` and survive a VM stop (only lost if the disk itself is deleted).

Expected real duration: **~11 hours if this is confirmed A100** (per `PLAN.md` §9.2, still pending real-veRL
verification from step 3a), **genuinely unknown for L4** (rough spec-based bound was 29-58 hours, never measured
for real — step 3a's actual numbers are the real answer, use those over this estimate).

## Cost awareness
This is real, billed compute. On-demand A100 ≈ $3.67/hr; L4 ≈ $0.70/hr (single) or roughly double for 2x L4. Do
not let this run unattended for the full duration without at least one check-in — `AUTO_SHUTDOWN` protects
against runaway idle billing after completion/failure, but won't catch a stuck-but-still-running process burning
budget without producing useful output. If step 3a showed something concerning (OOM, no reward movement, absurd
per-step time), stop and report back rather than proceeding to step 4 anyway.

## What to report back to the user once done (or once blocked)
- Step 0: actual machine type / GPU confirmed
- Step 3a: real per-step time, whether it matched or diverged from the estimate, whether reward moved
- Step 3b: whether resume actually worked
- Step 4: final status (completed 500 steps / failed at step N / still running), checkpoint location, wandb run URL if logging was enabled
