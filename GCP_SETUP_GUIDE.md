# GCP Setup Guide — Phase 1 Training

Real Phase 1 training run (`src/training/phase1_config.py`) on a raw GCP
Compute Engine A100 40GB VM (`a2-highgpu-1g`, on-demand — see PLAN.md for
why on-demand over Spot). This replaces the Modal-based Steps 1-7
infrastructure only for the real, expensive Phase 1 run itself — Modal
remains valid for anything else in this project.

Scripts referenced here: `scripts/gcp/create_vm.sh`, `scripts/gcp/setup_env.sh`,
`scripts/gcp/run_training.sh`, `configs/gcp/Dockerfile.verl`,
`scripts/run_phase1_on_gcp.py`.

---

## 0. One-time GCP account setup (skip anything already done)

1. **Create/select a project** and confirm billing is enabled:
   ```
   gcloud projects create YOUR_PROJECT_ID   # or use an existing one
   gcloud config set project YOUR_PROJECT_ID
   ```
   Billing must be linked via the Console (Billing → Link a billing account) — not a gcloud-only step.

2. **Enable the Compute Engine API**:
   ```
   gcloud services enable compute.googleapis.com
   ```

3. **Check/request A100 GPU quota** — the single most common real blocker. Quota is *not* automatic even with billing enabled, and A100s aren't available in every region.
   - Console path (most reliable for a first check): **IAM & Admin → Quotas**, filter for "NVIDIA A100 GPUs", check your target region (e.g. `us-central1`) shows a limit ≥ 1.
   - If it shows 0, request an increase from the same page (usually approved within minutes to a day for a small request like 1 GPU).

4. **Install/authenticate the `gcloud` CLI** locally, or use **Cloud Shell** (console.cloud.google.com → Activate Cloud Shell) — has `gcloud` pre-installed, avoids a local install entirely. Either works for the commands below; Cloud Shell itself has no GPU, it's just used to create/manage the real GPU VM.
   ```
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

---

## 1. Create the VM

```bash
export GCP_PROJECT_ID=YOUR_PROJECT_ID
export GCP_ZONE=us-central1-a   # pick a zone where your A100 quota was approved
bash scripts/gcp/create_vm.sh
```

This creates `vlm-pass-k-phase1` (`a2-highgpu-1g`, 300GB SSD boot disk, a Deep Learning VM image with NVIDIA drivers pre-installed). Takes a few minutes.

**Verify the image family before relying on this unmodified** — version-dated suffixes change over time:
```
gcloud compute images list --project deeplearning-platform-release --filter="family~pytorch" --format="value(family)" | sort -u
```
Override via `export IMAGE_FAMILY=...` before running `create_vm.sh` if the current one differs from the script's default.

---

## 2. SSH in and run setup

```bash
gcloud compute ssh vlm-pass-k-phase1 --zone=$GCP_ZONE --project=$GCP_PROJECT_ID
```

On the VM:
```bash
curl -fsSL https://raw.githubusercontent.com/antrip03/VLM_pass-k/Guneesh/scripts/gcp/setup_env.sh -o setup_env.sh
bash setup_env.sh
```
(Or clone the repo first and run it from there — either works; the script clones the repo itself if it isn't present.)

This installs Docker + `nvidia-container-toolkit` (explicitly, not assumed present on the base image), **verifies real GPU passthrough** with an actual `nvidia-smi` container run (not just assumed working), clones the repo, and builds the veRL image (`configs/gcp/Dockerfile.verl` — the same real, already-debugged environment from `configs/modal_app_verl.py`, translated to plain Docker). The image build pulls a large base image and installs veRL — expect several minutes the first time, cached after.

---

## 2b. Weights & Biases login (do this on the VM, never anywhere else)

Training logs to console + wandb (`trainer.logger=[console,wandb]`, real syntax confirmed against docs.wandb.ai). The key is read from this shell's environment at run time — it's never written into any file or committed:
```bash
export WANDB_API_KEY=your-key-here
# or: wandb login
```
**Only ever paste a wandb key directly into a terminal on a machine you control** (like this VM) — never into a chat, ticket, or any other channel. If a key is ever pasted somewhere it shouldn't be, treat it as compromised and rotate it (wandb.ai → Settings → API keys) rather than trusting it's fine.

Training still runs without this set — `run_training.sh` will just warn that wandb won't authenticate; console logging is unaffected.

---

## 3. Calibrate before committing the full budget

Two things have never been tested for real, and both are cheap to check before the full 500-step run:

**a) Real throughput/VRAM at production batch shape.** Our $67 estimate is still based on a raw-HF proxy, not real veRL (with `use_fused_kernels` + vLLM rollout):
```bash
TOTAL_STEPS=5 bash scripts/gcp/run_training.sh
```
Check the output for: no OOM, a sane per-step time (compare against the raw-HF-proxy-based 11.1hr/500-step estimate in PLAN.md §9.2 — if real veRL is meaningfully faster, that's good news worth knowing before the full run), and that the reward signal is moving (not stuck at exactly 0 or exactly 1 for every sample).

**b) Resume-from-checkpoint actually works.** `trainer.resume_mode=auto` was just wired in (checkpoints were always being saved every 100 steps, but nothing was using them on a restart before now) — this needs a real kill-and-restart test, not just trust:
```bash
TOTAL_STEPS=10 bash scripts/gcp/run_training.sh &
sleep 60   # let it get partway, past the first save if save_freq is low enough for this quick test
kill %1
TOTAL_STEPS=10 bash scripts/gcp/run_training.sh   # should resume, not restart from step 0 - check the log output confirms this
```
If it doesn't resume cleanly, that's important to know now, not 8 hours into the real run.

---

## 4. Run the real thing

```bash
TOTAL_STEPS=500 AUTO_SHUTDOWN=true bash scripts/gcp/run_training.sh
```

`AUTO_SHUTDOWN=true` stops the VM automatically when training exits (success or failure) — this is the real, easy-to-miss risk of a raw VM vs. Modal: **it bills continuously until explicitly stopped**, unlike Modal's automatic per-invocation lifecycle. Checkpoints save to `/mnt/disks/data/checkpoints` every 100 steps (survives VM stop, since it's on the boot disk — lost only if the VM/disk is *deleted*, not stopped).

Expect ~11-18+ real hours depending on what calibration showed. Consider running this inside `tmux`/`screen` on the VM (or just let `AUTO_SHUTDOWN` handle it and reconnect later) so an SSH disconnect doesn't kill the job:
```bash
tmux new -s training
TOTAL_STEPS=500 AUTO_SHUTDOWN=true bash scripts/gcp/run_training.sh
# Ctrl+B, D to detach; `tmux attach -t training` to reattach later
```

---

## 5. Monitor from your local machine

```bash
gcloud compute ssh vlm-pass-k-phase1 --zone=$GCP_ZONE --project=$GCP_PROJECT_ID --command="tmux attach -t training"
```
Or just check whether the instance is still running (billing signal):
```bash
gcloud compute instances describe vlm-pass-k-phase1 --zone=$GCP_ZONE --format="value(status)"
```

---

## 6. When done: stop or delete

**Stop** (pauses compute billing, keeps the disk — checkpoints/model cache survive, small residual disk cost continues):
```bash
gcloud compute instances stop vlm-pass-k-phase1 --zone=$GCP_ZONE
```

**Delete** (stops all billing, including disk — only do this after retrieving checkpoints, e.g. via `gcloud compute scp`):
```bash
gcloud compute instances delete vlm-pass-k-phase1 --zone=$GCP_ZONE
```

---

## Known gaps / things not yet verified live
- The exact Deep Learning VM image family name (verify per step 1 above — version-dated, changes over time).
- Whether Docker ships pre-installed on this image family (not confirmed against official docs — `setup_env.sh` installs it explicitly regardless, so this doesn't matter in practice, just noting it wasn't assumed).
- Real veRL throughput/VRAM ceiling at S=128 with `use_fused_kernels` (exactly what step 3a's calibration run answers — do not skip it).
- `trainer.resume_mode=auto` (checkpoint resume) has never been tested with a real kill-and-restart (step 3b — do not skip it either; a crash 8 hours into the real run is the wrong time to discover this doesn't work).
