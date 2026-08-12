#!/usr/bin/env bash
# Creates the real Phase 1 GCP VM. Defaults to a2-highgpu-1g (1x A100
# 40GB, on-demand, per PLAN.md's on-demand decision - not Spot) - override
# via MACHINE_TYPE for the L4 fallback path (A100 quota denied, real
# constraint hit 2026-08-12):
#   MACHINE_TYPE=g2-standard-24 bash scripts/gcp/create_vm.sh   # 2x L4
#   MACHINE_TYPE=g2-standard-8  bash scripts/gcp/create_vm.sh   # 1x L4
# The GPU is a fixed property of the machine type itself on GCP (a2-*
# always ships with A100s, g2-* always ships with L4s) - no separate
# --accelerator flag needed either way.
#
# Prerequisites this does NOT handle for you (see the guide this ships
# with for exact commands):
#   1. A GCP project with billing enabled.
#   2. Compute Engine API enabled.
#   3. GPU quota approved for the chosen region AND machine family - the
#      single most common real blocker; quota is NOT automatic even with
#      billing enabled, not available in every region, and is tracked
#      SEPARATELY per GPU type (A100 quota being 0 doesn't mean L4 quota
#      is also 0, and vice versa - check the specific one you're using).
#
# SPOT=true adds --provisioning-model=SPOT (real cost savings, real
# preemption risk - only reasonably safe now that trainer.resume_mode=auto
# is wired into phase1_config.py AND the real kill-and-restart test in
# GCP_SETUP_GUIDE.md step 3b has actually been run, not just written):
#   SPOT=true MACHINE_TYPE=g2-standard-8 bash scripts/gcp/create_vm.sh
# --instance-termination-action=STOP is set explicitly (not left to the
# default) - confirmed via docs.cloud.google.com/compute/docs/instances/spot
# (fetched 2026-08-12) that STOP actually IS the real default when
# preempted (stops the VM, disk survives, resumable later), not DELETE as
# might be assumed - but pinning it explicitly here means this script's
# behavior doesn't silently change if that default is ever revised.
#
# The image family below is a Deep Learning VM image (confirmed via
# cloud.google.com/deep-learning-vm/docs/images, 2026-08-12) - comes with
# the NVIDIA driver pre-installed (avoids the fragile manual driver-install
# path), but Docker/nvidia-container-toolkit are installed explicitly in
# setup_env.sh regardless, rather than assumed present - that was NOT
# confirmed in the docs, so this doesn't rely on an unverified assumption.
# Deliberately NOT the plain "Debian" default image the Console wizard
# suggests - that has no NVIDIA driver, no Docker, and would hit the
# fragile manual-install path this whole approach exists to avoid.
# Image family names carry a version-date suffix that changes over time -
# verify the current one before relying on this script unmodified:
#   gcloud compute images list --project deeplearning-platform-release \
#       --filter="family~pytorch" --format="value(family)" | sort -u

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID first, e.g. export GCP_PROJECT_ID=your-project}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-vlm-pass-k-phase1}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-1g}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="deeplearning-platform-release"
# 300GB regardless of machine type: the Docker base image alone
# (verlai/verl:vllm024.dev2) is likely 10-20GB, plus the veRL install,
# model cache (~7GB+ for Qwen2.5-VL-3B), and checkpoint saves on top -
# a 10GB disk (the Console wizard's default) fails during setup, not
# training.
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-300GB}"
SPOT="${SPOT:-false}"

SPOT_ARGS=()
if [ "$SPOT" = "true" ]; then
  SPOT_ARGS=(--provisioning-model=SPOT --instance-termination-action=STOP)
  echo "SPOT=true: this VM can be preempted at any time. Do NOT proceed to the"
  echo "real 500-step run until GCP_SETUP_GUIDE.md step 3b's real kill-and-restart"
  echo "test has actually been run and confirmed resume_mode=auto works."
fi

echo "Creating $INSTANCE_NAME ($MACHINE_TYPE, spot=$SPOT) in $ZONE, project $PROJECT_ID..."

gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$BOOT_DISK_SIZE" \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --no-restart-on-failure \
  "${SPOT_ARGS[@]}"

echo ""
echo "VM created. Next steps:"
echo "  1. SSH in:  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo "  2. On the VM, run:  bash scripts/gcp/setup_env.sh  (after cloning the repo - see the guide)"
echo ""
echo "IMPORTANT: this VM bills continuously once created, even if idle (or, if"
echo "SPOT=true, can be preempted at any time). When you're done for the day,"
echo "stop it (billing for compute pauses, disk storage keeps a small"
echo "residual cost):"
echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
