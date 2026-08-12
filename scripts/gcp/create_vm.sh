#!/usr/bin/env bash
# Creates the real Phase 1 GCP VM: a2-highgpu-1g (1x A100 40GB, on-demand,
# per PLAN.md/the on-demand decision - not Spot).
#
# Prerequisites this does NOT handle for you (see the guide this ships
# with for exact commands):
#   1. A GCP project with billing enabled.
#   2. Compute Engine API enabled.
#   3. A100 GPU quota approved for the chosen region - this is the single
#      most common real blocker; quota is NOT automatic even with billing
#      enabled, and A100s are not available in every region.
#
# The image family below is a Deep Learning VM image (confirmed via
# cloud.google.com/deep-learning-vm/docs/images, 2026-08-12) - comes with
# the NVIDIA driver pre-installed (avoids the fragile manual driver-install
# path), but Docker/nvidia-container-toolkit are installed explicitly in
# setup_env.sh regardless, rather than assumed present - that was NOT
# confirmed in the docs, so this doesn't rely on an unverified assumption.
# Image family names carry a version-date suffix that changes over time -
# verify the current one before relying on this script unmodified:
#   gcloud compute images list --project deeplearning-platform-release \
#       --filter="family~pytorch" --format="value(family)" | sort -u

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID first, e.g. export GCP_PROJECT_ID=your-project}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-vlm-pass-k-phase1}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-300GB}"

echo "Creating $INSTANCE_NAME (a2-highgpu-1g, A100 40GB) in $ZONE, project $PROJECT_ID..."

gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type=a2-highgpu-1g \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$BOOT_DISK_SIZE" \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --no-restart-on-failure

echo ""
echo "VM created. Next steps:"
echo "  1. SSH in:  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo "  2. On the VM, run:  bash scripts/gcp/setup_env.sh  (after cloning the repo - see the guide)"
echo ""
echo "IMPORTANT: this VM bills continuously once created, even if idle. When"
echo "you're done for the day, stop it (billing for compute pauses, disk"
echo "storage keeps a small residual cost):"
echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
