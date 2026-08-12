#!/usr/bin/env bash
# Runs ON the GCP VM (after SSH-ing in) - one-time environment setup.
# Idempotent throughout: safe to re-run if it fails partway (e.g. quota
# hiccup, network blip) without redoing completed steps.
#
# Does NOT assume Docker/nvidia-container-toolkit are pre-installed on the
# Deep Learning VM image, even though they plausibly are - that wasn't
# confirmed against real docs, so this installs/configures them explicitly
# and then verifies GPU passthrough actually works, rather than trusting
# an unverified assumption.

set -euo pipefail

REPO_URL="https://github.com/antrip03/VLM_pass-k.git"
REPO_BRANCH="Guneesh"
REPO_DIR="$HOME/VLM_pass-k"
DATA_DIR="/mnt/disks/data"   # on the boot disk here for simplicity - see the guide for a separate-disk variant

echo "=== 1/4: Docker ==="
if ! command -v docker &>/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  echo "Docker installed. You may need to log out/in for group membership to take effect."
else
  echo "Docker already present."
fi

echo "=== 2/4: nvidia-container-toolkit (GPU passthrough into containers) ==="
if ! dpkg -l | grep -q nvidia-container-toolkit; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update -qq
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
else
  echo "nvidia-container-toolkit already present."
fi

echo "=== Verifying GPU passthrough (real check, not assumed) ==="
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

echo "=== 3/4: Repo ==="
if [ ! -d "$REPO_DIR" ]; then
  git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
  (cd "$REPO_DIR" && git checkout "$REPO_BRANCH" && git pull)
fi

echo "=== 4/4: Build the veRL image (configs/gcp/Dockerfile.verl) ==="
echo "This pulls a large base image (verlai/verl:vllm024.dev2) and clones+installs"
echo "verl - expect this to take a while the first time, cached after."
sudo docker build -t vlm-pass-k-verl:latest -f "$REPO_DIR/configs/gcp/Dockerfile.verl" "$REPO_DIR"

mkdir -p "$DATA_DIR/hf_cache" "$DATA_DIR/checkpoints"

echo ""
echo "Setup complete. Data/checkpoint/model-cache directory: $DATA_DIR"
echo "Next: scripts/gcp/run_training.sh (see the guide for the calibration-first workflow)."
