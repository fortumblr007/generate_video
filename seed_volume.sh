#!/bin/bash
# Seed a RunPod Network Volume with required model weights (~45GB).
#
# Usage (on a Pod with the volume attached at /runpod-volume):
#   bash seed_volume.sh
#   # or from this repo after clone:
#   MODELS_ROOT=/runpod-volume/models bash download_models.sh
#
# Then attach the SAME volume to your Serverless endpoint (same datacenter).
set -euo pipefail

VOLUME_ROOT="${NETWORK_VOLUME_PATH:-/runpod-volume}"
if [ ! -d "$VOLUME_ROOT" ]; then
  echo "ERROR: $VOLUME_ROOT not found."
  echo "Create a Network Volume in the RunPod console, attach it to this Pod,"
  echo "and ensure it mounts at /runpod-volume."
  exit 1
fi

export MODELS_ROOT="${MODELS_ROOT:-$VOLUME_ROOT/models}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Seeding MODELS_ROOT=$MODELS_ROOT"
bash "$SCRIPT_DIR/download_models.sh"
echo "Done. Directory sizes:"
du -sh "$MODELS_ROOT"/* 2>/dev/null || true
df -h "$VOLUME_ROOT" | tail -1
