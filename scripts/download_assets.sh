#!/usr/bin/env bash
# Download the released assets into data/ and outputs/:
#   * calibration bundles, few-shot splits and LIBERO-Plus camera-track ids (always)
#   * per-frame metric depth for the error-aware fine-tuning, 8.4 GB   (WITH_DEPTH=1)
#   * the released WACO checkpoint for the LIBERO-Plus camera track    (WITH_MODEL=1)
# Public inputs (lerobot/libero, lerobot/pi05_base, lerobot/pi05_libero_finetuned) are fetched
# by LeRobot on first use.
set -euo pipefail
ROOT=${POSESHIFT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATASET=${DATASET:-Bluexxx/LIBERO-PoseShift}
MODEL=${MODEL:-Bluexxx/WACO-pi05-LIBERO}
EXCLUDE=(--exclude "depth_dav2_128.npy")
[[ ${WITH_DEPTH:-0} == 1 ]] && EXCLUDE=()
hf download "$DATASET" --repo-type dataset --local-dir "$ROOT/data" "${EXCLUDE[@]}"
if [[ ${WITH_MODEL:-0} == 1 ]]; then
  hf download "$MODEL" --local-dir "$ROOT/outputs/train/waco_pi05_libero/pretrained_model"
fi
echo "assets ready under $ROOT/data"
