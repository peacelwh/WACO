#!/usr/bin/env bash
# LIBERO-Plus camera track of the paper (1,599 tasks, one episode each; released and corrected
# prompts; evaluation seeds 0, 1, 2). Run in the LIBERO-Plus environment
# (scripts/setup_liberoplus_env.sh) after scripts/download_assets.sh.
#
#   bash scripts/reproduce_liberoplus.sh public    # public pi05 LIBERO checkpoint, no runtime
#   bash scripts/reproduce_liberoplus.sh runtime   # + WACO runtime (no training)
#   bash scripts/reproduce_liberoplus.sh waco      # WACO: error-aware fine-tuning + runtime (released checkpoint)
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PUBLIC=${PUBLIC:-lerobot/pi05_libero_finetuned}
WACO=${WACO:-Bluexxx/WACO-pi05-LIBERO}
lp() {  # lp TAG POLICY CANON SEEDS
  for seed in $4; do
    POLICY=$2 TAG=$1_released_s$seed  CANON=$3 PROMPT=released  SEED=$seed bash scripts/eval_liberoplus_camera.sh
    POLICY=$2 TAG=$1_corrected_s$seed CANON=$3 PROMPT=corrected SEED=$seed bash scripts/eval_liberoplus_camera.sh
  done
}
case ${1:?usage: reproduce_liberoplus.sh public|runtime|waco} in
public)  lp lp_public  "$PUBLIC" none 0 ;;
runtime) lp lp_runtime "$PUBLIC" waco "0 1 2" ;;
waco)    lp lp_waco    "$WACO"   waco "0 1 2" ;;
esac
