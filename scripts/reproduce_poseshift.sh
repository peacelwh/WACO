#!/usr/bin/env bash
# LIBERO-PoseShift evaluations of the paper (LIBERO-Spatial and LIBERO-10, 10 tasks x 10
# conditions x 10 episodes = 2,000 episodes per model, base-frame protocol).
#
#   bash scripts/reproduce_poseshift.sh sensitivity   # public pi05 checkpoint: none / world frame / WACO runtime / GT oracle
#   bash scripts/reproduce_poseshift.sh main          # main comparison (needs B1, B1 + 2D aug and W1 checkpoints)
#   bash scripts/reproduce_poseshift.sh ablation      # component ablations on W1
#   bash scripts/reproduce_poseshift.sh fewshot       # few-shot scaling (10 / 20 demonstrations per task)
#   bash scripts/reproduce_poseshift.sh calib_error   # sensitivity to calibration error (base L3, LIBERO-Spatial)
#
# Checkpoints are read from outputs/train/<run>/checkpoints/last/pretrained_model
# (scripts/reproduce_training.sh). Results: outputs/eval/<tag>/summary.md.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source scripts/presets.sh
export GPUS=${GPUS:-0} PER_GPU=${PER_GPU:-1}
ck() { echo "$ROOT/outputs/train/$1/checkpoints/last/pretrained_model"; }
PUBLIC=${PUBLIC:-lerobot/pi05_libero_finetuned}
run() {  # run TAG POLICY [VAR=VALUE ...]
  local tag=$1 pol=$2; shift 2
  env POLICY="$pol" TAG="$tag" "$@" bash scripts/eval_grid.sh
}

case ${1:?usage: reproduce_poseshift.sh sensitivity|main|ablation|fewshot|calib_error} in
sensitivity)
  waco_preset basic
  run pub_none   "$PUBLIC"
  run pub_world  "$PUBLIC" FRAME=world KINDS=base,both
  run pub_oracle "$PUBLIC" ORACLE=both KINDS=cam,base,both
  waco_preset wide
  run pub_waco   "$PUBLIC" CANON=waco ;;
main)
  waco_preset basic
  run b1             "$(ck b1)"
  run b1_aug         "$(ck b1_aug)"
  run b2             "$(ck w1)"                                         # W1's samples, no runtime
  run align_replay   "$(ck b1)"     CANON=waco POSESHIFT_CHANNELS=B     # front end + channel B on B1
  run base_gauge     "$(ck b1)"     CANON=waco POSESHIFT_ANCHOR=base_gauge
  run camera_gauge   "$(ck b1)"     CANON=waco POSESHIFT_ANCHOR=camera_gauge
  run b1_runtime     "$(ck b1)"     CANON=waco
  run b1_aug_runtime "$(ck b1_aug)" CANON=waco
  run waco           "$(ck w1)"     CANON=waco                          # WACO = W1 + runtime
  run oracle_b1      "$(ck b1)"     ORACLE=both
  run oracle_w1      "$(ck w1)"     ORACLE=both ;;
ablation)
  waco_preset basic
  W1=$(ck w1)
  run abl_channel_a      "$W1" CANON=waco POSESHIFT_CHANNELS=A
  run abl_channel_b      "$W1" CANON=waco POSESHIFT_CHANNELS=B
  run abl_no_homing      "$W1" CANON=waco EXTRA_ENV_ARGS=--env.canon_homing=false
  run abl_arm_only       "$W1" CANON=waco POSESHIFT_ANCHOR=camera_gauge  # dT_c = I
  run abl_scene_only     "$W1" CANON=waco POSESHIFT_ANCHOR=world         # dT_b = I
  run abl_no_gate        "$W1" CANON=waco POSESHIFT_NOGATE=1
  run abl_round_trip     "$(ck w1_round_trip_only)"  CANON=waco
  run abl_state_noise    "$(ck w1_state_noise_only)" CANON=waco
  run abl_ref_depth      "$W1" CANON=waco POSESHIFT_WARP_DEPTH=ref
  run abl_plane          "$W1" CANON=waco POSESHIFT_WARP_DEPTH=plane ;;
fewshot)
  waco_preset basic
  for k in 10 20; do
    run fs_b1_k$k         "$(ck b1_k$k)"
    run fs_b1_k${k}_waco  "$(ck b1_k$k)" CANON=waco
    run fs_w1_k$k         "$(ck w1_k$k)"
    run fs_w1_k${k}_waco  "$(ck w1_k$k)" CANON=waco
  done ;;
calib_error)
  # yaw error injected into the ground-truth base canonicalisation (base L3, LIBERO-Spatial, 200 episodes per point)
  waco_preset basic
  for e in 0 0.5 1 2 4; do
    for m in b1 w1; do
      run err_${m}_$e "$(ck $m)" SUITES=libero_spatial KINDS=base LEVELS=l3 N_EP=20 ORACLE=base POSESHIFT_ORACLE_YAW_ERR=$e
    done
  done ;;
esac
