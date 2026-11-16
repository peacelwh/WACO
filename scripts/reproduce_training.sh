#!/usr/bin/env bash
# Train the policies of the paper. All of them share the recipe of scripts/train_pi05.sh:
# global batch 32, 10k steps, lr 2.5e-5 (four GPUs x 8 samples with MICRO_BS=2 below; any
# split with the same global batch works).
#
#   bash scripts/reproduce_training.sh b1          # B1: standard few-shot recipe
#   bash scripts/reproduce_training.sh b1_aug      # B1 + 2D aug
#   bash scripts/reproduce_training.sh w1          # W1: error-aware fine-tuning (WACO = W1 + runtime)
#   bash scripts/reproduce_training.sh fewshot     # B1 / W1 with the first 10 and 20 demonstrations per task
#   bash scripts/reproduce_training.sh ablation    # W1 with round-trip reprojection only / state noise only
#   bash scripts/reproduce_training.sh liberoplus  # WACO fine-tuning of the public LIBERO checkpoint
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
COMMON=(GPUS=${GPUS:-0,1,2,3} PER_GPU_BS=8 MICRO_BS=2 STEPS=10000 SAVE_FREQ=2500)
train() { env "${COMMON[@]}" "$@" bash scripts/train_pi05.sh; }
episodes() { tr -d ' \n' < "data/fewshot_episodes_$1.json"; }

case ${1:?usage: reproduce_training.sh b1|b1_aug|w1|fewshot|ablation|liberoplus} in
b1)      train RUN=b1 ;;
b1_aug)  train RUN=b1_aug EXTRA_ARGS="--dataset.image_transforms.enable=true" ;;
w1)      train RUN=w1 POSESHIFT_RT_AUG=1 ;;
fewshot)
  for k in 10 20; do
    train RUN=b1_k$k EXTRA_ARGS="--dataset.episodes=$(episodes $k)"
    train RUN=w1_k$k POSESHIFT_RT_AUG=1 EXTRA_ARGS="--dataset.episodes=$(episodes $k)"
  done ;;
ablation)
  train RUN=w1_round_trip_only  POSESHIFT_RT_AUG=$ROOT/scripts/configs/aug_round_trip_only.json
  train RUN=w1_state_noise_only POSESHIFT_RT_AUG=$ROOT/scripts/configs/aug_state_noise_only.json ;;
liberoplus)
  # the public LIBERO checkpoint, fine-tuned further for 10k steps (300 warm-up steps, global batch
  # 32), with the augmentation rendered like the runtime's image channel on LIBERO-Plus
  # (inverse warp, unseen pixels from the reference); one GPU: 32 samples in micro-batches of 4
  env GPUS=${GPUS:-0} PER_GPU_BS=32 MICRO_BS=4 STEPS=10000 SAVE_FREQ=2500 WARMUP=300 \
    PRETRAINED=lerobot/pi05_libero_finetuned RUN=waco_pi05_libero \
    POSESHIFT_RT_AUG=1 POSESHIFT_RT_WARP=inverse POSESHIFT_RT_FILL=ref bash scripts/train_pi05.sh ;;
esac
