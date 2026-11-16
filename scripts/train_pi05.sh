#!/usr/bin/env bash
# Full fine-tuning of LeRobot pi05 with the paper's recipe (all weights, bf16, gradient
# checkpointing, global batch 32, AdamW lr 2.5e-5 with warm-up and cosine decay to 2.5e-6,
# n_action_steps = 10, vocabulary embedding frozen).
#
#   B1 (standard few-shot recipe)   RUN=b1 bash scripts/train_pi05.sh
#   W1 (error-aware fine-tuning)    RUN=w1 POSESHIFT_RT_AUG=1 bash scripts/train_pi05.sh
#
# The global batch is (number of GPUs) x PER_GPU_BS; keep it at 32. MICRO_BS splits each per-GPU
# batch into micro-batches with one optimiser update per step (scripts/apply_lerobot_microbatch.py).
# See scripts/reproduce_training.sh for every policy of the paper.
set -euo pipefail

ROOT=${POSESHIFT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export POSESHIFT_ROOT=$ROOT
RUN=${RUN:?set RUN to a run name}
GPUS=${GPUS:-0,1,2,3}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
PER_GPU_BS=${PER_GPU_BS:-8}
STEPS=${STEPS:-10000}
SAVE_FREQ=${SAVE_FREQ:-2500}
LOG_FREQ=${LOG_FREQ:-50}
LR=${LR:-2.5e-5}
WARMUP=${WARMUP:-1000}
DECAY_LR=${DECAY_LR:-2.5e-6}
NUM_WORKERS=${NUM_WORKERS:-8}
PRETRAINED=${PRETRAINED:-lerobot/pi05_base}
DATASET_REPO=${DATASET_REPO:-lerobot/libero}
DATASET_ROOT=${DATASET_ROOT:-}        # local copy of the dataset (default: the Hub cache)
OUT=$ROOT/outputs/train/$RUN
EXTRA_ARGS=${EXTRA_ARGS:-}

if [[ -e $OUT && ${RESUME:-0} != 1 ]]; then
  echo "output dir $OUT exists; set RESUME=1 or pick another RUN" >&2; exit 2
fi
mkdir -p "$ROOT/logs"

export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPUS
export WANDB_MODE=${WANDB_MODE:-disabled}
# memory: shard the AdamW state across ranks (ZeRO-1) and let DDP buckets alias the gradients
export POSESHIFT_OPTIM=${POSESHIFT_OPTIM:-zero1}
export POSESHIFT_DDP_BUCKET_VIEW=${POSESHIFT_DDP_BUCKET_VIEW:-1}
# micro-batch accumulation inside each step
export POSESHIFT_MICRO_BS=${MICRO_BS:-0}
# freeze the vocabulary embedding
export POSESHIFT_FREEZE_EMBED=${FREEZE_EMBED:-1}

LOG=$ROOT/logs/train_${RUN}.log
echo "[train] run=$RUN gpus=$GPUS nproc=$NPROC per_gpu_bs=$PER_GPU_BS micro_bs=${MICRO_BS:-0} global_bs=$((NPROC * PER_GPU_BS)) steps=$STEPS lr=$LR" | tee -a "$LOG"
echo "[train] pretrained=$PRETRAINED dataset=$DATASET_REPO" | tee -a "$LOG"

RESUME_ARGS=()
if [[ ${RESUME:-0} == 1 ]]; then
  RESUME_ARGS=(--resume=true --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json")
fi

torchrun --standalone --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29631}" \
  "$(command -v lerobot-train)" \
  --dataset.repo_id="$DATASET_REPO" \
  ${DATASET_ROOT:+--dataset.root="$DATASET_ROOT"} \
  --dataset.video_backend=pyav \
  --selected_image_features="${IMAGE_FEATURES:-[\"observation.images.image\",\"observation.images.image2\"]}" \
  --policy.type=pi05 \
  --policy.pretrained_path="$PRETRAINED" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.empty_cameras=1 \
  --policy.n_action_steps=10 \
  --policy.optimizer_lr="$LR" \
  --policy.scheduler_warmup_steps="$WARMUP" \
  --policy.scheduler_decay_steps="$STEPS" \
  --policy.scheduler_decay_lr="$DECAY_LR" \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --save_checkpoint_to_hub=false \
  --batch_size="$PER_GPU_BS" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq="$LOG_FREQ" \
  --env_eval_freq=0 \
  --seed="${SEED:-1000}" \
  --output_dir="$OUT" \
  --job_name="$RUN" \
  "${RESUME_ARGS[@]}" \
  $EXTRA_ARGS 2>&1 | tee -a "$LOG"
