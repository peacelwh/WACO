#!/usr/bin/env bash
# Evaluate one policy on the LIBERO-PoseShift grid:
#   suites x {clean, cam, base, both} x {l1, l2, l3}
# Each cell is one lerobot-eval run (10 tasks x N_EP episodes per suite). Finished cells are
# skipped, so the script is resumable; cells are spread over the listed GPUs.
#
# Usage:
#   POLICY=/path/to/pretrained_model TAG=b1 bash scripts/eval_grid.sh                 # no runtime
#   POLICY=/path/to/pretrained_model TAG=waco CANON=waco bash scripts/eval_grid.sh    # WACO runtime
#   POLICY=/path/to/pretrained_model TAG=oracle ORACLE=both bash scripts/eval_grid.sh # GT oracle
set -uo pipefail

ROOT=${POSESHIFT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export POSESHIFT_ROOT=$ROOT
POLICY=${POLICY:?set POLICY to a pretrained_model directory or a Hub id}
TAG=${TAG:?set TAG to a short run name}
SUITES=${SUITES:-libero_spatial,libero_10}
KINDS=${KINDS:-clean,cam,base,both}
LEVELS=${LEVELS:-l1,l2,l3}
SEEDS=${SEEDS:-0}               # perturbation / evaluation seed
N_EP=${N_EP:-10}                # episodes per task and condition
BATCH=${BATCH:-10}
FRAME=${FRAME:-base}            # base: base-frame protocol | world: LIBERO's world-frame convention
ORACLE=${ORACLE:-none}          # none | cam | base | both: canonicalise with the true shifts (upper bound)
CANON=${CANON:-none}            # none | waco: canonicalise with the shifts estimated at test time
TASK_IDS=${TASK_IDS:-}          # e.g. "[0,1,2]" to evaluate a subset of each suite
N_ACTION_STEPS=${N_ACTION_STEPS:-10}
GPUS=${GPUS:-0}
PER_GPU=${PER_GPU:-1}           # concurrent cells per GPU (pi05 inference + 10 envs need ~12 GB)
OUT=${OUT:-$ROOT/outputs/eval/$TAG}
PY=${PYTHON:-python}
mkdir -p "$OUT"

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
# one BLAS / OpenMP thread per process: every cell spawns BATCH environment workers
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cells=()
IFS=, read -ra S <<< "$SUITES"; IFS=, read -ra K <<< "$KINDS"; IFS=, read -ra L <<< "$LEVELS"; IFS=, read -ra SD <<< "$SEEDS"
for suite in "${S[@]}"; do
  for kind in "${K[@]}"; do
    if [[ $kind == clean ]]; then lv=(clean); else lv=("${L[@]}"); fi
    for level in "${lv[@]}"; do
      for seed in "${SD[@]}"; do cells+=("$suite:$kind:$level:$seed"); done
    done
  done
done
echo "[eval_grid] $TAG: ${#cells[@]} cells -> $OUT"

run_cell() {
  local cell=$1 gpu=$2 suite kind level seed
  IFS=: read -r suite kind level seed <<< "$cell"
  local name="${suite}__${kind}__${level}__s${seed}"
  local dir="$OUT/$name"
  if [[ -f "$dir/eval_info.json" ]]; then echo "[skip] $name"; return 0; fi
  mkdir -p "$dir"
  # Keep every GPU visible and select the policy GPU with --policy.device, and render on the
  # same GPU: restricting CUDA_VISIBLE_DEVICES can make robosuite ask for a missing EGL device.
  env -u CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE:-$gpu}" lerobot-eval \
    --policy.path="$POLICY" \
    --policy.device="cuda:$gpu" \
    --policy.compile_model=false \
    --policy.n_action_steps="$N_ACTION_STEPS" \
    --env.type=libero_poseshift \
    --env.task="$suite" \
    --env.perturb_kind="$kind" \
    --env.perturb_level="$level" \
    --env.perturb_seed="$seed" \
    --env.action_frame="$FRAME" \
    --env.oracle_canon="$ORACLE" \
    --env.canon="$CANON" \
    ${TASK_IDS:+--env.task_ids="$TASK_IDS"} \
    ${EXTRA_ENV_ARGS:-} \
    --env.observation_height=256 --env.observation_width=256 \
    --env.max_parallel_tasks=1 \
    --eval.n_episodes="$N_EP" \
    --eval.batch_size="$BATCH" \
    --seed="$seed" \
    --output_dir="$dir" > "$dir/eval.log" 2>&1
  echo "[done rc=$? gpu=$gpu] $name $(grep -o '"pc_success": [0-9.]*' "$dir/eval_info.json" 2>/dev/null | tail -1)"
}

# one slot per (GPU, PER_GPU) pair; every cell takes the first free slot
IFS=, read -ra G <<< "$GPUS"
slots=(); for g in "${G[@]}"; do for ((i = 0; i < PER_GPU; i++)); do slots+=("$g"); done; done
pids=()
for cell in "${cells[@]}"; do
  while :; do
    for s in "${!slots[@]}"; do
      pid=${pids[$s]:-}
      if [[ -z $pid ]] || ! kill -0 "$pid" 2>/dev/null; then
        run_cell "$cell" "${slots[$s]}" &
        pids[$s]=$!
        continue 3
      fi
    done
    sleep 5
  done
done
wait
"$PY" "$ROOT/scripts/aggregate_results.py" "$OUT"
