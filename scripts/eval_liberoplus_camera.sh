#!/usr/bin/env bash
# LIBERO-Plus camera track: 1,599 task variants, one episode each.
#
#   POLICY=... TAG=lp_waco CANON=waco PROMPT=released|corrected SEED=0 bash scripts/eval_liberoplus_camera.sh
#
# Run it in an environment in which the LIBERO-Plus fork replaces `libero`
# (scripts/setup_liberoplus_env.sh). The camera change is part of each LIBERO-Plus task variant,
# so the LIBERO-PoseShift perturbation stays "clean"; WACO loads the calibration bundle of the
# original task, matched by instruction. With CANON=waco the `liberoplus` front-end preset is used.
#   PROMPT=released   the benchmark's prompts, which append the variant's parameters (default)
#   PROMPT=corrected  the task's own instruction
set -uo pipefail
ROOT=${POSESHIFT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export POSESHIFT_ROOT=$ROOT
POLICY=${POLICY:?}; TAG=${TAG:?}; CANON=${CANON:-none}
SEED=${SEED:-0}
SUITES=${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}
CHUNK=${CHUNK:-25}                       # task variants per lerobot-eval process
GPUS=${GPUS:-0}; PER_GPU=${PER_GPU:-2}
PY=${PYTHON:-python}
export POSESHIFT_LP_PROMPT=${PROMPT:-released}
source "$ROOT/scripts/presets.sh"
if [[ $CANON == waco ]]; then waco_preset liberoplus; fi

IDS=${IDS:-$ROOT/data/liberoplus_camera_ids.json}   # task ids of the camera track per suite
[[ -f $IDS ]] || "$PY" "$ROOT/scripts/liberoplus_camera_ids.py" "$IDS"

jobs_list=()
for suite in ${SUITES//,/ }; do
  mapfile -t chunks < <("$PY" -c "
import json; ids=json.load(open('$IDS'))['$suite']
for s in range(0, len(ids), $CHUNK): print(json.dumps(ids[s:s+$CHUNK]).replace(' ', ''))")
  k=0
  for ch in "${chunks[@]}"; do jobs_list+=("$suite|$k|$ch"); k=$((k + 1)); done
done
echo "[liberoplus] $TAG: ${#jobs_list[@]} chunks, prompt=$POSESHIFT_LP_PROMPT, seed=$SEED"

IFS=, read -ra G <<< "$GPUS"
slots=(); for g in "${G[@]}"; do for ((i = 0; i < PER_GPU; i++)); do slots+=("$g"); done; done
pids=()
for job in "${jobs_list[@]}"; do
  IFS='|' read -r suite k ch <<< "$job"
  while :; do
    for s in "${!slots[@]}"; do
      pid=${pids[$s]:-}
      if [[ -z $pid ]] || ! kill -0 "$pid" 2>/dev/null; then
        SUITES=$suite KINDS=clean SEEDS=$SEED TASK_IDS="$ch" N_EP=1 BATCH=1 GPUS=${slots[$s]} PER_GPU=1 \
          POLICY=$POLICY TAG=$TAG CANON=$CANON FRAME=base \
          OUT=$ROOT/outputs/liberoplus/$TAG/${suite}_chunk$k EXTRA_ENV_ARGS="--env.is_libero_plus=true" \
          bash "$ROOT/scripts/eval_grid.sh" > /dev/null 2>&1 &
        pids[$s]=$!
        continue 3
      fi
    done
    sleep 5
  done
done
wait
"$PY" "$ROOT/scripts/aggregate_liberoplus.py" "$ROOT/outputs/liberoplus" "$TAG"
