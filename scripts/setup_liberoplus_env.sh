#!/usr/bin/env bash
# A separate environment for the LIBERO-Plus camera track: installing the LIBERO-Plus fork
# replaces the `libero` package, so it cannot share an environment with the original LIBERO.
#
#   SRC_ENV=/path/to/envs/waco DST_ENV=/path/to/envs/waco-liberoplus \
#   LIBERO_PLUS=/path/to/LIBERO-plus bash scripts/setup_liberoplus_env.sh
#
# SRC_ENV is the conda environment that already runs LIBERO-PoseShift; LIBERO_PLUS is a checkout
# of https://github.com/sylvestf/LIBERO-plus with its asset pack extracted (see its README).
set -euo pipefail
SRC=${SRC_ENV:?conda env that runs LIBERO-PoseShift}
DST=${DST_ENV:?path of the new env}
LP=${LIBERO_PLUS:?checkout of the LIBERO-Plus repository}
[[ -x $DST/bin/python ]] || conda create --yes --quiet -p "$DST" --clone "$SRC"
"$DST/bin/pip" uninstall -y hf-libero || true
"$DST/bin/pip" install -q wand scikit-image easydict
SITE=$("$DST/bin/python" -c "import site; print(site.getsitepackages()[0])")
# the fork is a namespace package: expose it through a .pth file
echo "$LP" > "$SITE/liberoplus_path.pth"
CFG=${LIBERO_CONFIG_PATH:-$HOME/.libero_plus}
mkdir -p "$CFG"
"$DST/bin/python" - "$LP" "$CFG" <<'PY'
import os, sys, yaml
root = os.path.join(sys.argv[1], "libero", "libero")
cfg = {k: os.path.join(root, v) for k, v in
       {"benchmark_root": "", "bddl_files": "bddl_files", "init_states": "init_files",
        "datasets": "../datasets", "assets": "assets"}.items()}
yaml.safe_dump(cfg, open(os.path.join(sys.argv[2], "config.yaml"), "w"))
print(cfg)
PY
LIBERO_CONFIG_PATH=$CFG "$DST/bin/python" -c "import libero, libero.libero.benchmark as b; print('libero from', list(libero.__path__)); print(sorted(b.get_benchmark_dict())[:6])"
echo "LIBERO-Plus environment ready: export LIBERO_CONFIG_PATH=$CFG before evaluating"
