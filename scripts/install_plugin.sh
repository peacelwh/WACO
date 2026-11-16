#!/usr/bin/env bash
# Make the `poseshift` package importable in the active Python environment and register the
# LIBERO-PoseShift environment (`--env.type=libero_poseshift`) and the WACO training options
# with LeRobot, without editing LeRobot.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
SITE=$($PY -c "import site; print(site.getsitepackages()[0])")
echo "$ROOT/src" > "$SITE/poseshift_path.pth"
echo "import poseshift.plugin" > "$SITE/poseshift_plugin.pth"
$PY -c "import poseshift, poseshift.plugin; from lerobot.envs.configs import EnvConfig; print('libero_poseshift registered:', 'libero_poseshift' in EnvConfig.get_known_choices())"
