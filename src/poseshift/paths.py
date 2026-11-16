"""Where data and outputs live.

`POSESHIFT_ROOT` (default: the repository root) holds `data/` (calibration bundles, per-frame
depth, few-shot splits) and `outputs/` (checkpoints, evaluation results).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = os.environ.get("POSESHIFT_ROOT", str(Path(__file__).resolve().parents[2]))

__all__ = ["ROOT"]
