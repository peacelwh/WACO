"""WACO: world-anchored canonicalization for pose-robust few-shot VLA models, and the
LIBERO-PoseShift benchmark (camera, base and compound shifts) for LeRobot pi05."""

from .frames import FrameAdapter
from .perturb import LEVELS, PerturbSpec, apply_perturbation, sample_perturbation

__all__ = ["FrameAdapter", "LEVELS", "PerturbSpec", "apply_perturbation", "sample_perturbation"]
__version__ = "0.1.0"
