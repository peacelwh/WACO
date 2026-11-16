"""Deterministic camera and robot-base pose shifts: the perturbation layer of LIBERO-PoseShift.

Design notes
------------
* Perturbations are drawn from a seed derived from (task name, task id, episode index, kind,
  level, seed) so that every model sees exactly the same perturbed scenes. Nothing depends
  on wall-clock time or global RNG state.
* The camera perturbation rotates about the camera's own optical centre and translates in
  the WORLD frame, which is what a bumped tripod does (`mode="bump"`, used by all levels).
  `mode="orbit"` additionally scales the camera's distance to the scene.
* The base perturbation moves `robot0_base` in the floor plane and rotates it about the
  vertical. The table, the objects and the third-person camera stay where they are, so only
  the robot moves -- exactly the "the robot got re-docked" case.
* `kind="both"` applies the two independently.

Magnitudes (LIBERO-Spatial task 0, 256x256 third-person view): a 2/5/10 cm camera translation
moves a target object by 6/15/31 px and a 3/5/10 degree camera yaw moves it by 12/21/42 px.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

from .geometry import mj2scipy, scipy2mj

__all__ = ["PerturbSpec", "LEVELS", "resolve_level", "sample_perturbation", "apply_perturbation"]


@dataclass
class PerturbSpec:
    """A concrete, already-sampled perturbation. Units: metres and degrees."""

    cam_dpos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cam_drot_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cam_dist_scale: float = 1.0
    base_dpos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_dyaw: float = 0.0

    def is_identity(self) -> bool:
        return (
            not np.any(self.cam_dpos)
            and not np.any(self.cam_drot_rpy)
            and self.cam_dist_scale == 1.0
            and not np.any(self.base_dpos)
            and self.base_dyaw == 0.0
        )

    def to_dict(self) -> dict:
        return {
            "cam_dpos": np.asarray(self.cam_dpos).tolist(),
            "cam_drot_rpy": np.asarray(self.cam_drot_rpy).tolist(),
            "cam_dist_scale": float(self.cam_dist_scale),
            "base_dpos": np.asarray(self.base_dpos).tolist(),
            "base_dyaw": float(self.base_dyaw),
        }


# Perturbation levels of LIBERO-PoseShift:
# level -> (max camera translation m, max camera rotation deg per axis, max base translation m,
#           max base yaw deg). Camera and base shifts are enabled independently by `kind`.
LEVELS: dict[str, tuple[float, float, float, float]] = {
    "clean": (0.00, 0.0, 0.00, 0.0),
    "l1": (0.02, 3.0, 0.02, 3.0),
    "l2": (0.05, 5.0, 0.05, 8.0),
    "l3": (0.10, 10.0, 0.10, 15.0),
}

KINDS = ("clean", "cam", "base", "both")


def _rng(*parts) -> np.random.Generator:
    key = "|".join(str(p) for p in parts).encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    return np.random.default_rng(seed)


def resolve_level(level: str) -> tuple[float, float, float, float]:
    level = level.lower()
    if level not in LEVELS:
        raise ValueError(f"unknown perturbation level {level!r}; expected one of {sorted(LEVELS)}")
    return LEVELS[level]


def sample_perturbation(
    kind: str,
    level: str,
    task_name: str,
    task_id: int,
    episode_index: int,
    seed: int = 0,
    mode: str = "bump",
) -> PerturbSpec:
    """Draw a reproducible perturbation.

    kind: 'clean' | 'cam' | 'base' | 'both'
    level: 'clean' | 'l1' | 'l2' | 'l3'
    mode:  'bump'  -- camera translates in world and rotates about its own centre
           'orbit' -- additionally scales the camera's distance to the scene
    """
    kind = kind.lower()
    if kind not in KINDS:
        raise ValueError(f"unknown perturbation kind {kind!r}; expected one of {KINDS}")
    if kind == "clean" or level == "clean":
        return PerturbSpec()

    cam_t, cam_r, base_t, base_yaw = resolve_level(level)
    g = _rng(task_name, task_id, episode_index, kind, level, seed, mode)
    spec = PerturbSpec()

    if kind in ("cam", "both"):
        # Translation: uniform direction, between half and the full radius of the level's ball;
        # rotation: uniform per axis.
        v = g.normal(size=3)
        v /= np.linalg.norm(v) + 1e-12
        spec.cam_dpos = v * cam_t * g.uniform(0.5, 1.0)
        spec.cam_drot_rpy = g.uniform(-1.0, 1.0, size=3) * cam_r
        if mode == "orbit":
            spec.cam_dist_scale = float(g.uniform(1.0, 1.0 + 0.5 * cam_t / 0.10))

    if kind in ("base", "both"):
        # Translation in the floor plane: uniform direction, between half and the full radius of
        # the level's disc; yaw uniform. A vertical offset an order of magnitude smaller models
        # the tolerance of the mount.
        ang = g.uniform(0, 2 * np.pi)
        r_mag = base_t * g.uniform(0.5, 1.0)
        spec.base_dpos = np.array(
            [r_mag * np.cos(ang), r_mag * np.sin(ang), g.uniform(-0.1, 0.1) * base_t]
        )
        spec.base_dyaw = float(g.uniform(-1.0, 1.0) * base_yaw)

    return spec


def apply_perturbation(sim, spec: PerturbSpec, camera_name: str = "agentview",
                       base_body: str = "robot0_base") -> dict:
    """Mutate the MuJoCo model in place. Call BEFORE the episode's init state is set.

    Returns the nominal (pre-perturbation) camera and base poses so a caller can restore
    them or log the ground-truth transform.
    """
    m = sim.model
    cid = m.camera_name2id(camera_name)
    bid = m.body_name2id(base_body)
    nominal = {
        "cam_pos": m.cam_pos[cid].copy(),
        "cam_quat": m.cam_quat[cid].copy(),
        "base_pos": m.body_pos[bid].copy(),
        "base_quat": m.body_quat[bid].copy(),
    }
    if spec.is_identity():
        return nominal

    # --- camera -------------------------------------------------------------------
    if np.any(spec.cam_dpos) or np.any(spec.cam_drot_rpy) or spec.cam_dist_scale != 1.0:
        pos = nominal["cam_pos"].astype(float).copy()
        rot = mj2scipy(nominal["cam_quat"])
        if spec.cam_dist_scale != 1.0:
            # Scale the distance from the workspace centre along the optical axis. The
            # camera looks down -z in its own frame.
            axis = rot.apply(np.array([0.0, 0.0, -1.0]))
            # Approximate the scene centre as the point 1 m in front of the nominal camera.
            centre = pos + axis * 1.0
            pos = centre - (centre - pos) * spec.cam_dist_scale
        pos = pos + np.asarray(spec.cam_dpos, dtype=float)
        # Rotate about the camera's own centre: pre-multiply in the world frame so that
        # roll/pitch/yaw read as world-axis rotations of the mount.
        drot = R.from_euler("xyz", np.asarray(spec.cam_drot_rpy, dtype=float), degrees=True)
        m.cam_pos[cid] = pos
        m.cam_quat[cid] = scipy2mj(drot * rot)

    # --- robot base ---------------------------------------------------------------
    if np.any(spec.base_dpos) or spec.base_dyaw:
        m.body_pos[bid] = nominal["base_pos"].astype(float) + np.asarray(spec.base_dpos, float)
        if spec.base_dyaw:
            rot = mj2scipy(nominal["base_quat"])
            m.body_quat[bid] = scipy2mj(R.from_euler("z", spec.base_dyaw, degrees=True) * rot)

    sim.forward()
    return nominal
