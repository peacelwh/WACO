"""Frame conventions: the base-frame protocol of LIBERO-PoseShift.

Conventions of the original LIBERO (robosuite 1.4, `OSC_POSE`):

* `obs["robot0_eef_pos"]` / `obs["robot0_eef_quat"]` are in the **world** frame.
* The 6-D `OSC_POSE` delta is applied in the **world** frame. Rotating `robot0_base` by
  30 degrees left the world-frame displacement of a `+x` action unchanged at `[1, 0, 0]`,
  while the base-frame displacement rotated to `[0.83, -0.53, 0]`.

As long as the base is fixed, world and base frames differ only by a constant and the choice
is immaterial. Once the base moves, the world-frame convention hands the policy the correct
target for free, since neither the object nor the reported state has moved in the world
(protocol leakage). A real robot has no world frame: its policy speaks the **base** frame.

Two conventions are supported, and the base-frame protocol is the default:

``world``  LIBERO's native behaviour (the world-frame column of the sensitivity table).
``base``   express observed EEF pose and commanded deltas in the robot base frame,
           **anchored at the nominal (calibration-time) base pose**.

The anchoring matters. LIBERO demonstrations record `robot0_eef_pos` in the world frame, and
the world frame coincides with "the base frame as it was at calibration". Anchoring keeps
the clean-scene state bit-identical to the training data, so no dataset conversion is
needed, and under a base shift the robot reports its pose in the frame it *believes* it is
in -- which is exactly what a real robot does after being nudged:

    p_policy = T_nominal * T_actual^{-1} * p_world
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

__all__ = ["FrameAdapter"]


class FrameAdapter:
    """Converts observations and actions between the world frame and the robot base frame."""

    def __init__(self, frame: str = "base", base_body: str = "robot0_base"):
        frame = frame.lower()
        if frame not in ("world", "base"):
            raise ValueError(f"frame must be 'world' or 'base'; got {frame!r}")
        self.frame = frame
        self.base_body = base_body
        self._t_base = np.zeros(3)
        self._R_base = R.identity()
        self._t_nom = np.zeros(3)
        self._R_nom = R.identity()

    # -- bookkeeping ------------------------------------------------------------------
    def sync(self, sim, nominal_pos=None, nominal_quat_wxyz=None) -> None:
        """Read the current base pose out of the simulator. Call after every reset.

        `nominal_*` is the calibration-time base pose (before any perturbation). If omitted
        the current pose is taken as nominal, i.e. the scene is assumed unperturbed.
        """
        bid = sim.model.body_name2id(self.base_body)
        self._t_base = sim.data.body_xpos[bid].copy()
        self._R_base = R.from_matrix(sim.data.body_xmat[bid].reshape(3, 3).copy())
        if nominal_pos is None:
            self._t_nom, self._R_nom = self._t_base.copy(), self._R_base
        else:
            q = np.asarray(nominal_quat_wxyz, dtype=float)
            self._t_nom = np.asarray(nominal_pos, dtype=float).copy()
            self._R_nom = R.from_quat([q[1], q[2], q[3], q[0]])

    @property
    def _R_rel(self) -> R:
        """Rotation taking actual-base-frame vectors to the nominal (policy) frame."""
        return self._R_nom * self._R_base.inv()

    @property
    def base_pos(self) -> np.ndarray:
        return self._t_base.copy()

    @property
    def base_rot(self) -> R:
        return self._R_base

    # -- controlled calibration error (oracle experiments) ---------------------------
    def set_world_error(self, yaw_deg: float = 0.0) -> None:
        """In the world (= perfectly canonicalised) frame, make the policy's frame wrong by a yaw
        about the robot base, as a base-shift estimate with that error would (sensitivity to
        calibration error)."""
        self._R_err = R.from_euler("z", yaw_deg, degrees=True) if yaw_deg else None

    def _err(self):
        return getattr(self, "_R_err", None) if self.frame == "world" else None

    # -- observations -----------------------------------------------------------------
    def obs_pos(self, p_world) -> np.ndarray:
        p_world = np.asarray(p_world, dtype=float)
        if self.frame == "world":
            e = self._err()
            return p_world if e is None else self._t_base + e.apply(p_world - self._t_base)
        return self._t_nom + self._R_rel.apply(p_world - self._t_base)

    def obs_rotmat(self, mat_world) -> np.ndarray:
        mat_world = np.asarray(mat_world, dtype=float)
        if self.frame == "world":
            e = self._err()
            return mat_world if e is None else e.as_matrix() @ mat_world
        return self._R_rel.as_matrix() @ mat_world

    def obs_quat_xyzw(self, quat_xyzw) -> np.ndarray:
        q = np.asarray(quat_xyzw, dtype=float)
        if self.frame == "world":
            e = self._err()
            if e is None:
                return q
            out = (e * R.from_quat(q)).as_quat()
            return out if np.dot(out, q) >= 0 else -out
        out = (self._R_rel * R.from_quat(q)).as_quat()
        # keep the input's hemisphere so the downstream quat->axis-angle stays continuous
        return out if np.dot(out, q) >= 0 else -out

    # -- actions ----------------------------------------------------------------------
    def action_to_world(self, action) -> np.ndarray:
        """Map a 7-D policy action (6-D EEF delta + gripper) into LIBERO's world frame."""
        a = np.asarray(action, dtype=float).copy()
        if self.frame == "world":
            e = self._err()
            if e is not None:
                a[:3] = e.inv().apply(a[:3])
                a[3:6] = e.inv().apply(a[3:6])
            return a
        a[:3] = self._R_rel.inv().apply(a[:3])
        a[3:6] = self._R_rel.inv().apply(a[3:6])
        return a

    def action_to_policy(self, action_world) -> np.ndarray:
        """Inverse of :meth:`action_to_world`."""
        a = np.asarray(action_world, dtype=float).copy()
        if self.frame == "world":
            return a
        a[:3] = self._R_rel.apply(a[:3])
        a[3:6] = self._R_rel.apply(a[3:6])
        return a
