"""LIBERO-PoseShift: a LIBERO environment with graded camera, base and compound pose shifts.

Subclasses LeRobot's `LiberoEnv` so that the rest of the LeRobot evaluation stack --
observation processor, vectorisation, success bookkeeping -- keeps working unchanged.

Two behaviours are added:

1. After every reset the MuJoCo model is re-perturbed. robosuite rebuilds the model on a
   hard reset, so a one-off edit at construction time would be silently undone; we re-apply
   the perturbation on each reset, before the episode's init state is restored.
2. Observations and actions can be expressed in the robot base frame (see `frames.py`).

The ground-truth camera extrinsics and base pose are exposed through `info` for analysis
only (oracle rows, estimator accuracy). WACO never reads them.
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation as R

from .frames import FrameAdapter
from .geometry import camera_extrinsics, invert, pose_to_matrix
from .paths import ROOT
from .perturb import PerturbSpec, apply_perturbation, sample_perturbation
from .waco_runtime import WacoRuntime
from .warp import focal_from_fovy, forward_splat, real_depth

# The released LIBERO-Plus evaluation builds each policy prompt from the variant's file name, which
# appends the variant's parameters to the instruction ("... on the plate view 0 0 100 2 352
# initstate 0"). POSESHIFT_LP_PROMPT=released (default) keeps this released protocol;
# =corrected passes the task's own instruction (the suffix is stripped by _LP_SUFFIX).
LP_PROMPT = os.environ.get("POSESHIFT_LP_PROMPT", "released")
# sensitivity to calibration error: yaw error (deg, random sign per episode) injected into the
# ground-truth base canonicalisation, i.e. the policy's frame is off by that much
ORACLE_YAW_ERR = float(os.environ.get("POSESHIFT_ORACLE_YAW_ERR", "0"))
_LP_SUFFIX = re.compile(r"\s+view(\s+-?\d+)+\s+initstate\s+\d+\s*$")

ORACLE_MODES = ("none", "cam", "base", "both")
CANON_MODES = ("none", "waco")
_CALIBRATION_CACHE: dict = {}   # task bddl path -> Calibration (one-off setup per task)
_CALIB_INDEX: dict = {}


def _calib_language_index(calib_dir: str) -> dict:
    """instruction (lower-case) -> calibration bundle path, for the original LIBERO tasks."""
    if calib_dir not in _CALIB_INDEX:
        idx = {}
        for fn in sorted(os.listdir(calib_dir)) if os.path.isdir(calib_dir) else []:
            if fn.endswith(".npz"):
                z = np.load(os.path.join(calib_dir, fn), allow_pickle=True)
                idx[str(z["language"]).strip().lower()] = os.path.join(calib_dir, fn)
        _CALIB_INDEX[calib_dir] = idx
    return _CALIB_INDEX[calib_dir]

__all__ = ["LiberoPoseShiftEnv"]


class LiberoPoseShiftEnv(LiberoEnv):
    def __init__(
        self,
        *args,
        perturb_kind: str = "clean",
        perturb_level: str = "clean",
        perturb_seed: int = 0,
        perturb_mode: str = "bump",
        action_frame: str = "base",
        expose_gt_pose: bool = True,
        oracle_canon: str = "none",
        oracle_homing: bool = True,
        need_depth: bool = False,
        canon: str = "none",
        canon_homing: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._suite_name = kwargs.get("task_suite_name") or (args[2] if len(args) > 2 else "")
        if self.is_libero_plus and LP_PROMPT == "corrected":
            self.task_description = _LP_SUFFIX.sub("", self.task_description).strip()
        if oracle_canon not in ORACLE_MODES:
            raise ValueError(f"oracle_canon must be one of {ORACLE_MODES}; got {oracle_canon!r}")
        # GT oracle (upper bound): canonicalise with the ground-truth shifts.
        #   cam  -> re-render the agentview into the nominal camera with GT depth + GT pose
        #   base -> report state / accept actions in the nominal base frame, and home the
        #           end effector back to its nominal world pose after reset
        self.oracle_canon = oracle_canon
        self.oracle_homing = oracle_homing
        self._oracle_cam = oracle_canon in ("cam", "both")
        self._oracle_base = oracle_canon in ("base", "both")
        if self._oracle_base:
            action_frame = "world"  # nominal base frame == LIBERO world frame (R0 = I)
        self.homing_steps = 0
        if canon not in CANON_MODES:
            raise ValueError(f"canon must be one of {CANON_MODES}; got {canon!r}")
        if canon != "none" and oracle_canon != "none":
            raise ValueError("canon (estimated) and oracle_canon (ground truth) are mutually exclusive")
        # WACO: canonicalisation from shifts estimated at test time (no ground truth)
        self.canon = canon
        self.canon_homing = canon_homing
        self._waco: WacoRuntime | None = None
        self._need_depth = bool(need_depth) or self._oracle_cam or canon == "waco"
        self.perturb_kind = perturb_kind
        self.perturb_level = perturb_level
        self.perturb_seed = perturb_seed
        self.perturb_mode = perturb_mode
        self.expose_gt_pose = expose_gt_pose
        self._frames = FrameAdapter(action_frame)
        self._spec: PerturbSpec | None = None
        self._nominal: dict | None = None
        # The camera the agentview observation comes from; wrist cameras ride on the arm
        # and so are perturbed implicitly by a base shift.
        self._agent_camera = "agentview"

    # -- perturbation ------------------------------------------------------------------
    def _current_spec(self) -> PerturbSpec:
        return sample_perturbation(
            kind=self.perturb_kind,
            level=self.perturb_level,
            task_name=str(getattr(self, "task", "")),
            task_id=int(self.task_id),
            episode_index=int(self.init_state_id),
            seed=int(self.perturb_seed),
            mode=self.perturb_mode,
        )

    def _apply(self) -> None:
        assert self._env is not None
        self._spec = self._current_spec()
        self._nominal = apply_perturbation(
            self._env.sim, self._spec, camera_name=self._agent_camera
        )
        self._sync_frames()

    def _sync_frames(self) -> None:
        assert self._nominal is not None
        self._frames.sync(
            self._env.sim,
            nominal_pos=self._nominal["base_pos"],
            nominal_quat_wxyz=self._nominal["base_quat"],
        )

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        self._env = OffScreenRenderEnv(
            bddl_file_name=self._task_bddl_file,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
            camera_depths=self._need_depth,
        )
        self._env.reset()

    # -- privileged oracle -------------------------------------------------------------
    def _nominal_cam_pose(self) -> np.ndarray:
        m = self._env.sim.model
        cid = m.camera_name2id(self._agent_camera)
        if m.cam_bodyid[cid] != 0:
            raise RuntimeError("oracle warp assumes the agentview camera is attached to the world body")
        return pose_to_matrix(self._nominal["cam_pos"], self._nominal["cam_quat"])

    def _oracle_warp(self, raw_obs: dict) -> dict:
        key = f"{self._agent_camera}_image"
        dkey = f"{self._agent_camera}_depth"
        if dkey not in raw_obs or self._spec is None or not (
            np.any(self._spec.cam_dpos) or np.any(self._spec.cam_drot_rpy) or self._spec.cam_dist_scale != 1.0
        ):
            return raw_obs
        sim = self._env.sim
        depth = real_depth(sim, raw_obs[dkey])
        H = depth.shape[0]
        f = focal_from_fovy(sim.model.cam_fovy[sim.model.camera_name2id(self._agent_camera)], H)
        T_src = invert(camera_extrinsics(sim, self._agent_camera))
        warped, valid = forward_splat(raw_obs[key], depth, f, T_src, self._nominal_cam_pose())
        raw_obs = dict(raw_obs)
        raw_obs[key] = warped
        self.last_valid_frac = float(valid.mean())
        return raw_obs

    def _home(self, raw_obs: dict) -> dict:
        """Drive the EEF back to the world pose it had in the unperturbed scene."""
        self.homing_steps = 0
        if self._spec is None or (not np.any(self._spec.base_dpos) and self._spec.base_dyaw == 0.0):
            return raw_obs
        sim = self._env.sim
        ctrl = self._env.robots[0].controller
        # base frame anchored at nominal: exactly where the EEF sits in the nominal scene
        anchor = FrameAdapter("base")
        anchor.sync(sim, nominal_pos=self._nominal["base_pos"], nominal_quat_wxyz=self._nominal["base_quat"])
        p_tgt = anchor.obs_pos(ctrl.ee_pos)
        R_tgt = R.from_matrix(anchor.obs_rotmat(ctrl.ee_ori_mat))
        for _ in range(25):
            p = ctrl.ee_pos
            Rc = R.from_matrix(ctrl.ee_ori_mat)
            dp = p_tgt - p
            dw = (R_tgt * Rc.inv()).as_rotvec()
            if np.linalg.norm(dp) < 0.005 and np.linalg.norm(dw) < np.deg2rad(1.0):
                break
            a = np.zeros(7)
            a[:3] = np.clip(dp / 0.05, -1, 1)
            a[3:6] = np.clip(dw / 0.5, -1, 1)
            a[6] = -1.0
            raw_obs, _, _, _ = self._env.step(a)
            self.homing_steps += 1
        return raw_obs

    # -- WACO (estimated, non-privileged) ------------------------------------------------
    CALIB_STATES = tuple(range(40, 50))   # disjoint from the evaluated episodes (0..n-1)
    # POSESHIFT_CALIB selects the calibration bundles: calib_libero (LIBERO scenes, default) or
    # calib_liberoplus (recorded in the LIBERO-Plus scenes, with the multi-view scan)
    CALIB_DIR = os.path.join(ROOT, "data",
                             os.environ.get("POSESHIFT_CALIB", "calib_libero"))

    def _hide_objects(self) -> None:
        """Move every free-jointed (movable) object far below the floor: an empty workspace."""
        env, sim = self._env.env, self._env.sim
        for i, obj in enumerate(getattr(env, "objects_dict", {}).values()):
            for j in getattr(obj, "joints", []) or []:
                try:
                    sim.data.set_joint_qpos(j, np.array([0.0, 0.0, -10.0 - i, 1.0, 0.0, 0.0, 0.0]))
                    sim.data.set_joint_qvel(j, np.zeros(6))
                except Exception:  # noqa: BLE001  (non-free joints are left alone)
                    pass
        sim.forward()

    def empty_scene_frames(self) -> list[dict]:
        frames = []
        states = self._init_states
        ids = [i for i in self.CALIB_STATES if states is not None and i < len(states)] or [None]
        for i in ids:
            if i is not None:
                self._env.set_init_state(states[i])
            self._hide_objects()
            frames.append(dict(self._env.env._get_observations(force_update=True)))
        return frames

    def _calib_path(self) -> str:
        if not self.is_libero_plus:
            return os.path.join(self.CALIB_DIR, f"{self._suite_name}__{self.task_id}.npz")
        # LIBERO-Plus variants bake their perturbation (camera pose / FOV, ...) into the task
        # itself, so the calibration must come from the ORIGINAL task, matched by instruction.
        # LIBERO-Plus appends the variant parameters to the instruction, e.g.
        # "... on the plate view 0 0 100 2 352 initstate 0", so match the longest original
        # instruction that is a prefix of this one.
        index = _calib_language_index(self.CALIB_DIR)
        key = str(self.task_description).strip().lower()
        hits = [lang for lang in index if key == lang or key.startswith(lang + " ")]
        if not hits:
            raise FileNotFoundError(f"no calibration bundle for LIBERO-Plus task {key!r}; "
                                    f"run scripts/build_calibration.py first")
        return index[max(hits, key=len)]

    def _ensure_calibration(self) -> None:
        """One-off setup: empty-workspace RGB-D from the nominal camera (cached on disk)."""
        key = self._task_bddl_file
        if key not in _CALIBRATION_CACHE:
            path = self._calib_path()
            if os.path.exists(path):
                _CALIBRATION_CACHE[key] = WacoRuntime.load(path, self._env.sim)
            elif self.is_libero_plus:
                raise FileNotFoundError(path)
            else:
                _CALIBRATION_CACHE[key] = WacoRuntime.calibrate(self._env.sim, self.empty_scene_frames(),
                                                                self._agent_camera)
                self._env.reset()   # back to a pristine nominal scene before perturbing
        self._waco = None
        self._calibration = _CALIBRATION_CACHE[key]

    def _waco_start(self, raw_obs: dict) -> dict:
        rt = WacoRuntime(self._calibration, camera=self._agent_camera, home=self.canon_homing)
        rt.estimate(self._env.sim, raw_obs)
        self._waco = rt
        self.homing_steps = 0
        if rt.use_base and self.canon_homing:
            sim = self._env.sim
            ctrl = self._env.robots[0].controller
            p0 = self._frames.obs_pos(ctrl.ee_pos)                  # believed pose at t0
            R0 = R.from_matrix(self._frames.obs_rotmat(ctrl.ee_ori_mat))
            Cr = R.from_matrix(rt.C[:3, :3])
            # homing target: the demonstrations' start pose expressed in the current base frame
            p_tgt = Cr.inv().apply(p0 - rt.C[:3, 3])
            R_tgt = Cr.inv() * R0
            for _ in range(25):
                p = self._frames.obs_pos(ctrl.ee_pos)
                Rc = R.from_matrix(self._frames.obs_rotmat(ctrl.ee_ori_mat))
                dp, dw = p_tgt - p, (R_tgt * Rc.inv()).as_rotvec()
                if np.linalg.norm(dp) < 0.005 and np.linalg.norm(dw) < np.deg2rad(1.0):
                    break
                a = np.zeros(7)
                a[:3] = np.clip(dp / 0.05, -1, 1)
                a[3:6] = np.clip(dw / 0.5, -1, 1)
                a[6] = -1.0
                raw_obs, _, _, _ = self._env.step(self._frames.action_to_world(a))
                self.homing_steps += 1
            _ = sim
        return raw_obs

    # -- gym API -----------------------------------------------------------------------
    def reset(self, seed=None, **kwargs):
        self._ensure_env()
        # Perturb the model first: robosuite's reset can rebuild it, and the episode's
        # init state must be restored into the already-perturbed scene.
        self._env.seed(seed)
        self._env.reset()
        if self.canon == "waco":
            self._ensure_calibration()
        self._apply()

        if self.init_states and self._init_states is not None:
            raw_obs = self._env.set_init_state(
                self._init_states[self.init_state_id % len(self._init_states)]
            )
            self.init_state_id += self._reset_stride
        else:
            raw_obs = self._env.env._get_observations()

        # Re-read the base pose after set_init_state, then settle the scene.
        self._sync_frames()
        if self._oracle_base and ORACLE_YAW_ERR:
            sign = 1.0 if np.random.default_rng(seed).uniform() < 0.5 else -1.0
            self._frames.set_world_error(sign * ORACLE_YAW_ERR)
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())
        if self._oracle_base and self.oracle_homing:
            raw_obs = self._home(raw_obs)
        if self.canon == "waco":
            raw_obs = self._waco_start(raw_obs)

        use_delta = self.control_mode == "relative"
        if self.control_mode not in ("relative", "absolute"):
            raise ValueError(f"Invalid control mode: {self.control_mode}")
        for robot in self._env.robots:
            robot.controller.use_delta = use_delta

        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        info.update(self._pose_info())
        return observation, info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=float)
        if action.ndim != 1:
            raise ValueError(f"Expected a 1-D action, got shape {action.shape}")
        if self._waco is not None:
            action = self._waco.decanon_action(action)
        world_action = self._frames.action_to_world(action)
        obs, reward, terminated, truncated, info = super().step(world_action)
        info.update(self._pose_info())
        return obs, reward, terminated, truncated, info

    # -- observation plumbing ------------------------------------------------------------
    def _format_raw_obs(self, raw_obs):
        if self._oracle_cam:
            raw_obs = self._oracle_warp(raw_obs)
        if self._waco is not None:
            raw_obs = self._waco.canon_image(self._env.sim, raw_obs)
        obs = super()._format_raw_obs(raw_obs)
        transform = self._frames.frame == "base" or self._frames._err() is not None
        if transform and isinstance(obs, dict) and "robot_state" in obs:
            eef = obs["robot_state"]["eef"]
            if eef.get("pos") is not None:
                eef["pos"] = self._frames.obs_pos(eef["pos"])
            if eef.get("mat") is not None:
                eef["mat"] = self._frames.obs_rotmat(eef["mat"])
            if eef.get("quat") is not None:
                eef["quat"] = self._frames.obs_quat_xyzw(eef["quat"])
        if self._waco is not None and isinstance(obs, dict) and "robot_state" in obs:
            eef = obs["robot_state"]["eef"]
            if eef.get("pos") is not None:
                eef["pos"] = self._waco.canon_pos(eef["pos"])
            if eef.get("mat") is not None:
                eef["mat"] = self._waco.canon_rotmat(eef["mat"])
            if eef.get("quat") is not None:
                eef["quat"] = self._waco.canon_quat_xyzw(eef["quat"])
        return obs

    def _pose_info(self) -> dict:
        if not self.expose_gt_pose or self._env is None:
            return {}
        sim = self._env.sim
        return {
            "gt_cam_extrinsics": camera_extrinsics(sim, self._agent_camera),
            "gt_base_pos": self._frames.base_pos,
            "gt_base_quat_xyzw": self._frames.base_rot.as_quat(),
            "nominal_base_pos": np.asarray(self._nominal["base_pos"]) if self._nominal else None,
            "perturb": self._spec.to_dict() if self._spec else None,
            "homing_steps": self.homing_steps,
            "waco": dict(self._waco.info) if self._waco is not None else None,
        }
