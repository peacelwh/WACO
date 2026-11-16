"""Error-aware fine-tuning: the estimator-matched augmentation of WACO (applied inside
LeRobot's dataset).

At test time WACO feeds pi05 a *canonicalised* agentview image: the frame from a shifted
camera, forward-splatted back to the nominal camera with the ESTIMATED shift. That image
has holes, out-of-view bands and a small residual misalignment. Proprioception is likewise
re-expressed with an estimated base shift and inherits a small error. A policy trained
only on pristine demos has never seen either. This module reproduces both effects on the
training demos, using nothing but the demos and the one-off calibration bundle:

  image (prob. p_rt):  I --warp(D, dT)--> I' --warp(D', (dT*delta)->nominal)--> I~
                       (warp = forward splat, or the inverse warp with reference hole fill
                        that the runtime uses for the wide views of LIBERO-Plus)
                       dT ~ virtual camera shift (<= cam_t metres, <= cam_r degrees per axis)
                       delta ~ estimator error (Gaussian, err_t metres / err_r degrees)
  state (prob. p_state): p~ = C_err p, R~ = R_Cerr R with C_err = T_wb0 delta_b T_wb0^-1,
                       delta_b a planar error of the base about its own origin.

Actions are left untouched: with an unchanged image the direction of a state error is not
observable, so rotating actions would only inject label noise.

Enabled with POSESHIFT_RT_AUG=1 (defaults) or POSESHIFT_RT_AUG=/path/to/config.json.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from .paths import ROOT
from .warp import _splat_depth, forward_splat, inverse_warp

__all__ = ["RTConfig", "RTAugmenter", "closest_rotvec", "maybe_patch_dataset"]

IMAGE_KEY = "observation.images.image"
STATE_KEY = "observation.state"


@dataclass
class RTConfig:
    p_rt: float = 0.5
    cam_t: float = 0.12        # m, virtual camera translation radius
    cam_r: float = 12.0        # deg, per-axis virtual camera rotation
    err_t: float = 0.008       # m, std of the estimator's camera-shift error per axis
    err_r: float = 0.6         # deg, std of the estimator's camera-shift error per axis
    p_state: float = 0.5
    base_err_t: float = 0.008  # m, std of the base-shift error (planar, per axis)
    base_err_r: float = 0.6    # deg, std of the base-shift yaw error
    calib_dir: str = ""
    depth_path: str = ""       # per-frame metric depth memmap (policy orientation); "" -> auto
    warp: str = "splat"        # splat | inverse  -- match the runtime's image channel
    fill: str = "gray"         # gray | ref       -- unseen pixels: grey, or the calibration reference


def closest_rotvec(r: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Among the axis-angle vectors of one rotation (r and r*(1-2pi/|r|)) pick the one nearest
    `ref`. LIBERO's end effector points down, so its angle sits near pi, where a naive
    conversion flips the sign of the whole vector and the state jumps out of distribution."""
    n = np.linalg.norm(r)
    if n < 1e-9:
        return r
    alt = r * (1.0 - 2.0 * np.pi / n)
    return r if np.linalg.norm(r - ref) <= np.linalg.norm(alt - ref) else alt


def _load_calib(calib_dir: str) -> dict[str, dict]:
    bundles = {}
    for path in sorted(glob.glob(os.path.join(calib_dir, "*.npz"))):
        z = np.load(path, allow_pickle=True)
        lang = str(z["language"]).strip().lower()
        bundles[lang] = {"depth": z["ref_depth"].astype(np.float64), "f": float(z["f_px"]),
                         "T_wb0": z["T_wb0"].astype(np.float64), "rgb": z["ref_rgb"], "mask": z["ref_mask"].astype(bool),
                         "T_wc0": z["T_wb0"].astype(np.float64) @ z["T_bc0"].astype(np.float64)}
    return bundles


def _rand_pose(rng, t_max, r_max_deg):
    v = rng.normal(size=3)
    v /= np.linalg.norm(v) + 1e-12
    T = np.eye(4)
    T[:3, 3] = v * t_max * rng.uniform(0.0, 1.0)
    T[:3, :3] = R.from_euler("xyz", rng.uniform(-r_max_deg, r_max_deg, size=3), degrees=True).as_matrix()
    return T


def _gauss_pose(rng, sd_t, sd_r_deg):
    T = np.eye(4)
    T[:3, 3] = rng.normal(scale=sd_t, size=3)
    T[:3, :3] = R.from_rotvec(np.deg2rad(rng.normal(scale=sd_r_deg, size=3))).as_matrix()
    return T


class RTAugmenter:
    def __init__(self, cfg: RTConfig, task_names: dict[int, str]):
        self.cfg = cfg
        self.calib = _load_calib(cfg.calib_dir)
        self.task_names = {int(k): str(v).strip().lower() for k, v in task_names.items()}
        missing = sorted({n for n in self.task_names.values() if n not in self.calib})
        if missing:
            print(f"[poseshift.rt_aug] no calibration for {len(missing)} task(s); they are left un-augmented",
                  flush=True)
        self._rng = None
        self._depth = None
        self._depth_ok = None

    def _frame_depth(self, gidx: int, shape) -> np.ndarray | None:
        """Per-frame metric depth in RAW orientation at the image resolution, or None."""
        if self._depth_ok is None:
            path = self.cfg.depth_path
            meta = path.replace(".npy", ".json")
            self._depth_ok = bool(path) and os.path.exists(path) and os.path.exists(meta) and \
                json.load(open(meta)).get("done_upto", 0) > 0
            if self._depth_ok:
                self._depth = np.load(path, mmap_mode="r")
                self._depth_upto = json.load(open(meta)).get("done_upto", 0)
        if not self._depth_ok or gidx >= self._depth_upto:
            return None
        d = np.asarray(self._depth[gidx], dtype=np.float64)
        if not np.isfinite(d).all() or d.max() <= 0:
            return None
        ry, rx = shape[0] // d.shape[0], shape[1] // d.shape[1]
        d = np.repeat(np.repeat(d, ry, axis=0), rx, axis=1)
        return np.ascontiguousarray(d[::-1, ::-1])

    def _gen(self):
        if self._rng is None:  # one independent stream per dataloader worker
            import torch

            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 63))
        return self._rng

    # -- image ---------------------------------------------------------------------------
    def _rt_image(self, img_chw, bundle, gidx=None):
        import torch

        rng = self._gen()
        is_float = img_chw.dtype.is_floating_point
        x = img_chw.detach().cpu().numpy()
        x = np.transpose(x, (1, 2, 0))
        x = (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8) if is_float else x.astype(np.uint8)
        raw = np.ascontiguousarray(x[::-1, ::-1])              # policy view -> raw MuJoCo view
        D = self._frame_depth(gidx, raw.shape[:2]) if gidx is not None else None
        if D is None:
            D = bundle["depth"]
        if D.shape != raw.shape[:2]:
            return img_chw
        dT = _rand_pose(rng, self.cfg.cam_t, self.cfg.cam_r)        # true virtual camera (nominal frame)
        delta = _gauss_pose(rng, self.cfg.err_t, self.cfg.err_r)
        f = bundle["f"]
        if self.cfg.warp == "inverse":
            # same renderer as the runtime's inverse warp: depth splat + depth-consistent backward sampling,
            # unseen pixels optionally filled from the empty-scene reference
            I1, seen1 = inverse_warp(raw, D, f, np.eye(4), dT, f_dst=f)
            z1, known1 = _splat_depth(D, f, np.linalg.inv(dT), f)
            D1 = np.where(seen1 & known1, z1, 0.0)
            ref = (bundle["rgb"], bundle["mask"]) if self.cfg.fill == "ref" else (None, None)
            I2, _ = inverse_warp(I1, D1, f, dT @ delta, np.eye(4), f_dst=f, fill_rgb=ref[0], fill_mask=ref[1])
        else:
            I1, _, D1 = forward_splat(raw, D, f, np.eye(4), dT, return_depth=True)
            I2, _ = forward_splat(I1, D1, f, dT @ delta, np.eye(4))   # runtime believes dT*delta
        y = np.ascontiguousarray(I2[::-1, ::-1])
        y = torch.from_numpy(np.transpose(y, (2, 0, 1)).copy())
        return y.to(img_chw.dtype) / 255.0 if is_float else y.to(img_chw.dtype)

    # -- state ---------------------------------------------------------------------------
    def _noisy_state(self, s, bundle):
        import torch

        rng = self._gen()
        v = s.detach().cpu().numpy().astype(np.float64).copy()
        db = np.eye(4)
        db[:2, 3] = rng.normal(scale=self.cfg.base_err_t, size=2)
        db[:3, :3] = R.from_euler("z", rng.normal(scale=self.cfg.base_err_r), degrees=True).as_matrix()
        T0 = bundle["T_wb0"]
        C = T0 @ db @ np.linalg.inv(T0)
        v[:3] = C[:3, :3] @ v[:3] + C[:3, 3]
        v[3:6] = closest_rotvec((R.from_matrix(C[:3, :3]) * R.from_rotvec(v[3:6])).as_rotvec(), v[3:6])
        return torch.as_tensor(v, dtype=s.dtype)

    def __call__(self, item: dict) -> dict:
        tix = item.get("task_index")
        if tix is None:
            return item
        name = self.task_names.get(int(tix))
        bundle = self.calib.get(name) if name else None
        if bundle is None:
            return item
        rng = self._gen()
        if IMAGE_KEY in item and rng.uniform() < self.cfg.p_rt:
            img = item[IMAGE_KEY]
            gidx = int(item["index"]) if "index" in item else None
            if img.ndim == 3:
                item[IMAGE_KEY] = self._rt_image(img, bundle, gidx)
            elif img.ndim == 4:  # (T, C, H, W) with n_obs_steps > 1
                import torch

                item[IMAGE_KEY] = torch.stack([self._rt_image(f, bundle) for f in img])
        if STATE_KEY in item and rng.uniform() < self.cfg.p_state:
            st = item[STATE_KEY]
            item[STATE_KEY] = self._noisy_state(st, bundle) if st.ndim == 1 else st
        return item


def _config_from_env() -> RTConfig | None:
    val = os.environ.get("POSESHIFT_RT_AUG", "")
    if not val or val == "0":
        return None
    cfg = RTConfig()
    if val not in ("1", "true", "True") and os.path.exists(val):
        cfg = RTConfig(**{**asdict(cfg), **json.load(open(val))})
    root = ROOT
    if not cfg.calib_dir:
        cfg.calib_dir = os.path.join(root, "data", "calib_libero")
    if not cfg.depth_path:
        cfg.depth_path = os.path.join(root, "data", "depth_dav2_128.npy")
    cfg.warp = os.environ.get("POSESHIFT_RT_WARP", cfg.warp)
    cfg.fill = os.environ.get("POSESHIFT_RT_FILL", cfg.fill)
    return cfg


def maybe_patch_dataset() -> None:
    cfg = _config_from_env()
    if cfg is None:
        return
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if getattr(LeRobotDataset, "_poseshift_rt", False):
        return
    original = LeRobotDataset.__getitem__

    def __getitem__(self, idx):
        item = original(self, idx)
        aug = getattr(self, "_poseshift_aug", None)
        if aug is None:
            tasks = self.meta.tasks
            if hasattr(tasks, "iterrows"):
                names = {int(r["task_index"]): str(t) for t, r in tasks.iterrows()} if "task_index" in tasks.columns \
                    else {int(i): str(t) for i, t in enumerate(tasks.index)}
            else:
                names = {int(k): str(v) for k, v in dict(tasks).items()}
            aug = RTAugmenter(cfg, names)
            self._poseshift_aug = aug
            if os.environ.get("RANK", "0") == "0":
                print(f"[poseshift.rt_aug] enabled {asdict(cfg)}; calibrated tasks: {len(aug.calib)}", flush=True)
        return aug(item)

    LeRobotDataset.__getitem__ = __getitem__
    LeRobotDataset._poseshift_rt = True
