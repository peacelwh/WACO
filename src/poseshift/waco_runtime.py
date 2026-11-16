"""WACO runtime: estimate the pose shift at the start of an episode, then canonicalise.

Channel A (image):   warp the agentview frame from the current camera back to the nominal
                     camera using the estimated camera shift dT_c and the measured depth.
Channel B (base):    re-express the robot's proprioception and commanded deltas as if the
                     base were at its calibrated pose, using the estimated base shift dT_b,
                     and home the end effector to its nominal start pose.

The policy therefore sees (approximately) the calibration viewpoint and base frame. Any pi05
checkpoint can run this runtime without retraining; the error-aware fine-tuning (rt_aug.py)
additionally trains the policy on the canonicaliser's own artefacts.

Nothing here reads ground-truth poses. Calibration constants (nominal camera-in-base pose,
nominal base pose, one reference RGB-D frame per task) are captured from an unperturbed
scene, which on a real robot corresponds to the one-off setup done before demo collection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

from .estimator import (RobotCloud, SceneScorer, _cloud, _support_planes, arm_global, arm_plane_search, arm_anchor, fuse, gray,
                        project, world_anchor, scene_global, scene_refine, scene_rematch, transform)
from .geometry import camera_extrinsics, invert
from .warp import _splat_depth, focal_from_fovy, forward_splat, inverse_warp, real_depth

__all__ = ["Calibration", "WacoRuntime"]

NOISE_T = 0.005   # m  -- noise floor: below this an estimated shift is treated as zero
NOISE_R = 0.5     # deg
OFFPLANE_T = 0.02  # m   -- gate: a base cannot leave the floor, so a larger out-of-plane residual
OFFPLANE_R = 2.0   # deg    (height, roll or pitch) means that an anchor has failed
MIN_INLIERS = 8    # world-anchor inliers below which the sparse match is not trusted
# Channel A rendering:
#   POSESHIFT_WARP=splat|inverse   forward colour splat (default), or depth splat + backward colour
#                                  sampling (for the wide view changes of LIBERO-Plus)
#   POSESHIFT_HOLE_FILL=gray|ref   unseen pixels: flat grey (default), or the empty-workspace reference
WARP_MODE = os.environ.get("POSESHIFT_WARP", "splat")
# Front end (POSESHIFT_FRONTEND):
#   basic  the two anchors: SIFT + 3D RANSAC + point-to-plane ICP (world anchor), multi-hypothesis
#          ICP (arm anchor), closed-form fusion and the out-of-plane gate
#   wide   adds the components for wide viewpoint changes: candidates from SIFT, the scan views and
#          a plane-constrained global search are refined and ranked by a dense agreement score
FRONTEND = os.environ.get("POSESHIFT_FRONTEND", "basic")
SCENE_GOOD = 0.85    # wide: a candidate this consistent with the reference ends the search
GLOBAL_MARGIN = 0.03  # wide: a global candidate must beat the local ones by this much
SCENE_ACCEPT = 0.70  # wide: below this, the camera shift is treated as unknown
# Appearance check (POSESHIFT_APPEARANCE_GATE=1): if the static scene, warped to the calibration
# view, no longer looks like its reference (median grey difference on the pixels both views see),
# canonicalising towards that reference is unsafe and the estimate is not applied.
APPEARANCE_GATE = os.environ.get("POSESHIFT_APPEARANCE_GATE", "0") == "1"
# Multi-view calibration scan (POSESHIFT_MULTIVIEW=1, bundles from build_calibration_liberoplus.py):
# extra empty-workspace views around the table, matched when the nominal view is too far away
MULTIVIEW = os.environ.get("POSESHIFT_MULTIVIEW", "0") == "1"
# Plausibility (POSESHIFT_PLAUSIBLE=1): an implied base shift beyond 25 cm or 30 deg, or an arm fit
# explaining fewer than 5 % of the model points, counts as inconsistent
PLAUSIBLE = os.environ.get("POSESHIFT_PLAUSIBLE", "0") == "1"
PLAUSIBLE_T, PLAUSIBLE_YAW = 0.25, 30.0
# Floor aliases (POSESHIFT_ALIAS=1): on a periodically textured floor, let the arm pick among the
# floor-plane translations of the scene estimate that the scene agrees with equally well
ALIAS = os.environ.get("POSESHIFT_ALIAS", "0") == "1"
ALIAS_SCENE_TOL = 0.06    # dense score may drop at most this much versus the scene's own estimate
ALIAS_MIN_FIT = 0         # render-and-compare arm score (explained - 2 x see-through, of 1500 points)
ALIAS_MARGIN = 200        # ... and it must beat the scene estimate's own arm score by this much
# Arm-only fallback (POSESHIFT_ARMGLOBAL=1): when no estimate is trusted, register the arm without
# any scene prior (base assumed unmoved) and accept the fit only if it explains enough of the arm
# and the static reference depth agrees; image channel only
ARMGLOBAL = os.environ.get("POSESHIFT_ARMGLOBAL", "0") == "1"
ARMG_MIN_OVERLAP = float(os.environ.get("POSESHIFT_ARMG_OV", "0.055"))  # share of arm model points the fit must explain
ARMG_MIN_GEO = 0.99       # the static reference depth must agree with the arm-derived pose
ARM_MIN_OVERLAP = 0.05    # an arm fit explaining fewer than 5 % of the model points is not trusted
APPEARANCE_MAX = 25.0
GEO_DISCOUNT = 0.9   # wide: geometry-only agreement counts slightly less than photometric
SUPPORT_OFF = 100    # wide: off-plane points needed before geometry alone is trusted
GLOBAL_REFINE = 4    # wide: global candidates refined after dense pre-ranking
GLOBAL_ICP_BOUNDS = ((10, 0.12), (25, 0.05), (None, 0.02))  # wide: association radius schedule (m)
HOLE_FILL = os.environ.get("POSESHIFT_HOLE_FILL", "gray")
# ablations and single-anchor baselines (defaults = full method)
CHANNELS = os.environ.get("POSESHIFT_CHANNELS", "AB")     # AB | A (image only) | B (state, action only)
ANCHOR = os.environ.get("POSESHIFT_ANCHOR", "both")       # both
#   world         world anchor only (dT_b = I)
#   camera_gauge  arm anchor only, its pose change read as a base shift (dT_c = I)
#   base_gauge    arm anchor only, its pose change read as a camera shift (dT_b = I)
NO_GATE = os.environ.get("POSESHIFT_NOGATE", "0") == "1"  # apply both channels whenever the shift is not small
WARP_DEPTH = os.environ.get("POSESHIFT_WARP_DEPTH", "live")  # live | ref (static reference depth) | plane (homography)


def _base_pose(sim, base_body="robot0_base") -> np.ndarray:
    b = sim.model.body_name2id(base_body)
    T = np.eye(4)
    T[:3, :3] = sim.data.body_xmat[b].reshape(3, 3)
    T[:3, 3] = sim.data.body_xpos[b]
    return T


def _small(T: np.ndarray) -> bool:
    ang = np.degrees(np.linalg.norm(R.from_matrix(T[:3, :3]).as_rotvec()))
    return np.linalg.norm(T[:3, 3]) < NOISE_T and ang < NOISE_R


def arm_mask(T_cb, pts_b, f, H, W, dilate=6, depth=None, occ_tol=0.03) -> np.ndarray:
    """True where the pixel is NOT covered by the (visible part of the) robot.

    With `depth`, model points hidden behind observed surfaces (e.g. the pedestal under the
    table) are ignored, so they do not punch robot-shaped holes into the table region.
    """
    P = transform(T_cb, pts_b)
    uv, ok = project(P, f, H, W)
    m = np.ones((H, W), bool)
    rc = np.round(uv).astype(int)
    inb = ok & (rc[:, 0] >= 0) & (rc[:, 0] < H) & (rc[:, 1] >= 0) & (rc[:, 1] < W)
    if depth is not None:
        zi = np.flatnonzero(inb)
        visible = depth[rc[zi, 0], rc[zi, 1]] > (-P[zi, 2]) - occ_tol
        inb[zi[~visible]] = False
    rc = rc[inb]
    for dr in range(-dilate, dilate + 1, 2):
        for dc in range(-dilate, dilate + 1, 2):
            m[np.clip(rc[:, 0] + dr, 0, H - 1), np.clip(rc[:, 1] + dc, 0, W - 1)] = False
    return m


@dataclass
class Calibration:
    """What is known after the one-off setup: all in the nominal (world == base^0) frame."""

    T_wb0: np.ndarray
    T_bc0: np.ndarray
    f_px: float
    ref_rgb: np.ndarray
    ref_depth: np.ndarray
    ref_mask: np.ndarray          # True = static scene usable for the world anchor
    cloud: RobotCloud
    scorer: object = None         # dense agreement scorer of the wide front end, built lazily
    views: list | None = None     # optional multi-view scan [(T_c0_v, rgb, depth, mask)] of the empty scene
    fused: np.ndarray | None = None  # static points of nominal view + scan (nominal frame), built lazily
    pivot_w: np.ndarray | None = None  # workspace point the nominal camera looks at (world), lazily

    def fused_points(self, n=8000, seed=0) -> np.ndarray:
        if self.fused is None:
            from .estimator import _sample_points
            views = [(np.eye(4), self.ref_rgb, self.ref_depth, self.ref_mask)] + list(self.views or [])
            per = max(n // len(views), 500)
            self.fused = np.concatenate([transform(T, _sample_points(rgb, d, m, self.f_px, per, seed + k)[0])
                                         for k, (T, rgb, d, m) in enumerate(views)])
        return self.fused


@dataclass
class WacoRuntime:
    cal: Calibration
    camera: str = "agentview"
    home: bool = True
    dT_c: np.ndarray = field(default_factory=lambda: np.eye(4))
    dT_b: np.ndarray = field(default_factory=lambda: np.eye(4))
    C: np.ndarray = field(default_factory=lambda: np.eye(4))   # nominal-frame conjugate of dT_b
    use_cam: bool = False
    use_base: bool = False
    f_cur: float = 0.0
    info: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- calibration
    @staticmethod
    def calibrate(sim, frames: list[dict], camera: str = "agentview", depth_tol: float = 0.002) -> Calibration:
        """Build the calibration from EMPTY-scene captures (movable objects removed).

        On a real robot: photograph the bare workspace from the nominal camera once. The
        reference then contains only the static scene -- table, walls, fixtures -- so any
        feature that matches it is static by construction, and objects in the live frame have
        no counterpart. `frames` may hold several captures (e.g. different fixture states);
        pixels whose depth varies by more than `depth_tol` across them are excluded.
        """
        ref = frames[0]
        depth = real_depth(sim, ref[f"{camera}_depth"])
        f = focal_from_fovy(sim.model.cam_fovy[sim.model.camera_name2id(camera)], depth.shape[0])
        T_wc0 = invert(camera_extrinsics(sim, camera))
        T_wb0 = _base_pose(sim)
        T_bc0 = invert(T_wb0) @ T_wc0
        cloud = RobotCloud(sim)
        not_arm = arm_mask(invert(T_bc0), cloud.points_base(sim), f, *depth.shape, dilate=8, depth=depth)
        if len(frames) > 1:
            dstack = np.stack([real_depth(sim, fr[f"{camera}_depth"]) for fr in frames])
            static = dstack.std(axis=0) < depth_tol
        else:
            static = np.ones(depth.shape, bool)
        static &= (depth > 1e-3) & (depth < 3.0)
        return Calibration(T_wb0, T_bc0, f, ref[f"{camera}_image"].copy(), depth, not_arm & static, cloud)

    @staticmethod
    def load(path: str, sim) -> Calibration:
        z = np.load(path, allow_pickle=True)
        views = None
        if MULTIVIEW and "views_T" in z.files:
            views = [(z["views_T"][k], z["views_rgb"][k], z["views_depth"][k].astype(np.float64),
                      z["views_mask"][k].astype(bool)) for k in range(len(z["views_T"]))]
        return Calibration(z["T_wb0"], z["T_bc0"], float(z["f_px"]), z["ref_rgb"], z["ref_depth"].astype(np.float64),
                           z["ref_mask"].astype(bool), RobotCloud(sim), views=views)

    # ---------------------------------------------------------------- estimation
    def estimate(self, sim, raw_obs) -> None:
        cal = self.cal
        depth = real_depth(sim, raw_obs[f"{self.camera}_depth"])
        rgb = raw_obs[f"{self.camera}_image"]
        # intrinsics of the LIVE camera (known from its datasheet on a real robot); they may
        # differ from the calibration camera's, e.g. LIBERO-Plus also varies the field of view
        self.f_cur = f_cur = focal_from_fovy(sim.model.cam_fovy[sim.model.camera_name2id(self.camera)],
                                             depth.shape[0])
        # 1) world anchor: static background vs the empty-workspace reference -> camera shift
        scene = world_anchor(rgb, depth, cal.ref_rgb, cal.ref_depth, f_cur, mask=None, ref_mask=cal.ref_mask,
                             f_ref=cal.f_px)
        dT_c = scene.dT_c if scene.inliers >= MIN_INLIERS else np.eye(4)
        #    point-to-plane ICP refinement on the static background
        dT_c, scene_rms = scene_refine(depth, cal.ref_depth, cal.ref_mask, f_cur, dT_c, f_ref=cal.f_px)
        scene_ok, scene_info = True, {}
        if FRONTEND == "wide":
            dT_c, scene_ok, scene_info = self._scene_dense(depth, rgb, f_cur, dT_c, scene.inliers)
        if APPEARANCE_GATE:
            diff = self._appearance_diff(rgb, depth, f_cur, dT_c)
            scene_info["appearance_diff"] = diff
            if diff is not None and diff > APPEARANCE_MAX:
                scene_ok = False
                scene_info["appearance_ok"] = False
        # 2) arm anchor: arm model at the measured joint state, initialised as "only the
        #    camera moved"; its hypothesis grid then covers the base moving
        pts_b = cal.cloud.points_base(sim)
        arm_fit, dT_b, raw_b, rpy, z_off, rp_off, consistent = self._arm_check(depth, f_cur, pts_b, dT_c)
        n_alias, arm_confirmed = 0, False
        if ALIAS and FRONTEND == "wide" and scene_ok and not consistent and not self._unambiguous(scene_info):
            # the scene alone cannot pin sliding along a bare, periodically textured floor; let the
            # arm decide among the floor-plane aliases of the scene estimate
            alt, n_alias = self._resolve_alias(depth, rgb, f_cur, pts_b, dT_c)
            if alt is not None:
                dT_c = alt
                arm_fit, dT_b, raw_b, rpy, z_off, rp_off, consistent = self._arm_check(depth, f_cur, pts_b, alt)
                if not consistent:
                    # chosen by the arm under "the base did not move": vouches for the camera
                    # channel only; the base channel stays off
                    arm_confirmed, dT_b = True, np.eye(4)
        scene_info["n_alias"] = n_alias
        scene_info["arm_confirmed"] = arm_confirmed
        self.dT_c, self.dT_b = dT_c, dT_b
        if FRONTEND == "wide":
            # the image channel needs only dT_c. When the arm anchor disagrees, which anchor is
            # wrong? If the reference has off-plane structure (walls, fixtures) that agrees, the
            # scene estimate is unambiguous and the arm anchor is blamed (arm partly out of
            # view, grazing views of the links): only the base channel is vetoed. Without such
            # structure (a bare, periodically textured floor) the scene estimate may be a
            # texture alias one period away, so both channels are vetoed as in the basic front end.
            unambiguous = self._unambiguous(scene_info) or scene_info.get("arm_confirmed", False)
            trusted = scene_ok and (consistent or unambiguous)
            self.use_cam = trusted and not _small(dT_c)
            self.use_base = scene_ok and consistent and not _small(dT_b)
            # last resort when no estimate could be trusted (large orbits over a bare table, where
            # the scene has nothing to match): the robot itself is the calibration target
            # (also when the appearance gate tripped: that gate judges the scene estimate's warp, and
            # the arm search does not use it -- its own check is on depth, which lighting cannot fool)
            if ARMGLOBAL and not trusted:
                alt, armg = self._arm_global(depth, rgb, f_cur, pts_b)
                scene_info.update(armg)
                if alt is not None:
                    dT_c, dT_b = alt, np.eye(4)
                    self.dT_c, self.dT_b = dT_c, dT_b
                    self.use_cam, self.use_base = not _small(dT_c), False
                    scene_info["scene_src"] = "arm_global"
        else:
            self.use_cam = scene_ok and consistent and not _small(dT_c)
            self.use_base = scene_ok and consistent and not _small(dT_b)
        if ANCHOR != "both" or NO_GATE or CHANNELS != "AB":
            dT_c, dT_b = self._ablate(depth, f_cur, pts_b, dT_c, dT_b, scene_ok, arm_fit)
            self.dT_c, self.dT_b = dT_c, dT_b
        self.C = cal.T_wb0 @ dT_b @ invert(cal.T_wb0)
        self.T_bc = arm_fit.T_bc
        self.info = {"icp_rms_mm": 1000 * arm_fit.residual, "arm_overlap": float(arm_fit.overlap), "scene_inliers": scene.inliers,
                     "scene_rms_mm": 1000 * scene_rms, "offplane_z_mm": 1000 * z_off, "offplane_rp_deg": rp_off,
                     "consistent": consistent, "use_cam": self.use_cam, "use_base": self.use_base,
                     "dT_c": dT_c.tolist(), "dT_b": dT_b.tolist(), "ablation": f"{ANCHOR}/{CHANNELS}/gate{int(not NO_GATE)}",
                     **scene_info}

    def _ablate(self, depth, f_cur, pts_b, dT_c, dT_b, scene_ok, arm_fit):
        """Ablation variants of the front end and the gate, and the single-anchor gauges."""
        cal = self.cal
        if ANCHOR in ("camera_gauge", "base_gauge"):
            # single anchor: the arm alone, registered from the nominal pose (no world-anchor seed).
            # Its pose change is read either as a base shift (camera gauge) or as a camera shift
            # (base gauge) -- the gauge ambiguity: each is right on one axis only
            b = arm_anchor(depth, f_cur, pts_b, cal.T_bc0, T_bc_init=cal.T_bc0)
            ok = np.isfinite(b.residual) and b.overlap >= 0.05
            if ANCHOR == "camera_gauge":
                dT_b, _, _ = fuse(b.T_bc, np.eye(4), cal.T_bc0)
                dT_c = np.eye(4)
                self.use_cam, self.use_base = False, ok and not _small(dT_b)
            else:
                dT_c, dT_b = invert(cal.T_bc0) @ b.T_bc, np.eye(4)
                self.use_cam, self.use_base = ok and not _small(dT_c), False
        elif ANCHOR == "world":
            dT_b = np.eye(4)
            self.use_cam, self.use_base = scene_ok and not _small(dT_c), False
        if NO_GATE:
            self.use_cam, self.use_base = not _small(dT_c), not _small(dT_b)
        if CHANNELS == "A":
            self.use_base = False
        elif CHANNELS == "B":
            self.use_cam = False
        return dT_c, dT_b

    def _arm_global(self, depth, rgb, f_cur, pts_b):
        """Arm-only camera estimate (base assumed unmoved), accepted only if the arm fit is clear
        and the static scene's depth agrees with it. Returns (dT_c or None, diagnostics)."""
        cal = self.cal
        up_ref = cal.T_bc0[:3, :3].T @ cal.T_wb0[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        cands = arm_plane_search(depth, f_cur, cal.ref_depth, cal.ref_mask, cal.f_px, up_ref, pts_b, cal.T_bc0)
        info = {"armg_overlap": cands[0][1] if cands else None, "armg_geo": None}
        if getattr(cal, "scorer", None) is None:
            return None, info
        g = gray(rgb)
        # the arm alone is ambiguous between a view and its mirror on the robot's other side (both
        # explain the arm similarly well): take the best arm fit whose static scene also agrees
        for dT_c, overlap, rms in cands:
            if overlap < ARMG_MIN_OVERLAP:
                break
            geo = cal.scorer(dT_c, depth, g, f_cur)["geo"]
            if geo >= ARMG_MIN_GEO:
                info.update(armg_overlap=overlap, armg_geo=geo)
                return dT_c, info
            info.setdefault("armg_rejected", []).append((round(overlap, 3), round(float(geo), 3)))
        return None, info

    @staticmethod
    def _unambiguous(scene_info) -> bool:
        """The reference has enough off-plane structure (walls, fixtures) that agrees."""
        return scene_info.get("scene_support_off", 0) >= SUPPORT_OFF and scene_info.get("scene_geo_off", 0.0) >= 0.9

    def _arm_check(self, depth, f_cur, pts_b, dT_c):
        """Arm anchor seeded by the world-anchor estimate, fusion, and the consistency gate."""
        cal = self.cal
        arm_fit = arm_anchor(depth, f_cur, pts_b, cal.T_bc0, T_bc_init=cal.T_bc0 @ dT_c)
        # fusion: dT_b = T_bc^0 dT_c T_bc^-1, projected to a planar motion
        dT_b, raw_b, _ = fuse(arm_fit.T_bc, dT_c, cal.T_bc0)
        # consistency (no ground truth needed): the base can only slide on the floor, so a large
        # out-of-plane component means one of the anchors failed -- then do nothing rather than
        # something wrong, and the policy falls back to its plain behaviour
        rpy = R.from_matrix(raw_b[:3, :3]).as_euler("xyz", degrees=True)
        z_off, rp_off = abs(raw_b[2, 3]), max(abs(rpy[0]), abs(rpy[1]))
        consistent = z_off <= OFFPLANE_T and rp_off <= OFFPLANE_R
        if PLAUSIBLE:
            # a base is nudged by centimetres, not slid across the room: an implied planar base
            # shift beyond the arm anchor's own search range means the scene estimate locked
            # onto the wrong match (e.g. one period of a repeating floor texture away)
            t_xy, yaw = float(np.linalg.norm(raw_b[:2, 3])), abs(float(rpy[2]))
            consistent = consistent and t_xy <= PLAUSIBLE_T and yaw <= PLAUSIBLE_YAW
            # a failed or barely supported arm fit (arm small / mostly out of view) returns an
            # untouched start hypothesis -- planar and "consistent", but meaningless
            arm_ok = bool(np.isfinite(arm_fit.residual)) and arm_fit.overlap >= ARM_MIN_OVERLAP
            consistent = consistent and arm_ok
        return arm_fit, dT_b, raw_b, rpy, z_off, rp_off, consistent

    def _resolve_alias(self, depth, rgb, f_cur, pts_b, dT_c, span=0.9, step=0.03, n_arm=1500):
        """Floor-plane aliases of the scene estimate (the camera displaced by v in the world
        xy-plane), ranked by how well the visible arm then matches the depth image under the
        hypothesis that the base did not move. Kept only among aliases the scene agrees with
        as much as with the estimate itself. Returns (dT_c or None, number of aliases scored)."""
        from .estimator import _depth_consistency
        cal = self.cal
        g = gray(rgb)
        T_wc0 = cal.T_wb0 @ cal.T_bc0
        base_sc = self._score(cal.scorer(dT_c, depth, g, f_cur))["score"]
        arm = pts_b[np.random.default_rng(0).choice(len(pts_b), min(n_arm, len(pts_b)), replace=False)]
        est_fit, _ = _depth_consistency(arm, invert(cal.T_bc0 @ dT_c), depth, f_cur)
        grid = np.arange(-span, span + 1e-9, step)
        best, best_fit = None, -np.inf
        n = 0
        for vx in grid:
            for vy in grid:
                Tv = np.eye(4)
                Tv[:2, 3] = (vx, vy)
                cand = invert(T_wc0) @ Tv @ T_wc0 @ dT_c
                fit, _ = _depth_consistency(arm, invert(cal.T_bc0 @ cand), depth, f_cur)
                n += 1
                if fit > best_fit:
                    sc = self._score(cal.scorer(cand, depth, g, f_cur))["score"]
                    if sc >= base_sc - ALIAS_SCENE_TOL:
                        best, best_fit = cand, fit
        # a far, small arm explains few points even at the truth, but a wrong alias makes the
        # camera see THROUGH where the arm would be -- a strongly negative score
        if best is None or best_fit < ALIAS_MIN_FIT or best_fit < est_fit + ALIAS_MARGIN:
            return None, n
        best = scene_refine(depth, cal.ref_depth, cal.ref_mask, f_cur, best, f_ref=cal.f_px,
                            model_pts=cal.fused_points() if cal.views else None)[0]
        return best, n

    def _scene_dense(self, depth, rgb, f_cur, dT_sift, sift_inliers):
        """Pick the camera shift among SIFT, identity and global candidates by dense agreement.

        The dense score separates right from wrong hypotheses but not a 2 mm estimate from a
        1 cm one, so a healthy local estimate (SIFT or identity, both ICP-refined) is kept as
        is; with a multi-view scan, SIFT against each scan view comes next; the global search
        runs only when nothing explains the reference, and must win by a clear margin.
        """
        cal = self.cal
        up_ref = cal.T_bc0[:3, :3].T @ cal.T_wb0[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        if getattr(cal, "scorer", None) is None:
            cal.scorer = SceneScorer(cal.ref_rgb, cal.ref_depth, cal.ref_mask, cal.f_px, up_ref=up_ref,
                                     views=cal.views)
        g = gray(rgb)
        model = cal.fused_points() if cal.views else None

        def refine(T0, **kw):
            return scene_refine(depth, cal.ref_depth, cal.ref_mask, f_cur, T0, f_ref=cal.f_px, model_pts=model, **kw)[0]

        def scored(src, T):
            return (self._score(cal.scorer(T, depth, g, f_cur)), src, T)

        cands = []
        if sift_inliers >= MIN_INLIERS:
            cands.append(scored("sift", refine(dT_sift) if model is not None else dT_sift))
        cands.append(scored("identity", refine(np.eye(4))))
        best = max(cands, key=lambda x: x[0]["score"])     # ties go to SIFT (listed first)
        n_global = n_rematch = n_view = 0
        if best[0]["score"] < SCENE_GOOD and cal.views:
            # the camera may have moved towards one of the scan views: match it there
            for k, (T_v, v_rgb, v_depth, v_mask) in enumerate(cal.views):
                res = world_anchor(rgb, depth, v_rgb, v_depth, f_cur, mask=None, ref_mask=v_mask, f_ref=cal.f_px)
                if res.inliers >= MIN_INLIERS:
                    cands.append(scored(f"view{k}", refine(T_v @ res.dT_c)))
                    n_view += 1
            best = max(cands, key=lambda x: x[0]["score"])
        if best[0]["score"] < SCENE_GOOD:
            extra = cal.fused_points() if cal.views else None
            coarse = scene_global(depth, cal.ref_depth, cal.ref_mask, f_cur, cal.f_px, up_ref, extra_pts=extra)
            # coarse candidates are off by up to a grid step, so pre-rank them on geometry alone
            # (texture does not line up yet) and refine only the most promising few
            pre = sorted(coarse, key=lambda T: -self._coarse(cal.scorer(T, depth, g, f_cur)))[:GLOBAL_REFINE]
            glob = []
            for T0 in pre:
                # coarse candidates can be ~5 deg / several cm off: wide association first
                T = refine(refine(T0, iters=60, bounds=GLOBAL_ICP_BOUNDS))
                glob.append(scored("global", T))
                n_global += 1
            if glob:
                gbest = max(glob, key=lambda x: x[0]["score"])
                # polish: SIFT against the reference re-rendered at the global estimate (a wide
                # baseline becomes a narrow one), then dense ICP; kept if it scores better
                T, n_rematch = scene_rematch(rgb, depth, cal.ref_rgb, cal.ref_depth, cal.ref_mask, f_cur, cal.f_px,
                                             gbest[2])
                if n_rematch >= MIN_INLIERS:
                    cand = scored("global+rematch", refine(T))
                    if cand[0]["score"] > gbest[0]["score"]:
                        gbest = cand
                if gbest[0]["score"] > best[0]["score"] + GLOBAL_MARGIN:
                    best = gbest
        sc, src, T = best
        ok = sc["score"] >= SCENE_ACCEPT
        return (T if ok else np.eye(4)), ok, {"scene_src": src, "n_global": n_global, "n_rematch": n_rematch,
                                              "n_view": n_view, **{f"scene_{k}": v for k, v in sc.items()}}

    def _appearance_diff(self, rgb, depth, f_cur, dT_c, min_pixels=500):
        """Median |grey difference| between the frame warped to the nominal view and the
        calibration reference, over static pixels both see (None if they overlap too little)."""
        cal = self.cal
        warped, seen = inverse_warp(rgb, depth, f_cur, dT_c, np.eye(4), f_dst=cal.f_px)
        m = seen & cal.ref_mask
        if m.sum() < min_pixels:
            return None
        return float(np.median(np.abs(gray(warped)[m] - gray(cal.ref_rgb)[m])))

    @staticmethod
    def _score(sc: dict) -> dict:
        """Geometric agreement times grey-value correlation on the agreeing points (absolute
        grey values would also accept sliding along a uniformly textured table, and fail on a
        re-lit scene; correlation does neither), or geometry alone when the reference has
        enough off-plane structure (walls, fixtures) to pin sliding along the table/floor."""
        sc = dict(sc)
        geo_only = min(sc["geo"], sc["geo_off"]) if sc["support_off"] >= SUPPORT_OFF else 0.0
        sc["score"] = max(sc["geo"] * max(sc["ncc"], 0.0), GEO_DISCOUNT * geo_only)
        return sc

    @staticmethod
    def _coarse(sc: dict) -> float:
        """Pre-ranking of unrefined global candidates: geometric agreement weighted by how much
        of the reference it explains (a hypothesis seeing only a sliver of the scene is weak)."""
        return sc["geo"] * min(1.0, sc["support"] / 1000.0) * (0.5 + 0.5 * sc["geo_off"])

    # ---------------------------------------------------------------- channel A
    def canon_image(self, sim, raw_obs) -> dict:
        if not self.use_cam:
            return raw_obs
        key = f"{self.camera}_image"
        depth = real_depth(sim, raw_obs[f"{self.camera}_depth"])
        # current camera expressed in the nominal camera frame is dT_c; target is identity
        f_src = self.f_cur or self.cal.f_px
        if WARP_DEPTH != "live":
            depth = self._proxy_depth(depth.shape, f_src)
        if WARP_MODE == "inverse":
            ref = (self.cal.ref_rgb, self.cal.ref_mask) if HOLE_FILL == "ref" else (None, None)
            warped, valid = inverse_warp(raw_obs[key], depth, f_src, self.dT_c, np.eye(4), f_dst=self.cal.f_px,
                                         fill_rgb=ref[0], fill_mask=ref[1])
        else:
            warped, valid = forward_splat(raw_obs[key], depth, f_src, self.dT_c, np.eye(4), f_dst=self.cal.f_px)
            if HOLE_FILL == "ref":
                use = ~valid & self.cal.ref_mask
                warped[use] = self.cal.ref_rgb[use]
        out = dict(raw_obs)
        out[key] = warped
        self.info["valid_frac"] = float(valid.mean())
        return out

    def _proxy_depth(self, shape, f_src):
        """Depth ablations for channel A: the static reference depth seen from the current camera
        ("ref": objects and arm are missing), or the support plane alone ("plane": a homography)."""
        cal = self.cal
        H, W = shape
        n, d = self._plane_cur(f_src, shape)
        rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        ray = np.stack([(cc + 0.5 - W / 2) / f_src, (rr + 0.5 - H / 2) / f_src, -np.ones_like(cc, float)], -1)
        den = ray @ n
        plane = np.where(den < -1e-6, -d / np.minimum(den, -1e-6), 0.0)      # depth along -z of the plane hit
        if WARP_DEPTH == "plane":
            return plane
        z, known = _splat_depth(cal.ref_depth, cal.f_px, invert(self.dT_c), f_src)
        return np.where(known & (z > 1e-3), z, plane)

    def _plane_cur(self, f_src, shape):
        """Support plane of the nominal view, expressed in the current camera frame (n.x + d = 0)."""
        cal = self.cal
        up_ref = cal.T_bc0[:3, :3].T @ cal.T_wb0[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        planes = _support_planes(_cloud(cal.ref_depth, cal.f_px, cal.ref_mask), up_ref, k=1)
        n0, d0 = planes[0] if planes else (up_ref, 1.0)
        Rc, tc = self.dT_c[:3, :3], self.dT_c[:3, 3]      # current camera pose in the nominal frame
        n = Rc.T @ n0
        return n, d0 + n0 @ tc

    # ---------------------------------------------------------------- channel B
    def canon_pos(self, p):
        return self.C[:3, :3] @ np.asarray(p, float) + self.C[:3, 3] if self.use_base else np.asarray(p, float)

    def canon_rotmat(self, M):
        return self.C[:3, :3] @ np.asarray(M, float) if self.use_base else np.asarray(M, float)

    def canon_quat_xyzw(self, q):
        q = np.asarray(q, float)
        if not self.use_base:
            return q
        out = (R.from_matrix(self.C[:3, :3]) * R.from_quat(q)).as_quat()
        # keep the input's hemisphere so the downstream quat->axis-angle stays continuous
        return out if np.dot(out, q) >= 0 else -out

    def decanon_action(self, a):
        """Policy action in the nominal frame -> action in the robot's own (believed) frame."""
        a = np.asarray(a, float).copy()
        if self.use_base:
            Rinv = self.C[:3, :3].T
            a[:3] = Rinv @ a[:3]
            a[3:6] = Rinv @ a[3:6]
        return a
