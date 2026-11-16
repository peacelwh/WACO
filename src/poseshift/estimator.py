"""WACO front end: estimate camera shift dT_c and base shift dT_b from sensor data only.

Allowed inputs (everything a real robot has after a one-off calibration):
  * the current agentview RGB-D frame and its intrinsics,
  * the robot's joint state, turned into a base-frame point cloud of the arm by forward
    kinematics of its meshes,
  * calibration constants from demo collection: the nominal camera-in-base pose T_bc^0 and
    one reference RGB-D frame of the scene taken from the nominal camera.
Never used: ground-truth camera or base poses, segmentation, object poses, simulator state.

Conventions: 4x4 homogeneous matrices; camera frames follow MuJoCo (x right, y up, looking
down -z); depth images are raw MuJoCo orientation (row 0 at the bottom). T_ab maps
coordinates in frame b to frame a.

    T_wb = T_wb^0 dT_b,   T_wc = T_wc^0 dT_c,   T_bc = dT_b^{-1} T_bc^0 dT_c
  =>  dT_b = T_bc^0 dT_c T_bc^{-1}                                    (fusion)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from .geometry import invert

__all__ = [
    "arm_global",
    "arm_plane_search",
    "orbit_pose",
    "SceneScorer",
    "scene_rematch",
    "scene_global",
    "scene_refine",
    "RobotCloud",
    "backproject",
    "umeyama",
    "arm_anchor",
    "world_anchor",
    "fuse",
    "se2_project",
    "pose_error",
]


# --------------------------------------------------------------------------- geometry
def backproject(depth: np.ndarray, f_px: float, stride: int = 1, max_depth: float = 3.0):
    """Metric depth (raw orientation) -> (N, 3) camera-frame points and their (row, col)."""
    H, W = depth.shape
    rr, cc = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    d = depth[rr, cc]
    ok = (d > 1e-3) & (d < max_depth)
    rr, cc, d = rr[ok], cc[ok], d[ok]
    x = (cc + 0.5 - W / 2.0) * d / f_px
    y = (rr + 0.5 - H / 2.0) * d / f_px
    return np.stack([x, y, -d], axis=1), np.stack([rr, cc], axis=1)


def project(points_cam: np.ndarray, f_px: float, H: int, W: int):
    z = -points_cam[:, 2]
    ok = z > 1e-4
    c = f_px * points_cam[:, 0] / np.maximum(z, 1e-9) + W / 2.0 - 0.5
    r = f_px * points_cam[:, 1] / np.maximum(z, 1e-9) + H / 2.0 - 0.5
    return np.stack([r, c], axis=1), ok


def umeyama(src: np.ndarray, dst: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Rigid T (no scale) minimising sum w_i ||T src_i - dst_i||^2."""
    if w is None:
        w = np.ones(len(src))
    w = w / w.sum()
    mu_s = (w[:, None] * src).sum(0)
    mu_d = (w[:, None] * dst).sum(0)
    S = ((dst - mu_d) * w[:, None]).T @ (src - mu_s)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ D @ Vt
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = mu_d - Rm @ mu_s
    return T


def transform(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    """(translation error in metres, rotation error in degrees)."""
    D = invert(T_gt) @ T_est
    return float(np.linalg.norm(D[:3, 3])), float(np.degrees(np.linalg.norm(R.from_matrix(D[:3, :3]).as_rotvec())))


def se2_project(T: np.ndarray) -> np.ndarray:
    """Closest planar motion: keep x, y and yaw; zero z, roll and pitch."""
    yaw = np.arctan2(T[1, 0], T[0, 0])
    out = np.eye(4)
    out[:3, :3] = R.from_euler("z", yaw).as_matrix()
    out[:2, 3] = T[:2, 3]
    return out


# --------------------------------------------------------------------------- robot model
class RobotCloud:
    """Surface samples of the robot's visual meshes, expressed in the robot base frame.

    Positions are computed from the simulator's kinematic tree *relative to the base body*,
    which is exactly forward kinematics of the current joint state; the absolute placement
    of the base in the world never enters.
    """

    PREFIXES = ("robot0_", "gripper0_", "mount0_")

    def __init__(self, sim, base_body: str = "robot0_base", n_points: int = 4000, seed: int = 0):
        m = sim.model
        self.base_id = m.body_name2id(base_body)
        rng = np.random.default_rng(seed)
        geoms, verts = [], []
        for g in range(m.ngeom):
            body = m.body_id2name(m.geom_bodyid[g]) or ""
            if not body.startswith(self.PREFIXES):
                continue
            if int(m.geom_type[g]) != 7 or int(m.geom_group[g]) != 1:  # visual meshes only
                continue
            mid = int(m.geom_dataid[g])
            a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            v = np.asarray(m.mesh_vert[a:a + n], dtype=np.float64)
            geoms.append(g)
            verts.append(v)
        counts = np.array([len(v) for v in verts], dtype=float)
        # allocate samples proportionally to vertex count, at least a few per mesh
        alloc = np.maximum(8, np.round(n_points * counts / counts.sum())).astype(int)
        self._samples = []
        for g, v, k in zip(geoms, verts, alloc, strict=True):
            idx = rng.choice(len(v), size=min(k, len(v)), replace=False)
            self._samples.append((g, v[idx]))

    def points_base(self, sim) -> np.ndarray:
        d = sim.data
        T_wb = np.eye(4)
        T_wb[:3, :3] = d.body_xmat[self.base_id].reshape(3, 3)
        T_wb[:3, 3] = d.body_xpos[self.base_id]
        T_bw = invert(T_wb)
        out = []
        for g, v in self._samples:
            T_wg = np.eye(4)
            T_wg[:3, :3] = d.geom_xmat[g].reshape(3, 3)
            T_wg[:3, 3] = d.geom_xpos[g]
            out.append(transform(T_bw @ T_wg, v))
        return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- arm anchor
def _visible(model_b, T_cb, depth, f_px, occ_tol):
    """Model points that land inside the image and are not hidden behind observed surfaces."""
    H, W = depth.shape
    P = transform(T_cb, model_b)
    uv, ok = project(P, f_px, H, W)
    rc = np.round(uv).astype(int)
    inside = ok & (rc[:, 0] >= 0) & (rc[:, 0] < H) & (rc[:, 1] >= 0) & (rc[:, 1] < W)
    vis = np.zeros(len(model_b), bool)
    idx = np.flatnonzero(inside)
    z_model = -P[idx, 2]
    z_obs = depth[rc[idx, 0], rc[idx, 1]]
    # an observed surface clearly in FRONT of the model point means the point is occluded
    vis[idx] = z_obs > z_model - occ_tol
    return vis, P


def _icp(model_b, depth, f_px, tree, obs, T_cb, schedule, trim):
    """Visibility-aware trimmed point-to-point ICP; schedule = [(iters, max_dist), ...]."""
    for iters, max_dist in schedule:
        for _ in range(iters):
            vis, P = _visible(model_b, T_cb, depth, f_px, occ_tol=max_dist)
            if vis.sum() < 50:
                return T_cb, np.inf, -np.inf
            dist, idx = tree.query(P[vis], distance_upper_bound=max_dist)
            ok = np.isfinite(dist)
            if ok.sum() < 50:
                return T_cb, np.inf, -np.inf
            d_ok = dist[ok]
            keep = d_ok <= np.quantile(d_ok, trim)
            T_cb = umeyama(model_b[vis][ok][keep], obs[idx[ok][keep]])
    score, rms = _depth_consistency(model_b, T_cb, depth, f_px)
    return T_cb, rms, score


def _depth_consistency(model_b, T_cb, depth, f_px, tol=0.015, gap=0.03):
    """Render-and-compare score of a pose hypothesis.

    For model points inside the image: |z_obs - z_model| < tol counts as explained;
    z_obs > z_model + gap is a free-space violation -- the camera sees *through* where the arm
    would be -- and is penalised twice. Points hidden behind observed surfaces are neutral.
    This rejects the degenerate solutions where the model lies flat on the table plane.
    """
    H, W = depth.shape
    P = transform(T_cb, model_b)
    uv, ok = project(P, f_px, H, W)
    rc = np.round(uv).astype(int)
    inside = ok & (rc[:, 0] >= 0) & (rc[:, 0] < H) & (rc[:, 1] >= 0) & (rc[:, 1] < W)
    idx = np.flatnonzero(inside)
    if len(idx) == 0:
        return -np.inf, np.inf
    dz = depth[rc[idx, 0], rc[idx, 1]] - (-P[idx, 2])
    fit = np.abs(dz) < tol
    viol = dz > gap
    rms = float(np.sqrt(np.mean(dz[fit] ** 2))) if fit.any() else np.inf
    return float(fit.sum() - 2.0 * viol.sum()), rms


@dataclass
class ArmResult:
    T_bc: np.ndarray
    residual: float
    overlap: float


def arm_anchor(depth, f_px, model_b, T_bc_prior, yaw_grid=(-16, -8, 0, 8, 16),
                xy_grid=((0, 0), (0.08, 0), (-0.08, 0), (0, 0.08), (0, -0.08)), trim=0.8,
                T_bc_init: np.ndarray | None = None, max_shift_m: float = 0.25,
                max_shift_deg: float = 30.0) -> ArmResult:
    """Register the arm model (base frame) to the observed depth cloud (camera frame).

    Only model points that are inside the image and not occluded under the current pose
    hypothesis take part (from this viewpoint the table hides the pedestal and much of the
    lower arm). Multi-start: base-yaw / base-translation hypotheses around the calibrated
    prior are ranked by how many model points they explain within 2 cm after a short
    coarse ICP; the best three are refined with a tightening tolerance.
    """
    obs, _ = backproject(depth, f_px, stride=1)
    tree = cKDTree(obs)
    # centre the hypotheses on the world-anchor estimate when available: it already accounts
    # for the camera moving, so the grid only has to cover the base moving
    T_cb0 = invert(T_bc_prior if T_bc_init is None else T_bc_init)
    starts = []
    for yaw in yaw_grid:
        for dx, dy in xy_grid:
            dTb = np.eye(4)
            dTb[:3, :3] = R.from_euler("z", yaw, degrees=True).as_matrix()
            dTb[:2, 3] = (dx, dy)
            starts.append(T_cb0 @ dTb)          # hypothesis: the base moved by dTb
    def plausible(T_cb):
        # the implied base motion relative to the initialisation must be physically plausible:
        # a robot is nudged by centimetres, not dropped onto the table
        d = invert(T_cb0) @ T_cb
        ang = np.degrees(np.linalg.norm(R.from_matrix(d[:3, :3]).as_rotvec()))
        return np.linalg.norm(d[:3, 3]) <= max_shift_m and ang <= max_shift_deg

    coarse = [_icp(model_b, depth, f_px, tree, obs, T, [(6, 0.10)], trim) for T in starts]
    coarse = [c for c in coarse if plausible(c[0])] or [(T_cb0, np.inf, -np.inf)]
    coarse.sort(key=lambda t: -t[2])
    fine = [_icp(model_b, depth, f_px, tree, obs, T, [(10, 0.05), (15, 0.02), (10, 0.01)], trim)
            for T, _, _ in coarse[:5]]
    fine = [c for c in fine if plausible(c[0])] or [(T_cb0, np.inf, -np.inf)]
    fine.sort(key=lambda t: -t[2])
    T_cb, rms, score = fine[0]
    return ArmResult(T_bc=invert(T_cb), residual=rms, overlap=float(score) / len(model_b))


# --------------------------------------------------------------------------- world anchor
@dataclass
class WorldResult:
    dT_c: np.ndarray
    inliers: int
    matches: int
    pairs_cur: np.ndarray | None = None   # inlier 3D points, current camera frame
    pairs_ref: np.ndarray | None = None   # the same points in the reference camera frame


_SIFT = None


def _sift():
    global _SIFT
    if _SIFT is None:
        import cv2

        _SIFT = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01, edgeThreshold=15)
    return _SIFT


def world_anchor(rgb, depth, ref_rgb, ref_depth, f_px, mask=None, ref_mask=None,
                 iters=2000, thresh=0.01, seed=0, f_ref: float | None = None) -> WorldResult:
    """Camera motion from the reference view to the current view via 3D-3D RANSAC.

    Points seen in both frames satisfy X_ref = dT_c X_cur, so the registration directly gives
    dT_c. `mask`/`ref_mask` (True = usable) exclude pixels that are not static scene, e.g.
    the arm predicted from the arm anchor.
    """
    import cv2

    def feats(img, m):
        g = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)
        mk = None if m is None else (m.astype(np.uint8) * 255)
        return _sift().detectAndCompute(g, mk)

    k1, d1 = feats(rgb, mask)
    k2, d2 = feats(ref_rgb, ref_mask)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return WorldResult(np.eye(4), 0, 0)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    H, W = depth.shape

    def lift(kps, dep, ids, f):
        out = []
        for i in ids:
            c, r = kps[i].pt
            ri, ci = int(round(r)), int(round(c))
            if not (0 <= ri < H and 0 <= ci < W):
                out.append(None)
                continue
            dd = dep[ri, ci]
            win = dep[max(0, ri - 2):ri + 3, max(0, ci - 2):ci + 3]
            # keypoints sit on corners/edges; reject those on a depth discontinuity, whose
            # lifted 3D position would mix foreground and background. A smooth slope (the
            # table seen at a grazing angle) keeps the centre close to the window median.
            if not (1e-3 < dd < 3.0) or abs(dd - np.median(win)) > 0.005:
                out.append(None)
                continue
            out.append(np.array([(c + 0.5 - W / 2) * dd / f, (r + 0.5 - H / 2) * dd / f, -dd]))
        return out

    P1 = lift(k1, depth, [m.queryIdx for m in good], f_px)
    P2 = lift(k2, ref_depth, [m.trainIdx for m in good], f_px if f_ref is None else f_ref)
    pts = [(a, b) for a, b in zip(P1, P2, strict=True) if a is not None and b is not None]
    if len(pts) < 6:
        return WorldResult(np.eye(4), 0, len(good))
    A = np.stack([a for a, _ in pts])
    B = np.stack([b for _, b in pts])
    rng = np.random.default_rng(seed)
    best, best_in = np.eye(4), np.zeros(len(A), bool)
    for _ in range(iters):
        s = rng.choice(len(A), 3, replace=False)
        T = umeyama(A[s], B[s])
        inl = np.linalg.norm(transform(T, A) - B, axis=1) < thresh
        if inl.sum() > best_in.sum():
            best, best_in = T, inl
    if best_in.sum() >= 6:
        best = umeyama(A[best_in], B[best_in])
    return WorldResult(best, int(best_in.sum()), len(pts), A[best_in], B[best_in])


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _normals(depth, f_px):
    """Per-pixel camera-frame points and normals from a depth map (central differences)."""
    H, W = depth.shape
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    X = np.stack([(cc + 0.5 - W / 2) * depth / f_px, (rr + 0.5 - H / 2) * depth / f_px, -depth], -1)
    dx = np.zeros_like(X)
    dy = np.zeros_like(X)
    dx[:, 1:-1] = X[:, 2:] - X[:, :-2]
    dy[1:-1] = X[2:] - X[:-2]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return X, n


def scene_refine(depth, ref_depth, ref_mask, f_px, dT_c_init, iters=25, n_model=6000, seed=0,
                 pairs_cur=None, pairs_ref=None, pair_weight=20.0, f_ref: float | None = None,
                 bounds=((8, 0.05), (None, 0.02)), model_pts=None):
    """Point-to-plane ICP of the static reference cloud against the current cloud.

    Model: reference points on static background (camera^0 frame). Target: current cloud
    with normals. Estimates A = dT_c^{-1} (maps camera^0 points into the current camera
    frame); returns dT_c. Trimmed + Huber-weighted so that objects and the arm, which moved,
    drop out as outliers. The table plane pins z/roll/pitch, walls and furniture pin the rest.
    Sparse feature matches (`pairs_*`, point-to-point) are added with a large weight: the
    table dominates the dense term and leaves in-plane sliding weakly constrained, whereas a
    textured feature pins all three translation directions.
    """
    rng = np.random.default_rng(seed)
    if model_pts is not None:          # e.g. a fused multi-view scan, already in the nominal frame
        pts = model_pts
    else:
        Xr, _ = _normals(ref_depth, f_px if f_ref is None else f_ref)
        ok = ref_mask & (ref_depth > 1e-3) & (ref_depth < 3.0)
        pts = Xr[ok]
    if len(pts) < 200:
        return dT_c_init, np.inf
    pts = pts[rng.choice(len(pts), min(n_model, len(pts)), replace=False)]
    Xc, Nc = _normals(depth, f_px)
    okc = (depth > 1e-3) & (depth < 3.0)
    tgt, tgt_n = Xc[okc], Nc[okc]
    tree = cKDTree(tgt)
    A = invert(dT_c_init)
    rms = np.inf
    for it in range(iters):
        src = transform(A, pts)
        bound = next(b for until, b in bounds if until is None or it < until)
        dist, idx = tree.query(src, distance_upper_bound=bound)
        m = np.isfinite(dist)
        if m.sum() < 100:
            break
        s_, d_, n_ = src[m], tgt[idx[m]], tgt_n[idx[m]]
        r = np.einsum("ij,ij->i", s_ - d_, n_)
        keep = np.abs(r) <= np.quantile(np.abs(r), 0.8)
        s_, n_, r = s_[keep], n_[keep], r[keep]
        w = 1.0 / np.maximum(1.0, np.abs(r) / 0.005)          # Huber, 5 mm
        J = np.hstack([np.cross(s_, n_), n_])
        H_ = (J * w[:, None]).T @ J + 1e-9 * np.eye(6)
        g = (J * w[:, None]).T @ r
        if pairs_cur is not None and pairs_ref is not None and len(pairs_cur) >= 3:
            q = transform(A, pairs_ref)                    # reference points mapped into current frame
            e = (q - pairs_cur)                            # (K, 3) residuals
            en = np.linalg.norm(e, axis=1)
            ok_p = en < (0.05 if it < 8 else 0.015)        # drop matches that disagree with the fit
            if ok_p.sum() >= 3:
                qk, ek = q[ok_p], e[ok_p]
                wk = pair_weight / np.maximum(1.0, np.linalg.norm(ek, axis=1) / 0.005)
                for k in range(len(qk)):
                    Jk = np.hstack([-_skew(qk[k]), np.eye(3)])     # d(q + w x q + t)/d(w, t)
                    H_ += wk[k] * Jk.T @ Jk
                    g += wk[k] * Jk.T @ ek[k]
        xi = -np.linalg.solve(H_, g)
        dA = np.eye(4)
        dA[:3, :3] = R.from_rotvec(xi[:3]).as_matrix()
        dA[:3, 3] = xi[3:]
        A = dA @ A
        rms = float(np.sqrt(np.mean(r ** 2)))
        if np.linalg.norm(xi) < 1e-6 and bound == bounds[-1][1]:
            break
    return invert(A), rms


# ------------------------------------------- wide viewpoint changes: dense scoring, global search
def gray(img: np.ndarray) -> np.ndarray:
    return img[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def _sample_points(rgb, depth, mask, f_px, n, seed=0):
    """Random static points (camera frame) with their grey values."""
    H, W = depth.shape
    rr, cc = np.nonzero(mask & (depth > 1e-3) & (depth < 3.0))
    sel = np.random.default_rng(seed).choice(len(rr), min(n, len(rr)), replace=False)
    rr, cc = rr[sel], cc[sel]
    d = depth[rr, cc]
    return np.stack([(cc + 0.5 - W / 2) * d / f_px, (rr + 0.5 - H / 2) * d / f_px, -d], 1), gray(rgb)[rr, cc]


class SceneScorer:
    """Dense check of a camera-shift hypothesis dT_c against the static reference.

    Reference static points (camera^0 frame) are mapped into the current camera and compared
    with the measured depth at the pixel they land on:
      agree     |d - z| < tol       the static surface is where the hypothesis puts it
      violation d > z + tol         we see THROUGH where a static surface should be
      occluded  d < z - tol         something (object, arm) in front -- no evidence either way
    `score` also requires the grey value to match on agreeing points, which disambiguates
    sliding along large planes (table, floor) that geometry alone cannot pin.
    """

    def __init__(self, ref_rgb, ref_depth, ref_mask, f_ref, up_ref=None, n=4000, seed=0, views=None):
        """`views`: optional extra calibration views [(T_c0_v, rgb, depth, mask)], fused in the
        nominal camera frame (a multi-view scan covers what a far-orbited camera sees)."""
        Ps, gs = [], []
        all_views = [(np.eye(4), ref_rgb, ref_depth, ref_mask)] + list(views or [])
        per = max(n // len(all_views), 500)
        for k, (T, rgb, dep, msk) in enumerate(all_views):
            P, g = _sample_points(rgb, dep, msk, f_ref, per if k else max(per, n // 2), seed + k)
            Ps.append(transform(T, P))
            gs.append(g)
        self.P = np.concatenate(Ps)
        self.g = np.concatenate(gs)
        # points off the support plane (walls, fixtures): geometry there pins what sliding
        # along the plane leaves free, without relying on appearance
        self.off = np.ones(len(self.P), bool)
        if up_ref is not None:
            planes = _support_planes(self.P, up_ref, k=1, seed=seed)
            if planes:
                n_, d_ = planes[0]
                self.off = np.abs(self.P @ n_ + d_) > 0.02

    def __call__(self, dT_c, depth, gray_cur, f_cur, tol=0.015, rel=0.02, photo_tol=25.0) -> dict:
        H, W = depth.shape
        Q = transform(invert(dT_c), self.P)
        z = -Q[:, 2]
        zs = np.maximum(z, 1e-9)
        c = np.floor(f_cur * Q[:, 0] / zs + W / 2).astype(np.int64)
        r = np.floor(f_cur * Q[:, 1] / zs + H / 2).astype(np.int64)
        idx = np.flatnonzero((z > 1e-3) & (c >= 0) & (c < W) & (r >= 0) & (r < H))
        dz = depth[r[idx], c[idx]] - z[idx]
        t = tol + rel * z[idx]
        agree = np.abs(dz) < t
        viol = dz > t
        gc, gr = gray_cur[r[idx], c[idx]], self.g[idx]
        photo = agree & (np.abs(gc - gr) < photo_tol)
        n_a, n_v = int(agree.sum()), int(viol.sum())
        off = self.off[idx]
        n_ao, n_vo = int((agree & off).sum()), int((viol & off).sum())
        # normalised cross-correlation of grey values on the agreeing points: insensitive to a
        # global brightness/contrast change of the scene since calibration
        ncc = 0.0
        if n_a >= 50:
            a, b = gc[agree] - gc[agree].mean(), gr[agree] - gr[agree].mean()
            ncc = float((a @ b) / (np.sqrt((a @ a) * (b @ b)) + 1e-9))
        geo = n_a / (n_a + n_v + 1)
        return {"photo": float(photo.sum() / (n_a + n_v + 1)), "ncc": ncc, "geo": float(geo),
                "geo_off": float(n_ao / (n_ao + n_vo + 1)), "support": n_a, "support_off": n_ao,
                "seen": float(len(idx) / len(self.P))}


def _cloud(depth, f_px, mask=None, n=6000, seed=0):
    H, W = depth.shape
    ok = (depth > 1e-3) & (depth < 3.0)
    if mask is not None:
        ok &= mask
    rr, cc = np.nonzero(ok)
    sel = np.random.default_rng(seed).choice(len(rr), min(n, len(rr)), replace=False)
    rr, cc = rr[sel], cc[sel]
    d = depth[rr, cc]
    return np.stack([(cc + 0.5 - W / 2) * d / f_px, (rr + 0.5 - H / 2) * d / f_px, -d], 1)


def _support_planes(P, up_hint, k=2, max_angle=45.0, iters=400, tol=0.01, seed=0):
    """Up to k large planes with normal within max_angle of up_hint (camera frame).

    Returns [(n, d)] with n.x + d = 0 and d > 0, i.e. n points from the plane towards the
    camera (upwards, for a camera looking down at a table or floor)."""
    rng = np.random.default_rng(seed)
    cos_max = np.cos(np.deg2rad(max_angle))
    out, free = [], np.ones(len(P), bool)
    for _ in range(k):
        Q = P[free]
        if len(Q) < 200:
            break
        best, best_cnt = None, 0
        for _ in range(iters):
            a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            d = -n @ a
            if d < 0:
                n, d = -n, -d
            if n @ up_hint < cos_max:
                continue
            cnt = int((np.abs(Q @ n + d) < tol).sum())
            if cnt > best_cnt:
                best, best_cnt = (n, d), cnt
        if best is None or best_cnt < 150:
            break
        n, d = best
        inl = np.abs(Q @ n + d) < tol                       # least-squares polish on the inliers
        c0 = Q[inl].mean(0)
        _, _, Vt = np.linalg.svd(Q[inl] - c0, full_matrices=False)
        n = Vt[-1] * (1 if Vt[-1] @ n > 0 else -1)
        d = -n @ c0
        if d < 0:
            n, d = -n, -d
        out.append((n, d))
        idx = np.flatnonzero(free)
        free[idx[np.abs(P[idx] @ n + d) < 2 * tol]] = False
    return out


def _plane_frame(n, d):
    """T_{camera <- plane}: origin at the camera's foot on the plane, z = n, x = optical axis
    projected onto the plane."""
    o = -d * n
    a = np.array([0.0, 0.0, -1.0])
    u = a - (a @ n) * n
    if np.linalg.norm(u) < 1e-6:
        u = np.array([1.0, 0.0, 0.0]) - n[0] * n
    u /= np.linalg.norm(u)
    T = np.eye(4)
    T[:3, :3] = np.stack([u, np.cross(n, u), n], 1)
    T[:3, 3] = o
    return T


def _height_map(Q, cell, N, h_min=0.015, h_max=0.4, flat=0.05):
    """Max-height map in plane coordinates; points on the plane mark its extent weakly."""
    m = np.zeros(N * N, np.float32)
    h = Q[:, 2]
    v = np.where(h > h_min, np.minimum(h, h_max), np.where(np.abs(h) < 0.01, flat, 0.0)).astype(np.float32)
    iu = np.floor(Q[:, 0] / cell).astype(np.int64) + N // 2
    iv = np.floor(Q[:, 1] / cell).astype(np.int64) + N // 2
    ok = (v > 0) & (iu >= 0) & (iu < N) & (iv >= 0) & (iv < N)
    np.maximum.at(m, iu[ok] * N + iv[ok], v[ok])
    return m.reshape(N, N)


def scene_global(depth, ref_depth, ref_mask, f_cur, f_ref, up_ref, bin_deg=10.0, yaw_step=2.0,
                 cell=0.025, N=192, seed=0, extra_pts=None) -> list[np.ndarray]:
    """Global camera-shift hypotheses that do not rely on feature matching.

    Large viewpoint changes (orbits of tens of degrees, cameras twice as far away) defeat
    SIFT, but both views still contain the support plane. Aligning it fixes tilt and height,
    leaving yaw + in-plane translation, which an exhaustive yaw sweep with 2D FFT
    cross-correlation of height maps (walls, fixtures, table outline) recovers. `up_ref` is
    the world up direction in the nominal camera frame (known from calibration); the current
    camera is assumed to be tilted by < 45 deg relative to it. Returns candidate dT_c's,
    best correlation first, for dense scoring and refinement by the caller.
    """
    Pr = _cloud(ref_depth, f_ref, ref_mask, seed=seed)
    if extra_pts is not None and len(extra_pts):
        Pr = np.concatenate([Pr, extra_pts])
    Pc = _cloud(depth, f_cur, None, seed=seed)
    ref_planes = _support_planes(Pr, up_ref, k=1, seed=seed)
    cur_planes = _support_planes(Pc, up_ref, k=2, seed=seed)
    if not ref_planes or not cur_planes:
        return []
    T_r = _plane_frame(*ref_planes[0])
    Qr = transform(invert(T_r), Pr)
    Am = _height_map(Qr, cell, N)
    A = np.fft.rfft2(Am)
    na = np.sqrt((Am ** 2).sum())
    peaks = []
    for pi, (n, d) in enumerate(cur_planes):
        T_c = _plane_frame(n, d)
        Qc = transform(invert(T_c), Pc)
        for yaw in np.arange(-180.0, 180.0, yaw_step):
            Rz = R.from_euler("z", yaw, degrees=True).as_matrix()
            Bm = _height_map(Qc @ Rz.T, cell, N)
            B = np.fft.rfft2(Bm)
            corr = np.fft.irfft2(A * np.conj(B), s=(N, N))
            k = int(np.argmax(corr))
            ku, kv = divmod(k, N)
            score = corr.flat[k] / (na * np.sqrt((Bm ** 2).sum()) + 1e-9)   # cosine similarity
            su = (ku if ku < N // 2 else ku - N) * cell
            sv = (kv if kv < N // 2 else kv - N) * cell
            M = np.eye(4)
            M[:3, :3] = Rz
            M[:3, 3] = [su, sv, 0.0]
            peaks.append((score, yaw, pi, T_r @ M @ invert(T_c)))
    # symmetric layouts (a rectangular table, a wall on each side) give strong mirrored peaks
    # that can outrank the true one, so return the best peak of EVERY yaw bin (per plane) and
    # let the caller's dense score decide
    best = {}
    for score, yaw, pi, T in peaks:
        key = (pi, int(((yaw + 180.0) % 360.0) // bin_deg))
        if key not in best or score > best[key][0]:
            best[key] = (score, yaw, pi, T)
    chosen = sorted(best.values(), key=lambda x: -x[0])
    return [T for *_, T in chosen]


def scene_rematch(rgb, depth, ref_rgb, ref_depth, ref_mask, f_cur, f_ref, dT_c, seed=0):
    """Polish a coarse camera shift by matching against the reference RE-RENDERED at it.

    A wide-baseline pair (tens of degrees apart) defeats SIFT; after warping the reference
    into the hypothesised current viewpoint the remaining difference is small, so features
    match again. The 3D-3D fit maps current points onto the re-rendered reference (both in
    the current camera frame), i.e. a correction dT with dT_c <- dT_c dT.
    Returns (dT_c, inliers).
    """
    from .warp import _splat_depth, inverse_warp

    # reference -> current viewpoint: the reference camera is the origin, the current one is dT_c
    ref_w, valid = inverse_warp(ref_rgb, ref_depth, f_ref, np.eye(4), dT_c, f_dst=f_cur)
    m3 = np.repeat(ref_mask[..., None].astype(np.uint8) * 255, 3, axis=2)
    mask_w, _ = inverse_warp(m3, ref_depth, f_ref, np.eye(4), dT_c, f_dst=f_cur, fill=0)
    mask_w = valid & (mask_w[..., 0] > 127)
    zbuf, known = _splat_depth(ref_depth, f_ref, invert(dT_c), f_cur)
    zbuf = np.where(valid & known, zbuf, 0.0)
    res = world_anchor(rgb, depth, ref_w, zbuf, f_cur, mask=None, ref_mask=mask_w, seed=seed, f_ref=f_cur)
    if res.inliers < 8:
        return dT_c, res.inliers
    return dT_c @ res.dT_c, res.inliers


def orbit_pose(T_wc0, P, az_deg, el_deg, scale):
    """Camera pose orbiting the workspace point P: raised by el, swung by az about the vertical,
    moved along the line of sight by `scale`, always keeping P in the same image position."""
    C0, R0 = T_wc0[:3, 3], T_wc0[:3, :3]
    r = C0 - P
    h = np.cross([0.0, 0.0, 1.0], r)
    h /= np.linalg.norm(h) + 1e-12
    Re = R.from_rotvec(h * np.deg2rad(el_deg)).as_matrix()
    if el_deg and (P + Re @ r)[2] < C0[2]:      # positive elevation raises the camera
        Re = R.from_rotvec(-h * np.deg2rad(el_deg)).as_matrix()
    Rm = R.from_euler("z", az_deg, degrees=True).as_matrix() @ Re
    T = np.eye(4)
    T[:3, :3] = Rm @ R0
    T[:3, 3] = P + scale * (Rm @ r)
    return T


def arm_global(depth, f_px, model_b, T_wb0, T_bc0, pivot_w, n_score=1500, top_k=8, seed=0,
               az=np.arange(-90.0, 90.1, 7.5), el=(-10.0, 0.0, 10.0, 20.0, 30.0),
               scale=(0.8, 1.0, 1.25, 1.5, 1.8, 2.1)):
    """Camera pose in the base frame from the ARM ALONE, globally (no scene, no prior pose).

    Hypotheses are viewpoints orbiting the workspace (as a bumped or re-mounted third-person
    camera would), ranked by render-and-compare of the arm model against the depth image; the
    best few are refined by visibility-aware ICP. Returns (T_bc, overlap, rms)."""
    T_wc0 = T_wb0 @ T_bc0
    rng = np.random.default_rng(seed)
    sub = model_b[rng.choice(len(model_b), min(n_score, len(model_b)), replace=False)]
    hyps = []
    for a in az:
        for e in el:
            for sc in scale:
                T_bc = invert(T_wb0) @ orbit_pose(T_wc0, pivot_w, a, e, sc)
                fit, _ = _depth_consistency(sub, invert(T_bc), depth, f_px)
                hyps.append((fit, T_bc))
    hyps.sort(key=lambda x: -x[0])
    obs, _ = backproject(depth, f_px, stride=1)
    tree = cKDTree(obs)
    best = (None, -np.inf, np.inf)
    for fit, T_bc in hyps[:top_k]:
        if not np.isfinite(fit):
            continue
        T_cb, rms, score = _icp(model_b, depth, f_px, tree, obs, invert(T_bc),
                                [(10, 0.08), (10, 0.03), (10, 0.015)], 0.8)
        if score > best[1]:
            best = (invert(T_cb), score, rms)
    if best[0] is None:
        return None, -np.inf, np.inf
    return best[0], float(best[1]) / len(model_b), best[2]


def _offset_fit(P0, off, depth, f_px, tol, gap):
    """Vectorised _depth_consistency for one point set P0 (N, 3) under G translations off (G, 3)."""
    H, W = depth.shape
    P = P0[None] + off[:, None]
    z = -P[..., 2]
    zs = np.maximum(z, 1e-6)
    c = np.floor(f_px * P[..., 0] / zs + W / 2).astype(np.int64)
    r = np.floor(f_px * P[..., 1] / zs + H / 2).astype(np.int64)
    inside = (z > 1e-3) & (c >= 0) & (c < W) & (r >= 0) & (r < H)
    dz = np.where(inside, depth[np.clip(r, 0, H - 1), np.clip(c, 0, W - 1)] - z, np.nan)
    with np.errstate(invalid="ignore"):
        fit = (np.abs(dz) < tol).sum(1) - 2.0 * (dz > gap).sum(1)
    return np.where(inside.any(1), fit, -np.inf)


def arm_plane_search(depth, f_cur, ref_depth, ref_mask, f_ref, up_ref, model_b, T_bc0, yaw_step=5.0,
                     cell=0.08, span=1.0, n_score=1000, keep=150, top_k=12, seed=0):
    """Camera pose from the ARM, searched within the support-plane constraint (no pivot assumed).

    Aligning the table/floor plane of the current and reference views fixes the camera's tilt
    and height; what remains -- yaw and in-plane translation -- is swept exhaustively. Each
    candidate is scored by render-and-compare of the arm model (base assumed unmoved): first
    with a loose depth tolerance so a coarse grid still sees the arm near the truth, then the
    best `keep` are refined on a fine local grid with the strict tolerance, and the best
    `top_k` are polished by visibility-aware ICP on the arm. Used when the static scene is too
    bare to tell these candidates apart. Returns [(dT_c, overlap, rms)], best overlap first."""
    rng = np.random.default_rng(seed)
    Pr = _cloud(ref_depth, f_ref, ref_mask, seed=seed)
    Pc = _cloud(depth, f_cur, None, seed=seed)
    ref_planes = _support_planes(Pr, up_ref, k=1, seed=seed)
    cur_planes = _support_planes(Pc, up_ref, k=2, seed=seed)
    if not ref_planes or not cur_planes:
        return []
    T_r = _plane_frame(*ref_planes[0])
    sub = model_b[rng.choice(len(model_b), min(n_score, len(model_b)), replace=False)]
    Q = transform(invert(T_r) @ invert(T_bc0), sub)                  # arm in the reference plane frame

    def fits(T_c, yaw, tx, ty, tol, gap):
        # camera-frame points for M = (R_z(yaw), [tx, ty, 0]):  T_c M^-1 Q
        Rz = R.from_euler("z", yaw, degrees=True).as_matrix()
        P0 = (Q @ Rz) @ T_c[:3, :3].T + T_c[:3, 3]
        off = np.stack([tx, ty, np.zeros_like(tx)], 1) @ (-T_c[:3, :3] @ Rz.T).T
        return _offset_fit(P0, off, depth, f_cur, tol, gap)

    def pose(T_c, yaw, tx, ty):
        M = np.eye(4)
        M[:3, :3] = R.from_euler("z", yaw, degrees=True).as_matrix()
        M[:3, 3] = (tx, ty, 0.0)
        return T_r @ M @ invert(T_c)

    g = np.arange(-span, span + 1e-9, cell)
    gx, gy = [a.ravel() for a in np.meshgrid(g, g, indexing="ij")]
    cands = []
    for pi, (n, d) in enumerate(cur_planes):
        T_c = _plane_frame(n, d)
        for yaw in np.arange(-180.0, 180.0, yaw_step):
            fit = fits(T_c, yaw, gx, gy, 0.05, 0.08)
            cands += [(fit[k], pi, yaw, gx[k], gy[k]) for k in np.argsort(-fit)[:3] if np.isfinite(fit[k])]
    cands.sort(key=lambda x: -x[0])
    lg = np.arange(-0.75 * cell, 0.75 * cell + 1e-9, 0.02)
    lx, ly = [a.ravel() for a in np.meshgrid(lg, lg, indexing="ij")]
    refined = []
    for _, pi, yaw, x0, y0 in cands[:keep]:
        T_c = _plane_frame(*cur_planes[pi])
        best = (-np.inf, None)
        for y in yaw + np.arange(-yaw_step / 2, yaw_step / 2 + 1e-9, 1.25):
            fit = fits(T_c, y, x0 + lx, y0 + ly, 0.015, 0.03)
            k = int(np.argmax(fit))
            if fit[k] > best[0]:
                best = (fit[k], pose(T_c, y, x0 + lx[k], y0 + ly[k]))
        refined.append(best)
    refined.sort(key=lambda x: -x[0])
    obs, _ = backproject(depth, f_cur, stride=1)
    tree = cKDTree(obs)
    out = []
    for fit, dT in refined[:top_k]:
        T_cb, rms, score = _icp(model_b, depth, f_cur, tree, obs, invert(T_bc0 @ dT),
                                [(10, 0.03), (10, 0.015)], 0.8)
        if np.isfinite(rms):
            out.append((invert(T_bc0) @ invert(T_cb), float(score) / len(model_b), rms))
    out.sort(key=lambda x: -x[1])
    return out

# --------------------------------------------------------------------------- fusion
def fuse(T_bc: np.ndarray, dT_c: np.ndarray, T_bc0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """dT_b = T_bc^0 dT_c T_bc^{-1}; returns (planar dT_b, raw 6-DoF dT_b, off-plane residual)."""
    raw = T_bc0 @ dT_c @ invert(T_bc)
    planar = se2_project(raw)
    rpy = R.from_matrix(raw[:3, :3]).as_euler("xyz", degrees=True)
    off = float(np.hypot(raw[2, 3] * 100.0, np.hypot(rpy[0], rpy[1])))  # cm and deg mixed on purpose
    return planar, raw, off
