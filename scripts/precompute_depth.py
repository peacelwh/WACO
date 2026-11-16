"""Per-frame metric depth for the LIBERO third-person frames (round-trip reprojection of the
error-aware fine-tuning).

1. Depth-Anything-V2 (metric, indoor, small) predicts depth for every agentview frame.
2. Per frame, fit an affine map in inverse depth, 1/z_true ~ s * (1/z_pred) + b, on pixels
   that show the static workspace: inside the calibration static mask and with colour close
   to the empty-workspace reference. This anchors the prediction to the calibrated metric
   scale (the calibration RGB-D is a one-off capture, like on a real robot).
3. Store float16 depth at 128x128 in POLICY orientation (the dataset's 180-degree-rotated
   view) as a memmap indexed by the dataset's global frame index.
"""

from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from poseshift.paths import ROOT  # noqa: E402

DS_ROOT = os.environ.get("LIBERO_DS_ROOT")   # local copy of lerobot/libero (default: the Hub cache)
OUT = os.path.join(ROOT, "data", "depth_dav2_128.npy")
META = OUT.replace(".npy", ".json")
RES = 128
IN = 266  # DA-V2 input side, multiple of 14
DEVICE = os.environ.get("DEVICE", "cuda:0")
BATCH = int(os.environ.get("BATCH", "64"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from transformers import AutoModelForDepthEstimation  # noqa: E402

ds = LeRobotDataset("lerobot/libero", root=DS_ROOT, video_backend="pyav")
N = len(ds)
names = {int(r["task_index"]): str(t).strip().lower() for t, r in ds.meta.tasks.iterrows()}

# calibration bundles in policy orientation, downsampled
calib = {}
for p in sorted(glob.glob(os.path.join(ROOT, "data", "calib_libero", "*.npz"))):
    z = np.load(p, allow_pickle=True)
    ref_rgb = torch.from_numpy(np.ascontiguousarray(z["ref_rgb"][::-1, ::-1])).permute(2, 0, 1).float() / 255
    ref_d = torch.from_numpy(np.ascontiguousarray(z["ref_depth"][::-1, ::-1])).float()[None]
    mask = torch.from_numpy(np.ascontiguousarray(z["ref_mask"][::-1, ::-1])).float()[None]
    calib[str(z["language"]).strip().lower()] = (
        F.interpolate(ref_rgb[None], size=(RES, RES), mode="area")[0],
        F.interpolate(ref_d[None], size=(RES, RES), mode="nearest")[0, 0],
        F.interpolate(mask[None], size=(RES, RES), mode="nearest")[0, 0] > 0.5,
    )
print(f"frames {N}, tasks {len(names)}, calibrated {sum(n in calib for n in names.values())}", flush=True)

model = AutoModelForDepthEstimation.from_pretrained(
    os.environ.get("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")).to(DEVICE).eval().half()
mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

out = np.lib.format.open_memmap(OUT, mode="w+" if not os.path.exists(OUT) else "r+", dtype=np.float16, shape=(N, RES, RES))
done = np.zeros(N, bool)
if os.path.exists(META):
    done[json.load(open(META)).get("done_upto", 0):] = False
    done[: json.load(open(META)).get("done_upto", 0)] = True
start = int(done.sum())


class Frames(torch.utils.data.Dataset):
    def __len__(self):
        return N - start

    def __getitem__(self, i):
        it = ds[start + i]
        return it["observation.images.image"], int(it["task_index"]), int(it["index"])


dl = torch.utils.data.DataLoader(Frames(), batch_size=BATCH, num_workers=int(os.environ.get("WORKERS", "16")), shuffle=False)
stats = {"fit_ok": 0, "fit_fail": 0, "scale": [], "resid": []}
t0 = time.time()
for bi, (img, tix, gidx) in enumerate(dl):
    with torch.no_grad():
        x = F.interpolate(img.to(DEVICE).float(), size=(IN, IN), mode="bilinear", align_corners=False)
        pred = model(pixel_values=((x - mean) / std).half()).predicted_depth.float()
        pred = F.interpolate(pred[:, None], size=(RES, RES), mode="bilinear", align_corners=False)[:, 0].cpu()
    small = F.interpolate(img.float(), size=(RES, RES), mode="area")
    for k in range(len(gidx)):
        d = pred[k].clamp(min=0.05)
        name = names.get(int(tix[k]))
        if name in calib:
            ref_rgb, ref_d, mask = calib[name]
            m = mask & ((small[k] - ref_rgb).abs().mean(0) < 0.06) & (ref_d > 0.05)
            if m.sum() > 200:
                xi, yi = 1.0 / d[m], 1.0 / ref_d[m]
                for _ in range(2):  # robust LS in inverse depth
                    A = torch.stack([xi, torch.ones_like(xi)], 1)
                    sol = torch.linalg.lstsq(A, yi[:, None]).solution[:, 0]
                    r = (A @ sol - yi).abs()
                    keep = r < torch.quantile(r, 0.8)
                    xi, yi = xi[keep], yi[keep]
                s_, b_ = float(sol[0]), float(sol[1])
                inv = s_ / d + b_
                if s_ > 0 and torch.isfinite(inv).all() and (inv > 0).float().mean() > 0.99:
                    d = 1.0 / inv.clamp(min=1e-3)
                    stats["fit_ok"] += 1
                    stats["scale"].append(s_)
                    stats["resid"].append(float(r[keep].median()))
                else:
                    stats["fit_fail"] += 1
            else:
                stats["fit_fail"] += 1
        out[int(gidx[k])] = d.numpy().astype(np.float16)
    if bi % 50 == 0:
        n = start + (bi + 1) * BATCH
        el = time.time() - t0
        print(f"{n}/{N} frames | {n / max(el, 1):.0f} fps | fit ok {stats['fit_ok']} fail {stats['fit_fail']} | "
              f"median scale {np.median(stats['scale'][-2000:]) if stats['scale'] else float('nan'):.3f}", flush=True)
        out.flush()
        json.dump({"done_upto": min(n, N), "res": RES, "orientation": "policy"}, open(META, "w"))
out.flush()
json.dump({"done_upto": N, "res": RES, "orientation": "policy", "fit_ok": stats["fit_ok"], "fit_fail": stats["fit_fail"]}, open(META, "w"))
print("DEPTH_DONE", stats["fit_ok"], stats["fit_fail"], flush=True)
