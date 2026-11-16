"""Export the shifts of LIBERO-PoseShift: one entry per (task, episode, kind, level).

The benchmark's sampler is deterministic, so this file is a reference for other
implementations of the perturbation layer: every method sees exactly these scenes.

  python scripts/export_perturbations.py data/poseshift_perturbations.json
"""

from __future__ import annotations

import json
import sys

from libero.libero import benchmark

from poseshift.perturb import LEVELS, sample_perturbation

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
N_EP = 10   # episodes per task and condition
SEED = 0    # the perturbation seed shared by all methods


def rounded(spec: dict) -> dict:
    return {k: ([round(x, 6) for x in v] if isinstance(v, list) else round(v, 6)) for k, v in spec.items()}


out = {}
for sname in SUITES:
    suite = benchmark.get_benchmark_dict()[sname]()
    tasks = []
    for tid in range(len(suite.tasks)):
        task = suite.get_task(tid)
        episodes = []
        for ep in range(N_EP):
            episodes.append({f"{kind}/{level}": rounded(sample_perturbation(kind, level, task.name, tid, ep, seed=SEED).to_dict())
                             for kind in ("cam", "base", "both") for level in ("l1", "l2", "l3")})
        tasks.append({"task_id": tid, "name": task.name, "language": task.language, "episodes": episodes})
    out[sname] = tasks

doc = {
    "benchmark": "LIBERO-PoseShift",
    "units": {"cam_dpos": "m (world frame)", "cam_drot_rpy": "deg (world axes, about the camera centre)",
              "base_dpos": "m (world frame)", "base_dyaw": "deg (about the vertical)"},
    "levels": {k: dict(zip(("cam_t_m", "cam_r_deg", "base_t_m", "base_yaw_deg"), v)) for k, v in LEVELS.items() if k != "clean"},
    "perturbation_seed": SEED,
    "suites": out,
}
with open(sys.argv[1], "w") as fh:
    json.dump(doc, fh, separators=(",", ":"))
print("wrote", sys.argv[1], sum(len(t["episodes"]) * 9 for s in out.values() for t in s), "shifts")
