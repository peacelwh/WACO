"""Scan every LIBERO task for base perturbations that are physically invalid.

For each (suite, task, level) and several sampled base perturbations, reset the scene and
record (a) penetration between robot/mount geoms and the environment right after reset and
(b) whether the arm is stable after settling. A perturbation that embeds the robot in the
table cannot be solved by any policy; such samples are reported here and are kept in the
evaluation, so they bound the success at base and compound L3.
"""

from __future__ import annotations

import json
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark  # noqa: E402

from poseshift.libero_env import LiberoPoseShiftEnv  # noqa: E402

from poseshift.paths import ROOT  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "feasibility")
os.makedirs(OUT, exist_ok=True)
SUITES = os.environ.get("SUITES", "libero_spatial,libero_object,libero_goal,libero_10").split(",")
N_EP = int(os.environ.get("N_EP", "6"))


def robot_contacts(sim) -> float:
    """Most negative penetration depth between a robot/mount geom and a non-robot geom."""
    m, d = sim.model, sim.data
    worst = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = m.body_id2name(m.geom_bodyid[c.geom1]) or ""
        b2 = m.body_id2name(m.geom_bodyid[c.geom2]) or ""
        r1 = b1.startswith(("robot0", "mount0", "gripper0"))
        r2 = b2.startswith(("robot0", "mount0", "gripper0"))
        if r1 != r2:
            worst = min(worst, c.dist)
    return worst


report = []
for sname in SUITES:
    su = benchmark.get_benchmark_dict()[sname]()
    for tid in range(len(su.tasks)):
        for level in ("l1", "l2", "l3"):
            bad = 0
            worst = 0.0
            drift = []
            for ep in range(N_EP):
                e = LiberoPoseShiftEnv(task_suite=su, task_id=tid, task_suite_name=sname, obs_type="pixels_agent_pos",
                                       observation_height=64, observation_width=64, episode_index=ep,
                                       perturb_kind="base", perturb_level=level, action_frame="base")
                o, info = e.reset(seed=ep)
                pen = robot_contacts(e._env.sim)
                q0 = e._env.robots[0]._joint_positions.copy()
                for _ in range(20):
                    o, *_ = e.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
                q1 = e._env.robots[0]._joint_positions.copy()
                drift.append(float(np.abs(q1 - q0).max()))
                worst = min(worst, pen)
                if pen < -0.005 or drift[-1] > 0.1:
                    bad += 1
                e.close()
            row = {"suite": sname, "task_id": tid, "level": level, "n": N_EP, "invalid": bad,
                   "worst_penetration_m": worst, "max_joint_drift_rad": max(drift)}
            report.append(row)
            print(f"{sname}[{tid}] {level}: invalid {bad}/{N_EP} worst_pen={100 * worst:.2f} cm "
                  f"max_drift={max(drift):.3f} rad", flush=True)

with open(os.path.join(OUT, "base_feasibility.json"), "w") as f:
    json.dump(report, f, indent=2)
tot = sum(r["invalid"] for r in report)
n = sum(r["n"] for r in report)
print(f"FEASIBILITY_DONE invalid {tot}/{n}")
