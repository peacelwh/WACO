"""Success rate vs perturbation level, one panel per perturbation kind, one line per method.

Usage: python scripts/plot_results.py OUT.png B1=outputs/eval/b1 WACO=outputs/eval/waco ...
Reads the per-cell eval_info.json files (aggregated over suites).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LEVELS = ["clean", "l1", "l2", "l3"]
KINDS = [("cam", "camera shift"), ("base", "robot-base shift"), ("both", "camera + base")]
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]


def load(run: Path):
    acc = defaultdict(lambda: [0, 0])
    for d in run.iterdir():
        f = d / "eval_info.json"
        if not f.exists():
            continue
        parts = d.name.split("__")
        if len(parts) != 4:
            continue
        _, kind, level, _ = parts
        o = json.loads(f.read_text())["overall"]
        n = int(o["n_episodes"])
        acc[(kind, level)][0] += round(o["pc_success"] * n / 100)
        acc[(kind, level)][1] += n
    return acc


def main():
    out = sys.argv[1]
    runs = [a.split("=", 1) for a in sys.argv[2:]]
    data = {lab: load(Path(p)) for lab, p in runs}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (kind, title) in zip(axes, KINDS, strict=True):
        for i, (lab, _) in enumerate(runs):
            xs, ys = [], []
            for j, lv in enumerate(LEVELS):
                key = ("clean", "clean") if lv == "clean" else (kind, lv)
                k, n = data[lab].get(key, (0, 0))
                if n:
                    xs.append(j)
                    ys.append(100 * k / n)
            if xs:
                ax.plot(xs, ys, marker="o", lw=2, color=COLORS[i % len(COLORS)], label=lab)
        ax.set_xticks(range(len(LEVELS)), ["clean", "L1", "L2", "L3"])
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("success rate (%)")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
