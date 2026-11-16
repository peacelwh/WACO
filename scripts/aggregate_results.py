"""Collect lerobot-eval outputs from an eval_grid run into one table.

Writes <run_dir>/summary.json and <run_dir>/summary.md with per-suite success rates for
every (kind, level) cell, the retention ratio relative to clean, and Wilson 95% intervals.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def read_cell(d: Path):
    f = d / "eval_info.json"
    if not f.exists():
        return None
    info = json.loads(f.read_text())
    # lerobot-eval writes overall + per-task aggregates; count successes from episodes when present
    eps = info.get("per_episode") or []
    if eps:
        n = len(eps)
        k = sum(1 for e in eps if e.get("success"))
        return k, n
    agg = info.get("overall") or info.get("aggregated") or {}
    pc = agg.get("pc_success")
    n = agg.get("n_episodes") or info.get("n_episodes")
    if pc is None or n is None:
        return None
    return round(pc * n / 100), int(n)


def main(run_dir: str) -> None:
    root = Path(run_dir)
    table: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        parts = d.name.split("__")
        if len(parts) != 4:
            continue
        suite, kind, level, _seed = parts
        r = read_cell(d)
        if r is None:
            continue
        cell = table[(kind, level)][suite]
        cell[0] += r[0]
        cell[1] += r[1]

    suites = sorted({s for v in table.values() for s in v})
    order = [("clean", "clean")] + [(k, l) for k in ("cam", "base", "both") for l in ("l1", "l2", "l3")]
    rows, out = [], {}
    clean_avg = None
    for key in order:
        if key not in table:
            continue
        per = table[key]
        k_all = sum(per[s][0] for s in suites if s in per)
        n_all = sum(per[s][1] for s in suites if s in per)
        avg = 100 * k_all / n_all if n_all else float("nan")
        if key == ("clean", "clean"):
            clean_avg = avg
        lo, hi = wilson(k_all, n_all)
        ret = avg / clean_avg if clean_avg else float("nan")
        cells = [f"{100 * per[s][0] / per[s][1]:.1f}" if s in per and per[s][1] else "-" for s in suites]
        rows.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + f" | **{avg:.1f}** [{lo:.1f}, {hi:.1f}] | {ret:.2f} | {n_all} |")
        out[f"{key[0]}/{key[1]}"] = {
            "per_suite": {s: 100 * per[s][0] / per[s][1] for s in suites if s in per and per[s][1]},
            "avg": avg, "ci95": [lo, hi], "retention": ret, "n": n_all,
        }
    header = "| kind | level | " + " | ".join(suites) + " | avg [95% CI] | retention | n |"
    sep = "|" + "---|" * (len(suites) + 5)
    md = "\n".join([f"# {root.name}", "", header, sep, *rows, ""])
    (root / "summary.md").write_text(md)
    (root / "summary.json").write_text(json.dumps(out, indent=2))
    print(md)


if __name__ == "__main__":
    main(sys.argv[1])
