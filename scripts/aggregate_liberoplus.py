"""Camera-track success on LIBERO-Plus, per suite and overall, for one or more runs.

Usage: python scripts/aggregate_liberoplus.py outputs/liberoplus TAG [TAG ...]
Each run directory contains <suite>_chunk<k>/<cell>/eval_info.json from eval_liberoplus_camera.sh.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for tag in sys.argv[2:]:
    acc = defaultdict(lambda: [0, 0])
    for f in (root / tag).glob("*_chunk*/*/eval_info.json"):
        suite = f.parent.parent.name.rsplit("_chunk", 1)[0]
        o = json.loads(f.read_text())["overall"]
        n = int(o["n_episodes"])
        acc[suite][0] += round(o["pc_success"] * n / 100)
        acc[suite][1] += n
    k = sum(v[0] for v in acc.values())
    n = sum(v[1] for v in acc.values())
    per = {s: f"{100 * v[0] / v[1]:.1f} ({v[1]})" for s, v in sorted(acc.items())}
    rows.append((tag, per, 100 * k / n if n else float("nan"), n))
suites = sorted({s for _, per, _, _ in rows for s in per})
print("| run | " + " | ".join(suites) + " | camera track (all) |")
print("|---|" + "---|" * (len(suites) + 1))
for tag, per, tot, n in rows:
    print(f"| {tag} | " + " | ".join(per.get(s, "–") for s in suites) + f" | **{tot:.1f}** ({n}) |")
