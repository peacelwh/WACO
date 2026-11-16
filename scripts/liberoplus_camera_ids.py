"""Print, per suite, the LIBERO-Plus task indices whose perturbation category is the camera.

LIBERO-Plus ships task_classification.json with one entry per task variant:
  {"id": <1-based position in the suite>, "name": <bddl stem>, "category": ..., "difficulty_level": ...}
We verify the id convention against the suite's own task list before trusting it.
Output: JSON {suite: [0-based task ids]} written to argv[1] (LIBERO-Plus prints to stdout).
"""

import json
import os
import sys

from libero.libero import benchmark

root = os.path.dirname(benchmark.__file__)
cls = json.load(open(os.path.join(root, "task_classification.json")))
out = {}
for suite_name, entries in cls.items():
    su = benchmark.get_benchmark_dict()[suite_name]()
    names = [su.get_task(i).name for i in range(len(su.tasks))]
    pos = {n: i for i, n in enumerate(names)}
    ids, mismatches = [], 0
    for e in entries:
        if e["category"] != "Camera Viewpoints":
            continue
        i = pos.get(e["name"])
        if i is None:
            mismatches += 1
            continue
        if i != e["id"] - 1:
            mismatches += 1
        ids.append(i)
    print(f"{suite_name}: {len(names)} tasks, camera {len(ids)}, id-convention mismatches {mismatches}", file=sys.stderr)
    out[suite_name] = sorted(ids)
json.dump(out, open(sys.argv[1], "w"))
print("wrote", sys.argv[1], file=sys.stderr)
