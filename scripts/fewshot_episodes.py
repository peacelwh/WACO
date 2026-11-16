"""First K demonstrations of every task in a LeRobot dataset, as a list for --dataset.episodes.

  python fewshot_episodes.py DATASET_ROOT K  ->  prints "[0,1,...]" (JSON, no spaces)
Episodes are taken in index order within each task, so the K=10 set is a subset of K=20.
"""

import glob
import json
import sys
from collections import defaultdict

import pandas as pd

root, k = sys.argv[1], int(sys.argv[2])
df = pd.concat(pd.read_parquet(f, columns=["episode_index", "task_index"])
               for f in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True)))
first = df.groupby("episode_index")["task_index"].first().sort_index()
by_task = defaultdict(list)
for ep, task in first.items():
    by_task[int(task)].append(int(ep))
eps = sorted(e for v in by_task.values() for e in v[:k])
print(f"{len(by_task)} tasks, {len(eps)} episodes (of {len(first)})", file=sys.stderr)
print(json.dumps(eps).replace(" ", ""))
