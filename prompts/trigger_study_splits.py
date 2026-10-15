#!/usr/bin/env python3
"""Fixed, disjoint splits of the pool for curating the trigger list.

  calib   1000 prompts, read by hand in 40 batches of 25, to discover triggers
  val_a    200 prompts, held out -- validation round 1 (patching allowed)
  val_b    200 prompts, held out -- validation round 2 (confirmation, never patched on)
  eval   20000 prompts, held out -- used ONLY to score coverage after each batch

Coverage is scored on `eval`, not on the 25-prompt batch: 25 prompts gives a standard
error of roughly +-10pp on a coverage estimate, which would swamp the per-batch signal
we are trying to read. `eval` is fixed across all batches so the curve is comparable.

All splits are disjoint and drawn uniformly at random from pool v2.0 with a fixed seed.
None of them is training data, so pass every file written here to
select_prompts.py --exclude.

    python trigger_study_splits.py --pool pool/pool_v2.0.jsonl --out splits/
"""
import argparse
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True, help="pool_v2.0.jsonl from prompt_pipeline.py")
ap.add_argument("--out", default="splits")
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.pool)]
print("pool: %d" % len(rows))

rng = random.Random(20260815)
idx = list(range(len(rows)))
rng.shuffle(idx)

spans = {"calib": 1000, "val_a": 200, "val_b": 200, "eval": 20000}
os.makedirs(a.out, exist_ok=True)
pos = 0
for name, n in spans.items():
    take = idx[pos:pos + n]
    pos += n
    with open(f"{a.out}/{name}.jsonl", "w") as f:
        for j, i in enumerate(take):
            r = dict(rows[i])
            r["split"] = name
            r["n"] = j
            if name == "calib":
                r["batch"] = j // 25
            f.write(json.dumps(r) + "\n")
    print("%-8s %6d  -> %s/%s.jsonl" % (name, n, a.out, name))

# plain-text copy of calib for hand reading, one prompt per line, index-prefixed
with open(f"{a.out}/calib.txt", "w") as f:
    for l in open(f"{a.out}/calib.jsonl"):
        r = json.loads(l)
        f.write("%04d\t%s\n" % (r["n"], r["prompt"].replace("\n", " ").replace("\t", " ")))
print("wrote %s/calib.txt" % a.out)
