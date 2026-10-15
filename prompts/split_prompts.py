#!/usr/bin/env python3
"""Split generated pairs into test, validation and train by PROMPT ID.

Why prompt id and not pair: several interventions reuse one anchor clip, so
excluding a pair leaves its anchor in training under the other pairs built on
it. Measured on a pair-level split, 99.8% of some families' benchmark pairs had
both clips seen in training. Splitting on prompt id removes the clip itself.

Test. A prompt id is eligible for test when sha256("gen2holdout:" + pid) mod
100 falls below --holdout-pct (20). A pair enters the test pool only when every
prompt id it references is eligible, and the test split takes
--test-per-intervention pairs of each intervention from that pool. A test file
built some other way can be passed with --test instead; in the release, the
test pairs of mq_gen, mq_hold, size_1p3b, size_5b and size_i2v came from this
rule, while shuffle, vq_steps and ta_attrib were built separately (high-motion
anchors, 8 denoising steps, single-attribute edits) and joined to them.

Train and validation. Every pair referencing a prompt id used by the test split
is dropped, so no test clip or prompt reaches training. Prompt ids are then
reserved for validation before anything is cut, so no anchor is shared between
validation and training; otherwise validation measures new pairings of scenes
the model already trained on. Prompt ids are taken in shuffled order until
every intervention has --val-per-intervention pairs, and the rest supply
--train-per-intervention pairs each, so no intervention is weighted by how much
of it happened to render.

    python split_prompts.py --pairs pairs.jsonl --out data/
    python split_prompts.py --pairs pairs.jsonl --test test.jsonl --out data/

Input is one pair per line with `prompt_id`, `intervention`, and `video_a` /
`video_b` objects holding a `path`, as in the released data/*.jsonl. A path
segment of the form p123456/ or p123456_ also counts as a prompt id the pair
references.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re

PID = re.compile(r"(?:^|/)(p\d+)[/_]")


def pids_of(row):
    out = {row["prompt_id"]} if row.get("prompt_id") else set()
    for side in ("video_a", "video_b"):
        m = PID.search((row.get(side) or {}).get("path") or "")
        if m:
            out.add(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--test", help="an existing test split; skips the hash rule")
    ap.add_argument("--out", default="data")
    ap.add_argument("--holdout-pct", type=int, default=20)
    ap.add_argument("--test-per-intervention", type=int, default=1000)
    ap.add_argument("--val-per-intervention", type=int, default=500)
    ap.add_argument("--train-per-intervention", type=int, default=20000)
    ap.add_argument("--test-seed", type=int, default=20260902)
    ap.add_argument("--seed", type=int, default=20260910)
    a = ap.parse_args()

    def held(pid):
        h = int(hashlib.sha256(f"gen2holdout:{pid}".encode()).hexdigest(), 16)
        return (h % 100) < a.holdout_pct

    rows = [json.loads(line) for line in open(a.pairs)]
    rows = [r for r in rows if pids_of(r)]
    split = {"test": [], "validation": [], "train": []}

    # ---- test ----
    if a.test:
        split["test"] = [json.loads(line) for line in open(a.test)]
    else:
        pool = collections.defaultdict(list)
        for r in rows:
            if all(held(p) for p in pids_of(r)):
                pool[r["intervention"]].append(r)
        rng = random.Random(a.test_seed)
        for iv in sorted(pool):
            cand = sorted(pool[iv], key=lambda r: json.dumps(r, sort_keys=True))
            if len(cand) < a.test_per_intervention:
                print(f"  {iv}: only {len(cand)} test pairs")
            take = cand if len(cand) <= a.test_per_intervention \
                else rng.sample(cand, a.test_per_intervention)
            split["test"].extend(take)
    test_pids = {p for r in split["test"] for p in pids_of(r)}
    rest = [r for r in rows if not (pids_of(r) & test_pids)]
    print(f"test {len(split['test'])} pairs on {len(test_pids)} prompt ids; "
          f"{len(rest)} of {len(rows)} pairs remain for train and validation")

    # ---- validation, reserved by prompt id ----
    rng = random.Random(a.seed)
    interventions = sorted({r["intervention"] for r in rest})
    bypid = collections.defaultdict(list)
    for r in rest:
        bypid[r["prompt_id"]].append(r)
    pids = sorted(bypid)
    rng.shuffle(pids)
    val_pids, got = set(), collections.Counter()
    for pid in pids:
        ivs = {r["intervention"] for r in bypid[pid]}
        if all(got[iv] >= a.val_per_intervention for iv in ivs):
            continue
        val_pids.add(pid)
        for r in bypid[pid]:
            got[r["intervention"]] += 1
        if all(got[iv] >= a.val_per_intervention for iv in interventions):
            break
    missing = [iv for iv in interventions if got[iv] < a.val_per_intervention]
    if missing:
        raise SystemExit("cannot fill validation for %s" % ", ".join(missing))
    print("reserved %d prompts for validation" % len(val_pids))

    # ---- fixed counts per intervention ----
    short = []
    for iv in interventions:
        v = [r for p in sorted(val_pids) for r in bypid[p] if r["intervention"] == iv]
        t = [r for p in sorted(bypid) if p not in val_pids
             for r in bypid[p] if r["intervention"] == iv]
        rng.shuffle(v)
        rng.shuffle(t)
        if len(t) < a.train_per_intervention:
            short.append("%s has %d for training, needs %d"
                         % (iv, len(t), a.train_per_intervention))
        split["validation"].extend(v[:a.val_per_intervention])
        split["train"].extend(t[:a.train_per_intervention])
    if short:
        raise SystemExit("not enough pairs: " + "; ".join(short))

    # refuse to write splits that share a prompt id
    seen = {k: {p for r in rs for p in pids_of(r)} for k, rs in split.items()}
    for x, y in (("test", "validation"), ("test", "train"), ("validation", "train")):
        if seen[x] & seen[y]:
            raise SystemExit("%d prompt ids in both %s and %s" % (len(seen[x] & seen[y]), x, y))

    os.makedirs(a.out, exist_ok=True)
    for k, rs in split.items():
        with open(f"{a.out}/{k}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(dict(r, split=k), ensure_ascii=False) + "\n")
        print("%-10s %7d pairs  %6d prompt ids -> %s/%s.jsonl"
              % (k, len(rs), len(seen[k]), a.out, k))


if __name__ == "__main__":
    main()
