#!/usr/bin/env python3
"""Select the prompts to generate from, out of the pool prompt_pipeline.py builds.

Three things this has to get right:

CONTAMINATION. Prompts that were read to curate the trigger list, or that are
held out for any other reason, must never become training data. Every file
passed to --exclude is excluded in full, so the rule is "this file is not
training data" rather than a list of names to maintain. An earlier version
excluded only the coverage split and left 487 selected prompts inside the
splits the trigger list was read from.

TRIGGER FIRST. 86.3% of pool prompts carried at least one trigger under the
list the release used, so requiring one costs almost nothing and lets ta_attrib draw
from the same anchors as every other intervention. A prompt with no trigger
cannot serve ta_attrib.

DIMENSION BALANCE. The raw hit distribution is lopsided: location-outdoor
appears about twice as often as the next dimension and roughly 30x the tail.
Selecting on availability alone would give ta_attrib a corpus that is mostly
location swaps. Each prompt is assigned its RAREST available dimension as its
primary, then primaries are filled round-robin, so scarce dimensions are taken
whenever they appear and abundant ones fill the remainder.

    python select_prompts.py --pool pool/pool_v2.0.jsonl --n 75000 \
        --exclude splits/*.jsonl --out prompts.jsonl

Prompt ids are assigned after sorting by prompt text, as p000000, p000001, ...
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "IsoGen"))
import interventions as IV                                            # noqa: E402


def excluded_prompts(paths):
    """Every prompt in the given files. jsonl rows give "prompt" (or
    "prompt_original"); any other file is one prompt per line."""
    bad = set()
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            if p.endswith(".jsonl"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                v = r.get("prompt") or r.get("prompt_original") or ""
            else:
                v = line.split("\t")[-1]
            if v.strip():
                bad.add(v.strip())
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="pool_v2.0.jsonl from prompt_pipeline.py")
    ap.add_argument("--triggers", default=os.path.join(HERE, "..", "IsoGen", "triggers.json"))
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="files whose prompts must not be selected")
    ap.add_argument("--n", type=int, default=75000)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--scan", type=int, default=400000,
                    help="pool lines to screen; needs headroom over --n for balancing")
    ap.add_argument("--out", default="prompts.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    triggers = json.load(open(a.triggers))["entries"]

    bad = excluded_prompts(a.exclude)
    print(f"exclusions: {len(bad)} distinct prompts from {len(a.exclude)} files")

    cands, seen = [], set()
    skipped = collections.Counter()
    with open(a.pool) as f:
        for i, l in enumerate(f):
            if i >= a.scan:
                break
            try:
                pr = json.loads(l)["prompt"].strip()
            except (ValueError, KeyError):
                skipped["unparsable"] += 1
                continue
            if not (8 <= len(pr.split()) <= 60):
                skipped["length"] += 1
                continue
            if pr in bad:
                skipped["excluded"] += 1
                continue
            if pr in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(pr)
            low = pr.lower()
            hits = [e for e in triggers if e.get("targets") and e["trigger"].lower() in low]
            if not hits:
                skipped["no trigger"] += 1
                continue
            # k_max is what ta_attrib can actually apply together, one trigger
            # per dimension, which is fewer than the raw hit count
            k_max = len(IV.find_triggers(pr, triggers, 3))
            cands.append({"prompt": pr,
                          "dims": sorted({e["dimension"] for e in hits}),
                          "n_swaps": len(hits), "k_max": k_max})
    print(f"screened {a.scan} lines -> {len(cands)} usable candidates")
    for k, v in skipped.most_common():
        print(f"    skipped {k}: {v}")

    # rarest available dimension becomes the prompt's primary
    freq = collections.Counter(d for c in cands for d in c["dims"])
    for c in cands:
        c["primary"] = min(c["dims"], key=lambda d: (freq[d], d))
    groups = collections.defaultdict(list)
    for c in cands:
        groups[c["primary"]].append(c)
    for g in groups.values():
        rng.shuffle(g)

    # round-robin over primaries so scarce dimensions are exhausted before abundant ones
    order = sorted(groups, key=lambda d: len(groups[d]))
    chosen, idx = [], {d: 0 for d in order}
    while len(chosen) < a.n:
        progressed = False
        for d in order:
            if len(chosen) >= a.n:
                break
            if idx[d] < len(groups[d]):
                chosen.append(groups[d][idx[d]])
                idx[d] += 1
                progressed = True
        if not progressed:
            break
    print(f"\nselected {len(chosen)} prompts across "
          f"{len({c['primary'] for c in chosen})} primary dimensions")
    if len(chosen) < a.n:
        print(f"  SHORT by {a.n - len(chosen)} -- raise --scan")

    chosen.sort(key=lambda c: c["prompt"])
    for n, c in enumerate(chosen):
        c["pid"] = "p%06d" % n

    with open(a.out, "w") as f:
        for c in chosen:
            f.write(json.dumps({k: c[k] for k in
                                ("pid", "prompt", "primary", "dims", "n_swaps", "k_max")},
                               ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    kk = collections.Counter(min(c["n_swaps"], 3) for c in chosen)
    print(f"\nswaps available: 1={kk[1]} 2={kk[2]} 3+={kk[3]}")
    print(f"wrote {a.out}\n  sha256 {sha}")


if __name__ == "__main__":
    main()
