#!/usr/bin/env python3
"""Reference scorer for IsoVRBench.

You supply one score per clip; this reproduces every number the paper reports.

    python score.py --scores my_scores.jsonl --data .

Each line of --scores is one scored clip:

    {"path": "...", "prompt": "...", "score": 0.42}

A model with per-dimension heads may give them instead of `score`, and each
pair is then read on the head named by its `scored_on` field:

    {"path": "...", "prompt": "...", "VQ": 0.1, "MQ": -0.3, "TA": 0.8}

The key is (path, prompt), not path. Every path in the released data carries
exactly one prompt, so path alone would work; the prompt is in the key because
the score depends on it.

Two conventions are reported for the isolated subset, over each
intervention's 1,000 pairs:

    ties_wrong = correct / n           a tie counts as an error
    decided    = correct / (n - ties)  ties dropped

`ties_wrong` is the paper's main table: every pair carries an intervention a
person can see, so scoring the two clips equally is a failure to perceive it.
`decided` is the convention VideoGen-RewardBench uses. The reported average is
the mean of the eight per-intervention accuracies, not correct / 8000.

The conflicted subset is 4,000 conflict pairs and 4,000 matched controls. The
controls have a correct answer and are scored as above. The conflict pairs have
none, and what is reported for them is the follow rate: of the decided pairs,
the fraction where the model's preferred side is the side that dimension
favours. Subtracting the matched control gives what the competing cue costs.
"""
import argparse
import collections
import json
import os


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_scores(p):
    out = {}
    for r in read_jsonl(p):
        key = (r["path"], r.get("prompt", ""))
        out[key] = r
    return out


def get(scores, side, head):
    r = scores.get((side["path"], side["prompt"]))
    if r is None:
        return None
    if head and head in r:
        return r[head]
    return r.get("score")


def pref(scores, rec, head_mode):
    """Which side the model prefers: 'A', 'B', or None for a tie."""
    head = rec.get("scored_on") if head_mode else None
    a = get(scores, rec["video_a"], head)
    b = get(scores, rec["video_b"], head)
    if a is None or b is None:
        return "missing"
    if a == b:
        return None
    return "A" if a > b else "B"


def isolated(recs, scores, head_mode):
    per = collections.defaultdict(lambda: collections.Counter())
    for r in recs:
        p = pref(scores, r, head_mode)
        c = per[r["intervention"]]
        if p == "missing":
            c["missing"] += 1
        elif p is None:
            c["tie"] += 1
        else:
            c["decided"] += 1
            c["correct"] += (p == r["correct"])

    rows, tw, dc = {}, [], []
    for fam, c in sorted(per.items()):
        n = c["decided"] + c["tie"]
        a1 = 100 * c["correct"] / n if n else None
        a2 = 100 * c["correct"] / c["decided"] if c["decided"] else None
        rows[fam] = {"ties_wrong": round(a1, 2) if a1 is not None else None,
                     "decided": round(a2, 2) if a2 is not None else None,
                     "n": n, "ties": c["tie"], "missing": c["missing"]}
        if a1 is not None:
            tw.append(a1)
        if a2 is not None:
            dc.append(a2)
    return {"by_intervention": rows,
            "average_ties_wrong": round(sum(tw) / len(tw), 2) if tw else None,
            "average_decided": round(sum(dc) / len(dc), 2) if dc else None}


def conflicted(recs, scores, head_mode):
    ctrl = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        if r["intervention"].startswith("ctrl_"):
            p = pref(scores, r, head_mode)
            if p in ("A", "B"):
                ctrl[r["intervention"]][1] += 1
                ctrl[r["intervention"]][0] += (p == r["correct"])

    out = {"controls": {k: round(100 * a / n, 2) if n else None
                        for k, (a, n) in sorted(ctrl.items())},
           "oppositions": {}}

    for r in recs:
        fam = r["intervention"]
        if not fam.startswith("conflict_"):
            continue
        p = pref(scores, r, head_mode)
        if p not in ("A", "B"):
            continue
        d = out["oppositions"].setdefault(
            fam, {"n": 0, "follows": collections.Counter()})
        d["n"] += 1
        for dim, side in r["favours"].items():
            if side == p:
                d["follows"][dim] += 1

    for fam, d in out["oppositions"].items():
        n = d["n"]
        d["follow_rate"] = {k: round(100 * v / n, 2) for k, v in d["follows"].items()}
        # the matched control measures the same dimension with nothing opposing
        # it, so the difference is the cost of the competing cue
        d["vs_control"] = {}
        for cname in next(x["matched_controls"] for x in recs
                          if x["intervention"] == fam):
            dim = cname.split("_")[1]
            base = out["controls"].get(cname)
            got = d["follow_rate"].get(dim)
            if base is not None and got is not None:
                d["vs_control"][cname] = round(got - base, 2)
        del d["follows"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--data", default=".")
    ap.add_argument("--single-head", action="store_true",
                    help="use one score for every pair, ignoring scored_on")
    ap.add_argument("--out")
    a = ap.parse_args()

    sc = load_scores(a.scores)
    if not sc:
        raise SystemExit("no scores read from %s" % a.scores)
    hm = not a.single_head
    res = {
        "n_scored_clips": len(sc),
        "isolated": isolated(read_jsonl(os.path.join(a.data, "isolated.jsonl")),
                             sc, hm),
        "conflicted": conflicted(read_jsonl(os.path.join(a.data, "conflicted.jsonl")),
                                 sc, hm),
    }
    # The commonest first-run mistake is scoring by path and omitting the
    # prompt, which makes every lookup miss. Say so rather than printing nulls.
    iso_n = sum(v["n"] for v in res["isolated"]["by_intervention"].values())
    if iso_n == 0:
        miss = sum(v["missing"] for v in res["isolated"]["by_intervention"].values())
        raise SystemExit(
            "matched 0 of %d isolated pairs. Scores are keyed on (path, prompt); "
            "check that each line carries the prompt from videos.jsonl." % miss)

    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)


if __name__ == "__main__":
    main()
