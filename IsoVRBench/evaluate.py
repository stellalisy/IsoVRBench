#!/usr/bin/env python3
"""Run a reward model on IsoVRBench.

Fetches the benchmark, scores every clip with your model, and prints accuracy
per intervention.

    python evaluate.py --model my_module:score_clip

--model names a function as `module:function`. It is called once per clip as

    score_clip(video_path, prompt) -> float

or, for a model with separate heads, returning a dict:

    score_clip(video_path, prompt) -> {"VQ": .., "MQ": .., "TA": ..}

With heads, each pair is read on the one its `scored_on` field names, which is
how a model that scores motion separately gets judged on motion.

The benchmark is private, so `huggingface-cli login` first.

Scores are cached to scores.jsonl, so a rerun after a crash resumes rather than
rescoring 20,047 clips.
"""
import argparse
import importlib
import json
import os
import sys


def load_scorer(spec):
    if ":" not in spec:
        sys.exit("--model must be module:function, e.g. my_module:score_clip")
    mod, fn = spec.split(":", 1)
    sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(mod), fn)


def fetch(repo, local):
    """Download the benchmark. Skips anything already present."""
    from huggingface_hub import snapshot_download
    print("fetching %s" % repo)
    return snapshot_download(repo_id=repo, repo_type="dataset", local_dir=local)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="module:function")
    ap.add_argument("--data", default="IsoVRBench",
                    help="where to put or find the benchmark")
    ap.add_argument("--repo", default="stellalisy/IsoVRBench")
    ap.add_argument("--scores", default="scores.jsonl")
    ap.add_argument("--subset", choices=("isolated", "conflicted", "both"),
                    default="isolated")
    ap.add_argument("--no-download", action="store_true",
                    help="use --data as is")
    a = ap.parse_args()

    root = a.data if a.no_download else fetch(a.repo, a.data)
    score_clip = load_scorer(a.model)

    clips = [json.loads(l) for l in open(os.path.join(root, "videos.jsonl"))]
    print("%d clips to score" % len(clips))

    # resume: a clip already scored is not scored again
    done = {}
    if os.path.exists(a.scores):
        for line in open(a.scores):
            r = json.loads(line)
            done[(r["path"], r.get("prompt", ""))] = True
        print("  %d already scored, resuming" % len(done))

    out = open(a.scores, "a")
    for i, c in enumerate(clips, 1):
        key = (c["path"], c.get("prompt", ""))
        if key in done:
            continue
        v = score_clip(os.path.join(root, "videos", c["path"]), c["prompt"])
        rec = {"path": c["path"], "prompt": c["prompt"]}
        rec.update(v if isinstance(v, dict) else {"score": float(v)})
        out.write(json.dumps(rec) + "\n")
        if i % 500 == 0:
            out.flush()
            print("  %d/%d" % (i, len(clips)), flush=True)
    out.close()

    # score.py does the accounting; it lives beside this file
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import score

    probe = json.loads(open(a.scores).readline())
    single = "score" in probe
    res = {"n_scored_clips": sum(1 for _ in open(a.scores))}
    if a.subset in ("isolated", "both"):
        res["isolated"] = score.isolated(
            score.read_jsonl(os.path.join(root, "isolated.jsonl")),
            score.load_scores(a.scores), not single)
    if a.subset in ("conflicted", "both"):
        res["conflicted"] = score.conflicted(
            score.read_jsonl(os.path.join(root, "conflicted.jsonl")),
            score.load_scores(a.scores), not single)
    print(json.dumps(res, indent=1))

    if "isolated" in res:
        by = res["isolated"]["by_intervention"]
        print("\n%-12s %8s %8s %7s" % ("", "ties=wrong", "decided", "ties"))
        for k in ("mq_hold", "shuffle", "mq_gen", "ta_attrib", "vq_steps",
                  "size_i2v", "size_5b", "size_1p3b"):
            v = by.get(k)
            if v:
                print("%-12s %8s %8s %7d" % (k, v["ties_wrong"], v["decided"],
                                             v["ties"]))
        print("%-12s %8s" % ("average", res["isolated"]["average_ties_wrong"]))


if __name__ == "__main__":
    main()
