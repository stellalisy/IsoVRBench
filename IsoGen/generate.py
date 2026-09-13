#!/usr/bin/env python3
"""Turn prompts, or videos you already have, into labelled preference pairs.

    python generate.py --prompts prompts.txt --out pairs/

One prompt in, up to eight pairs out. Each pair is the anchor against one
degraded counterpart, and the label follows from what was changed.

mq_hold and shuffle need no model. If you already have videos, point at them
and run on CPU with nothing downloaded:

    python generate.py --anchors my_videos/ --out pairs/ --families mq_hold,shuffle

--anchors takes any directory of mp4 files, from any generator. That is the
point: the procedure is not tied to the models listed below. Families that
re-render need the prompt too, so pass --prompts alongside for those.

Rerunning skips anything already on disk.
"""
import argparse
import glob
import json
import os
import shutil
import sys

# run from anywhere, not only from inside this directory
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import yaml

import config as C
import interventions as IV


# family -> (models it needs, the dimension its label is on)
FAMILIES = {
    "mq_hold":   ((), "MQ"),
    "shuffle":   ((), "MQ"),
    "vq_steps":  (("anchor",), "VQ"),
    "ta_attrib": (("anchor",), "TA"),
    "size_1p3b": (("weaker_t2v",), "overall"),
    "size_5b":   (("weaker_t2v_2",), "overall"),
    "size_i2v":  (("i2v",), "overall"),
    "mq_gen":    (("i2v_strong",), "MQ"),
}

_LOADED = {}


def pipeline(key, cfg, root=None, i2v=False):
    """Load a generator, keeping at most one on the card.

    `i2v` picks the pipeline class. TI2V-5B is used both ways, from the prompt
    alone for size_5b and from the anchor's first frame for size_i2v, so the
    family decides rather than the checkpoint.

    The anchor model is 14B. Holding it while loading a second fills an 80 GB
    card, and everything after that fails on a few megabytes. Swapping costs a
    reload, so at scale shard by family rather than by prompt and each worker
    loads once. See the scaling notes in the README.
    """
    slot = key + (":i2v" if i2v else "")
    if slot in _LOADED:
        return _LOADED[slot]
    if key not in cfg.models:
        sys.exit("no model configured for %r; configs name %s"
                 % (key, ", ".join(sorted(cfg.models))))
    path = os.path.join(root, key) if root else cfg.models[key]
    import gc
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline
    for k in list(_LOADED):
        _LOADED[k].to("cpu")
        del _LOADED[k]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae",
                                           torch_dtype=torch.float32)
    cls = WanImageToVideoPipeline if i2v else WanPipeline
    _LOADED[slot] = cls.from_pretrained(path, vae=vae,
                                        torch_dtype=torch.bfloat16).to("cuda")
    print("loaded %s" % slot, flush=True)
    return _LOADED[slot]


def read_frames(path):
    """Frames of an mp4, as PIL images."""
    import imageio.v2 as iio
    import numpy as np
    from PIL import Image
    r = iio.get_reader(path, "ffmpeg")
    try:
        return [Image.fromarray(np.asarray(f)) for f in r]
    finally:
        r.close()


def write(frames, path, fps):
    """Write to a temp name and rename, so an interrupted run never leaves a
    partial clip that later looks finished."""
    import imageio.v2 as iio
    import numpy as np
    tmp = "%s.%d.partial.mp4" % (path[:-4], os.getpid())
    w = iio.get_writer(tmp, "ffmpeg", fps=fps, codec="libx264", quality=8,
                       macro_block_size=1)
    try:
        for f in frames:
            w.append_data(np.asarray(f))
    finally:
        w.close()
    if os.path.getsize(tmp) < 1024:
        os.remove(tmp)
        raise RuntimeError("wrote an empty clip")
    os.replace(tmp, path)


def make(prompt, pid, families, outdir, triggers, cfg, root, anchor_src=None):
    d = os.path.join(outdir, pid)
    os.makedirs(d, exist_ok=True)
    anchor = os.path.join(d, "anchor.mp4")

    if not os.path.exists(anchor):
        if anchor_src:
            shutil.copy2(anchor_src, anchor)
        elif prompt is None:
            raise RuntimeError("no anchor supplied and no prompt to render one")
        else:
            t2v = pipeline("anchor", cfg, root)
            f = t2v(prompt=prompt, height=cfg.height, width=cfg.width,
                    num_frames=cfg.num_frames, num_inference_steps=cfg.steps,
                    guidance_scale=cfg.guidance,
                    generator=IV.seeded_generator(cfg)).frames[0]
            write(f, anchor, cfg.fps)
            # this frame holds the pipeline for the rest of the call, and a 14B
            # model left on the card means every later family fails to load
            del t2v

    frames = read_frames(anchor)

    pairs = []
    for fam in families:
        needs_prompt = bool(FAMILIES[fam][0])
        if needs_prompt and prompt is None:
            continue
        out = os.path.join(d, fam + ".mp4")
        rec = {"id": "%s-%s" % (pid, fam), "intervention": fam,
               "prompt": prompt, "prompt_id": pid,
               "chosen": os.path.relpath(anchor, outdir),
               "rejected": os.path.relpath(out, outdir),
               "dimension": FAMILIES[fam][1]}
        if os.path.exists(out):
            pairs.append(rec)
            continue
        try:
            if fam == "mq_hold":
                v = IV.mq_hold(frames, pid, cfg)
            elif fam == "shuffle":
                m = IV.motion_score(frames)
                if m < cfg.shuffle_motion_min:
                    continue          # reordering would not be visible
                rec["motion"] = round(m, 2)
                v = IV.shuffle(frames, pid, cfg)
            elif fam == "vq_steps":
                v = IV.vq_steps(pipeline("anchor", cfg, root), prompt, cfg)
                rec["steps"] = cfg.vq_steps
            elif fam == "ta_attrib":
                v, edited, meta = IV.ta_attrib(pipeline("anchor", cfg, root), prompt,
                                               triggers, cfg)
                if v is None:
                    continue          # no trigger phrase in this prompt
                rec.update(meta)
                rec["rejected_prompt"] = edited
            elif fam in ("size_1p3b", "size_5b"):
                key = "weaker_t2v" if fam == "size_1p3b" else "weaker_t2v_2"
                v = IV.size_swap(pipeline(key, cfg, root), prompt, cfg)
            elif fam == "size_i2v":
                v = IV.size_i2v(pipeline("i2v", cfg, root, i2v=True), prompt,
                                frames, cfg)
            elif fam == "mq_gen":
                v = IV.mq_gen(pipeline("i2v_strong", cfg, root, i2v=True), prompt,
                              frames, pid, cfg)
            write(v, out, cfg.fps)
        except Exception as e:
            print("  %s %s failed: %s: %s" % (pid, fam, type(e).__name__, e),
                  file=sys.stderr, flush=True)
            continue
        pairs.append(rec)
    return pairs


def load_rows(a):
    """(pid, prompt, anchor_path) for each item to process."""
    prompts = {}
    if a.prompts:
        for i, line in enumerate(open(a.prompts)):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                r = json.loads(line)
                prompts[r.get("pid") or "p%06d" % i] = r["prompt"]
            else:
                prompts["p%06d" % i] = line

    if a.anchors:
        rows = []
        for f in sorted(glob.glob(os.path.join(a.anchors, "*.mp4"))):
            pid = os.path.splitext(os.path.basename(f))[0]
            rows.append((pid, prompts.get(pid), f))
        return rows
    return [(pid, p, None) for pid, p in prompts.items()]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts",
                    help="one prompt per line, or jsonl with prompt and pid")
    ap.add_argument("--anchors",
                    help="directory of existing mp4 files to use as anchors "
                         "instead of rendering them; the file stem is the id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", default="all",
                    help="comma separated, or all, or none to render only the "
                         "anchors")
    ap.add_argument("--config", default="benchmark",
                    help="a preset name (%s) or a path to a yaml file of your "
                         "own" % ", ".join(C.PRESETS))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config field, e.g. --set vq_steps=4")
    ap.add_argument("--triggers", default=os.path.join(HERE, "triggers.json"))
    ap.add_argument("--models-root",
                    help="directory holding the generators; "
                         "otherwise they come from the hub")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()

    if not a.prompts and not a.anchors:
        sys.exit("give --prompts, or --anchors, or both")

    cfg = C.load(a.config, a.set)
    print(yaml.safe_dump(vars(cfg), sort_keys=False))

    if a.families == "all":
        families = list(FAMILIES)
    elif a.families in ("", "none"):
        # anchors only, which is the first pass when scaling up: render every
        # anchor once, then run each family against them with --anchors
        families = []
    else:
        families = a.families.split(",")
    bad = [f for f in families if f not in FAMILIES]
    if bad:
        sys.exit("unknown families: %s" % ", ".join(bad))

    triggers = []
    if "ta_attrib" in families:
        if not os.path.exists(a.triggers):
            sys.exit("ta_attrib needs %s" % a.triggers)
        triggers = json.load(open(a.triggers))["entries"]

    rows = load_rows(a)[a.shard::a.num_shards]
    print("%d items, families: %s" % (len(rows), ", ".join(families)))

    os.makedirs(a.out, exist_ok=True)
    out = open(os.path.join(a.out, "pairs.%03d.jsonl" % a.shard), "a")
    made = 0
    for n, (pid, prompt, src) in enumerate(rows, 1):
        try:
            for rec in make(prompt, pid, families, a.out, triggers, cfg,
                            a.models_root, src):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                made += 1
        except Exception as e:
            print("  %s failed: %s: %s" % (pid, type(e).__name__, e),
                  file=sys.stderr, flush=True)
        out.flush()
        if n % 25 == 0:
            print("  %d/%d items, %d pairs" % (n, len(rows), made), flush=True)
    print("done: %d items, %d pairs" % (len(rows), made))


if __name__ == "__main__":
    main()
