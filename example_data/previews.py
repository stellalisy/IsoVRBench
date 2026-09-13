#!/usr/bin/env python3
"""Side-by-side preview GIFs, one per intervention.

A GitHub README cannot play an mp4 that lives in the repository, so the clips
are only visible to someone who clicks through. A GIF renders inline.

Left is the preferred clip, right is the variant. The point is that a reader
sees what each intervention does without downloading anything, so these are
small and lossy on purpose. The mp4s in videos/ are the real pairs.
"""
import importlib.util
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VID = os.path.join(R, "example_data", "videos")
OUT = os.path.join(R, "example_data", "previews")


def mod(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m


sys.path.insert(0, os.path.join(R, "IsoGen"))
mod("config", os.path.join(R, "IsoGen", "config.py"))
mod("interventions", os.path.join(R, "IsoGen", "interventions.py"))
gen = mod("generate", os.path.join(R, "IsoGen", "generate.py"))

# intervention -> (preferred file, variant file, which prompt shows it best)
PANELS = [
    ("vq_steps", "s50.mp4", "vq_steps_k8.mp4"),
    ("size_1p3b", "s50.mp4", "size_1p3b.mp4"),
    ("size_5b", "s50.mp4", "size_5b.mp4"),
    ("size_i2v", "s50.mp4", "size_i2v.mp4"),
    ("mq_gen", "s50.mp4", "mq_gen.mp4"),
    ("mq_hold", "s50.mp4", "mq_hold.mp4"),
    ("shuffle", "s50.mp4", "shuffle.mp4"),
    ("ta_attrib", "s50.mp4", "ta_attrib.mp4"),
]

W = int(os.environ.get("PREVIEW_W", 256))      # per panel
N = int(os.environ.get("PREVIEW_N", 24))       # frames
FPS = int(os.environ.get("PREVIEW_FPS", 8))
GAP = 4
BAR = 16


def strip(path, n):
    f = gen.read_frames(path)
    step = (len(f) - 1) / max(1, n - 1)
    return [f[round(i * step)] for i in range(n)]


def panel(a, b, left, right):
    h = round(a[0].height * W / a[0].width)
    out = []
    for fa, fb in zip(a, b):
        ca = fa.resize((W, h), Image.LANCZOS)
        cb = fb.resize((W, h), Image.LANCZOS)
        im = Image.new("RGB", (W * 2 + GAP, h + BAR), (255, 255, 255))
        im.paste(ca, (0, BAR))
        im.paste(cb, (W + GAP, BAR))
        d = ImageDraw.Draw(im)
        d.text((2, 3), left, fill=(0, 0, 0))
        d.text((W + GAP + 2, 3), right, fill=(0, 0, 0))
        out.append(im)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    pid = sys.argv[1] if len(sys.argv) > 1 else "p003441"
    import imageio.v2 as iio
    total = 0
    for name, keep, rej in PANELS:
        kp, rp = os.path.join(VID, pid, keep), os.path.join(VID, pid, rej)
        if not (os.path.exists(kp) and os.path.exists(rp)):
            print("  skip %s" % name)
            continue
        frames = panel(strip(kp, N), strip(rp, N), "preferred", name)
        # a shared 128 colour palette, no dithering: dithering costs more than
        # it buys here and speckles the flat regions that vq_steps is about
        frames = [f.quantize(colors=128, dither=Image.NONE).convert("RGB")
                  for f in frames]
        dst = os.path.join(OUT, name + ".gif")
        iio.mimsave(dst, [np.asarray(f) for f in frames],
                    duration=1000 / FPS, loop=0)
        kb = os.path.getsize(dst) // 1024
        total += kb
        print("  %-12s %5d KB  %dx%d" % (name, kb, frames[0].width,
                                         frames[0].height))
    print("total %d KB at W=%d N=%d fps=%d" % (total, W, N, FPS))


if __name__ == "__main__":
    main()
