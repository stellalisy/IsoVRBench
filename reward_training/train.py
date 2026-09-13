#!/usr/bin/env python3
"""Train a video reward model on pairs built by IsoGen.

    python train.py --videos /path/to/clips --out ckpt/

Everything else lives in configs/*.yaml. Pick one and override any field:

    python train.py --config paper --set lora=false --set frames=16 ...

Out of the box it trains on example_data, 64 pairs over 8 prompts that ship
with the repository. Point `pairs` and `videos` at a corpus.jsonl and its clips
to train on something real.

The resolved settings are written to the checkpoint as config.yaml, so a run
can be traced back to what produced it.

A minimal, readable reference. One GPU, one process, plain PyTorch. Same recipe
as the paper's models, not the code that produced them: those ran on 32 GPUs
under DeepSpeed.

The model is a vision language backbone with a scalar head read at marker
tokens, one marker per dimension, trained with a Bradley Terry objective. A
pair carries a label only on the dimensions its intervention licenses, so the
loss covers those and the rest contribute nothing.
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

try:
    # peft reads torch.distributed.tensor.DTensor without importing the
    # submodule, which `import torch` alone does not pull in on torch 2.6.
    import torch.distributed.tensor        # noqa: F401
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config as C


def load_pairs(path, videos_root, split=""):
    """Records in `split` whose clips are both on disk and which label at least
    one dimension. A pair labelling nothing cannot contribute a gradient.

    IsoGen's own output has no split field, so an empty split takes everything.
    """
    out, missing, wrong_split = [], 0, 0
    for line in open(path):
        r = json.loads(line)
        if split and r.get("split", split) != split:
            wrong_split += 1
            continue
        pref = r.get("preferred") or {}
        if not pref:
            continue
        a = os.path.join(videos_root, r["video_a"]["path"])
        b = os.path.join(videos_root, r["video_b"]["path"])
        if not (os.path.exists(a) and os.path.exists(b)):
            missing += 1
            continue
        out.append({"a": a, "b": b, "prompt": r["prompt"], "pref": pref,
                    "intervention": r.get("intervention", "")})
    if missing:
        print("  %d pairs skipped, a clip was not under %s" % (missing, videos_root))
    if not out:
        sys.exit("no usable pairs (%d were in another split). Does `videos` "
                 "point at the clips?" % wrong_split)
    return out


def sample_frames(path, n):
    """Uniformly sample n frames. Uniform rather than the first n, because the
    frame reuse families differ from the anchor only in the middle."""
    import imageio.v2 as iio
    import numpy as np
    r = iio.get_reader(path, "ffmpeg")
    try:
        frames = [np.asarray(f) for f in r]
    finally:
        r.close()
    if not frames:
        raise RuntimeError("no frames in %s" % path)
    step = (len(frames) - 1) / max(1, n - 1)
    return [frames[round(i * step)] for i in range(n)]


class RewardModel(nn.Module):
    """A backbone plus a scalar head read at per dimension marker tokens.

    One head of width one read at several positions, rather than one head per
    dimension, so the dimensions share every parameter except where they are
    read.
    """

    def __init__(self, backbone, hidden, markers):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden, 1, bias=False)
        self.markers = markers

    def forward(self, **inputs):
        hs = self.backbone(**inputs, output_hidden_states=True).hidden_states[-1]
        ids = inputs["input_ids"]
        scores = {}
        for dim, tok in self.markers.items():
            hit = (ids == tok).nonzero()
            if not len(hit):
                continue
            per = {}
            for b, t in hit.tolist():
                per[b] = t          # last occurrence per sequence
            sel = torch.stack([hs[b, per[b]] for b in sorted(per)])
            # the backbone runs in bf16 and the head stays in fp32, since it is
            # one scalar per dimension and the precision is free
            scores[dim] = self.head(sel.to(self.head.weight.dtype)).squeeze(-1)
        return scores


def bt_loss(sa, sb, pref):
    """Bradley Terry over the dimensions this pair actually labels.

    A dimension the intervention says nothing about is absent from `pref` and
    contributes no gradient, which is what keeps `shuffle` from teaching
    anything about visual quality.
    """
    total, n = 0.0, 0
    for dim, winner in pref.items():
        if dim not in sa or dim not in sb:
            continue
        margin = sa[dim] - sb[dim] if winner == "A" else sb[dim] - sa[dim]
        total = total + F.softplus(-margin).mean()
        n += 1
    return total / max(1, n)


def video_prompt(processor, prompt):
    """Wrap the prompt so the processor knows where the video goes.

    Without a placeholder the processor produces video features that the model
    has no tokens to attach them to, and the forward pass raises rather than
    quietly ignoring the video.
    """
    try:
        return processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"},
                                          {"type": "text", "text": prompt}]}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        # no chat template, as on a base checkpoint
        return "<|vision_start|><|video_pad|><|vision_end|>" + prompt


def build(cfg):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    processor = AutoProcessor.from_pretrained(cfg.model)
    backbone = AutoModelForVision2Seq.from_pretrained(
        cfg.model, torch_dtype=dtype).to("cuda")

    marks = ["<|reward_%s|>" % d for d in cfg.dims]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": marks})
    backbone.resize_token_embeddings(len(processor.tokenizer))
    markers = {d: processor.tokenizer.convert_tokens_to_ids(m)
               for d, m in zip(cfg.dims, marks)}

    hidden = getattr(backbone.config, "hidden_size", None) or \
        backbone.config.text_config.hidden_size
    model = RewardModel(backbone, hidden, markers).to("cuda")

    if cfg.lora:
        from peft import LoraConfig, get_peft_model
        model.backbone = get_peft_model(model.backbone, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout, target_modules=cfg.lora_targets))
        model.backbone.print_trainable_parameters()
    return model, processor, marks


@torch.no_grad()
def validate(model, encode, pairs):
    """Fraction of labelled dimensions the model ranks the right way round. A
    tie counts as wrong, since every pair carries a change a person can see."""
    right = total = 0
    for p in pairs:
        try:
            sa, sb = model(**encode(p["a"], p["prompt"])), \
                model(**encode(p["b"], p["prompt"]))
        except Exception:
            continue
        for dim, winner in p["pref"].items():
            if dim not in sa or dim not in sb:
                continue
            margin = (sa[dim] - sb[dim]) if winner == "A" else (sb[dim] - sa[dim])
            right += int(margin.item() > 0)
            total += 1
    return right / total if total else float("nan")


def save(model, processor, cfg, out):
    os.makedirs(out, exist_ok=True)
    torch.save({"head": model.head.state_dict(), "markers": model.markers},
               os.path.join(out, "reward_head.pt"))
    model.backbone.save_pretrained(out)
    processor.save_pretrained(out)
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(vars(cfg), f, sort_keys=False)
    print("  saved to %s" % out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="default",
                    help="a preset name (%s) or a path to a yaml file of your "
                         "own" % ", ".join(C.PRESETS))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config field, e.g. --set frames=16")
    ap.add_argument("--videos", help="shorthand for --set videos=...")
    ap.add_argument("--pairs", help="shorthand for --set pairs=...")
    ap.add_argument("--limit", type=int, help="stop after this many pairs")
    a = ap.parse_args()

    sets = list(a.set)
    for name in ("videos", "pairs"):
        if getattr(a, name):
            sets.append("%s=%s" % (name, getattr(a, name)))
    cfg = C.load(a.config, sets)
    print(yaml.safe_dump(vars(cfg), sort_keys=False))
    if not cfg.videos:
        sys.exit("set `videos` to the directory holding the clips")

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    src = cfg.pairs
    pairs = load_pairs(src, cfg.videos, cfg.split)
    if a.limit:
        pairs = pairs[:a.limit]
    val = load_pairs(src, cfg.val_videos or cfg.videos, cfg.val_split) \
        if cfg.val_split else []
    print("%d training pairs, %d validation" % (len(pairs), len(val)))

    model, processor, marks = build(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    total = max(1, len(pairs) * cfg.epochs // (cfg.batch * cfg.accum))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=total, pct_start=cfg.warmup_ratio)

    def encode(path, prompt):
        text = video_prompt(processor, prompt) + "".join(marks)
        return processor(text=[text], videos=[sample_frames(path, cfg.frames)],
                         return_tensors="pt").to("cuda")

    def checkpoint():
        if val:
            print("  validation accuracy %.3f" % validate(model, encode, val))
        save(model, processor, cfg, a.out)

    step, used, skipped = 0, 0, 0
    for epoch in range(cfg.epochs):
        random.Random(cfg.seed + epoch).shuffle(pairs)
        for i, p in enumerate(pairs, 1):
            try:
                sa = model(**encode(p["a"], p["prompt"]))
                sb = model(**encode(p["b"], p["prompt"]))
            except Exception as e:
                skipped += 1
                # the first is usually the only one worth reading; the rest
                # repeat it once per pair
                if skipped <= 3:
                    print("  skipped %s: %s" % (p["intervention"], e))
                elif skipped == 4:
                    print("  (further skips not printed)")
                continue
            used += 1
            (bt_loss(sa, sb, p["pref"]) / cfg.accum).backward()
            if i % cfg.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % cfg.log_every == 0:
                    print("  epoch %d step %d/%d lr %.2e"
                          % (epoch, step, total, sched.get_last_lr()[0]),
                          flush=True)
                if cfg.save_every and step % cfg.save_every == 0:
                    checkpoint()

    print("%d pairs used, %d skipped" % (used, skipped))
    if not used:
        sys.exit("every pair was skipped, so nothing was trained. The message "
                 "above is why.")
    if skipped > used:
        print("WARNING: more pairs were skipped than used")
    checkpoint()


if __name__ == "__main__":
    main()
