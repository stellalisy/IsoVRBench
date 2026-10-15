#!/usr/bin/env python3
"""The eight interventions.

Each takes the anchor, or the prompt that produced it, plus a config, and
returns the frames of the degraded clip. Nothing here writes files.
"""
import random

import numpy as np


def seeded_generator(cfg, device="cuda"):
    # Imported here rather than at the top so the families that only rearrange
    # the anchor's frames run without torch installed.
    import torch
    return torch.Generator(device=device).manual_seed(cfg.seed)


def motion_score(frames):
    """Mean absolute difference between every fourth frame. What shuffle
    filters on, measured on the anchor."""
    a = np.asarray([np.asarray(f) for f in frames[::4]], dtype=np.int16)
    return float(np.abs(np.diff(a, axis=0)).mean())


# --- no model needed, the anchor's own frames rearranged -------------------

def mq_hold(frames, key, cfg):
    """Hold each of a few real frames until the next is due, so the clip covers
    the same movement in visible jumps. Every pixel is the anchor's own, so
    only motion changes."""
    n = len(frames)
    k = random.Random(str(key) + "k").choice(cfg.mq_hold_keyframes)
    rnd = random.Random(str(key) + "hold")
    keys = sorted(set([0] + rnd.sample(range(1, n), min(k - 1, n - 1))))
    out, j = [], 0
    for t in range(n):
        while j + 1 < len(keys) and keys[j + 1] <= t:
            j += 1
        out.append(frames[keys[j]])
    return out


def shuffle(frames, key, cfg):
    """Cut into contiguous segments and permute them so none stays in place.
    The frames themselves are untouched, so only their order changes."""
    n, s = len(frames), cfg.shuffle_segments
    if n < s * 2:
        raise ValueError("clip too short for %d segments" % s)
    chunks = [c for c in (frames[i * n // s:(i + 1) * n // s] for i in range(s)) if c]
    rnd = random.Random(str(key) + "shuf")
    idx = list(range(len(chunks)))
    for _ in range(400):
        q = idx[:]
        rnd.shuffle(q)
        if all(a != b for a, b in zip(idx, q)):
            break
    else:
        q = idx[::-1]
    return [f for i in q for f in chunks[i]]


# --- the same model, re-rendered -------------------------------------------

def vq_steps(t2v, prompt, cfg):
    """Re-render the prompt at fewer denoising steps, same seed."""
    return t2v(prompt=prompt, height=cfg.height, width=cfg.width,
               num_frames=cfg.num_frames, num_inference_steps=cfg.vq_steps,
               guidance_scale=cfg.guidance, generator=seeded_generator(cfg)).frames[0]


def ta_attrib(t2v, prompt, triggers, cfg):
    """Edit attributes in the prompt, render that, and pair it against the
    anchor under the original prompt.

    `triggers` is an attribute_edit.TriggerList. The edit is a lookup in the
    frozen trigger list, seeded by the prompt, so it is deterministic. A prompt
    that cannot take ta_attrib_k disjoint swaps is skipped. Returns frames, the
    edited prompt, and what was changed.
    """
    edited, swaps = triggers.edit(prompt, cfg.ta_attrib_k, cfg.seed)
    if len(swaps) < cfg.ta_attrib_k:
        return None, None, None
    frames = t2v(prompt=edited, height=cfg.height, width=cfg.width,
                 num_frames=cfg.num_frames, num_inference_steps=cfg.steps,
                 guidance_scale=cfg.guidance, generator=seeded_generator(cfg)).frames[0]
    meta = {"k": len(swaps),
            "trigger": [s["trigger"] for s in swaps],
            "target": [s["target"] for s in swaps],
            "swap_dimension": [s["dimension"] for s in swaps]}
    return frames, edited, meta


# --- another text to video model, same prompt ------------------------------

def size_swap(weaker_t2v, prompt, cfg):
    """Render the same prompt with a weaker generator. This moves visual
    quality, motion and prompt adherence together, so the pair carries an
    overall label rather than a per dimension one."""
    return weaker_t2v(prompt=prompt, height=cfg.height, width=cfg.width,
                      num_frames=cfg.num_frames, num_inference_steps=cfg.steps,
                      guidance_scale=cfg.guidance, generator=seeded_generator(cfg)).frames[0]


# --- an image to video model, conditioned on the anchor --------------------

def size_i2v(i2v_5b, prompt, frames, cfg):
    """A weaker generator continuing the anchor's first frame. Varies the
    generator and the conditioning together."""
    first = frames[0].convert("RGB").resize((cfg.width, cfg.height))
    return i2v_5b(image=first, prompt=prompt, height=cfg.height, width=cfg.width,
                  num_frames=len(frames), num_inference_steps=cfg.steps,
                  guidance_scale=cfg.guidance, generator=seeded_generator(cfg)).frames[0]


def mq_gen(i2v, prompt, frames, key, cfg):
    """Keep real keyframes, regenerate the gaps between them, and snap the last
    frame of each gap back to the real keyframe.

    The generated frames drift from the path the anchor took and are corrected
    at each keyframe, so movement is unsteady while the content stays on
    course. Random spacing varies how far the drift runs.
    """
    n = len(frames)
    k = random.Random(str(key) + "k").choice(cfg.mq_gen_keyframes)
    rnd = random.Random(str(key) + "gen")
    keys = sorted(set([0] + rnd.sample(range(1, n - 1), min(k - 2, n - 2)) + [n - 1]))

    out = [frames[0]]
    for a, b in zip(keys, keys[1:]):
        span = b - a
        if span <= 1:
            out.append(frames[b])
            continue
        seg = i2v(image=frames[a].convert("RGB").resize((cfg.width, cfg.height)),
                  prompt=prompt, height=cfg.height, width=cfg.width,
                  num_frames=span + 1, num_inference_steps=cfg.mq_gen_steps,
                  guidance_scale=cfg.guidance, generator=seeded_generator(cfg)).frames[0]
        out.extend(seg[1:span])
        out.append(frames[b])
    return out[:n]
