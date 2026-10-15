#!/usr/bin/env python3
"""The eight interventions.

Each takes the anchor, or the prompt that produced it, plus a config, and
returns the frames of the degraded clip. Nothing here writes files.
"""
import functools
import random
import re

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

    The edit is a lookup in the frozen trigger list, so it is deterministic.
    Returns frames, the edited prompt, and what was changed.
    """
    edits = find_triggers(prompt, triggers, cfg.ta_attrib_k)
    if not edits:
        return None, None, None
    edited = prompt
    for trigger, target, _ in edits:
        edited = replace_once(edited, trigger, target)
    frames = t2v(prompt=edited, height=cfg.height, width=cfg.width,
                 num_frames=cfg.num_frames, num_inference_steps=cfg.steps,
                 guidance_scale=cfg.guidance, generator=seeded_generator(cfg)).frames[0]
    meta = {"k": len(edits),
            "trigger": [t for t, _, _ in edits],
            "target": [g for _, g, _ in edits],
            "swap_dimension": [d for _, _, d in edits]}
    return frames, edited, meta


@functools.lru_cache(maxsize=None)
def _trigger_pattern(trigger):
    return re.compile(r"(?<![a-z0-9])" + re.escape(trigger.lower()) + r"(?![a-z0-9])")


def trigger_match(prompt, trigger):
    """The first whole-word occurrence of trigger in prompt, ignoring case,
    so "king" does not fire inside "attacking"."""
    return _trigger_pattern(trigger).search(prompt.lower())


def find_triggers(prompt, triggers, k):
    """Up to k trigger phrases present in the prompt as whole words, longest
    first so "12 year" wins over "year". Each dimension is used at most once,
    and entries carrying a review_flag, mostly a wrong dimension label, are
    skipped."""
    found, used = [], set()
    for e in sorted(triggers, key=lambda e: -len(e["trigger"])):
        if len(found) >= k:
            break
        d = e.get("dimension")
        if d in used or not e.get("targets") or e.get("review_flag"):
            continue
        if trigger_match(prompt, e["trigger"]):
            found.append((e["trigger"], e["targets"][0], d))
            used.add(d)
    return found


def replace_once(prompt, trigger, target):
    m = trigger_match(prompt, trigger)
    return prompt if m is None else prompt[:m.start()] + target + prompt[m.end():]


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
