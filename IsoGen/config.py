#!/usr/bin/env python3
"""Load a generation config from YAML.

Settings live in configs/*.yaml, one file per preset. This module reads one,
checks it, and hands back an object the interventions can read fields off.
"""
import os
import sys
from types import SimpleNamespace

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# YAML leaves a bare 2e-6 as a string and nobody should have to remember that,
# so every field is coerced on load. This doubles as the list of settings that
# exist: anything else in a config file is a typo and is reported as one.
SCHEMA = {
    # the anchor video, shared by every family
    "width": int, "height": int, "num_frames": int, "fps": int,
    "guidance": float, "steps": int, "seed": int,
    # per family knobs
    "vq_steps": int,
    "ta_attrib_k": int,
    "mq_hold_keyframes": list,
    "mq_gen_keyframes": list, "mq_gen_steps": int,
    "shuffle_segments": int, "shuffle_motion_min": float,
    "models": dict,
}

PRESETS = ("benchmark", "training")

# benchmark.yaml carries every field with its explanation. Any other config
# lists only what it changes, so a preset reads as the difference it makes.
BASE = os.path.join(HERE, "configs", "benchmark.yaml")


def resolve(name_or_path):
    """A preset name, or a path to a config file of your own."""
    if os.path.sep in name_or_path or name_or_path.endswith((".yaml", ".yml")):
        return name_or_path
    return os.path.join(HERE, "configs", name_or_path + ".yaml")


def read(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load(name_or_path="benchmark", overrides=()):
    path = resolve(name_or_path)
    if not os.path.exists(path):
        sys.exit("no config at %s (presets: %s)" % (path, ", ".join(PRESETS)))
    raw = read(BASE)
    if os.path.abspath(path) != os.path.abspath(BASE):
        raw.update(read(path))
    for k, v in (kv.split("=", 1) for kv in overrides):
        raw[k.strip()] = v

    unknown = set(raw) - set(SCHEMA)
    if unknown:
        sys.exit("unknown setting(s): %s" % ", ".join(sorted(unknown)))
    missing = set(SCHEMA) - set(raw)
    if missing:
        sys.exit("missing setting(s): %s" % ", ".join(sorted(missing)))

    cfg = {}
    for k, want in SCHEMA.items():
        v = raw[k]
        try:
            if want is list:
                cfg[k] = [int(s) for s in v.split(",")] if isinstance(v, str) \
                    else [int(x) for x in v]
            elif want is dict:
                cfg[k] = dict(v)
            else:
                cfg[k] = want(v)
        except (TypeError, ValueError):
            sys.exit("%s should be %s, got %r" % (k, want.__name__, v))
    return SimpleNamespace(**cfg)
