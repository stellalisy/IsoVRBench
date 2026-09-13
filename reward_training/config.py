#!/usr/bin/env python3
"""Load a training config from YAML.

Settings live in configs/*.yaml, one file per preset. This module reads one,
checks it, and hands back an object whose fields the trainer reads.
"""
import os
import sys
from types import SimpleNamespace

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# YAML leaves a bare 2e-6 as a string and nobody should have to remember that,
# so every field is coerced on load. This doubles as the list of settings that
# exist: anything else in a config file is a typo and is reported as one.
SCHEMA = {
    # where the pairs come from
    "pairs": str, "videos": str, "val_videos": str,
    "split": str, "val_split": str,
    # what to start from
    "model": str,
    # LoRA or full finetune
    "lora": bool, "lora_r": int, "lora_alpha": int, "lora_dropout": float,
    "lora_targets": list,
    # what the model sees
    "frames": int, "dims": list,
    # optimisation
    "lr": float, "epochs": int, "batch": int, "accum": int,
    "warmup_ratio": float, "grad_clip": float, "bf16": bool,
    # bookkeeping
    "seed": int, "log_every": int, "save_every": int,
}

PRESETS = ("default", "paper", "full_finetune")

# default.yaml carries every field with its explanation. Any other config lists
# only what it changes, so a preset reads as the difference it makes.
BASE = os.path.join(HERE, "configs", "default.yaml")


def resolve(name_or_path):
    """A preset name, or a path to a config file of your own."""
    if os.path.sep in name_or_path or name_or_path.endswith((".yaml", ".yml")):
        return name_or_path
    return os.path.join(HERE, "configs", name_or_path + ".yaml")


def read(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load(name_or_path="default", overrides=()):
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
            if want is bool:
                cfg[k] = v if isinstance(v, bool) else \
                    str(v).strip().lower() in ("1", "true", "yes", "on")
            elif want is list:
                cfg[k] = [s.strip() for s in v.split(",")] if isinstance(v, str) \
                    else list(v)
            elif want is str:
                cfg[k] = "" if v is None else str(v)
            else:
                cfg[k] = want(v)
        except (TypeError, ValueError):
            sys.exit("%s should be %s, got %r" % (k, want.__name__, v))

    # so the shipped defaults work from any working directory
    for k in ("pairs", "videos", "val_videos"):
        if cfg[k] and not os.path.isabs(cfg[k]):
            cfg[k] = os.path.join(REPO, cfg[k])
    return SimpleNamespace(**cfg)
