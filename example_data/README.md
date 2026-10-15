# example_data

Eight prompts, each with an anchor clip and all eight interventions applied to
it, in the same format as the training corpus. Enough to run everything in this
repository end to end without downloading anything.

```bash
python reward_training/train.py --out /tmp/ckpt --limit 20
```

That is the default, so no path is needed.

## Layout

```
prompts.txt            what each clip is, with every prompt and segment order
prompts.jsonl          the same prompts, for IsoGen's --prompts
pairs.jsonl            64 pairs, the schema the corpus uses
build.py               writes the two jsonl files from prompts.txt
videos/<pid>/*.mp4     the anchor and one clip per intervention applied to it
```

`prompts.txt` is the authored description of the folder. The two jsonl files
are derived from it and from what is on disk, so edit `prompts.txt` and rerun
`build.py` rather than editing them by hand.

`pairs.jsonl` is the one file the trainer reads. It carries what a directory
listing cannot: which side wins on which dimension, and which prompts are
validation.

A record:

```json
{"id": "mq_hold-000041", "split": "train", "intervention": "mq_hold",
 "prompt": "Sea Monster attacking a ship full of penguins",
 "video_a": {"path": "p030135/mq_hold.mp4", "prompt": "Sea Monster ..."},
 "video_b": {"path": "p030135/s50.mp4", "prompt": "Sea Monster ..."},
 "preferred": {"MQ": "B"},
 "construction": "real frames of the anchor held for several frame durations",
 "prompt_id": "p030135"}
```

`preferred` names the better side on each dimension the intervention licenses,
and only those. `mq_hold` rearranges the anchor's own pixels, so it says
nothing about visual quality and carries no VQ label. Paths are relative to
`videos/`. The preferred clip is on side A in 32 of the 64 records, so
answering the same side every time scores exactly 50.

Two of the eight prompts are validation. The split is on prompt rather than on
pair, because the families re-pair the same anchor and holding out a pair would
leave both its clips in training under a different pairing.

## What is here

Eight pairs for each of the eight interventions, one per prompt. Every pair
prefers the anchor, `s50.mp4`.

| intervention | labels |
|---|---|
| `vq_steps` | VQ |
| `size_1p3b` | VQ, MQ |
| `size_5b` | VQ, MQ |
| `size_i2v` | VQ, MQ |
| `mq_gen` | MQ |
| `mq_hold` | MQ |
| `shuffle` | MQ |
| `ta_attrib` | TA |

`shuffle` records carry the anchor's motion score, which runs from 29.1 to 70.0
here, against a default `shuffle_motion_min` of 25. Reordering a clip that
barely moves changes nothing a person could see, so the procedure skips anchors
below that floor, and prompts that fell below it were left out of this folder
rather than shipped without a shuffle pair.

`ta_attrib` records carry `edited_prompt`, the rewriting that rendered the
rejected clip. Both clips are scored against the original prompt; the edit is
there to be read, not used. Prompts offering fewer than three disjoint
attribute swaps have no edited prompt, and were likewise left out.

`vq_steps` uses 8 denoising steps, the benchmark setting. The corpus uses 16, a
milder degradation worth about 13 points of model accuracy on the same items.

## Rebuilding it

The metadata comes from `prompts.txt`, and rebuilding it needs no model and no
GPU:

```bash
python example_data/build.py
```

That rewrites `pairs.jsonl` and `prompts.jsonl`, recomputing each anchor's
motion score and checking that every prompt still carries all eight families.
It refuses rather than writing a short file if one does not.

Rebuilding the clips themselves is a different job. `shuffle` and `mq_hold`
rearrange the anchor's own frames and need no model:

```bash
python IsoGen/generate.py --anchors example_data/videos --out /tmp/pairs \
  --families mq_hold,shuffle
```

The rest render video, which needs the generators named in `IsoGen/README.md`:

```bash
python IsoGen/generate.py --prompts example_data/prompts.jsonl --out /tmp/pairs
```

## What this is not

Sixty four pairs will not train a reward model. This is for checking that the
code runs, reading the data format, and seeing what each intervention looks
like. The corpus behind the paper is 164,000 pairs and the benchmark is 8,000.

The prompts come from VidProM, filtered and deduplicated as described in
[`prompts/`](../prompts).
