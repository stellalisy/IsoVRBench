# prompts

How the prompts behind IsoVRBench and its training corpus were chosen, from raw
VidProM to the train, validation and test splits.

| step | script | in | out |
|---|---|---|---|
| 1 | `prompt_pipeline.py` | `VidProM_semantic_unique.csv` | the prompt pool, 880,052 prompts |
| 2 | `trigger_study_splits.py` | the pool | the held-out splits used to curate `IsoGen/triggers.json` |
| 3 | `select_prompts.py` | the pool, minus step 2's splits | the prompts to generate from, with prompt ids |
| 4 | `split_prompts.py` | the generated pairs | `train`, `validation` and `test` |

Generation itself, from a prompt to a labeled pair, is [`IsoGen`](../IsoGen).

The scripts need only the Python standard library. `select_prompts.py` uses
the trigger matching in `IsoGen/attribute_edit.py`.

## 1. The pool

```bash
python prompt_pipeline.py --src VidProM_semantic_unique.csv --measure   # audit report
python prompt_pipeline.py --src VidProM_semantic_unique.csv --build --out pool/
```

The source is VidProM's semantically deduplicated release, 1,038,805 prompts.
The pipeline has four steps.

| step | what it does |
|---|---|
| 0 source | `VidProM_semantic_unique.csv` |
| 1 clean | strip tool and UI syntax that is not scene content (Midjourney flags, aspect ratios, "prompt:" prefixes, HTML, a recurring negative-prompt template, "Create a video of" wrappers); never remove scene words |
| 2 validity | all six Detoxify scores VidProM ships below 0.5, 20 to 400 characters after cleaning, at least 85% ASCII |
| 3 blacklist | reject the prompts that are unusable as a scene |

896,550 prompts pass steps 0 to 2, and the blacklist rejects 16,498 of them
(1.84%).

| rule | rejects | n |
|---|---|---|
| `foreign_latin` | non-English text in Latin script, by English and foreign function-word ratios together | 12,974 |
| `logo_typography` | logo and lettering jobs, when the logo is the subject | 2,286 |
| `dangling_edit` | edit instructions pointing at an image that is not there ("make it brighter") | 856 |
| `empty_after_clean` | fewer than two words left after cleaning | 161 |
| `tool_request` | requests about the tool rather than a scene, when under 12 content words remain | 152 |
| `non_latin_script` | prompts mostly in a non-Latin script | 69 |

Step 3 is a blacklist because a whitelist failed. The first version kept only
prompts containing one of 176 scene words, and rejected 38.8% of valid prompts.
About 40% of those were usable scenes missing from the word list ("samurai
katana cutting through a pig", "pilots in the aircraft"). The blacklist rejects
1.84%. The script records every rule tried and removed, and names the prompt
that motivated each rule kept.

`--build` writes `pool_v2.0.jsonl` and a manifest with the sha256 of the source
and of the script, the per-rule counts and every threshold.
`pool_v2.0.manifest.json` here is the manifest of the pool the paper used. Its
`script_sha256` is that of the original script, which differs from this one
only in taking its paths as arguments.

## 2. Splits for curating the trigger list

```bash
python trigger_study_splits.py --pool pool/pool_v2.0.jsonl --out splits/
```

`triggers.json` was curated by reading prompts by hand. This draws the
prompts that were read (`calib`, 1,000), two held-out validation rounds
(`val_a`, `val_b`, 200 each) and a fixed set of 20,000 for measuring coverage
(`eval`). None of them is training data.

## 3. Selecting prompts

```bash
python select_prompts.py --pool pool/pool_v2.0.jsonl --n 75000 \
    --exclude splits/*.jsonl splits/calib.txt --out prompts.jsonl
```

A prompt is kept when it has 8 to 60 words, is not in any `--exclude` file, is
not a duplicate, and contains at least one trigger from `triggers.json`. 86.3%
of the pool contained one under the trigger list the release used. Attribute Edit can then use the same anchors as every
other intervention.

Selection balances the attribute that Attribute Edit would change. Each prompt
is assigned the rarest dimension among its triggers, and dimensions are filled
round-robin, rarest first. Selecting on availability alone would give a corpus
of mostly location swaps, since `location-outdoor` appears about twice as often
as the next dimension and about 30 times as often as the rarest.

Prompts are sorted by text and numbered `p000000` onward. These are the
`prompt_id` values in the released data.

## 4. Splitting by prompt id

```bash
python split_prompts.py --pairs pairs.jsonl --out data/
```

All three splits are disjoint on prompt id, not on pair, because several
interventions reuse one anchor clip. On a pair-level split, 99.8% of the
benchmark pairs of some interventions had both clips in training.

- **Test.** A prompt id is eligible when `sha256("gen2holdout:" + pid) mod 100`
  is below 20, and the test split samples 1,000 pairs per intervention from
  pairs whose prompt ids are all eligible. In the release, Motion Re-gen, Frame
  Hold and the three generator substitutions follow this rule. Temporal
  Shuffle, Reduced Steps and Attribute Edit were built separately, on the
  highest-motion anchors, at 8 denoising steps and with a single edited
  attribute, and joined to them. Pass a test file built that way with `--test`.
- **Train and validation.** Every pair touching a test prompt id is dropped.
  Validation prompt ids are then reserved until each intervention has 500
  pairs, and the remaining prompt ids supply 20,000 training pairs per
  intervention. Reserving whole prompts keeps validation anchors out of
  training.

The script refuses to write splits that share a prompt id.
