# IsoVRBench

The benchmark is 8,000 pairs built by [IsoGen](../IsoGen), 1,000 for each of
eight interventions. One clip was changed in exactly one respect, so the correct
answer follows from the construction rather than from anyone judging the result.
They are the `test` split of the dataset, which also carries the training and
validation pairs.

The data lives at
[stellalisy/IsoVRBench](https://huggingface.co/datasets/stellalisy/IsoVRBench).

## Run it

Write a function that scores one clip:

```python
# my_model.py
def score_clip(video_path, prompt):
    return float(...)          # higher is better
```

Then:

```bash
huggingface-cli login          # the dataset is private
python evaluate.py --model my_model:score_clip
```

```
             ties=wrong  decided    ties
mq_hold          71.6     71.9       4
shuffle          60.8     61.0       4
mq_gen           84.7     84.9       2
ta_attrib        74.1     74.6       7
vq_steps         78.5     78.7       5
size_i2v         73.6     74.0       6
size_5b          77.9     78.2       4
size_1p3b        64.9     65.2       5
average         73.3
```

That is VideoReward's row, for reference.

A model with separate heads returns a dict instead, and each pair is then read
on the head its `scored_on` field names:

```python
def score_clip(video_path, prompt):
    return {"VQ": ..., "MQ": ..., "TA": ...}
```

## How accuracy is computed

A pair is **decided** if the model gives its two clips different scores, and a
**tie** otherwise. Within one intervention, over its n = 1,000 pairs:

```
ties_wrong = correct / n
decided    = correct / (n - ties)
```

`ties_wrong` is the paper's main table. Every pair carries a change a person can
see, so scoring the two clips equally is a failure to perceive it. `decided`
drops ties and matches the convention VideoGen-RewardBench uses. The two differ only
by the tie rate, which is under one percent for most models.

The headline average is the mean of the eight per-intervention accuracies, not
`correct / 8000`. They coincide here because the interventions are equal sized,
and the macro average is what is reported so that a subset dropped for missing
clips does not silently reweight the rest.

At n = 1,000 a 95 percent Wilson interval reaches at most 3.1 points either
side, so differences smaller than about 4 points between two models on one
intervention are not separable.

A model with per-dimension heads is read on the head each pair's `scored_on`
field names, so a model that scores motion separately is judged on motion. A
model returning one number is read on that number everywhere.

## Practical notes

Scores are keyed on `(path, prompt)` rather than on path alone, because the
score depends on the prompt. If you build a scores file
yourself and omit the prompt, every lookup misses and the scorer says so rather
than reporting nulls.

Scores are cached to `scores.jsonl`, so a rerun after a crash resumes rather
than rescoring 15,047 clips.

## Files

`evaluate.py` fetches the test split only, not the training shards, scores every clip, and reports per
intervention accuracy. `score.py` does the accounting and is the same file the
dataset card ships, so numbers from here and from the card agree.
