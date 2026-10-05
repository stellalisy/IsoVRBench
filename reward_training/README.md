# reward_training

Train a video reward model on pairs built by [IsoGen](../IsoGen).

```bash
python train.py --out ckpt/
```

## Where the pairs come from

With nothing else set, from [example_data](../example_data): 64 pairs over 8
prompts that ship with the repository. Nothing is downloaded, and 64 pairs
train nothing, but the run exercises every code path.

Point `pairs` and `videos` elsewhere to train on something real:

```bash
python train.py --pairs my_pairs.jsonl --videos my_clips/ --out ckpt/
```

`pairs` is a jsonl, one record per pair, naming two clips and which side each
dimension prefers. It is the form IsoGen writes and the form the corpus ships
in. `videos` is the directory those paths are relative to.

The corpus behind the paper is 164,000 pairs at
[stellalisy/IsoVRBench-corpus](https://huggingface.co/datasets/stellalisy/IsoVRBench-corpus),
split 160,000 training and 4,000 validation, prompt disjoint from each other
and from the benchmark. It is metadata only, so the clips have to come from
somewhere `videos` can reach. Its validation clips sit in their own tree, which
is what `val_videos` is for.

The split is on prompt rather than on pair. Several families re-pair the same
anchor, so holding out a pair would leave both its clips in training under a
different pairing.

## How it works

A vision language backbone with a scalar head read at marker tokens, one marker
per dimension, trained with a Bradley Terry objective.

One head of width one read at three positions, rather than three heads, so the
dimensions share every parameter except where they are read.

A pair carries a label only on the dimensions its intervention licenses, and
the loss sums over those alone. Reordering frames says nothing about visual
quality, so a `shuffle` pair contributes no visual quality gradient. That is
what makes dimension separated supervision possible: an ordinary preference
over two independently sampled videos cannot say which dimension it is about.

## Settings

Hyperparameters, which backbone to load, whether to use LoRA, and where the
data lives all live in `configs/*.yaml`.

| config | what it is |
|---|---|
| `default.yaml` | one GPU, LoRA, a single pass; every field explained here |
| `paper.yaml` | the recipe behind the paper's models, effective batch 32 |
| `full_finetune.yaml` | every parameter moves, wants several cards |

`default.yaml` is the base. The other two are layered on top of it and list
only what they change, so a preset reads as the difference it makes.

```bash
python train.py --config paper --videos /path/to/clips --out ckpt/
```

Copy one and edit it, or change a field without editing anything:

```bash
python train.py ... --set lora=false --set frames=16 --set lr=1e-5
```

A misspelled key is an error rather than a silent default, and the resolved
settings are written to the checkpoint as `config.yaml`, so a run can be traced
back to what produced it. `--videos` and `--pairs` are shorthand for `--set`
on those two fields, since they change per machine. The one flag with no config
field is `--limit`, which stops after a given number of pairs for a smoke test.

Validation runs at every save, reporting the fraction of labelled dimensions
ranked the right way round. A tie counts as wrong, since every pair carries a
change a person can see. Set `val_split` empty to skip it.

## This is a reference, not the production code

One GPU, one process, plain PyTorch, written to be read. The paper's models
trained on 32 GPUs under DeepSpeed with an effective batch of 32, and that
setup lives in the research repository rather than here.

Two things to know if you scale it up. Sample frames uniformly rather than
taking the first n, because the frame reuse families differ from the anchor
only in the middle. And hold out by prompt rather than by pair: several
interventions reuse anchors, so dropping a pair leaves both its clips in
training under a different pairing.
