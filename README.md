# IsoVGRBench: Video Reward Modeling Without Human Feedback

The benchmark is on HuggingFace:
[IsoVGRBench](https://huggingface.co/datasets/stellalisy/IsoVGRBench).

Video reward models are trained and evaluated on human preferences over
independently sampled videos. Two such videos differ along many dimensions at
once, so a preference label records which video is better without recording
why.

This repository changes one thing about one generated video instead. The
counterpart is matched everywhere else, so the label follows from the change
and arrives tagged with the dimension it moves.

![shuffle](example_data/previews/shuffle.gif)

Left and right hold the same frames of the same clip. Only their order differs,
so the left one is better on motion quality and nothing else separates them. Reward
models are near chance on this. The other seven interventions are shown in
[IsoGen](IsoGen#what-they-look-like).

| directory | what it is |
|---|---|
| [IsoGen/](IsoGen) | the generation procedure. A prompt in, a labelled pair out |
| [IsoVGRBench/](IsoVGRBench) | run the benchmark, 16,000 pairs in two subsets of 8,000 |
| [reward_training/](reward_training) | train a reward model on generated pairs |
| [example_data/](example_data) | 64 pairs over 8 prompts, so everything here runs with no download |

Neither the benchmark nor the training corpus
([IsoVGRBench-corpus](https://huggingface.co/datasets/stellalisy/IsoVGRBench-corpus))
is needed to run any of this. `example_data` ships in the repository and is
what the trainer reads by default, and `IsoGen` builds pairs from your own
prompts or your own videos.

## Install

One environment per directory, each pinned to the versions it was tested on.
IsoGen runs on transformers 5 and the trainer on transformers 4, so they do not
share one.

```bash
pip install -r IsoGen/requirements.txt            # generate pairs
pip install -r IsoVGRBench/requirements.txt       # run the benchmark
pip install -r reward_training/requirements.txt   # train a reward model
```

`mq_hold` and `shuffle` rearrange the anchor's own frames and run without torch
installed, on numpy, pillow, pyyaml and imageio alone. Every other family loads
a generator on first use.

## What it finds

Reward models read frame content and largely ignore frame order. Where two
clips are separate renders the best of nine reaches 79.1 to 91.1. Where the
frames are the anchor's own and only their arrangement differs it reaches 71.6
and 60.8, with six of nine at or below chance, while people are at 93 and 100.

## License

Apache 2.0, see [LICENSE](LICENSE).
