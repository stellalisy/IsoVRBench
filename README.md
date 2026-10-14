<div align="center">

# IsoVRBench

### Video Reward Modeling Without Human Feedback

**Shuyue Stella Li**<sup>1,2</sup>, **Jacqueline He**<sup>1,2</sup>, **Jiahui Chen**<sup>2</sup>, **Sreya Dutta Roy**<sup>2</sup>,
**Yulia Tsvetkov**<sup>1</sup>, **Luke Zettlemoyer**<sup>1,2</sup>, **Marjan Ghazvininejad**<sup>2†</sup>, **Xiaochuang Han**<sup>1,2†</sup>

<sup>1</sup>University of Washington &nbsp;&nbsp; <sup>2</sup>Meta FAIR &nbsp;&nbsp; <sup>†</sup>Equal last author

[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-IsoVRBench-yellow)](https://huggingface.co/datasets/stellalisy/IsoVRBench)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

---

Video generators are post-trained against reward models, so a generator improves only along the
properties its reward model perceives. Reward models are trained and evaluated on human
preferences between independently generated videos, which differ in many properties at once, so
a model's accuracy on them does not show which properties it perceives.

**IsoVRBench** builds preference pairs by *controlled intervention* instead. Each intervention
changes one property of a generated video and holds the rest fixed, so the construction fixes
both the preferred side and the property under test. No pair needs a human label, and human
validation is needed only once per intervention to confirm that the change is perceptible.

<div align="center">
<img src="example_data/previews/shuffle.gif" width="720" alt="Temporal Shuffle: the same frames of one clip, in their original and in a shuffled order">
<br>
<sub><b>Temporal Shuffle.</b> Both sides hold the same frames of the same clip in a different order.
People pick the original 100% of the time. The best reward model does so 60.8% of the time.</sub>
</div>

## Findings

- **Reward models perceive frame content but miss frame order.** The best of nine reward models
  reaches 60.8% on Temporal Shuffle against 100% for people, and six of nine are at or below
  chance. No model exceeds 66.1% at any frame count up to 48.
- **Existing benchmarks do not predict this.** Rankings on VideoGen-RewardBench, GenAI-Bench,
  VBench-2.0, TemporalBench and TempCompass correlate with Temporal Shuffle accuracy at
  &minus;0.15 to +0.37, and none differs significantly from zero.
- **The same pairs train better reward models.** IsoVRM-Q2, trained only on intervention pairs
  from the same base model as VideoReward, outperforms VideoReward by 3.8 points on VBench-2.0.
  Used to post-train Wan2.2-TI2V-5B, it yields the generator people prefer most in a three-way
  comparison (55.4% win frequency, against 43.3% under VideoReward).

## The eight interventions

| Intervention | Code | What changes | Labeled on |
|---|---|---|---|
| Temporal Shuffle | `shuffle` | the anchor is cut into six segments and reordered | motion |
| Frame Hold | `mq_hold` | 8 to 12 anchor frames are each held until the next | motion |
| Motion Re-gen | `mq_gen` | motion is regenerated between real keyframes | motion |
| Attribute Edit | `ta_attrib` | one attribute of the prompt is changed before re-rendering | text alignment |
| Reduced Steps | `vq_steps` | 8 denoising steps instead of 50, same prompt and seed | visual quality |
| Aligned Weak Gen | `size_i2v` | Wan2.2-TI2V-5B continues the anchor's first frame | overall |
| Weak Gen (5B) | `size_5b` | the prompt is rendered by Wan2.2-TI2V-5B | overall |
| Weakest Gen (1.3B) | `size_1p3b` | the prompt is rendered by Wan2.1-T2V-1.3B | overall |

Anchors are rendered by Wan2.2-T2V-A14B, Wan2.2-TI2V-5B and Wan2.1-T2V-1.3B with 50 denoising
steps. Previews of every intervention are in [IsoGen](IsoGen#what-they-look-like).

## Results

Accuracy (%) on the 8,000 benchmark pairs, ties counted as wrong. Chance is 50.

| Model | Avg. | Temporal Shuffle | Frame Hold | Motion Re-gen | Attribute Edit | Reduced Steps | Aligned Weak Gen | Weak Gen (5B) | Weakest Gen (1.3B) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VideoReward | **73.3** | **60.8** | **71.6** | **84.7** | 74.1 | 78.5 | 73.6 | 77.9 | 64.9 |
| PickScore | 69.7 | 45.0 | 54.2 | 68.3 | 78.0 | 68.0 | 75.0 | **89.6** | 79.4 |
| UnifiedReward-Flex-9B | 68.4 | 47.7 | 49.5 | 54.3 | **78.9** | 77.4 | 69.2 | 88.2 | **81.6** |
| UnifiedReward-2.0-8B | 68.1 | 52.6 | 54.3 | 56.8 | 71.7 | 76.8 | 66.7 | 86.7 | 79.4 |
| VideoScore v1 | 66.2 | 49.5 | 39.8 | 75.2 | 49.6 | **80.5** | 74.4 | 81.5 | 78.8 |
| VideoScore v1.1 | 64.5 | 54.2 | 34.9 | 64.7 | 60.1 | 70.2 | **76.3** | 82.4 | 72.9 |
| VisionReward | 58.0 | 27.7 | 31.7 | 45.0 | 66.2 | 80.0 | 54.8 | 86.3 | 72.6 |
| VideoScore2 | 47.9 | 38.2 | 35.9 | 42.2 | 62.1 | 47.6 | 47.0 | 61.0 | 49.4 |
| LiFT | 19.5 | 11.9 | 8.9 | 18.0 | 26.0 | 22.8 | 21.9 | 21.5 | 24.8 |
| *GPT-4.1 (judge)* | 70.1 | 58.9 | 64.7 | 57.5 | 71.4 | 77.1 | 69.6 | 85.5 | 76.0 |
| *Qwen3-VL-8B (judge)* | 69.3 | 57.1 | 60.8 | 64.8 | 74.2 | 73.5 | 68.9 | 82.6 | 72.8 |
| *People* | 96.4 | 100 | 93 | 100 | 98 | 84 | 98 | 98 | 100 |

Bold marks the best reward model in each column. The two judges are general vision-language
models prompted to score each clip.

## Repository

| Directory | Contents |
|---|---|
| [`IsoGen/`](IsoGen) | the generation procedure: a prompt in, a labeled pair out |
| [`IsoVRBench/`](IsoVRBench) | evaluate any reward model on the benchmark |
| [`reward_training/`](reward_training) | train a reward model on intervention pairs |
| [`example_data/`](example_data) | 64 pairs over 8 prompts, so everything here runs with no download |

## Installation

Each directory has its own environment, pinned to the versions it was tested on. IsoGen runs on
transformers 5 and the trainer on transformers 4.

```bash
git clone https://github.com/stellalisy/IsoVRBench.git
cd IsoVRBench
pip install -r IsoGen/requirements.txt            # generate pairs
pip install -r IsoVRBench/requirements.txt        # run the benchmark
pip install -r reward_training/requirements.txt   # train a reward model
```

Temporal Shuffle and Frame Hold rearrange the anchor's own frames and need only numpy, pillow,
pyyaml and imageio. Every other intervention loads a generator on first use.

## Quick start

**Evaluate a reward model.** Write a function that scores one clip, higher being better:

```python
# my_model.py
def score_clip(video_path, prompt):
    return float(...)
```

```bash
huggingface-cli login            # the dataset is currently private
python IsoVRBench/evaluate.py --model my_model:score_clip
```

A model with separate heads can return `{"VQ": ..., "MQ": ..., "TA": ...}`, and each pair is
then read on the head for the property it tests. See [`IsoVRBench/`](IsoVRBench) for how
accuracy is computed.

**Generate pairs from your own prompts.**

```bash
python IsoGen/generate.py --prompts IsoGen/example_prompts.txt --out pairs/
```

**Train a reward model** on the bundled example pairs, or on the
[training split](https://huggingface.co/datasets/stellalisy/IsoVRBench) (see [`reward_training/`](reward_training) for downloading it):

```bash
python reward_training/train.py --out ckpt/
```

## Data

Everything is in one dataset,
[stellalisy/IsoVRBench](https://huggingface.co/datasets/stellalisy/IsoVRBench).

| split | pairs | clips | contents |
|---|---|---|---|
| `test` | 8,000 | 15,047 | the benchmark, 1,000 pairs per intervention over 7,129 prompts |
| `train` | 160,000 | 234,075, shared with validation | 20,000 pairs per intervention |
| `validation` | 4,000 | | 500 pairs per intervention |

Splits are disjoint on prompt id, and every benchmark prompt is excluded from
training and validation.

## Citation

```bibtex
@misc{li2026isovrbench,
  title  = {IsoVRBench: Video Reward Modeling Without Human Feedback},
  author = {Li, Shuyue Stella and He, Jacqueline and Chen, Jiahui and Dutta Roy, Sreya and
            Tsvetkov, Yulia and Zettlemoyer, Luke and Ghazvininejad, Marjan and Han, Xiaochuang},
  year   = {2026}
}
```

## License

Released under the [Apache 2.0 License](LICENSE).
