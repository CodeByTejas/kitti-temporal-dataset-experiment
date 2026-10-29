# Does a sequence of frames tell you more than one frame? (KITTI ego-motion state)

Small, reproducible experiment on **temporal visual intelligence**: frozen DINOv2 frame embeddings -> single-frame
baseline vs. temporal models (GRU / Transformer), with controlled experiments (progressive observation, temporal order,
corruption, cross-sequence generalisation), temporal-attention visualisation and a zero-shot Qwen2-VL study.

## Task
*Ego-motion state classification* on **KITTI odometry (grayscale left camera)**. A sample is a window of `T=8` frames
(every 2nd frame, i.e. 0.2 s apart -> 1.4 s span). Label from **ground-truth poses** over the whole window span:
`left` / `right` if |yaw| > 9 deg, `straight` if |yaw| < 3 deg (and >= 1.5 m travelled). **Ambiguous windows are dropped**
(dead zone: yaw 3-9 deg, slow motion). Default `--mode 3class`; `--mode 4class` adds `stopped` (< 0.3 m), which KITTI
odometry has too few examples of to evaluate reliably (see the report).
Why this task: motion is defined over time, so single frames are inherently ambiguous - but we test whether temporal
models really exploit that instead of assuming it. All thresholds are CLI arguments.

**Splits are by sequence** (no leakage through overlapping windows): train 00,02,03,05,08 / val 06,09 / test 07,10,01,04.
Sequence 01 (highway) is an explicit environment-shift case. Override with `--splits_json`.

## 1. Environment setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # a CUDA build of torch is recommended (pick it from pytorch.org)
```
Developed for a single local GPU; DINOv2-S feature extraction for all sequences takes a few minutes.

## 2. Dataset preparation
Download from the KITTI odometry page: `data_odometry_gray.zip`, `data_odometry_poses.zip` (and calib if you like).
Unzip **both** into `data/kitti/` so that the layout is:
```
data/kitti/dataset/sequences/00/image_0/000000.png ... , times.txt
data/kitti/dataset/poses/00.txt
```
(or pass `--data_root /your/path` to every script).
```bash
python scripts/prepare_data_step01.py       # integrity checks, statistics, frame strips  -> results/{tables,figures}
```
Checks: contiguous frame ids, #frames == #poses == #timestamps, monotonic timestamps, constant image size, dt stats.

## 3. Embedding extraction (frozen DINOv2, cached to `cache/`)
```bash
python scripts/extract_embeddings_step_02.py --corruptions all --frac 0.5
python scripts/analyze_embeddings_03.py   # consecutive cosine, drift vs lag, PCA trajectory, sequence similarity, pooling, ambiguity pairs
```
Frames are resized to 224x742 (keeps KITTI's aspect ratio, no crop), feature = [CLS ; mean patch token] (768-d).
`--corruptions all` additionally embeds the **test** sequences with blur / noise / occlusion applied to a random 50 % of frames.

## 4. Training
```bash
python scripts/train_step04.py --seeds 3
```
Models (`src/models/temporal.py`), all on frozen embeddings, class-weighted CE, AdamW, early stopping on val macro-F1:
`single` (random frame during training; evaluated on the last frame AND averaged over frame position), `single_wide`
(single frame, widened to the Transformer's parameter count = capacity control), `meanpool` (all frames, **no order**),
`gru`, `transformer` (CLS token + learned positions), `transformer_nopos` (permutation-invariant control), and
`gru_delta` / `transformer_delta` (input = [E_t ; 3(E_t - E_{t-1})], i.e. explicit frame-to-frame change).
Outputs: `results/main_results.json`, `results/bootstrap.json` (95 % **block-bootstrap** CIs over 5 s blocks and *paired*
differences vs the single-frame baseline - windows overlap, so plain seed std understates uncertainty).

## 5. Evaluation / experiments
```bash
python scripts/run_experiments_step05.py --seeds 3   # progressive | order | ablation | corruption | persequence
python scripts/attention_viz_step06.py --model transformer          # also: --model transformer_delta
python scripts/failure_cases_step07.py               # temporal helps / hurts / unnecessary / both wrong
python scripts/vlm_experiment_step08.py              # Qwen2-VL-2B zero-shot; try --model Qwen/Qwen2-VL-7B-Instruct
python scripts/architecture_diagram_step09.py        # system design figure
```


## How "does the temporal model use time?" is tested
| Question | Experiment |
|---|---|
| more frames helps? | progressive observation, k = 1..8 (separate model per k) |
| more information or real temporal use? | `meanpool` / `transformer_nopos` (same info, no order) vs ordered models; **shuffled / reversed** test windows |
| does reversing time flip left <-> right? | reversed windows + L<->R swap rate in `order.json` |
| which frames matter? | frame ablation (zero one frame at a time) + CLS->frame attention / rollout |
| capacity or temporal? | `single_wide` (same #params, one frame) |
| representation or model limit? | `*_delta` inputs and `--feat grid` spatial features |
| robustness | blur / noise / occlusion on 50 % of frames, frame drops (hold and zero-fill) |
| environments | per-sequence results, results with and without highway seq 01 |
| significance | block-bootstrap CIs and paired differences vs single frame |

## Repository layout
```
src/config.py           configuration + shared CLI args
src/data/               KITTI loading, integrity, labels, windows, strips
src/embeddings/         frozen encoders, corruptions, caching
src/models/             single-frame / temporal heads
src/evaluation/         training + metrics
src/vlm/                Qwen2-VL wrapper + answer parsing
scripts/                numbered pipeline steps
results/                tables, figures, vlm outputs, predictions (checkpoints are git-ignored)
report/REPORT_TEMPLATE.md
```

## Acknowledgements
KITTI odometry dataset (Geiger et al.), DINOv2 (Meta AI, via HuggingFace `transformers`), Qwen2-VL (Alibaba, via
`transformers`). Models are used frozen / zero-shot, no fine-tuning.
