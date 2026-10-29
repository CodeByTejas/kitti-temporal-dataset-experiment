"""Step 8 - VLM experiment (Qwen2-VL, zero-shot): single frame vs ordered sequence vs shuffled sequence,
plus free-text temporal-change descriptions and early-vs-late comparison."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.config import CLASSES, add_common_args, cfg_from_args
from src.data.viz import entry, plot_strips
from src.data.windows import get_windows
from src.vlm.qwen import QwenVL, parse_label

ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
ap.add_argument("--n_per_class", type=int, default=15)
ap.add_argument("--n_describe", type=int, default=6)
ap.add_argument("--frames", type=int, default=8, help="frames given to the VLM in sequence prompts")
a = ap.parse_args()
cfg = cfg_from_args(a)
res = cfg.results_dir; fig_dir = res / "figures"; (res / "vlm").mkdir(exist_ok=True, parents=True); fig_dir.mkdir(exist_ok=True)
_, y, meta, idx = get_windows(cfg, cfg.splits["test"])
rng = np.random.default_rng(0)
sel = np.concatenate([rng.choice(np.where(y == c)[0], min(a.n_per_class, (y == c).sum()), replace=False)
                      for c in range(4) if (y == c).any()])
dt = cfg.gap * 0.1
has_stop = bool((y == 0).any())
opts = "stopped, straight, left, right" if has_stop else "straight, left, right"
state_q = ("stopped, driving straight, turning left or turning right" if has_stop
           else "driving straight, turning left or turning right")
Q = f"Answer with exactly one word from: {opts}."
P_SINGLE = ("This is a single frame from a forward-facing camera mounted on a moving car. "
            f"Is the car currently {state_q}? " + Q)
P_SEQ = (f"These are {a.frames} consecutive frames, in temporal order, from a forward-facing camera mounted on a car, "
         f"{dt:.1f} seconds apart. Looking at how the scene changes between frames, is the car {state_q}? " + Q)
P_DESC = ("These are consecutive frames in temporal order from a car's forward-facing camera. In at most three "
          "sentences: describe what changes between the frames and what information becomes available over time "
          "that a single frame would not show.")
P_EARLY_LATE = ("The first image is an early frame and the second image a later frame from a car's forward camera. "
                "Compare them: has the car moved, and how (forward, turning, or not at all)? Answer in two sentences.")

vlm = QwenVL(a.model)
rows = []
for n, j in enumerate(sel):
    seq, ii = meta.seq[j], idx[j][-a.frames:]
    imgs = [vlm.load(cfg, seq, i) for i in ii]
    labels = [f"Frame {k + 1}:" for k in range(len(imgs))]
    shuf = rng.permutation(len(imgs))
    ans = dict(single=vlm.ask([imgs[-1]], P_SINGLE),
               sequence=vlm.ask(imgs, P_SEQ, labels),
               shuffled=vlm.ask([imgs[k] for k in shuf], P_SEQ, labels))
    rows.append(dict(seq=seq, start=int(meta.start[j]), true=int(y[j]), **{f"{k}_raw": v for k, v in ans.items()},
                     **{f"{k}_pred": parse_label(v) for k, v in ans.items()}))
    if n % 10 == 0:
        print(f"{n + 1}/{len(sel)} true={CLASSES[y[j]]} single={ans['single']!r} seq={ans['sequence']!r}")
df = pd.DataFrame(rows); df.to_csv(res / "vlm" / "vlm_predictions.csv", index=False)

summ = {}
for k in ["single", "sequence", "shuffled"]:
    p = df[f"{k}_pred"].fillna(-1).astype(int).values
    summ[k] = dict(acc=float((p == df.true.values).mean()),
                   macro_f1=float(f1_score(df.true, p, labels=sorted(df.true.unique()), average="macro", zero_division=0)),
                   invalid=int((p < 0).sum()),
                   answer_distribution={CLASSES[c]: int((p == c).sum()) for c in range(4)},
                   cm=confusion_matrix(df.true, p, labels=list(range(4))).tolist())
# compare with the embedding-based models on the SAME windows
for name in ["single", "transformer"]:
    try:
        P = np.mean([np.load(res / "preds" / f"{name}_s{s}.npy") for s in range(3)], 0).argmax(1)[sel]
        summ[f"embedding_{name}_same_windows"] = dict(acc=float((P == df.true.values).mean()))
    except FileNotFoundError:
        pass
(res / "vlm" / "vlm_summary.json").write_text(json.dumps(summ, indent=1))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "cm"} for k, v in summ.items()}, indent=1))

fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
for a_, k in zip(axs, ["single", "sequence", "shuffled"]):
    cm = np.array(summ[k]["cm"], float); cm /= cm.sum(1, keepdims=True).clip(1)
    a_.imshow(cm, vmin=0, vmax=1, cmap="Blues"); a_.set_title(f"VLM {k} (acc {summ[k]['acc']:.2f})")
    a_.set_xticks(range(4)); a_.set_xticklabels(CLASSES, rotation=30); a_.set_yticks(range(4)); a_.set_yticklabels(CLASSES)
    for i in range(4):
        for j in range(4):
            a_.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.tight_layout(); fig.savefig(fig_dir / "vlm_confusions.png", dpi=130); plt.close(fig)

# qualitative: free-text temporal change analysis + early-vs-late comparison
desc_ids = np.concatenate([rng.choice(np.where(y == c)[0], 2, replace=False) for c in range(4) if (y == c).sum() >= 2])
desc_ids = desc_ids[: a.n_describe]
md = ["# VLM qualitative outputs\n"]
for j in desc_ids:
    seq, ii = meta.seq[j], idx[j][-a.frames:]
    imgs = [vlm.load(cfg, seq, i) for i in ii]
    d = vlm.ask(imgs, P_DESC, [f"Frame {k + 1}:" for k in range(len(imgs))], max_new_tokens=140)
    e = vlm.ask([imgs[0], imgs[-1]], P_EARLY_LATE, ["Early frame:", "Later frame:"], max_new_tokens=90)
    md.append(f"## seq {seq} start {int(meta.start[j])} - ground truth: **{CLASSES[y[j]]}** "
              f"({meta.disp[j]:.1f} m, {meta.yaw[j]:+.1f} deg)\n- **Sequence description:** {d}\n- **Early vs late:** {e}\n")
(res / "vlm" / "vlm_descriptions.md").write_text("\n".join(md))
plot_strips(cfg, [entry(cfg, meta, idx, int(j)) for j in desc_ids], fig_dir / "vlm_described_windows.png")
print("VLM outputs ->", res / "vlm")
