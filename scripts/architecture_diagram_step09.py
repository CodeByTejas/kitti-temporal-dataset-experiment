"""Step 9 - architecture diagram: how this experiment would grow into a multimodal perception system."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/figures/architecture.png")
out = Path(ap.parse_args().out); out.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 7.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 7.2); ax.axis("off")


def box(x, y, w, h, text, fc, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None)


def arrow(x1, y1, x2, y2, style="-|>", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle=style, lw=1.3, color="#333", ls=ls))


cols = ["#dbeafe", "#dcfce7", "#fef9c3", "#fde4cf", "#fbd5e5", "#e5dbff"]
# layer titles
for i, t in enumerate(["1. Input", "2. Representation", "3. Temporal model", "4. Multimodal fusion", "5. Semantic / VLM", "6. Prediction"]):
    ax.text(0.15 + [0, 2.3, 4.45, 6.6, 8.75, 11.05][i], 6.95, t, fontsize=9, fontweight="bold", ha="left")

# inputs
ins = ["RGB / gray video\n(multi-cam)", "Depth / LiDAR\n(range, structure)", "IMU + wheel odom\n(100-200 Hz)",
       "Pose / GPS-INS\n(spatial prior)", "Audio\n(engine, sirens)"]
for k, t in enumerate(ins):
    box(0.15, 5.7 - k * 1.3, 1.9, 0.95, t, cols[0])
# encoders
enc = ["Frozen DINOv2 / SigLIP\n(+ patch tokens)", "Depth / point-cloud\nencoder", "1-D CNN / SSM\nIMU encoder",
       "Pose / ego-motion\nembedding", "Audio spectrogram\nencoder"]
for k, t in enumerate(enc):
    box(2.3, 5.7 - k * 1.3, 1.9, 0.95, t, cols[1]); arrow(2.05, 6.17 - k * 1.3, 2.3, 6.17 - k * 1.3)
# temporal
box(4.45, 3.4, 1.9, 3.0, "Temporal encoder\n\ncausal Transformer / SSM\nwith time-stamp\nencodings\n\nstreaming KV-cache\n= bounded memory\n\n[this repo: GRU /\nTransformer + attn viz]", cols[2], 8)
box(4.45, 0.9, 1.9, 2.1, "Event / long-term\nmemory\n\nkeyframes + summaries\nfor minutes-hours", cols[2], 8)
for k in range(5):
    arrow(4.2, 6.17 - k * 1.3, 4.45, 4.9 if k < 3 else 4.3)
arrow(5.4, 3.4, 5.4, 3.0, "<|-|>")
# fusion
box(6.6, 2.4, 1.9, 3.9, "Cross-modal fusion\n\nPerceiver-style latent\ncross-attention\n\ntemporally aligned\nvisual <-> IMU <-> depth\ntokens\n\nmodality dropout for\nrobustness to sensor\nfailure", cols[3], 8)
arrow(6.35, 4.85, 6.6, 4.85); arrow(6.35, 1.95, 6.6, 3.2)
# VLM
box(8.75, 3.9, 2.0, 2.4, "VLM / LLM layer\n\nQwen2-VL-class model\nreads temporal tokens +\ntext query\n\n-> descriptions, QA,\nambiguity flags", cols[4], 8)
box(8.75, 1.4, 2.0, 2.1, "Symbolic / semantic\nstate\n\nmotion state, objects,\nevents, scene graph", cols[4], 8)
arrow(8.5, 4.9, 8.75, 5.1); arrow(8.5, 3.6, 8.75, 2.9); arrow(9.75, 3.9, 9.75, 3.5, "<|-|>")
# outputs
outs = ["Ego-motion state\n(stop/straight/turn)", "Future-state\nprediction",
        "Events + anomalies", "Natural-language\nexplanation"]
for k, t in enumerate(outs):
    box(11.05, 5.55 - k * 1.4, 1.8, 1.0, t, cols[5], 8); arrow(10.75, 4.9 if k < 2 else 2.4, 11.05, 6.05 - k * 1.4)
# training signals
box(4.45, 0.05, 6.3, 0.6, "Training signals: supervised motion labels (poses)  +  self-supervised contrastive / next-frame prediction  +  VLM distillation (pseudo-labels)", "#f3f4f6", 7.5)
ax.set_title("Evolving the experiment into a multimodal temporal perception system", fontsize=11, fontweight="bold", pad=2)
fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
