"""Step 6 - temporal attention visualisation of the CLS-token Transformer."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import CLASSES, add_common_args, cfg_from_args
from src.data.viz import entry, plot_strips
from src.data.windows import get_windows, load_norm
from src.evaluation.train import DEV, load_ckpt

ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default="transformer", help="transformer | transformer_delta | transformer_nopos")
ap.add_argument("--n_examples", type=int, default=3, help="strip examples per class")
a = ap.parse_args()
cfg = cfg_from_args(a)
fig_dir = cfg.results_dir / "figures"; fig_dir.mkdir(exist_ok=True)
norm = load_norm(cfg)
X, y, meta, idx = get_windows(cfg, cfg.splits["test"], norm=norm)


@torch.no_grad()
def cls_attention(model, X, bs=1024):
    """Returns last-layer CLS->frame attention (head-avg) and attention rollout CLS->frame, both [N, T]."""
    last, roll = [], []
    for i in range(0, len(X), bs):
        _, atts = model(torch.as_tensor(X[i:i + bs]).to(DEV), return_attn=True)
        A = [w.mean(1) for w in atts]                                  # [B, T+1, T+1] head-avg per layer
        eye = torch.eye(A[0].shape[-1], device=A[0].device)
        R = eye.expand_as(A[0])
        for w in A:
            w = 0.5 * w + 0.5 * eye
            R = (w / w.sum(-1, keepdim=True)) @ R
        last.append(A[-1][:, 0, 1:].cpu()); roll.append(R[:, 0, 1:].cpu())
    return torch.cat(last).numpy(), torch.cat(roll).numpy()


model = load_ckpt(cfg.results_dir / "checkpoints" / f"{a.model}_s{a.seed}.pt")
att, rol = cls_attention(model, X)
pred = torch.cat([model(torch.as_tensor(X[i:i + 2048]).to(DEV)).argmax(1).cpu() for i in range(0, len(X), 2048)]).numpy()

# (1) mean attention per frame position, per TRUE class
t = np.arange(cfg.T) * cfg.gap * 0.1
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a_, (A, ttl) in zip(ax, [(att, "last-layer CLS attention"), (rol, "attention rollout")]):
    M = np.stack([A[y == c].mean(0) if (y == c).any() else np.full(cfg.T, np.nan) for c in range(4)])
    im = a_.imshow(M, aspect="auto", cmap="viridis"); plt.colorbar(im, ax=a_)
    a_.set_yticks(range(4)); a_.set_yticklabels(CLASSES); a_.set_xticks(range(cfg.T))
    a_.set_xticklabels([f"{x:.1f}s" for x in t]); a_.set_title(ttl + " (mean per true class)")
    a_.set_xlabel("frame time within window")
fig.tight_layout(); fig.savefig(fig_dir / f"attention_per_class_{a.model}.png", dpi=130); plt.close(fig)

# (2) attention is content-driven or position-driven? same windows shuffled
rng = np.random.default_rng(0)
perm = np.argsort(rng.random(X.shape[:2]), 1)
Xs = np.take_along_axis(X, perm[:, :, None], 1)
att_s, _ = cls_attention(model, Xs)
print("mean attention by position (correct):", np.round(att.mean(0), 3))
print("mean attention by position (shuffled):", np.round(att_s.mean(0), 3))
# entropy of the attention (1 = uniform over frames)
ent = lambda A: float((-(A / A.sum(1, keepdims=True) * np.log(A / A.sum(1, keepdims=True) + 1e-9)).sum(1) / np.log(cfg.T)).mean())
print(f"normalised attention entropy: correct {ent(att):.3f}  shuffled {ent(att_s):.3f}")

# (3) frame strips with attention weights printed under every frame
rng = np.random.default_rng(1)
ents, bars = [], []
for c in range(4):
    ok = np.where((y == c) & (pred == c))[0]
    for j in rng.choice(ok, min(a.n_examples, len(ok)), replace=False):
        ents.append(entry(cfg, meta, idx, int(j), "(pred ok)")); bars.append(att[j])
plot_strips(cfg, ents, fig_dir / f"attention_strips_{a.model}.png", bars=bars)
print("attention figures ->", fig_dir)
