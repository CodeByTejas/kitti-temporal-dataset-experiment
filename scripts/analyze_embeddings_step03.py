"""Step 3 - how do frame representations change over time? (+ single-frame ambiguity examples)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.config import CLASSES, add_common_args, cfg_from_args
from src.data.viz import entry, plot_strips
from src.data.windows import get_windows, load_emb

ap = add_common_args(argparse.ArgumentParser())
cfg = cfg_from_args(ap.parse_args())
fig_dir = cfg.results_dir / "figures"; fig_dir.mkdir(exist_ok=True, parents=True)
seqs = cfg.splits["train"] + cfg.splits["test"]
X, y, meta, idx = get_windows(cfg, seqs)               # raw (un-normalised) embeddings
Xn = X / np.linalg.norm(X, axis=-1, keepdims=True)
colors = ["tab:gray", "tab:blue", "tab:green", "tab:red"]

# (a) cosine similarity between consecutive frames of a window, per motion class
cs = (Xn[:, 1:] * Xn[:, :-1]).sum(-1).mean(1)
# (e) similarity to the first frame as a function of lag
lag = np.stack([(Xn[:, 0] * Xn[:, t]).sum(-1) for t in range(cfg.T)], 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].boxplot([cs[y == c] if (y == c).any() else [np.nan] for c in range(4)], showfliers=False); ax[0].set_xticklabels(CLASSES)
ax[0].set_title(f"mean cosine sim. of consecutive frames ({cfg.gap * 0.1:.1f}s apart)")
for c in range(4):
    if (y == c).any():
        ax[1].errorbar(np.arange(cfg.T) * cfg.gap * 0.1, lag[y == c].mean(0), lag[y == c].std(0), label=CLASSES[c],
                       c=colors[c], capsize=2)
ax[1].set_xlabel("lag from first frame [s]"); ax[1].set_ylabel("cos(E_0, E_t)"); ax[1].legend()
ax[1].set_title("embedding drift vs lag")
fig.tight_layout(); fig.savefig(fig_dir / "emb_consecutive_cosine.png", dpi=130); plt.close(fig)
cos_tab = {CLASSES[c]: float(cs[y == c].mean()) for c in range(4) if (y == c).any()}
print("mean consecutive cosine per class:", cos_tab)

# (b) temporal embedding trajectory: PCA of one test sequence, coloured by time
seq = cfg.splits["test"][0]
E = load_emb(cfg, seq); P = PCA(2).fit_transform(E)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sc = ax[0].scatter(P[:, 0], P[:, 1], c=np.arange(len(P)), s=3, cmap="viridis"); ax[0].plot(P[:, 0], P[:, 1], lw=.2, c="k")
ax[0].set_title(f"PCA trajectory of seq {seq} (colour = time)"); plt.colorbar(sc, ax=ax[0])
step = np.linalg.norm(np.diff(E, axis=0), axis=1)
ax[1].plot(step); ax[1].set_title("||E_t - E_{t-1}|| over the sequence"); ax[1].set_xlabel("frame")
fig.tight_layout(); fig.savefig(fig_dir / "emb_trajectory.png", dpi=130); plt.close(fig)

# (c) similarity between different sequences (mean embedding cosine)
M = np.stack([load_emb(cfg, s).mean(0) for s in seqs]); M /= np.linalg.norm(M, axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(M @ M.T, cmap="magma"); ax.set_xticks(range(len(seqs))); ax.set_xticklabels(seqs)
ax.set_yticks(range(len(seqs))); ax.set_yticklabels(seqs); plt.colorbar(im)
ax.set_title("cosine sim. of sequence-mean embeddings")
fig.tight_layout(); fig.savefig(fig_dir / "emb_sequence_similarity.png", dpi=130); plt.close(fig)

# (d) simple temporal pooling: PCA of last-frame vs mean-pooled embedding, coloured by class
sub = np.random.default_rng(0).choice(len(X), min(4000, len(X)), replace=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a_, (name, F) in zip(ax, [("last frame", X[sub, -1]), ("mean-pooled window", X[sub].mean(1))]):
    Z = PCA(2).fit_transform(F)
    for c in range(4):
        m = y[sub] == c
        a_.scatter(Z[m, 0], Z[m, 1], s=3, c=colors[c], label=CLASSES[c], alpha=.5)
    a_.set_title(f"PCA of {name} embedding"); a_.legend(markerscale=4)
fig.tight_layout(); fig.savefig(fig_dir / "emb_pooling_pca.png", dpi=130); plt.close(fig)

# (f) single-frame ambiguity: most similar LAST frames of a stopped vs a moving-straight window
last = Xn[:, -1]
st, mv = np.where(y == 0)[0], np.where(y == 1)[0]
if len(st) and len(mv):
    st, mv = np.random.default_rng(0).choice(st, min(1500, len(st)), replace=False), \
        np.random.default_rng(1).choice(mv, min(3000, len(mv)), replace=False)
    S = last[st] @ last[mv].T
    order = np.dstack(np.unravel_index(np.argsort(-S, axis=None)[:200], S.shape))[0]
    chosen, used = [], set()
    for i, j in order:                                   # pick 3 distinct, non-adjacent pairs
        if all(abs(meta.start[st[i]] - meta.start[st[u]]) > 40 or meta.seq[st[i]] != meta.seq[st[u]] for u in used):
            chosen.append((st[i], mv[j], S[i, j])); used.add(i)
        if len(chosen) == 3:
            break
    ents = []
    for a_, b_, s_ in chosen:
        ents += [entry(cfg, meta, idx, int(a_), f"last-frame cos={s_:.3f}"), entry(cfg, meta, idx, int(b_))]
    plot_strips(cfg, ents, fig_dir / "ambiguous_pairs_stopped_vs_moving.png")
    print("saved ambiguity pairs (nearly identical last frame, different motion label)")
print("figures ->", fig_dir)
