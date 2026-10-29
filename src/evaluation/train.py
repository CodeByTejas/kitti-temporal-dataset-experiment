"""Training / evaluation utilities shared by all experiments."""
from __future__ import annotations

import copy
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from src.models.temporal import count_params, make_model

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
n_params = count_params


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, X, k=None, bs=1024):
    model.eval()
    X = torch.as_tensor(X)
    if k:
        X = X[:, -k:]
    out = [model(X[i:i + bs].to(DEV)).float().cpu() for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()


def metrics(y, logits, n_cls=4):
    p = logits.argmax(1)
    labs = sorted(set(y.tolist()))                      # macro-F1 over classes actually present
    return dict(acc=float((p == y).mean()),
                macro_f1=float(f1_score(y, p, labels=labs, average="macro", zero_division=0)),
                bal_acc=float(balanced_accuracy_score(y, p)),
                cm=confusion_matrix(y, p, labels=list(range(n_cls))).tolist())


def train_model(name, Xtr, ytr, Xva, yva, n_cls=4, seed=0, k=None, epochs=40, lr=1e-3, wd=1e-2,
                bs=256, patience=8, max_len=16):
    """Train on the most recent k frames of each window (k=None -> all). Early stop on val macro-F1.
    Single-frame models see a RANDOM frame of the window each step (position-agnostic baseline)."""
    set_seed(seed)
    if k:
        Xtr, Xva = Xtr[:, -k:], Xva[:, -k:]
    rand_frame = name.startswith("single")
    Xt = torch.as_tensor(Xtr).to(DEV); yt = torch.as_tensor(ytr).to(DEV)
    model = make_model(name, Xtr.shape[-1], n_cls, max_len).to(DEV)
    cnt = np.bincount(ytr, minlength=n_cls).astype(np.float64)
    w = torch.tensor(cnt.sum() / (max((cnt > 0).sum(), 1) * np.maximum(cnt, 1)), dtype=torch.float32, device=DEV)
    w[torch.as_tensor(cnt == 0, device=DEV)] = 0.0
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            xb = Xt[b]
            if rand_frame:
                t = torch.randint(0, xb.shape[1], (len(b),), device=DEV)
                xb = xb[torch.arange(len(b), device=DEV), t][:, None]
            opt.zero_grad()
            lossf(model(xb), yt[b]).backward()
            opt.step()
        sched.step()
        f1 = metrics(yva, predict(model, Xva))["macro_f1"]
        if f1 > best:
            best, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


def save_ckpt(path, model, name, d_in, n_cls, max_len, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), name=name, d_in=d_in, n_cls=n_cls, max_len=max_len, seed=seed), path)


def load_ckpt(path):
    c = torch.load(path, map_location=DEV)
    m = make_model(c["name"], c["d_in"], c["n_cls"], c["max_len"]).to(DEV)
    m.load_state_dict(c["state"])
    return m.eval()


# ------------------------------------------------------------------ uncertainty (block bootstrap)
def block_ids(meta, block_frames=50):
    """Windows overlap heavily -> resample contiguous 5 s blocks (50 frames) instead of single windows."""
    return (meta.seq.astype(str) + "_" + (meta.start.values // block_frames).astype(str)).values


def _scores(cm):
    tp, sup, pr = np.diag(cm), cm.sum(1), cm.sum(0)
    prec = np.divide(tp, pr, out=np.zeros(len(tp)), where=pr > 0)
    rec = np.divide(tp, sup, out=np.zeros(len(tp)), where=sup > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(tp)), where=(prec + rec) > 0)
    return tp.sum() / max(cm.sum(), 1), f1[sup > 0].mean()


def bootstrap_summary(y, preds, blocks, ref="single", n_boot=2000, seed=0, n_cls=4):
    """preds: {model: predicted labels}. Returns 95 % block-bootstrap CIs of acc / macro-F1 per model and
    PAIRED differences (model - ref) with the share of resamples where the model is not better."""
    ub, inv = np.unique(blocks, return_inverse=True)
    cms = {n: np.zeros((len(ub), n_cls, n_cls)) for n in preds}
    for n, p in preds.items():
        np.add.at(cms[n], (inv, y, p), 1)
    rng = np.random.default_rng(seed)
    S = {n: np.zeros((n_boot, 2)) for n in preds}
    for b in range(n_boot):
        s = rng.integers(0, len(ub), len(ub))
        for n in preds:
            S[n][b] = _scores(cms[n][s].sum(0))
    out = {}
    for n in preds:
        pt = _scores(cms[n].sum(0))
        row = {m: dict(point=float(pt[i]), lo=float(np.percentile(S[n][:, i], 2.5)),
                       hi=float(np.percentile(S[n][:, i], 97.5))) for i, m in enumerate(["acc", "macro_f1"])}
        if ref in preds and n != ref:
            for i, m in enumerate(["acc", "macro_f1"]):
                d = S[n][:, i] - S[ref][:, i]
                row[f"diff_{m}"] = dict(mean=float(d.mean()), lo=float(np.percentile(d, 2.5)),
                                        hi=float(np.percentile(d, 97.5)), p_not_better=float((d <= 0).mean()))
        out[n] = row
    return out
