"""Step 5 - controlled temporal experiments:
  progressive observation, temporal order (shuffle / reverse), frame ablation, corruption + frame drops, per-sequence."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import CLASSES, add_common_args, cfg_from_args
from src.data.windows import get_windows, load_norm, variant_name
from src.embeddings.corruptions import PRESETS, drop_frames
from src.evaluation.train import load_ckpt, metrics, predict, train_model

ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--exps", default="progressive,order,ablation,corruption,persequence")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--frac", type=float, default=0.5, help="corruption frac used in 02_extract_embeddings.py")
ap.add_argument("--drop_rates", default="0.25,0.5")
ap.add_argument("--prog_models", default="gru,transformer,transformer_nopos,meanpool,transformer_delta")
a = ap.parse_args()
cfg = cfg_from_args(a)
exps = a.exps.split(",")
res = cfg.results_dir; fig_dir = res / "figures"; exp_dir = res / "experiments"
fig_dir.mkdir(exist_ok=True); exp_dir.mkdir(exist_ok=True)
norm = load_norm(cfg)
Xtr, ytr, _, _ = get_windows(cfg, cfg.splits["train"], norm=norm)
Xva, yva, _, _ = get_windows(cfg, cfg.splits["val"], norm=norm)
Xte, yte, mte, _ = get_windows(cfg, cfg.splits["test"], norm=norm)
main = json.loads((res / "main_results.json").read_text()) if (res / "main_results.json").exists() else {}
COL = dict(single="tab:gray", single_wide="silver", meanpool="tab:orange", gru="tab:green", transformer="tab:blue",
           transformer_nopos="tab:purple", gru_delta="tab:olive", transformer_delta="tab:red")


def avail(names):
    return [n for n in names if all((res / "checkpoints" / f"{n}_s{s}.pt").exists() for s in range(a.seeds))]


def ckpts(name):
    return [load_ckpt(res / "checkpoints" / f"{name}_s{s}.pt") for s in range(a.seeds)]


def ms(v):
    return float(np.mean(v)), float(np.std(v))


def lr_swap(cms):
    cm = np.sum(cms, 0)
    return float((cm[2, 3] + cm[3, 2]) / max(cm[2].sum() + cm[3].sum(), 1))


# ----------------------------------------------------------------------------------------------
if "progressive" in exps:
    ks = [k for k in [1, 2, 3, 4, 6, 8, 12, 16] if k <= cfg.T]
    names = a.prog_models.split(",")
    out = {n: {} for n in names}
    for n in names:
        for k in ks:
            f1s, accs = [], []
            for s in range(a.seeds):
                m, _ = train_model(n, Xtr, ytr, Xva, yva, seed=s, k=k, epochs=a.epochs, max_len=cfg.T)
                r = metrics(yte, predict(m, Xte, k=k)); f1s.append(r["macro_f1"]); accs.append(r["acc"])
            out[n][k] = dict(macro_f1=ms(f1s), acc=ms(accs))
            print(f"[progressive] {n:18s} k={k:2d} F1 {out[n][k]['macro_f1'][0]:.3f} acc {out[n][k]['acc'][0]:.3f}")
    (exp_dir / "progressive.json").write_text(json.dumps(out, indent=1))
    fig, ax = plt.subplots(figsize=(6.8, 4))
    for n in names:
        mu_ = np.array([out[n][k]["macro_f1"][0] for k in ks]); sd_ = np.array([out[n][k]["macro_f1"][1] for k in ks])
        ax.plot(ks, mu_, "-o", c=COL.get(n), label=n); ax.fill_between(ks, mu_ - sd_, mu_ + sd_, color=COL.get(n), alpha=.15)
    if "single" in main:
        b = main["single"]["mean_std"]["macro_f1"]
        ax.axhline(b[0], c="k", ls="--", label="single frame (last)"); ax.axhspan(b[0] - b[1], b[0] + b[1], color="k", alpha=.08)
        if "pos_avg_macro_f1" in main["single"]:
            ax.axhline(main["single"]["pos_avg_macro_f1"], c="k", ls=":", label="single frame (avg over position)")
    ax.set_xlabel(f"# observed frames k (most recent k of {cfg.T}; spacing {cfg.gap * .1:.1f}s)"); ax.set_ylabel("test macro-F1")
    ax.set_title("Progressive observation"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(fig_dir / "exp_progressive.png", dpi=130); plt.close(fig)

# ----------------------------------------------------------------------------------------------
if "order" in exps:
    MODELS = avail(["single", "single_wide", "meanpool", "gru", "transformer", "transformer_nopos", "gru_delta", "transformer_delta"])
    out, rng = {}, np.random.default_rng(0)
    perm = np.argsort(rng.random(Xte.shape[:2]), 1)
    conds = {"correct": Xte, "shuffled": np.take_along_axis(Xte, perm[:, :, None], 1),
             "reversed": np.ascontiguousarray(Xte[:, ::-1])}
    for n in MODELS:
        ms_ = ckpts(n); out[n] = {}
        for c, X in conds.items():
            runs = [metrics(yte, predict(m, X)) for m in ms_]
            out[n][c] = dict(macro_f1=ms([r["macro_f1"] for r in runs]), acc=ms([r["acc"] for r in runs]),
                             lr_swap=lr_swap([r["cm"] for r in runs]), cm_seed0=runs[0]["cm"])
        print(f"[order] {n:18s} " + "  ".join(f"{c}: F1 {out[n][c]['macro_f1'][0]:.3f}" for c in conds)
              + f"  | L<->R swap correct {out[n]['correct']['lr_swap']:.3f} reversed {out[n]['reversed']['lr_swap']:.3f}")
    (exp_dir / "order.json").write_text(json.dumps(out, indent=1))
    fig, ax = plt.subplots(figsize=(max(7.5, 1.1 * len(MODELS)), 4)); w = .26
    for i, c in enumerate(conds):
        ax.bar(np.arange(len(MODELS)) + i * w, [out[n][c]["macro_f1"][0] for n in MODELS], w,
               yerr=[out[n][c]["macro_f1"][1] for n in MODELS], label=c, capsize=2)
    ax.set_xticks(np.arange(len(MODELS)) + w); ax.set_xticklabels(MODELS, rotation=25, ha="right")
    ax.set_ylabel("test macro-F1"); ax.set_title("Temporal order: correct vs shuffled vs reversed"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "exp_order.png", dpi=130); plt.close(fig)
    for n in [m for m in ["transformer", "transformer_delta", "gru_delta"] if m in out]:
        cm = np.array(out[n]["reversed"]["cm_seed0"], float); cm /= cm.sum(1, keepdims=True).clip(1)
        fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=30); ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
        ax.set_title(f"{n}, time-REVERSED input"); ax.set_xlabel("predicted"); ax.set_ylabel("true")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.tight_layout(); fig.savefig(fig_dir / f"exp_order_reversed_cm_{n}.png", dpi=130); plt.close(fig)

# ----------------------------------------------------------------------------------------------
if "ablation" in exps:
    names = avail(["single", "meanpool", "gru", "transformer", "gru_delta", "transformer_delta"])
    out = {}
    t = np.arange(cfg.T) * cfg.gap * 0.1
    for n in names:
        ms_ = ckpts(n)
        base = np.mean([metrics(yte, predict(m, Xte))["macro_f1"] for m in ms_])
        out[n] = {"clean_f1": float(base), "drop_by_position": []}
        for j in range(cfg.T):                                   # replace frame j by the train mean (zeros)
            X2 = Xte.copy(); X2[:, j] = 0.0
            f = np.mean([metrics(yte, predict(m, X2))["macro_f1"] for m in ms_])
            out[n]["drop_by_position"].append(float(base - f))
        print(f"[ablation] {n:18s} F1 drop when frame j removed: " + " ".join(f"{d:+.3f}" for d in out[n]["drop_by_position"]))
    (exp_dir / "frame_ablation.json").write_text(json.dumps(out, indent=1))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for n in names:
        ax.plot(t, out[n]["drop_by_position"], "-o", c=COL.get(n), label=n)
    ax.axhline(0, c="k", lw=.5); ax.set_xlabel("time of removed frame in window [s]"); ax.set_ylabel("macro-F1 drop")
    ax.set_title("Frame ablation: which observations does each model rely on?"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(fig_dir / "exp_frame_ablation.png", dpi=130); plt.close(fig)

# ----------------------------------------------------------------------------------------------
if "corruption" in exps:
    names = avail(["single", "meanpool", "gru", "transformer", "transformer_delta"])
    conds = {"clean": Xte}
    for p in PRESETS:
        if p == "clean":
            continue
        try:
            conds[p], _, _, _ = get_windows(cfg, cfg.splits["test"], variant=variant_name(p, a.frac), norm=norm)
        except FileNotFoundError:
            print(f"[corruption] skipping {p} (run 02_extract_embeddings.py --corruptions all)")
    rng = np.random.default_rng(1)
    for r in map(float, a.drop_rates.split(",")):
        conds[f"drophold_{r:g}"] = drop_frames(Xte, r, rng, "hold")
        conds[f"dropzero_{r:g}"] = drop_frames(Xte, r, rng, "zero")
    out = {}
    for n in names:
        ms_ = ckpts(n); out[n] = {}
        for c, X in conds.items():
            runs = [metrics(yte, predict(m, X)) for m in ms_]
            out[n][c] = dict(macro_f1=ms([r["macro_f1"] for r in runs]), acc=ms([r["acc"] for r in runs]))
        print(f"[corruption] {n:18s} " + " ".join(f"{c}:{out[n][c]['macro_f1'][0]:.2f}" for c in conds))
    (exp_dir / "corruption.json").write_text(json.dumps(out, indent=1))
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(conds)), 4.2)); w = .8 / len(names)
    for i, n in enumerate(names):
        ax.bar(np.arange(len(conds)) + i * w, [out[n][c]["macro_f1"][0] for c in conds], w,
               yerr=[out[n][c]["macro_f1"][1] for c in conds], label=n, color=COL.get(n), capsize=2)
    ax.set_xticks(np.arange(len(conds)) + .4 - w / 2); ax.set_xticklabels(list(conds), rotation=35, ha="right")
    ax.set_ylabel("test macro-F1"); ax.legend(fontsize=7)
    ax.set_title(f"Robustness: {a.frac:.0%} of frames corrupted / dropped (trained on clean only)")
    fig.tight_layout(); fig.savefig(fig_dir / "exp_corruption.png", dpi=130); plt.close(fig)

# ----------------------------------------------------------------------------------------------
if "persequence" in exps:
    names = avail(["single", "single_wide", "meanpool", "gru", "transformer", "transformer_delta"])
    seqs = cfg.splits["test"]
    per = {n: {s: ms([r["per_seq"][s]["acc"] for r in main[n]["runs"]]) for s in seqs} for n in names if n in main}
    seq_arr = mte.seq.values
    summ = {}
    for tag, mask in [("excl_seq01", seq_arr != "01"), ("only_seq01", seq_arr == "01")]:
        if not mask.any():
            continue
        summ[tag] = {}
        for n in names:
            rs = [metrics(yte[mask], np.load(res / "preds" / f"{n}_s{s}.npy")[mask]) for s in range(a.seeds)]
            summ[tag][n] = dict(acc=ms([r["acc"] for r in rs]), macro_f1=ms([r["macro_f1"] for r in rs]))
        print(f"[{tag}] " + " ".join(f"{n}:{summ[tag][n]['macro_f1'][0]:.3f}" for n in names))
    (exp_dir / "per_sequence.json").write_text(json.dumps(dict(per_seq_acc=per, **summ), indent=1))
    fig, ax = plt.subplots(figsize=(7.5, 3.8)); w = .8 / max(len(per), 1)
    for i, n in enumerate(per):
        ax.bar(np.arange(len(seqs)) + i * w, [per[n][s][0] for s in seqs], w, yerr=[per[n][s][1] for s in seqs],
               label=n, color=COL.get(n), capsize=2)
    ax.set_xticks(np.arange(len(seqs)) + .4 - w / 2); ax.set_xticklabels([f"seq {s}" for s in seqs]); ax.set_ylim(0, 1)
    ax.set_ylabel("test accuracy"); ax.set_title("Unseen sequences / environments (seq 01 = highway)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "exp_per_sequence.png", dpi=130); plt.close(fig)
print("experiment results ->", exp_dir)
