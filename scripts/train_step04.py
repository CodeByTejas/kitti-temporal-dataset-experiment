"""Step 4 - train single-frame baselines and temporal models on frozen embeddings (+ bootstrap CIs)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import CLASSES, add_common_args, cfg_from_args
from src.data.windows import fit_norm, get_windows
from src.evaluation.train import block_ids, bootstrap_summary, metrics, n_params, predict, save_ckpt, train_model
from src.models.temporal import MODEL_NAMES, make_model

ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--models", default=",".join(MODEL_NAMES))
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=40)
a = ap.parse_args()
cfg = cfg_from_args(a)
res_dir = cfg.results_dir
mu, sd = fit_norm(cfg)                                   # statistics from TRAIN sequences only
np.savez(res_dir / "norm.npz", mu=mu, sd=sd)
Xtr, ytr, mtr, _ = get_windows(cfg, cfg.splits["train"], norm=(mu, sd))
Xva, yva, _, _ = get_windows(cfg, cfg.splits["val"], norm=(mu, sd))
Xte, yte, mte, _ = get_windows(cfg, cfg.splits["test"], norm=(mu, sd))
print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")
print("class counts train", np.bincount(ytr, minlength=4), "val", np.bincount(yva, minlength=4),
      "test", np.bincount(yte, minlength=4))
(res_dir / "preds").mkdir(exist_ok=True)

out, ens = {}, {}
for name in a.models.split(","):
    runs = []
    for seed in range(a.seeds):
        model, best_val = train_model(name, Xtr, ytr, Xva, yva, seed=seed, epochs=a.epochs, max_len=cfg.T)
        logits = predict(model, Xte)
        m = metrics(yte, logits); m["val_macro_f1"] = best_val
        per_seq = {s: metrics(yte[mte.seq.values == s], logits[mte.seq.values == s]) for s in cfg.splits["test"]}
        m["per_seq"] = {s: dict(acc=v["acc"], macro_f1=v["macro_f1"]) for s, v in per_seq.items()}
        if name.startswith("single"):                    # baseline averaged over WHICH frame it sees
            pos = [metrics(yte, predict(model, Xte[:, j:j + 1])) for j in range(cfg.T)]
            m["pos_macro_f1"] = [p["macro_f1"] for p in pos]; m["pos_acc"] = [p["acc"] for p in pos]
        runs.append(m)
        np.save(res_dir / "preds" / f"{name}_s{seed}.npy", logits)
        save_ckpt(res_dir / "checkpoints" / f"{name}_s{seed}.pt", model, name, Xtr.shape[-1], 4, cfg.T, seed)
        print(f"{name:18s} seed{seed} test acc {m['acc']:.3f} macroF1 {m['macro_f1']:.3f} (val F1 {best_val:.3f})")
    agg = {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
           for k in ["acc", "macro_f1", "bal_acc"]}
    out[name] = dict(params=n_params(make_model(name, Xtr.shape[-1], 4, cfg.T)), mean_std=agg,
                     cm_seed0=runs[0]["cm"], runs=runs)
    if name.startswith("single"):
        out[name]["pos_avg_macro_f1"] = float(np.mean([r["pos_macro_f1"] for r in runs]))
        out[name]["pos_avg_acc"] = float(np.mean([r["pos_acc"] for r in runs]))
    P = []
    for s in range(a.seeds):
        z = np.load(res_dir / "preds" / f"{name}_s{s}.npy"); z = np.exp(z - z.max(1, keepdims=True))
        P.append(z / z.sum(1, keepdims=True))
    ens[name] = np.mean(P, 0).argmax(1)

# 95 % block-bootstrap CIs (5 s blocks) and PAIRED differences w.r.t. the single-frame baseline
boot = bootstrap_summary(yte, ens, block_ids(mte), ref="single" if "single" in ens else list(ens)[0])
(res_dir / "main_results.json").write_text(json.dumps(out, indent=1))
(res_dir / "bootstrap.json").write_text(json.dumps(boot, indent=1))

print("\n%-18s %8s  %-13s %-13s | %-22s | %s" % ("model", "params", "acc", "macro-F1", "F1 95% CI (bootstrap)", "dF1 vs single [95% CI]"))
for n, r in out.items():
    b = boot[n]["macro_f1"]; d = boot[n].get("diff_macro_f1")
    ds = f"{d['mean']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}] p={d['p_not_better']:.3f}" if d else "-"
    print("%-18s %8d  %.3f+-%.3f  %.3f+-%.3f | %.3f [%.3f, %.3f]  | %s" % (
        n, r["params"], *r["mean_std"]["acc"], *r["mean_std"]["macro_f1"], b["point"], b["lo"], b["hi"], ds))
for n in out:
    if "pos_avg_macro_f1" in out[n]:
        print(f"{n}: macro-F1 averaged over the frame it is shown = {out[n]['pos_avg_macro_f1']:.3f} "
              f"(last frame only: {out[n]['mean_std']['macro_f1'][0]:.3f})")

fig_dir = res_dir / "figures"; fig_dir.mkdir(exist_ok=True)
names = list(out)
fig, ax = plt.subplots(1, 1, figsize=(max(7, 1.0 * len(names)), 3.8))
w = 0.38
for i, (k, lab) in enumerate([("acc", "accuracy"), ("macro_f1", "macro-F1")]):
    ax.bar(np.arange(len(names)) + i * w, [out[n]["mean_std"][k][0] for n in names], w,
           yerr=[out[n]["mean_std"][k][1] for n in names], label=lab, capsize=3)
ax.set_xticks(np.arange(len(names)) + w / 2); ax.set_xticklabels(names, rotation=25, ha="right"); ax.legend(); ax.set_ylim(0, 1)
ax.set_title("test performance (mean +- std over seeds)")
fig.tight_layout(); fig.savefig(fig_dir / "main_comparison.png", dpi=130); plt.close(fig)
sel = [n for n in ["single", "meanpool", "transformer", "transformer_delta"] if n in out]
fig, axs = plt.subplots(1, len(sel), figsize=(4.0 * len(sel), 3.8), squeeze=False)
for a_, n in zip(axs[0], sel):
    cm = np.array(out[n]["cm_seed0"], dtype=float); cm /= cm.sum(1, keepdims=True).clip(1)
    a_.imshow(cm, vmin=0, vmax=1, cmap="Blues"); a_.set_title(n)
    a_.set_xticks(range(4)); a_.set_xticklabels(CLASSES, rotation=30); a_.set_yticks(range(4)); a_.set_yticklabels(CLASSES)
    for i in range(4):
        for j in range(4):
            a_.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.tight_layout(); fig.savefig(fig_dir / "confusion_matrices.png", dpi=130); plt.close(fig)
