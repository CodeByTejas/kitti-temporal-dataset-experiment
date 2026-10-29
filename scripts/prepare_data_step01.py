"""Step 1 - load sequences, verify ordering/integrity, dataset statistics, frame strips."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json

import multiprocessing as mp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CLASSES, add_common_args, cfg_from_args
from src.data.kitti import build_windows, verify_sequence
from src.data.viz import entry, plot_strips


def main():
    ap = add_common_args(argparse.ArgumentParser())
    cfg = cfg_from_args(ap.parse_args())
    fig_dir, tab_dir = cfg.results_dir / "figures", cfg.results_dir / "tables"
    fig_dir.mkdir(exist_ok=True, parents=True); tab_dir.mkdir(exist_ok=True, parents=True)

    # ---- 1. integrity -------------------------------------------------------------------------
    rows = [verify_sequence(cfg, s) for s in cfg.all_seqs()]
    integ = pd.DataFrame(rows)
    integ.to_csv(tab_dir / "integrity.csv", index=False)
    print(integ.to_string(index=False))
    if not integ.ok.all():
        print("\n!! integrity problems in:", integ[~integ.ok].seq.tolist())

    # ---- 2. windows + stats --------------------------------------------------------------------
    split_of = {s: k for k, v in cfg.splits.items() for s in v}
    dfs, idxs, alls = [], [], []
    for s in cfg.all_seqs():
        df, idx = build_windows(cfg, s)
        df["split"] = split_of[s]
        dfs.append(df); idxs.append(idx)
        alls.append(build_windows(cfg, s, filter=False)[0])
    W = pd.concat(dfs, ignore_index=True); IDX = np.concatenate(idxs)
    A = pd.concat(alls, ignore_index=True)
    print(f"\nmode={cfg.mode}: kept {len(W)} of {len(A)} windows ({len(W) / len(A):.1%}); "
        f"dropped = ambiguous dead zone (|yaw| {cfg.yaw_lo}-{cfg.yaw_hi} deg, slow motion {cfg.stop_disp}-{cfg.move_disp} m"
        f"{'' if cfg.mode == '4class' else ', or stopped'})")
    W["cls"] = W.label.map(dict(enumerate(CLASSES)))

    per_seq = W.pivot_table(index="seq", columns="cls", values="label", aggfunc="count", fill_value=0)
    per_seq = per_seq.reindex(columns=CLASSES, fill_value=0)
    per_seq["n_windows"] = per_seq.sum(axis=1)
    per_seq["n_frames"] = integ.set_index("seq").n_frames
    per_seq["split"] = per_seq.index.map(split_of)
    per_seq.to_csv(tab_dir / "windows_per_sequence.csv")
    print("\nwindows per sequence / class:\n", per_seq.to_string())

    split_cls = W.pivot_table(index="split", columns="cls", values="label", aggfunc="count", fill_value=0)
    split_cls = split_cls.reindex(columns=CLASSES, fill_value=0)
    split_cls.to_csv(tab_dir / "class_counts_per_split.csv")
    print("\nclass counts per split:\n", split_cls.to_string())
    for sp, r in split_cls.iterrows():
        for c in CLASSES:
            if c == "stopped" and cfg.mode == "3class":
                continue
            if r[c] < 50:
                print(f"!! WARNING: split '{sp}' has only {r[c]} '{c}' windows - consider --splits_json / thresholds")

    summary = dict(window=dict(T=cfg.T, gap=cfg.gap, stride=cfg.stride, span_s=cfg.span_s),
                thresholds=dict(mode=cfg.mode, stop_disp_m=cfg.stop_disp, move_disp_m=cfg.move_disp,
                            yaw_lo_deg=cfg.yaw_lo, yaw_hi_deg=cfg.yaw_hi),
                windows_before_filter=int(len(A)), windows_after_filter=int(len(W)),
                total_frames=int(integ.n_frames.sum()), total_windows=int(len(W)))
    (tab_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- 3. plots --------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    split_cls.plot.bar(ax=ax[0], rot=0); ax[0].set_title("windows per class and split")
    ax[1].hist(A.disp, bins=60); ax[1].axvline(cfg.stop_disp, c="r", ls="--"); ax[1].axvline(cfg.move_disp, c="r", ls="--")
    ax[1].set_title(f"displacement over window ({cfg.span_s:.1f}s) [m]")
    ax[2].hist(A.yaw, bins=80)
    for v in (cfg.yaw_lo, cfg.yaw_hi, -cfg.yaw_lo, -cfg.yaw_hi):
        ax[2].axvline(v, c="r", ls="--")
    ax[2].set_title("yaw change over window [deg] (+ = right)"); ax[2].set_yscale("log")
    fig.tight_layout(); fig.savefig(fig_dir / "dataset_stats.png", dpi=130); plt.close(fig)

    # ---- 4. temporal frame strips (2 random examples per class) ----------------------------------
    rng = np.random.default_rng(0)
    ents = []
    for c in range(4):
        cand = np.where(W.label.values == c)[0]
        for j in rng.choice(cand, size=min(2, len(cand)), replace=False):
            ents.append(entry(cfg, W, IDX, int(j)))
    plot_strips(cfg, ents, fig_dir / "strips_by_class.png")
    print(f"\nsaved figures to {fig_dir}")


if __name__ == "__main__":
    mp.freeze_support()
    main()