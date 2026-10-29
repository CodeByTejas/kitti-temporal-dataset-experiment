"""Step 7 - where does temporal information help / hurt / not matter? (failure-case analysis)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np
import pandas as pd

from src.config import CLASSES, add_common_args, cfg_from_args
from src.data.viz import entry, plot_strips
from src.data.windows import get_windows

ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--temporal", default="transformer")
ap.add_argument("--per_group", type=int, default=5)
a = ap.parse_args()
cfg = cfg_from_args(a)
res = cfg.results_dir; fig_dir = res / "figures"; (res / "tables").mkdir(exist_ok=True, parents=True); fig_dir.mkdir(exist_ok=True)
_, y, meta, idx = get_windows(cfg, cfg.splits["test"])


def mean_prob(name):
    P = []
    for s in range(a.seeds):
        z = np.load(res / "preds" / f"{name}_s{s}.npy"); z = np.exp(z - z.max(1, keepdims=True))
        P.append(z / z.sum(1, keepdims=True))
    return np.mean(P, 0)


Ps, Pt = mean_prob("single"), mean_prob(a.temporal)
ps, pt = Ps.argmax(1), Pt.argmax(1)
groups = {
    "temporal_helps (single wrong, temporal right)": (ps != y) & (pt == y),
    "temporal_hurts (single right, temporal wrong)": (ps == y) & (pt != y),
    "both_wrong": (ps != y) & (pt != y),
    "temporal_unnecessary (both right, single conf>0.9)": (ps == y) & (pt == y) & (Ps.max(1) > 0.9),
}
summary = {k: int(v.sum()) for k, v in groups.items()}
print(json.dumps(summary, indent=1))
rows = []
for gname, mask in groups.items():
    ids = np.where(mask)[0]
    sc = (Ps[ids, y[ids]] if "helps" in gname else Pt[ids, pt[ids]] if "hurts" in gname else Ps.max(1)[ids])
    top = ids[np.argsort(-sc)][: a.per_group] if "unnec" not in gname else ids[np.argsort(-sc)][: a.per_group]
    ents = [entry(cfg, meta, idx, int(j), f"single:{CLASSES[ps[j]]} temp:{CLASSES[pt[j]]}") for j in top]
    tag = gname.split(" ")[0]
    if ents:
        plot_strips(cfg, ents, fig_dir / f"cases_{tag}.png")
    for j in top:
        rows.append(dict(group=tag, seq=meta.seq[j], start=int(meta.start[j]), true=CLASSES[y[j]],
                         single=CLASSES[ps[j]], temporal=CLASSES[pt[j]], disp=round(float(meta.disp[j]), 2),
                         yaw=round(float(meta.yaw[j]), 2)))
pd.DataFrame(rows).to_csv(res / "tables" / "failure_cases.csv", index=False)
# which true classes does temporal help on?
tab = pd.DataFrame({"true": [CLASSES[i] for i in y], "single_ok": ps == y, "temporal_ok": pt == y})
print(tab.groupby("true")[["single_ok", "temporal_ok"]].mean().round(3))
tab.groupby("true")[["single_ok", "temporal_ok"]].mean().to_csv(res / "tables" / "per_class_recall.csv")
