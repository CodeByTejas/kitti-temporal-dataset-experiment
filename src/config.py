"""Central configuration + shared CLI arguments for every script."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["stopped", "straight", "left", "right"]  # label ids 0..3 (id 0 unused in 3class mode)

# Sequence-level splits (no window crosses a sequence boundary -> no leakage through overlapping windows).
# Test = 3 urban/residential sequences + seq 01 (highway) as an explicit environment-shift case.
DEFAULT_SPLITS = {
    "train": ["00", "02", "03", "05", "08"],
    "val": ["06", "09"],
    "test": ["07", "10", "01", "04"],
}


@dataclass
class Cfg:
    data_root: Path = ROOT / "data" / "kitti"      # contains dataset/sequences, dataset/poses
    cache_dir: Path = ROOT / "cache"
    results_dir: Path = ROOT / "results"
    encoder: str = "dinov2-small"                  # dinov2-small | dinov2-base | dummy
    feat: str = "clsmean"                          # cls | clsmean | grid (clsmean + 2x6 pooled patch grid)
    cam: str = "image_0"                           # KITTI odometry grayscale left camera
    T: int = 8                                     # frames per window
    gap: int = 2                                   # frame gap inside a window (10 Hz / gap)
    stride: int = 2                                # window stride (in frames)
    # ---- labels (from GT poses, over the whole window span) ----
    mode: str = "3class"                           # 3class: straight/left/right | 4class: + stopped
    stop_disp: float = 0.3                         # m: below -> "stopped" (4class only)
    move_disp: float = 1.5                         # m: moving windows need at least this; slower windows are dropped
    yaw_lo: float = 3.0                            # deg: |yaw| < lo -> straight
    yaw_hi: float = 9.0                            # deg: |yaw| > hi -> left/right; lo..hi = ambiguous -> dropped
    img_h: int = 224
    img_w: int = 742                               # keeps KITTI's ~3.3:1 aspect, multiple of 14
    seed: int = 0
    splits: dict = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_SPLITS)))

    @property
    def seq_root(self) -> Path:
        return self.data_root / "dataset" / "sequences"

    @property
    def pose_root(self) -> Path:
        return self.data_root / "dataset" / "poses"

    @property
    def emb_root(self) -> Path:
        return self.cache_dir / f"{self.encoder}_{self.feat}"

    @property
    def span_s(self) -> float:
        return (self.T - 1) * self.gap * 0.1

    def all_seqs(self):
        return sum(self.splits.values(), [])


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    d = Cfg()
    p.add_argument("--data_root", type=Path, default=d.data_root)
    p.add_argument("--cache_dir", type=Path, default=d.cache_dir)
    p.add_argument("--results_dir", type=Path, default=d.results_dir)
    p.add_argument("--encoder", default=d.encoder, choices=["dinov2-small", "dinov2-base", "dummy"])
    p.add_argument("--feat", default=d.feat, choices=["cls", "clsmean", "grid"])
    p.add_argument("--T", type=int, default=d.T)
    p.add_argument("--gap", type=int, default=d.gap)
    p.add_argument("--stride", type=int, default=d.stride)
    p.add_argument("--mode", default=d.mode, choices=["3class", "4class"])
    p.add_argument("--stop_disp", type=float, default=d.stop_disp)
    p.add_argument("--move_disp", type=float, default=d.move_disp)
    p.add_argument("--yaw_lo", type=float, default=d.yaw_lo)
    p.add_argument("--yaw_hi", type=float, default=d.yaw_hi)
    p.add_argument("--splits_json", type=Path, default=None,
                   help='optional JSON {"train":[..],"val":[..],"test":[..]} to override splits')
    return p


def cfg_from_args(a: argparse.Namespace) -> Cfg:
    c = Cfg(data_root=a.data_root, cache_dir=a.cache_dir, results_dir=a.results_dir,
            encoder=a.encoder, feat=a.feat, T=a.T, gap=a.gap, stride=a.stride, mode=a.mode,
            stop_disp=a.stop_disp, move_disp=a.move_disp, yaw_lo=a.yaw_lo, yaw_hi=a.yaw_hi)
    if a.splits_json:
        c.splits = json.loads(Path(a.splits_json).read_text())
    c.results_dir.mkdir(parents=True, exist_ok=True)
    return c
