"""Create a tiny fake KITTI-odometry-shaped dataset (for smoke-testing the pipeline WITHOUT the real data).
Motion plan per sequence: cycles of stop / straight / left / right segments with matching poses + images.
Usage: python scripts/make_synthetic_kitti.py --out /tmp/fake_kitti --n_seq 6
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n_seq", type=int, default=6)
ap.add_argument("--frames", type=int, default=360)
a = ap.parse_args()
rng = np.random.default_rng(0)
H, W = 96, 320
tex = rng.random((H * 4, W * 4)).astype(np.float32)
for _ in range(3):  # smooth the texture
    tex = (tex + np.roll(tex, 1, 0) + np.roll(tex, 1, 1) + np.roll(tex, -1, 0) + np.roll(tex, -1, 1)) / 5
tex = (tex - tex.min()) / (tex.max() - tex.min())
plans = {"stop": (0.0, 0.0), "straight": (1.2, 0.0), "left": (1.0, -0.012), "right": (1.0, 0.012)}  # (m/frame, rad/frame)

for s in range(a.n_seq):
    name = f"{s:02d}"
    d = a.out / "dataset" / "sequences" / name / "image_0"; d.mkdir(parents=True, exist_ok=True)
    (a.out / "dataset" / "poses").mkdir(parents=True, exist_ok=True)
    x = z = th = 0.0; poses, kinds = [], list(plans)
    seg = 0
    for f in range(a.frames):
        if f % 45 == 0:
            seg = rng.integers(0, 4)
        v, w = plans[kinds[seg]]
        R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        poses.append(np.hstack([R, [[x], [0], [z]]]).reshape(-1))
        r0 = int(z * 6) % (H * 3); c0 = int(th * 900) % (W * 3)
        img = tex[r0:r0 + H, c0:c0 + W] + 0.01 * rng.standard_normal((H, W))
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(d / f"{f:06d}.png")
        x += v * np.sin(th); z += v * np.cos(th); th += w
    np.savetxt(a.out / "dataset" / "poses" / f"{name}.txt", np.array(poses))
    np.savetxt(a.out / "dataset" / "sequences" / name / "times.txt", np.arange(a.frames) * 0.1)
print("fake dataset at", a.out)
