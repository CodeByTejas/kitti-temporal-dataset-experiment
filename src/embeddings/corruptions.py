"""Image-level corruptions (applied on the resized [1,H,W] float tensor in [0,1])."""
from __future__ import annotations

import math

import torch
import torchvision.transforms.functional as TF

PRESETS = {
    "clean": None,
    "blur_mid": ("blur", 3.0), "blur_high": ("blur", 8.0),          # gaussian sigma (px)
    "noise_mid": ("noise", 0.10), "noise_high": ("noise", 0.30),    # gaussian std on [0,1] scale
    "occ_mid": ("occlusion", 0.25), "occ_high": ("occlusion", 0.50)  # black rectangle, area fraction
}


def apply_corruption(img: torch.Tensor, preset: str, gen: torch.Generator) -> torch.Tensor:
    kind, v = PRESETS[preset]
    if kind == "blur":
        k = int(2 * math.ceil(3 * v) + 1)
        return TF.gaussian_blur(img, [k, k], [v, v])
    if kind == "noise":
        return (img + v * torch.randn(img.shape, generator=gen)).clamp(0, 1)
    if kind == "occlusion":
        _, H, W = img.shape
        h, w = max(1, int(H * math.sqrt(v))), max(1, int(W * math.sqrt(v)))
        top = int(torch.randint(0, H - h + 1, (1,), generator=gen))
        left = int(torch.randint(0, W - w + 1, (1,), generator=gen))
        img = img.clone()
        img[..., top:top + h, left:left + w] = 0
        return img
    raise ValueError(preset)


def drop_frames(X, p: float, rng, mode: str = "hold"):
    """Embedding-level frame dropout: a dropped frame is replaced by the previous available one
    (mode="hold") or by the train mean = zeros in normalised space (mode="zero"). X: [N,T,D]."""
    import numpy as np
    X = X.copy()
    drop = rng.random(X.shape[:2]) < p
    drop[:, 0] = False
    if mode == "zero":
        X[drop] = 0.0
        return X
    for t in range(1, X.shape[1]):
        X[:, t] = np.where(drop[:, t, None], X[:, t - 1], X[:, t])
    return X
