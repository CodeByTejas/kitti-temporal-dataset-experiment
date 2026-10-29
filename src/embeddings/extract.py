"""Frozen per-frame encoders + cached embedding extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import Cfg
from src.data.kitti import frame_paths
from src.data.windows import variant_name
from src.embeddings.corruptions import PRESETS, apply_corruption


class FrameDS(Dataset):
    def __init__(self, cfg: Cfg, seq, paths, preset="clean", frac=1.0, seed=0):
        self.cfg, self.seq, self.paths, self.preset, self.frac, self.seed = cfg, seq, paths, preset, frac, seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("L").resize((self.cfg.img_w, self.cfg.img_h), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).unsqueeze(0)
        if self.preset != "clean":  # deterministic per (seq, frame): reproducible corruption
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(self.seq) * 100_003 + i)
            if torch.rand(1, generator=g).item() < self.frac:
                x = apply_corruption(x, self.preset, g)
        return x


class DinoV2Encoder:
    HF = {"dinov2-small": "facebook/dinov2-small", "dinov2-base": "facebook/dinov2-base"}

    def __init__(self, name, feat, device, gh=16, gw=53):
        from transformers import AutoModel
        self.device, self.feat, self.gh, self.gw = device, feat, gh, gw
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.m = AutoModel.from_pretrained(self.HF[name]).to(device).to(self.dtype).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, x):  # x: [B,1,H,W] in [0,1]
        x = x.to(self.device).repeat(1, 3, 1, 1)
        x = ((x - self.mean) / self.std).to(self.dtype)
        h = self.m(pixel_values=x).last_hidden_state
        cls, pm = h[:, 0].float(), h[:, 1:].float().mean(1)
        if self.feat == "cls":
            f = cls
        elif self.feat == "clsmean":
            f = torch.cat([cls, pm], -1)
        else:  # grid: keep coarse spatial layout (2x6 cells) so horizontal motion / parallax survives pooling
            B = h.shape[0]
            g = h[:, 1:].float().reshape(B, self.gh, self.gw, -1).permute(0, 3, 1, 2)
            g = torch.nn.functional.adaptive_avg_pool2d(g, (2, 6)).flatten(2).permute(0, 2, 1).flatten(1)
            f = torch.cat([cls, pm, g], -1)
        return f.cpu()


class DummyEncoder:
    """Tiny deterministic encoder (pooled pixels -> random projection). Only for smoke tests."""

    def __init__(self, *_):
        g = torch.Generator().manual_seed(0)
        self.W = torch.randn(8 * 24, 64, generator=g) / 12.0

    @torch.no_grad()
    def __call__(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(x, (8, 24)).flatten(1) @ self.W


def get_encoder(cfg: Cfg, device):
    if cfg.encoder == "dummy":
        return DummyEncoder()
    return DinoV2Encoder(cfg.encoder, cfg.feat, device, cfg.img_h // 14, cfg.img_w // 14)


def extract_sequence(cfg, enc, seq, preset="clean", frac=1.0, seed=0, bs=32, workers=4,
                     max_frames=None, overwrite=False):
    out = cfg.emb_root / variant_name(preset, frac) / f"{seq}.npy"
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = frame_paths(cfg, seq)[:max_frames]
    dl = DataLoader(FrameDS(cfg, seq, paths, preset, frac, seed), batch_size=bs, num_workers=workers)
    feats = [enc(x) for x in tqdm(dl, desc=f"{seq}/{variant_name(preset, frac)}", leave=False)]
    np.save(out, torch.cat(feats).numpy().astype(np.float16))
    return out
