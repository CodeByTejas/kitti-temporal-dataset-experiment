"""Small single-frame / temporal heads that sit on top of frozen frame embeddings.

All models take x: [B, k, D] (k = number of observed frames, most recent last).
  single             : MLP on the LAST frame only                     (baseline)
  meanpool           : MLP on the mean over frames                    (more frames, NO order)
  gru                : GRU over frames                                (order-aware)
  transformer        : CLS-token Transformer + learned positions      (order-aware, attention viz)
  transformer_nopos  : same without positions = permutation-invariant (control: set, not sequence)
  single_wide        : single-frame MLP widened to the Transformer's parameter count (capacity control)
  gru_delta / transformer_delta : same models fed [E_t ; 3*(E_t - E_{t-1})] -> explicit frame-to-frame change
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SingleFrameMLP(nn.Module):
    def __init__(self, d_in, n_cls, hidden=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(p), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(hidden, n_cls))

    def forward(self, x):
        return self.net(x[:, -1])


class MeanPoolMLP(SingleFrameMLP):
    def forward(self, x):
        return self.net(x.mean(1))


class GRUModel(nn.Module):
    def __init__(self, d_in, n_cls, d=128, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(p), nn.Linear(d_in, d), nn.GELU())
        self.gru = nn.GRU(d, d, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(d, n_cls))

    def forward(self, x):
        _, h = self.gru(self.proj(x))
        return self.head(h[-1])


class Block(nn.Module):
    def __init__(self, d, heads, p):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(p)
        self.last_attn = None

    def forward(self, x, keep_attn=False):
        h = self.n1(x)
        a, w = self.attn(h, h, h, need_weights=keep_attn, average_attn_weights=False)
        self.last_attn = w if keep_attn else None
        x = x + self.drop(a)
        return x + self.drop(self.mlp(self.n2(x)))


class TemporalTransformer(nn.Module):
    def __init__(self, d_in, n_cls, d=128, heads=4, layers=2, max_len=16, p=0.1, use_pos=True):
        super().__init__()
        self.use_pos = use_pos
        self.proj = nn.Sequential(nn.Dropout(p), nn.Linear(d_in, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads, p) for _ in range(layers)])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, n_cls)

    def forward(self, x, return_attn=False):
        t = self.proj(x)
        if self.use_pos:
            t = t + self.pos[:, : t.shape[1]]
        tok = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)
        for b in self.blocks:
            tok = b(tok, keep_attn=return_attn)
        out = self.head(self.norm(tok[:, 0]))
        return (out, [b.last_attn for b in self.blocks]) if return_attn else out


class DeltaInput(nn.Module):
    """Concatenate each frame embedding with its change w.r.t. the previous frame (first delta = 0)."""

    def __init__(self, inner, scale=3.0):
        super().__init__()
        self.inner, self.scale = inner, scale

    def forward(self, x, **kw):
        d = torch.zeros_like(x)
        d[:, 1:] = x[:, 1:] - x[:, :-1]
        return self.inner(torch.cat([x, self.scale * d], -1), **kw)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


MODEL_NAMES = ["single", "single_wide", "meanpool", "gru", "transformer", "transformer_nopos",
               "gru_delta", "transformer_delta"]


def make_model(name, d_in, n_cls, max_len=16):
    if name.endswith("_delta"):
        return DeltaInput(make_model(name[:-6], 2 * d_in, n_cls, max_len))
    if name == "single":
        return SingleFrameMLP(d_in, n_cls)
    if name == "single_wide":
        target = count_params(TemporalTransformer(d_in, n_cls, max_len=max_len))
        return SingleFrameMLP(d_in, n_cls, hidden=max(16, (target - n_cls) // (d_in + 1 + n_cls)))
    if name == "meanpool":
        return MeanPoolMLP(d_in, n_cls)
    if name == "gru":
        return GRUModel(d_in, n_cls)
    if name == "transformer":
        return TemporalTransformer(d_in, n_cls, max_len=max_len)
    if name == "transformer_nopos":
        return TemporalTransformer(d_in, n_cls, max_len=max_len, use_pos=False)
    raise ValueError(name)
