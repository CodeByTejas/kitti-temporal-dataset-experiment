"""Step 2 - extract frozen per-frame embeddings (clean + corrupted test variants)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse

import torch

from src.config import add_common_args, cfg_from_args
from src.embeddings.corruptions import PRESETS
from src.embeddings.extract import extract_sequence, get_encoder
import multiprocessing as mp


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--corruptions", default="none",
                    help="'none', 'all' or comma list of presets (blur_mid,noise_high,...); applied to TEST seqs only")
    ap.add_argument("--frac", type=float, default=0.5, help="probability that a given frame is corrupted")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=None, help="debug: only first N frames per sequence")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = cfg_from_args(a)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = get_encoder(cfg, dev)
    kw = dict(bs=a.bs, workers=a.workers, max_frames=a.max_frames, overwrite=a.overwrite)

    for s in cfg.all_seqs():
        extract_sequence(cfg, enc, s, **kw)
    print("clean embeddings ->", cfg.emb_root / "clean")

    if a.corruptions != "none":
        presets = [p for p in PRESETS if p != "clean"] if a.corruptions == "all" else a.corruptions.split(",")
        for p in presets:
            for s in cfg.splits["test"]:
                extract_sequence(cfg, enc, s, preset=p, frac=a.frac, **kw)
        print("corrupted test embeddings ->", cfg.emb_root)


if __name__ == "__main__":
    mp.freeze_support()
    main()