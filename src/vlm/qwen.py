"""Thin wrapper around Qwen2-VL (open-source VLM, no fine-tuning, greedy decoding)."""
from __future__ import annotations

import re

import torch
from PIL import Image

from src.config import CLASSES, Cfg

KEYWORDS = {"stopped": ["stopped", "stationary", "not moving", "standing still", "stop"],
            "straight": ["straight", "forward"], "left": ["left"], "right": ["right"]}


def parse_label(text: str):
    """First motion keyword mentioned in the answer -> class id (None if nothing matches)."""
    t, best = text.lower(), (10 ** 9, None)
    for ci, c in enumerate(CLASSES):
        for kw in KEYWORDS[c]:
            m = re.search(r"\b" + re.escape(kw) + r"\b", t)
            if m and m.start() < best[0]:
                best = (m.start(), ci)
    return best[1]


class QwenVL:
    def __init__(self, name="Qwen/Qwen2-VL-2B-Instruct", img_size=(644, 196)):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(name, torch_dtype=dtype, device_map="auto").eval()
        self.img_size = img_size  # multiples of 28 -> ~161 visual tokens per frame

    def load(self, cfg: Cfg, seq, i):
        im = Image.open(cfg.seq_root / seq / cfg.cam / f"{int(i):06d}.png").convert("RGB")
        return im.resize(self.img_size, Image.BILINEAR)

    @torch.no_grad()
    def ask(self, images, prompt, frame_labels=None, max_new_tokens=48):
        content = []
        for k, _ in enumerate(images):
            if frame_labels:
                content.append({"type": "text", "text": frame_labels[k]})
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        text = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
