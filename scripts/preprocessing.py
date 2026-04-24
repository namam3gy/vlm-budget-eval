#!/usr/bin/env python3
"""Preprocess ScienceQA into the budget-eval input format.

Downloads `derek-thomas/ScienceQA`, filters to samples that have both an image
and usable text context (hint + lecture), splits the text into sentence-sized
chunks, and crops the image into tile_grid x tile_grid tiles.

Outputs under <out_dir>:
- samples.parquet / samples.jsonl  (one row per sample)
- images/full/<sample_id>.png      (full image, for reference)
- images/tiles/<sample_id>_tile<NN>.png  (per-tile crops)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from datasets import load_dataset
from PIL import Image


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class PreprocConfig:
    out_dir: Path = Path("preproc")
    hf_repo: str = "derek-thomas/ScienceQA"
    split: str = "test"
    tile_grid: int = 2
    max_samples: int = 500
    min_text_chars: int = 60
    min_sentences: int = 2
    min_image_side: int = 64
    seed: int = 42


def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter. Good enough for ScienceQA hint/lecture text."""
    text = (text or "").strip()
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def tile_label(index: int, grid: int) -> str:
    r, c = index // grid, index % grid
    if grid == 2:
        rows = ["top", "bottom"]
        cols = ["left", "right"]
    elif grid == 3:
        rows = ["top", "middle", "bottom"]
        cols = ["left", "middle", "right"]
    else:
        return f"r{r}c{c}"
    return f"{rows[r]}-{cols[c]}"


def crop_tiles(image: Image.Image, grid: int) -> List[Image.Image]:
    """Split image into grid x grid tiles, row-major."""
    W, H = image.size
    x_edges = [(W * i) // grid for i in range(grid + 1)]
    y_edges = [(H * i) // grid for i in range(grid + 1)]
    tiles = []
    for r in range(grid):
        for c in range(grid):
            tiles.append(image.crop((x_edges[c], y_edges[r], x_edges[c + 1], y_edges[r + 1])))
    return tiles


def letter_for_choice(i: int) -> str:
    return chr(ord("A") + i)


def main(cfg: PreprocConfig) -> pd.DataFrame:
    out = Path(cfg.out_dir)
    (out / "images" / "full").mkdir(parents=True, exist_ok=True)
    (out / "images" / "tiles").mkdir(parents=True, exist_ok=True)

    print(f"Loading {cfg.hf_repo} split={cfg.split} ...")
    ds = load_dataset(cfg.hf_repo, split=cfg.split)
    print(f"  total examples: {len(ds)}")

    rows: List[dict] = []
    kept = 0
    skipped_no_image = 0
    skipped_small = 0
    skipped_text = 0

    for i, ex in enumerate(ds):
        if kept >= cfg.max_samples:
            break

        img = ex.get("image")
        if img is None:
            skipped_no_image += 1
            continue
        if img.size[0] < cfg.min_image_side or img.size[1] < cfg.min_image_side:
            skipped_small += 1
            continue

        hint = (ex.get("hint") or "").strip()
        lecture = (ex.get("lecture") or "").strip()
        combined = "\n".join([p for p in (hint, lecture) if p])
        if len(combined) < cfg.min_text_chars:
            skipped_text += 1
            continue

        sentences = split_sentences(combined)
        if len(sentences) < cfg.min_sentences:
            skipped_text += 1
            continue

        sample_id = f"sqa_{i:06d}"
        img_rgb = img.convert("RGB")
        full_path = out / "images" / "full" / f"{sample_id}.png"
        img_rgb.save(full_path)

        tiles = crop_tiles(img_rgb, cfg.tile_grid)
        tile_rel_paths: List[str] = []
        for t_idx, tile in enumerate(tiles):
            rel = f"images/tiles/{sample_id}_tile{t_idx:02d}.png"
            tile.save(out / rel)
            tile_rel_paths.append(rel)

        rows.append({
            "sample_id": sample_id,
            "question": ex["question"],
            "choices": list(ex["choices"]),
            "choice_letters": [letter_for_choice(k) for k in range(len(ex["choices"]))],
            "answer_idx": int(ex["answer"]),
            "answer_letter": letter_for_choice(int(ex["answer"])),
            "subject": ex.get("subject", ""),
            "topic": ex.get("topic", ""),
            "category": ex.get("category", ""),
            "skill": ex.get("skill", ""),
            "grade": ex.get("grade", ""),
            "text_sentences": sentences,
            "n_sentences": len(sentences),
            "full_image_rel_path": str(full_path.relative_to(out)),
            "tile_rel_paths": tile_rel_paths,
            "tile_labels": [tile_label(t, cfg.tile_grid) for t in range(len(tiles))],
            "tile_grid": cfg.tile_grid,
            "n_tiles": len(tiles),
        })
        kept += 1

    df = pd.DataFrame(rows)
    df.to_parquet(out / "samples.parquet", index=False)
    df.to_json(out / "samples.jsonl", orient="records", lines=True, force_ascii=False)

    with open(out / "preproc_meta.json", "w") as f:
        json.dump({
            "hf_repo": cfg.hf_repo,
            "split": cfg.split,
            "tile_grid": cfg.tile_grid,
            "max_samples": cfg.max_samples,
            "kept": kept,
            "skipped_no_image": skipped_no_image,
            "skipped_small": skipped_small,
            "skipped_text": skipped_text,
        }, f, indent=2)

    print(f"Kept {kept} samples. "
          f"Skipped: no_image={skipped_no_image}, small={skipped_small}, thin_text={skipped_text}")
    print(f"Wrote {out / 'samples.parquet'}")
    return df


def parse_args() -> PreprocConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("preproc"))
    p.add_argument("--hf-repo", default="derek-thomas/ScienceQA")
    p.add_argument("--split", default="test")
    p.add_argument("--tile-grid", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--min-text-chars", type=int, default=60)
    p.add_argument("--min-sentences", type=int, default=2)
    p.add_argument("--min-image-side", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return PreprocConfig(**{k.replace("-", "_"): v for k, v in vars(args).items()})


if __name__ == "__main__":
    main(parse_args())
