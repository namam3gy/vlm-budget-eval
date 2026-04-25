#!/usr/bin/env python3
"""Preprocess V*Bench into the budget-eval input format.

V*Bench is a vision-only visual-search benchmark (191 high-res samples,
typically ~2000×1500 pixels). Each sample is a multiple-choice question over
fine-grained visual cues; there is no text hint. We reuse the ScienceQA-shaped
samples.parquet schema but emit `text_sentences=[]`, so any REQUEST_TEXT issued
by the model is registered as a wasted action — itself a clean test of whether
the model still defaults to text on a domain where text content does not exist.

Outputs under `<out_dir>` (default `preproc_vstar/`):
- samples.parquet / samples.jsonl
- images/full/<sample_id>.jpg
- images/tiles/<sample_id>_tile<NN>.png  (tile_grid × tile_grid crops)
- preproc_meta.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image


@dataclass
class PreprocConfig:
    out_dir: Path = Path("preproc_vstar")
    hf_repo: str = "craigwu/vstar_bench"
    split: str = "test"
    tile_grid: int = 2
    max_samples: int = -1  # -1 = no cap
    min_image_side: int = 64
    seed: int = 42


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


# V*Bench `text` field is shaped like:
#   "Question text...?\n(A) optA\n(B) optB\n[(C) optC\n(D) optD\n]Answer with the option's ..."
# We need (question, [(letter, choice_text), ...]).
_CHOICE_RE = re.compile(r"^\(([A-Z])\)\s*(.+?)\s*$")


def parse_text_field(text: str) -> Tuple[str, List[str], List[str]]:
    """Split a V*Bench `text` field into question, choices, choice_letters."""
    lines = [ln.rstrip() for ln in text.split("\n")]
    q_lines: List[str] = []
    choices: List[str] = []
    letters: List[str] = []
    in_choices = False
    for ln in lines:
        if not ln.strip():
            continue
        if ln.startswith("Answer with"):
            break
        m = _CHOICE_RE.match(ln)
        if m:
            in_choices = True
            letters.append(m.group(1))
            choices.append(m.group(2))
        elif not in_choices:
            q_lines.append(ln)
    return " ".join(q_lines).strip(), choices, letters


def main(cfg: PreprocConfig) -> pd.DataFrame:
    out = Path(cfg.out_dir)
    (out / "images" / "full").mkdir(parents=True, exist_ok=True)
    (out / "images" / "tiles").mkdir(parents=True, exist_ok=True)

    print(f"Loading {cfg.hf_repo} split={cfg.split} ...")
    ds = load_dataset(cfg.hf_repo, split=cfg.split)
    print(f"  total examples: {len(ds)}")

    # Snapshot the .jpg image folders. Subsequent calls hit the local cache.
    print(f"Snapshotting image folders from {cfg.hf_repo} ...")
    snap = Path(snapshot_download(
        repo_id=cfg.hf_repo,
        repo_type="dataset",
        allow_patterns=["direct_attributes/*.jpg", "relative_position/*.jpg"],
    ))
    print(f"  cache: {snap}")

    rows: List[dict] = []
    skipped_missing = 0
    skipped_small = 0
    skipped_parse = 0

    for i, ex in enumerate(ds):
        if cfg.max_samples > 0 and len(rows) >= cfg.max_samples:
            break

        rel_img = ex["image"]                    # e.g. "direct_attributes/sa_4690.jpg"
        src_img = snap / rel_img
        if not src_img.exists():
            skipped_missing += 1
            continue

        try:
            img = Image.open(src_img).convert("RGB")
        except Exception:
            skipped_missing += 1
            continue

        if img.size[0] < cfg.min_image_side or img.size[1] < cfg.min_image_side:
            skipped_small += 1
            continue

        question, choices, letters = parse_text_field(ex["text"])
        if not question or not choices or not letters or len(choices) != len(letters):
            skipped_parse += 1
            continue
        gold_letter = str(ex["label"]).strip().upper()
        if gold_letter not in letters:
            skipped_parse += 1
            continue
        gold_idx = letters.index(gold_letter)

        sample_id = f"vstar_{int(ex['question_id']):06d}_{ex['category']}"

        # Save full image (preserve original .jpg for the dataset)
        full_rel = f"images/full/{sample_id}.jpg"
        shutil.copy(src_img, out / full_rel)

        tiles = crop_tiles(img, cfg.tile_grid)
        tile_rel_paths: List[str] = []
        for t_idx, tile in enumerate(tiles):
            rel = f"images/tiles/{sample_id}_tile{t_idx:02d}.png"
            tile.save(out / rel)
            tile_rel_paths.append(rel)

        rows.append({
            "sample_id": sample_id,
            "question": question,
            "choices": choices,
            "choice_letters": letters,
            "answer_idx": int(gold_idx),
            "answer_letter": gold_letter,
            "subject": "vision",                 # 단일 도메인이라 placeholder
            "topic": ex["category"],             # direct_attributes / relative_position
            "category": ex["category"],
            "skill": "",
            "grade": "",
            "text_sentences": [],                # V*Bench has no hints
            "n_sentences": 0,
            "full_image_rel_path": full_rel,
            "tile_rel_paths": tile_rel_paths,
            "tile_labels": [tile_label(t, cfg.tile_grid) for t in range(len(tiles))],
            "tile_grid": cfg.tile_grid,
            "n_tiles": len(tiles),
            "vstar_question_id": int(ex["question_id"]),
            "image_w": int(img.size[0]),
            "image_h": int(img.size[1]),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out / "samples.parquet", index=False)
    df.to_json(out / "samples.jsonl", orient="records", lines=True, force_ascii=False)

    with open(out / "preproc_meta.json", "w") as f:
        json.dump({
            "hf_repo": cfg.hf_repo,
            "split": cfg.split,
            "tile_grid": cfg.tile_grid,
            "max_samples": cfg.max_samples,
            "kept": len(df),
            "skipped_missing": skipped_missing,
            "skipped_small": skipped_small,
            "skipped_parse": skipped_parse,
            "category_counts": df["category"].value_counts().to_dict() if not df.empty else {},
        }, f, indent=2)

    print(f"Kept {len(df)} samples (categories: {df['category'].value_counts().to_dict()}).")
    print(f"Skipped: missing={skipped_missing}, small={skipped_small}, parse={skipped_parse}")
    print(f"Wrote {out / 'samples.parquet'}")
    return df


def parse_args() -> PreprocConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("preproc_vstar"))
    p.add_argument("--hf-repo", default="craigwu/vstar_bench")
    p.add_argument("--split", default="test")
    p.add_argument("--tile-grid", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--min-image-side", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return PreprocConfig(**{k.replace("-", "_"): v for k, v in vars(args).items()})


if __name__ == "__main__":
    main(parse_args())
