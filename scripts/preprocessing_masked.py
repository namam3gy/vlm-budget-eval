"""Phase 1d / I13: build a sufficiency-known mini-set with masked images.

Selects 100 samples from preproc/ (50 zi_correct + 50 zi_wrong), then
writes a parallel preproc_masked/ directory with the same 4 tiles per sample
replaced with white PNGs of matching dimensions. Text hints (sentences),
labels, and metadata are unchanged. Runs at multiple budgets compare abstain
rate on these "image-insufficient" samples vs the same samples with original
images — the cleanest test of whether vanilla abstention tracks sufficiency
at all.

Outputs:
  preproc_masked/samples.parquet (and .jsonl)
  preproc_masked/images/tiles/<sample_id>_tile{00..03}.png   (all white)
  preproc_masked/preproc_meta.json
  preproc_masked/cohort_split.csv   -- which samples landed in which cohort
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

WORKSPACE = Path("/mnt/ddn/prod-runs/thyun.park/src/vlm_budget_eval")
SRC = WORKSPACE / "preproc"
DST = WORKSPACE / "preproc_masked"
ZERO_INFO = WORKSPACE / "output" / "zero_info" / "predictions.parquet"

N_PER_COHORT = 50
SEED = 42


def main() -> None:
    if not ZERO_INFO.exists():
        raise SystemExit(f"need {ZERO_INFO} to label cohorts -- run zero_info first")

    samples = pd.read_parquet(SRC / "samples.parquet")
    zero = pd.read_parquet(ZERO_INFO)[["sample_id", "is_correct"]].rename(
        columns={"is_correct": "zi_correct"}
    )
    df = samples.merge(zero, on="sample_id", how="inner")
    print(f"merged: {len(df)} samples")

    rng = random.Random(SEED)
    zi_correct_ids = sorted(df[df["zi_correct"]]["sample_id"].tolist())
    zi_wrong_ids = sorted(df[~df["zi_correct"]]["sample_id"].tolist())
    print(f"zi_correct pool: {len(zi_correct_ids)}, zi_wrong pool: {len(zi_wrong_ids)}")

    rng.shuffle(zi_correct_ids)
    rng.shuffle(zi_wrong_ids)
    chosen_correct = zi_correct_ids[:N_PER_COHORT]
    chosen_wrong = zi_wrong_ids[:N_PER_COHORT]
    chosen = sorted(chosen_correct + chosen_wrong)
    print(f"selected: {N_PER_COHORT} zi_correct + {N_PER_COHORT} zi_wrong = {len(chosen)} samples")

    sub = df[df["sample_id"].isin(chosen)].copy().reset_index(drop=True)
    sub["cohort"] = sub["zi_correct"].map({True: "zi_correct", False: "zi_wrong"})

    DST.mkdir(parents=True, exist_ok=True)
    (DST / "images" / "tiles").mkdir(parents=True, exist_ok=True)

    # Generate white tiles per sample (matching original tile dimensions).
    # tile_rel_paths is updated to point into preproc_masked/images/tiles/.
    new_rel_paths = []
    for row in tqdm(sub.to_dict("records"), total=len(sub), desc="masking tiles"):
        sid = row["sample_id"]
        new_paths = []
        for orig_rel in row["tile_rel_paths"]:
            orig_abs = SRC / orig_rel
            with Image.open(orig_abs) as img:
                size = img.size  # (w, h)
            new_rel = f"images/tiles/{Path(orig_rel).name}"
            white = Image.new("RGB", size, (255, 255, 255))
            white.save(DST / new_rel, format="PNG")
            new_paths.append(new_rel)
        new_rel_paths.append(new_paths)
    sub["tile_rel_paths"] = new_rel_paths

    # Drop the merge column so the schema matches budget_eval.load_samples
    sub = sub.drop(columns=["zi_correct"])

    sub.to_parquet(DST / "samples.parquet", index=False)
    sub.to_json(DST / "samples.jsonl", orient="records", lines=True, force_ascii=False)

    cohort_split = sub[["sample_id", "cohort", "subject"]]
    cohort_split.to_csv(DST / "cohort_split.csv", index=False)

    meta = {
        "source_preproc": str(SRC),
        "n_samples": int(len(sub)),
        "n_per_cohort": N_PER_COHORT,
        "seed": SEED,
        "masking": "all tiles replaced with white PNGs of matching dimensions; text untouched",
        "selected_zi_correct": chosen_correct,
        "selected_zi_wrong": chosen_wrong,
    }
    (DST / "preproc_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"\nwrote {DST}/samples.parquet + {len(sub) * 4} white tile PNGs")
    print(f"cohort breakdown:\n{sub['cohort'].value_counts()}")


if __name__ == "__main__":
    main()
