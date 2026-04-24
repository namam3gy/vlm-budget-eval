"""Densify the accuracy-budget curve with extra budget values.

Existing runs cover b=0 (zero_info), b=4 (sweep_b4), b=6 (main_b6), b=8
(sweep_b8), and b=∞ (full_info). This script adds b ∈ {1, 2, 3, 5, 7, 10}
using the same model policy, saving per-sample traces so we can use a few of
them in the notebook's side-by-side example.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from budget_eval import (
    EvalConfig,
    aggregate,
    load_model_and_processor,
    load_samples,
    run_episode,
)

WORKSPACE = Path("/mnt/ddn/prod-runs/thyun.park/src/vlm_budget_eval")
OUT_ROOT = WORKSPACE / "output"
PREPROC = WORKSPACE / "preproc"

SHARED = dict(
    preproc_dir=PREPROC,
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="bfloat16",
    max_new_tokens=128,
    temperature=0.0,
    random_seed=42,
    max_eval_rows=-1,
    tile_order="shuffled",
    text_order="natural",
    save_trace=True,
    max_wasted_before_force=2,
    max_forced_attempts=2,
)

BUDGETS = [1, 2, 3, 5, 7, 10]


def save_outputs(pred_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    flat = pred_df.drop(columns=[c for c in ["trace"] if c in pred_df.columns])
    flat.to_parquet(out_dir / "predictions.parquet", index=False)
    flat.to_csv(out_dir / "predictions.csv", index=False)
    pred_df.to_json(out_dir / "predictions.jsonl", orient="records", lines=True, force_ascii=False)
    aggs = aggregate(flat)
    for k, a in aggs.items():
        a.to_csv(out_dir / f"summary_{k}.csv", index=False)
        a.to_json(out_dir / f"summary_{k}.jsonl", orient="records", lines=True, force_ascii=False)
    return aggs["overall"]


def main() -> None:
    samples_df = load_samples(PREPROC)
    print(f"Loaded {len(samples_df)} samples")

    bootstrap = EvalConfig(**SHARED, out_dir=OUT_ROOT, policy="model", budget=1)
    print(f"Loading model {bootstrap.model_id} ...")
    model, processor = load_model_and_processor(bootstrap)

    rows = []
    for b in BUDGETS:
        name = f"sweep_b{b}"
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = EvalConfig(**{**SHARED, "policy": "model", "budget": b, "out_dir": out_dir})
        print(f"\n=== {name}: policy=model budget={b} ===")

        rng = random.Random(cfg.random_seed)
        results = []
        for row in tqdm(samples_df.to_dict(orient="records"), total=len(samples_df)):
            results.append(run_episode(model, processor, row, cfg, rng))
        pred_df = pd.DataFrame(results)
        overall = save_outputs(pred_df, out_dir)
        overall = overall.assign(run_name=name, policy="model", budget=b)
        rows.append(overall)
        print(overall.to_string(index=False))

    roll = pd.concat(rows, ignore_index=True)
    roll.to_csv(OUT_ROOT / "dense_sweep_summary.csv", index=False)
    print("\n=== Dense sweep summary ===")
    print(roll.to_string(index=False))


if __name__ == "__main__":
    main()
