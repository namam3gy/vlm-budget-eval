"""Phase 2.0c: V*Bench baseline runs (3-action vocab, vanilla policy).

Loads Qwen2.5-VL-7B once and iterates 5 configurations against the full 191-
sample V*Bench preproc. Per-config outputs go under `output/<name>/` with the
standard schema. Roll-up at `output/vstar_baseline_summary.csv`.

Five runs:
- vstar_zero_info          : b=0 (no info, model is forced to commit on prior knowledge)
- vstar_b2                 : b=2 (low-budget model policy)
- vstar_b4                 : b=4 (== n_tiles; model picks order)
- vstar_always_visual_b4   : auto-reveal every tile (deterministic)
- vstar_full_info          : reveal all tiles up-front, single forced-answer turn
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vlm_budget_eval import (
    EvalConfig,
    aggregate,
    load_model_and_processor,
    load_samples,
    run_episode,
)

WORKSPACE = Path("/mnt/ddn/prod-runs/thyun.park/src/vlm_budget_eval")
OUT_ROOT = WORKSPACE / "output"
PREPROC = WORKSPACE / "preproc_vstar"

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
    # 1024 x 28 x 28 = 802,816 pixels per tile (~1024 vision tokens). With
    # 2x2 cropped V*Bench tiles (typically ~1000x750) this is just below the
    # native size but caps memory predictably on a shared GPU.
    max_pixels=1024 * 28 * 28,
)

RUNS = [
    dict(name="vstar_zero_info",        policy="model",         budget=0),
    dict(name="vstar_b2",               policy="model",         budget=2),
    dict(name="vstar_b4",               policy="model",         budget=4),
    dict(name="vstar_always_visual_b4", policy="always_visual", budget=4),
    dict(name="vstar_full_info",        policy="full_info",     budget=0),
]


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
    print(f"Loaded {len(samples_df)} samples from {PREPROC}")

    bootstrap = EvalConfig(**SHARED, out_dir=OUT_ROOT, policy="model", budget=0)
    print(f"Loading model {bootstrap.model_id} ...")
    model, processor = load_model_and_processor(bootstrap)

    summary_rows = []
    for run in RUNS:
        name = run["name"]
        overrides = {k: v for k, v in run.items() if k != "name"}
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = EvalConfig(**{**SHARED, **overrides, "out_dir": out_dir})

        # Resume: skip if a complete predictions.jsonl is already on disk.
        existing = out_dir / "predictions.jsonl"
        if existing.exists():
            with existing.open() as f:
                n_done = sum(1 for _ in f)
            if n_done == len(samples_df):
                print(f"\n=== {name}: SKIP (already have {n_done}/{len(samples_df)} predictions) ===")
                ov = pd.read_csv(out_dir / "summary_overall.csv").assign(
                    run_name=name, policy=cfg.policy, budget=cfg.budget)
                summary_rows.append(ov)
                continue

        print(f"\n=== {name}: policy={cfg.policy} budget={cfg.budget} ===")
        rng = random.Random(cfg.random_seed)
        results = []
        for row in tqdm(samples_df.to_dict(orient="records"), total=len(samples_df)):
            results.append(run_episode(model, processor, row, cfg, rng))

        pred_df = pd.DataFrame(results)
        overall = save_outputs(pred_df, out_dir)
        overall = overall.assign(run_name=name, policy=cfg.policy, budget=cfg.budget)
        summary_rows.append(overall)
        print(overall.to_string(index=False))

    if summary_rows:
        roll = pd.concat(summary_rows, ignore_index=True)
        roll.to_csv(OUT_ROOT / "vstar_baseline_summary.csv", index=False)
        roll.to_json(OUT_ROOT / "vstar_baseline_summary.jsonl",
                     orient="records", lines=True, force_ascii=False)
        print("\n=== V*Bench baseline summary ===")
        print(roll.to_string(index=False))


if __name__ == "__main__":
    main()
