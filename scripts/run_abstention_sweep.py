"""Phase 1d / I14: abstention budget sweep.

`abstain_b0` and `abstain_b6` are already on disk from Phase 1c. This script
runs the same abstain-enabled policy at all the remaining budget points so we
can plot abstain rate (cohort × budget) and watch how the calibration→anti-
calibration transition unfolds with budget. Each run shares the seed and 500
samples used by the Phase 1c runs.

Outputs land under `output/abstain_b{1,2,3,4,5,7,8,10}/` with the standard
schema. The roll-up updates `output/abstention_summary.csv` (overwritten) so
it covers all 10 budgets.
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
    enable_abstain=True,
)

# Skip 0 and 6 — already on disk from Phase 1c run_abstention.py.
SWEEP_BUDGETS = [1, 2, 3, 4, 5, 7, 8, 10]


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
    # carry forward existing b=0 and b=6 if available so the rolled-up summary
    # is complete after this script runs
    existing = OUT_ROOT / "abstention_summary.csv"
    if existing.exists():
        prior = pd.read_csv(existing)
        summary_rows.append(prior)
        print(f"loaded prior summary: {len(prior)} runs")

    for b in SWEEP_BUDGETS:
        name = f"abstain_b{b}"
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = EvalConfig(**{**SHARED, "policy": "model", "budget": b, "out_dir": out_dir})
        print(f"\n=== {name}: policy={cfg.policy} budget={cfg.budget} abstain={cfg.enable_abstain} ===")

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
        # de-dup in case prior csv already had a row for one of these
        roll = roll.drop_duplicates(subset=["run_name"], keep="last").sort_values("budget")
        roll.to_csv(OUT_ROOT / "abstention_summary.csv", index=False)
        roll.to_json(OUT_ROOT / "abstention_summary.jsonl", orient="records", lines=True, force_ascii=False)
        print("\n=== Abstention summary (all budgets) ===")
        print(roll.to_string(index=False))


if __name__ == "__main__":
    main()
