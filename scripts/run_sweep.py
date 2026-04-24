"""Run baseline policies and budget sweep in one pass.

Loads Qwen2.5-VL-7B once and iterates through multiple EvalConfigs against the
full 500-sample preproc set. Per-config outputs go under output/<name>/.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vlm_budget_eval import EvalConfig, aggregate, load_model_and_processor, load_samples, run_episode

WORKSPACE = Path("/mnt/ddn/prod-runs/thyun.park/src/vlm_budget_eval")
OUT_ROOT = WORKSPACE / "output"
PREPROC = WORKSPACE / "preproc"

# Shared defaults; per-run dicts override a subset.
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
    save_trace=False,
    max_wasted_before_force=2,
)

RUNS = [
    # Baselines (task 3)
    dict(name="zero_info",     policy="model",         budget=0),
    dict(name="full_info",     policy="full_info",     budget=0),
    dict(name="always_text",   policy="always_text",   budget=6),
    dict(name="always_visual", policy="always_visual", budget=6),
    # Budget sweep (task 4). budget=6 already in output/main_b6/.
    dict(name="sweep_b4",      policy="model",         budget=4),
    dict(name="sweep_b8",      policy="model",         budget=8),
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
        roll.to_csv(OUT_ROOT / "sweep_summary.csv", index=False)
        roll.to_json(OUT_ROOT / "sweep_summary.jsonl", orient="records", lines=True, force_ascii=False)
        print("\n=== Sweep summary ===")
        print(roll.to_string(index=False))


if __name__ == "__main__":
    main()
