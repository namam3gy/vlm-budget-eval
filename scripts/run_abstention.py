"""Phase 1c probe: how does the vanilla model use ABSTAIN?

Adds the ABSTAIN action to the system prompt and runs the model at two budget
points to see how it changes behaviour:

  - `abstain_b0`: budget=0. The model is forced to either answer (with no info)
    or abstain. If the model has any "I don't know" instinct, abstention rate
    here should be high. The naive comparison is `zero_info` (forced_answer
    rate 0.866; n=500), where 100% of force-answered samples are guesses.

  - `abstain_b6`: budget=6, model-driven policy. Abstention is now an option at
    every step. Compare to `main_b6` (no abstain): does giving the model an
    "out" change accuracy of non-abstained samples (selective accuracy)?

Outputs follow the same schema as other runs and land under
`output/abstain_b0/` and `output/abstain_b6/`. A small roll-up CSV at
`output/abstention_summary.csv` aggregates the new vs. baseline rates.
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

RUNS = [
    dict(name="abstain_b0", policy="model", budget=0),
    dict(name="abstain_b6", policy="model", budget=6),
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
        roll.to_csv(OUT_ROOT / "abstention_summary.csv", index=False)
        roll.to_json(OUT_ROOT / "abstention_summary.jsonl", orient="records", lines=True, force_ascii=False)
        print("\n=== Abstention summary ===")
        print(roll.to_string(index=False))


if __name__ == "__main__":
    main()
