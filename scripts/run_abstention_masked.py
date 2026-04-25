"""Phase 1d / I13: abstention on the image-masked mini-set.

Runs the same abstain-enabled policy on `preproc_masked/` (100 samples, all
tiles replaced with white) at three budget points so we can compare against
the matching abstain runs on the unmasked full set:

  - b=0  : tied to abstain_b0  (no info either way)
  - b=4  : tied to abstain_b4  (mid-budget; image was the main lever)
  - b=6  : tied to abstain_b6  (full Phase-1c budget)

Hypothesis (per ⑭): if vanilla abstention has any sufficiency detection,
masked abstain rate should be materially higher than unmasked at every
non-zero budget — the model has been told the image is gone (visually)
and should prefer ABSTAIN over guessing.

Outputs land at `output/abstain_masked_b{0,4,6}/` with the standard schema.
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
PREPROC_MASKED = WORKSPACE / "preproc_masked"

SHARED = dict(
    preproc_dir=PREPROC_MASKED,
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
    dict(name="abstain_masked_b0", policy="model", budget=0),
    dict(name="abstain_masked_b4", policy="model", budget=4),
    dict(name="abstain_masked_b6", policy="model", budget=6),
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
    samples_df = load_samples(PREPROC_MASKED)
    print(f"Loaded {len(samples_df)} samples from {PREPROC_MASKED}")

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
        roll.to_csv(OUT_ROOT / "abstention_masked_summary.csv", index=False)
        roll.to_json(OUT_ROOT / "abstention_masked_summary.jsonl",
                     orient="records", lines=True, force_ascii=False)
        print("\n=== Masked abstention summary ===")
        print(roll.to_string(index=False))


if __name__ == "__main__":
    main()
