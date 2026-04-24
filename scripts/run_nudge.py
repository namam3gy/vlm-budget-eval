"""Prompt-nudge experiment.

Runs the model policy at budget=6 with a system prompt that explicitly tells
the model not to over-rely on text. We saw in the baseline analysis that
always_visual (b=6, 4 tiles) beats both always_text and the model's adaptive
policy. This run probes whether a simple textual nudge closes the gap.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vlm_budget_eval import (
    SYSTEM_INSTRUCTION,
    EvalConfig,
    aggregate,
    load_model_and_processor,
    load_samples,
    run_episode,
)

WORKSPACE = Path("/mnt/ddn/prod-runs/thyun.park/src/vlm_budget_eval")
OUT_ROOT = WORKSPACE / "output"
PREPROC = WORKSPACE / "preproc"

NUDGE_BLURB = (
    "\n\nIMPORTANT MODALITY GUIDANCE: For ScienceQA-style questions the source "
    "image is split into a small number of tiles (typically 4) and often contains "
    "the diagram, chart, photograph, or visual cue that the question hinges on. "
    "Text hints are individual sentences from a longer passage and many of them "
    "are background, not directly the answer. Do NOT default to REQUEST_TEXT. "
    "When a question references something visual ('Which map...', 'In the picture...', "
    "'Compare these two organisms...', etc.), prefer REQUEST_VISUAL. Spend at "
    "least one budget unit on a visual tile before relying solely on text."
)

NUDGED_INSTRUCTION = SYSTEM_INSTRUCTION + NUDGE_BLURB

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
    max_forced_attempts=2,
)


def main() -> None:
    samples_df = load_samples(PREPROC)
    print(f"Loaded {len(samples_df)} samples")

    out_dir = OUT_ROOT / "nudge_b6"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = EvalConfig(
        **SHARED,
        out_dir=out_dir,
        budget=6,
        policy="model",
        system_instruction=NUDGED_INSTRUCTION,
    )

    print(f"Loading model {cfg.model_id} ...")
    model, processor = load_model_and_processor(cfg)

    rng = random.Random(cfg.random_seed)
    results = []
    for row in tqdm(samples_df.to_dict(orient="records"), total=len(samples_df)):
        results.append(run_episode(model, processor, row, cfg, rng))

    pred_df = pd.DataFrame(results)
    flat = pred_df.drop(columns=[c for c in ["trace"] if c in pred_df.columns])
    flat.to_parquet(out_dir / "predictions.parquet", index=False)
    flat.to_csv(out_dir / "predictions.csv", index=False)
    pred_df.to_json(out_dir / "predictions.jsonl", orient="records", lines=True, force_ascii=False)

    aggs = aggregate(flat)
    for k, a in aggs.items():
        a.to_csv(out_dir / f"summary_{k}.csv", index=False)
        a.to_json(out_dir / f"summary_{k}.jsonl", orient="records", lines=True, force_ascii=False)

    # Save the nudge blurb so the run is self-documenting.
    (out_dir / "system_instruction.txt").write_text(NUDGED_INSTRUCTION)

    print("\n=== Nudge b=6 Overall ===")
    print(aggs["overall"].to_string(index=False))

    base = OUT_ROOT / "main_b6" / "summary_overall.csv"
    if base.exists():
        b = pd.read_csv(base)
        print(f"\nBaseline main_b6:   accuracy={b['accuracy'].iloc[0]:.4f} "
              f"text={b['mean_text_requests'].iloc[0]:.2f} "
              f"visual={b['mean_visual_requests'].iloc[0]:.2f}")
        print(f"Nudge    nudge_b6: accuracy={aggs['overall']['accuracy'].iloc[0]:.4f} "
              f"text={aggs['overall']['mean_text_requests'].iloc[0]:.2f} "
              f"visual={aggs['overall']['mean_visual_requests'].iloc[0]:.2f}")


if __name__ == "__main__":
    main()
