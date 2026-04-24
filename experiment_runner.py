"""Thin driver for budget-constrained sequential info-seeking eval."""

from pathlib import Path

from budget_eval import EvalConfig, main

WORKSPACE = Path("/mnt/ddn/prod-runs/thyun.park/src/eval_sufficiency")

cfg = EvalConfig(
    preproc_dir=WORKSPACE / "preproc",
    out_dir=WORKSPACE / "output" / "main_b6",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="bfloat16",
    budget=6,
    max_new_tokens=128,
    temperature=0.0,
    random_seed=42,
    max_eval_rows=-1,
    tile_order="shuffled",
    text_order="natural",
    save_trace=True,
)

if __name__ == "__main__":
    main(cfg)
