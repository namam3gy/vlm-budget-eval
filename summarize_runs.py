"""Roll up summary_overall.csv from every run subdir under output/ into one table.

Reads all output/<run>/summary_overall.csv, attaches the run name + policy/budget
parsed from the matching predictions, and writes a comparison table to
output/all_runs_summary.csv (and prints to stdout).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

WORKSPACE = Path("/mnt/ddn/prod-runs/thyun.park/src/eval_sufficiency")
OUT_ROOT = WORKSPACE / "output"

# Friendly ordering for the comparison table.
RUN_ORDER = [
    "zero_info",
    "sweep_b1",
    "sweep_b2",
    "sweep_b3",
    "sweep_b4",
    "sweep_b5",
    "main_b6",
    "sweep_b7",
    "sweep_b8",
    "sweep_b10",
    "full_info",
    "always_text",
    "always_visual",
    "nudge_b6",
]


def load_run(name: str) -> pd.DataFrame | None:
    overall = OUT_ROOT / name / "summary_overall.csv"
    preds = OUT_ROOT / name / "predictions.parquet"
    if not overall.exists():
        return None
    df = pd.read_csv(overall)
    df.insert(0, "run", name)
    if preds.exists():
        pdf = pd.read_parquet(preds)
        if "budget" in pdf.columns:
            df["budget"] = int(pdf["budget"].iloc[0])
    return df


def main() -> None:
    rows = []
    for name in RUN_ORDER:
        r = load_run(name)
        if r is not None:
            rows.append(r)
    # Plus any other run dirs not in RUN_ORDER
    for sub in sorted(p.name for p in OUT_ROOT.iterdir() if p.is_dir()):
        if sub in RUN_ORDER or sub == "smoke_20":
            continue
        r = load_run(sub)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No completed runs found.")
        return

    table = pd.concat(rows, ignore_index=True)
    cols = ["run", "n", "accuracy", "mean_budget_used", "mean_text_requests",
            "mean_visual_requests", "mean_wasted", "parse_fail_rate",
            "forced_answer_rate", "answered_without_info_rate"]
    table = table[[c for c in cols if c in table.columns]]
    table = table.round({"accuracy": 4, "mean_budget_used": 3,
                         "mean_text_requests": 3, "mean_visual_requests": 3,
                         "mean_wasted": 3, "parse_fail_rate": 4,
                         "forced_answer_rate": 4, "answered_without_info_rate": 4})

    table.to_csv(OUT_ROOT / "all_runs_summary.csv", index=False)
    table.to_json(OUT_ROOT / "all_runs_summary.jsonl", orient="records", lines=True)

    print(table.to_string(index=False))
    print(f"\nWrote {OUT_ROOT / 'all_runs_summary.csv'}")


if __name__ == "__main__":
    main()
