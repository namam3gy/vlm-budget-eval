"""Post-run analysis: budget curves, subject cross-tabs, modality-bias breakdown.

Reads output/{run}/predictions.parquet for every completed run and writes:
- output/plots/accuracy_vs_budget.png
- output/plots/modality_mix.png
- output/subject_crosstab.csv          (config x subject accuracy)
- output/visual_bias_breakdown.csv     (model_b6 vs always_visual disagreement)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

WORKSPACE = Path("/mnt/ddn/prod-runs/thyun.park/src/vlm_budget_eval")
OUT_ROOT = WORKSPACE / "output"
PLOT_DIR = OUT_ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def load_pred(run: str) -> pd.DataFrame | None:
    p = OUT_ROOT / run / "predictions.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ---------------------------------------------------------------------------
# 1. Accuracy vs. budget curve (model policy + reference baselines)
# ---------------------------------------------------------------------------

MODEL_SWEEP_RUNS = [
    ("zero_info", 0),
    ("sweep_b1", 1), ("sweep_b2", 2), ("sweep_b3", 3), ("sweep_b4", 4),
    ("sweep_b5", 5), ("main_b6", 6), ("sweep_b7", 7), ("sweep_b8", 8),
    ("sweep_b10", 10),
]


def plot_budget_curve():
    points = []
    for run, b in MODEL_SWEEP_RUNS:
        df = load_pred(run)
        if df is None:
            continue
        points.append((run, b, float(df["is_correct"].mean())))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    ax.plot(xs, ys, marker="o", lw=2, color="#1f77b4", label="model policy (budget sweep)")
    for label, x, y in points:
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, ha="center", color="#1f77b4")

    # full_info as a horizontal reference line (ceiling)
    full = load_pred("full_info")
    if full is not None:
        y = float(full["is_correct"].mean())
        ax.axhline(y, color="#9467bd", linestyle="--", lw=1.5, alpha=0.8,
                   label=f"full_info ceiling = {y:.3f}")

    # Always-one-modality baselines at b=6
    for run, color, marker, dy in [("always_text", "#d62728", "s", -18),
                                    ("always_visual", "#2ca02c", "^", 8),
                                    ("nudge_b6", "#ff7f0e", "D", -24)]:
        df = load_pred(run)
        if df is None:
            continue
        b = int(df["budget"].iloc[0])
        y = float(df["is_correct"].mean())
        ax.scatter([b], [y], color=color, marker=marker, s=90, zorder=5,
                   label=f"{run} = {y:.3f}", edgecolors="black", linewidths=0.6)

    ax.set_xlabel("Budget (info requests)")
    ax.set_ylabel("Accuracy")
    ax.set_title("ScienceQA accuracy vs. budget — Qwen2.5-VL-7B, n=500 per point")
    ax.set_xticks([b for _, b in MODEL_SWEEP_RUNS])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = PLOT_DIR / "accuracy_vs_budget.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# 2. Modality mix per config
# ---------------------------------------------------------------------------

def plot_modality_mix():
    rows = []
    for run in ["zero_info", "sweep_b1", "sweep_b2", "sweep_b3", "sweep_b4",
                "sweep_b5", "main_b6", "sweep_b7", "sweep_b8", "sweep_b10",
                "always_text", "always_visual", "full_info", "nudge_b6"]:
        df = load_pred(run)
        if df is None:
            continue
        rows.append({
            "run": run,
            "text": float(df["text_requests"].mean()),
            "visual": float(df["visual_requests"].mean()),
            "accuracy": float(df["is_correct"].mean()),
        })
    mix = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    x = range(len(mix))
    ax.bar([i - width / 2 for i in x], mix["text"], width, label="text", color="#d62728")
    ax.bar([i + width / 2 for i in x], mix["visual"], width, label="visual", color="#2ca02c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(mix["run"], rotation=15)
    ax.set_ylabel("Mean requests per sample")
    ax.set_title("Mean text vs. visual requests by config (n=500 per config)")
    for i, acc in enumerate(mix["accuracy"]):
        ax.annotate(f"acc={acc:.3f}", (i, max(mix["text"].iloc[i], mix["visual"].iloc[i])),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    out = PLOT_DIR / "modality_mix.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# 3. Subject cross-tab
# ---------------------------------------------------------------------------

def subject_crosstab():
    rows = []
    for run in ["zero_info", "sweep_b1", "sweep_b2", "sweep_b3", "sweep_b4",
                "sweep_b5", "main_b6", "sweep_b7", "sweep_b8", "sweep_b10",
                "always_text", "always_visual", "full_info", "nudge_b6"]:
        df = load_pred(run)
        if df is None:
            continue
        for subj, g in df.groupby(df["subject"].fillna("unknown")):
            rows.append({"run": run, "subject": subj, "n": len(g),
                         "accuracy": float(g["is_correct"].mean())})
    long = pd.DataFrame(rows)
    wide = long.pivot(index="run", columns="subject", values="accuracy").round(3)
    counts = long.pivot(index="run", columns="subject", values="n").iloc[0].astype(int)
    wide.loc["__n__"] = counts
    out = OUT_ROOT / "subject_crosstab.csv"
    wide.to_csv(out)
    print(f"wrote {out}")
    print(wide.to_string())
    return wide


# ---------------------------------------------------------------------------
# 4. Visual-bias breakdown: where always_visual succeeds and main_b6 fails
# ---------------------------------------------------------------------------

def visual_bias_breakdown():
    main = load_pred("main_b6")
    av = load_pred("always_visual")
    if main is None or av is None:
        print("skipping visual_bias_breakdown — runs missing")
        return None

    j = main[["sample_id", "subject", "is_correct", "text_requests",
              "visual_requests", "budget_used"]].rename(columns={
        "is_correct": "main_correct",
        "text_requests": "main_text",
        "visual_requests": "main_visual",
        "budget_used": "main_budget",
    }).merge(
        av[["sample_id", "is_correct"]].rename(columns={"is_correct": "av_correct"}),
        on="sample_id",
    )

    def bucket(row):
        if row["main_correct"] and row["av_correct"]:
            return "both_right"
        if row["main_correct"] and not row["av_correct"]:
            return "main_only"
        if not row["main_correct"] and row["av_correct"]:
            return "visual_only"
        return "both_wrong"

    j["bucket"] = j.apply(bucket, axis=1)
    overall = j["bucket"].value_counts(normalize=True).round(3)
    by_subject = (j.groupby("subject")["bucket"]
                  .value_counts(normalize=True)
                  .unstack(fill_value=0).round(3))

    visual_only = j[j["bucket"] == "visual_only"]
    print("\n=== Bucket counts (n=500) ===")
    print(j["bucket"].value_counts())
    print("\n=== Bucket fractions ===")
    print(overall)
    print("\n=== Bucket fractions by subject ===")
    print(by_subject)
    print(f"\n=== visual_only cohort ({len(visual_only)} samples) ===")
    print(f"  mean main_text:   {visual_only['main_text'].mean():.2f}")
    print(f"  mean main_visual: {visual_only['main_visual'].mean():.2f}")
    print(f"  mean main_budget_used: {visual_only['main_budget'].mean():.2f}")

    out = OUT_ROOT / "visual_bias_breakdown.csv"
    j.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return j


def main():
    plot_budget_curve()
    plot_modality_mix()
    subject_crosstab()
    visual_bias_breakdown()


if __name__ == "__main__":
    main()
