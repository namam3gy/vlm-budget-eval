"""Difficulty stratification: zi_correct vs zi_wrong cohorts across budgets.

Phase 1b of ROADMAP. Re-uses `zero_info` predictions to label every sample by
prior-knowledge difficulty and then re-projects all sweep runs through that
split. Surfaces three views:

  A. Cohort accuracy curves -- accuracy vs budget for each cohort
  B. Cohort modality mix -- text/visual/wasted requests per (budget, cohort)
  C. Subject x cohort breakdown -- where does each subject's zi_correct
     cohort live, and at which budget does it actually peak?

The motivating question (from INSIGHTS finding ⑧): vanilla policy treats easy
and hard samples almost identically. Where does that hurt most? Are there
subjects/cohorts where adding info actively destroys accuracy, vs subjects
where info helps even the hard cohort?

Outputs:
  output/difficulty/cohort_curve.csv          -- A
  output/difficulty/cohort_modality.csv       -- B
  output/difficulty/subject_cohort.csv        -- C
  output/difficulty/peak_budget.csv           -- best budget per cohort
  output/plots/difficulty/{A,B,C,D}_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE = Path("/mnt/ddn/prod-runs/thyun.park/src/vlm_budget_eval")
OUT_ROOT = WORKSPACE / "output"
DIFF_DIR = OUT_ROOT / "difficulty"
PLOT_DIR = OUT_ROOT / "plots" / "difficulty"
DIFF_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_RUNS: list[tuple[str, int]] = [
    ("zero_info", 0),
    ("sweep_b1", 1), ("sweep_b2", 2), ("sweep_b3", 3),
    ("sweep_b4", 4), ("sweep_b5", 5), ("main_b6", 6),
    ("sweep_b7", 7), ("sweep_b8", 8), ("sweep_b10", 10),
]

# Reference policies at b=6 we want plotted as horizontal bands
REF_RUNS = [("always_text", "#d62728", "s"),
            ("always_visual", "#2ca02c", "^"),
            ("full_info", "#9467bd", "D")]


def load_jsonl(run: str) -> list[dict] | None:
    p = OUT_ROOT / run / "predictions.jsonl"
    if not p.exists():
        return None
    with p.open() as f:
        return [json.loads(line) for line in f]


def to_df(recs: list[dict], run: str, budget: int) -> pd.DataFrame:
    rows = []
    for r in recs:
        rows.append({
            "sample_id": r["sample_id"],
            "subject": r["subject"],
            "answer_letter": r["answer_letter"],
            "run": run,
            "budget": budget,
            "is_correct": int(r["is_correct"]),
            "final_action": r["final_action"],
            "final_choice": r.get("final_choice"),
            "text_requests": r["text_requests"],
            "visual_requests": r["visual_requests"],
            "wasted_requests": r["wasted_requests"],
            "budget_used": r["budget_used"],
        })
    return pd.DataFrame(rows)


def cohort_label(zero_info_df: pd.DataFrame) -> pd.Series:
    """Returns sample_id -> cohort label."""
    return zero_info_df.set_index("sample_id")["is_correct"].map(
        lambda x: "zi_correct" if x else "zi_wrong"
    )


# ---------------------------------------------------------------------------
# A. Cohort accuracy curves + reference-policy bands
# ---------------------------------------------------------------------------

def cohort_curve(big: pd.DataFrame) -> pd.DataFrame:
    g = (big.groupby(["budget", "cohort"])
         .agg(n=("is_correct", "size"),
              accuracy=("is_correct", "mean"),
              text=("text_requests", "mean"),
              visual=("visual_requests", "mean"),
              wasted=("wasted_requests", "mean"),
              budget_used=("budget_used", "mean"),
              spon_frac=("final_action", lambda s: (s == "ANSWER").mean()))
         .round(4)
         .reset_index())
    return g.sort_values(["cohort", "budget"])


def plot_cohort_curve(curve: pd.DataFrame, refs: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    palette = {"zi_correct": "#2ca02c", "zi_wrong": "#d62728"}
    for cohort, color in palette.items():
        s = curve[curve["cohort"] == cohort].sort_values("budget")
        ax.plot(s["budget"], s["accuracy"], marker="o", color=color, lw=2,
                label=f"{cohort} (n={int(s['n'].iloc[0])})")
        for _, r in s.iterrows():
            ax.annotate(f"{r['accuracy']:.3f}", (r["budget"], r["accuracy"]),
                        textcoords="offset points", xytext=(0, 7),
                        fontsize=8, ha="center", color=color)

    # reference policies as horizontal bands per cohort
    for run, color, marker in REF_RUNS:
        if run not in refs:
            continue
        for cohort, edge in [("zi_correct", "#2ca02c"), ("zi_wrong", "#d62728")]:
            y = refs[run].get(cohort)
            if y is None:
                continue
            ax.axhline(y, color=color, lw=0.8, alpha=0.4, linestyle=":")
            ax.scatter([6], [y], color=color, marker=marker, s=70, zorder=5,
                       edgecolors=edge, linewidths=1.0)
    legend_extra = [plt.Line2D([0], [0], marker=m, color="w", markerfacecolor=c,
                               markeredgecolor="black", label=run, markersize=8)
                    for run, c, m in REF_RUNS if run in refs]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_extra, labels + [r for r, _, _ in REF_RUNS if r in refs],
              loc="center right", fontsize=9)

    ax.set_xlabel("Budget")
    ax.set_ylabel("Accuracy")
    ax.set_title("A — Cohort accuracy vs budget (zi_correct vs zi_wrong)\n"
                 "reference policies at b=6 marked with shapes")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = PLOT_DIR / "A_cohort_accuracy_curve.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# B. Cohort modality mix
# ---------------------------------------------------------------------------

def plot_cohort_modality(curve: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    palette = {"zi_correct": "#2ca02c", "zi_wrong": "#d62728"}
    for ax, modality in zip(axes, ["text", "visual"]):
        for cohort, color in palette.items():
            s = curve[curve["cohort"] == cohort].sort_values("budget")
            ax.plot(s["budget"], s[modality], marker="o", color=color,
                    label=cohort, lw=2)
        ax.set_xlabel("Budget")
        ax.set_ylabel(f"Mean {modality} requests")
        ax.set_title(f"B — {modality} requests by cohort")
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p = PLOT_DIR / "B_cohort_modality.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# C. Subject x cohort breakdown
# ---------------------------------------------------------------------------

def subject_cohort(big: pd.DataFrame, zero_info_df: pd.DataFrame) -> pd.DataFrame:
    """For each (subject, cohort): cohort size, accuracy at each budget."""
    # cohort size per subject
    zi = zero_info_df.assign(
        cohort=zero_info_df["is_correct"].map(lambda x: "zi_correct" if x else "zi_wrong")
    )
    sizes = (zi.groupby(["subject", "cohort"]).size()
             .unstack(fill_value=0)
             .assign(total=lambda d: d.sum(axis=1)))
    sizes["zi_correct_frac"] = sizes["zi_correct"] / sizes["total"]
    sizes = sizes.round(3)

    # per (subject, cohort, budget) accuracy
    acc = (big.groupby(["subject", "cohort", "budget"])["is_correct"]
           .mean().round(3).unstack("budget"))
    return sizes, acc


def plot_subject_cohort(acc: pd.DataFrame, sizes: pd.DataFrame) -> None:
    """Per-subject zi_correct vs zi_wrong curve panels."""
    subjects = sizes.index.tolist()
    fig, axes = plt.subplots(1, len(subjects), figsize=(5 * len(subjects), 4.5),
                              sharey=True)
    if len(subjects) == 1:
        axes = [axes]
    palette = {"zi_correct": "#2ca02c", "zi_wrong": "#d62728"}
    budgets = sorted(acc.columns.tolist())
    for ax, subj in zip(axes, subjects):
        for cohort, color in palette.items():
            try:
                row = acc.loc[(subj, cohort)]
            except KeyError:
                continue
            n = int(sizes.loc[subj, cohort])
            ax.plot(budgets, [row[b] for b in budgets], marker="o", color=color,
                    label=f"{cohort} (n={n})", lw=2)
        ax.set_title(f"{subj}\n(total n={int(sizes.loc[subj, 'total'])}, "
                     f"zi_correct={sizes.loc[subj, 'zi_correct_frac']:.2f})")
        ax.set_xlabel("Budget")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("C — Subject × Cohort accuracy curves", y=1.02)
    fig.tight_layout()
    p = PLOT_DIR / "C_subject_cohort.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# D. "Info hurts easy samples" — accuracy delta from b=1 reference, per cohort
# ---------------------------------------------------------------------------

def plot_delta_from_b1(curve: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    palette = {"zi_correct": "#2ca02c", "zi_wrong": "#d62728"}
    for cohort, color in palette.items():
        s = curve[curve["cohort"] == cohort].sort_values("budget")
        s = s[s["budget"] >= 1]
        baseline = s[s["budget"] == 1]["accuracy"].values
        if len(baseline) == 0:
            continue
        delta = s["accuracy"] - baseline[0]
        ax.plot(s["budget"], delta, marker="o", color=color, lw=2,
                label=f"{cohort} (b=1 acc = {baseline[0]:.3f})")
        for x, y in zip(s["budget"], delta):
            ax.annotate(f"{y:+.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 7), fontsize=8, ha="center", color=color)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("Budget")
    ax.set_ylabel("Accuracy delta vs b=1")
    ax.set_title("D — Accuracy change as budget grows, per cohort\n"
                 "negative = adding info hurt this cohort")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p = PLOT_DIR / "D_delta_from_b1.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    # Load all runs
    all_recs: dict[str, list[dict]] = {}
    for run, _ in MODEL_RUNS:
        recs = load_jsonl(run)
        if recs is None:
            print(f"skip {run}")
            continue
        all_recs[run] = recs

    if "zero_info" not in all_recs:
        raise RuntimeError("zero_info run missing — cannot build cohort labels")

    zero_df = to_df(all_recs["zero_info"], "zero_info", 0)
    cohort_map = cohort_label(zero_df)
    print(f"cohort sizes: {cohort_map.value_counts().to_dict()}")

    # Build the big long-format frame, attaching cohort
    frames = []
    for run, recs in all_recs.items():
        budget = next(b for r, b in MODEL_RUNS if r == run)
        df = to_df(recs, run, budget)
        df["cohort"] = df["sample_id"].map(cohort_map)
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    big = big[big["cohort"].notna()].copy()

    # Reference policies (always_text / always_visual / full_info) at b=6
    # We want their per-cohort accuracy too
    refs: dict[str, dict] = {}
    for run, *_ in REF_RUNS:
        recs = load_jsonl(run)
        if recs is None:
            continue
        d = to_df(recs, run, 6)
        d["cohort"] = d["sample_id"].map(cohort_map)
        d = d[d["cohort"].notna()]
        refs[run] = d.groupby("cohort")["is_correct"].mean().to_dict()
        print(f"ref {run}: {refs[run]}")

    # A
    curve = cohort_curve(big)
    curve.to_csv(DIFF_DIR / "cohort_curve.csv", index=False)
    print(f"\n=== A cohort curve ===\n{curve.to_string(index=False)}")

    # B (modality mix is in curve already, dump as separate view)
    mod = curve[["budget", "cohort", "text", "visual", "wasted", "budget_used",
                 "spon_frac", "accuracy", "n"]]
    mod.to_csv(DIFF_DIR / "cohort_modality.csv", index=False)

    # C
    sizes, acc_table = subject_cohort(big, zero_df)
    sizes.to_csv(DIFF_DIR / "subject_cohort_sizes.csv")
    acc_table.to_csv(DIFF_DIR / "subject_cohort_accuracy.csv")
    print(f"\n=== C subject sizes ===\n{sizes}")
    print(f"\n=== C subject x cohort accuracy by budget ===\n{acc_table}")

    # Peak-budget summary
    peak = (curve.loc[curve.groupby("cohort")["accuracy"].idxmax()]
            [["cohort", "budget", "accuracy", "n"]]
            .rename(columns={"budget": "peak_budget", "accuracy": "peak_acc"}))
    peak.to_csv(DIFF_DIR / "peak_budget.csv", index=False)
    print(f"\n=== peak budget per cohort ===\n{peak.to_string(index=False)}")

    # Plots
    plot_cohort_curve(curve, refs)
    plot_cohort_modality(curve)
    plot_subject_cohort(acc_table, sizes)
    plot_delta_from_b1(curve)

    print(f"\nall outputs under {DIFF_DIR}/ and {PLOT_DIR}/")


if __name__ == "__main__":
    main()
