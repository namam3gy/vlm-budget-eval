"""Phase 1d analysis: anti-calibration evolution + sufficiency-known masking.

Combines two views:

  I14. Anti-calibration evolution
       Loads abstain_b{0..10} (when available) and plots abstain rate split by
       zi_correct / zi_wrong cohort vs budget. Phase 1c (⑭) showed the
       calibration→anti-calibration flip between b=0 and b=6; this view
       reveals where in the budget range that flip happens.

  I13. Sufficiency-known masking
       Loads `output/abstain_masked_b{0,4,6}/` (100 samples) and the matching
       100-sample subset from `output/abstain_b{0,4,6}/` (same sample_ids).
       Compares abstain rate on masked-image vs unmasked-image variants of
       the same questions, split by cohort. If vanilla abstention has any
       sufficiency detection, masked abstain rate should rise for at least
       one cohort.

Outputs:
  output/abstention_phase1d/I14_cohort_x_budget.csv
  output/abstention_phase1d/I13_masked_vs_unmasked.csv
  output/plots/abstention_phase1d/I14_cohort_x_budget.png
  output/plots/abstention_phase1d/I13_masked_vs_unmasked.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

WORKSPACE = Path("/mnt/ddn/prod-runs/thyun.park/src/vlm_budget_eval")
OUT_ROOT = WORKSPACE / "output"
P1D_DIR = OUT_ROOT / "abstention_phase1d"
PLOT_DIR = OUT_ROOT / "plots" / "abstention_phase1d"
P1D_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ABSTAIN_BUDGETS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]


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
            "run": run,
            "budget": budget,
            "is_correct": int(r["is_correct"]),
            "final_action": r["final_action"],
            "abstain": int(r.get("final_action") in ("ABSTAIN", "FORCED_ABSTAIN")),
        })
    return pd.DataFrame(rows)


def cohort_map() -> pd.Series:
    zero = pd.read_parquet(OUT_ROOT / "zero_info" / "predictions.parquet")
    return zero.set_index("sample_id")["is_correct"].map(
        lambda x: "zi_correct" if x else "zi_wrong"
    )


# ---------------------------------------------------------------------------
# I14 -- anti-calibration evolution
# ---------------------------------------------------------------------------

def i14_analysis(cohort: pd.Series) -> pd.DataFrame:
    rows = []
    for b in ABSTAIN_BUDGETS:
        run = f"abstain_b{b}"
        recs = load_jsonl(run)
        if recs is None:
            print(f"skip {run}")
            continue
        df = to_df(recs, run, b)
        df["cohort"] = df["sample_id"].map(cohort)
        df = df[df["cohort"].notna()]
        g = (df.groupby("cohort")
             .agg(n=("abstain", "size"),
                  abstain_rate=("abstain", "mean"),
                  selective_acc=("is_correct", lambda s: s[df.loc[s.index, "abstain"] == 0].mean()
                                  if (df.loc[s.index, "abstain"] == 0).any() else float("nan")),
                  naive_acc=("is_correct", "mean"))
             .reset_index())
        g["budget"] = b
        rows.append(g)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out


def plot_i14(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    palette = {"zi_correct": "#2ca02c", "zi_wrong": "#d62728"}
    for cohort, color in palette.items():
        s = df[df["cohort"] == cohort].sort_values("budget")
        axes[0].plot(s["budget"], s["abstain_rate"], marker="o", lw=2,
                     color=color, label=f"{cohort} (n={int(s['n'].iloc[0])})")
        axes[1].plot(s["budget"], s["selective_acc"], marker="o", lw=2,
                     color=color, label=cohort)
    axes[0].set_title("I14 — Abstain rate by cohort × budget")
    axes[0].set_xlabel("Budget"); axes[0].set_ylabel("Abstain rate")
    axes[0].set_ylim(0, 1.0); axes[0].grid(alpha=0.3); axes[0].legend()
    # mark the calibration -> anti-calibration cross-over visually
    cross_b = None
    cmp = df.pivot(index="budget", columns="cohort", values="abstain_rate")
    if {"zi_correct", "zi_wrong"}.issubset(cmp.columns):
        diff = cmp["zi_wrong"] - cmp["zi_correct"]
        # find first budget where sign flips (zi_wrong > zi_correct -> calibrated; flips when neg)
        signs = diff.apply(lambda v: "+" if v > 0 else ("-" if v < 0 else "0"))
        # transition from + to - or vice versa
        for i in range(1, len(signs)):
            if signs.iloc[i] != signs.iloc[i - 1] and signs.iloc[i - 1] != "0":
                cross_b = signs.index[i]
                break
        if cross_b is not None:
            axes[0].axvline(cross_b, color="grey", linestyle=":", lw=1.5,
                            label=f"sign flip at b={cross_b}")
            axes[0].legend()

    axes[1].set_title("I14 — Selective accuracy (answered samples only)")
    axes[1].set_xlabel("Budget"); axes[1].set_ylabel("Accuracy on non-abstained")
    axes[1].set_ylim(0, 1.0); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.tight_layout()
    p = PLOT_DIR / "I14_cohort_x_budget.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# I13 -- masked vs unmasked
# ---------------------------------------------------------------------------

def i13_analysis(cohort: pd.Series) -> pd.DataFrame:
    rows = []
    for b in [0, 4, 6]:
        masked = load_jsonl(f"abstain_masked_b{b}")
        unmasked = load_jsonl(f"abstain_b{b}")
        if masked is None or unmasked is None:
            print(f"skip b={b} (missing masked or unmasked)")
            continue

        masked_df = to_df(masked, f"abstain_masked_b{b}", b)
        unmasked_df = to_df(unmasked, f"abstain_b{b}", b)
        masked_df["cohort"] = masked_df["sample_id"].map(cohort)
        unmasked_df["cohort"] = unmasked_df["sample_id"].map(cohort)

        sample_ids = set(masked_df["sample_id"])
        unmasked_subset = unmasked_df[unmasked_df["sample_id"].isin(sample_ids)].copy()

        for cohort_name in ["zi_correct", "zi_wrong"]:
            m = masked_df[masked_df["cohort"] == cohort_name]
            u = unmasked_subset[unmasked_subset["cohort"] == cohort_name]
            if len(m) == 0 or len(u) == 0:
                continue
            rows.append({
                "budget": b,
                "cohort": cohort_name,
                "n_masked": len(m),
                "n_unmasked_subset": len(u),
                "abstain_rate_unmasked": float(u["abstain"].mean()),
                "abstain_rate_masked": float(m["abstain"].mean()),
                "delta_masked_minus_unmasked": float(m["abstain"].mean() - u["abstain"].mean()),
                "naive_acc_unmasked": float(u["is_correct"].mean()),
                "naive_acc_masked": float(m["is_correct"].mean()),
            })
    return pd.DataFrame(rows)


def plot_i13(df: pd.DataFrame) -> None:
    if df.empty:
        return
    budgets = sorted(df["budget"].unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.5 * len(budgets), 4.8),
                              sharey=True)
    if len(budgets) == 1:
        axes = [axes]
    palette = {"zi_correct": "#2ca02c", "zi_wrong": "#d62728"}
    width = 0.35
    for ax, b in zip(axes, budgets):
        s = df[df["budget"] == b].sort_values("cohort")
        x = range(len(s))
        ax.bar([i - width / 2 for i in x], s["abstain_rate_unmasked"],
               width, label="unmasked", color="#7f7f7f", edgecolor="black")
        ax.bar([i + width / 2 for i in x], s["abstain_rate_masked"],
               width, label="masked (image=white)",
               color=[palette[c] for c in s["cohort"]], edgecolor="black")
        for i, (uu, mm) in enumerate(zip(s["abstain_rate_unmasked"], s["abstain_rate_masked"])):
            ax.text(i - width / 2, uu + 0.01, f"{uu:.2f}", ha="center", fontsize=8)
            ax.text(i + width / 2, mm + 0.01, f"{mm:.2f}", ha="center", fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(s["cohort"])
        ax.set_title(f"b={b}")
        ax.set_ylim(0, 1.0); ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Abstain rate")
    fig.suptitle("I13 — abstain rate: image-masked vs unmasked (same 100 samples, 50+50 cohort)",
                 y=1.02)
    fig.tight_layout()
    p = PLOT_DIR / "I13_masked_vs_unmasked.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")


def main() -> None:
    cohort = cohort_map()

    # I14
    i14 = i14_analysis(cohort)
    if not i14.empty:
        i14.to_csv(P1D_DIR / "I14_cohort_x_budget.csv", index=False)
        print(f"\n=== I14 cohort x budget abstain rate ===")
        print(i14.pivot(index="budget", columns="cohort", values="abstain_rate"))
        plot_i14(i14)

    # I13
    i13 = i13_analysis(cohort)
    if not i13.empty:
        i13.to_csv(P1D_DIR / "I13_masked_vs_unmasked.csv", index=False)
        print(f"\n=== I13 masked vs unmasked abstain rate ===")
        print(i13.to_string(index=False))
        plot_i13(i13)

    print(f"\nall outputs under {P1D_DIR}/ and {PLOT_DIR}/")


if __name__ == "__main__":
    main()
