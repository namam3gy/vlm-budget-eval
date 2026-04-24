"""Phase 1c analysis: how well does the vanilla model use the ABSTAIN action?

Reads:
  output/abstain_b0/predictions.jsonl    (forced to commit-or-abstain w/ no info)
  output/abstain_b6/predictions.jsonl    (full b=6 budget, abstain available)
  output/zero_info/predictions.jsonl     (b=0 reference, no abstain option)
  output/main_b6/predictions.jsonl       (b=6 reference, no abstain option)

Surfaces three views:

  A. Abstention rate, selective accuracy, coverage at each budget.
     Comparison vs no-abstain reference: did giving the model an "out" change
     accuracy of the samples it DID answer (selective accuracy)?
  B. Where does the model abstain? Cohort × abstain crosstab using the
     zi_correct/zi_wrong cohorts from Phase 1b. Calibrated abstention should
     concentrate on zi_wrong (the samples the model truly cannot answer with
     prior knowledge alone).
  C. Effective Reliability Φ proxy (Whitehead et al., ECCV 2022):
     Φ = (n_correct - c · n_wrong) / n_attempted, with abstention worth 0.
     Sweep c ∈ {0.0, 0.5, 1.0, 2.0} for the abstain run vs the no-abstain
     reference (where wrong answers contribute negatively because the model
     could not abstain).

Outputs:
  output/abstention/summary.csv      -- A
  output/abstention/cohort_xtab.csv  -- B
  output/abstention/phi_curve.csv    -- C
  output/plots/abstention/{A,B,C}_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE = Path("/mnt/ddn/prod-runs/thyun.park/src/vlm_budget_eval")
OUT_ROOT = WORKSPACE / "output"
ABS_DIR = OUT_ROOT / "abstention"
PLOT_DIR = OUT_ROOT / "plots" / "abstention"
ABS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(run: str) -> list[dict] | None:
    p = OUT_ROOT / run / "predictions.jsonl"
    if not p.exists():
        return None
    with p.open() as f:
        return [json.loads(line) for line in f]


def to_df(recs: list[dict], run: str) -> pd.DataFrame:
    rows = []
    for r in recs:
        rows.append({
            "sample_id": r["sample_id"],
            "subject": r["subject"],
            "answer_letter": r["answer_letter"],
            "run": run,
            "budget": r["budget"],
            "is_correct": int(r["is_correct"]),
            "final_action": r["final_action"],
            "final_choice": r.get("final_choice"),
            "abstain": int(r.get("final_action") in ("ABSTAIN", "FORCED_ABSTAIN")),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A. summary -- abstain rate + selective accuracy + coverage
# ---------------------------------------------------------------------------

def metric_a(name: str, df: pd.DataFrame) -> dict:
    n = len(df)
    abst = df["abstain"].sum()
    answered = df[df["abstain"] == 0]
    n_ans = len(answered)
    n_correct = int(answered["is_correct"].sum())
    return {
        "run": name,
        "n": n,
        "abstain_n": int(abst),
        "abstain_rate": float(abst) / n if n else 0.0,
        "answered_n": n_ans,
        "coverage": n_ans / n if n else 0.0,
        "answered_acc (selective)": float(n_correct) / n_ans if n_ans else float("nan"),
        "naive_acc (any)": float(df["is_correct"].mean()),
    }


# ---------------------------------------------------------------------------
# B. cohort x abstain crosstab
# ---------------------------------------------------------------------------

def cohort_crosstab(zero_df: pd.DataFrame, abst_df: pd.DataFrame) -> pd.DataFrame:
    cohort = zero_df.set_index("sample_id")["is_correct"].map(
        lambda x: "zi_correct" if x else "zi_wrong"
    )
    df = abst_df.copy()
    df["cohort"] = df["sample_id"].map(cohort)
    df = df[df["cohort"].notna()]
    table = (df.groupby(["cohort", "abstain"]).size()
             .unstack(fill_value=0)
             .rename(columns={0: "answered", 1: "abstained"}))
    table["total"] = table.sum(axis=1)
    table["abstain_rate"] = (table["abstained"] / table["total"]).round(3)
    # answered subset accuracy per cohort
    ans = df[df["abstain"] == 0].groupby("cohort")["is_correct"].agg(["mean", "size"])
    ans = ans.rename(columns={"mean": "answered_acc", "size": "answered_n"})
    table = table.join(ans)
    return table


# ---------------------------------------------------------------------------
# C. Effective Reliability Phi sweep
#
#   Phi(c) = (n_correct - c * n_wrong) / n_total
#   with abstain contributing 0 to numerator (no credit, no penalty).
#
# For the no-abstain reference, n_abstain=0 so wrong answers always penalise.
# For the abstain run, the model can shed wrong-answer penalty by abstaining,
# but loses credit. The curve shows when abstention pays off.
# ---------------------------------------------------------------------------

def phi(df: pd.DataFrame, c: float) -> float:
    n_total = len(df)
    answered = df[df["abstain"] == 0]
    n_correct = int(answered["is_correct"].sum())
    n_wrong = len(answered) - n_correct
    return (n_correct - c * n_wrong) / n_total if n_total else float("nan")


# ---------------------------------------------------------------------------
# D. Cohort-aligned comparison -- the real calibration test.
#
#   The naive "abstain_b0 answered_acc 0.833 beats zero_info overall 0.562"
#   comparison is confounded by selection: the subsets are different.
#   The fair test is: on the SAME subset the abstain run chose to answer,
#   what accuracy does the no-abstain reference achieve? If the reference
#   already sits at ~0.83 there, the "decision to answer" signal is perfectly
#   aligned with prior knowledge and abstention adds no new information.
#   If materially lower, abstention adds real selectivity.
# ---------------------------------------------------------------------------

def aligned_comparison(abstain_df: pd.DataFrame, ref_df: pd.DataFrame,
                       abstain_name: str, ref_name: str) -> dict:
    answered = abstain_df[abstain_df["abstain"] == 0]
    answered_ids = set(answered["sample_id"])
    ref_on_same = ref_df[ref_df["sample_id"].isin(answered_ids)]
    return {
        "run": abstain_name,
        "reference_run": ref_name,
        "subset_size": len(answered_ids),
        "own_acc": float(answered["is_correct"].mean()) if len(answered) else float("nan"),
        "reference_acc_on_same_subset": float(ref_on_same["is_correct"].mean())
            if len(ref_on_same) else float("nan"),
        "reference_overall_acc": float(ref_df["is_correct"].mean()),
        "uplift_vs_reference_on_subset": (
            float(answered["is_correct"].mean() - ref_on_same["is_correct"].mean())
            if len(answered) and len(ref_on_same) else float("nan")
        ),
    }


def plot_metric_a(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(df))
    width = 0.27
    ax.bar(x - width, df["coverage"], width, label="coverage (1 − abstain_rate)", color="#1f77b4")
    ax.bar(x, df["answered_acc (selective)"], width, label="selective accuracy", color="#2ca02c")
    ax.bar(x + width, df["naive_acc (any)"], width, label="naive accuracy (abstain = wrong)", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(df["run"], rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("A — Abstention coverage vs selective vs naive accuracy")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    p = PLOT_DIR / "A_summary.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


def plot_metric_b(tables: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, len(tables), figsize=(5.5 * len(tables), 4.2),
                              sharey=True)
    if len(tables) == 1:
        axes = [axes]
    for ax, (name, t) in zip(axes, tables.items()):
        cohorts = t.index.tolist()
        ax.bar(cohorts, t["abstain_rate"], color=["#2ca02c", "#d62728"])
        for i, (rate, n) in enumerate(zip(t["abstain_rate"], t["total"])):
            ax.text(i, rate + 0.01, f"{rate:.2f}\nn={int(n)}",
                    ha="center", fontsize=9)
        ax.set_title(name); ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Abstain rate")
    fig.suptitle("B — Abstention rate by zero_info cohort", y=1.02)
    fig.tight_layout()
    p = PLOT_DIR / "B_cohort_abstain.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")


def plot_metric_c(curve: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    palette = {"zero_info": "#9467bd", "abstain_b0": "#1f77b4",
               "main_b6": "#ff7f0e", "abstain_b6": "#2ca02c"}
    for run in curve["run"].unique():
        s = curve[curve["run"] == run].sort_values("c")
        ax.plot(s["c"], s["phi"], marker="o", lw=2,
                color=palette.get(run, None), label=run)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("Wrong-answer cost c (per Effective Reliability Φ)")
    ax.set_ylabel("Φ(c)")
    ax.set_title("C — Effective Reliability Φ vs wrong-answer cost\n"
                 "abstain pays off where the abstain run beats the no-abstain reference")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p = PLOT_DIR / "C_phi_curve.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    runs = {
        "zero_info":  load_jsonl("zero_info"),
        "abstain_b0": load_jsonl("abstain_b0"),
        "main_b6":    load_jsonl("main_b6"),
        "abstain_b6": load_jsonl("abstain_b6"),
    }
    missing = [k for k, v in runs.items() if v is None]
    if missing:
        print(f"missing runs: {missing} -- run run_abstention.py first")
    dfs = {k: to_df(v, k) for k, v in runs.items() if v is not None}

    # A
    rows = [metric_a(k, df) for k, df in dfs.items()]
    a_df = pd.DataFrame(rows).round(4)
    a_df.to_csv(ABS_DIR / "summary.csv", index=False)
    print(f"\n=== A summary ===\n{a_df.to_string(index=False)}")

    # B
    if "zero_info" in dfs:
        zero = dfs["zero_info"]
        cohort_tables = {}
        for run in ["abstain_b0", "abstain_b6"]:
            if run in dfs:
                cohort_tables[run] = cohort_crosstab(zero, dfs[run])
        if cohort_tables:
            xtab_dump = pd.concat({k: t for k, t in cohort_tables.items()}, names=["run"])
            xtab_dump.to_csv(ABS_DIR / "cohort_xtab.csv")
            print(f"\n=== B cohort × abstain ===\n{xtab_dump}")

    # C
    cs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    rows_c = []
    for run in ["zero_info", "abstain_b0", "main_b6", "abstain_b6"]:
        if run not in dfs:
            continue
        for c in cs:
            rows_c.append({"run": run, "c": c, "phi": phi(dfs[run], c)})
    c_df = pd.DataFrame(rows_c).round(4)
    c_df.to_csv(ABS_DIR / "phi_curve.csv", index=False)
    print(f"\n=== C Phi sweep ===")
    print(c_df.pivot(index="c", columns="run", values="phi"))

    # D -- cohort-aligned: did the decision to answer pick samples the reference
    # would also get right?
    aligned_rows = []
    if "abstain_b0" in dfs and "zero_info" in dfs:
        aligned_rows.append(aligned_comparison(
            dfs["abstain_b0"], dfs["zero_info"], "abstain_b0", "zero_info"))
    if "abstain_b6" in dfs and "main_b6" in dfs:
        aligned_rows.append(aligned_comparison(
            dfs["abstain_b6"], dfs["main_b6"], "abstain_b6", "main_b6"))
    if aligned_rows:
        d_df = pd.DataFrame(aligned_rows).round(4)
        d_df.to_csv(ABS_DIR / "aligned_comparison.csv", index=False)
        print(f"\n=== D cohort-aligned comparison ===")
        print(d_df.to_string(index=False))

    # Plots
    plot_metric_a(rows)
    if "zero_info" in dfs and cohort_tables:
        plot_metric_b(cohort_tables)
    plot_metric_c(c_df)

    print(f"\nall outputs under {ABS_DIR}/ and {PLOT_DIR}/")


if __name__ == "__main__":
    main()
