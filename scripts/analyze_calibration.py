"""Calibration analysis from existing budget-sweep traces.

Phase 1a of ROADMAP. Does NOT invoke the model — re-analyses existing
output/{run}/predictions.jsonl produced by run_dense_sweep.py /
experiment_runner.py / run_sweep.py.

Six calibration signals:

  A. spontaneous vs forced answer accuracy delta, per run
     (does the model know when to stop?)
  B. cross-budget choice stability per sample
     (how often does the answer flip when budget changes?)
  C. spontaneous stop-step vs accuracy, pooled across runs
     (early-stop = "I'm sure" — is that reliable?)
  D. forced -> spontaneous answer concordance
     (when the same sample gets forced at small b but spontaneous at large b,
      do the two answers agree? if no, force-answer is unreliable)
  E. stop-step distribution shift vs budget
     (does the model anchor stop-step on budget or stop based on internal cue?)
  F. zero_info-correct cohort: does the model recognise "I already know"
     and stop at step 1 when given a higher budget?

Outputs:
  output/calibration/summary.csv            -- per-run aggregates (A, E)
  output/calibration/per_sample.csv         -- sample x run choice matrix (B)
  output/calibration/stop_step_acc.csv      -- stop_step accuracy curve (C)
  output/calibration/forced_vs_spon.csv     -- per-sample concordance (D)
  output/calibration/zero_info_cohort.csv   -- F breakdown
  output/plots/calibration/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE = Path("/mnt/ddn/prod-runs/thyun.park/src/vlm_budget_eval")
OUT_ROOT = WORKSPACE / "output"
CAL_DIR = OUT_ROOT / "calibration"
PLOT_DIR = OUT_ROOT / "plots" / "calibration"
CAL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_RUNS: list[tuple[str, int]] = [
    ("zero_info", 0),
    ("sweep_b1", 1), ("sweep_b2", 2), ("sweep_b3", 3),
    ("sweep_b4", 4), ("sweep_b5", 5), ("main_b6", 6),
    ("sweep_b7", 7), ("sweep_b8", 8), ("sweep_b10", 10),
]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_jsonl(run: str) -> list[dict] | None:
    p = OUT_ROOT / run / "predictions.jsonl"
    if not p.exists():
        return None
    with p.open() as f:
        return [json.loads(line) for line in f]


def stop_step_of(rec: dict) -> int | None:
    """Step index (1-based) at which the model emitted ANSWER.

    Returns None for FORCED_ANSWER and PARSE_FAIL — these did not stop
    on their own. For spontaneous answers this equals len(trace).
    """
    if rec["final_action"] != "ANSWER":
        return None
    return rec["n_steps"]


# ---------------------------------------------------------------------------
# Metric A: spontaneous vs forced accuracy
# ---------------------------------------------------------------------------

def metric_a(records: list[dict]) -> dict:
    spon = [r for r in records if r["final_action"] == "ANSWER"]
    forced = [r for r in records if r["final_action"] == "FORCED_ANSWER"]
    fail = [r for r in records if r["final_action"] == "PARSE_FAIL"]
    return {
        "n": len(records),
        "spontaneous_n": len(spon),
        "spontaneous_acc": (sum(r["is_correct"] for r in spon) / len(spon)) if spon else float("nan"),
        "forced_n": len(forced),
        "forced_acc": (sum(r["is_correct"] for r in forced) / len(forced)) if forced else float("nan"),
        "parse_fail_n": len(fail),
        "spontaneous_frac": len(spon) / len(records),
    }


# ---------------------------------------------------------------------------
# Metric B: cross-budget choice stability
# ---------------------------------------------------------------------------

def cross_budget_stability(all_runs: dict[str, list[dict]]) -> pd.DataFrame:
    """Wide table: sample_id x run -> final_choice. Plus per-sample stats."""
    rows = []
    for run, recs in all_runs.items():
        for r in recs:
            rows.append({
                "sample_id": r["sample_id"],
                "run": run,
                "budget": r["budget"],
                "choice": r.get("final_choice"),
                "is_correct": r["is_correct"],
                "answer_letter": r["answer_letter"],
                "subject": r["subject"],
            })
    long = pd.DataFrame(rows)
    wide = long.pivot_table(
        index="sample_id", columns="run", values="choice", aggfunc="first"
    )

    # per-sample stability statistics
    stats = []
    for sid, row in wide.iterrows():
        choices = [c for c in row.dropna().tolist() if c is not None]
        if not choices:
            continue
        counts = pd.Series(choices).value_counts()
        modal = counts.index[0]
        stability = counts.iloc[0] / len(choices)
        truth = long[long["sample_id"] == sid]["answer_letter"].iloc[0]
        subj = long[long["sample_id"] == sid]["subject"].iloc[0]
        stats.append({
            "sample_id": sid,
            "subject": subj,
            "answer_letter": truth,
            "n_runs": len(choices),
            "n_unique_choices": int(len(counts)),
            "modal_choice": modal,
            "stability": stability,
            "modal_correct": int(modal == truth),
        })
    sdf = pd.DataFrame(stats)
    return wide, sdf


# ---------------------------------------------------------------------------
# Metric C: stop-step vs accuracy (pooled)
# ---------------------------------------------------------------------------

def stop_step_accuracy(all_runs: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for run, recs in all_runs.items():
        for r in recs:
            stop = stop_step_of(r)
            if stop is None:
                continue
            rows.append({
                "run": run,
                "budget": r["budget"],
                "stop_step": stop,
                # spontaneous stop_step = N implies (N-1) info requests issued
                # because ANSWER is the N-th step and costs nothing
                "info_requests": stop - 1,
                "is_correct": int(r["is_correct"]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    by_step = df.groupby("info_requests").agg(
        n=("is_correct", "size"),
        acc=("is_correct", "mean"),
    ).round(4).reset_index()
    return by_step, df


# ---------------------------------------------------------------------------
# Metric D: forced -> spontaneous concordance
# ---------------------------------------------------------------------------

def forced_vs_spontaneous(all_runs: dict[str, list[dict]]) -> pd.DataFrame:
    """For each sample, find the smallest budget at which it answered
    spontaneously (gold = 'mature' answer). Compare to choices made under
    forced answer at smaller budgets."""
    by_sample: dict[str, dict[int, dict]] = {}
    for run, recs in all_runs.items():
        for r in recs:
            sid = r["sample_id"]
            by_sample.setdefault(sid, {})[r["budget"]] = r

    rows = []
    for sid, b_map in by_sample.items():
        spon_budgets = [b for b, r in b_map.items() if r["final_action"] == "ANSWER"]
        if not spon_budgets:
            continue
        b_spon = min(spon_budgets)
        spon_choice = b_map[b_spon]["final_choice"]
        spon_correct = b_map[b_spon]["is_correct"]
        for b, r in b_map.items():
            if b >= b_spon:
                continue
            if r["final_action"] != "FORCED_ANSWER":
                continue
            rows.append({
                "sample_id": sid,
                "subject": r["subject"],
                "answer_letter": r["answer_letter"],
                "b_forced": b,
                "b_spon": b_spon,
                "forced_choice": r["final_choice"],
                "spon_choice": spon_choice,
                "agreement": int(r["final_choice"] == spon_choice),
                "forced_correct": int(r["is_correct"]),
                "spon_correct": int(spon_correct),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metric E: stop-step distribution shift
# ---------------------------------------------------------------------------

def stop_step_distribution(all_runs: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for run, recs in all_runs.items():
        budget = recs[0]["budget"] if recs else None
        for r in recs:
            if r["final_action"] != "ANSWER":
                continue
            rows.append({
                "run": run,
                "budget": budget,
                "info_requests": r["n_steps"] - 1,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metric F: zero_info-correct cohort behaviour at higher budget
# ---------------------------------------------------------------------------

def zero_info_cohort(all_runs: dict[str, list[dict]]) -> pd.DataFrame:
    if "zero_info" not in all_runs:
        return pd.DataFrame()
    z_correct = {r["sample_id"] for r in all_runs["zero_info"] if r["is_correct"]}
    z_wrong = {r["sample_id"] for r in all_runs["zero_info"] if not r["is_correct"]}

    rows = []
    for run, recs in all_runs.items():
        if run == "zero_info":
            continue
        budget = recs[0]["budget"]
        for r in recs:
            sid = r["sample_id"]
            cohort = "zi_correct" if sid in z_correct else (
                "zi_wrong" if sid in z_wrong else "unknown"
            )
            rows.append({
                "sample_id": sid,
                "run": run,
                "budget": budget,
                "cohort": cohort,
                "spontaneous": int(r["final_action"] == "ANSWER"),
                "info_requests": r["n_steps"] - 1 if r["final_action"] == "ANSWER" else r["budget_used"],
                "is_correct": int(r["is_correct"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_metric_a(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    s = summary.sort_values("budget")
    ax.plot(s["budget"], s["spontaneous_acc"], marker="o", color="#1f77b4",
            label="spontaneous ANSWER accuracy")
    ax.plot(s["budget"], s["forced_acc"], marker="s", color="#d62728",
            label="FORCED_ANSWER accuracy")
    ax.set_xlabel("Budget")
    ax.set_ylabel("Accuracy")
    ax.set_title("Calibration A — spontaneous vs forced answer accuracy per budget")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = PLOT_DIR / "A_spontaneous_vs_forced_acc.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


def plot_metric_b(stats: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(stats["stability"], bins=np.arange(0.0, 1.05, 0.05),
                 color="#1f77b4", edgecolor="black")
    axes[0].set_xlabel("Per-sample modal-choice fraction across budgets")
    axes[0].set_ylabel("Sample count")
    axes[0].set_title("B — choice stability distribution")
    axes[0].grid(alpha=0.3)

    bins = pd.cut(stats["stability"],
                  bins=[0, 0.5, 0.7, 0.85, 1.001], include_lowest=True,
                  labels=["≤0.5", "0.5-0.7", "0.7-0.85", "0.85-1.0"])
    g = stats.assign(bin=bins).groupby("bin", observed=True)["modal_correct"].agg(["mean", "size"])
    axes[1].bar(g.index.astype(str), g["mean"], color="#2ca02c", edgecolor="black")
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        axes[1].text(i, m + 0.01, f"n={n}", ha="center", fontsize=9)
    axes[1].set_xlabel("Stability bucket")
    axes[1].set_ylabel("Modal-choice accuracy")
    axes[1].set_title("B — accuracy of modal choice by stability")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = PLOT_DIR / "B_choice_stability.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


def plot_metric_c(by_step: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(by_step["info_requests"], by_step["acc"], color="#9467bd",
           edgecolor="black", alpha=0.85)
    for _, row in by_step.iterrows():
        ax.text(row["info_requests"], row["acc"] + 0.01, f"n={int(row['n'])}",
                ha="center", fontsize=8)
    ax.set_xlabel("Info requests issued before spontaneous ANSWER")
    ax.set_ylabel("Accuracy")
    ax.set_title("C — calibration curve: stop-step → accuracy (pooled across budgets)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = PLOT_DIR / "C_stop_step_accuracy.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


def plot_metric_e(dist: pd.DataFrame) -> None:
    pivot = (dist.groupby(["budget", "info_requests"]).size()
             .unstack(fill_value=0))
    # normalize per row to fraction of spontaneous answers at each budget
    norm = pivot.div(pivot.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(norm.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(norm.columns)), [int(c) for c in norm.columns])
    ax.set_yticks(range(len(norm.index)), [int(i) for i in norm.index])
    ax.set_xlabel("Info requests issued before stop")
    ax.set_ylabel("Budget")
    ax.set_title("E — stop-step distribution per budget (row-normalized)")
    fig.colorbar(im, ax=ax, label="fraction of spontaneous answers")
    fig.tight_layout()
    p = PLOT_DIR / "E_stop_step_distribution.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


def plot_metric_f(coh: pd.DataFrame) -> None:
    if coh.empty:
        return
    only = coh[coh["cohort"].isin(["zi_correct", "zi_wrong"])]
    g = (only.groupby(["budget", "cohort"])
         .agg(spon_frac=("spontaneous", "mean"),
              mean_info=("info_requests", "mean"),
              acc=("is_correct", "mean"),
              n=("is_correct", "size"))
         .reset_index())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for cohort, color in [("zi_correct", "#2ca02c"), ("zi_wrong", "#d62728")]:
        s = g[g["cohort"] == cohort].sort_values("budget")
        if s.empty:
            continue
        axes[0].plot(s["budget"], s["spon_frac"], marker="o", color=color, label=cohort)
        axes[1].plot(s["budget"], s["mean_info"], marker="o", color=color, label=cohort)
        axes[2].plot(s["budget"], s["acc"], marker="o", color=color, label=cohort)
    axes[0].set_title("F — spontaneous-stop fraction"); axes[0].set_xlabel("Budget"); axes[0].set_ylabel("Fraction")
    axes[1].set_title("F — mean info requests used"); axes[1].set_xlabel("Budget"); axes[1].set_ylabel("Requests")
    axes[2].set_title("F — accuracy"); axes[2].set_xlabel("Budget"); axes[2].set_ylabel("Accuracy"); axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("F — zero_info-correct vs zero_info-wrong cohorts at higher budget", y=1.02)
    fig.tight_layout()
    p = PLOT_DIR / "F_zero_info_cohort.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    all_runs: dict[str, list[dict]] = {}
    for run, _ in MODEL_RUNS:
        recs = load_jsonl(run)
        if recs is None:
            print(f"skip {run} — no jsonl")
            continue
        all_runs[run] = recs
        print(f"loaded {run}: {len(recs)} records")

    # Metric A
    a_rows = []
    for run, recs in all_runs.items():
        budget = recs[0]["budget"]
        a = metric_a(recs)
        a_rows.append({"run": run, "budget": budget, **a})
    a_df = pd.DataFrame(a_rows).sort_values("budget").round(4)

    # Metric E aggregates fold into the same summary table
    dist_df = stop_step_distribution(all_runs)
    if not dist_df.empty:
        e_means = (dist_df.groupby("budget")["info_requests"]
                   .agg(["mean", "median", "std"]).round(3)
                   .add_prefix("stop_step_"))
        a_df = a_df.merge(e_means, left_on="budget", right_index=True, how="left")

    a_df.to_csv(CAL_DIR / "summary.csv", index=False)
    print(f"\n=== A + E summary ===\n{a_df.to_string(index=False)}\n")
    print(f"wrote {CAL_DIR / 'summary.csv'}")

    # Metric B
    wide, b_stats = cross_budget_stability(all_runs)
    wide.to_csv(CAL_DIR / "per_sample_choice.csv")
    b_stats.to_csv(CAL_DIR / "per_sample_stability.csv", index=False)
    print(f"\n=== B stability summary ===")
    print(f"  mean stability: {b_stats['stability'].mean():.3f}")
    print(f"  perfectly stable (1.0): {(b_stats['stability'] == 1.0).mean():.3f}")
    print(f"  modal-choice accuracy: {b_stats['modal_correct'].mean():.3f}")
    bin_summary = (b_stats.assign(bin=pd.cut(b_stats["stability"],
                                             bins=[0, 0.5, 0.7, 0.85, 1.001],
                                             include_lowest=True,
                                             labels=["<=0.5", "0.5-0.7", "0.7-0.85", "0.85-1.0"]))
                   .groupby("bin", observed=True)["modal_correct"]
                   .agg(["mean", "size"]).round(3))
    print(f"  acc by stability bin:\n{bin_summary}")

    # Metric C
    by_step, c_raw = stop_step_accuracy(all_runs)
    by_step.to_csv(CAL_DIR / "stop_step_acc.csv", index=False)
    print(f"\n=== C stop-step calibration curve ===\n{by_step.to_string(index=False)}")

    # Metric D
    d = forced_vs_spontaneous(all_runs)
    d.to_csv(CAL_DIR / "forced_vs_spon.csv", index=False)
    if not d.empty:
        print(f"\n=== D forced->spontaneous concordance ===")
        print(f"  pairs: {len(d)}")
        print(f"  agreement rate: {d['agreement'].mean():.3f}")
        print(f"  forced accuracy: {d['forced_correct'].mean():.3f}")
        print(f"  spon accuracy:   {d['spon_correct'].mean():.3f}")
        by_b = (d.groupby("b_forced")
                 .agg(n=("agreement", "size"),
                      agreement=("agreement", "mean"),
                      forced_acc=("forced_correct", "mean"),
                      spon_acc=("spon_correct", "mean"))
                 .round(3))
        print(f"  by forced-budget:\n{by_b}")

    # Metric F
    coh = zero_info_cohort(all_runs)
    coh.to_csv(CAL_DIR / "zero_info_cohort.csv", index=False)
    if not coh.empty:
        print(f"\n=== F zero_info cohort behaviour ===")
        only = coh[coh["cohort"].isin(["zi_correct", "zi_wrong"])]
        f_summary = (only.groupby(["budget", "cohort"])
                     .agg(spon_frac=("spontaneous", "mean"),
                          mean_info=("info_requests", "mean"),
                          acc=("is_correct", "mean"),
                          n=("is_correct", "size"))
                     .round(3))
        print(f_summary)

    # Plots
    plot_metric_a(a_df)
    plot_metric_b(b_stats)
    plot_metric_c(by_step)
    plot_metric_e(dist_df)
    plot_metric_f(coh)

    print(f"\nall calibration outputs under {CAL_DIR}/ and {PLOT_DIR}/")


if __name__ == "__main__":
    main()
