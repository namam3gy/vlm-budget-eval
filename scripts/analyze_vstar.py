"""Phase 2.0d: V*Bench baseline analysis.

Mirrors the structure of analyze_runs.py / analyze_difficulty.py but on the
new V*Bench preproc. Key questions:

  A. Policy comparison (zero_info / b=2 / b=4 / always_visual / full_info) —
     does the rank order match ScienceQA's, or does V*Bench (vision-only,
     image-required) flip it?
  B. Modality mix — does the model still default to text on a domain where
     no text exists? (mean_text_requests / mean_wasted)
  C. Cohort behavior — split by V*Bench's vstar_zero_info correctness.
     Does the ⑪ b=1 cross-over reproduce here? Does ⑭/⑮ anti-calibration?
  D. Subject (direct_attributes vs relative_position) crosstab.

Outputs:
  output/vstar/policy_summary.csv         -- A
  output/vstar/cohort_curve.csv           -- C
  output/vstar/subject_xtab.csv           -- D
  output/plots/vstar/{A,B,C,D}_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

WORKSPACE = Path("/mnt/ddn/prod-runs/thyun.park/src/vlm_budget_eval")
OUT_ROOT = WORKSPACE / "output"
VSTAR_DIR = OUT_ROOT / "vstar"
PLOT_DIR = OUT_ROOT / "plots" / "vstar"
VSTAR_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RUNS: list[tuple[str, int, str]] = [
    ("vstar_zero_info",        0, "zero_info"),
    ("vstar_b2",               2, "b=2 model"),
    ("vstar_b4",               4, "b=4 model"),
    ("vstar_always_visual_b4", 4, "always_visual b=4"),
    ("vstar_full_info",        0, "full_info"),  # budget field is 0 for full_info
]


def load_jsonl(run: str) -> list[dict] | None:
    p = OUT_ROOT / run / "predictions.jsonl"
    if not p.exists():
        return None
    with p.open() as f:
        return [json.loads(line) for line in f]


def to_df(recs: list[dict], run: str, label: str) -> pd.DataFrame:
    rows = []
    for r in recs:
        rows.append({
            "sample_id": r["sample_id"],
            "subject": r["subject"],
            "topic": r.get("topic", ""),
            "run": run,
            "label": label,
            "is_correct": int(r["is_correct"]),
            "final_action": r["final_action"],
            "final_choice": r.get("final_choice"),
            "text_requests": r.get("text_requests", 0),
            "visual_requests": r.get("visual_requests", 0),
            "wasted_requests": r.get("wasted_requests", 0),
            "budget_used": r.get("budget_used", 0),
            "n_steps": r.get("n_steps", 1),
        })
    return pd.DataFrame(rows)


def cohort_map(zero_run: str = "vstar_zero_info") -> pd.Series:
    recs = load_jsonl(zero_run) or []
    rows = []
    for r in recs:
        rows.append({"sample_id": r["sample_id"], "is_correct": int(r["is_correct"])})
    df = pd.DataFrame(rows)
    return df.set_index("sample_id")["is_correct"].map(
        lambda x: "zi_correct" if x else "zi_wrong"
    )


# ---------------------------------------------------------------------------
# A. Policy summary
# ---------------------------------------------------------------------------

def policy_summary(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for run, df in dfs.items():
        rows.append({
            "run": run,
            "n": len(df),
            "accuracy": float(df["is_correct"].mean()),
            "mean_text": float(df["text_requests"].mean()),
            "mean_visual": float(df["visual_requests"].mean()),
            "mean_wasted": float(df["wasted_requests"].mean()),
            "forced_rate": float((df["final_action"] == "FORCED_ANSWER").mean()),
        })
    return pd.DataFrame(rows).round(4)


def plot_a(summary: pd.DataFrame, label_map: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    summary = summary.copy()
    summary["label"] = summary["run"].map(label_map).fillna(summary["run"])
    summary = summary.sort_values("accuracy")
    bars = ax.barh(summary["label"], summary["accuracy"], color="#1f77b4", edgecolor="black")
    for bar, acc in zip(bars, summary["accuracy"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.0); ax.set_xlabel("Accuracy")
    ax.set_title("A — V*Bench accuracy by policy (n=191)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    p = PLOT_DIR / "A_policy_accuracy.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# B. Modality mix (does model still ask for text?)
# ---------------------------------------------------------------------------

def plot_b(summary: pd.DataFrame, label_map: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    summary = summary.copy()
    summary["label"] = summary["run"].map(label_map).fillna(summary["run"])
    width = 0.27
    x = range(len(summary))
    ax.bar([i - width for i in x], summary["mean_text"], width,
           label="mean text req (no text exists)", color="#d62728")
    ax.bar(list(x), summary["mean_visual"], width,
           label="mean visual req", color="#2ca02c")
    ax.bar([i + width for i in x], summary["mean_wasted"], width,
           label="mean wasted", color="#ff7f0e")
    ax.set_xticks(list(x)); ax.set_xticklabels(summary["label"], rotation=10)
    ax.set_title("B — modality mix per policy (V*Bench has no text hints)")
    ax.set_ylabel("Mean requests per sample")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = PLOT_DIR / "B_modality_mix.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# C. Cohort behavior (zi_correct vs zi_wrong on V*Bench)
# ---------------------------------------------------------------------------

def cohort_curve(dfs: dict[str, pd.DataFrame], cohort: pd.Series) -> pd.DataFrame:
    rows = []
    for run, df in dfs.items():
        d = df.copy()
        d["cohort"] = d["sample_id"].map(cohort)
        d = d[d["cohort"].notna()]
        for cohort_name, sub in d.groupby("cohort"):
            rows.append({
                "run": run,
                "cohort": cohort_name,
                "n": len(sub),
                "accuracy": float(sub["is_correct"].mean()),
                "mean_text": float(sub["text_requests"].mean()),
                "mean_visual": float(sub["visual_requests"].mean()),
                "mean_wasted": float(sub["wasted_requests"].mean()),
            })
    return pd.DataFrame(rows).round(4)


def plot_c(curve: pd.DataFrame, label_map: dict[str, str]) -> None:
    if curve.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    palette = {"zi_correct": "#2ca02c", "zi_wrong": "#d62728"}
    runs = sorted(curve["run"].unique(), key=lambda r: list(label_map).index(r))
    width = 0.35
    x = range(len(runs))
    for i, cohort in enumerate(["zi_correct", "zi_wrong"]):
        ys = []
        ns = []
        for run in runs:
            sel = curve[(curve["run"] == run) & (curve["cohort"] == cohort)]
            ys.append(float(sel["accuracy"].iloc[0]) if len(sel) else 0.0)
            ns.append(int(sel["n"].iloc[0]) if len(sel) else 0)
        offset = (i - 0.5) * width
        ax.bar([j + offset for j in x], ys, width, label=cohort, color=palette[cohort])
        for j, (y, n) in enumerate(zip(ys, ns)):
            ax.text(j + offset, y + 0.01, f"{y:.2f}\nn={n}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([label_map.get(r, r) for r in runs], rotation=10)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title("C — V*Bench cohort accuracy (split by zero_info correctness)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = PLOT_DIR / "C_cohort_accuracy.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# D. Subject crosstab (direct_attributes vs relative_position)
# ---------------------------------------------------------------------------

def subject_xtab(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for run, df in dfs.items():
        for topic, sub in df.groupby("topic"):
            rows.append({
                "run": run,
                "topic": topic,
                "n": len(sub),
                "accuracy": float(sub["is_correct"].mean()),
            })
    return pd.DataFrame(rows).round(4)


def plot_d(xtab: pd.DataFrame, label_map: dict[str, str]) -> None:
    if xtab.empty:
        return
    runs = sorted(xtab["run"].unique(), key=lambda r: list(label_map).index(r))
    topics = ["direct_attributes", "relative_position"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.35
    x = range(len(runs))
    palette = {"direct_attributes": "#1f77b4", "relative_position": "#ff7f0e"}
    for i, t in enumerate(topics):
        ys = []
        ns = []
        for run in runs:
            sel = xtab[(xtab["run"] == run) & (xtab["topic"] == t)]
            ys.append(float(sel["accuracy"].iloc[0]) if len(sel) else 0.0)
            ns.append(int(sel["n"].iloc[0]) if len(sel) else 0)
        offset = (i - 0.5) * width
        ax.bar([j + offset for j in x], ys, width, label=t, color=palette[t])
        for j, (y, n) in enumerate(zip(ys, ns)):
            ax.text(j + offset, y + 0.01, f"{y:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([label_map.get(r, r) for r in runs], rotation=10)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title("D — V*Bench accuracy by topic (direct_attributes vs relative_position)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = PLOT_DIR / "D_topic_xtab.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    label_map = {r: lbl for r, _, lbl in RUNS}
    dfs: dict[str, pd.DataFrame] = {}
    for run, _, lbl in RUNS:
        recs = load_jsonl(run)
        if recs is None:
            print(f"skip {run} -- no jsonl")
            continue
        dfs[run] = to_df(recs, run, lbl)
        print(f"loaded {run}: {len(dfs[run])} rows")

    if not dfs:
        print("no runs loaded -- did run_vstar_baseline.py finish?")
        return

    # A
    summary = policy_summary(dfs)
    summary.to_csv(VSTAR_DIR / "policy_summary.csv", index=False)
    print(f"\n=== A policy summary ===\n{summary.to_string(index=False)}")
    plot_a(summary, label_map)
    plot_b(summary, label_map)

    # C
    cohort = cohort_map() if "vstar_zero_info" in dfs else pd.Series(dtype=object)
    if not cohort.empty:
        cdf = cohort_curve(dfs, cohort)
        cdf.to_csv(VSTAR_DIR / "cohort_curve.csv", index=False)
        print(f"\n=== C cohort breakdown ===\n{cdf.to_string(index=False)}")
        plot_c(cdf, label_map)

    # D
    xt = subject_xtab(dfs)
    xt.to_csv(VSTAR_DIR / "subject_xtab.csv", index=False)
    print(f"\n=== D subject crosstab ===\n{xt.to_string(index=False)}")
    plot_d(xt, label_map)

    print(f"\nall outputs under {VSTAR_DIR}/ and {PLOT_DIR}/")


if __name__ == "__main__":
    main()
