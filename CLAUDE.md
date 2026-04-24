# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The parent workspace `CLAUDE.md` at `/mnt/ddn/prod-runs/thyun.park/src/CLAUDE.md` covers workspace-level conventions (uv, per-project venvs, git-ignored artifact dirs). This file is specific to `vlm_budget_eval/`.

## What this project does

**Budget-constrained sequential information-seeking evaluation** for Qwen2.5-VL-7B-Instruct on ScienceQA (MC). The research question is: *given a small fixed information budget, can a VLM pick the right modality (text vs visual) to request next, and stop asking once it has enough?*

Per sample:

1. Initial state shown to the model: the question and its multiple-choice options. No text hint, no image.
2. At each turn the model emits ONE JSON action:
   - `{"action": "ANSWER", "choice": "A"}` — commit and end (cost 0).
   - `{"action": "REQUEST_TEXT"}` — reveal the next sentence from the hint/lecture (cost 1).
   - `{"action": "REQUEST_VISUAL"}` — reveal the next image tile (cost 1).
3. Budget starts at `cfg.budget` (default 6). When it hits 0 (or the model wastes too many requests on exhausted modalities), a forced-answer turn is issued.
4. Outputs record the final choice, correctness, modality mix (text_requests / visual_requests), wasted requests, parse failures, and the full per-step trace.

Visual hints are implemented as **multi-image reveal**: the source image is pre-cropped into a `tile_grid × tile_grid` grid (default 2×2 = 4 tiles), and each `REQUEST_VISUAL` appends the next tile as a separate image input, labeled by spatial position (`top-left`, `bottom-right`, ...). Tile reveal order is seeded-shuffle per sample so the model cannot assume a fixed scan order.

Text hints are sentences split from `hint + lecture`, revealed in natural order by default.

## Layout

```
src/vlm_budget_eval/budget_eval.py  # core engine (EvalConfig + run_episode + main)
scripts/                            # executable drivers (preprocessing, experiment_runner, run_*, analyze_*, summarize_runs)
notebooks/experiment.ipynb          # 37-cell pre-executed demo
docs/figures/                       # headline figures (tracked)
docs/insights/                      # insights.md (EN) + insights_ko.md (KO)
references/                         # project.md/_ko.md (paper-scope plan), roadmap.md/_ko.md (living progress)
configs/, tests/                    # reserved
```

## Pipeline

| Script | When to run | Reads | Writes |
|---|---|---|---|
| `scripts/preprocessing.py` | Once, to build the eval subset | `derek-thomas/ScienceQA` via `datasets.load_dataset` | `preproc/samples.{parquet,jsonl}`, `preproc/preproc_meta.json`, `preproc/images/full/*.png`, `preproc/images/tiles/*.png` |
| `src/vlm_budget_eval/budget_eval.py` (via `scripts/experiment_runner.py`) | Each eval run | `preproc/` | `<out_dir>/predictions.{parquet,csv,jsonl}`, `<out_dir>/summary_{overall,by_subject}.{csv,jsonl}` |

`preproc/` must be rebuilt for this project — the old VQAv2-era files under `preproc/` (`qa_subset.parquet`, `images/original/...`) are NOT consumed by the new pipeline. Safe to delete them.

## Running

```bash
cd /mnt/ddn/prod-runs/thyun.park/src/vlm_budget_eval
uv sync

# 1. one-time preprocessing (downloads ScienceQA + writes tiles)
uv run python scripts/preprocessing.py --max-samples 500 --tile-grid 2

# 2. eval
uv run python scripts/experiment_runner.py
```

`scripts/experiment_runner.py` is a thin `EvalConfig(...); main(cfg)` driver. The core module is importable as `from vlm_budget_eval import EvalConfig, main, run_episode, ...` and also exposes an argparse CLI via `parse_args()` if you'd rather pass flags directly.

Key `EvalConfig` knobs:
- `budget`: total actions across both modalities. `ANSWER` costs 0; each reveal costs 1. Wasted requests (exhausted modality, unparseable JSON) also cost 1.
- `max_eval_rows`: caps samples processed. `-1` = no cap.
- `tile_order`: `"shuffled"` (default, seeded) or `"row_major"`.
- `text_order`: `"natural"` (default) or `"shuffled"`.
- `random_seed`: fixes tile/text shuffling.
- `save_trace`: include the full per-step action trace in `predictions.jsonl`. Adds no per-step images; tiles are already on disk under `preproc/`.
- `max_wasted_before_force`: after this many wasted actions, next turn is forced to ANSWER.

## Correctness criterion

Plain letter match: `final_choice == answer_letter`. No VQA-style normalization — the output is a single MC letter.

## Output schema cheat sheet

- `predictions.*` — one row per sample. Key columns:
  - `final_action`: `ANSWER` | `FORCED_ANSWER` | `PARSE_FAIL`
  - `final_choice`: letter, possibly None if PARSE_FAIL
  - `is_correct`: bool
  - `budget_used`, `text_requests`, `visual_requests`, `wasted_requests`, `parse_failures`
  - `tile_reveal_order`, `text_reveal_order` — indices actually revealed, in order
  - `trace` (jsonl only) — per-step `{step, action, budget_before, raw, revealed_{text|tile}_idx, ...}`
- `summary_overall.*` — one-row aggregate (accuracy, mean budget used, mean text/visual requests, forced/parse-fail rates).
- `summary_by_subject.*` — same metrics grouped by ScienceQA `subject`.

## Model & hardware

- Default model `Qwen/Qwen2.5-VL-7B-Instruct`, `device_map="auto"`, `torch_dtype=bfloat16`.
- The project pins `torch==2.9.1` against the `pytorch-cu130` index (see `pyproject.toml`) — do not bump without matching the CUDA toolchain on the host.
- No test suite. Old `output/`, `output2/` dirs are VQAv2-era and not consumed here.
