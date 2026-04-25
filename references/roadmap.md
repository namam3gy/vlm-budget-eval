# ROADMAP — Budget-conditional VLM agents

> A living progress-tracking document. **The paper-scope plan lives in [`project_ko.md`](./project_ko.md)** (near-immutable reference); this document tracks how far we've come relative to that plan + next unit of work + insight-mining actions. Updated whenever there is a meaningful change.

Last updated: 2026-04-25 (Phase 1a + 1b + 1c + 1d complete)

---

## 0. One-line summary

Phase 0 (diagnostic baseline) + Phase 1a (calibration) + Phase 1b (difficulty stratification) + Phase 1c (abstention proxy) + Phase 1d (anti-calibration curve + masking) from `project.md` **are all complete**. 6-action / SFT+GRPO / Pareto training is still at 0%. Next steps are **Phase 2 (expand action space to 6 + introduce V*Bench/HR-Bench) → Phase 3 (SFT data pipeline) → Phase 4 (GRPO training)**.

Phase 1 uncovered a total of **12 additional insights** (INSIGHTS ⑤–⑯). The four strongest as paper motivation:

- **⑧**: The model requests almost the same amount of information across the zi_correct / zi_wrong cohorts (4.06 vs 4.33 at b=6) — it fails to recognize "I already know this".
- **⑪**: The budget-accuracy curves for zi_correct and zi_wrong cross exactly at b=1 — on easy samples information hurts, on hard samples information helps. This is the mechanism behind the macro plateau (the 0.700 ceiling).
- **⑭**: At b=6, vanilla abstention is **anti-calibrated** (zi_correct abstains 12.1% > zi_wrong 7.3%) — direct evidence that abstention is available but, without training, goes in the wrong direction.
- **⑮**: The calibration → anti-calibration **flip happens exactly at b=3**, the same location as the macro dip in ④ → both phenomena share one mechanism. The strongest single-figure quantitative justification for designing budget-conditioning and abstention reward **together** in the paper.

---

## 1. Current state (Phase 0 complete)

| Item | Status | Notes |
|---|---|---|
| Model | Qwen2.5-VL-7B-Instruct, bf16, H200 | Matches the plan (correct backbone) |
| Data | ScienceQA n=500 | V*Bench / HR-Bench / MM-UPD etc. from the plan **not yet adopted** |
| Action vocab | 3 actions (`ANSWER` / `REQUEST_TEXT` / `REQUEST_VISUAL`) | Plan calls for 6 (`ANSWER`/`ABSTAIN`/`THINK`/`REQUEST_HI_RES`/`ZOOM`/`RETRIEVE`) |
| Cost model | Uniform per-call (1 unit) | Plan calls for token-based deterministic cost |
| Budget conditioning | Only exposed in prompt as `Remaining budget: N` | Not trained (used to diagnose the limits of a vanilla policy) |
| Training | None (inference-only) | Both Stage-1 SFT and Stage-2 GRPO not yet done |
| Pareto curve | Budget sweep b=0..10, **single task** | Multi-benchmark Pareto not yet |
| Abstention | None | One of the core contributions; not implemented |
| Calibration metric | None | Φ, AURC and the other plan metrics not measured |

**Completed artifacts**:
- `budget_eval.py` engine (3-action episode loop, force-answer, wasted accounting)
- `preprocessing.py` (ScienceQA → samples.parquet + 2×2 tile PNG)
- 4 baselines: `zero_info` / `always_text` / `always_visual` / `full_info`
- Budget sweep b=1..10 (model policy)
- `nudge_b6` (visual-favor system prompt variant)
- Analysis artifacts: `output/all_runs_summary.csv`, `output/visual_bias_breakdown.csv`, `output/plots/`
- Notebook `experiment.ipynb` (37 cells pre-executed) + `insights_ko.md`

---

## 2. Four things Phase 0 showed — candidates for the paper's motivation paragraph

These diagnostic findings plug straight into the motivation: **"a vanilla VLM fails to understand / exploit the budget signal → we need a learned budget-conditional policy"**.

1. **The model has a strong text bias** — `main_b6` uses text 3.28 / visual 0.87. Yet an automatic `always_visual` policy is +2.4pp higher. The model's policy runs opposite to information efficiency.
2. **Budget-accuracy curve is non-monotonic** — local max at b=2 → dip at b=3 → saturation at b≥7. Same seed, same data, yet accuracy swings with budget alone due to path-dependence. Direct evidence that **the model cannot read the budget signal**.
3. **A single prompt nudge fully flips behavior but only adds +0.4pp accuracy** — modality preference flips instantly with one line (text 3.28→0.01) but accuracy only moves 0.656→0.660. Quantitative evidence of the **limits of prompt-only intervention.**
4. **43 visual-only-correct samples (8.6%) that main misses** — even on cases where vision was the answer, the model uses on average text 3.91 / visual 0.81. In language science this climbs to 21.4%, the worst. Simultaneously demonstrates a domain effect and a bias effect.

(Details in [`insights_ko.md`](../docs/insights/insights_ko.md))

---

## 3. Gap matrix vs. the plan

| project_ko.md element | Current | Work needed to close the gap |
|---|---|---|
| 6-action vocab | 3-action | Add `ABSTAIN` / `THINK` / `ZOOM(bbox)` / `REQUEST_HI_RES` (Phase 2) |
| Token-based cost | Per-call cost | Accurate vision-token count and CoT-token count (Phase 2) |
| Budget conditioning π(a\|s,B) | Prompt-injection only | Learn via SFT + GRPO (Phase 3-4) |
| Multi-benchmark (V*Bench, HR-Bench, MM-UPD, POPE, MM-AQA, MMBench, MMMU) | ScienceQA only | Add V*Bench + MM-UPD at minimum (Phase 1c, Phase 5) |
| Stage-1 SFT (LoRA r=16) | None | Synthesize sufficiency-labeled data + tool-use trajectories (Phase 3) |
| Stage-2 GRPO (budget-conditional) | None | Reward function, group sampling, λ tuning (Phase 4) |
| Pareto curve (multi-bench, AUPC) | Single-task 1D sweep | Remeasure after training (Phase 5) |
| Abstention metric (Φ, AURC) | None | Introduce MM-UPD + masked-image diagnostic (Phase 1c, Phase 5) |
| Hallucination regression (POPE, CHAIR) | None | Post-training non-regression check (Phase 5) |

---

## 4. Phase roadmap

Each phase is an independent deliverable. Phase 1 can proceed without training, so it's a fast way to pile up paper-appendix material. Code changes grow from Phase 2 onward, and Phase 3-4 are where serious GPU-hours get spent.

### Phase 1 — Diagnostic hardening (no training, 1 week)

**Goal**: Harden the current diagnostics and secure data for baselines / ablations. Feeds both Section 3 (motivation) and Section 5 (baselines) of the paper.

- [x] **1a. Calibration analysis** ✅ 2026-04-24 — `analyze_calibration.py` + `output/calibration/`. Six metrics, results written up in INSIGHTS ⑤–⑨. Headlines: ⑤ early-stop is only calibrated at b≤3, ⑥ cross-budget stability = a free confidence proxy, ⑦ per-stop accuracy peaks at info=2 and drops −20pp at info=3, ⑧ zi_correct / zi_wrong cohorts request the same amount of information + extra info hurts easy samples, ⑨ 84% agreement between forced and spontaneous answers.
- [x] **1b. Difficulty stratification** ✅ 2026-04-24 — `analyze_difficulty.py` + `output/difficulty/`. Adds INSIGHTS ⑩–⑫. Headlines: ⑩ modality bias hurts only on zi_wrong (on zi_correct, always_text == always_visual at exactly 0.911), ⑪ the zi_correct and zi_wrong budget curves cross exactly at b=1 (clean cross-over), ⑫ natural science is high-volatility, social science is low-volatility — optimal budget differs by domain.
- [x] **1c. Abstention proxy measurement** ✅ 2026-04-24 — Added an `enable_abstain` flag + `SYSTEM_INSTRUCTION_WITH_ABSTAIN` to `budget_eval.py`. Ran `abstain_b0` / `abstain_b6` on 500 samples each via `run_abstention.py`. Produced four metrics (A summary, B cohort xtab, C Phi sweep, D cohort-aligned comparison) via `analyze_abstention.py`. Adds INSIGHTS ⑬–⑭. Headlines: ⑬ the "should I answer" signal nearly coincides with zero-info prior knowledge (selectivity only +3.3pp), ⑭ at b=6 vanilla abstention is **anti-calibrated** (zi_correct 12.1% > zi_wrong 7.3% abstain).
  - ⚠️ **Scope-cut**: The original MM-UPD subset / image-masked ScienceQA sufficiency-known mini-set is **skipped**. We use `zi_correct` / `zi_wrong` cohorts as a sufficiency proxy. Direct test on unanswerable-by-construction stimuli deferred to Phase 1d or the start of Phase 2 — see backlog I13 below.
- [x] **1d. Anti-calibration curve + sufficiency-known masking** ✅ 2026-04-25 — `run_abstention_sweep.py` ran abstain at b=1..5,7,8,10 (I14); `preprocessing_masked.py` built a 100-sample (50/50 cohort) image-masked variant under `preproc_masked/`, and `run_abstention_masked.py` ran abstain at b=0/4/6 on the masked set (I13); `analyze_abstention_phase1d.py` integrates both. Adds INSIGHTS ⑮–⑯. Headlines: ⑮ the calibration→anti-calibration **flip happens exactly at b=3**, same location as ④'s macro dip → both phenomena share one mechanism. ⑯ image masking changes abstain rate by Δ ≤ 2pp → vanilla abstention is image-level sufficiency-blind.
  - The original 1d candidate (second VLM validation) is moved to a Phase 2 / 5 sanity step. Whether the text bias and the b=3 flip reproduce on another VLM is needed for paper generalization, but Phase 1c+1d already make the ScienceQA diagnostic solid enough.

**Exit criterion for Phase 1**: 1a + 1b + 1c + 1d all complete. INSIGHTS now carries ⑤–⑯, twelve findings. Paper motivation candidates: ⑧, ⑪, ⑭, ⑮ as the headline figures.

### Phase 2 — Action-space expansion (no training, 1–1.5 weeks)

**Goal**: Lay down the plan's 6-action vocab in an **inference-only** form first. If the new actions "exist" but the vanilla model fails to use them well, that itself is more motivation.

- [ ] **2a. Officially introduce the `ABSTAIN` action** — terminates at cost 0 on any task
- [ ] **2b. `ZOOM(bbox)` action** — propose bbox candidates via Qwen2.5-VL's native grounding → add the cropped tile as input. Account for token cost.
- [ ] **2c. `REQUEST_HI_RES` action** — start with ¼-res input, tokens ↑ on full-res request
- [ ] **2d. `THINK(text)` action** — one free-form CoT segment, counted in output-token cost
- [ ] **2e. Cost-model refactor** — uniform-1 → token-based deterministic. Vision tokens = `floor(H'·W'·grid_factor)`, computed exactly.
- [ ] **2f. One new benchmark** — V*Bench OR HR-Bench 4K (a domain where a high-res zoom action pays off)
- [ ] **2g. 6-action vanilla baseline re-measurement** — re-run budget sweep under the new action vocab. Same as the plan's baseline (1) "vanilla budget forcing".

**Exit criterion**: 6-action interface is stabilized, sample-level predictions land on the new benchmark.

### Phase 3 — SFT data pipeline (1 week)

**Goal**: Build ~15k tool-use trajectories for Stage-1 LoRA SFT. Follows Stage-1 of the plan directly.

- [ ] **3a. Sufficiency-labeled subset** — Take 5k from VQAv2 / GQA, apply 4×4 masking, produce 1/16 · 4/16 · 8/16 · 16/16 unmask variants, label sufficiency via majority vote of a teacher VLM (GPT-4o or Qwen2.5-VL-72B)
- [ ] **3b. Teacher-forced trajectory synthesis** — generate 6-action solution traces on the V*Bench training split + a portion of Visual CoT (438k), keep only the correct ones
- [ ] **3c. Rule-based trajectories** — synthesize `ZOOM` sequences from the ground-truth bboxes of V*Bench / Visual CoT
- [ ] **3d. Unified JSONL format** — `{messages, tools, expected_action_sequence}` schema
- [ ] **3e. Held-out 500-sample val set split**

**Exit criterion**: 15k train + 500 val JSONL generated, distribution stats (per-action frequency, average length) reported.

### Phase 4 — Stage-1 SFT → Stage-2 GRPO (1.5–2 weeks, serious GPU)

**Goal**: Stage-1 + Stage-2 from the plan. Train the budget-conditional policy.

- [ ] **4a. Stage-1 LoRA SFT** — Qwen2.5-VL-7B + LoRA (r=16, α=32) on the last 8 blocks + LM head, 2 epochs, batch 32, 4×H200, ~8h
- [ ] **4b. Reward function implementation** — `R = 1[correct]·r_acc - λ_cost·max(0, cost-B) - λ_abs·1[wrong abstain] + λ_cal·1[right abstain]`
- [ ] **4c. GRPO trainer integration** — `trl` GRPOTrainer or `verl` base, group size G=8, B ∈ {512, 1024, 2048, 4096, 8192}
- [ ] **4d. λ sweep** — 3-point grid for each of λ_cost, λ_abs, λ_cal on the 500-sample val set
- [ ] **4e. Main training run** — 4-8×H200, ~48-72h, ~30k samples
- [ ] **4f. Judge** — Qwen2.5-VL-72B or GPT-4o (model distinct from the policy to avoid contamination)

**Exit criterion**: Stage-2 training converges, val Pareto moves up-right vs. vanilla.

### Phase 5 — Ablations + final evaluation + paper draft (1 week)

- [ ] **5a. No-budget-conditioning ablation** — input with shuffled B. Specified in the plan.
- [ ] **5b. No-abstention ablation** — retrain under a 5-action vocab without abstain
- [ ] **5c. SFT-only ablation** — Stage-1 only, skip Stage-2
- [ ] **5d. Final Pareto** — B ∈ {256, 512, 1024, 2048, 4096, 8192, ∞} sweep on V*Bench, HR-Bench (4K/8K), MM-UPD, POPE, MM-AQA (if possible), MMBench, MMMU
- [ ] **5e. Calibration / abstention metrics** — Φ, AURC, abstention F1, hallucination rate (POPE adv + CHAIR_i)
- [ ] **5f. AUPC computation + headline plot** — paper figure 1
- [ ] **5g. First draft**

**Exit criterion**: NeurIPS 2026 submission ready.

---

## 5. Insight-mining backlog

Not a separate phase, but things to keep rolling throughout to generate insights:

| # | Action | Artifact | Which phase will fold it in |
|---|---|---|---|
| I1 | Visualize the modality-switching pattern (text→visual transition matrix) from traces | plot + one paragraph of analysis | Phase 1b option (1a done, no meaningful outcome) |
| I2 | Analyze model prompts right before `wasted_request` events (does it recognize modality exhaustion?) | qualitative table | Phase 1c by-product |
| I3 | Diff the b=2/3/5/7 traces of the same sample (5 case studies of path divergence) | notebook cells + 5 cases | Phase 1b by-product (from the 16+59 unstable-bin samples of metric B) |
| I4 | subject × action × correctness 3-way table | csv | Phase 1b by-product |
| I5 | Separate "over-spending on easy samples" vs "under-spending on hard samples" asymmetry | two plots | ✅ Phase 1a metric F already answers it — both easy and hard samples request the same amount (symmetric over-spend) |
| I6 | Trade-off curve between abstention and budget (vanilla, pre-training) | plot | Phase 1c → Phase 2 |
| I7 | Do the 4 + 5 findings reproduce on a different VLM? | 3×9 comparison table | Phase 1d |
| I8 | Which actions does the model barely use even under the 6-action vocab? | per-action frequencies | Phase 2g |
| I9 | Does the text bias vanish in a high-res domain (V*Bench)? | budget-sweep comparison | Phase 2f |
| I10 | Can SFT alone make the model follow the budget signal (without RL)? | Pareto compare | Phase 4a vs 4e |
| I11 | Can choice stability (metric B) be used as a train-time abstention target? | SFT-label synthesis experiment | Phase 3a |
| I12 | Does the "more info hurts" phenomenon (⑧, zi_correct b=1→b=6 −5pp) appear on other datasets? | b=1 vs b=6 comparison plot | Phase 1c-1d or Phase 2f |
| I13 | Properly build the sufficiency-known mini-set skipped in Phase 1c (MM-UPD subset or 100 image-masked ScienceQA samples) → check whether vanilla abstention is calibrated on "unanswerable" samples | `preproc_masked/` + `run_abstention_masked.py` + comparison plot | ✅ Phase 1d (2026-04-25). Δ ≤ 2pp → vanilla abstention is image-level sufficiency-blind. The decisive test moves to I15 because of the ScienceQA caveat. |
| I14 | How does the anti-calibration (⑭) evolve across budgets b∈{1..10} — did it emerge sequentially? | abstain run per b → cohort-rate plot | ✅ Phase 1d (2026-04-25). The flip is exactly at b=3, anti-calibration persists, gap reaches −6.9pp at b=10. |
| I15 | Decisive version of I13: V*Bench / HR-Bench (image-required-by-construction) + image-masked variants → directly test whether vanilla abstention tracks image-level sufficiency | masked V*Bench/HR-Bench 50–100 samples + abstain run + comparison | Early Phase 2 (after benchmark introduction) |

---

## 6. Recent decisions / open questions

- **Q1**: Keep ScienceQA as the testbed, or move primary to V*Bench? → Current answer: keep ScienceQA through Phase 1, promote V*Bench to primary from Phase 2, demote ScienceQA to an OOD check.
- **Q2**: Single-GPU iteration vs 4-GPU main training, split? → Phase 1-3 on a single H200, Phase 4 onward on 4-8 H200.
- **Q3**: 7B vs 3B for fast iteration? → A one-shot 3B dry-run before kicking off the 7B main training in Phase 4, then 7B.
- **Q4**: Judge model? → Try Qwen2.5-VL-72B first, fall back to InternVL 2.5-38B if API quota / time becomes a problem.
- **Q5**: trl vs verl? → Decide again at the end of Phase 3.

---

## 7. Next unit of work (now-doing)

Phase 1 is fully closed. Next: **start Phase 2 — 6-action vocab expansion + introduction of V*Bench / HR-Bench**.

Recommended order:
1. **2f. Introduce one new benchmark** — V*Bench (high-res detail detection) OR HR-Bench 4K. Data download + preprocessing + 3-action vanilla baseline first.
2. **2g. 6-action vanilla baseline (current vocab)** — budget sweep on the new benchmark. Confirm whether the b=3 dip in ④ / anti-calibration in ⑭ generalize off ScienceQA — addresses backlog I9 / I12 / I15 simultaneously.
3. **2a–2d. Add actions**: ABSTAIN (already there) → ZOOM(bbox) (use Qwen2.5-VL native grounding) → REQUEST_HI_RES → THINK
4. **2e. Cost-model refactor** — uniform-1 → token-based deterministic.

Total Phase 2 time: ~1–1.5 weeks. Core artifact: stable 6-action interface + sample-level predictions on V*Bench or HR-Bench. Phase 3 SFT-data synthesis can then use the new benchmark's correct trajectories.

Open backlog handled in parallel during Phase 2:
- **I9, I12, I15**: Whether ⑧ / ⑪ / ⑭ / ⑮ reproduce on the new benchmark → strengthens the paper's generalization evidence.
- **I7**: Whether the same diagnostics reproduce on a second VLM (LLaVA-OneVision-7B or InternVL2.5-8B).

---

## 8. Change history

| Date | Change |
|---|---|
| 2026-04-24 | Initial draft. Phase 0 completion captured, gap matrix vs project_ko.md drafted, 5-phase roadmap + insight backlog defined. |
| 2026-04-24 | Phase 1a complete. Added `analyze_calibration.py`; six-metric results land in `output/calibration/` + `output/plots/calibration/`. Added INSIGHTS ⑤–⑨. Next work switched to Phase 1b. I5 in the backlog marked ✅; I11 / I12 added. |
| 2026-04-24 | Phase 1b complete. Added `analyze_difficulty.py`; cohort curves + modality mix + subject crosstab + delta-from-b1 produced. INSIGHTS ⑩–⑫ added (especially ⑪ cross-over at b=1, a paper-figure candidate). |
| 2026-04-24 | Phase 1c complete (scope-cut noted). Added `enable_abstain` + `SYSTEM_INSTRUCTION_WITH_ABSTAIN` to `budget_eval.py`. New: `run_abstention.py` + `analyze_abstention.py`. Ran `abstain_b0` + `abstain_b6` at 500 samples. INSIGHTS ⑬–⑭ added — ⑭ anti-calibration is direct evidence for the necessity of budget-conditional training. Sufficiency-known mini-set deferred to backlog I13. Five headline figures copied to `docs/figures/`. Committed Phase 1a+1b+1c as a bundle. |
| 2026-04-25 | Phase 1d complete. New: `run_abstention_sweep.py` (8 additional abstain runs at b=1..5,7,8,10) + `preprocessing_masked.py` (100-sample image-masked variant under `preproc_masked/`) + `run_abstention_masked.py` (b=0/4/6 on the masked set) + `analyze_abstention_phase1d.py`. Findings: I13 (sufficiency-known masking, ⑯) and I14 (anti-calibration curve, ⑮). **⑮ The flip is exactly at b=3, the same location as ④'s macro dip** — strongest single-figure paper motivation candidate. ⑯ Δ ≤ 2pp under masking → vanilla abstention is image-blind. Two new headline plots copied to `docs/figures/abstention_phase1d_*.png`. New backlog I15 (the decisive sufficiency test moves to V*Bench / HR-Bench). |