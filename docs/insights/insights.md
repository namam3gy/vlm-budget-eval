# Insights Note — Budget-Constrained Sequential Information-Seeking

> Results from evaluating Qwen2.5-VL-7B-Instruct on 500 ScienceQA samples with a budget sweep (b=0…10) + 4 baseline policies. See [`experiment.ipynb`](../../notebooks/experiment.ipynb) for the detailed walkthrough and [`output/`](../../output) for the raw data.
>
> **Figure note**: The 5 headline figures are copied (tracked) under `docs/figures/` and the main text links to those. All other auxiliary-plot references point to gitignored paths under `output/plots/` — they are regenerated locally by re-running `analyze_calibration.py` / `analyze_difficulty.py` / `analyze_abstention.py`.

## Terminology: Sweep

**Parameter sweep** = a method where every other condition (model · data · seed · prompt · policy code) is held fixed, **one hyperparameter is run at several values**, and the response of a metric is plotted.

In this project, only `budget` is swept — 11 points (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, ∞), each at 500 samples × the same seed (42). The output is the accuracy curve in §④ below.

Concretely, inside `run_dense_sweep.py`:

```python
BUDGETS = [1, 2, 3, 5, 7, 10]
for b in BUDGETS:
    cfg = EvalConfig(policy="model", budget=b, random_seed=42, ...)
    for sample in samples:
        run_episode(model, processor, sample, cfg, rng)   # same 500 samples
    save(...)
```

Notebook `§3-a` contains a trace comparison of **how the same sample (`sqa_000001`) is solved differently under budgets 1/2/3/5/6/7/10**.

---

## 0. At a glance

### Budget vs. accuracy
![accuracy vs budget](../figures/accuracy_vs_budget.png)

### Modality usage by policy
![modality mix](../figures/modality_mix.png)

### Full comparison table

Model-policy budget sweep (b=0…10) + baselines + full_info ceiling + nudge:

| run | budget | accuracy | text | visual | wasted | forced |
|---|---:|---:|---:|---:|---:|---:|
| `zero_info` | 0 | 0.562 | 0.00 | 0.00 | 0.00 | 0.866 |
| `sweep_b1` | 1 | 0.622 | 0.95 | 0.03 | 0.00 | 0.930 |
| `sweep_b2` | 2 | **0.646** | 1.57 | 0.31 | 0.00 | 0.916 |
| `sweep_b3` | 3 | 0.626 | 2.08 | 0.49 | 0.00 | 0.722 |
| `sweep_b4` | 4 | 0.638 | 2.54 | 0.64 | 0.00 | 0.542 |
| `sweep_b5` | 5 | 0.654 | 2.91 | 0.79 | 0.01 | 0.432 |
| **`main_b6`** | **6** | **0.656** | **3.28** | **0.87** | 0.02 | 0.316 |
| `sweep_b7` | 7 | 0.666 | 3.52 | 1.01 | 0.03 | 0.256 |
| `sweep_b8` | 8 | 0.668 | 3.67 | 1.10 | 0.04 | 0.194 |
| `sweep_b10` | 10 | 0.666 | 3.92 | 1.13 | 0.05 | 0.136 |
| **baseline** | | | | | | |
| `always_text` | 6 | 0.638 | 5.64 | 0.00 | 0.00 | 1.000 |
| **`always_visual`** | **6** | **0.680** | **0.00** | **4.00** | 0.00 | 1.000 |
| **`nudge_b6`** | **6** | **0.660** | **0.01** | **3.72** | 1.00 | 0.410 |
| `full_info` (∞) | — | **0.700** | 11.72 | 4.00 | 0.00 | 0.978 |

---

## Quick glossary

| Term | Meaning |
|---|---|
| **VLM** | Vision-Language Model. Image + text input → text output. |
| **MC** | Multiple Choice. Single-letter A/B/C/D answer format. |
| **Budget** | Total number of information requests per sample. ANSWER cost 0, REQUEST_* cost 1. |
| **Modality** | Type of information source. Here we have two: `text` (sentence-level hint) and `visual` (image tile). |
| **Tile** | Patch obtained by cutting the original image into an N×N grid (default 2×2 = 4). Fed one image at a time. |
| **Action** | One-line JSON emitted by the model at each step. `ANSWER` / `REQUEST_TEXT` / `REQUEST_VISUAL`. |
| **Force-answer** | When budget=0 or the wasted cap is exceeded, the system injects a "now answer" prompt. |
| **Wasted request** | Case where only the budget is burned — e.g. requesting an already-exhausted modality, or emitting unparseable JSON. |
| **Refused answer** | If the model still won't emit ANSWER after force-answer, the episode terminates. An infinite-loop cap. |
| **Policy** | Actor that chooses actions. `model` (model decides), `always_text` / `always_visual` (auto single modality), `full_info` (all information revealed at once). |
| **Floor / Ceiling** | `zero_info` (0.562) is the prior-knowledge floor with no information; `full_info` (0.700) is the ceiling with all information available. |
| **Sweep** | A method that holds every other condition (model · data · seed · policy implementation) fixed and **runs only one hyperparameter across multiple values** to obtain a response curve. In this project only `budget` is swept (11 points: 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, ∞). |

---

## ① The model has a strong text bias, and it is costing it

On `main_b6` the model requests, on average, **3.28 text requests / 0.87 visual requests** — preferring text by roughly 3.8×.

Yet the accuracy comparison shows the model picks the wrong side:

| Policy (b=6) | text req | visual req | accuracy |
|---|---:|---:|---:|
| `always_text` | 5.64 | 0 | 0.638 |
| `main_b6` (model) | 3.28 | 0.87 | 0.656 |
| `always_visual` | 0 | 4.00 | **0.680** |

A policy that automatically uses only vision beats the model's adaptive policy by **+2.4pp**. A single vision tile carries more information per unit budget than a single text sentence, yet the model does the opposite.

**Why text bias?** Plausible causes:
- Text-based reasoning is more frequently rewarded in pre-training / instruct stages
- ScienceQA text hints are in a "lecture" style that is more familiar to the model
- The prompt is modality-neutral, so the default tends to fall toward text

---

## ② 43 visual_only-correct cases (8.6%) are missed by main

Decomposing 500 samples into 4 buckets by (`main_b6` × `always_visual`) correctness:

| bucket | n | Meaning |
|---|---:|---|
| both_right | 297 | Both correct |
| both_wrong | 129 | Both incorrect |
| **visual_only** | **43** | Only always_visual correct → vision should have been used more |
| main_only | 31 | Only main correct → solved via text |

In the `visual_only` cohort, **the model uses on average 3.91 text / 0.81 visual requests** — it floods with text even on cases where vision was the answer.

Visual_only rate by subject:

| subject | n | visual_only rate |
|---|---:|---:|
| language science | 14 | **21.4%** |
| natural science | 410 | 9.3% |
| social science | 76 | 2.6% |

Worst in language science (primarily chart / map / photo based), nearly absent in social science (text-dominant). A clear domain effect.

---

## ③ A prompt nudge flips behavior but barely moves accuracy

With `run_nudge.py`, the following guidance is added to the system prompt:

> "ScienceQA images are usually cut into 4 tiles and often carry decisive visual information. Text hints are split sentence-by-sentence and are frequently background context. **Do not default to REQUEST_TEXT.**"

Result:

| | text req | visual req | wasted | accuracy |
|---|---:|---:|---:|---:|
| `main_b6` (no nudge) | 3.28 | 0.87 | 0.02 | 0.656 |
| `nudge_b6` (with nudge) | **0.01** | **3.72** | **1.00** | 0.660 |
| `always_visual` | 0 | 4.00 | 0 | **0.680** |

**Observations**:
- Modality preference flips almost 100% with a single line of prompt (text 3.28 → 0.01).
- Overall accuracy only moves +0.4pp (0.656 → 0.660).
- `wasted=1.0` — after exhausting all 4 tiles, the model still emits "REQUEST_VISUAL". It does not notice modality exhaustion well.
- Falls short of `always_visual` (0.680) → the wasted penalty eats into the gain by exactly that much.

Effect of the nudge by subject:

| subject | main_b6 | nudge_b6 | change |
|---|---:|---:|---:|
| language science | 0.714 | **0.929** | **+21.5pp** |
| natural science | 0.666 | 0.673 | +0.7pp |
| social science | 0.592 | 0.539 | **−5.3pp** |

→ A uniform visual-favor nudge **hurts** text-dominant domains (social science). **A simple nudge is too blunt a tool.** What is really needed is question-type conditional routing.

---

## ④ The budget-accuracy curve is **non-monotonic** and saturates near b≈7

Dense sweep (b=0,1,2,3,4,5,6,7,8,10):

| budget | accuracy | Δ vs prev |
|---:|---:|---:|
| 0 | 0.562 | (floor) |
| 1 | 0.622 | **+6.0pp** |
| 2 | **0.646** | +2.4pp |
| 3 | 0.626 | **−2.0pp** ← dip |
| 4 | 0.638 | +1.2pp |
| 5 | 0.654 | +1.6pp |
| 6 | 0.656 | +0.2pp |
| 7 | 0.666 | +1.0pp |
| 8 | 0.668 | +0.2pp |
| 10 | 0.666 | −0.2pp ← saturated |
| ∞ (full_info) | 0.700 | +3.2pp |

**Key observations**:
- **The 0 → 1 lift is the largest (+6.0pp)**. Even a single budget unit already lets the model bring in meaningful information.
- **Local max at b=2 (0.646), dip at b=3 (0.626)**. Non-monotonic due to seed and path dependence — if the model commits to an answer after 2 turns it is correct, but extending to 3 turns it sometimes requests a different modality and falls into a wrong answer.
- **Saturation beyond b≥7**. b=7/8/10 all plateau at 0.666–0.668. Giving more budget is not usefully consumed.
- +3.4pp gap between full_info (0.700) and b=10 (0.666): more information is available but the model policy does not request that much (average text 3.92 / visual 1.13 vs full's 11.72 / 4.00). **Lack of drive to gather information**.
- `zero_info` alone reaches 0.562 → more than half of ScienceQA is solvable from prior knowledge alone. The **total achievable lift from information use is capped at around +14pp** (0.562 → 0.700).

### What the non-monotonicity means
`temperature=0`, and **the tile / sentence reveal order also uses the same seed (random=42)**, yet swinging only the budget makes accuracy oscillate.

- The model's early-turn decisions are sensitive to the "Remaining budget: N" string visible in the prompt. At budget=2 it "commits fast"; at budget=3 it "requests one more piece of information" — and that extra request can steer it into a wrong answer.
- This is evidence that the model has a weak **stopping rule when its own information is enough to decide**. An inertia toward asking more when more budget is available ≈ under-calibrated confidence.

Implication of this curve: Qwen2.5-VL-7B already saturates at b=7. Since larger budgets do not produce lift, in actual deployment it may be more efficient to start at budget=2~3 and encourage early exit.

---

## ⑤ The early-stop signal is calibrated only at b≤3 — useless from b≥4 onwards

**Phase 1a Metric A** ([`output/calibration/summary.csv`](../../output/calibration/summary.csv), figure: [`docs/calibration_A_spontaneous_vs_forced_acc.png`](../figures/calibration_A_spontaneous_vs_forced_acc.png))

For the same budget run, accuracy on samples that terminated with a spontaneous ANSWER vs. samples that terminated with a FORCED_ANSWER.

| budget | spontaneous_n | spontaneous_acc | forced_n | forced_acc | gap |
|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 0.714 | 465 | 0.658 | +0.06 |
| 2 | 42 | **0.738** | 458 | 0.638 | **+0.10** |
| 3 | 139 | **0.763** | 361 | 0.573 | **+0.19** |
| 4 | 229 | 0.638 | 271 | 0.638 | 0 |
| 5 | 284 | 0.659 | 216 | 0.648 | +0.01 |
| 6 | 342 | 0.655 | 158 | 0.658 | 0 |
| 7 | 372 | 0.659 | 128 | 0.688 | −0.03 |
| 8 | 403 | 0.658 | 97 | 0.711 | −0.05 |
| 10 | 432 | 0.641 | 68 | **0.824** | **−0.18** |

At small budgets (b=2~3), samples the model stopped on voluntarily are +10~+19pp more accurate than those that got cut off — **the "I can answer" signal is genuinely calibrated**. But from b=4 the gap disappears, and at b=10 forced is actually **+18pp more accurate**: once budget is abundant, voluntary stops become habit, and the 68 that couldn't stop to the end are the genuinely hard cases where extra information does raise accuracy.

→ The model **has a stopping rule only at small budgets and loses that signal once budget becomes abundant**. Direct evidence for the need for a learned budget-conditional stopping policy.

---

## ⑥ Cross-budget answer stability = a free confidence proxy

**Phase 1a Metric B** ([`per_sample_stability.csv`](../../output/calibration/per_sample_stability.csv), figure: [`B_choice_stability.png`](../../output/plots/calibration/B_choice_stability.png))

How consistent `final_choice` is when the same sample is solved at 10 budget points (b=0,1,2,3,4,5,6,7,8,10):

- Mean modal-choice fraction = **0.918**
- Samples that produced exactly the same answer across all 10 runs = **69.4%**
- Modal-choice accuracy = 0.658 (matches the overall average)

Stability bin × modal accuracy:

| stability bucket | n | modal accuracy |
|---|---:|---:|
| ≤0.5 (answer flips often) | 16 | 0.625 |
| 0.5–0.7 | 59 | **0.356** ← barely above random |
| 0.7–0.85 | 35 | 0.486 |
| 0.85–1.0 (stable) | 390 | **0.721** |

→ **Samples whose answer wobbles even when only the budget dimension is perturbed are the samples the model really does not know.** Stability is a strong confidence proxy obtainable without logits / entropy. Reusable both as an abstention-target label at train time and as a self-consistency-style gate at inference.

---

## ⑦ Accuracy by stop-step — peaks at info=2, crashes −20pp from info=3 onward

**Phase 1a Metric C** ([`stop_step_acc.csv`](../../output/calibration/stop_step_acc.csv), figure: [`C_stop_step_accuracy.png`](../../output/plots/calibration/C_stop_step_accuracy.png))

Group all spontaneous-ANSWER trajectories (regardless of run) by stop step and take accuracy:

| info_requests | n | accuracy |
|---:|---:|---:|
| 0 (instant answer) | 128 | 0.711 |
| 1 | 278 | 0.745 |
| **2** | **384** | **0.794** ← peak |
| 3 | 411 | **0.596** ← **−20pp** |
| 4 | 383 | 0.637 |
| 5 | 370 | 0.600 |
| 6 | 176 | 0.597 |
| 7 | 84 | 0.524 |

**Trajectories that ask little and then stop are more accurate.** The b=3 dip in the macro budget curve (④) reproduces at the per-trajectory level — **"the moment the model starts asking more"** is itself a confidence-drop signal.

Caveat: there is selection bias (which budget runs a given info=k stop comes from shifts the average). Still, the 0.794 → 0.596 crash is not noise (all n ≥ 380).

---

## ⑧ The model fails to recognize "I already know" — easy and hard samples get almost the same amount of information

**Phase 1a Metric F** ([`zero_info_cohort.csv`](../../output/calibration/zero_info_cohort.csv), figure: [`docs/calibration_F_zero_info_cohort.png`](../figures/calibration_F_zero_info_cohort.png))

Split by zero_info (answering without any information): **281 samples correct (zi_correct)** vs **219 samples incorrect (zi_wrong)**, and compare:

| budget | cohort | spon_stop rate | avg info requests | accuracy |
|---:|---|---:|---:|---:|
| 6 | zi_correct | 0.665 | **4.06** | 0.904 |
| 6 | zi_wrong | 0.708 | **4.33** | 0.338 |
| 10 | zi_correct | 0.826 | 5.06 | 0.907 |
| 10 | zi_wrong | 0.913 | 5.16 | 0.356 |

Key points:
1. **The model requests nearly the same amount of information from both cohorts** (gap of 0.27 at b=6). It **does not distinguish** samples solvable by prior knowledge alone from those that really need information.
2. **Extra information actually hurts easy samples**: zi_correct accuracy goes from **0.954** at b=1 to **0.904** at b=6 (**−5pp**). More information breaks otherwise-correct answers.
3. zi_wrong goes 0.196 → 0.356 (+16pp). Information partially rescues hard samples while simultaneously breaking easy ones — this **asymmetric effect** is one reason the macro curve caps at 0.700.

This is the strongest finding of Phase 1a. It explains in one picture *why* a budget-conditional policy is necessary: quantitative evidence that even at the same budget, sample-level routing (stop on easy, spend on hard) is not in place.

---

## ⑨ Forced ↔ Spontaneous answer agreement 84% — extra information barely changes the model's answer

**Phase 1a Metric D** ([`forced_vs_spon.csv`](../../output/calibration/forced_vs_spon.csv))

1,981 pairs in which the same sample ended with FORCED_ANSWER under a small budget and with a spontaneous ANSWER under a large budget:

- Agreement rate between the two answers: **0.843**
- Forced accuracy 0.617 / spontaneous accuracy 0.648 (only +3pp)
- Agreement rate rises with larger forced-budget (b_forced=0 → 0.79, b_forced=6 → 0.94, b_forced=8 → 1.00)

Interpretation:
- **Extra information barely changes the model's answer**. Another signal of weak information-integration ability.
- At the same time, **force-answer itself is not a bottleneck in the budget curve** — if forced termination broke the answer every time, small-b accuracy would have to be much worse, but in reality the answer is largely the same. The dip in ④ is not a force-answer side effect but a **real path-dependence of the model's policy**.

---

## ⑩ Modality bias hurts only on the zi_wrong cohort — modality is irrelevant on zi_correct

**Phase 1b** ([`output/difficulty/cohort_curve.csv`](../../output/difficulty/cohort_curve.csv), figure: [`docs/difficulty_A_cohort_accuracy_curve.png`](../figures/difficulty_A_cohort_accuracy_curve.png))

Decomposing the b=6 reference policies by cohort:

| Policy (b=6) | zi_correct (n=281) | zi_wrong (n=219) | gap |
|---|---:|---:|---:|
| `always_text` | **0.911** | 0.288 | — |
| `always_visual` | **0.911** | 0.384 | **+9.6pp** |
| `full_info` | 0.890 | 0.457 | **+16.9pp** |
| `main_b6` (model) | 0.904 | 0.338 | — |

Two key takeaways:
1. **On zi_correct, always_text == always_visual == 0.911 exactly.** Easy samples yield the same answer under any modality → modality choice is itself meaningless.
2. **Only on zi_wrong is always_visual better than always_text, by +9.6pp.** The "model's text bias is costly" finding from ① is in fact **a phenomenon confined to the zi_wrong cohort**. The model applies nearly identical modality ratios to both cohorts (text 3.45/3.07, visual 0.58/1.24) — **the default is wrong exactly on the cohort where modality choice actually matters**, a painful miss.

→ This also explains why a "use more visual" nudge collapses on social science (③): social science is ~60% zi_wrong (see ⑫ below), the degree to which zi_wrong benefits from visual is modest, zi_correct is modality-invariant, and the nudge's forced visual inflates wasted counts on zi_correct → losses on both sides.

---

## ⑪ An asymmetry where budget hurts easy samples and saves hard samples (clean cross-over at b=1)

**Phase 1b** (figure: [`docs/difficulty_D_delta_from_b1.png`](../figures/difficulty_D_delta_from_b1.png))

Taking b=1 accuracy as baseline and tracking per-cohort accuracy change as budget grows:

| budget | zi_correct Δ | zi_wrong Δ |
|---:|---:|---:|
| 1 | 0.000 (baseline 0.954) | 0.000 (baseline 0.196) |
| 2 | −0.011 | +0.068 |
| 3 | **−0.050** | +0.073 |
| 4 | −0.053 | +0.105 |
| 5 | −0.053 | +0.142 |
| 6 | −0.050 | +0.142 |
| 7 | −0.046 | +0.160 |
| 10 | −0.046 | +0.160 |

- zi_correct is trapped at a −5pp plateau from b=3 onwards. Extra information breaks otherwise-correct answers.
- zi_wrong rises monotonically to +16pp through b=7 before saturating.
- The two curves cross exactly at b=1 → **the sweet spot where 1 unit of information per sample helps every sample exactly once**. Beyond that the ROI splits along cohort lines.

This is the mechanistic explanation for the budget-curve non-monotonicity seen in ④. The macro curve is a weighted average of the two cohort curves, and the −5pp on zi_correct eats into the +x pp on zi_wrong to produce the b=3 dip. **Core motivation for a budget-conditional policy**: even at the same average budget, the macro 0.700 ceiling can be broken only if routing is done per-sample (easy → answer fast, hard → spend to the end).

---

## ⑫ Subject economics: natural science is high-volatility, social science is low-volatility

**Phase 1b** ([`subject_cohort_sizes.csv`](../../output/difficulty/subject_cohort_sizes.csv), [`subject_cohort_accuracy.csv`](../../output/difficulty/subject_cohort_accuracy.csv), figure: [`C_subject_cohort.png`](../../output/plots/difficulty/C_subject_cohort.png))

| subject | total n | zi_correct | zi_wrong | zi_correct frac |
|---|---:|---:|---:|---:|
| language science | 14 | 10 | 4 | 0.714 |
| natural science | 410 | 240 | 170 | 0.585 |
| social science | 76 | 31 | 45 | 0.408 |

The two most interesting cells in the cohort × subject × budget accuracy table:

- **natural science zi_correct**: b=1 0.950 → b=6 0.908 (−4pp). zi_wrong: 0.182 → 0.359 (+18pp). Biggest cost-benefit of the domains.
- **social science zi_correct**: b=1 0.968 → b=6 0.968 (no change). zi_wrong: 0.222 → 0.333 (+11pp), dropping to 0.267 at b=10. **On social science, giving more budget barely rescues the hard cohort.**

So:
- **natural science**: extra information has strong effects on both cohorts. The domain where budget choice matters most.
- **social science**: zi_correct is robust, zi_wrong unresponsive. Raising the budget yields little utility — a policy close to zero_info is more efficient.
- **language science**: n=14, statistically meaningless.

→ Quantitative hint that optimal budget differs by domain. This also explains the −5.3pp collapse of the nudge on social science (③): on social science, the marginal utility of extra information is low to begin with, so forcing modality to visual just inflates the wasted count.

---

## ⑬ The model can use ABSTAIN, but its "should I answer" decision overlaps heavily with the zero_info prior-knowledge signal

**Phase 1c Metric A + D** ([`output/abstention/summary.csv`](../../output/abstention/summary.csv), [`aligned_comparison.csv`](../../output/abstention/aligned_comparison.csv))

Expose the 4-action version (ANSWER / ABSTAIN / REQUEST_TEXT / REQUEST_VISUAL) in the system prompt and measure at two budgets:

| run | coverage (answer rate) | selective acc (answered only) | notes |
|---|---:|---:|---|
| `abstain_b0` (b=0) | 12.0% (60/500) | **0.833** | abstains on 88% — the model recognizes "I don't know" well under starvation |
| `abstain_b6` (b=6) | 90.0% (450/500) | 0.644 | only 10% abstain |

The naive read "0.833 is way above zero_info's 0.562, so the model is using the confidence signal well" is a **selection confound**. The fair test is "on the exact 60 samples that abstain_b0 chose to answer, how often is zero_info correct?":

| Comparison | own acc | reference acc on the same subset | uplift |
|---|---:|---:|---:|
| `abstain_b0` vs `zero_info` (60-sample subset) | 0.833 | **0.800** | **+3.3pp** |
| `abstain_b6` vs `main_b6` (450-sample subset) | 0.644 | 0.651 | **−0.7pp** |

→ **The "should I answer" decision is essentially the same signal as the prior-knowledge boundary zero_info already had.** The newly acquired selectivity is only +3.3pp at b=0 and 0pp at b=6. Abstention does not add a *new* axis of confidence to the model; it merely converts samples zero_info would have answered incorrectly into "no answer".

---

## ⑭ Vanilla abstention is actually anti-calibrated at b=6

**Phase 1c Metric B** ([`output/abstention/cohort_xtab.csv`](../../output/abstention/cohort_xtab.csv), figure: [`docs/abstention_B_cohort_abstain.png`](../figures/abstention_B_cohort_abstain.png))

Abstain rate by zi_correct / zi_wrong cohort:

| budget | cohort | abstain rate | n |
|---:|---|---:|---:|
| 0 | zi_correct | 0.829 | 281 |
| 0 | zi_wrong | **0.945** | 219 |
| 6 | zi_correct | **0.121** | 281 |
| 6 | zi_wrong | 0.073 | 219 |

**At b=0, weakly calibrated** (zi_wrong abstain 94.5% > zi_correct 82.9%) — without information, the model follows the intuition of "abstain more often on the ones you know less".

**At b=6 the direction flips — anti-calibration**: the model abstains **more often on zi_correct** (12.1%) and **less on zi_wrong** (7.3%). Once budget is available, it just answers the hard samples anyway and sometimes gets scared into abstaining on the easy ones. Consequences:

- abstain_b6 selective acc (0.644) < main_b6 overall (0.656)
- The Effective Reliability Φ curve has abstain_b6 < main_b6 for every wrong-answer cost c ∈ [0, 2] ([`phi_curve.csv`](../../output/abstention/phi_curve.csv)). In other words, **using vanilla abstention at b=6 is a loss under any cost regime**.
- In contrast, the abstain_b0 Φ overtakes zero_info for c ≥ 1.25 (abstention becomes valuable when wrong-answer cost is high).

→ **Mere prompt exposure does not get the model to use abstention as "I don't know right now, so I won't answer" — when budget is available, it treats abstention as a "useless gadget" or deploys it in the wrong places.** This is direct evidence for why **training-for-abstention** in the style of `project_ko.md` Thread D (R-Tuning / MM-UPD) is **necessary** — the design rationale for including abstention terms (`λ_cal` for correct abstain, `λ_abs` for incorrect abstain) in the Phase 4 GRPO reward.

---

## ⑮ The calibration → anti-calibration flip happens exactly at b=3 — same location as the macro dip in ④

**Phase 1d / I14** ([`output/abstention_phase1d/I14_cohort_x_budget.csv`](../../output/abstention_phase1d/I14_cohort_x_budget.csv), figure: [`docs/figures/abstention_phase1d_I14_cohort_x_budget.png`](../figures/abstention_phase1d_I14_cohort_x_budget.png))

⑭ compared only b=0 and b=6. Adding b=1..5,7,8,10 yields the full curve of **how abstain rate evolves per cohort across budgets**.

| budget | zi_correct abstain | zi_wrong abstain | direction |
|---:|---:|---:|---|
| 0 | 0.829 | **0.945** | calibrated (zi_wrong > zi_correct) |
| 1 | 0.544 | **0.817** | calibrated (+27.3pp) |
| 2 | 0.320 | **0.397** | calibrated (+7.7pp) |
| **3** | **0.157** | 0.137 | **flipped** (−2.0pp) |
| 4 | **0.146** | 0.114 | anti-calibrated (−3.2pp) |
| 5 | **0.125** | 0.087 | anti-calibrated (−3.8pp) |
| 6 | **0.121** | 0.073 | anti-calibrated (−4.8pp) |
| 7 | **0.114** | 0.082 | anti-calibrated (−3.2pp) |
| 8 | **0.114** | 0.078 | anti-calibrated (−3.6pp) |
| 10 | **0.110** | 0.041 | strongly anti-calibrated (**−6.9pp**) |

**Key**: the sign flip is not gradual — it happens **between b=2 and b=3 in one step and never reverts**. At b≤2 the model abstains more on hard samples (zi_wrong) → calibrated. From b=3 onwards it abstains more on zi_correct → permanent anti-calibration, with the gap widening to −6.9pp at b=10.

This b=3 transition lines up exactly with the **b=3 dip in the macro budget curve** from ④. Strong indication that both phenomena share a mechanism:
- At b≤2 the model genuinely registers "not enough information" → abstains on hard samples, answers the easy ones.
- At b=3 the model decides "this much is enough to answer" → it produces wrong answers on the hard cohort and occasionally even abstains on easy ones, the (anti-)calibration pattern.
- The macro b=3 dip happens because abstaining-then-answering on hard samples flips them to wrong (acc ↓), while a slice of easy samples leaks into the abstain bucket (acc ↓).

→ Two phenomena resolve to a single figure. Concrete justification for designing budget-conditioning and abstention reward **together** during training (the reason the Phase-4 GRPO setup must not separate `λ_cost` from `λ_abs` / `λ_cal`).

---

## ⑯ Image masking barely changes the abstain rate (Δ ≤ 2pp) — vanilla abstention is image-level sufficiency-blind

**Phase 1d / I13** ([`output/abstention_phase1d/I13_masked_vs_unmasked.csv`](../../output/abstention_phase1d/I13_masked_vs_unmasked.csv), figure: [`docs/figures/abstention_phase1d_I13_masked_vs_unmasked.png`](../figures/abstention_phase1d_I13_masked_vs_unmasked.png))

To test directly whether ⑭'s anti-calibration is "the model genuinely cannot read sufficiency from image content", I picked 100 ScienceQA samples (50 zi_correct + 50 zi_wrong, seed 42), built a variant (`preproc_masked/`) where **all 4 tiles are replaced with white (255,255,255)**, and compared abstain behaviour at the same b=0/4/6.

| budget | cohort | unmasked abstain | masked abstain | Δ |
|---:|---|---:|---:|---:|
| 0 | zi_correct | 0.74 | 0.74 | **0.00** |
| 0 | zi_wrong | 0.96 | 0.96 | **0.00** |
| 4 | zi_correct | 0.12 | 0.14 | +0.02 |
| 4 | zi_wrong | 0.20 | 0.20 | **0.00** |
| 6 | zi_correct | 0.10 | 0.10 | **0.00** |
| 6 | zi_wrong | 0.10 | 0.12 | +0.02 |

**Δ ≤ 2pp in every cell** — within a 50-sample cohort that is a single-sample difference, i.e. noise. The model **barely registers** that all four tiles are white when deciding to abstain.

Naive accuracy is also nearly unchanged under masking (b=4 zi_correct 0.84→0.82, b=6 zi_correct 0.86→0.84, b=6 zi_wrong 0.24→0.24). Two possibilities:
1. **ScienceQA text (hint+lecture) is enough to derive the answer in many cases** — the image is nominally present but practically redundant.
2. **The model treats white tiles as "image as usual" rather than "no information"** — the visual encoder accepts white as meaningful pixels, and the model reads sufficiency from text confidence rather than image content.

These two effects mix and we cannot fully separate them, but the conclusion that **vanilla abstention does not track image-level sufficiency** is safe (if it did, the masked zi_wrong cohort would show an abstain-rate spike).

**Caveat**: 100 samples is small (per-cohort 50 → 95% CI ±~14%). And ScienceQA is not an image-required-by-construction benchmark. The **decisive sufficiency test** would re-run this on V\*Bench / HR-Bench (image is essential to the answer) + image-masked variants — registered in ROADMAP as backlog I15 (early Phase 2).

→ ⑮ (flip at b=3) and ⑯ (image-blind) together tighten the picture: vanilla abstention reads **text-based confidence**, not image content, and that confidence becomes mis-calibrated starting at b=3. R-Tuning / GRPO training would have to fix both deficiencies.

---

## Limitations and caveats

- **Single model**: only Qwen2.5-VL-7B evaluated. Modality bias may depend on the model's fine-tuning regime — LLaVA / InternVL could differ.
- **Greedy decoding (T=0)**: temperature=0. Turning sampling on would shake the model's decisions and could yield different results.
- **2×2 tiles = 4 images**: with a finer cut (3×3=9, 4×4=16) the vision cost unit shrinks and the budget-allocation dynamics would change.
- **Text chunk size**: sentence-level. Coarser chunks (paragraphs) would raise information per budget unit and could shift the comparisons.
- **Subject-distribution skew**: 410/500 are natural science. With social (76) / language (14) small, those conclusions are weak.
- **Single nudge prompt**: different wording / position / length could change results. Only one nudge was measured here.
- **Tile reveal order**: shuffled (seeded). If tiles were always revealed in a fixed order, the model could latch onto heuristics like "the first-seen tile is always at a specific position", so random shuffle is the default. Row-major comparison is not tested.
- **Phase 1c sufficiency set is skipped**: the original ROADMAP explicitly called for a sufficiency-known mini-set using an MM-UPD subset or an image-masked ScienceQA variant, but this phase uses the zi_correct / zi_wrong cohorts as a proxy instead. ⑬–⑭ diagnose the model's abstention behavior on **naturally occurring ScienceQA samples**, not a direct test against unanswerable-by-construction stimuli. The latter is deferred to a Phase 1d or Phase 2 follow-up (see ROADMAP).

---

## Candidate next steps

Next steps are organized by phase in [`roadmap.md`](../../references/roadmap.md). Phase 1a (calibration ⑤–⑨) is complete. Upcoming priorities:

- **Phase 1b** Difficulty stratification: redraw the budget curve by zero_info cohort (to separate which budget is effective for which cohort)
- **Phase 1c** Abstention proxy: how often the vanilla model uses the ABSTAIN action + build a small sufficiency-known set
- **Phase 2** Expand the action vocab to 6 (`ABSTAIN`/`THINK`/`ZOOM(bbox)`/`REQUEST_HI_RES`) + add one new benchmark (V*Bench or HR-Bench)
- **Phase 3+** SFT data → GRPO budget-conditional training (the paper's main contribution)

---

## Artifact locations

- `notebooks/experiment.ipynb` — demo notebook (contains all visualizations / tables from this doc)
- `output/all_runs_summary.csv` — source of the comparison table above
- `output/subject_crosstab.csv` — subject × policy cross-tab
- `output/visual_bias_breakdown.csv` — sample-level main vs always_visual bucket
- `output/plots/accuracy_vs_budget.png`, `output/plots/modality_mix.png` — source plots
- `output/<run>/predictions.{parquet,csv,jsonl}` — per-run sample-level raw (jsonl includes step traces)
- `output/<run>/summary_{overall,by_subject}.{csv,jsonl}` — per-run aggregates
- `output/calibration/` — Phase 1a artifacts (summary.csv / per_sample_*.csv / stop_step_acc.csv / forced_vs_spon.csv / zero_info_cohort.csv)
- `output/plots/calibration/A..F_*.png` — calibration visualizations
- `output/difficulty/` — Phase 1b artifacts (cohort_curve.csv / cohort_modality.csv / subject_cohort_*.csv / peak_budget.csv)
- `output/plots/difficulty/{A,B,C,D}_*.png` — difficulty visualizations
- `output/abstain_b0/`, `output/abstain_b6/` — Phase 1c raw predictions (output of run_abstention.py)
- `output/abstention/` — Phase 1c analysis artifacts (summary / cohort_xtab / phi_curve / aligned_comparison)
- `output/plots/abstention/{A,B,C}_*.png` — abstention visualizations
- `docs/figures/*.png` — 5 headline figures for the insights (copies; see the "figure" section above)