# ROADMAP — Budget-conditional VLM agents

> 살아 있는 진행 추적 문서. **paper-scope 계획은 [`project_ko.md`](./project_ko.md)** (변경 금지에 가까운 reference), 이 문서는 그 계획 대비 현재 어디까지 왔는지 + 다음 단위 작업 + 인사이트 발굴 액션을 정리한다. 의미 있는 변화가 있을 때마다 업데이트.

마지막 업데이트: 2026-04-25 (Phase 1 완료 + Phase 2.0 V*Bench baseline 완료)

---

## 0. 한 줄 요약

`project_ko.md`의 Phase 0 (diagnostic baseline) + Phase 1a–d (calibration / difficulty / abstention / anti-calibration & masking) + Phase 2.0 (V*Bench 도입 + 3-action baseline) **완료**. 6-action vocab 확장 / SFT+GRPO / Pareto 학습은 아직 0%. 다음 단계는 **Phase 2.1+ (6-action vocab 도입) → Phase 3 (SFT 데이터 파이프라인) → Phase 4 (GRPO 학습)**.

Phase 1+2.0에서 총 **16개 추가 인사이트 발굴** (INSIGHTS ⑤–⑳). paper motivation으로 가장 강한 5가지:

- **⑧**: 모델은 zi_correct/zi_wrong 코호트에 거의 같은 양의 정보를 요청 (b=6에서 4.06 vs 4.33) — "내가 안다"를 인지 못 함.
- **⑪**: budget-accuracy 곡선이 zi_correct와 zi_wrong에서 b=1에서 정확히 교차 — 쉬운 샘플은 정보가 망치고, 어려운 샘플은 정보가 살림. 매크로 plateau(0.700 ceiling)의 메커니즘.
- **⑭**: b=6에서 vanilla abstention은 **anti-calibrated** (zi_correct 12.1% abstain > zi_wrong 7.3%) — training 없이는 방향까지 거꾸로 간다는 직접 증거.
- **⑮**: Calibration → anti-calibration **flip이 정확히 b=3에서 일어남**, ④ macro dip과 같은 위치 → 두 phenomenon 한 메커니즘. paper의 budget-conditional + abstention reward를 **함께** 설계해야 한다는 가장 단단한 정량적 근거.
- **⑱** (Phase 2.0): V*Bench `relative_position`이 b=2 peak (0.45) → full_info dip (0.21) **−24pp**. ④의 비단조성이 spatial reasoning에서 극대화 — paper의 ZOOM(bbox) action(2b) 정당화의 직접 증거. ⑰(adaptive > brute-force on V*Bench, ScienceQA ① reverse)와 함께 multi-domain 일반화 evidence.

---

## 1. 현재 상태 (Phase 0 완료)

| 항목 | 상태 | 비고 |
|---|---|---|
| 모델 | Qwen2.5-VL-7B-Instruct, bf16, H200 | 계획서와 일치 (정답 backbone) |
| 데이터 | ScienceQA n=500 | 계획서의 V*Bench/HR-Bench/MM-UPD 등은 **미도입** |
| Action vocab | 3개 (`ANSWER` / `REQUEST_TEXT` / `REQUEST_VISUAL`) | 계획서는 6개 (`ANSWER`/`ABSTAIN`/`THINK`/`REQUEST_HI_RES`/`ZOOM`/`RETRIEVE`) |
| Cost model | uniform per-call (1 unit) | 계획서는 token-based deterministic cost |
| Budget conditioning | 프롬프트에 `Remaining budget: N` 노출만 | 학습 안 함 (vanilla policy의 한계 진단용) |
| Training | 없음 (inference-only) | Stage-1 SFT, Stage-2 GRPO 모두 미수행 |
| Pareto curve | budget sweep b=0..10, **단일 task** | 멀티 벤치마크 Pareto는 아직 |
| Abstention | 없음 | 핵심 contribution 중 하나인데 미구현 |
| Calibration metric | 없음 | Φ, AURC 등 계획서 metric 미측정 |

**완료된 산출물**:
- `budget_eval.py` 엔진 (3-action episode loop, force-answer, wasted accounting)
- `preprocessing.py` (ScienceQA → samples.parquet + 2×2 tile PNG)
- 베이스라인 4종: `zero_info` / `always_text` / `always_visual` / `full_info`
- Budget sweep b=1..10 (모델 정책)
- `nudge_b6` (visual-favor system prompt 변형)
- 분석 산출물: `output/all_runs_summary.csv`, `output/visual_bias_breakdown.csv`, `output/plots/`
- 노트북 `experiment.ipynb` (37셀 사전 실행) + `insights_ko.md`

---

## 2. Phase 0이 보여준 4가지 — paper의 motivation paragraph 후보

이 진단 결과들은 그대로 **"vanilla VLM은 budget 신호를 이해/활용하지 못한다 → 학습된 budget-conditional policy가 필요하다"** 라는 motivation으로 쓰임.

1. **모델은 강한 텍스트 편향** — `main_b6` text 3.28 / visual 0.87. 그런데 자동 `always_visual`이 +2.4pp 더 높음. 모델 정책이 정보 효율 반대로 감.
2. **Budget-accuracy 곡선 비단조** — b=2 local max → b=3 dip → b≥7 포화. 같은 seed, 같은 데이터인데 budget만 바꿔도 path-dependence로 정확도 출렁임. **모델이 budget signal을 못 읽음**의 직접 증거.
3. **단일 prompt nudge는 행동 100% 뒤집지만 정확도 +0.4pp만** — modality 선호는 한 줄로 즉시 반전(text 3.28→0.01)되는데 accuracy는 0.656→0.660. **prompt-only intervention 한계의 정량 증거.**
4. **Visual_only 정답 43개(8.6%)를 main이 놓침** — 비전이 답이었던 케이스에서조차 모델은 평균 text 3.91/visual 0.81 사용. language science에서 21.4%로 가장 심각. 도메인 효과 + bias 동시 입증.

(자세한 내용은 [`insights_ko.md`](../docs/insights/insights_ko.md))

---

## 3. 계획서 대비 갭 매트릭스

| project_ko.md 요소 | 현재 | 갭 메우는 작업 |
|---|---|---|
| 6-action vocab | 3-action | `ABSTAIN` / `THINK` / `ZOOM(bbox)` / `REQUEST_HI_RES` 추가 (Phase 2) |
| Token-based cost | per-call cost | 비전 토큰 수, CoT 토큰 수 정확 산정 (Phase 2) |
| Budget conditioning π(a\|s,B) | prompt-injection만 | SFT + GRPO로 학습 (Phase 3-4) |
| 멀티 벤치마크 (V*Bench, HR-Bench, MM-UPD, POPE, MM-AQA, MMBench, MMMU) | ScienceQA 1개 | 최소 V*Bench + MM-UPD 추가 (Phase 1c, Phase 5) |
| Stage-1 SFT (LoRA r=16) | 없음 | sufficiency-labeled 데이터 + tool-use trajectory 합성 (Phase 3) |
| Stage-2 GRPO (budget-conditional) | 없음 | reward 함수, group sampling, λ 튜닝 (Phase 4) |
| Pareto curve (multi-bench, AUPC) | 단일 task 1D sweep | 학습 후 재측정 (Phase 5) |
| Abstention metric (Φ, AURC) | 없음 | MM-UPD + masked-image diagnostic 도입 (Phase 1c, Phase 5) |
| Hallucination regression (POPE, CHAIR) | 없음 | 학습 후 비-퇴행 체크 (Phase 5) |

---

## 4. Phase 로드맵

각 phase는 독립 deliverable. Phase 1은 학습 없이 갈 수 있어서 빠르게 paper appendix material을 쌓는다. Phase 2부터 코드 변경량이 커지고, Phase 3-4는 GPU-시간이 본격 투입.

### Phase 1 — Diagnostic 강화 (training 없이, 1주)

**목표**: 현재 진단을 단단하게 만들고 baseline/ablation에 쓸 데이터 확보. Paper의 Section 3 (motivation)와 Section 5 (baselines) 양쪽에 재활용됨.

- [x] **1a. Calibration 분석** ✅ 2026-04-24 — `analyze_calibration.py` + `output/calibration/`. 6개 metric 결과 INSIGHTS ⑤–⑨에 정리. 핵심: ⑤ early-stop은 b≤3에서만 calibrated, ⑥ cross-budget stability = 무료 confidence proxy, ⑦ stop-step별 정답률 info=2 peak / info=3 −20pp 급락, ⑧ zi_correct/zi_wrong 코호트가 같은 양 정보 요청 + 추가 정보가 쉬운 샘플 망침, ⑨ forced↔spontaneous 답 84% 일치.
- [x] **1b. Difficulty stratification** ✅ 2026-04-24 — `analyze_difficulty.py` + `output/difficulty/`. INSIGHTS ⑩–⑫ 추가. 핵심: ⑩ modality bias는 zi_wrong에서만 손해 (zi_correct는 always_text == always_visual 정확히 0.911), ⑪ budget-zi_correct 곡선과 zi_wrong 곡선이 b=1에서 정확히 교차 (clean cross-over), ⑫ natural science high-volatility / social science low-volatility — 도메인별 optimal budget 다름.
- [x] **1c. Abstention proxy 측정** ✅ 2026-04-24 — `budget_eval.py`에 `enable_abstain` flag + `SYSTEM_INSTRUCTION_WITH_ABSTAIN` 추가. `run_abstention.py`로 `abstain_b0` / `abstain_b6` 500샘플 각각 실행. `analyze_abstention.py`로 4개 metric(A summary, B cohort xtab, C Phi sweep, D cohort-aligned comparison) 산출. INSIGHTS ⑬–⑭ 추가. 핵심: ⑬ "답할지 말지" 신호가 zero_info 사전지식과 거의 겹침 (selectivity +3.3pp만), ⑭ b=6에서 vanilla abstention이 **anti-calibrated** (zi_correct 12.1% > zi_wrong 7.3% abstain).
  - ⚠️ **Scope-cut**: 원안의 MM-UPD subset / image-masked ScienceQA sufficiency-known mini-set은 **생략**. `zi_correct` / `zi_wrong` cohort를 sufficiency proxy로 사용. unanswerable-by-construction stimuli에 대한 직접 테스트는 Phase 1d 또는 Phase 2 초반으로 이월 — 아래 backlog I13 참조.
- [x] **1d. Anti-calibration 곡선 + sufficiency-known masking** ✅ 2026-04-25 — `run_abstention_sweep.py`로 b=1..5,7,8,10 추가 abstain 측정 (I14), `preproc_masked/` 100샘플(50/50 cohort) + `run_abstention_masked.py`로 image-masked 변형 b=0/4/6 측정 (I13), `analyze_abstention_phase1d.py`로 통합 분석. INSIGHTS ⑮–⑯ 추가. 핵심: ⑮ calibration→anti-calibration **flip이 정확히 b=3에서**, ④ macro dip과 같은 위치 → 두 현상이 한 메커니즘. ⑯ 이미지 마스킹해도 abstain rate Δ ≤ 2pp → vanilla abstention은 image-level sufficiency-blind.
  - **두 번째 VLM 검증**(원래 1d 후보)은 Phase 2 / 5 sanity 단계로 이월. 텍스트 편향 + b=3 flip이 다른 VLM에서도 재현되는지는 paper 일반화 위해 필요하지만 Phase 1c+1d로 ScienceQA 진단은 충분히 단단해짐.

**Exit criterion of Phase 1**: 1a + 1b + 1c + 1d 모두 완료. insights에 ⑤–⑯ 12개 발견 누적. paper motivation은 ⑧, ⑪, ⑭, ⑮가 핵심 figure 후보.

### Phase 2 — Action space 확장 (training 없이, 1-1.5주)

**목표**: 계획서의 6-action vocab을 **inference-only**로 먼저 깔아둔다. 새 action들이 "있어도" vanilla 모델이 잘 못 쓴다는 걸 보이면 그것 자체로 또 motivation.

- [ ] **2a. `ABSTAIN` action 정식 도입** — 모든 task에서 cost 0으로 종료 (Phase 1c에서 이미 구현, 도큐먼테이션만 정리하면 됨)
- [ ] **2b. `ZOOM(bbox)` action** — Qwen2.5-VL native grounding으로 bbox 후보 제안 → crop tile 추가 입력. 토큰 비용 산정. ⑱ relative_position 케이스가 직접적 motivation.
- [ ] **2c. `REQUEST_HI_RES` action** — 초기 입력을 ¼-res로 시작, full-res 요청 시 토큰 ↑
- [ ] **2d. `THINK(text)` action** — free-form CoT 한 segment, output token cost로 카운트
- [ ] **2e. Cost model 리팩터** — uniform-1 → token-based deterministic. 비전 token = `floor(H'·W'·grid_factor)` 정확히 계산.
- [x] **2f. New benchmark 1종** ✅ 2026-04-25 — V*Bench (191 sample, 115 direct_attributes + 76 relative_position) 도입. `preprocessing_vstar.py` + `preproc_vstar/`. `budget_eval.py`에 `max_pixels` flag 추가 (high-res ergonomics, shared-GPU 환경에서 vision token 캡).
- [x] **2g. 3-action vanilla baseline (V*Bench, 5 runs)** ✅ 2026-04-25 — `run_vstar_baseline.py` (zero_info / b=2 / b=4 / always_visual_b4 / full_info) + `analyze_vstar.py`. INSIGHTS ⑰–⑳ 추가. 핵심: ⑰ 모델 adaptive (b=4: 0.440)이 always_visual (0.387) 능가 → ScienceQA의 ① reverse, ⑱ relative_position이 b=2에서 peak / full_info에서 dip −24pp → ④ 비단조성이 spatial reasoning에서 극대화, ⑲ text request 0회 (text 없는 도메인이면 default text 편향 즉시 사라짐), ⑳ V*Bench zero_info 0.120 = 진짜 vision-only QA.
- [ ] **2h. 6-action 통합 baseline 재측정** — 6-action vocab으로 V*Bench 한 번 더. 새 action들이 vanilla 모델에서 어떻게 쓰이는지 측정.

**Exit criterion**: 6-action 인터페이스 안정화, V*Bench에서 6-action sample-level prediction. **2.0 (V*Bench 도입 + 3-action baseline) 완료** — 2a–2e + 2h가 남은 Phase 2 작업.

### Phase 3 — SFT 데이터 파이프라인 (1주)

**목표**: Stage-1 LoRA SFT를 위한 ~15k tool-use trajectory 만들기. 계획서 Stage-1 그대로.

- [ ] **3a. Sufficiency-labeled subset** — VQAv2/GQA 중 5k에 4×4 마스킹, 1/16·4/16·8/16·16/16 unmask 비율로 변형, teacher VLM(GPT-4o or Qwen2.5-VL-72B) 답을 majority vote로 sufficiency 라벨링
- [ ] **3b. Teacher-forced trajectory 합성** — V*Bench train + Visual CoT 438k 일부에 6-action으로 풀이 trace 생성, 정답 일치하는 것만 keep
- [ ] **3c. Rule-based trajectory** — V*Bench / Visual CoT의 ground-truth bbox로 합성 `ZOOM` 시퀀스
- [ ] **3d. Format 통일 JSONL** — `{messages, tools, expected_action_sequence}` 스키마
- [ ] **3e. Held-out val 500개 분리**

**Exit criterion**: 15k 학습용 + 500 val JSONL 생성 완료, 분포 통계 (per-action 빈도, 평균 길이) 리포트.

### Phase 4 — Stage-1 SFT → Stage-2 GRPO (1.5-2주, GPU 본격 투입)

**목표**: 계획서 Stage-1 + Stage-2. budget-conditional policy 학습.

- [ ] **4a. Stage-1 LoRA SFT** — Qwen2.5-VL-7B + LoRA(r=16, α=32) on 마지막 8 block + LM head, 2 epoch, batch 32, 4×H200 ~8h
- [ ] **4b. Reward 함수 구현** — `R = 1[correct]·r_acc - λ_cost·max(0, cost-B) - λ_abs·1[wrong abstain] + λ_cal·1[right abstain]`
- [ ] **4c. GRPO trainer 통합** — `trl` GRPOTrainer 또는 `verl` 베이스, group size G=8, B ∈ {512, 1024, 2048, 4096, 8192}
- [ ] **4d. λ 스윕** — 500 val에서 λ_cost, λ_abs, λ_cal 각 3 point grid
- [ ] **4e. 본 학습** — 4-8×H200 ~48-72h, ~30k samples
- [ ] **4f. Judge** — Qwen2.5-VL-72B 또는 GPT-4o (policy랑 다른 모델로 contamination 회피)

**Exit criterion**: Stage-2 학습 수렴, val Pareto에서 vanilla 대비 우상향.

### Phase 5 — Ablation + 최종 평가 + paper draft (1주)

- [ ] **5a. No-budget-conditioning ablation** — B를 shuffle해서 input. 계획서 명시.
- [ ] **5b. No-abstention ablation** — abstain 없는 5-action vocab으로 재학습
- [ ] **5c. SFT-only ablation** — Stage-1만, Stage-2 skip
- [ ] **5d. 최종 Pareto** — V*Bench, HR-Bench(4K/8K), MM-UPD, POPE, MM-AQA(가능하면), MMBench, MMMU에서 B∈{256,512,1024,2048,4096,8192,∞} sweep
- [ ] **5e. Calibration/abstention metric** — Φ, AURC, abstention F1, hallucination rate(POPE adv + CHAIR_i)
- [ ] **5f. AUPC 계산 + 헤드라인 plot** — paper figure 1
- [ ] **5g. 1차 draft 작성**

**Exit criterion**: NeurIPS 2026 submission ready.

---

## 5. 인사이트 발굴 우선순위 (insight-mining backlog)

별도 phase는 아니지만 진행 중에 인사이트 만들기 위해 항상 굴려야 할 것:

| # | 행동 | 산출 | 어느 phase에 묻혀가나 |
|---|---|---|---|
| I1 | trace에서 modality switching pattern 시각화 (text→visual transition matrix) | plot + 한 문단 해석 | Phase 1b 옵션 (1a 완료, 별 outcome 없었음) |
| I2 | wasted_request 발생 직전 모델 prompt 분석 (모달리티 소진 인지 여부) | qualitative 표 | Phase 1c 부산물 |
| I3 | 같은 sample의 b=2/3/5/7 trace diff (path divergence 사례 study) | notebook 셀 + 사례 5개 | Phase 1b 부산물 (B의 unstable bin 16+59 샘플 기준) |
| I4 | subject × action × correctness 3-way table | csv | Phase 1b 부산물 |
| I5 | "쉬운 샘플 over-spend" vs "어려운 샘플 under-spend" 비대칭 분리 | 두 개 plot | ✅ Phase 1a Metric F가 이미 답 — easy 샘플도 hard 샘플도 같은 양 요청 (대칭 over-spend) |
| I6 | abstention과 budget의 trade-off 곡선 (학습 전 vanilla) | plot | Phase 1c → Phase 2 |
| I7 | 다른 VLM에서 같은 4가지 + 5가지 발견 재현되는지 | 3×9 비교표 | Phase 1d |
| I8 | 6-action vocab으로도 모델이 어떤 action을 거의 안 쓰는지 | per-action 빈도 | Phase 2g |
| I9 | high-res 도메인(V*Bench)에서 텍스트 편향 사라지는지 | budget sweep 비교 | Phase 2f |
| I10 | SFT만으로 budget signal을 따르게 만들 수 있는지 (RL 없이) | Pareto compare | Phase 4a vs 4e |
| I11 | choice stability(metric B)를 train-time abstention target으로 쓸 수 있는가 | SFT label 합성 실험 | Phase 3a |
| I12 | "추가 정보가 망친다" (⑧, zi_correct b=1→b=6 −5pp) 현상이 다른 데이터셋에서도 나타나는가 | b=1 vs b=6 비교 plot | Phase 1c-1d 또는 Phase 2f |
| I13 | Phase 1c에서 skip한 sufficiency-known mini-set 제대로 만들기 (MM-UPD subset 또는 ScienceQA image-masked 100개) → vanilla abstention이 "답 불가능" 샘플에서는 calibrated인지 확인 | `preproc_masked/` + `run_abstention_masked.py` + 비교 plot | ✅ Phase 1d (2026-04-25). Δ ≤ 2pp → vanilla abstention은 image-level sufficiency-blind. ScienceQA caveat 때문에 결정적 테스트는 I15로 이월. |
| I14 | Anti-calibration (⑭)이 다른 budget b∈{1..10}에서 어떻게 진화하는가 — 순차적으로 학습됐을 가능성 | b마다 abstain run → cohort rate plot | ✅ Phase 1d (2026-04-25). flip이 정확히 b=3, 이후 영구 anti-calibrated, b=10에서 격차 −6.9pp. |
| I15 | I13 결정판: V*Bench / HR-Bench (image-required-by-construction) + image-masked 변형으로 vanilla abstention의 image-level sufficiency 추적 능력 직접 테스트 | masked V*Bench/HR-Bench 50-100개 + abstain run + 비교 | Phase 2 초반 (벤치마크 도입 후) |

---

## 6. 직전 결정 / open questions

- **Q1**: ScienceQA를 계속 testbed로 둘지, V*Bench로 primary를 옮길지. → 현재 답: ScienceQA는 Phase 1까지 유지, Phase 2부터 V*Bench primary, ScienceQA는 OOD 체크용으로 강등.
- **Q2**: 1-GPU iteration vs 4-GPU 본 학습 분리. → Phase 1-3는 single H200, Phase 4부터 4-8 H200.
- **Q3**: 7B vs 3B for fast iteration. → Phase 4 본 학습 시작 전 3B로 1회 dry-run, 그 후 7B.
- **Q4**: Judge 모델. → Qwen2.5-VL-72B를 우선 시도, API 한도/시간 문제면 InternVL 2.5-38B fallback.
- **Q5**: trl vs verl. → 추후 Phase 3 끝날 때 재결정.

---

## 7. 다음 단위 작업 (now-doing)

Phase 2.0 (V*Bench 도입 + 3-action baseline) 완료. 다음 후보 순서:

**Phase 2.1 — V*Bench abstention 측정 (I15 결정판, 1h)**: 현 3-action vocab + ABSTAIN을 V*Bench에서 sweep + masked 변형. 가설: ⑳ V*Bench는 image-required → 4 tile masking이 진짜 unanswerable → ⑭/⑮ anti-calibration이 진짜 sufficiency 환경에서 어떻게 나오는지 직접 측정. paper의 sufficiency-conditional abstention 주장 결정적 근거.

**Phase 2.2 — ZOOM(bbox) action 도입 (2b)**: ⑱이 직접 motivation. relative_position에서 모델이 두 객체를 한 crop에서 비교 가능하도록. Qwen2.5-VL native grounding으로 bbox 후보 제안. ~1-2일.

**Phase 2.3 — Cost model 리팩터 + REQUEST_HI_RES (2c, 2e)**: token-based cost. ¼-res로 시작해서 full-res 요청.

권장 순서: **2.1 → 2.2 → 2.3**. 2.1은 Phase 1d 직접 후속이라 paper narrative 매끄럽고, ZOOM 도입 전에 "vanilla abstention이 진짜 sufficiency에 어떻게 반응하는가" 답을 확보. ⑱이 ZOOM의 자연스러운 처방이 되어 2.2로 이어짐.

미해결 backlog (Phase 2 진행 중 동시 처리):
- **I9 ✅ partial** (Phase 2.0): V*Bench에서 ⑩ 일반화 검증 — text 편향이 multimodal-domain artifact임 확인 (⑲).
- **I12 ✅ partial** (Phase 2.0): V*Bench cohort breakdown으로 "more info hurts easy" 패턴 일부 재현 (zi_correct n=23 too small).
- **I15** Phase 2.1에서 결정적 테스트 예정.
- **I7**: 두 번째 VLM (LLaVA-OneVision-7B 또는 InternVL2.5-8B)에서 ⑤–⑳ 재현. Phase 5 sanity로 이월.

---

## 8. 변경 이력

| 날짜 | 변경 |
|---|---|
| 2026-04-24 | 초기 작성. Phase 0 완료 캡처, project_ko.md와 갭 매트릭스 작성, 5-phase 로드맵 + insight backlog 정의. |
| 2026-04-24 | Phase 1a 완료. `analyze_calibration.py` 신규 추가, 6개 metric 결과 `output/calibration/` + `output/plots/calibration/`로 떨어짐. INSIGHTS에 ⑤–⑨ 다섯 발견 추가. 다음 작업 Phase 1b로 전환. Insight backlog I5에 ✅ 표시, I11/I12 추가. |
| 2026-04-24 | Phase 1b 완료. `analyze_difficulty.py` 추가, cohort curve + modality mix + subject crosstab + delta-from-b1 산출. INSIGHTS ⑩–⑫ 추가 (특히 ⑪ cross-over at b=1은 paper figure 후보). |
| 2026-04-24 | Phase 1c 완료 (scope-cut noted). `budget_eval.py`에 `enable_abstain` + `SYSTEM_INSTRUCTION_WITH_ABSTAIN` 추가. `run_abstention.py` + `analyze_abstention.py` 신규. `abstain_b0` + `abstain_b6` 500샘플 실행. INSIGHTS ⑬–⑭ 추가 — ⑭ anti-calibration 이 budget-conditional training 필요성의 직접 증거. sufficiency-known mini-set은 backlog I13으로 이월. 헤드라인 figure 5장 `docs/figures/`로 복사. Phase 1a+1b+1c 번들로 commit. |
| 2026-04-25 | Phase 1d 완료. `run_abstention_sweep.py` (b=1..5,7,8,10 추가 abstain 8 run) + `preprocessing_masked.py` (100샘플 image-masked 변형) + `run_abstention_masked.py` (b=0/4/6 masked) + `analyze_abstention_phase1d.py` 신규. I13(sufficiency-known masking, ⑯) + I14(anti-calibration 곡선, ⑮) 발견. **⑮ flip이 정확히 b=3 = ④ macro dip 위치** — paper의 가장 단단한 single-figure motivation 후보. ⑯ 마스킹 Δ ≤ 2pp → vanilla abstention image-blind. 헤드라인 plot 2장 `docs/figures/abstention_phase1d_*.png` 추가. backlog I15 신규 (decisive sufficiency 테스트는 V*Bench/HR-Bench로 이월). |
| 2026-04-25 | Phase 2.0 (V*Bench 도입 + 3-action vanilla baseline) 완료. `preprocessing_vstar.py` (191 sample, 115 direct_attributes + 76 relative_position) + `run_vstar_baseline.py` (5 runs: zero_info/b=2/b=4/always_visual_b4/full_info) + `analyze_vstar.py` 신규. `budget_eval.py`에 `max_pixels` flag 추가 (shared GPU 환경 대응). INSIGHTS ⑰–⑳ 추가. 핵심: ⑰ 모델 adaptive (b=4: 0.440)이 always_visual (0.387) 능가 → ScienceQA ① reverse, ⑱ relative_position 비단조 (b=2 0.45 peak → full_info 0.21 dip), ⑲ text request 0회 (modality 부재 빨리 인지), ⑳ V*Bench zero_info 0.120 = vision-only QA. 헤드라인 plot 2장 `docs/figures/vstar_*.png` 추가. backlog I9/I12/I15 partial 진행. |
