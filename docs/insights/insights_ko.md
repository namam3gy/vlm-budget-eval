# 인사이트 노트 — Budget-Constrained Sequential Information-Seeking

> Qwen2.5-VL-7B-Instruct를 ScienceQA 500개 샘플에서 budget sweep(b=0…10) + 베이스라인 정책 4종으로 평가한 결과. 자세한 데모는 [`experiment.ipynb`](../../notebooks/experiment.ipynb), 원시 데이터는 [`output/`](../../output) 참고.
>
> **Figure 노트**: 헤드라인 figure 5장은 `docs/figures/` 아래에 tracked로 복사되어 있고 본문이 이를 링크함. 그 외 보조 plot 참조는 `output/plots/` 아래의 gitignored 경로 — `analyze_calibration.py` / `analyze_difficulty.py` / `analyze_abstention.py` 재실행 시 로컬에서 재생성됨.

## 용어: Sweep

**Parameter sweep** = 다른 조건(모델 · 데이터 · seed · 프롬프트 · 정책 코드)은 모두 고정한 채 **하이퍼파라미터 하나를 여러 값으로 돌리고** 지표가 어떻게 반응하는지 그래프로 그리는 방법.

이 프로젝트의 sweep은 `budget` 하나만 움직인다. 11 point(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, ∞)에 각각 500 샘플 × 동일 seed(42)로 돌아간다. 산출물이 아래 §④의 accuracy curve.

구체적으로, `run_dense_sweep.py` 내부:

```python
BUDGETS = [1, 2, 3, 5, 7, 10]
for b in BUDGETS:
    cfg = EvalConfig(policy="model", budget=b, random_seed=42, ...)
    for sample in samples:
        run_episode(model, processor, sample, cfg, rng)   # 동일 500 샘플
    save(...)
```

노트북 `§3-a`에 **같은 sample(`sqa_000001`)이 budget 1/2/3/5/6/7/10에서 어떻게 다르게 풀리는지** trace 비교가 실려 있음.

---

## 0. 한 눈에 보기

### Budget vs. accuracy
![accuracy vs budget](../figures/accuracy_vs_budget.png)

### 정책별 모달리티 사용량
![modality mix](../figures/modality_mix.png)

### 전체 비교표

모델 정책 budget sweep (b=0…10) + 베이스라인 + full_info ceiling + nudge:

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

## 용어 빠른 참조

| 용어 | 의미 |
|---|---|
| **VLM** | Vision-Language Model. 이미지+텍스트 입력 → 텍스트 출력. |
| **MC** | Multiple Choice. A/B/C/D 한 letter로 답하는 형식. |
| **Budget** | 한 샘플당 정보 요청 횟수 총합. ANSWER cost 0, REQUEST_* cost 1. |
| **Modality** | 정보 출처 종류. 본 실험은 `text`(문장 단위 hint)와 `visual`(이미지 타일) 둘. |
| **Tile** | 원본 이미지를 N×N 그리드(default 2×2 = 4)로 자른 patch. 한 번에 1장씩 별도 이미지로 입력. |
| **Action** | 매 스텝 모델이 출력하는 한 줄 JSON. `ANSWER` / `REQUEST_TEXT` / `REQUEST_VISUAL`. |
| **Force-answer** | budget=0 또는 wasted 한도 초과 시 시스템이 "지금 답해라" 프롬프트를 강제. |
| **Wasted request** | 소진된 모달리티 재요청 또는 unparseable JSON 출력으로 budget만 까먹은 케이스. |
| **Refused answer** | force-answer 프롬프트에도 모델이 ANSWER를 안 내면 episode 종료. 무한 루프 방지 cap. |
| **Policy** | 액션 결정 주체. `model`(모델이 결정), `always_text`/`always_visual`(자동 단일 모달리티), `full_info`(모든 정보 한 번에 공개). |
| **Floor / Ceiling** | `zero_info`(0.562)는 정보 없을 때 사전지식 floor, `full_info`(0.700)는 모든 정보 가용 시 ceiling. |
| **Sweep** | 다른 조건(모델·데이터·seed·정책 구현)은 모두 같게 두고 **하이퍼파라미터 하나만 여러 값**으로 돌려 지표의 반응 곡선을 얻는 방법. 이 프로젝트에서는 `budget` 하나만 sweep (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, ∞ 등 11개 point). |

---

## ① 모델은 강한 텍스트 편향이고, 그게 손해다

`main_b6`에서 모델은 평균 **text 3.28회 / visual 0.87회** 요청 — 텍스트를 약 3.8배 더 선호.

그런데 정확도 비교를 보면 모델이 잘못 고른 쪽이다:

| 정책 (b=6) | text req | visual req | accuracy |
|---|---:|---:|---:|
| `always_text` | 5.64 | 0 | 0.638 |
| `main_b6` (model) | 3.28 | 0.87 | 0.656 |
| `always_visual` | 0 | 4.00 | **0.680** |

자동으로 비전만 쓴 정책이 모델의 적응적 정책을 **+2.4pp** 이긴다. 비전 타일 한 장이 텍스트 한 문장보다 정보 효율이 높음에도 모델은 반대로 행동.

**왜 텍스트 편향?** 추정 원인:
- 사전훈련/instruct 단계에서 텍스트 추론이 더 자주 보상받음
- ScienceQA 텍스트 hint가 "lecture" 스타일이라 모델 입장에서 더 익숙한 형식
- prompt가 모달리티에 중립적이므로 default 선택이 텍스트로 기우는 경향

---

## ② Visual_only 정답 케이스 43개(8.6%)가 main에서 누락됨

500 샘플을 `main_b6` × `always_visual` 정답 여부로 4-bucket 분해:

| bucket | n | 의미 |
|---|---:|---|
| both_right | 297 | 둘 다 정답 |
| both_wrong | 129 | 둘 다 오답 |
| **visual_only** | **43** | always_visual만 정답 → 비전을 더 썼어야 했던 케이스 |
| main_only | 31 | main만 정답 → 텍스트로 풀어낸 케이스 |

`visual_only` 코호트에서 **모델은 평균 text 3.91회 / visual 0.81회** 사용 — 비전이 답이었던 케이스에서조차 텍스트 도배.

Subject별 visual_only 비율:

| subject | n | visual_only 비율 |
|---|---:|---:|
| language science | 14 | **21.4%** |
| natural science | 410 | 9.3% |
| social science | 76 | 2.6% |

언어과학(주로 도표/지도/사진 기반)에서 가장 심각, 사회과학(텍스트 우세)에서는 거의 없음. 도메인 효과 뚜렷.

---

## ③ Prompt nudge로 행동은 뒤집히지만 정확도는 거의 그대로

`run_nudge.py`로 system prompt에 다음 가이드 추가:

> "ScienceQA 이미지는 보통 4 타일로 쪼개지고 종종 결정적 시각 정보를 담는다. 텍스트 hint는 한 문장씩 끊겨서 자주 배경 설명이다. **REQUEST_TEXT를 default로 하지 마라.**"

결과:

| | text req | visual req | wasted | accuracy |
|---|---:|---:|---:|---:|
| `main_b6` (no nudge) | 3.28 | 0.87 | 0.02 | 0.656 |
| `nudge_b6` (with nudge) | **0.01** | **3.72** | **1.00** | 0.660 |
| `always_visual` | 0 | 4.00 | 0 | **0.680** |

**관찰**:
- 모달리티 선호는 한 줄 prompt로 즉시 거의 100% 뒤집힘 (text 3.28 → 0.01).
- 그러나 전체 accuracy는 +0.4pp만 (0.656 → 0.660).
- `wasted=1.0` — 4개 타일을 다 본 뒤에도 "REQUEST_VISUAL" 또 요청. 모델이 모달리티 소진을 잘 인지하지 못함.
- `always_visual`(0.680) 못 미침 → wasted 패널티가 정확히 갉아먹는 만큼.

Subject별 nudge 영향:

| subject | main_b6 | nudge_b6 | 변화 |
|---|---:|---:|---:|
| language science | 0.714 | **0.929** | **+21.5pp** |
| natural science | 0.666 | 0.673 | +0.7pp |
| social science | 0.592 | 0.539 | **−5.3pp** |

→ 일률적 visual-favor가 텍스트 우세 도메인(사회과학)을 **악화**시킴. **단순 nudge는 너무 무딘 도구.** 진짜 필요한 건 질문 유형 conditional routing.

---

## ④ Budget-accuracy 곡선은 **비단조적**이며 b≈7에서 포화

dense sweep (b=0,1,2,3,4,5,6,7,8,10):

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
| 10 | 0.666 | −0.2pp ← 포화 |
| ∞ (full_info) | 0.700 | +3.2pp |

**주요 관찰**:
- **0 → 1의 lift가 가장 큼 (+6.0pp)**. 첫 1 budget만으로도 모델이 유의미한 정보를 끌어온다.
- **b=2에서 local max (0.646), b=3에서 dip (0.626)**. seed와 path dependence 때문에 monotonic하지 않음 — 2턴만에 답을 committed했다면 correct지만, 3턴으로 늘리면 모델이 다른 modality를 요청했다가 오답으로 빠지는 케이스가 생김.
- **b≥7 이후 포화**. b=7/8/10 모두 0.666–0.668로 플래튜. 더 많은 budget을 줘도 모델이 유용하게 활용 못함.
- full_info(0.700)와 b=10(0.666) 사이의 +3.4pp 격차: 가용 정보는 더 있는데 모델 정책이 그만큼 요청하지 않음 (평균 text 3.92 / visual 1.13 vs full의 11.72 / 4.00). **information gathering 의지 부족**.
- `zero_info`만으로 0.562 → ScienceQA의 절반 이상이 사전지식만으로 풀림. 정보 활용으로 가능한 **총 lift는 약 +14pp가 한계** (0.562 → 0.700).

### 비단조성이 의미하는 것
`temperature=0`이고 **tile/sentence 공개 순서도 같은 seed(random=42)** 인데 budget만 바꿨더니 accuracy가 오르락내리락.

- 초기 몇 턴의 모델 결정은 prompt에 보이는 "Remaining budget: N"에 민감. budget이 2일 때는 "빨리 commit"하고, budget이 3이면 "한 번 더 정보 요청" — 그 추가 요청이 오답 쪽으로 끌고 갈 수 있음.
- 모델이 **자기 정보로 판단이 섰을 때 멈추는 능력(stopping rule)** 이 약하다는 방증. budget이 더 있으면 더 달라는 관성 ≈ confidence calibration 부족.

이 곡선의 함의: Qwen2.5-VL-7B는 b=7에서 이미 포화. 더 큰 budget을 줘도 lift가 안 나오니 실제 배치 시 budget=2~3으로 시작해 early-exit 유도가 더 효율적일 수 있음.

---

## ⑤ Early-stop 신호는 b≤3에서만 calibrated — b≥4부터는 무용

**Phase 1a Metric A** ([`output/calibration/summary.csv`](../../output/calibration/summary.csv), 그림: [`docs/calibration_A_spontaneous_vs_forced_acc.png`](../figures/calibration_A_spontaneous_vs_forced_acc.png))

같은 budget run에서 자발 ANSWER로 종료한 샘플 vs FORCED_ANSWER로 종료한 샘플의 정답률 차이.

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

작은 budget(b=2~3)에서는 모델이 자발적으로 멈춘 샘플이 강제로 끊긴 샘플보다 +10~+19pp 더 정확하다 — **"나 답할 수 있어"라는 신호가 진짜 calibrated**. 그런데 b=4부터 격차가 사라지고, b=10에서는 오히려 **forced가 +18pp 더 정확**: budget이 클 때 자발 stop은 그냥 버릇이 되고, 끝까지 못 멈춘 68개가 진짜 어려운 케이스라서 추가 정보로 정답률이 더 오름.

→ 모델은 **작은 budget에서만 stopping rule이 작동하고, budget이 풍부해지면 그 신호를 잃어버림**. 학습된 budget-conditional stopping policy의 필요성에 대한 직접 증거.

---

## ⑥ Cross-budget 답 안정성 = 무료 confidence proxy

**Phase 1a Metric B** ([`per_sample_stability.csv`](../../output/calibration/per_sample_stability.csv), 그림: [`B_choice_stability.png`](../../output/plots/calibration/B_choice_stability.png))

같은 sample을 budget 10 point(b=0,1,2,3,4,5,6,7,8,10)에서 풀었을 때 final_choice가 얼마나 일관적인가:

- 평균 modal-choice fraction = **0.918**
- 10개 run에서 정확히 같은 답을 낸 샘플 = **69.4%**
- modal choice 정답률 = 0.658 (전체 평균과 일치)

stability bin × modal 정답률:

| stability bucket | n | modal 정답률 |
|---|---:|---:|
| ≤0.5 (답이 자주 흔들림) | 16 | 0.625 |
| 0.5–0.7 | 59 | **0.356** ← random보다 약간 위 |
| 0.7–0.85 | 35 | 0.486 |
| 0.85–1.0 (안정) | 390 | **0.721** |

→ **budget 한 차원만 흔들어도 답이 흔들리는 샘플은 모델이 실제로 모르는 샘플.** stability가 logit/entropy 없이도 얻을 수 있는 강한 confidence proxy. 학습 시 abstention target 라벨링이나, inference 시 self-consistency-style 게이트 모두에 재활용 가능.

---

## ⑦ Stop-step 별 정답률 — info=2에서 peak, info=3부터 −20pp 급락

**Phase 1a Metric C** ([`stop_step_acc.csv`](../../output/calibration/stop_step_acc.csv), 그림: [`C_stop_step_accuracy.png`](../../output/plots/calibration/C_stop_step_accuracy.png))

모든 자발 ANSWER trajectory를 (run에 무관하게) stop step별로 묶어서 정답률:

| info_requests | n | accuracy |
|---:|---:|---:|
| 0 (즉답) | 128 | 0.711 |
| 1 | 278 | 0.745 |
| **2** | **384** | **0.794** ← peak |
| 3 | 411 | **0.596** ← **−20pp** |
| 4 | 383 | 0.637 |
| 5 | 370 | 0.600 |
| 6 | 176 | 0.597 |
| 7 | 84 | 0.524 |

**적게 묻고 멈춘 trajectory가 더 정확하다.** 매크로 budget curve의 b=3 dip(④)이 per-trajectory 수준에서도 재현됨 — "더 묻기 시작하는 순간"이 곧 confidence drop 시그널.

Caveat: selection bias 있음 (info=k에서 stop하는 샘플들이 어떤 budget run에서 왔는지에 따라 평균이 달라짐). 그래도 0.794→0.596 급락은 noise 아님 (n 모두 380+).

---

## ⑧ 모델은 "이미 안다"를 인지하지 못한다 — easy/hard 샘플이 거의 같은 양 정보 요구

**Phase 1a Metric F** ([`zero_info_cohort.csv`](../../output/calibration/zero_info_cohort.csv), 그림: [`docs/calibration_F_zero_info_cohort.png`](../figures/calibration_F_zero_info_cohort.png))

zero_info(정보 없이 답)에서 **정답이었던 샘플 281개(zi_correct)** vs **오답이었던 샘플 219개(zi_wrong)** 두 코호트로 나눠서 비교:

| budget | cohort | spon_stop 비율 | 평균 info 요청 | accuracy |
|---:|---|---:|---:|---:|
| 6 | zi_correct | 0.665 | **4.06** | 0.904 |
| 6 | zi_wrong | 0.708 | **4.33** | 0.338 |
| 10 | zi_correct | 0.826 | 5.06 | 0.907 |
| 10 | zi_wrong | 0.913 | 5.16 | 0.356 |

핵심:
1. **모델은 두 코호트에 거의 같은 양의 정보를 요청한다** (b=6에서 차이 0.27건). 사전지식만으로 풀 수 있는 샘플과 정말 정보가 필요한 샘플을 **구분 못 함**.
2. **추가 정보가 쉬운 샘플을 오히려 망친다**: zi_correct accuracy는 b=1에서 **0.954** → b=6에서 **0.904** (**−5pp**). 정보를 더 줄수록 멀쩡한 답을 깨뜨림.
3. zi_wrong은 0.196→0.356 (+16pp). 정보가 어려운 샘플은 일부 구해주지만 동시에 쉬운 샘플을 망가뜨리는 **비대칭 효과**가 매크로 곡선이 0.700에 cap되는 한 이유.

이게 Phase 1a 최강 발견. budget-conditional policy가 *왜* 필요한지 한 그림으로 설명: 동일 budget에서도 sample-level routing(쉬운 건 stop, 어려운 건 spend)이 안 되어 있다는 정량 증거.

---

## ⑨ Forced→Spontaneous 답 일치율 84% — 추가 정보가 모델 답을 거의 안 바꾼다

**Phase 1a Metric D** ([`forced_vs_spon.csv`](../../output/calibration/forced_vs_spon.csv))

같은 sample이 작은 budget에서 FORCED_ANSWER로 끝났고 큰 budget에서는 자발 ANSWER로 끝난 1,981 페어:

- 두 답 일치율 **0.843**
- forced 정답률 0.617 / spontaneous 정답률 0.648 (+3pp만)
- forced budget이 커질수록 일치율도 ↑ (b_forced=0 → 0.79, b_forced=6 → 0.94, b_forced=8 → 1.00)

해석:
- **추가 정보가 모델 답을 크게 바꾸지 않음**. 정보 통합 능력 약함의 또 다른 신호.
- 동시에 **force-answer 자체는 budget curve의 bottleneck 아님** — 만약 강제로 끊는 게 답을 매번 망가뜨린다면 작은 b accuracy가 더 나빠야 했을 텐데, 실제로는 거의 같은 답을 내고 있음. ④의 dip은 force-answer 부작용이 아니라 **모델 정책의 진짜 path-dependence** 때문.

---

## ⑩ Modality bias는 zi_wrong 코호트에서만 손해 — zi_correct에서는 modality 무관

**Phase 1b** ([`output/difficulty/cohort_curve.csv`](../../output/difficulty/cohort_curve.csv), 그림: [`docs/difficulty_A_cohort_accuracy_curve.png`](../figures/difficulty_A_cohort_accuracy_curve.png))

b=6 reference policy를 코호트별로 분해:

| 정책 (b=6) | zi_correct (n=281) | zi_wrong (n=219) | gap |
|---|---:|---:|---:|
| `always_text` | **0.911** | 0.288 | — |
| `always_visual` | **0.911** | 0.384 | **+9.6pp** |
| `full_info` | 0.890 | 0.457 | **+16.9pp** |
| `main_b6` (model) | 0.904 | 0.338 | — |

핵심 두 가지:
1. **zi_correct에서는 always_text == always_visual == 0.911로 정확히 동일.** 쉬운 샘플은 어떤 modality를 줘도 같은 답이 나옴 → modality 선택 자체가 무의미.
2. **zi_wrong에서만 always_visual이 always_text보다 +9.6pp 더 좋다.** ①에서 발견한 "모델 텍스트 편향이 손해"는 사실상 **zi_wrong 코호트에서만 발생하는 현상**. 모델은 두 코호트에 대해 거의 같은 modality 비율(text 3.45/3.07, visual 0.58/1.24)을 적용 — **modality 선택이 실제로 중요한 코호트에서만 그것이 잘못된 default**라는 뼈저린 미스.

→ 단순히 "visual을 더 써라" nudge가 사회과학에서 망가지는 이유(③)도 이걸로 설명: 사회과학은 zi_wrong이 ~60%인데(아래 ⑫) zi_wrong이 visual로 도움받는 정도는 작음 + zi_correct는 modality 무관 + nudge가 forced visual로 zi_correct에서 wasted를 늘림 → 양쪽에서 손해.

---

## ⑪ Budget이 쉬운 샘플을 망치고 어려운 샘플을 살리는 비대칭 (clean cross-over at b=1)

**Phase 1b** (그림: [`docs/difficulty_D_delta_from_b1.png`](../figures/difficulty_D_delta_from_b1.png))

b=1 정확도를 baseline으로 놓고 budget 늘릴 때 코호트별 정답률 변화:

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

- zi_correct는 b=3 이후 −5pp plateau에 갇힘. 추가 정보가 멀쩡한 답을 깨뜨림.
- zi_wrong은 b=7까지 +16pp 단조 증가 후 포화.
- 두 곡선이 정확히 b=1에서 교차 → **샘플 1개당 1 unit의 정보가 모든 sample에 정확히 한 번씩 도움이 되는 sweet spot**, 그 이상은 코호트 따라 ROI가 양분.

이게 ④에서 본 budget curve 비단조성의 메커니즘적 설명. 매크로 곡선은 두 코호트 곡선의 가중 평균이고, zi_correct의 −5pp가 zi_wrong의 +x pp를 갉아먹어서 b=3 dip이 만들어짐. **budget-conditional policy의 핵심 동기**: 같은 평균 budget을 줘도 sample 단위로 routing(easy → 빨리 답, hard → 끝까지 정보)을 해야 매크로 0.700 ceiling을 넘는다.

---

## ⑫ Subject 경제학: natural science는 high-volatility, social science는 low-volatility

**Phase 1b** ([`subject_cohort_sizes.csv`](../../output/difficulty/subject_cohort_sizes.csv), [`subject_cohort_accuracy.csv`](../../output/difficulty/subject_cohort_accuracy.csv), 그림: [`C_subject_cohort.png`](../../output/plots/difficulty/C_subject_cohort.png))

| subject | total n | zi_correct | zi_wrong | zi_correct frac |
|---|---:|---:|---:|---:|
| language science | 14 | 10 | 4 | 0.714 |
| natural science | 410 | 240 | 170 | 0.585 |
| social science | 76 | 31 | 45 | 0.408 |

코호트 × subject × budget 정확도 table에서 가장 흥미로운 두 셀:

- **natural science zi_correct**: b=1 0.950 → b=6 0.908 (−4pp). zi_wrong: 0.182 → 0.359 (+18pp). cost-benefit이 가장 큼.
- **social science zi_correct**: b=1 0.968 → b=6 0.968 (변화 없음). zi_wrong: 0.222 → 0.333 (+11pp), b=10에서 0.267로 하락. **사회과학은 budget을 더 줘도 hard 코호트가 별로 안 살아남.**

즉:
- **natural science**: 추가 정보가 양쪽 코호트에 강한 영향. budget choice가 가장 중요한 도메인.
- **social science**: zi_correct는 견고, zi_wrong는 불응. budget 늘려봤자 효용 적음 — 차라리 zero_info에 가까운 정책이 효율적.
- **language science**: n=14로 통계적으로 무의미.

→ 도메인별 optimal budget이 다르다는 정량 단서. ③에서 본 nudge의 사회과학 −5.3pp 악화도 이걸로 설명: 사회과학은 추가 정보 자체의 marginal utility가 낮은데 거기에 modality까지 강제로 visual로 돌리면 wasted만 늘어남.

---

## ⑬ 모델은 ABSTAIN을 쓸 수 있지만, "답할지 말지" 판단은 zero_info 사전지식 신호와 거의 겹친다

**Phase 1c Metric A + D** ([`output/abstention/summary.csv`](../../output/abstention/summary.csv), [`aligned_comparison.csv`](../../output/abstention/aligned_comparison.csv))

4-action 버전(ANSWER / ABSTAIN / REQUEST_TEXT / REQUEST_VISUAL)을 system prompt에 노출하고 두 budget에서 측정:

| run | coverage (답변 비율) | selective acc (답한 것만) | 비고 |
|---|---:|---:|---|
| `abstain_b0` (b=0) | 12.0% (60/500) | **0.833** | 88%를 abstain — 모델은 starvation에서 "모른다" 잘 인지 |
| `abstain_b6` (b=6) | 90.0% (450/500) | 0.644 | 10%만 abstain |

순진한 해석은 "0.833은 zero_info의 0.562보다 훨씬 높으니 모델이 confidence signal을 잘 쓴다"지만, 이건 **selection confound**. 공정한 테스트는 "abstain_b0이 답하기로 선택한 **바로 그 60개**에서 zero_info는 얼마나 맞히나?":

| 비교 | own acc | 같은 subset에서 reference acc | uplift |
|---|---:|---:|---:|
| `abstain_b0` vs `zero_info` (60개 subset) | 0.833 | **0.800** | **+3.3pp** |
| `abstain_b6` vs `main_b6` (450개 subset) | 0.644 | 0.651 | **−0.7pp** |

→ **"답할지 말지" 결정은 사실상 zero_info가 이미 가지고 있던 사전지식 경계와 같은 신호**. 새로 얻은 selectivity는 b=0에서 +3.3pp, b=6에서 0pp. Abstention이 모델에 *새로운* confidence 축을 더해주지는 않음. 단지 zero_info가 이미 틀리게 답했을 샘플을 "답 안 함"으로 바꿔주는 것.

---

## ⑭ Vanilla abstention은 b=6에서 오히려 anti-calibrated

**Phase 1c Metric B** ([`output/abstention/cohort_xtab.csv`](../../output/abstention/cohort_xtab.csv), 그림: [`docs/abstention_B_cohort_abstain.png`](../figures/abstention_B_cohort_abstain.png))

zi_correct/zi_wrong 코호트별 abstain rate:

| budget | cohort | abstain rate | n |
|---:|---|---:|---:|
| 0 | zi_correct | 0.829 | 281 |
| 0 | zi_wrong | **0.945** | 219 |
| 6 | zi_correct | **0.121** | 281 |
| 6 | zi_wrong | 0.073 | 219 |

**b=0에서는 약하게 calibrated** (zi_wrong abstain 94.5% > zi_correct 82.9%) — 모델이 정보 없을 때는 "더 모르는 쪽을 더 자주 포기"하는 직관을 따름.

**b=6에서는 방향 뒤집힘 — anti-calibration**: 모델은 zi_correct에서 **더 자주 abstain**(12.1%)하고 zi_wrong에서 **덜 abstain**(7.3%). Budget이 있으면 어려운 샘플도 그냥 답해버리고, 오히려 쉬운 샘플에서 가끔 겁먹고 abstain. 결과:

- abstain_b6 selective acc (0.644) < main_b6 overall (0.656)
- Effective Reliability Φ curve는 모든 wrong-answer cost c ∈ [0, 2]에서 abstain_b6 < main_b6 ([`phi_curve.csv`](../../output/abstention/phi_curve.csv)). 즉 **b=6에서 vanilla abstention을 쓰는 건 어떤 cost regime에서도 손해**.
- 반면 abstain_b0 Φ는 c ≥ 1.25에서 zero_info를 역전함 (wrong-answer cost 높을 때 abstention이 값어치 가짐).

→ **단순 prompt 노출만으로는 모델이 abstention을 "지금 나 모르니까 안 답한다"로 쓰지 않고, budget이 있으면 "필요 없는 장치"로 취급하거나 잘못 배치함**. 이게 `project_ko.md` Thread D (R-Tuning / MM-UPD) 방식의 **training-for-abstention이 왜 필수**인지에 대한 직접 증거 — Phase 4 GRPO reward에 abstention 항(`λ_cal` 올바른 abstain, `λ_abs` 잘못된 abstain)을 넣는 디자인 근거.

---

## ⑮ Calibration → anti-calibration flip이 정확히 b=3에서 일어남 — ④의 macro dip과 같은 위치

**Phase 1d / I14** ([`output/abstention_phase1d/I14_cohort_x_budget.csv`](../../output/abstention_phase1d/I14_cohort_x_budget.csv), 그림: [`docs/figures/abstention_phase1d_I14_cohort_x_budget.png`](../figures/abstention_phase1d_I14_cohort_x_budget.png))

⑭에서는 b=0과 b=6 두 점만 비교했지만, 이번엔 b=1..5,7,8,10을 추가 측정해서 **abstain rate가 cohort별로 budget에 따라 어떻게 진화하는지** 풀 곡선을 그렸다.

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

**핵심**: sign flip은 점진적이 아니라 **b=2와 b=3 사이에서 단번에 일어나고, 그 이후로는 안 돌아온다**. b≤2에서는 모델이 어려운 샘플(zi_wrong)에 더 자주 abstain → calibrated. b≥3부터는 zi_correct에서 더 자주 abstain → anti-calibrated 영구 유지, 게다가 b=10에서 격차 −6.9pp까지 벌어짐.

이 b=3 transition은 ④에서 본 **macro budget curve의 b=3 dip**과 정확히 같은 위치다. 두 현상이 같은 메커니즘을 공유한다는 강한 시사:
- b≤2에서 모델은 "정보 부족"을 진심으로 인지 → 어려운 샘플에 abstain, 쉬운 샘플엔 답.
- b=3을 기점으로 모델이 "이 정도면 답해야 한다"고 판단 → 어려운 샘플에서도 잘못된 답을 내고, 오히려 쉬운 샘플에서 가끔 abstain하는 (anti-)calibration 패턴 발생.
- macro 곡선이 b=3에서 dip하는 이유: 어려운 샘플에서 abstain → 잘못된 답으로 전환 (정답률 ↓), 동시에 쉬운 샘플에선 일부가 abstain로 빠짐 (역시 정답률 ↓).

→ 두 phenomenon이 한 figure로 정리됨. 학습 시 budget-conditioning과 abstention reward를 **함께** 설계해야 하는 구체적 근거 (paper Phase 4 GRPO setup의 `λ_cost`와 `λ_abs`/`λ_cal`이 분리되면 안 되는 이유).

---

## ⑯ 이미지 마스킹해도 abstain rate 거의 안 변함 (Δ ≤ 2pp) — vanilla abstention은 image-level sufficiency-blind

**Phase 1d / I13** ([`output/abstention_phase1d/I13_masked_vs_unmasked.csv`](../../output/abstention_phase1d/I13_masked_vs_unmasked.csv), 그림: [`docs/figures/abstention_phase1d_I13_masked_vs_unmasked.png`](../figures/abstention_phase1d_I13_masked_vs_unmasked.png))

⑭의 anti-calibration이 "모델이 image content에 대한 sufficiency를 정말 못 본다"인지 직접 테스트하려고, ScienceQA에서 100개(50 zi_correct + 50 zi_wrong, seed 42)를 골라 **4개 타일 모두 white(255,255,255)로 대체**한 변형(`preproc_masked/`)을 만들고 같은 b=0/4/6에서 abstain 행동을 비교.

| budget | cohort | unmasked abstain | masked abstain | Δ |
|---:|---|---:|---:|---:|
| 0 | zi_correct | 0.74 | 0.74 | **0.00** |
| 0 | zi_wrong | 0.96 | 0.96 | **0.00** |
| 4 | zi_correct | 0.12 | 0.14 | +0.02 |
| 4 | zi_wrong | 0.20 | 0.20 | **0.00** |
| 6 | zi_correct | 0.10 | 0.10 | **0.00** |
| 6 | zi_wrong | 0.10 | 0.12 | +0.02 |

**모든 셀에서 Δ ≤ 2pp** — 50개 코호트 기준 1 샘플 차이 = noise 수준. 모델은 4개 타일이 전부 white라는 사실을 **abstention 결정에 거의 반영하지 않는다**.

게다가 정확도(naive accuracy)도 마스킹해도 거의 떨어지지 않음 (b=4 zi_correct 0.84→0.82, b=6 zi_correct 0.86→0.84, b=6 zi_wrong 0.24→0.24). 두 가능성:
1. **ScienceQA text(hint+lecture)만으로 답이 충분히 derivable** — 이미지가 명목상으로만 있고 실질적으로는 redundant인 케이스가 많음.
2. **모델이 white tile을 "정보 없음"이 아닌 "image as usual"로 처리** — visual encoder가 white를 의미 있는 픽셀로 받고, 모델은 sufficiency를 이미지 content가 아니라 텍스트 confidence에서만 본다.

두 효과가 섞여서 결정적 분리는 못 하지만, 적어도 **vanilla abstention은 image-level sufficiency를 추적하지 않는다**는 결론은 안전하다 (추적했다면 zi_wrong 코호트의 마스킹된 케이스에서 abstain rate ↑가 보여야 함).

**Caveat**: 100 샘플은 작음 (cohort당 50 → 95% CI ±~14%). 그리고 ScienceQA는 image-required-by-construction 벤치마크가 아님. **결정적 sufficiency 테스트**는 V\*Bench / HR-Bench(이미지가 답에 필수) + image-masked 변형으로 재실행 — ROADMAP backlog I15로 등록(Phase 2 초반).

→ ⑮(b=3에서 flip)와 ⑯(image-blind)을 합치면 그림이 단단해짐: vanilla abstention은 image content가 아니라 **text-based confidence**만 본다, 그리고 그 confidence는 b=3을 기점으로 잘못 calibrated되기 시작. R-Tuning / GRPO 학습은 이 두 결함을 모두 해결해야 함.

---

## 한계와 caveats

- **단일 모델**: Qwen2.5-VL-7B만 평가. 모달리티 편향이 모델 fine-tuning 패턴에 종속될 가능성 — LLaVA/InternVL은 다를 수 있음.
- **Greedy decoding (T=0)**: temperature=0. sampling을 켜면 모델 결정이 흔들리고 결과 다를 수 있음.
- **2×2 타일 = 4장**: 더 잘게 자르면 (3×3=9, 4×4=16) 비전 cost 단위가 작아져 budget 분배 dynamics가 달라짐.
- **텍스트 chunk 크기**: 문장 단위. 더 큰 청크(문단)로 자르면 budget당 정보량이 늘어 비교 결과 흔들릴 수 있음.
- **Subject 분포 편향**: 410/500이 natural science. social(76)/language(14)는 표본 적어 결론 약함.
- **단일 nudge prompt**: 다른 표현/위치/길이로 시도하면 결과 다를 수 있음. 본 실험은 한 가지 nudge만 측정.
- **Tile reveal order**: shuffled (seeded). 항상 같은 순서로 공개되면 모델이 "처음 본 타일이 항상 같은 위치" 같은 휴리스틱으로 갈 수 있어 random shuffle을 default로 사용. row-major 비교는 미실험.
- **Phase 1c sufficiency set은 skip**: 원래 ROADMAP은 MM-UPD subset 또는 ScienceQA image-masked 변형을 쓰는 sufficiency-known mini-set을 명시했지만 이 phase에서는 zi_correct/zi_wrong cohort를 proxy로 사용. ⑬–⑭는 **자연스럽게 존재하는 ScienceQA 샘플**에서의 모델 abstention 행동 진단이지, unanswerable-by-construction stimuli에 대한 직접 테스트는 아님. 후자는 Phase 1d 또는 Phase 2 follow-up으로 이월 (ROADMAP 참조).

---

## 다음 단계 후보

다음 단계는 [`roadmap_ko.md`](../../references/roadmap_ko.md)에 phase별로 정리되어 있음. Phase 1a (calibration ⑤–⑨) 완료. 다음 우선순위:

- **Phase 1b** Difficulty stratification: zero_info 코호트로 budget curve 다시 그리기 (어떤 budget이 어떤 코호트에 효과적인지 분리)
- **Phase 1c** Abstention proxy: vanilla 모델이 ABSTAIN action을 얼마나 쓰는지 + sufficiency-known 작은 set 만들기
- **Phase 2** 6-action vocab 확장 (`ABSTAIN`/`THINK`/`ZOOM(bbox)`/`REQUEST_HI_RES`) + 새 벤치마크 1종 (V*Bench or HR-Bench)
- **Phase 3+** SFT 데이터 → GRPO budget-conditional training (논문 본 컨트리뷰션)

---

## 산출물 위치

- `notebooks/experiment.ipynb` — 데모 노트북 (이 문서의 시각화·표 모두 포함)
- `output/all_runs_summary.csv` — 위의 비교표 원본
- `output/subject_crosstab.csv` — subject × 정책 cross-tab
- `output/visual_bias_breakdown.csv` — sample-level main vs always_visual bucket
- `output/plots/accuracy_vs_budget.png`, `output/plots/modality_mix.png` — plot 원본
- `output/<run>/predictions.{parquet,csv,jsonl}` — run별 sample-level raw (jsonl은 step trace 포함)
- `output/<run>/summary_{overall,by_subject}.{csv,jsonl}` — run별 집계
- `output/calibration/` — Phase 1a 산출물 (summary.csv / per_sample_*.csv / stop_step_acc.csv / forced_vs_spon.csv / zero_info_cohort.csv)
- `output/plots/calibration/A..F_*.png` — calibration 시각화
- `output/difficulty/` — Phase 1b 산출물 (cohort_curve.csv / cohort_modality.csv / subject_cohort_*.csv / peak_budget.csv)
- `output/plots/difficulty/{A,B,C,D}_*.png` — difficulty 시각화
- `output/abstain_b0/`, `output/abstain_b6/` — Phase 1c raw prediction (run_abstention.py 결과)
- `output/abstention/` — Phase 1c 분석 산출물 (summary / cohort_xtab / phi_curve / aligned_comparison)
- `output/plots/abstention/{A,B,C}_*.png` — abstention 시각화
- `docs/figures/*.png` — insights 헤드라인 figure 5장 (복사본, 아래 "figure" 절 참조)
