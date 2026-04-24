# 인사이트 노트 — Budget-Constrained Sequential Information-Seeking

> Qwen2.5-VL-7B-Instruct를 ScienceQA 500개 샘플에서 budget sweep(b=0…10) + 베이스라인 정책 4종으로 평가한 결과. 자세한 데모는 [`experiment.ipynb`](./experiment.ipynb), 원시 데이터는 [`output/`](./output) 참고.

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
![accuracy vs budget](output/plots/accuracy_vs_budget.png)

### 정책별 모달리티 사용량
![modality mix](output/plots/modality_mix.png)

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

## 한계와 caveats

- **단일 모델**: Qwen2.5-VL-7B만 평가. 모달리티 편향이 모델 fine-tuning 패턴에 종속될 가능성 — LLaVA/InternVL은 다를 수 있음.
- **Greedy decoding (T=0)**: temperature=0. sampling을 켜면 모델 결정이 흔들리고 결과 다를 수 있음.
- **2×2 타일 = 4장**: 더 잘게 자르면 (3×3=9, 4×4=16) 비전 cost 단위가 작아져 budget 분배 dynamics가 달라짐.
- **텍스트 chunk 크기**: 문장 단위. 더 큰 청크(문단)로 자르면 budget당 정보량이 늘어 비교 결과 흔들릴 수 있음.
- **Subject 분포 편향**: 410/500이 natural science. social(76)/language(14)는 표본 적어 결론 약함.
- **단일 nudge prompt**: 다른 표현/위치/길이로 시도하면 결과 다를 수 있음. 본 실험은 한 가지 nudge만 측정.
- **Tile reveal order**: shuffled (seeded). 항상 같은 순서로 공개되면 모델이 "처음 본 타일이 항상 같은 위치" 같은 휴리스틱으로 갈 수 있어 random shuffle을 default로 사용. row-major 비교는 미실험.

---

## 다음 단계 후보

1. **Subject conditional routing**: 질문 텍스트의 키워드(예: "map", "diagram", "passage")로 visual/text 우선순위 결정. nudge보다 정밀.
2. **Few-shot ICL**: 질문 유형별 modality 결정 예시를 1-3개 넣고 그 효과 측정.
3. **다른 VLM 비교**: LLaVA-1.6, InternVL2, Pixtral 등에 동일 프로토콜 적용. 텍스트 편향이 모델 의존인지 task 의존인지 분리.
4. **Tile order 효과**: `row_major` vs `shuffled` 비교 — 모델이 spatial 위치 정보를 활용하는지.
5. **Budget 단위 비대칭화**: visual cost를 0.5로, text cost를 1로 설정해서 모델이 "비싼" 모달리티를 쓰는지 본다.
6. **모델 calibration**: trace에서 force-answer 직전과 자발 ANSWER 시점의 답이 얼마나 다른지 → 모델이 자신의 confidence를 얼마나 잘 추정하는지 정량화.
7. **샘플 difficulty stratification**: zero_info에서 틀린 샘플 vs 맞은 샘플로 나눠 budget 활용 패턴 비교.

---

## 산출물 위치

- `experiment.ipynb` — 데모 노트북 (이 문서의 시각화·표 모두 포함)
- `output/all_runs_summary.csv` — 위의 비교표 원본
- `output/subject_crosstab.csv` — subject × 정책 cross-tab
- `output/visual_bias_breakdown.csv` — sample-level main vs always_visual bucket
- `output/plots/accuracy_vs_budget.png`, `output/plots/modality_mix.png` — plot 원본
- `output/<run>/predictions.{parquet,csv,jsonl}` — run별 sample-level raw (jsonl은 step trace 포함)
- `output/<run>/summary_{overall,by_subject}.{csv,jsonl}` — run별 집계
