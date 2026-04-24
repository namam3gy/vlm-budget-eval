# 예산 조건 기반 에이전트형 VLM: 혼잡하지만 갈라낼 수 있는 최전선

**핵심 결론.** 제안된 아이디어 — VLM에 토큰/연산 예산을 주고, 그 예산을 확대(zoom), 추론(reasoning), 검색(retrieval), 기권(abstention) 사이에 배분하도록 학습시키는 것 — 은 **부분적으로 선점되었지만 여전히 출판 가능**하다. 2025년 4월부터 2026년 4월 사이에, 최소한 12편 이상의 논문이 RL을 통해 VLM이 확대하거나 시각 도구를 호출하도록 학습시켰다(VisionThink, DeepEyes, Pixel Reasoner, Chain-of-Focus, Pixel Reasoner, Active-O3, Reinforcing-VLMs-Under-Resource-Constraints). 별도로, 텍스트 LLM 커뮤니티는 예산 조건부 정책을 이미 해결해 왔고(L1/LCPO, Elastic Reasoning, TALE), 에이전트 커뮤니티는 웹 에이전트를 위한 예산 인식 테스트타임 스케일링을 도입했다(BATS, 2025년 11월). **깨끗한 빈 공간은 연구자가 지적하고 있는 바로 그 교차점이다:** (a) 명시적 토큰/달러 예산 **B를 입력으로 받는** VLM 정책, (b) 기권을 포함하는 **이질적인 action space** 전반에 예산을 배분하는 정책, (c) 하나의 효율적 operating point가 아니라 전체 비용–정확도 **Pareto frontier**를 추적하는 정책. 이 세 가지를 모두 멀티모달 에이전트에 대해 수행한 단일 공개 논문은 없다. 해당 framing을 중심으로 범위를 타이트하게 잡는다면, 1–8대의 H200에서 2–3주 실행으로도 NeurIPS 또는 EMNLP 제출에 충분히 신뢰할 수 있는 결과를 만들 수 있다.

---

## Part 1 — 문헌 리뷰

### Thread A: VLM에서의 적응적 및 동적 시각 토큰

토큰 압축 문헌은 **정적 압축**(고정 비율)과 **적응적 획득**(시각 토큰을 어디에 쓸지에 대한 학습된 정책)으로 나뉜다. 정적 계열 — **FastV** (Chen et al., ECCV 2024, 2403.06764), **VisionZip** (Yang et al., CVPR 2025, 2412.04467), **PyramidDrop** (Xing et al., CVPR 2025, 2410.17247), **LLaVA-PruMerge** (Shang et al., 2403.15388), **SparseVLM** (Zhang et al., ICML 2025, 2410.04417), **TokenPacker** (Li et al., IJCV 2025, 2407.02392), **VoCo-LLaMA** (Ye et al., CVPR 2025, 2406.12275), 그리고 **FasterVLM** (Zhang et al., 2412.01818) — 은 모두 고정 비율이나 attention heuristic을 통해 시각 토큰을 prune한다. 이들은 제안과는 주변적 관계에 있다. 고정된 예산을 줄일 뿐, **얼마나** 쓸지를 결정하지는 않기 때문이다.

두 개의 정적 논문은 *substrate*로 인용할 만한 구조적 enabler이다. **Matryoshka Multimodal Models** (Cai et al., ICLR 2025, 2405.17430)는 downstream policy가 instance별로 해상도를 선택할 수 있도록 nested visual-token granularity를 학습한다 — 거의 이상적인 action-space substrate이다. **Qwen2-VL / Qwen2.5-VL** (Wang et al., 2409.12191 / 2502.13923), **LLaVA-NeXT AnyRes**, **InternVL 1.5/2.5** (Chen et al., 2404.16821 / 2412.05271), **Monkey** (Li et al., CVPR 2024), **LLaVA-UHD** (Guo et al., ECCV 2024, 2403.11703), 그리고 **mPLUG-DocOwl 1.5** (Hu et al., EMNLP Findings 2024)의 dynamic resolution은 모든 budget-aware 방법에 필요한 variable-token-count backbone을 제공한다.

적응적 획득 계열이 바로 제안이 실제로 경쟁하는 영역이다. **V\*** (Wu & Xie, CVPR 2024, 2312.14135)는 VQA LLM이 evidence가 충분하지 않을 때 visual search를 호출하는 SEAL meta-architecture를 도입했다. **Visual CoT** (Shao et al., NeurIPS 2024, 2403.16999)는 438k QA pair로 bbox-zoom을 supervised했다. **Chain-of-Spot** (Liu et al., 2403.12966)은 two-turn ROI policy를 학습한다. **CogCoM** (Qi et al., 2402.04236)은 zoom, crop, ground, count를 포함하는 full chain-of-manipulations action set을 학습한다. **Visual Sketchpad** (Hu et al., NeurIPS 2024, 2406.09403)는 모델에 중간 추론 도구로 drawing tool을 제공한다. Training-free zoom에는 **ZoomEye** (Shen et al., EMNLP 2025, 2411.16044), **ViCrop** (Zhang et al., ICLR 2025, 2502.17422), 그리고 **DC²** (Wang et al., AAAI 2025, 2408.15556)가 있다.

**가장 직접적으로 관련된 작업 — 반드시 경쟁 논문으로 다루어야 하는 것 — 은 2025년의 RL-trained visual-agent policy들이다.** **VisionThink** (Yang et al., NeurIPS 2025, 2507.13348)는 ¼ 해상도에서 시작하여 full resolution을 요청하는 단일 tool call을 RL로 학습하며, resize-call ratio에 penalty를 둔다. **DeepEyes** (Zheng et al., 2505.14362)는 `image_zoom_in_tool`과 함께 cold-start SFT 없이 end-to-end GRPO를 수행한다. **Pixel Reasoner** (Su et al., NeurIPS 2025, 2505.15966)는 curiosity-driven reward와 efficiency penalty를 사용해 `zoom-in`과 `select-frame`을 학습한다. **Chain-of-Focus** (2505.15436)는 adaptive sufficient-evidence detection을 추가한다. **Active-O3** (Zhu et al., 2505.21457)는 GRPO를 통해 MLLM의 active perception을 정식화한다. **Reinforcing VLMs to Use Tools under Resource Constraints** (2506.14821)는 제목상 가장 많이 겹치는 단일 선행 연구로, training-time tool-call limit 아래에서 Qwen2.5-VL-3B를 GRPO로 학습한다.

### Thread B: 테스트타임 연산 및 추론 예산 최적화

텍스트 LLM 커뮤니티는 아직 멀티모달 equivalent가 없는 성숙한 도구 상자를 가지고 있다. 기초에는 OpenAI o1 (2412.16720), **DeepSeek-R1** (Guo et al., Nature 2025, 2501.12948), **Snell et al.의 "compute-optimal test-time scaling"** (2408.03314), **Large Language Monkeys** (Brown et al., 2407.21787), 그리고 **s1 / budget forcing** (Muennighoff et al., EMNLP 2025, 2501.19393)이 포함된다. 예산 인식 reasoning 자체는 **TALE** (Han et al., ACL Findings 2025, 2412.18547)에 의해 뒷받침된다. TALE은 token-budget adherence를 위한 prompting+SFT+DPO를 선구적으로 제시한다. **L1/LCPO** (Aggarwal & Welleck, CMU 2025, 2503.04697)는 prompt에 지정된 length를 준수하도록 모델을 RL 학습한다. **Elastic Reasoning** (Xu et al., Salesforce 2025, 2505.05315)은 separated thinking/solution phase에 대한 budget-constrained GRPO rollout을 사용한다. **AdaCoT** (Lou et al., 2505.11896)는 복잡한 query에서만 CoT를 trigger한다. **Chain of Draft** (Xu et al., Zoom, 2502.18600)는 step별 aggressive compression을 위한 것이다. 표준적인 Sui et al.의 **"Stop Overthinking" survey** (TMLR 2025, 2503.16419)와 Feng et al.의 multimodal-aware survey (2503.21614)는 multimodal gap을 명시적으로 확인한다.

Adaptive computation은 Graves의 **ACT** (1603.08983), **PonderNet** (Banino et al., 2107.05407), 그리고 **Mixture-of-Depths** (Raposo et al., 2404.02258)로 거슬러 올라간다 — vision-token-level MoD에 대한 자연스러운 템플릿이다. Cascade와 routing은 **FrugalGPT** (Chen, Zaharia, Zou, TMLR 2024, 2305.05176), **RouteLLM** (Ong et al., ICLR 2025, 2406.18665), **Hybrid LLM** (Ding et al., ICLR 2024, 2404.14618), 그리고 **AutoMix** (Madaan et al., NeurIPS 2024, 2310.12963)가 주도한다.

**중요한 발견: 2024–2025년의 multimodal-reasoning 논문들 중 어느 것도** — LLaVA-CoT/LLaVA-o1 (Xu et al., ICCV 2025, 2411.10440), Mulberry (Yao et al., NeurIPS 2025, 2412.18319), Insight-V (Dong et al., CVPR 2025, 2411.14432), Virgo (Du et al., 2501.01904), Vision-R1 (Huang, 2503.06749), MM-Eureka (Meng et al., 2503.07365) — **모델이 준수하도록 학습하는 명시적 inference-time budget을 부과하지 않는다.** 이들은 긴 CoT를 유도하지만, 그것을 제약하지는 않는다. 이것이 가장 명확한 opening이다.

### Thread C: 에이전트형 cost efficiency와 tool-use 최적화

Budget-aware evaluation에 관련된 benchmark에는 **GAIA** (Mialon et al., 2311.12983)가 있으며, Princeton HAL leaderboard를 통해 first-class token accounting을 제공한다. **OSWorld** (Xie et al., NeurIPS 2024, 2404.07972)와 **OSWorld-Human** (2506.16042)은 step/latency breakdown을 추가한다. **SWE-agent** (Yang et al., 2405.15793)는 이미 instance당 $4 budget을 부과한다. **VisualWebArena** (Koh et al., ACL 2024, 2401.13649), 그리고 공개된 test-time compute curve가 있는 **BrowseComp** (Wei et al., OpenAI 2025, 2504.12516)도 관련된다. **WAREX** (2510.03285)와 **WABER**는 WebArena에 cost logging을 retrofit한다. 2–3주 timeline에서는 이들은 과도하게 큰 범위다. 더 단순한 academic benchmark(V\*Bench, HR-Bench, MM-UPD)가 더 현실적이다.

Adaptive-retrieval 문헌은 가장 깔끔한 방법론적 선례다. **Toolformer** (Schick et al., NeurIPS 2023, 2302.04761), **Adaptive-RAG** (Jeong et al., NAACL 2024, 2403.14403), **Self-RAG** (Asai et al., ICLR 2024 Oral, 2310.11511), **FLARE** (Jiang et al., EMNLP 2023, 2305.06983), **When to Retrieve / Adapt-LLM** (Labruna et al., 2404.19705), 그리고 특히 **SmartRAG** (Gao et al., ICLR 2025, 2410.18141) — 명시적인 cost-minimizing reward를 사용해 *언제* 검색할지에 대한 joint policy를 RL로 학습한다 — 는 제안된 action space에 직접 매핑된다.

명시적으로 예산 제약을 둔 agent는 여전히 드물다. **BudgetMLAgent** (Gandhi et al., 2411.07464)는 hard GPT-4-call cap과 cheap-to-expensive cascade를 사용한다. **Efficient Agents / cost-of-pass** (Wang et al., 2508.02694)는 GAIA에서 OWL 성능의 96.7%를 유지하면서 cost-of-pass를 28.4% 낮추며, 제안이 선호해야 할 metric vocabulary를 제공한다. **Scaling Test-time Compute for LLM Agents** (Zhu et al., 2506.12928)는 parallel vs. sequential scaling을 연구한다. **BATS / Budget Tracker** (2511.17006, 2025년 11월)는 budget-constrained tool-augmented agent에 대한 최초의 체계적 연구이며 unified token+tool-call cost metric을 도입한다 — **가장 중요한 framing competitor**이지만 LLM-only이다(visual action 없음). Visual/GUI agent — **CogAgent** (Hong et al., CVPR 2024 Highlight, 2312.08914), UI-guided visual-token pruning을 갖춘 **ShowUI** (Lin et al., CVPR 2025, 2411.17465), **UGround** (Gou et al., ICLR 2025 Oral, 2410.05243), **OS-Atlas** (Wu et al., ICLR 2025, 2410.23218), **Aguvis** (Xu et al., ICML 2025, 2412.04454), **UI-TARS** (2501.12326), 그리고 **UI-TARS-2** (2509.02544) — 는 잠재적 downstream testbed이지만, 이들의 budget-aware training은 탐구되지 않았다.

### Thread D: VLM에서의 기권, calibration, 그리고 "모르겠다"

Hallucination benchmark는 잘 확립되어 있다. **POPE** (Li et al., EMNLP 2023, 2305.10355), **MME** (Fu et al., 2306.13394), **HallusionBench** (Guan et al., CVPR 2024, 2310.14566), **MMHal-Bench** (Sun et al., 2309.14525), **AMBER** (Wang et al., 2311.07397), **CHAIR** (Rohrbach et al., EMNLP 2018), 그리고 **MMVP / Eyes Wide Shut** (Tong et al., CVPR 2024, 2401.06209)가 있다. 이들은 hallucination을 post-hoc으로 측정한다. 기권을 학습시키지는 않는다.

Training-for-abstention은 **R-Tuning** (Zhang et al., NAACL 2024 Outstanding, 2311.09677)에 기반한다. R-Tuning은 knowledge-intersection SFT data를 구성하며 가장 transferable한 recipe이다. 여기에 Lin et al.의 **"Teaching Models to Express Uncertainty in Words"** (TMLR 2022, 2205.14334), Kamath et al.의 **selective QA under domain shift** (ACL 2020), **SelfCheckGPT** (Manakul et al., EMNLP 2023, 2303.08896), LLM이 지속적으로 overconfident하다는 것을 보인 Xiong et al.의 ICLR 2024 confidence-elicitation study (2306.13063), Tian et al.의 **Just Ask for Calibration** (EMNLP 2023, 2305.14975), 그리고 Band et al.의 **Linguistic Calibration** (2404.00474)이 함께 놓인다.

VLM-specific abstention은 가장 가까운 adjacent sub-field이다. **Miyai et al.의 Unsolvable Problem Detection / MM-UPD** (ECCV 2024, 2403.20331)는 세 가지 subtype(AAD/IASD/IVQD)을 갖는 gold-standard benchmark이다. **Visually Dehallucinative Instruction Generation / IDK-Instructions** (Cha et al., 2402.09717)는 R-Tuning의 multimodal analog와 VQAv2-IDK evaluation set을 제공한다. **UNK-VQA** (Guo et al., TPAMI 2024, 2310.10942)는 image 또는 question을 의도적으로 perturb한다 — 제안된 masking diagnostic과 가장 가까운 기존 mirror이다. **VL-Uncertainty** (Zhang et al., 2411.11919), **AvisC** (Woo et al., ACL Findings 2025, 2405.17820), 그리고 Effective Reliability Φ metric을 갖춘 **Reliable VQA** (Whitehead et al., ECCV 2022, 2204.13631)는 calibration을 다룬다. 두 편의 2026년 논문은 특히 주의해야 한다. **"Reading Between the Lines"** (2511.19806)는 white box와 Gaussian blur를 이용한 visual occlusion을 stress test로 사용하고, corruption type 전반에 generalize되는 **latent-representation probe**를 학습한다. **MM-AQA** (2604.14799)는 partial masking을 통해 "Missing Visual Information"과 "Occlusion Ambiguity" category를 명시적으로 구성하고, 이것이 가장 어려운 abstention regime이라는 점을 발견한다(26.2% unanswerable accuracy coverage) — **이 두 논문은 제안된 masking diagnostic의 novelty를 크게 좁히므로 반드시 인용해야 한다**.

Information-seeking QA는 또 다른 템플릿을 제공한다. **CLAM** (Kuhn et al., 2212.07769)은 clarifying-question generation을 위한 것이고, **Self-Ask** (Press et al., EMNLP Findings 2023, 2210.03350)도 있다. 특히 중요한 것은 **MediQ** (Li et al., NeurIPS 2024, 2406.00922)로, static QA를 abstention과 follow-up question이 first-class action인 interactive benchmark로 변환한다. **AbstentionBench** (Kirichenko et al., 2506.09038)는 reasoning fine-tuning이 오히려 abstention을 *악화*시킨다고 보고한다 — 이는 training-time abstention-aware objective를 위한 강력한 동기다.

### Thread E: Active perception과 partial observation

계보는 **Recurrent Models of Visual Attention / RAM** (Mnih et al., NeurIPS 2014, 1406.6247), **Ba et al.의 multi-object attention** (ICLR 2015, 1412.7755), 그리고 **DRAW** (Gregor et al., ICML 2015, 1502.04623)에서 시작된다 — 모두 RL 또는 differentiable glimpse policy이다. 인지과학적 anchor는 **Ullman et al.의 "Atoms of Recognition" / MIRCs** (PNAS 2016, 113(10):2744–2749)로, 인간 인식에 필요한 minimal image configuration을 식별한다 — 제안된 masking diagnostic이 묻는 바로 그 epistemic question이다. Active VQA는 Shih et al.의 **Where To Look** (CVPR 2016, 1511.07394)까지 거슬러 올라가며, 현대적인 embodied version에는 **Active Neural SLAM** (Chaplot et al., ICLR 2020, 2004.05155)과 매우 최근의 **VG-AVS / Toward Ambulatory Vision** (2512.13250)이 있다. 가장 중요한 최근 항목은 **ActiView** (Wang et al., ACL 2025, 2410.04659)이다 — perceptual field를 제한하고 active zoom/view-shift를 요구하는 MLLM benchmark로, 27–30개 모델을 평가하고 큰 active-perception gap을 드러낸다. **SalBench** (2507.04741)는 low-level saliency를 probe한다. GPT-4o조차 47.6%에 그친다.

### Thread F: 가장 가까운 선행 연구와 scooping risk

아래 12편의 논문은 scooping risk 순으로 정렬되어 있다. **반드시 인용해야 하는 경쟁 연구**는 ⚠️로 표시했다.

| 순위 | 논문 | arXiv | 선점 위험 | 남은 white space |
|---|---|---|---|---|
| 1 | VisionThink (NeurIPS 2025) ⚠️ | 2507.13348 | **높음** | budget conditioning 없음; binary action만 있음; abstention 없음 |
| 2 | Reinforcing VLMs Under Resource Constraints ⚠️ | 2506.14821 | **높음** | input-conditioned budget이 아니라 training-time constraint; zoom-only |
| 3 | Chain-of-Focus / Adaptive-CoF ⚠️ | 2505.15436 | **높음** | 명시적 budget 없음; non-visual tool 없음; abstention 없음 |
| 4 | DeepEyes ⚠️ | 2505.14362 | **높음** | reward에 budget term 없음; Pareto 없음; single tool |
| 5 | Pixel Reasoner (NeurIPS 2025) ⚠️ | 2505.15966 | **높음** | curiosity reward이지 budget 아님; abstention 없음; cost-Pareto 없음 |
| 6 | BATS / Budget Tracker (Nov 2025) ⚠️ | 2511.17006 | **중-높음** | LLM web-search only; visual action 없음 |
| 7 | VRAG-RL (NeurIPS 2025) | 2505.22019 | 중간 | 명시적 budget 없음; RAG 중심 |
| 8 | Active-O3 | 2505.21457 | 중간 | cost constraint 없음; localization 중심 |
| 9 | AVR / Adaptive VLM Routing | 2603.12823 | 중간 | model-selection routing만 있음; action policy 없음 |
| 10 | AwaRes | 2603.16932 | 중간 | HR-crop retrieval만 있음; heterogeneous action 없음 |
| 11 | VTool-R1 | 2505.19255 | 중간 | tool-use RL; budget 없음 |
| 12 | OpenThinkIMG / V-ToolRL | 2505.08617 | 중간 | infrastructure + RL; budget 없음 |

낮은 위험도의 supporting cast: Visual Sketchpad, ZoomEye, T3-Agent (ICLR 2025 Spotlight, 2412.15606), CropVLM (2511.19820), VLM-R³, Argus, GRIT, "Don't Look Only Once" (2505.18842), ReCoVERR (2402.15610).

---

## Part 2 — Gap analysis

### 진정으로 새로운 것

서로 맞물린 세 가지 contribution은 결합될 경우 방어 가능하다. **첫째, 명시적 token 또는 dollar budget B를 input state의 일부로 받고 전체 cost–accuracy Pareto curve를 추적하는 budget-conditional VLM policy π(a|s,B).** L1/LCPO는 text-only CoT length에 대해 이를 수행하고, Elastic Reasoning은 partitioned phase에 대해 이를 수행하지만, **B를 policy의 입력으로 받는 multimodal agent paper는 없다** — VisionThink, DeepEyes, Pixel Reasoner, Chain-of-Focus, Reinforcing-VLMs-Under-Resource-Constraints는 모두 하나의 "efficient" operating point를 학습한다. **둘째, visual acquisition, reasoning, abstention을 통합하는 heterogeneous action space.** 기존 VLM-RL 작업은 하나 또는 두 개의 action(zoom, select-frame)을 노출한다. `answer / abstain / extend-CoT / request-high-res / zoom(bbox) / retrieve`를 하나의 budget-aware policy 아래 통합하는 것은 공개되지 않았다. **셋째, 명시적으로 budgeted된 abstention action.** Abstention은 LLM(R-Tuning)과 VLM benchmark(MM-UPD, VQAv2-IDK, MM-AQA)에 존재하지만, tool call과 trade off되는 budget-consuming action으로 다루어진 적은 없다.

네 번째, 더 선택적인 contribution은 **masking diagnostic을 information-theoretic 또는 minimal-image framework**(Ullman et al.) 안에서 framing하는 것이다. 이는 engineering-motivated 논문이 많은 이 영역에서 드문 방식이다. 그러나 MM-AQA (2604.14799)와 Reading-Between-the-Lines (2511.19806)는 이미 체계적인 VLM masking test를 수행했다. 따라서 diagnostic 자체는 더 이상 novel하지 않으며, 연구자가 이미 의도한 것처럼 contribution이 아니라 **budget-optimization method 내부의 signal**로 framing해야 한다.

### 가장 심각한 scooping risk

겹침은 실제로 존재한다. **VisionThink와 DeepEyes**는 RL-for-visual-tools paradigm을 확립한다. **Pixel Reasoner**는 pixel-space action-set template을 제공한다. **Chain-of-Focus**는 "adaptive sufficiency" framing을 주장한다. **Reinforcing-VLMs-Under-Resource-Constraints** (2506.14821)는 제목과 명시적인 "resource constraints" framing에서 가장 가깝지만, test-time budget에 policy를 condition하는 것이 아니라 train time에 tool call을 cap한다. 가장 높은 수준의 framing risk는 **BATS (2511.17006)**이다. 이 논문은 2025년 11월 "tool-augmented agents의 budget-aware test-time scaling"이라는 표현을 만든다 — 제안된 논문은 근본적으로 다른 action semantics를 가진 multimodal counterpart로 자신을 positioning해야 한다(visual token은 web-search call이 아니며, cost structure가 다르고, image는 BATS가 모델링하지 않는 partial-observability dimension을 도입한다).

### NeurIPS와 EMNLP를 위한 framing

**NeurIPS 2026**(5월 deadline 가능성이 높음)의 경우, 승리하는 framing은 *방법론적이고 경험적인* 것이다: "cost–accuracy Pareto frontier에 대한 이론적 특성을 갖는 multimodal agent의 budget-conditional policy." RL formulation, Pareto curve, action subset ablation을 강조한다. 예상되는 reviewer concern은 다음과 같다. (i) VisionThink/DeepEyes/Pixel Reasoner 대비 novelty — budget-conditional axis와 heterogeneous action space로 방어해야 한다. (ii) scale — reviewer는 7B+ model과 multiple benchmark를 원할 것이다. (iii) baseline — L1과 BATS는 text-only 또는 simulated baseline 형태로라도 비교되어야 한다.

**EMNLP 2026**(6월 deadline)의 경우, 더 강한 framing은 *calibration 및 abstention 중심*이다: "When should a vision-language agent stop, look again, or say I don't know?" — information-seeking literature(MediQ, CLAM, Self-Ask), abstention benchmark(MM-UPD, VQAv2-IDK, MM-AQA), R-Tuning-style training에 기대는 것이 좋다. EMNLP는 NeurIPS보다 linguistic-abstention framing에 더 가치를 둘 것이다. 반대로 RL machinery는 덜 인정받을 수 있다. Reviewer concern은 다음에 집중될 것이다. (i) contribution의 multimodal-ness(EMNLP reviewer가 "CV"라고 반발할 수 있음), (ii) abstention contribution이 MM-UPD/MM-AQA와 구분되는지.

**추천: NeurIPS primary.** Contribution은 지배적으로 methodological(RL, budget conditioning, Pareto analysis)이고 visual action space는 EMNLP의 일반적 scope보다 풍부하다. NeurIPS가 reject할 경우, EMNLP를 fallback으로 두고 abstention-forward narrative로 reframing한다.

---

## Part 3 — 구체적인 실험 계획 (2–3주, 1–8 H200)

### Base VLM 선택

**Qwen2.5-VL-7B-Instruct**가 올바른 backbone이다. 다섯 가지 이유가 수렴한다. (1) 2D-RoPE를 통한 native dynamic resolution 덕분에 token count가 input image size에 자연스럽게 scale한다 — 이는 모든 "더 많은 visual token 요청" action에 필수적이다. (2) 강한 native grounding ability가 있으며, DeepEyes가 보여주었듯 이는 external detector 없이 zoom tool을 사용하기 위해 중요하다. (3) community-standard RL substrate이다 — VisionThink, Chain-of-Focus, Active-O3, Reinforcing-VLMs-Under-Resource-Constraints 모두 Qwen2.5-VL을 사용하므로 apples-to-apples comparison이 쉽다. (4) LoRA + GRPO로 single H200(141 GB)에 여유롭게 들어간다. 2B와 3B variant는 ablation-scale fallback을 제공한다. (5) robust tool-calling schema가 내장되어 있다.

Rejected alternatives: **InternVL 2.5-8B**는 성능은 유사하지만 RL ecosystem이 덜 성숙했고 tile-based image encoding이 더 무거워 token accounting을 복잡하게 만든다. **LLaVA-OneVision-7B**는 native grounding이 부족하여 external detector에 의존하게 만든다. Compute가 빠듯하다면 **Qwen2.5-VL-3B**로 fallback한다. 8대 H200을 전체 기간 사용할 수 있다면, 3주차 final "scale experiment"로 Qwen2.5-VL-32B도 가능하다.

### 데이터셋

Training mixture(약 30–60k examples; format-unified tool-use JSONL로 결합):
- **V\*Bench training split** (from 2312.14135)과 **Visual CoT 438k subset** (2403.16999) — supervised zoom/bbox action.
- **HR-Bench train partition** (from DC², 2408.15556) — high-resolution spend action.
- **VQAv2-IDK + UNK-VQA train** — supervised abstention label.
- VQAv2/GQA에 4×4-grid masking diagnostic을 적용해 programmatically 생성한 **"sufficiency-labeled"** subset: 각 (image, question)에 대해 1/16, 4/16, 8/16, 16/16 unmask ratio variant를 생성한다. strong teacher VLM(GPT-4o 또는 Qwen2.5-VL-72B)의 majority-vote judgment로 각 variant를 correct/hallucinate/abstain으로 labeling한다. 이것이 연구자가 제안하는 "optimal action" supervision을 제공한다.
- Abstention calibration을 위한 **MM-UPD training-safe subset**과 **IDK-Instructions** (2402.09717).

Evaluation(primary reporting surface):
- **V\*Bench** (high-res detail detection) — zoom action을 위한 primary task.
- **HR-Bench 4K/8K** — budget-sensitive high-res.
- **MM-UPD** (AAD + IASD + IVQD subset) — primary abstention evaluation.
- **POPE** (random / popular / adversarial) — hallucination regression check.
- **MM-AQA** (2604.14799) "Missing Visual Information" slice — accessible하다면 masking-diagnostic premise에 대한 직접 평가.
- **MMBench**와 **MMMU** — generous budget에서 normal query에 대한 non-regression sanity check.
- Optional stretch: **ActiView** (2410.04659) — integration effort가 맞으면 수행.

### Action space와 token accounting

XML-tagged 방식(DeepEyes/VisionThink와 유사)으로 emit되는 six-action vocabulary:

| Action | Cost (input tokens) | Cost (output tokens) | Semantics |
|---|---|---|---|
| `ANSWER(x)` | 0 | ≤ 64 | answer x로 trajectory 종료 |
| `ABSTAIN("insufficient: r")` | 0 | ≤ 32 | refusal + reason r로 종료 |
| `THINK(text)` | 0 | ≤ configurable | CoT 확장; 각 call bounded |
| `REQUEST_HI_RES()` | +N_hires visual tokens (Qwen2.5-VL dynamic-res formula로 계산) | 0 | ¼-res를 full-res image로 대체 |
| `ZOOM(bbox)` | +N_zoom visual tokens (crop size에 따라 보통 256–729) | 0 | cropped region을 새 visual segment로 추가 |
| `RETRIEVE(q)` | +K retrieved text tokens (K=500 cap) | 0 | Optional; stretch goal |

**Budget B**는 token 단위의 single scalar로, system prompt에 `Budget: B tokens remaining.` 형태로 삽입되고 Anthropic의 `task_budget` beta를 mirror하듯 각 action 이후 running countdown으로 업데이트된다. Pareto curve를 sweep하기 위해 B ∈ {512, 1024, 2048, 4096, 8192}에서 시작한다. Cost는 action의 deterministic function이므로 accounting은 trivial하다.

### Training approach

**Stage 1 (Days 1–4) — Head-only probe + SFT warm start.** 마지막 8개 transformer block의 attention + MLP 및 LM head에 LoRA(r=16, α=32)를 적용한다. 두 채널을 통해 합성된 약 15k tool-use trajectory로 SFT한다. (a) teacher-forced — GPT-4o 또는 Qwen2.5-VL-72B가 training query를 풀면서 action vocabulary에서 tool call을 emit하게 하고, final-answer correctness로 filter한다. (b) rule-based — V\*Bench와 Visual CoT data에 대해 ground-truth bbox를 사용하여 `ZOOM` action을 합성한다. Batch 32로 약 2 epoch, 4×H200에서 약 8시간 학습한다.

**Stage 2 (Days 5–12) — Budget-conditional GRPO.** Stage-1 policy에서 initialize한다. 위 budget schedule에서 draw된 budget을 사용해 group size G=8로 GRPO를 수행한다. Reward:
$$R = \mathbb{1}[\text{correct}] \cdot r_{\text{acc}} - \lambda_{\text{cost}} \cdot \max(0, \text{cost} - B) - \lambda_{\text{abstain}} \cdot \mathbb{1}[\text{wrongly abstained}] + \lambda_{\text{cal}} \cdot \mathbb{1}[\text{correctly abstained when insufficient}]$$
λ 값들은 500-example val set에서 tune한다. Budget-overflow penalty는 policy가 B를 준수하게 만들고, abstention term은 calibration을 밀어준다. Answer가 자동 채점되지 않는 경우 reasoning correctness judge로 Qwen2.5-VL-72B를 사용한다. 이는 VisionThink setup과 유사하지만 state에 budget이 있고 action space에 abstention이 포함된다. Qwen2.5-VL-7B에 대해 약 30k sample에서 4–8×H200 기준 약 48–72 GPU-hour가 예상된다.

**Stage 3 (Days 13–18) — Ablations and final runs.** 세 가지 핵심 ablation을 학습한다. (i) no-budget-conditioning(input의 B를 shuffle); (ii) no-abstention action; (iii) SFT-only, no RL. 각각 4×H200에서 ≤12 hrs. Final Pareto eval은 B∈{256,512,1024,2048,4096,8192,∞}에서 실행한다.

**Fallback narrower scopes** (Day 10까지 RL이 converge하지 않을 경우): (a) RL을 완전히 건너뛰고 SFT-only budget-conditional variant를 "compute-conditioned instruction tuning"으로 framing하여 제출한다. (b) retrieval action을 제거하고 visual-only task에서만 평가한다. (c) Stage-1 → Stage-2 pipeline을 budget level별 accept/reject pair에 대한 pure DPO로 대체한다 — 훨씬 저렴하고 안정적이며 "TALE-for-VLMs" analog로 출판 가능하다.

### Baselines

다섯 가지 baseline이 명백한 비교 축을 포괄한다. **(1)** 각 budget에서 s1 방식의 budget forcing을 적용한 Qwen2.5-VL-7B vanilla(prepend budget to prompt, overflow 시 truncate). **(2)** VisionThink-style single binary escalation(가장 가까운 competitor이므로 re-implement). **(3)** ZoomEye training-free tree search. **(4)** vanilla Qwen과 full-tool Qwen 사이를 cascade하는 Adaptive-RAG-style difficulty classifier. **(5)** 모든 query에서 maximum budget을 쓰는 oracle upper bound.

### Metrics

필수 reporting surface는 네 가지다. **(1) Pareto accuracy-under-budget curves** — B sweep에 따라 task accuracy vs. average tokens consumed를 plot한다. Area under the Pareto curve(AUPC)가 headline scalar다. **(2) Effective Reliability Φ** (Whitehead et al., ECCV 2022)와 risk-coverage AURC — 각 budget에서 abstention quality를 정량화하는 standard selective-prediction metric. **(3) Abstention F1** on MM-UPD splits and the masked-image diagnostic. **(4) Hallucination rate** via POPE adversarial and CHAIR_i on descriptive outputs. Secondary: **cost-of-pass** (Efficient Agents, 2508.02694), **average tokens-to-answer**, 그리고 **budget-compliance rate**(actual cost ≤ B인 trajectory 비율).

### Week-by-week milestones

| Week | Focus | Deliverables |
|---|---|---|
| **Week 1 (Days 1–7)** | Data pipeline + Stage-1 SFT + baselines | Action-space JSONL training set (15k examples); masking-diagnostic data (5k examples with sufficiency labels); Qwen2.5-VL-7B의 SFT checkpoint; baselines (1)+(2)+(3) 재현 가능하게 실행; 초기 V\*Bench와 POPE 결과. |
| **Week 2 (Days 8–14)** | Stage-2 GRPO training + primary eval | GRPO-trained budget-conditional policy; V\*Bench, HR-Bench, MM-UPD, POPE에서 full Pareto curve; preliminary abstention-F1 결과. |
| **Week 3 (Days 15–21)** | Ablations, writing, buffer | 세 가지 ablation run; MMBench/MMMU non-regression check; figure가 포함된 첫 full paper draft. Benchmark가 예상 밖일 경우 rerun을 위한 buffer day. |

### Risk mitigation

GRPO instability가 가장 큰 risk다 — Pixel Reasoner의 two-phase SFT→RL recipe와 VisionThink의 call-ratio penalty를 따라 "always zoom" 또는 "never zoom" collapse를 방지한다. Day 10까지 GRPO가 diverge하면 rollout pair에 대한 DPO로 fallback한다(훨씬 안정적). 두 번째로 큰 risk는 **evaluation saturation**이다 — V\*Bench accuracy는 강한 7B 모델에서 85%에 가까워 Pareto headroom이 적다. HR-Bench 4K/8K가 차별화할 여지가 더 크다. 세 번째 risk는 **judge contamination**이다 — policy와 judge 모두 Qwen2.5-VL을 쓰는 것을 피한다. GPT-4o 또는 held-out InternVL 2.5-38B judge를 사용한다.

---

## Part 4 — Paper framing

### 세 가지 후보 제목

1. **"Budget-conditional policies for agentic vision-language models"** — methodological하고 NeurIPS 스타일이며, π(a|s,B) novelty를 강조한다.
2. **"Learning to look, think, or abstain under a token budget"** — narrative 스타일이며, heterogeneous action space와 abstention contribution을 포착한다.
3. **"Frugal eyes: pareto-optimal visual information acquisition for VLM agents"** — 기억하기 쉽고, Pareto frontier를 강조하며 FrugalGPT lineage와 정렬된다.

Preference: **NeurIPS에는 title 1, EMNLP에는 title 2.**

### Narrative arc

Intro는 inference-economics observation으로 시작해야 한다. Agentic VLM system(Operator, Claude agents, UI-TARS)은 이제 task당 수십 번 VLM을 호출하며, inference cost가 지배적인 system constraint가 되었다. Motivating anecdote는 frontier VLM이 image가 인식 불가능할 정도로 mask되어도 거의 기권하지 않고(MM-AQA의 26.2% UAC finding 인용), 이미지가 ¼ resolution에서도 trivially answerable한 경우에도 거의 항상 maximum token budget을 소비한다는 것이다(VisionThink의 motivation 인용). Gap은 세 가지로 framing한다 — 기존 efficient-VLM work(VisionThink, DeepEyes, Pixel Reasoner)는 single efficient operating point를 학습할 뿐 budget-conditional policy가 아니다. Budget-aware LLM work(L1, TALE, BATS)는 visual action이나 partial observability를 다루지 않는다. Abstention work(R-Tuning, MM-UPD, VQAv2-IDK)는 abstention을 budget-consuming action이 아니라 binary로 취급한다. Method section은 six-action vocabulary에 대한 budget-conditional policy π(a|s,B), two-stage SFT→GRPO training, 내부 signal generator로서의 masking-diagnostic을 소개한다. Experiments는 Pareto curve, ablation, abstention/calibration number를 제시한다. Conclusion은 Pareto-conditional framing이 VLM을 넘어 모든 budget-constrained agentic system으로 일반화된다고 주장한다.

### Venue selection

**Primary: NeurIPS 2026.** 세 가지 방식으로 적합하다. Contribution은 methodological(RL + policy learning), empirical(Pareto curves), architecturally multimodal하다. NeurIPS reviewer는 VLM-RL paper를 꾸준히 accept해 왔다(VisionThink, Pixel Reasoner, Mulberry 모두 NeurIPS 2025). 예상되는 reviewer concern: VisionThink/DeepEyes 대비 novelty(budget-conditioning으로 방어), scale question(7B + 3B ablation으로 방어), GRPO의 reward-hacking(λ에 대한 budget-compliance metric과 explicit ablation으로 방어).

**Secondary: EMNLP 2026.** Abstention과 information-seeking 중심으로 reframing한다: "when VLMs should say I don't know"로 시작하고 abstention action을 중심에 둔다. Risk: reviewer가 이 작업을 너무 CV라고 볼 수 있다. Strength: multimodal LM의 calibration이라는, EMNLP가 명시적으로 환영하는 topic에 대한 더 깔끔한 narrative thread.

**Not recommended:** CVPR 2026(RL-policy framing이 덜 자연스럽다); ICLR 2026 deadline은 이미 지났을 가능성이 높다; ACL 2026은 EMNLP가 reject할 경우 backup이다.

## Conclusion

이 분야는 충분히 혼잡해서 단순한 "VLM에게 RL로 zoom을 가르친다" 논문은 novelty 측면에서 reject될 것이다. 그러나 세 가지 깔끔하게 분리되는 gap은 여전히 열려 있다. **budget-conditioning**(policy가 B를 input으로 받고 Pareto를 추적 — multimodal에서는 아직 없음), **heterogeneous actions**(abstain + route + retrieve + zoom + CoT 통합 — 역시 아직 없음), 그리고 **budgeted action으로서의 information-theoretic abstention**(거의 미개척). 연구자의 직감은 맞다 — masking diagnostic은 headline이 아니라 training signal로 사용하는 것이 가장 좋다 — 왜냐하면 MM-AQA와 Reading-Between-the-Lines가 diagnostic-contribution의 문을 대체로 닫았기 때문이다. 명시적인 budget-overflow penalty를 갖는 six-action vocabulary 위에서 Qwen2.5-VL-7B를 two-stage SFT-then-GRPO로 학습하는 것은 4×H200에서 3주 이내에 가능하며, 모든 12개 competitor paper와 동시에 구분되는 Pareto-curve headline figure를 만들어낼 수 있다. NeurIPS가 더 나은 venue이고, EMNLP는 abstention reframing을 통해 일관된 fallback으로 남는다. 가장 중요하게는: acceptance를 가장 크게 예측하는 것은 reviewer가 1페이지에서 **어떤 경쟁 논문도 만들 수 없는** 단일 Pareto curve를 볼 수 있는지 여부다.
