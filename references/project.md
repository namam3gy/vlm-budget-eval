# Budget-conditional agentic VLMs: a crowded but cleavable frontier

**The bottom line.** The proposed idea — giving a VLM a token/compute budget and training it to spend that budget across zoom, reasoning, retrieval, and abstention — is **partially scooped but still publishable** if reframed. Between April 2025 and April 2026, at least a dozen papers have taught VLMs to zoom or call visual tools via RL (VisionThink, DeepEyes, Pixel Reasoner, Chain-of-Focus, Pixel Reasoner, Active-O3, Reinforcing-VLMs-Under-Resource-Constraints). Separately, the text-LLM community has solved budget-conditional policies (L1/LCPO, Elastic Reasoning, TALE), and the agent community has introduced budget-aware test-time scaling for web agents (BATS, Nov 2025). **The clean white space is the intersection the researcher is pointing at:** a VLM policy that (a) takes an explicit token/dollar budget **B as input**, (b) allocates it across a **heterogeneous action space** including abstention, and (c) traces the full cost–accuracy **Pareto frontier** rather than a single efficient operating point. No single published paper does all three for multimodal agents. A 2–3 week execution on 1–8 H200s can produce a credible NeurIPS or EMNLP submission if scoped tightly around that framing.

---

## Part 1 — Literature review

### Thread A: Adaptive and dynamic visual tokens in VLMs

The token-compression literature bifurcates into **static compression** (fixed ratios) and **adaptive acquisition** (learned policies over where to spend visual tokens). The static branch — **FastV** (Chen et al., ECCV 2024, 2403.06764), **VisionZip** (Yang et al., CVPR 2025, 2412.04467), **PyramidDrop** (Xing et al., CVPR 2025, 2410.17247), **LLaVA-PruMerge** (Shang et al., 2403.15388), **SparseVLM** (Zhang et al., ICML 2025, 2410.04417), **TokenPacker** (Li et al., IJCV 2025, 2407.02392), **VoCo-LLaMA** (Ye et al., CVPR 2025, 2406.12275), and **FasterVLM** (Zhang et al., 2412.01818) — all prune visual tokens by fixed fractions or attention heuristics. They are peripheral to the proposal: they shrink a fixed budget but never decide **how much** to spend.

Two static papers are architectural enablers worth citing as *substrate*. **Matryoshka Multimodal Models** (Cai et al., ICLR 2025, 2405.17430) trains nested visual-token granularities so that a downstream policy can pick resolution per instance — a near-ideal action-space substrate. Dynamic resolution in **Qwen2-VL / Qwen2.5-VL** (Wang et al., 2409.12191 / 2502.13923), **LLaVA-NeXT AnyRes**, **InternVL 1.5/2.5** (Chen et al., 2404.16821 / 2412.05271), **Monkey** (Li et al., CVPR 2024), **LLaVA-UHD** (Guo et al., ECCV 2024, 2403.11703), and **mPLUG-DocOwl 1.5** (Hu et al., EMNLP Findings 2024) provide variable-token-count backbones required for any budget-aware method.

The adaptive-acquisition branch is where the proposal actually competes. **V\*** (Wu & Xie, CVPR 2024, 2312.14135) introduced the SEAL meta-architecture in which the VQA LLM invokes visual search when evidence is insufficient. **Visual CoT** (Shao et al., NeurIPS 2024, 2403.16999) supervised bbox-zoom with 438k QA pairs. **Chain-of-Spot** (Liu et al., 2403.12966) trains a two-turn ROI policy. **CogCoM** (Qi et al., 2402.04236) trains a full chain-of-manipulations action set (zoom, crop, ground, count). **Visual Sketchpad** (Hu et al., NeurIPS 2024, 2406.09403) gives the model drawing tools as intermediate reasoning. Training-free zoom includes **ZoomEye** (Shen et al., EMNLP 2025, 2411.16044), **ViCrop** (Zhang et al., ICLR 2025, 2502.17422), and **DC²** (Wang et al., AAAI 2025, 2408.15556).

**The most directly related work — which must be treated as competitor papers — are the 2025 RL-trained visual-agent policies.** **VisionThink** (Yang et al., NeurIPS 2025, 2507.13348) starts at ¼ resolution and RL-trains a single tool call that requests full resolution, with a penalty on resize-call ratio. **DeepEyes** (Zheng et al., 2505.14362) does end-to-end GRPO with an `image_zoom_in_tool` and no cold-start SFT. **Pixel Reasoner** (Su et al., NeurIPS 2025, 2505.15966) trains `zoom-in` and `select-frame` with a curiosity-driven reward plus efficiency penalty. **Chain-of-Focus** (2505.15436) adds adaptive sufficient-evidence detection. **Active-O3** (Zhu et al., 2505.21457) formalizes active perception in MLLMs via GRPO. **Reinforcing VLMs to Use Tools under Resource Constraints** (2506.14821) is the single most title-overlapping prior work, training Qwen2.5-VL-3B with GRPO under a training-time tool-call limit.

### Thread B: Test-time compute and inference budget optimization

The text-LLM community has a mature toolbox that does **not yet have a multimodal equivalent**. Foundations include OpenAI o1 (2412.16720), **DeepSeek-R1** (Guo et al., Nature 2025, 2501.12948), **Snell et al.'s "compute-optimal test-time scaling"** (2408.03314), **Large Language Monkeys** (Brown et al., 2407.21787), and **s1 / budget forcing** (Muennighoff et al., EMNLP 2025, 2501.19393). Budget-aware reasoning per se is anchored by **TALE** (Han et al., ACL Findings 2025, 2412.18547) which pioneers prompting+SFT+DPO for token-budget adherence, **L1/LCPO** (Aggarwal & Welleck, CMU 2025, 2503.04697) which RL-trains models that honor a length specified in the prompt, **Elastic Reasoning** (Xu et al., Salesforce 2025, 2505.05315) which uses budget-constrained GRPO rollouts over separated thinking/solution phases, **AdaCoT** (Lou et al., 2505.11896) which triggers CoT only on complex queries, and **Chain of Draft** (Xu et al., Zoom, 2502.18600) for aggressive per-step compression. The canonical Sui et al. **"Stop Overthinking" survey** (TMLR 2025, 2503.16419) and Feng et al.'s multimodal-aware survey (2503.21614) confirm the multimodal gap explicitly.

Adaptive computation traces to Graves's **ACT** (1603.08983), **PonderNet** (Banino et al., 2107.05407), and **Mixture-of-Depths** (Raposo et al., 2404.02258) — a natural template for a vision-token-level MoD. Cascades and routing are dominated by **FrugalGPT** (Chen, Zaharia, Zou, TMLR 2024, 2305.05176), **RouteLLM** (Ong et al., ICLR 2025, 2406.18665), **Hybrid LLM** (Ding et al., ICLR 2024, 2404.14618), and **AutoMix** (Madaan et al., NeurIPS 2024, 2310.12963).

**Crucial finding: none of the 2024–2025 multimodal-reasoning papers** — LLaVA-CoT/LLaVA-o1 (Xu et al., ICCV 2025, 2411.10440), Mulberry (Yao et al., NeurIPS 2025, 2412.18319), Insight-V (Dong et al., CVPR 2025, 2411.14432), Virgo (Du et al., 2501.01904), Vision-R1 (Huang, 2503.06749), MM-Eureka (Meng et al., 2503.07365) — **impose an explicit inference-time budget that the model learns to honor.** They elicit long CoT; they do not constrain it. This is the single clearest opening.

### Thread C: Agentic cost efficiency and tool-use optimization

Benchmarks relevant for budget-aware evaluation include **GAIA** (Mialon et al., 2311.12983) with first-class token accounting via the Princeton HAL leaderboard, **OSWorld** (Xie et al., NeurIPS 2024, 2404.07972) plus **OSWorld-Human** (2506.16042) which adds step/latency breakdowns, **SWE-agent** (Yang et al., 2405.15793) which already imposes a $4-per-instance budget, **VisualWebArena** (Koh et al., ACL 2024, 2401.13649), and **BrowseComp** (Wei et al., OpenAI 2025, 2504.12516) with published test-time compute curves. **WAREX** (2510.03285) and **WABER** retrofit cost logging onto WebArena. For a 2–3 week timeline, these are over-scoped; simpler academic benchmarks (V\*Bench, HR-Bench, MM-UPD) are more realistic.

Adaptive-retrieval literature is the cleanest methodological precedent. **Toolformer** (Schick et al., NeurIPS 2023, 2302.04761), **Adaptive-RAG** (Jeong et al., NAACL 2024, 2403.14403), **Self-RAG** (Asai et al., ICLR 2024 Oral, 2310.11511), **FLARE** (Jiang et al., EMNLP 2023, 2305.06983), **When to Retrieve / Adapt-LLM** (Labruna et al., 2404.19705), and especially **SmartRAG** (Gao et al., ICLR 2025, 2410.18141) — which RL-trains a joint policy over *when* to retrieve with an explicit cost-minimizing reward — map directly onto the proposed action space.

Explicitly budget-constrained agents are still rare. **BudgetMLAgent** (Gandhi et al., 2411.07464) uses hard GPT-4-call caps with cheap-to-expensive cascades. **Efficient Agents / cost-of-pass** (Wang et al., 2508.02694) retains 96.7% of OWL performance at 28.4% lower cost-of-pass on GAIA and gives the proposal its preferred metric vocabulary. **Scaling Test-time Compute for LLM Agents** (Zhu et al., 2506.12928) studies parallel vs. sequential scaling. **BATS / Budget Tracker** (2511.17006, Nov 2025) is the first systematic study of budget-constrained tool-augmented agents and introduces the unified token+tool-call cost metric — **it is the chief framing competitor** but is LLM-only (no visual actions). Visual/GUI agents — **CogAgent** (Hong et al., CVPR 2024 Highlight, 2312.08914), **ShowUI** (Lin et al., CVPR 2025, 2411.17465) with its UI-guided visual-token pruning, **UGround** (Gou et al., ICLR 2025 Oral, 2410.05243), **OS-Atlas** (Wu et al., ICLR 2025, 2410.23218), **Aguvis** (Xu et al., ICML 2025, 2412.04454), **UI-TARS** (2501.12326) and **UI-TARS-2** (2509.02544) — are potential downstream testbeds but their budget-aware training is unexplored.

### Thread D: Abstention, calibration and "I don't know" in VLMs

Hallucination benchmarks are well-established: **POPE** (Li et al., EMNLP 2023, 2305.10355), **MME** (Fu et al., 2306.13394), **HallusionBench** (Guan et al., CVPR 2024, 2310.14566), **MMHal-Bench** (Sun et al., 2309.14525), **AMBER** (Wang et al., 2311.07397), **CHAIR** (Rohrbach et al., EMNLP 2018), and **MMVP / Eyes Wide Shut** (Tong et al., CVPR 2024, 2401.06209). These measure hallucination post-hoc; they do not train abstention.

Training-for-abstention draws on **R-Tuning** (Zhang et al., NAACL 2024 Outstanding, 2311.09677) which constructs knowledge-intersection SFT data and is the most transferable recipe, alongside Lin et al.'s **"Teaching Models to Express Uncertainty in Words"** (TMLR 2022, 2205.14334), Kamath et al.'s **selective QA under domain shift** (ACL 2020), **SelfCheckGPT** (Manakul et al., EMNLP 2023, 2303.08896), Xiong et al.'s ICLR 2024 confidence-elicitation study (2306.13063) showing LLMs are persistently overconfident, Tian et al.'s **Just Ask for Calibration** (EMNLP 2023, 2305.14975), and Band et al.'s **Linguistic Calibration** (2404.00474).

VLM-specific abstention is the closest-adjacent sub-field. **Miyai et al.'s Unsolvable Problem Detection / MM-UPD** (ECCV 2024, 2403.20331) is the gold-standard benchmark with three subtypes (AAD/IASD/IVQD). **Visually Dehallucinative Instruction Generation / IDK-Instructions** (Cha et al., 2402.09717) provides the multimodal analog of R-Tuning plus the VQAv2-IDK evaluation set. **UNK-VQA** (Guo et al., TPAMI 2024, 2310.10942) deliberately perturbs images or questions — the closest existing mirror of the proposed masking diagnostic. **VL-Uncertainty** (Zhang et al., 2411.11919), **AvisC** (Woo et al., ACL Findings 2025, 2405.17820), and **Reliable VQA** (Whitehead et al., ECCV 2022, 2204.13631) with the Effective Reliability Φ metric cover calibration. Two 2026 papers merit close attention: **"Reading Between the Lines"** (2511.19806) uses visual occlusion with white boxes and Gaussian blur as a stress test and trains **latent-representation probes** that generalize across corruption types; **MM-AQA** (2604.14799) explicitly constructs "Missing Visual Information" and "Occlusion Ambiguity" categories by partial masking and finds it is the hardest abstention regime (26.2% unanswerable accuracy coverage) — **these two papers significantly narrow the novelty of the proposed masking diagnostic and must be cited**.

Information-seeking QA provides another template: **CLAM** (Kuhn et al., 2212.07769) for clarifying-question generation, **Self-Ask** (Press et al., EMNLP Findings 2023, 2210.03350), and crucially **MediQ** (Li et al., NeurIPS 2024, 2406.00922) which converts static QA into interactive benchmarks where abstention and follow-up questions are first-class actions. **AbstentionBench** (Kirichenko et al., 2506.09038) reports that reasoning fine-tuning *degrades* abstention — a sharp motivating finding for training-time abstention-aware objectives.

### Thread E: Active perception and partial observation

The lineage starts with **Recurrent Models of Visual Attention / RAM** (Mnih et al., NeurIPS 2014, 1406.6247), **Ba et al.'s multi-object attention** (ICLR 2015, 1412.7755), and **DRAW** (Gregor et al., ICML 2015, 1502.04623) — all RL or differentiable glimpse policies. The cognitive-science anchor is **Ullman et al.'s "Atoms of Recognition" / MIRCs** (PNAS 2016, 113(10):2744–2749), which identifies minimal image configurations required for human recognition — the exact epistemic question the proposed masking diagnostic asks. Active VQA goes back to Shih et al.'s **Where To Look** (CVPR 2016, 1511.07394) with modern embodied versions in **Active Neural SLAM** (Chaplot et al., ICLR 2020, 2004.05155) and very recent **VG-AVS / Toward Ambulatory Vision** (2512.13250). The most important recent entry is **ActiView** (Wang et al., ACL 2025, 2410.04659) — an MLLM benchmark that restricts perceptual field and requires active zoom/view-shift, evaluating 27–30 models and revealing a large active-perception gap. **SalBench** (2507.04741) probes low-level saliency; even GPT-4o achieves only 47.6%.

### Thread F: Closest prior work and scooping risks

The 12 papers below are ranked by scooping risk. **Must-cite competitors** are marked ⚠️.

| Rank | Paper | arXiv | Scoop risk | Residual white space |
|---|---|---|---|---|
| 1 | VisionThink (NeurIPS 2025) ⚠️ | 2507.13348 | **HIGH** | No budget conditioning; binary action only; no abstention |
| 2 | Reinforcing VLMs Under Resource Constraints ⚠️ | 2506.14821 | **HIGH** | Training-time constraint, not input-conditioned budget; zoom-only |
| 3 | Chain-of-Focus / Adaptive-CoF ⚠️ | 2505.15436 | **HIGH** | No explicit budget; no non-visual tools; no abstention |
| 4 | DeepEyes ⚠️ | 2505.14362 | **HIGH** | No budget term in reward; no Pareto; single tool |
| 5 | Pixel Reasoner (NeurIPS 2025) ⚠️ | 2505.15966 | **HIGH** | Curiosity reward, not budget; no abstention; no cost-Pareto |
| 6 | BATS / Budget Tracker (Nov 2025) ⚠️ | 2511.17006 | **MED-HIGH** | LLM web-search only; no visual actions |
| 7 | VRAG-RL (NeurIPS 2025) | 2505.22019 | MEDIUM | No explicit budget; RAG-focused |
| 8 | Active-O3 | 2505.21457 | MEDIUM | No cost constraint; localization-centric |
| 9 | AVR / Adaptive VLM Routing | 2603.12823 | MEDIUM | Model-selection routing only, no action policy |
| 10 | AwaRes | 2603.16932 | MEDIUM | HR-crop retrieval only; no heterogeneous actions |
| 11 | VTool-R1 | 2505.19255 | MEDIUM | Tool-use RL; no budget |
| 12 | OpenThinkIMG / V-ToolRL | 2505.08617 | MEDIUM | Infrastructure + RL; no budget |

Supporting cast with lower risk: Visual Sketchpad, ZoomEye, T3-Agent (ICLR 2025 Spotlight, 2412.15606), CropVLM (2511.19820), VLM-R³, Argus, GRIT, "Don't Look Only Once" (2505.18842), ReCoVERR (2402.15610).

---

## Part 2 — Gap analysis

### What is genuinely novel

Three interlocking contributions remain defensible if combined. **First, a budget-conditional VLM policy π(a|s,B)** that takes an explicit token or dollar budget B as part of the input state and traces out the full cost–accuracy Pareto curve. L1/LCPO does this for text-only CoT length, Elastic Reasoning does it for partitioned phases, but **no multimodal agent paper takes B as input to the policy** — VisionThink, DeepEyes, Pixel Reasoner, Chain-of-Focus, and Reinforcing-VLMs-Under-Resource-Constraints all learn a single "efficient" operating point. **Second, a heterogeneous action space that unifies visual acquisition, reasoning, and abstention.** Existing VLM-RL work exposes one or two actions (zoom, select-frame). Unifying `answer / abstain / extend-CoT / request-high-res / zoom(bbox) / retrieve` under a single budget-aware policy is unpublished. **Third, an abstention action that is explicitly budgeted.** Abstention exists in LLMs (R-Tuning) and in VLM benchmarks (MM-UPD, VQAv2-IDK, MM-AQA) but is never treated as a budget-consuming action that trades off against tool calls.

A fourth, more optional contribution is **framing the masking diagnostic within an information-theoretic or minimal-image framework** (Ullman et al.), which is rare in this crowd of engineering-motivated papers. However, MM-AQA (2604.14799) and Reading-Between-the-Lines (2511.19806) have both performed systematic VLM masking tests; the diagnostic itself is no longer novel and must be framed as a **signal internal to the budget-optimization method**, not as the contribution, exactly as the researcher already intends.

### Most serious scooping risks

The overlap is real. **VisionThink and DeepEyes** establish the RL-for-visual-tools paradigm; **Pixel Reasoner** provides the pixel-space action-set template; **Chain-of-Focus** claims the "adaptive sufficiency" framing; **Reinforcing-VLMs-Under-Resource-Constraints** (2506.14821) has the closest title and explicit "resource constraints" framing, though it caps tool calls at train time rather than conditioning the policy on a test-time budget. The highest-level framing risk is **BATS (2511.17006)** which coined "budget-aware test-time scaling of tool-augmented agents" in November 2025 — the proposed paper must position itself as the multimodal counterpart with fundamentally different action semantics (visual tokens are not web-search calls; the cost structure is different; images introduce a partial-observability dimension BATS does not model).

### Framings for NeurIPS vs EMNLP

For **NeurIPS 2026** (May deadline likely), the winning framing is *methodological and empirical*: "budget-conditional policies for multimodal agents with theoretical characterization of the cost–accuracy Pareto frontier." Emphasize RL formulation, Pareto curves, and ablations over action subsets. Anticipated reviewer concerns: (i) novelty vs. VisionThink/DeepEyes/Pixel Reasoner — must be defended by the budget-conditional axis plus heterogeneous action space; (ii) scale — reviewers will want 7B+ models and multiple benchmarks; (iii) baselines — L1 and BATS must be compared, if only via text-only or simulated baselines.

For **EMNLP 2026** (June deadline), the stronger framing is *calibration- and abstention-centric*: "When should a vision-language agent stop, look again, or say I don't know?" — lean on the information-seeking literature (MediQ, CLAM, Self-Ask), abstention benchmarks (MM-UPD, VQAv2-IDK, MM-AQA), and R-Tuning-style training. EMNLP will value the linguistic-abstention framing more than NeurIPS; conversely the RL machinery will attract less credit there. Reviewer concerns would center on: (i) multimodal-ness of the contribution (EMNLP reviewers may push back that this is "CV"), (ii) whether the abstention contribution is distinct from MM-UPD/MM-AQA.

**Recommendation: NeurIPS primary.** The contribution is dominantly methodological (RL, budget conditioning, Pareto analysis) and the visual action space is richer than EMNLP's typical scope. Keep EMNLP as a fallback, with a reframed abstention-forward narrative if NeurIPS rejects.

---

## Part 3 — Concrete experiment plan (2–3 weeks, 1–8 H200s)

### Base VLM choice

**Qwen2.5-VL-7B-Instruct** is the correct backbone, for five converging reasons. (1) Native dynamic resolution via 2D-RoPE means token count scales naturally with input image size — essential for any "request more visual tokens" action. (2) Strong native grounding ability, which DeepEyes demonstrated is critical for zoom tools without needing an external detector. (3) It is the community-standard RL substrate — VisionThink, Chain-of-Focus, Active-O3, and Reinforcing-VLMs-Under-Resource-Constraints all use Qwen2.5-VL, so apples-to-apples comparisons are straightforward. (4) It fits comfortably on a single H200 (141 GB) with LoRA + GRPO; 2B and 3B variants offer ablation-scale fallbacks. (5) A robust tool-calling schema is built in.

Rejected alternatives: **InternVL 2.5-8B** has comparable capability but a less mature RL ecosystem and heavier tile-based image encoding that complicates token accounting. **LLaVA-OneVision-7B** lacks native grounding, forcing dependence on external detectors. If compute is tight, fall back to **Qwen2.5-VL-3B**; if 8 H200s are available for the whole window, Qwen2.5-VL-32B becomes feasible for a final "scale experiment" in week 3.

### Datasets

Training mixture (about 30–60k examples; combined via format-unified tool-use JSONL):
- **V\*Bench training split** (from 2312.14135) and **Visual CoT 438k subset** (2403.16999) — supervised zoom/bbox actions.
- **HR-Bench train partition** (from DC², 2408.15556) — high-resolution spend actions.
- **VQAv2-IDK + UNK-VQA train** — supervised abstention labels.
- A programmatically generated **"sufficiency-labeled"** subset by applying the 4×4-grid masking diagnostic to VQAv2/GQA: for each (image, question) generate variants at 1/16, 4/16, 8/16, 16/16 unmask ratios; label each with correct/hallucinate/abstain via majority-vote judgments from a strong teacher VLM (GPT-4o or Qwen2.5-VL-72B). This yields the "optimal action" supervision the researcher proposes.
- **MM-UPD training-safe subset** and **IDK-Instructions** (2402.09717) for abstention calibration.

Evaluation (primary reporting surface):
- **V\*Bench** (high-res detail detection) — primary task for zoom actions.
- **HR-Bench 4K/8K** — budget-sensitive high-res.
- **MM-UPD** (AAD + IASD + IVQD subsets) — primary abstention evaluation.
- **POPE** (random / popular / adversarial) — hallucination regression check.
- **MM-AQA** (2604.14799) "Missing Visual Information" slice — direct evaluation of the masking-diagnostic premise, *if accessible*.
- **MMBench** and **MMMU** — general-purpose sanity checks for non-regression on normal queries under generous budget.
- Optional stretch: **ActiView** (2410.04659) — if integration effort fits.

### Action space and token accounting

Six-action vocabulary, emitted as structured tool calls (XML-tagged à la DeepEyes/VisionThink):

| Action | Cost (input tokens) | Cost (output tokens) | Semantics |
|---|---|---|---|
| `ANSWER(x)` | 0 | ≤ 64 | Terminates trajectory with answer x |
| `ABSTAIN("insufficient: r")` | 0 | ≤ 32 | Terminates with refusal + reason r |
| `THINK(text)` | 0 | ≤ configurable | Extends CoT; each call bounded |
| `REQUEST_HI_RES()` | +N_hires visual tokens (computed from Qwen2.5-VL dynamic-res formula) | 0 | Replaces ¼-res with full-res image |
| `ZOOM(bbox)` | +N_zoom visual tokens (usually 256–729 depending on crop size) | 0 | Adds cropped region as new visual segment |
| `RETRIEVE(q)` | +K retrieved text tokens (K=500 cap) | 0 | Optional; stretch goal |

The **budget B** is a single scalar (in tokens) inserted into the system prompt as `Budget: B tokens remaining.` and updated after every action via a running countdown, mirroring Anthropic's `task_budget` beta. Start with B ∈ {512, 1024, 2048, 4096, 8192} to sweep the Pareto curve. Costs are deterministic functions of the action, so the accounting is trivial.

### Training approach

**Stage 1 (Days 1–4) — Head-only probe + SFT warm start.** LoRA (r=16, α=32) on attention + MLP of the last 8 transformer blocks plus the LM head. SFT on ~15k tool-use trajectories synthesized via two channels: (a) teacher-forced — have GPT-4o or Qwen2.5-VL-72B solve training queries while emitting tool calls from the action vocabulary, filter by final-answer correctness; (b) rule-based — for V\*Bench and Visual CoT data, use ground-truth bboxes to synthesize `ZOOM` actions. Train ~2 epochs at batch 32, ~8 hours on 4×H200.

**Stage 2 (Days 5–12) — Budget-conditional GRPO.** Initialize from Stage-1 policy. GRPO with group size G=8, sampling with budgets drawn from the schedule above. Reward:
$$R = \mathbb{1}[\text{correct}] \cdot r_{\text{acc}} - \lambda_{\text{cost}} \cdot \max(0, \text{cost} - B) - \lambda_{\text{abstain}} \cdot \mathbb{1}[\text{wrongly abstained}] + \lambda_{\text{cal}} \cdot \mathbb{1}[\text{correctly abstained when insufficient}]$$
with λ values tuned on a 500-example val set. The budget-overflow penalty makes the policy honor B; the abstention terms push calibration. Use Qwen2.5-VL-72B as a judge for reasoning correctness where answers are not automatable. This mirrors VisionThink's setup but with budget in the state and abstention in the action space. Estimated ~48–72 GPU-hours on 4–8×H200 for Qwen2.5-VL-7B at ~30k samples.

**Stage 3 (Days 13–18) — Ablations and final runs.** Train three key ablations: (i) no-budget-conditioning (shuffled B in input); (ii) no-abstention action; (iii) SFT-only, no RL. Each ≤12 hrs on 4×H200. Final Pareto eval runs at B∈{256,512,1024,2048,4096,8192,∞}.

**Fallback narrower scopes** (if RL does not converge by Day 10): (a) skip RL entirely and ship the SFT-only budget-conditional variant, framed as "compute-conditioned instruction tuning"; (b) drop the retrieval action and only evaluate on visual-only tasks; (c) drop Stage-1 → Stage-2 pipeline in favor of pure DPO on accept/reject pairs at each budget level — much cheaper and publishable as a "TALE-for-VLMs" analog.

### Baselines

Five baselines span the obvious comparisons: **(1)** Qwen2.5-VL-7B vanilla at each budget via budget forcing à la s1 (prepend budget to prompt, truncate on overflow); **(2)** VisionThink-style single binary escalation (re-implement since it is the closest competitor); **(3)** ZoomEye training-free tree search; **(4)** Adaptive-RAG-style difficulty classifier cascading between vanilla Qwen and full-tool Qwen; **(5)** an oracle upper bound that spends the maximum budget on every query.

### Metrics

Four required reporting surfaces: **(1) Pareto accuracy-under-budget curves** — plot task accuracy vs. average tokens consumed, swept across B; area under the Pareto curve (AUPC) is the headline scalar. **(2) Effective Reliability Φ** (Whitehead et al., ECCV 2022) and risk-coverage AURC — standard selective-prediction metrics quantifying abstention quality at each budget. **(3) Abstention F1** on MM-UPD splits and the masked-image diagnostic. **(4) Hallucination rate** via POPE adversarial and CHAIR_i on descriptive outputs. Secondary: **cost-of-pass** (Efficient Agents, 2508.02694), **average tokens-to-answer**, and a **budget-compliance rate** (% of trajectories where actual cost ≤ B).

### Week-by-week milestones

| Week | Focus | Deliverables |
|---|---|---|
| **Week 1 (Days 1–7)** | Data pipeline + Stage-1 SFT + baselines | Action-space JSONL training set (15k examples); masking-diagnostic data (5k examples with sufficiency labels); SFT checkpoint of Qwen2.5-VL-7B; baselines (1)+(2)+(3) reproducibly running; initial V\*Bench and POPE numbers. |
| **Week 2 (Days 8–14)** | Stage-2 GRPO training + primary eval | GRPO-trained budget-conditional policy; full Pareto curves on V\*Bench, HR-Bench, MM-UPD, POPE; preliminary abstention-F1 numbers. |
| **Week 3 (Days 15–21)** | Ablations, writing, buffer | Three ablation runs; MMBench/MMMU non-regression checks; first full paper draft with figures. Buffer days for reruns if a benchmark surprises. |

### Risk mitigation

GRPO instability is the single highest risk — follow Pixel Reasoner's two-phase SFT→RL recipe and VisionThink's call-ratio penalty to prevent collapse to "always zoom" or "never zoom". If GRPO diverges by Day 10, fall back to DPO over rollout pairs (much more stable). The second-highest risk is **evaluation saturation** — V\*Bench accuracy is close to 85% for strong 7B models, leaving little Pareto headroom; HR-Bench 4K/8K has more room to differentiate. The third risk is **judge contamination** — avoid using Qwen2.5-VL as both policy and judge; use GPT-4o or a held-out InternVL 2.5-38B judge.

---

## Part 4 — Paper framing

### Three candidate titles

1. **"Budget-conditional policies for agentic vision-language models"** — methodological, NeurIPS-style, emphasizes the π(a|s,B) novelty.
2. **"Learning to look, think, or abstain under a token budget"** — narrative-style, captures the heterogeneous action space and the abstention contribution.
3. **"Frugal eyes: pareto-optimal visual information acquisition for VLM agents"** — memorable, emphasizes the Pareto frontier and aligns with FrugalGPT lineage.

Preference: **title 1 for NeurIPS, title 2 for EMNLP.**

### Narrative arc

The intro should open with the inference-economics observation: agentic VLM systems (Operator, Claude agents, UI-TARS) now call VLMs dozens of times per task, and inference cost is the dominant system constraint. The motivating anecdote is that frontier VLMs almost never abstain even when the image is masked beyond recognition (cite MM-AQA's 26.2% UAC finding) and almost always consume the maximum token budget even when the image is trivially answerable at ¼ resolution (cite VisionThink's motivation). The gap is framed as threefold — existing efficient-VLM work (VisionThink, DeepEyes, Pixel Reasoner) learns a single efficient operating point, not a budget-conditional policy; budget-aware LLM work (L1, TALE, BATS) does not handle visual actions or partial observability; and abstention work (R-Tuning, MM-UPD, VQAv2-IDK) treats abstention as binary rather than as a budget-consuming action. The method section introduces the budget-conditional policy π(a|s,B) over the six-action vocabulary, the two-stage SFT→GRPO training, and the masking-diagnostic as an internal signal generator. The experiments deliver Pareto curves, ablations, and the abstention/calibration numbers. The conclusion argues the Pareto-conditional framing generalizes beyond VLMs to any budget-constrained agentic system.

### Venue selection

**Primary: NeurIPS 2026.** Fits in three ways: the contribution is methodological (RL + policy learning), empirical (Pareto curves), and architecturally multimodal. NeurIPS reviewers routinely accept VLM-RL papers (VisionThink, Pixel Reasoner, Mulberry were all NeurIPS 2025). Likely reviewer concerns: novelty vs. VisionThink/DeepEyes (defended by budget-conditioning), scale questions (defended by 7B + ablations down to 3B), and reward-hacking in GRPO (defended by budget-compliance metric and explicit ablations on λ).

**Secondary: EMNLP 2026.** Reframe around abstention and information-seeking: lead with "when VLMs should say I don't know" and center the abstention action. Risk: reviewers perceive the work as too CV. Strength: a cleaner narrative thread about calibration in multimodal LMs, a topic EMNLP explicitly welcomes.

**Not recommended:** CVPR 2026 (the RL-policy framing is less native); ICLR 2026 deadline has likely passed; ACL 2026 is a backup if EMNLP rejects.

## Conclusion

The field is crowded enough that a naive "teach a VLM to zoom with RL" paper would be rejected on novelty grounds. But three cleanly separable gaps remain open: **budget-conditioning** (policy takes B as input, traces Pareto — nobody has done this for multimodal), **heterogeneous actions** (abstain + route + retrieve + zoom + CoT unified — nobody has done this either), and **information-theoretic abstention as a budgeted action** (close to uncharted). The researcher's instincts are right — the masking diagnostic is best used as a training signal, not the headline — because MM-AQA and Reading-Between-the-Lines have largely closed the diagnostic-contribution door. A two-stage SFT-then-GRPO training on Qwen2.5-VL-7B over a six-action vocabulary with explicit budget-overflow penalty is feasible on 4×H200 in under three weeks, and produces the Pareto-curve headline figure that distinguishes the work from all twelve competitor papers simultaneously. NeurIPS is the better venue; EMNLP remains a coherent fallback via abstention reframing. Most critically: the single biggest predictor of acceptance will be whether reviewers see, on page one, a single Pareto curve that **no competitor paper can produce**.