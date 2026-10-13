# AHMAD-1 — RESEARCH LITERATURE REVIEW
**Deliverable §5.3 (+ §2.2, §2.5, §3.2, §5.4) per PRD_BRIEF**
Prepared: 2026-09-19 · Qadir research pass · All citations verified via live web search. Anything not verifiable is marked **NOT-VERIFIED** — nothing below is invented.

---

## §1 — TOP 20 PAPERS TABLE

| # | Title / Authors / Year | One-sentence relevance to AHMAD-1 | 2× RTX 5090 (32GB) feasible? |
|---|---|---|---|
| 1 | **The Geometry of Algorithms with Orthogonality Constraints** — Edelman, Arias, Smith, 1998, *SIAM J. Matrix Anal. Appl.* | The foundational reference for Stiefel/Grassmannian geometry: geodesics, principal angles, exponential/log maps — the math layer under the new concept layer. | Yes — pure math; no GPU needed. |
| 2 | **Optimization Algorithms on Matrix Manifolds** — Absil, Mahony, Sepulchre, 2008, Princeton UP | Riemannian SGD/CG/trust-region on G(k, R^d): the optimizer family for training subspace-constrained concept parameters without leaving the manifold. | Yes — small-matrix operations; k×d projections are trivial at our scale. |
| 3 | **Efficient Algorithms for Inferences on Grassmann Manifolds** — Gallivan, Srivastava, Liu, Van Dooren, 2003, IEEE SSP | O(n·k²) algorithms for geodesics and principal angles — makes per-token Grassmannian distance computation during inference cheap enough for real-time anomaly scoring. | Yes. |
| 4 | **Grassmann Discriminant Analysis: A Unifying View on Subspace-Based Learning** — Hamm & Lee, 2008, ICML | Classification by projection/Binet–Cauchy kernels over principal angles — the template for mapping a latent state to the nearest root-subspace ("which concept is active"). | Yes. |
| 5 | **Building Deep Networks on Grassmann Manifolds** — Huang et al., 2018, AAAI | GrNet shows end-to-end layers can operate natively on Grassmannian points with mapping/projection/pooling blocks — an architectural pattern for the geometric concept layer. | Yes — GrNet blocks are small orthonormal matrices. |
| 6 | **Riemannian Approach to Batch Normalization** — Cho & Lee, 2017, NeurIPS | Demonstrates stochastic Grassmannian SGD with momentum inside ordinary deep training — proof manifold-constrained parameters can be optimized stably alongside Euclidean ones. | Yes. |
| 7 | **Infeasible Deterministic, Stochastic, and Variance-Reduction Algorithms for Optimization under Orthogonality Constraints** — Ablin & Peyré, 2023, arXiv:2303.16510 | The "landing method" makes orthogonality-constrained SGD practical at network scale without expensive retractions — the right optimizer if concept bases are trained jointly with a transformer. | Yes. |
| 8 | **The Role of Principal Angles in Subspace Classification** — Huang, Qiu, Calderbank, 2016, *IEEE Trans. Signal Processing* | Proves misclassification probability between subspaces is governed (low noise: product of sines; high noise: sum of squares of sines of principal angles) — the theoretical basis for using principal angles as the ISNAD-grade distance metric. | Yes. |
| 9 | **Subspace Tracking with Dynamical Models on the Grassmannian** — Saad-Falcon et al., 2023 (IEEE TSP / arXiv) | Tracks time-varying subspaces under distribution shift with geodesic/chordal regularizers — directly the "training-distribution subspace drifts; flag queries outside the neighborhood" mechanism of the 8200 framework. | Yes — chordal variant avoids per-step SVD (4.7× faster per the paper). |
| 10 | **ViM: Out-of-Distribution with Virtual-Logit Matching** — Wang et al., 2022, CVPR | OOD score = residual after projecting features onto the principal ID subspace of the classifier — a one-file implementation pattern for AHMAD-1's anomaly head. | Yes — post-hoc, CPU-cheap. |
| 11 | **Out-of-Distribution Detection with Deep Nearest Neighbors** — Sun, Ming, Zhu, Li, 2022, ICML | Non-parametric kNN on penultimate features beats parametric Mahalanobis with no distributional assumptions — a zero-training baseline and fallback for the anomaly product. | Yes. |
| 12 | **Activation Subspaces for Out-of-Distribution Detection (ActSub)** — Zöngür et al., 2025, ICCV | State of the art among subspace-based OOD scores using decisive/insignificant activation subspaces — the current quality bar the Grassmannian detector must beat. | Yes. |
| 13 | **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** — Shao et al., 2024, arXiv:2402.03300 | Introduces GRPO, the critic-free RL algorithm that halves PPO memory — the core training engine specified in the PRD. | Yes — GRPO's whole point is small-memory RL. |
| 14 | **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** — DeepSeek-AI (Guo et al.), 2025, arXiv:2501.12948 | Establishes the base→RL-with-verifiable-rewards recipe and its distills (1.5B–70B) — the strategic template for AHMAD-1's pipeline and the bar for "reasoning model" claims. | Yes for 1.5B-distill-scale work; the R1-Zero run itself was large-cluster. |
| 15 | **SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild** — Zeng et al., 2025, arXiv:2503.18892 | Systematic small-scale GRPO study: simple +1/0 correctness rewards, difficulty-matched data, and the warning that format rewards can *hurt* base-model exploration — the hyperparameter playbook for our 0.6B–4B runs. | Yes — trains 0.5B–7B with one recipe. |
| 16 | **DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL** — Luo et al., 2025 (tech report) | Iterative context-length scaling (8K→24K) with GRPO took a 1.5B model to 43.1% AIME — the reference point for what "domain expert at small scale" actually costs in compute. | Partially — full recipe is heavy; a reduced 2–4K-context version fits. |
| 17 | **Qwen3 Technical Report** — Yang et al. (Qwen Team), 2025, arXiv:2505.09388 | Documents Qwen3-0.6B/1.7B/4B with thinking/non-thinking modes (Apache-2.0) — our named base models (d=1024 for 0.6B, d=2560 for 4B) and their pre-RL baselines. | Yes — inference of all sizes; 4B GRPO fine-tuning fits with 8-bit optimizers. |
| 18 | **SmolLM3: smol, multilingual, long-context reasoner** — Allal et al. (Hugging Face), 2025 | Fully-open 3B Apache-2.0 model trained on 11.2T tokens with dual think/no-think modes — the strongest fully-reproducible sub-4B training recipe in the literature and the methodology benchmark for "publishable in 30 days." | As inference base: yes. Reproducing its pretraining: no. |
| 19 | **The Era of 1-bit LLMs: All Large Language Models Are in 1.58 Bits** — Ma et al., 2024, arXiv:2402.17764 | BitNet b1.58 ternary training matches FP16 at ≥3B with 3.55× less memory — the only credible path to genuinely sub-1GB *with* reasoning intact, but requires training-from-scratch (QAT), not post-hoc quantization. | Not for 3B pretraining; the *fine-tuning/QAT continuation* variant on 0.6B is feasible. |
| 20 | **Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct** — O'Neal et al., 2026, arXiv:2601.14277 | Controlled data on GGUF tiers: sub-3-bit (Q3_K_S) collapses instruction-following benchmarks; 4-bit K-quants stay near-FP16 — sets the Q2_K "compression floor" risk line for §5.4. | Yes — pure conversion + lm-eval. |

### Supplementary verified references (beyond the top 20, all confirmed to exist)

- **TinyZero** — Pan et al., 2025 (github.com/Jiayi-Pan/TinyZero): minimal R1-Zero reproduction; verified note that **Qwen2.5-0.5B base fails to learn reasoning under GRPO** — directly relevant to 0.6B base selection (use Qwen3, not Qwen2.5, at the floor).
- **Memory-R+ / "Reasoning Under 1 Billion"** — Le, Do, Nguyen, Venkatesh, 2025, arXiv:2504.02273: memory-augmented GRPO for 0.5B–1B models on a single H100 — closest published analog to sub-1B GRPO on consumer GPUs.
- **Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?** — Yue et al., 2025, arXiv:2504.13837: pass@k evidence that RLVR sharpens but does not create capability — caps what GRPO alone can promise (§5.4).
- **RGR-GRPO (rubric-driven GRPO)** — Chen et al., 2025, arXiv:2511.12344: dense rubric rewards for non-math multi-domain reasoning — a reward-design template for ISNAD-graded linguistic rewards.
- **S2K** — 2025, arXiv:2508.15213: accuracy + format GRPO rewards in a non-math domain reach parity with domain-pretrained LLMs using 2–3 orders of magnitude less data — precedent for high-rejection-rate curation (Bukhari-standard training).
- **Enigmata** — Chen et al., 2025, arXiv:2505.19914: synthetic verifiable puzzles scale logical reasoning — candidate task family for Packet A/B reasoning exams.
- **LogicPuzzleRL** — Wong et al., 2025, arXiv:2506.04821: RL on logic puzzles transfers to out-of-distribution math reasoning — supports training reasoning *as* a transferable skill rather than per-domain knowledge.
- **Geometry-Calibrated Conformal Abstention for Language Models** — Xu, Chen, Xie, Xiong, 2026, arXiv:2604.27914: post-hoc conformal abstention with finite-sample correctness guarantees using representation geometry — the calibration method for the ISNAD self-grade head (mutawatir/sahih/… as participation-tier thresholds).
- **ArabicMMLU** — Koto et al., 2024: 14K native Arabic MCQs, MENA-focused — primary Arabic benchmark target.
- **AceGPT: Localizing Large Language Models in Arabic** — Huang et al., 2023, arXiv:2309.12053: SFT + RLAIF recipe that set open Arabic SOTA — evidence RL-style feedback works for Arabic cultural/linguistic alignment.
- **Evaluating Arabic Large Language Models: A Survey of Benchmarks, Methods, and Gaps** — 2025, arXiv:2510.13430; and **Open Arabic LLM Leaderboard** — El Filali et al., 2024/2025 (OALL v2: ArabicMMLU, ALRAGE, AraTrust, MadinahQA): the evaluation suite AHMAD-1 must report on to be credible.
- **BitNet b1.58 2B4T Technical Report** — Ma et al., 2025, arXiv:2504.12285; **bitnet.cpp** — Wang et al., 2025, arXiv:2502.11880: the shipped artifact + edge inference stack for the ternary path.
- **DICE: Leveraging Sparsification for OOD Detection** — Sun & Li, 2022, ECCV: sparsify weights to maximize Mahalanobis separation — cheap strong baseline for the anomaly detector.

### NOT-VERIFIED items (searched, could not confirm — do not cite in external docs)
- "Qwen3.5-35B-A3B MoE" as named in the PRD: a Qwen3.5-27B GGUF was found, but the specific 35B-A3B single-H100-trainable config is **NOT-VERIFIED**.
- The Modal "#1 pick = Qwen3-4B-Base + GRPO" recommendation: **NOT-VERIFIED** (source not locatable); treat as founder-supplied.
- "Incentivizing Reasoning Capability in LLMs via RL for Language Puzzles" (Shi et al.): **NOT-VERIFIED** in my searches — use the verified Enigmata / LogicPuzzleRL instead.
- ISNAD chain-grading as an ML framework: no literature exists — it is founder IP; only its calibration target (conformal/selective prediction) has literature, cited above.

---

## §2.2 — Why Subspaces (Grassmannian) Can Succeed Where the Simplex Loss Failed

**What died in MUTHALLATH-1 v0.1:** the simplex loss forced each concept to a *single point* at a regular-simplex vertex — a 0-dimensional object with a fixed, equal-angle geometry. The ablation showed ON ≈ OFF: the constraint added no usable signal to a frozen Qwen2.5-7B.

**Five literature-grounded reasons a Grassmannian formulation is different in kind, not degree:**

1. **Basis invariance vs. point brittleness.** A point on the simplex breaks the moment the backbone's representation drifts; a point on G(k, R^d) is an entire equivalence class — the *span* of a basis, quotienting out all k×k rotations (Edelman 1998; Absil 2008). The concept survives any in-subspace reparameterization of the frozen backbone. The simplex loss was fighting the backbone's geometry; the Grassmannian quotients over it.

2. **Continuous graded similarity instead of binary vertex assignment.** Principal angles give a differentiable similarity geometry between concepts (Hamm & Lee 2008), and classification error is provably controlled by them (product/sum of sines — Huang, Qiu & Calderbank 2016). Arabic morphology *needs* graded similarity: cognate roots, shared radicals (ك-ت-ب family), and verb-form derivations are "close but distinct," which a hard vertex assignment cannot express but a small-principal-angle relation expresses natively.

3. **The empirical thesis was already about subspaces.** The founder's Core Thesis §1.1 — mutually exclusive concepts sit near simplex vertices with pairwise cosine ≈ −1/(k−1) — is in the literature a statement about *near-orthogonal subspaces* (superposition/feature-geometry line of work; ActSub 2025 and ViM 2022 both find ID structure concentrates in top principal subspaces of activations). MUTHALLATH-1 projected that subspace observation down to points and lost the information. Grassmannian form keeps the object the evidence actually supports.

4. **Group-action structure for the awzan.** On the simplex there is no natural operation "transform Form I → Form X." On G(k, R^d), rotations/reflections in O(d) act transitively on subspaces, and geodesics give a canonical interpolation path (Edelman 1998; Gallivan 2003). The 10 verb forms become *learned shared operators* on root subspaces — a much richer hypothesis class than vertex displacement, and one that explains why Form X (استفعل, "seeking the root meaning") should be a *composable transformation* rather than a new point.

5. **Measurable before trained.** A point-vertex loss can only be evaluated through end-task accuracy (which is why the NULL ablation was discovered so late). Grassmannian structure can be *probed* in the frozen base model first: compute principal-angle statistics between root-conditioned activation subspaces (cheap, O(n·k²) — Gallivan 2003; chordal metrics avoid per-step SVD — Saad-Falcon 2023) and establish an effect size *before* any training. This directly fixes the MUTHALLATH-1 process failure (committing to an architecture before measuring it). **Sequencing rule: probe → post-hoc scoring → only then train-time constraint.**

**Honest caveat:** none of this guarantees a nonzero ablation. The literature shows manifold-constrained optimization is stable (Cho & Lee 2017; Ablin & Peyré 2023) and that subspace geometry carries discriminative signal (Hamm & Lee 2008; Huang et al. 2016), but no published work shows a Grassmannian *concept layer* improves a frozen-transformer downstream task. AHMAD-1's gate must remain: kill on NULL, like MUTHALLATH-1.

---

## §2.5 — Reasonable k for G(k, R^d)

Requirements: ~4,500 trilateral roots × 10 awzan ≈ 45,000 semantic units; d = 1024 (Qwen3-0.6B) or d = 2560 (Qwen3-4B). dim G(k, R^d) = k(d−k).

**Capacity math:**

| d | k | dim G(k, d) | Verdict |
|---|---|---|---|
| 1024 | 3 | 3,063 | Root-atom scale; ample for 4,500 3-subspaces (see packing argument below) |
| 2560 | 3 | 7,671 | Ample; ~2.5× more room |
| 1024 | 8 | 8,128 | Layer-subspace scale; better noise tolerance |
| 2560 | 8 | 20,416 | Generous |
| 1024 | 16 | 16,128 | Max-discriminability headroom |
| 2560 | 16 | 40,704 | Near-maximal capacity (max at k = d/2) |

**Packing check (can 4,500 roots actually coexist?):** with d ≫ k, random k-subspaces of R^d are already nearly orthogonal in every principal angle (expected θ_i ≈ π/2); the Welch-type bound for N subspaces gives mean squared chordal correlation ≈ k·(d−k)·N / (d·(N−1)). For k=3, d=2560, N=4,500 this evaluates to ≈ 2.99 ≈ k — i.e., the bound is saturated at *orthogonality*, meaning 4,500 three-dimensional root subspaces fit with all principal angles near 90°. Discriminability is not the binding constraint; the binding constraints are (a) per-subspace energy under noise and (b) memory.

**Recommendation — split the role of k:**
- **k = 3 for the semantic root layer** (matches the trilateral thesis: three consonantal radicals → 3-dim meaning atom; PRD §2.4). dim G(3, R^2560) = 7,671 degrees of freedom per concept is enormous.
- **k = 8–16 for detection/ISNAD scoring subspaces** (layer-level "normal behavior" subspace for the 8200 anomaly head): more directions = stabler principal-angle estimates at inference.
- **Awzan as operators, not subspaces:** store 10 shared k×k (or low-rank d×d) transformations rather than 45,000 subspaces. This collapses memory from 45,000 to 4,500 bases + 10 operators.

**Memory footprint (stored as bf16 orthobases d×k):**

| Storage choice | d=1024, k=3 | d=2560, k=3 |
|---|---|---|
| All 45,000 units | 276 MB | 691 MB |
| 4,500 roots + 10 awzan operators | 28 MB | 69 MB |
| Same, fp8/int8 | 14 MB | 35 MB |

Conclusion: even the naïve scheme fits inside a sub-4GB GGUF budget; the roots+operators scheme is negligible. **k = 3 (semantic) / 8–16 (detection) is the defensible choice; k = 16 at d = 2560 if a single k must be chosen for maximal discriminability.**

---

## §3.2 — Minimum Compute Estimates for GRPO at 0.6B / 4B

**Cost model (stated assumptions):** FLOPs ≈ 6·N per training token (fwd+bwd), ≈ 2·N per rollout token; per step: P = 64 prompts × G = 8 rollouts; prompt 500 tok, generation 1,500 tok ⇒ ~1.02M tokens/step (~0.77M generated). 2× RTX 5090 ≈ ~160–200 TFLOPS effective BF16 each ⇒ ~350 TFLOPS combined at 35–45% MFU.

| | 0.6B (d=1024) | 4B (d=2560) |
|---|---|---|
| FLOPs/step | ≈ 4.6 × 10^15 (3.7 train + 0.9 rollout) | ≈ 3.1 × 10^16 (24 train + 7 rollout) |
| Realistic step time (vLLM colocated) | ~90–150 s | ~8–15 min |
| Steps/day | ~600–900 | ~100–180 |
| Memory for full fine-tune (bf16 + AdamW) | ~8–10 GB → **fits on one card**; card 2 runs vLLM rollouts | ~56 GB (fp32 Adam) → needs 8-bit Adam + bf16 master (~24 GB) + grad checkpointing → **fits, tight**; or FSDP over both cards (PCIe, acceptable at this size) |
| Minimum meaningful run (1.5–3K steps, per SimpleRL-Zoo/DeepScaleR-lite scale) | **2–4 days local** | **2–4 weeks local** — the 4B run is a stretch on local hardware |
| Context ceiling locally | 4K comfortable; 8K with checkpointing | 2–4K recommended; DeepScaleR-style 24K is off-limits locally |

**$500 rental comparison (market rates ~$1.5–2.5/GPU-hr for A100/H100-class in 2026):** ≈ 200–250 H100 GPU-hours ≈ 25–30 hours on an 8×H100 node. One such node ≈ 12–15× the sustained throughput of the 2×5090 box (plus NVLink, which GRPO's weight-sync phase loves). $500 therefore buys roughly **2–4 weeks of local-equivalent compute** — enough for: (a) 2–4 serious 4B GRPO runs at 4K context, or (b) one extended 0.6B curriculum + full OALL evaluation, or (c) the 4B GRPO stage done in 2–3 days instead of a month.

**Recommendation:** do all 0.6B iteration and pipeline shakedown locally (free, fast feedback); spend the $500 rental on the 4B GRPO stage and any 8K+ context experiments. Consistent with the PRD's $0–500 envelope.

---

## §5.4 — Risk Assessment

**Top 5 failure modes (probability = honest estimate for this program; mitigations = actions, not hopes):**

| # | Failure mode | P(fail) | Mitigation |
|---|---|---|---|
| 1 | **Grassmannian layer ablates NULL again** — same failure as MUTHALLATH-1 v0.1: geometry carries no measurable downstream signal | **~40%** | Probe-first sequencing (§2.2, point 5): establish principal-angle effect size on the frozen base *before* building the layer; ship geometry first as a post-hoc inference-time scorer (ViM/ActSub-style, zero training risk); kill-on-NULL gate at day 7 per the goal map. |
| 2 | **Non-math reward design fails** — no verifier for Arabic linguistic/epistemic reasoning ⇒ GRPO reward is hackable or uninformative ⇒ training plateaus or reward-hacks | **~45%** | ISNAD rubric rewards (RGR-GRPO pattern): decompose into checkable sub-grades (format, root-correctness via arabic_root.py, citation validity, answer match) so each component is separately verifiable; keep binary correctness dominant (SimpleRL-Zoo's warning: dense/format rewards can hurt base-model exploration); Packet A/B separation to detect memorization-hacking. |
| 3 | **Small-scale GRPO instability or capability ceiling** — entropy collapse, or Yue et al. 2025's finding that RLVR sharpens but does not create capability ⇒ 0.6B/4B base can't reach "domain expert" | **~35%** | Anchor on Qwen3 (TinyZero verified Qwen2.5-0.5B fails to learn reasoning; Qwen3-0.6B is post-thinking-era); difficulty-matched data (SimpleRL-Zoo); keep context ≤4K locally; if GRPO stalls, pivot to distillation from a larger teacher (the one method Yue et al. show *does* add new patterns). |
| 4 | **Compression destroys the gains** — Q2_K-class GGUF collapses instruction-following (O'Neal 2026: Q3_K_S already breaks IFEval) ⇒ the sub-4GB artifact isn't the model you trained | **~30%** | Set Q4_K_M as the quality floor (4B → ~2.4GB, inside budget — no extreme quantization needed); treat sub-1GB as a separate BitNet-continuation experiment, never as the only artifact; always ship a perplexity + IFEval delta report with each GGUF. |
| 5 | **7-day / 30-day timeline slip on arabic_root.py and pipeline build** — the critical missing piece (never built in 6 months) blocks reward verification and Packet A/B exams | **~50%** | arabic_root.py is task #1 and gates everything else (per goal map); timebox to 48h; fallback to an existing Arabic root lexicon (BAMA/Qalsadi-pattern finite-state analyzers) wrapped to the spec interface rather than perfect morphology. |

**What MUTHALLATH-1 v0.1 got wrong that this plan fixes:** (a) committed to architecture before measurement → fixed by probe-first; (b) point-geometry constraint fighting a frozen backbone → fixed by basis-invariant subspaces (§2.2); (c) evaluation only through end-task accuracy → fixed by direct principal-angle statistics; (d) regex-baseline lesson (77.5% vs 85.8%): every claimed capability ships with an embarrassingly simple baseline comparison.

**Honest probability: frontier-class domain reasoning at sub-1GB.**
- Narrow, well-defined domain + verifiable rewards + multi-root Monte Carlo test-time scaling: **~10%**. (Grounding: DeepScaleR-1.5B reached 43% AIME — impressive but not "frontier-class"; the sub-1B regime is strictly harder, Memory-R+ shows modest gains; Yue et al. caps what RL adds.)
- Broad, open-domain frontier-class reasoning at sub-1GB: **≤2%**. Current scaling evidence gives no reason to expect this.
- What *is* high-probability (~70%): a sub-4GB Arabic-first reasoner that is **best-in-class for its size on Arabic benchmarks** with built-in calibrated abstention (ISNAD grades) and a competitive sub-1GB anomaly-detection product — that product is achievable on this hardware and budget, and is the safer commercial thesis.

---

*Prepared as the §5.3 deliverable companion to PRD_BRIEF.md. Every cited work was verified against live sources on 2026-09-19; items that could not be verified are explicitly marked NOT-VERIFIED.*
