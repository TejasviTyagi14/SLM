# BrainLift: Training an LLM to do Organic Chemistry

## Owners
- [Your Name]
- Research assistant (AI)

## Purpose

### Purpose
Build the sharpest possible working model of **how to train an LLM that genuinely reasons about organic reaction mechanisms and predicts reaction outcomes** — not one that pattern-matches its way to plausible-looking SMILES. The North Star: a specialist model whose *causal, step-by-step mechanistic reasoning* holds up under expert-curated, programmatically-verifiable evaluation (oMeBench/oMeS), closing the gap between "chemically valid output" (largely solved) and "chemically correct mechanism" (the open frontier). This document curates the benchmarks, training recipes, representations, foundational models, failure modes, and the people to follow, and distills them into a defensible point of view on *where the leverage actually is* — so that AI conversations about this project start from evidence, not vibes.

### In Scope
- Organic **reaction-mechanism reasoning** (multi-step, arrow-pushing-level logic) and **reaction/product prediction**.
- Benchmarks and evaluation methodology for mechanism reasoning (oMeBench/oMeS, ChemBench, GPQA-Diamond organic subset, ChemCoTBench, RxnBench, ChemLLMBench).
- Training methods: in-context learning (ICL), SFT, CoT-SFT / reasoning distillation, RL/GRPO/RLVR with verifiable rewards, tool-use/agents, multimodal literature reading, and domain pretraining.
- Molecular **representations** for sequence models (SMILES/canonical/C-SMILES/R-SMILES/SELFIES/IUPAC) vs graph methods.
- Foundational reaction-prediction models (Molecular Transformer, RSGPT, RetroDFM-R, LlaSMol, ether0, Chem-R, ChemDFM).
- **Failure modes**: memorization vs reasoning, overconfidence, reward hacking, in-domain overfitting.
- First-party empirical findings from our own oMeBench harness runs.

### Out of Scope
- Inorganic, organometallic catalysis design, materials, or protein/biologics modeling except where a cited source spans them.
- Wet-lab autonomous experimentation as an end goal (Coscientist/ChemCrow are cited only as agent-orchestration evidence).
- De novo molecular *generation* for drug discovery, property optimization, and ADMET prediction as primary objectives.
- Full pretraining-corpus engineering beyond what informs the pretraining-vs-post-training debate.
- Cheminformatics tooling internals (RDKit implementation) beyond their use as verifiers/reward signals.

## DOK 4: Spiky Points of View (SPOVs)

**1. For narrow mechanistic reasoning, specialization beats scale — a small fine-tuned specialist can outclass frontier closed models, and the field keeps mistaking parameter count for chemical competence.**
Elaboration: The oMeBench preprint (DOK2) reports that a fine-tuned **4B specialist scored ~+50% over Claude-Sonnet-4** on mechanism reasoning, while the best frontier model (Gemini-Pro-2.5) reached only S_part 37.9 / S_tot 34.9 (DOK1). On the reaction-prediction side, ether0 (24B) hit **70% reaction prediction after 46,000 examples, beating Molecular Transformer's 64.1% trained on ~480,000 USPTO reactions** — roughly 10x less data (DOK1). LlaSMol's LoRA-tuned Mistral beat GPT-4 on retrosynthesis **32.9% EM vs ~0.0%** (DOK1). This synthesizes DOK3 Insight ("data-efficiency crossover") and directly contradicts the default "bigger frontier model wins" prior. Implication for the project: our budget is best spent on a well-targeted specialist (SFT + RL on oMe-Silver) rather than chasing frontier-scale general capability — with the honest caveat from our own runs that GPT-5.5 (~48 S_partial) still sits ~18 points above the paper's 4B (~30), so "specialization wins" holds for *data/param efficiency and beating peer-scale baselines*, not for trivially dethroning the very best general reasoner without the untapped RL lever.

**2. RLVR is the single most under-exploited lever in chemistry precisely because oMeS-style metrics make the reward programmatically verifiable — and almost no prior mechanism work has actually used it.**
Elaboration: oMeS is a *deterministic* weighted Needleman-Wunsch alignment producing SMILES validity (V), logical/step-type fidelity (L), and structural scores (S_tot/S_part) (DOK1) — i.e., a rule-based verifier that requires no human or LLM judge. The 2025–26 dominant recipe (ether0, Chem-R, RetroDFM-R, ChemDFM-R) is RL/GRPO/RLVR with rule/tool rewards like RDKit validity and canonical-SMILES match (DOK2), and RetroDFM-R's RL pushed USPTO-50k top-1 to **65.0%**, beating EditRetro 60.8% and Graph2Edits 55.1% (DOK1). Yet, as our first-party findings note, prior mechanism work **did not apply RL against the verifiable oMeS reward** — that is the untapped lever. This bridges DOK3 (verifiable-reward Insight) to action: because we have oMe-Silver's **2,493 reactions with per-step rationales** and a programmatic oMeS scorer, we can run GRPO/RLVR directly on mechanism reasoning. Implication: the highest-EV experiment in the whole project is RL against oMeS, not more SFT. Caveat (SPOV-honesty): ether0 documents **reward hacking** (peroxides to inflate O-count, hydrazines for solubility) (DOK1), so the reward must be hardened against degenerate exploits.

**3. Validity is solved; causal correctness is the frontier — chemistry LLMs already write well-formed molecules and still get the mechanism wrong.**
Elaboration: Across oMeBench, **SMILES validity sits around ~90% everywhere, yet S_tot collapses with difficulty** (DOK1), and our own harness shows the same shape: GPT-5.5 scored **easy 0.65 / medium 0.46 / hard 0.27 S_partial**, a steep multi-step drop-off (first-party DOK1). ChemBench (peer-reviewed, Nature Chemistry) finds models **overconfident and weak on structural reasoning** despite beating human experts overall (DOK1). This synthesizes the DOK3 "syntax vs semantics" Insight: fluency in the representation language is not the same as modeling electron flow. Implication: metrics and training must target *stepwise causal fidelity* (L and S_tot, per-step rationales), and headline "accuracy"/validity numbers are actively misleading for prioritization. Our own truncation finding reinforces this — raising the output-token budget moved **validity 0.80→0.99 with S_partial only reaching 0.469**, i.e., fixing validity did not confer correctness.

**4. Domain pretraining is not the prerequisite the field assumes — for reaction/mechanism competence, RL post-training on top of a strong general base can match or beat the "pretrain-on-chemistry-first" school.**
Elaboration: There is an *active, live disagreement* in the literature (DOK2): the domain-pretraining school (ChemDFM ~34B tokens of papers+textbooks; ChemPile >75B-token corpus) versus ether0's explicit claim that **chemistry post-training works WITHOUT domain pretraining and with less data** (DOK1). Damningly, oMeBench reports that domain-tuned models **ChemDFM and BioT5+ score near-zero on mechanism reasoning** (DOK1), and the Memorization study (Zhao/Edwards/Ji, NeurIPS 2023) shows domain LMs like Galactica are **brittle, memorize, and don't propagate knowledge between similar molecules** (DOK1). This bridges DOK3's "pretraining ≠ reasoning" Insight to a concrete bet. Implication: rather than assembling a giant chemistry pretraining corpus, start from a capable general reasoning base and invest in CoT-SFT + RLVR — with the caveat that ether0 "does not perform especially well on ChemBench" (DOK1), so this route buys *targeted* mechanism/reaction skill, not broad chemical breadth.

**5. Benchmark in-domain overfitting inflates apparent skill — much "chemical reasoning" is retrieval, memorization, and copying, and domain-tuned models can underperform general models on genuinely novel mechanisms.**
Elaboration: The Memorization study finds ICL benefit is **~2x larger for computed than experimental properties** and that apparent gains can be copying rather than reasoning (DOK1); ICL scaffolds help big models most but "gains can be memorization/copying" (DOK2). Meanwhile oMeBench shows **domain models near-zero on mechanism reasoning** even as general frontier models lead (DOK1), and GPQA-Diamond analysis shows organic chemistry is the **hardest subfield — 70% of the 40 hardest questions vs 36% of the full set** — precisely because it needs spatial/diagram/causal reasoning that resists memorization (DOK1). This synthesizes the DOK3 "held-out mechanisms are the real test" Insight. Implication: we must evaluate on literature-verified, held-out mechanisms (oMe-Gold) and treat strong scores on templated/in-distribution splits (oMe-Template) with suspicion; a model that looks great in-domain may be memorizing scaffolds, not pushing arrows.

### From SPOV to code: the implemented training recipe

The SPOVs above are now operationalized in `training/` (full evidence map in
[`docs/sota_features.md`](docs/sota_features.md)). Each lever traces to the SPOV
it acts on:

- **CoT rejection-sampling distillation** (`training/distill_cot.py`) — operationalizes SPOV-2 + Insight-8 (distillation transfers reusable *process*). A frontier model writes answer-conditioned reasoning; only traces whose final answer passes oMeS (`S_partial ≥ 0.9`, validity `= 1.0`) are kept, then decontaminated. Warm-starting RL on verified long CoT is the field's highest-payoff move (ether0, RetroDFM-R).
- **DAPO RL** (`training/algo.py`, `--algo dapo`) — operationalizes SPOV-2. Token-level loss + clip-higher + dynamic sampling remove GRPO's short-completion bias, which matters because mechanism CoT is long (SPOV-3's multi-step collapse is where the reward must reach).
- **Hardened, gated reward** (`training/reward.py`) — directly answers SPOV-2's reward-hacking caveat and SPOV-3. A multiplicative format gate refuses to reward invalid-but-lucky outputs; per-step *subtype + structure* fidelity (not endpoint similarity) is the signal; an explicit validity term and a monitoring callback (`training/callbacks.py`) guard against the RL validity collapse documented in PSV-PPO.
- **Difficulty curriculum** (`training/curriculum.py`) — operationalizes Insight-3 (the easy→hard *slope* is the real failure surface): train easy→medium→hard so effort concentrates where multi-step reasoning breaks.
- **SMILES augmentation** (`build_sft_data.py --augment`) — operationalizes SPOV-5: randomizing input representations (canonical targets) discourages scaffold memorization and rewards representation-invariant reasoning.
- **Decontamination everywhere** — the SPOV-5 discipline made non-negotiable: every new data path (distilled, augmented, external) flows through the Phase-6 oMe-Gold/Template blacklist, tested, so held-out mechanisms stay the real test.
- **External elementary-step corpora** (`ingest/{pmechdb,rmechdb}.py`) — the scarce genuine-mechanism data SPOV-1/SPOV-2 want more of, but **license-gated**: CC-BY-NC-ND ⇒ quarantined, disabled, never bundled — honesty about provenance is itself part of SPOV-5.

## Experts

- **Kevin Jablonka** — Who: PI at FSU Jena / HIPOLE Jena. Focus: LLM evaluation for chemistry, chemistry pretraining corpora (ChemBench, ChemPile). Why Follow: co-created the peer-reviewed ChemBench benchmark (Nature Chemistry) and the >75B-token ChemPile corpus; central voice in the domain-pretraining school and in rigorous chem-LLM evaluation. Where: https://kjablonka.com/bio.html

- **Philippe Schwaller** — Who: PI at EPFL LIAC (Laboratory of Artificial Chemical Intelligence). Focus: reaction prediction as sequence translation, tool-augmented chemistry agents. Why Follow: lead author of the Molecular Transformer (ACS Cent. Sci. 2019) and co-author of ChemCrow (Nat. Mach. Intell. 2024); his SMILES→SMILES framing underpins IBM RXN and most modern reaction-prediction LLMs. Where: https://schwallergroup.github.io/team.html

- **Andrew White** — Who: Adjunct at University of Rochester; scientist at FutureHouse. Focus: LLM agents and RL for chemistry. Why Follow: co-author of ChemCrow and driving force behind ether0 — the strongest evidence for "RL post-training beats domain pretraining" and for documenting reward hacking. Where: https://www.hajim.rochester.edu/che/people/adjunct-part-time/white_andrew/index.html

- **Connor Coley** — Who: PI at MIT. Focus: computer-aided synthesis planning, retrosynthesis (ASKCOS), mechanistic grounding. Why Follow: sets the standard for mechanistically-grounded, tool-integrated synthesis prediction; essential counterweight to pure end-to-end LLM approaches. Where: https://coley.mit.edu/research

- **Regina Barzilay** — Who: Professor, MIT CSAIL / Jameel Clinic. Focus: molecular machine learning and NLP-for-science methodology. Why Follow: foundational methodological voice on molecular ML and representation learning that informs how chemistry is encoded for models. Where: https://coley.mit.edu/

- **Heng Ji** — Who: Professor, UIUC (Siebel School). Focus: NLP for science; mechanism reasoning benchmarks. Why Follow: senior author on oMeBench and on the Memorization study — the most direct sources on evaluating (and debunking) LLM mechanism reasoning. Where: https://siebelschool.illinois.edu/about/people/all-faculty/hengji

- **Carl Edwards** — Who: Researcher at Genentech (formerly UIUC). Focus: molecule–language modeling, multimodal chemistry. Why Follow: co-author on oMeBench and the Memorization study; bridges molecule representations and language models where mechanism reasoning lives. Where: https://scholar.google.com/citations?user=oMErt30AAAAJ

- **Gabe Gomes** — Who: PI at Carnegie Mellon University. Focus: autonomous chemical experimentation agents. Why Follow: co-led Coscientist (Nature 2023), a canonical demonstration of LLM agents orchestrating real chemistry — evidence for the "LLM-as-orchestrator-of-verified-tools" thesis. Where: https://www.nature.com/articles/s41586-023-06792-0

- **Teodoro Laino** — Who: Distinguished Researcher, IBM Research Zurich. Focus: reaction informatics, IBM RXN platform. Why Follow: co-author of the Molecular Transformer and architect of IBM RXN; deep authority on productionizing reaction prediction at scale. Where: https://pmc.ncbi.nlm.nih.gov/articles/PMC6764164/

## DOK 3: Insights

### Evaluation & the correctness gap
1. **Syntax is solved, semantics is not.** Because SMILES validity sits at ~90% across models on oMeBench while S_tot collapses with difficulty, "the model outputs a real molecule" and "the model got the mechanism right" have decoupled. The binding constraint on progress is causal, stepwise fidelity (L / S_tot), not representational fluency — so both metrics and training objectives must be per-step and structure-aware, not answer-level. This is the empirical backbone of SPOV-3.

2. **Output-token budget is a first-order evaluation variable for reasoning models, not a hyperparameter footnote.** Our harness showed 38/196 oMe-Gold reactions returned empty at a 4k budget (score 0.419), and raising the budget alone moved validity 0.80→0.99 and S_partial→0.469. Any benchmark comparison that doesn't control decode budget can silently mis-rank reasoning models — an evaluation-integrity insight that generalizes beyond chemistry.

3. **Difficulty stratification exposes the real failure surface.** Both the oMeBench paper and our GPT-5.5 run (easy 0.65 / medium 0.46 / hard 0.27) show a monotonic collapse with multi-step depth. Aggregate scores hide this; the actionable signal is the *slope* from easy→hard, which is where multi-step electron-flow reasoning breaks and where training effort should concentrate.

4. **Organic chemistry is a memorization-resistant stress test.** GPQA-Diamond makes organic chemistry the hardest subfield (70% of the 40 hardest questions vs 36% overall) specifically because it demands spatial/diagram/causal reasoning. This reframes organic mechanism reasoning as a *general* reasoning benchmark, not a niche domain — strengthening the case (SPOV-5) that in-domain-templated scores overstate skill.

### Training-method leverage
5. **There is a data-efficiency crossover where specialists overtake generalists.** A 4B fine-tuned model beating Claude-Sonnet-4 by ~50% on mechanism, and ether0 beating Molecular Transformer with ~10x less data, are not anomalies — they show that for a narrow, well-specified task, targeted fine-tuning converts a modest parameter budget into task competence more efficiently than scale. This is the analytical core of SPOV-1.

6. **Verifiable, programmatic rewards make chemistry an ideal RLVR domain — and oMeS is a ready-made reward.** Retrosynthesis/prediction rewards (RDKit validity, canonical-SMILES match, purchasability) already power the 2025–26 RL wave (RetroDFM-R 65.0% top-1). oMeS extends this to *mechanism* reasoning with a deterministic alignment score. The insight bridging DOK2 to SPOV-2: we can drop mechanism reasoning into a GRPO loop with zero human/LLM judging.

7. **Reward hacking is the predictable tax on RLVR and must be designed against up front.** ether0's documented exploits (peroxides to game O-count, hydrazines to game solubility) show that any rule-based reward will be gamed if it's a proxy. For an oMeS-based reward, the mitigation is to reward *per-step type + structure* fidelity (S_tot, L), not just endpoint similarity, so the model can't shortcut the mechanism.

8. **Reasoning distillation transfers reusable process, not just answers.** Chem-R's Chemical Reasoning Protocol distillation and ether0's "distill correct traces → SFT" step both suggest that CoT-SFT teaches transferable reasoning patterns. oMe-Silver's per-step rationales are exactly the substrate for this — CoT-SFT first to install the reasoning shape, then RLVR to optimize correctness against oMeS.

### Pretraining vs post-training, and the memorization trap
9. **Domain pretraining is neither necessary nor sufficient for mechanism reasoning.** ChemDFM and BioT5+ — both products of heavy domain pretraining — score near-zero on oMeBench mechanism reasoning, while ether0 claims strong reaction skill with no domain pretraining. Pretraining installs vocabulary and recall, not causal reasoning; the reasoning must be induced by CoT-SFT/RL. This is the evidentiary spine of SPOV-4.

10. **Apparent reasoning is frequently retrieval in disguise.** The Memorization study's findings (ICL benefit ~2x larger for computed than experimental properties; knowledge failing to propagate between similar molecules; brittleness of Galactica) imply that benchmark gains can reflect copying/interpolation over templated distributions. Combined with domain models' near-zero mechanism scores, this drives SPOV-5: trust held-out, literature-verified splits (oMe-Gold) far more than templated ones.

11. **The right architectural posture is often LLM-as-orchestrator, not LLM-as-oracle.** ChemCrow (GPT-4 + 18 tools) and Coscientist show LLMs excel at planning over *verified* tools; ChemCrow's own caveat — GPT-4-as-judge can't distinguish wrong GPT-4 from ChemCrow — is itself evidence that self-evaluation is unreliable, reinforcing the need for external verifiers (RDKit, oMeS) both at train and eval time.

12. **Sequence/LLM representations have overtaken graph methods for reaction tasks.** C-SMILES reaches 67.2% on USPTO-50k and R-SMILES/sequence-LLM methods now lead, while Graph2Edits (55.1%) trails. This means representation engineering (canonicalization, root-alignment) is a cheaper, higher-yield lever than switching to graph neural architectures — and keeps us in the LLM-native regime where RLVR applies directly.

## DOK 2: Knowledge Tree

### Category A — Benchmarks & Evaluation

#### A.1 Mechanism-reasoning benchmarks

**Source: oMeBench (preprint)**
- DOK 1 - Facts:
  - First large-scale, expert-curated benchmark for organic reaction **mechanism** reasoning; UIUC (Heng Ji / Carl Edwards).
  - >10,000 annotated mechanistic steps; datasets: **oMe-Gold** (literature-verified), **oMe-Template**, **oMe-Silver** (LLM-expanded, includes per-step rationales; 2,493 reactions).
  - **oMeS metric** = weighted Needleman-Wunsch alignment yielding V (SMILES validity), L (logical/step-type fidelity), S_tot (strict exact type+structure), S_part (Tanimoto-weighted partial).
  - Best model **Gemini-Pro-2.5 S_part 37.9 / S_tot 34.9**; GPT-5 S_part 29.1; o3 28.1; Claude-Sonnet-4 17.9; GPT-4o 5.0; DeepSeek-R1 (685B) 25.0.
  - Open models much lower: LLaMA-3-8B 3.3, Qwen-3-4B 4.2. Validity ~90% everywhere but S_tot collapses with difficulty.
  - A fine-tuned **4B specialist ~+50% over Claude-Sonnet-4**; domain models ChemDFM/BioT5+ near-zero on mechanism reasoning.
- DOK 2 - Summary: oMeBench is the field's first serious instrument for measuring *mechanistic* (not just product) reasoning, and its design — a deterministic, per-step alignment metric plus literature-verified and LLM-expanded splits — is exactly what makes both rigorous evaluation and verifiable-reward training possible. Its headline result (near-90% validity but collapsing structural correctness, and domain models failing outright) is the single strongest evidence for the "validity solved, correctness open" and "specialization beats scale" theses. Preprint status warrants caution on exact numbers but the qualitative shape is corroborated by our own harness.
- Link to source: https://arxiv.org/abs/2510.07731

**Source: This project's oMeBench harness runs (first-party primary source)**
- DOK 1 - Facts:
  - Built an API evaluation harness around oMeBench's oMeS metric; measured **GPT-5.5 = 46.9 S_partial** on oMe-Gold (196 reactions, chain-of-thought prompting), closely matching the paper's reported GPT-5.5 = 48.2 (harness validation).
  - By difficulty, GPT-5.5 scored **easy 0.65 / medium 0.46 / hard 0.27** (S_partial).
  - Low output-token budget caused reasoning-model truncation: **38/196 reactions returned empty at 4k tokens (score 0.419)**; raising the budget fixed it (validity **0.80→0.99**, S_partial → **0.469**).
  - oMe-Silver provides **2,493 reactions with per-step rationales** — sufficient for CoT-SFT and, because oMeS is programmatic, a **verifiable reward for GRPO/RLVR**. Built an SFT+GRPO pipeline on this basis.
  - Feasibility judgment: paper's best fine-tuned model is a **4B reaching ~30 S_partial (≈ Claude-Sonnet-4.6)**, still ~18 points below GPT-5.5's ~48; a 1B specialist beating GPT-5.5 head-on is a stretch, but matching mid-tier baselines and applying RL against the verifiable oMeS reward (untapped by prior work) is the key lever.
- DOK 2 - Summary: Our own runs independently reproduce the oMeBench paper's GPT-5.5 result within ~1.3 points, validating the harness, and add two methodological findings the paper does not foreground: (a) decode/output-token budget is a first-order eval variable for reasoning models, and (b) the easy→hard collapse is real and steep. Critically, we identify oMe-Silver + programmatic oMeS as a ready-made RLVR setup that prior mechanism work never exploited — reframing the project's highest-EV bet as RL against oMeS rather than more scale or SFT.
- Link to source: (internal — this project's evaluation harness and pipeline)

#### A.2 General & multimodal chemistry benchmarks

**Source: ChemBench (peer-reviewed, Nature Chemistry 2025)**
- DOK 1 - Facts:
  - Jablonka / Schwaller et al.; **2,788 QA pairs**; leaderboard at chembench.lamalab.org.
  - Best model **o1-preview beats the best human expert ~2x overall**; Llama-3.1-405B competitive.
  - Models **struggle with basic tasks, are overconfident, and are weak on structural reasoning** (e.g., predicting NMR from SMILES).
- DOK 2 - Summary: ChemBench is the peer-reviewed anchor for the "superhuman on knowledge, yet failing basics and overconfident" paradox. It demonstrates that aggregate dominance over experts coexists with brittle structural reasoning — a warning that headline superiority does not imply mechanistic competence, and motivation for evaluating calibration, not just accuracy.
- Link to source: https://www.nature.com/articles/s41557-025-01815-x

**Source: GPQA-Diamond (arXiv 2311.12022; Epoch analysis)**
- DOK 1 - Facts:
  - 198 hardest graduate-level questions.
  - **Organic chemistry = 70% of the 40 hardest questions** vs 36% of the full set — the hardest subfield.
  - Hardest items need spatial/diagram reasoning.
- DOK 2 - Summary: GPQA-Diamond independently corroborates that organic chemistry is disproportionately hard for LLMs, and that the difficulty is concentrated in spatial/causal reasoning. This positions organic mechanism reasoning as a demanding general-reasoning probe, not a narrow domain quirk.
- Link to source: https://epoch.ai/gradient-updates/gpqa-diamond-whats-left

**Source: ChemCoTBench (peer-reviewed, NeurIPS 2025 D&B)**
- DOK 1 - Facts:
  - Frames chemistry as **verifiable modular SMILES operations**; **22,000 expert CoT** traces.
  - Tasks include retrosynthesis, forward/by-product prediction, reaction condition, and **reaction mechanism prediction**.
  - Validated by 13 chemists.
- DOK 2 - Summary: ChemCoTBench operationalizes the idea that chemistry reasoning can be decomposed into *verifiable* modular steps with expert CoT — conceptually aligned with the RLVR thesis and a useful complementary training/eval source for stepwise reasoning beyond oMeBench.
- Link to source: https://arxiv.org/abs/2505.21318

**Source: RxnBench (preprint)**
- DOK 1 - Facts:
  - DP Technology; **multimodal** reaction understanding from literature.
  - **SF-QA: 1,525 MCQs + FD-QA: 540 multi-select** from 108 articles.
  - **No model exceeds 50% on FD-QA.**
- DOK 2 - Summary: RxnBench isolates the multimodal literature-reading bottleneck: extracting reactions from schemes/figures in papers is where models fail hardest (<50% on the harder FD-QA split). This flags multimodal scheme parsing as a distinct, unsolved capability separate from text-only mechanism reasoning.
- Link to source: https://arxiv.org/abs/2512.23565

**Source: ChemLLMBench (peer-reviewed, NeurIPS 2023 D&B)**
- DOK 1 - Facts:
  - First broad **8-task** chemistry benchmark.
  - **GPT-4 best of tested**, but general LLMs are weak at structured chemistry tasks.
- DOK 2 - Summary: The original broad chemistry benchmark; historically important for establishing that general LLMs underperform on structured chemistry, setting the baseline expectation that specialist tuning or tools would be required.
- Link to source: https://arxiv.org/abs/2305.18365

### Category B — Training Methods

#### B.1 RL / GRPO / RLVR post-training

**Source: ether0 (preprint)**
- DOK 1 - Facts:
  - FutureHouse / Andrew White; **24B** (Mistral-Small-24B base); reasons in natural language, outputs SMILES.
  - RL on **640,730 problems across 375 tasks**.
  - Reaction prediction **70% after 46,000 examples** vs Molecular Transformer **64.1% on ~480,000 USPTO** (beats the dedicated model with ~10x less data); MT retrained on a small set scores <30%.
  - Pipeline: r1-distill init → per-task GRPO → distill correct traces → SFT → all-task GRPO → safety.
  - Documents **reward hacking** (peroxides for O-count, hydrazines for solubility).
  - NOT a general chat model; "does not perform especially well on ChemBench"; claims chemistry post-training works **without domain pretraining and with less data**. Apache-2.0.
- DOK 2 - Summary: ether0 is the flagship evidence for both SPOV-1 (data-efficient specialization) and SPOV-4 (RL post-training can substitute for domain pretraining). Its multi-stage GRPO pipeline is a template for our own SFT+GRPO plan, and its documented reward-hacking failures are a direct warning about designing the oMeS reward to be exploit-resistant. Preprint, so treat efficiency claims as strong-but-unrefereed.
- Link to source: https://arxiv.org/abs/2506.17238

**Source: Chem-R (preprint, accepted KDD 2026)**
- DOK 1 - Facts:
  - Shanghai AI Lab; **3-phase**: Chemical Foundation Training (SFT) → Chemical Reasoning Protocol distillation → multi-task GRPO.
  - Surpasses Gemini-2.5-Pro / DeepSeek-R1 by **up to 32% on molecular** and **48% on reaction** tasks; text-only.
- DOK 2 - Summary: Chem-R validates the canonical 2025–26 recipe — foundation SFT, then reasoning-protocol distillation, then multi-task GRPO — with large margins over frontier general models on reaction tasks. It reinforces that CoT distillation + RL is the productive stack and provides a phase structure we can mirror on oMe-Silver.
- Link to source: https://arxiv.org/abs/2510.16880

**Source: RetroDFM-R (preprint)**
- DOK 1 - Facts:
  - RL for retrosynthesis.
  - **USPTO-50k top-1 = 65.0%**, beating EditRetro (60.8%) and Graph2Edits (55.1%).
- DOK 2 - Summary: RetroDFM-R is concrete proof that RL with verifiable rewards raises the state of the art on a standard reaction benchmark and that sequence/LLM RL methods now beat graph-edit methods. It de-risks the RLVR bet for our mechanism setting.
- Link to source: https://arxiv.org/abs/2507.17448

#### B.2 SFT / instruction tuning

**Source: LlaSMol / SMolInstruct (preprint / COLM 2024)**
- DOK 1 - Facts:
  - OSU NLP; **14 tasks, >3.3M samples**.
  - LoRA-tuned Mistral beats GPT-4/Claude-3-Opus hugely: retrosynthesis **32.9% EM vs GPT-4 ~0.0%**; SMILES→Formula **93.2% vs 4.8%**.
- DOK 2 - Summary: LlaSMol is the clearest SFT-side evidence that a modest LoRA-tuned open model can crush frontier general models on structured chemistry, provided the instruction data is right. It underscores that base model + targeted data can outperform scale — a pillar of SPOV-1 — while cautioning that these tasks are structured/extractive, not full mechanism reasoning.
- Link to source: https://osu-nlp-group.github.io/LLM4Chem/

#### B.3 Domain pretraining (the corpus school)

**Source: ChemDFM / ChemDFM-R (preprint / Cell Rep. Phys. Sci. 2025 — mixed)**
- DOK 1 - Facts:
  - SJTU; domain pretraining on **~34B tokens** (3.8M papers + 1.4K textbooks) + 2.7M instructions; v2.0-14B on Qwen2.5.
  - ChemDFM-R adds atomized knowledge + **DAPO RL**.
  - Base is **weak on mechanism reasoning** (near-zero per oMeBench).
- DOK 2 - Summary: ChemDFM is the canonical domain-pretraining effort, and its near-zero oMeBench mechanism score is the field's sharpest counterexample to "pretrain on chemistry and reasoning follows." The ChemDFM-R addition of RL is itself an admission that pretraining alone is insufficient. Mixed publication status; the mechanism-failure finding comes via oMeBench.
- Link to source: https://arxiv.org/abs/2401.14818

**Source: ChemPile (peer-reviewed, NeurIPS 2025 D&B)**
- DOK 1 - Facts:
  - Jablonka; **>75B token / 250GB** chemistry pretraining corpus.
  - Spans SMILES / SELFIES / IUPAC / InChI / images.
- DOK 2 - Summary: ChemPile is the largest curated chemistry pretraining corpus and the infrastructural backbone of the domain-pretraining school. It's the resource against which ether0's "no pretraining needed" claim is implicitly argued; useful if we ever want breadth, but not obviously the lever for mechanism correctness.
- Link to source: https://arxiv.org/abs/2505.12534

**Source: BioT5+ (peer-reviewed, ACL 2024 Findings)**
- DOK 1 - Facts:
  - **252M-parameter T5**; integrates IUPAC + SELFIES; multi-task.
  - **Fails mechanism reasoning** per oMeBench.
- DOK 2 - Summary: BioT5+ shows that clever multi-representation integration in a small encoder-decoder yields multi-task competence but still fails mechanism reasoning — further evidence that mechanism reasoning is a distinct capability not conferred by representation tricks or scale-modest domain training.
- Link to source: https://arxiv.org/abs/2402.17810

### Category C — Foundational Reaction-Prediction Models

**Source: Molecular Transformer (peer-reviewed, ACS Cent. Sci. 2019)**
- DOK 1 - Facts:
  - Schwaller / Laino, IBM; frames reaction prediction as **SMILES→SMILES translation**.
  - **90.4% top-1 on USPTO_MIT**, 78.1% on USPTO_STEREO; powers IBM RXN.
- DOK 2 - Summary: The foundational, peer-reviewed reaction-prediction model whose sequence-translation framing set the paradigm all modern LLM approaches inherit. It's the strong dedicated baseline that ether0 beats with ~10x less data — the reference point for the specialization/efficiency argument.
- Link to source: https://pmc.ncbi.nlm.nih.gov/articles/PMC6764164/

**Source: RSGPT (peer-reviewed, Nat. Commun. 2025)**
- DOK 1 - Facts:
  - GPT pretrained on **10B reactions + RL**.
  - **63.4% top-1 on USPTO-50k**.
- DOK 2 - Summary: RSGPT is a peer-reviewed, large-scale reaction-pretrained + RL model, corroborating that RL improves reaction prediction and providing a strong published USPTO-50k reference just below RetroDFM-R's RL result.
- Link to source: https://www.nature.com/articles/s41467-025-62308-6

### Category D — Tools, Agents & Orchestration

**Source: ChemCrow (peer-reviewed, Nat. Mach. Intell. 2024)**
- DOK 1 - Facts:
  - M. Bran / White / Schwaller; **GPT-4 + 18 expert tools**; autonomous synthesis planning/execution.
  - Caveat: **GPT-4-as-judge can't distinguish wrong GPT-4 from ChemCrow.**
- DOK 2 - Summary: ChemCrow is the canonical evidence that LLMs are strongest as orchestrators of verified tools, and its self-evaluation caveat is a load-bearing argument for external, programmatic verifiers (RDKit, oMeS) rather than LLM-as-judge — directly relevant to how we score and reward our model.
- Link to source: https://www.nature.com/articles/s42256-024-00832-8

**Source: Coscientist (peer-reviewed, Nature 2023)**
- DOK 1 - Facts:
  - Boiko / Gomes, CMU; **GPT-4 agent optimized Pd cross-couplings on a cloud lab.**
- DOK 2 - Summary: Coscientist is the landmark demonstration of an LLM agent planning and executing real chemistry via tools/automation, reinforcing the orchestration thesis and the value of closed-loop verification (here, physical experiments; for us, programmatic oMeS).
- Link to source: https://www.nature.com/articles/s41586-023-06792-0

### Category E — Representations

**Source: SMILES variants & graph methods (C-SMILES / R-SMILES / Graph2Edits)**
- DOK 1 - Facts:
  - **C-SMILES reaches 67.2% on USPTO-50k** (arXiv 2510.16588); R-SMILES (root-aligned) is a competitive variant.
  - Graph methods (**Graph2Edits 55.1%**) now trail sequence/LLM methods.
- DOK 2 - Summary: Representation choice is a cheap, high-yield lever: canonicalized/root-aligned SMILES variants now lead USPTO-50k while graph-edit methods trail, meaning we can stay in the LLM-native sequence regime (where RLVR applies directly) and still be state-of-the-art, rather than switching to graph architectures.
- Link to source: https://arxiv.org/abs/2510.16588

### Category F — Failure Modes & Limitations

**Source: Memorization study (Zhao / Edwards / Ji, NeurIPS 2023 AI4Science)**
- DOK 1 - Facts:
  - **Galactica-1.3B is brittle and memorizes.**
  - **ICL benefit ~2x larger for computed vs experimental properties.**
  - **Knowledge does not propagate between similar molecules.**
- DOK 2 - Summary: This study is the empirical basis for treating apparent chemical reasoning as often-memorization: the computed-vs-experimental ICL gap and the failure to generalize across similar molecules imply that benchmark gains can be interpolation over seen distributions. It anchors SPOV-5 and mandates evaluation on held-out, literature-verified mechanisms.
- Link to source: https://blender.cs.illinois.edu/paper/scientificlm2023.pdf
