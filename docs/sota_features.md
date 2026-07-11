# SOTA mechanism fine-tuning features

This document maps each training feature added to the repo to its published
evidence and its expected effect on the oMeS `S_partial` headline metric. All of
these are layered *around* the scorer and data pipeline without touching the two
invariants:

1. **Zero oMe-Gold leakage** — every new data path (distilled traces, augmented
   rows, external corpora) flows through the Phase-6 blacklist
   (`rxndata.decontaminate`) and is tested.
2. **oMeS comparability** — `omebench_eval.scoring.oMeS` is imported unchanged.
   New signals (validity, format gate, round-trip) are *separate reward terms*,
   never edits to the alignment/scoring algorithm.

Recommended recipe: **SFT (on silver + distilled CoT) → GRPO/DAPO (gated oMeS
reward, validity-monitored)**, optionally with a difficulty curriculum and
SMILES-augmented inputs.

| # | Feature | Where | Evidence | Expected effect |
|---|---------|-------|----------|-----------------|
| 1 | **CoT rejection-sampling distillation** | `training/distill_cot.py` | ether0 (arXiv:2506.17238), RetroDFM-R (arXiv:2507.17448) | Biggest single lever. Warm-starts the policy on long, *verified* reasoning traces so RL starts from a much higher floor. |
| 2 | **DAPO RL objective** | `training/algo.py`, `training/grpo_train.py --algo dapo` | DAPO (arXiv:2503.14476), RetroDFM-R | Token-level loss + clip-higher + dynamic sampling stop GRPO's short-completion bias — important for long mechanism CoT. |
| 3 | **Reward upgrades** | `training/reward.py`, `training/forward_model.py` | ether0, PSV-PPO (arXiv:2505.00530), RTRL (arXiv:2510.01527) | Multiplicative format gate prevents "invalid-but-lucky" reward hacking; explicit validity term guards against validity collapse; optional round-trip feasibility term. |
| 4 | **Validity/reward monitoring** | `training/callbacks.py` | PSV-PPO | Detects the well-documented SMILES-validity collapse during RL; warns or fails loud below a floor. |
| 5 | **Difficulty curriculum** | `training/curriculum.py`, `--curriculum` | RxnNano (arXiv:2603.02215), ether0, "Rethinking Retrosynthesis" (JCIM 2025) | easy→medium→hard ordering; smoother optimization on hard reactions. |
| 6 | **SMILES augmentation** | `training/build_sft_data.py --augment N` | Bjerrum enumeration (arXiv:1703.07076), RetroDFM-R, LlaSMol | Randomized *input* SMILES (canonical targets) → invariance to input representation, more robust generalization. |
| 7 | **DoRA + rank + seq-len** | `sft_train.py`/`grpo_train.py --dora --lora-r`, `--max-seq-len 6000` | DoRA (arXiv:2402.09353), LlaSMol, oMeBench (~6000-tok outputs) | DoRA ≈ +3–4 pts on reasoning vs LoRA at equal params; higher rank and full seq-len avoid truncating long mechanisms. |
| 8 | **External elementary-step corpora** | `src/rxndata/ingest/{pmechdb,rmechdb}.py` | PMechDB (JCIM 2024), RMechDB (JCIM 2023), FlowER | The only public corpora with genuine *elementary* steps. **License-gated**: CC-BY-NC-ND → quarantined, not bundled. |

## 1. CoT rejection-sampling distillation

```bash
python -m training.distill_cot --dataset silver --model opus \
    --limit 1000 --keep-threshold 0.9 --require-validity 1.0 \
    --max-attempts 2 --out-dir training/data
```

Answer-conditioned prompting: the frontier model is shown the *known gold
mechanism* and asked to write the reasoning that derives it, then re-emit it in
the `[ANSWER]...[/ANSWER]` format. Each trace is scored with `oMeS`; only traces
with `S_partial ≥ keep-threshold` **and** `validity ≥ require-validity` survive.
Kept traces are decontaminated against oMe-Gold and written in the same chat
schema `sft_train.py` consumes (`messages` + `prompt` + `reference`). The student
then SFTs on these long verified traces before RL.

## 2. DAPO vs GRPO

`--algo grpo` (default) leaves TRL's defaults untouched. `--algo dapo` sets, on
`GRPOConfig`, `loss_type="dapo"` (token-level aggregation), `epsilon_high=0.28`
(clip-higher), `scale_rewards="none"` (no per-group std normalization), and
`mask_truncated_completions=True` (drop degenerate/truncated rollouts — the
closest stock-TRL analogue to DAPO dynamic sampling). Knobs are filtered against
the installed `GRPOConfig` fields, so an older TRL drops unsupported keys with a
warning rather than crashing. Requires a recent TRL (`loss_type="dapo"` support,
roughly ≥ 0.21).

## 3. Reward composition

`training/reward.py` weights (module constants, overridable via `grpo_train`
flags `--w-spartial/--w-validity/--w-format/--w-roundtrip`):

- **Multiplicative gate**: unparseable or all-invalid output → `FORMAT_FAIL`
  (`-0.5`); oMeS is never called, so no validity/format credit leaks to garbage.
- **`omes_reward`**: `W_SPARTIAL·S_partial + W_VALIDITY·validity + W_FORMAT`.
- **`validity_reward`**: standalone fraction-of-valid-SMILES signal, always in
  the reward_funcs list so the loss keeps an explicit validity term.
- **`--reward omes+roundtrip`**: adds a forward-model feasibility term. The
  forward model is pluggable (`training/forward_model.py`); the default is a
  **no-op** that abstains (contributes 0), so enabling it needs no extra weights
  and never penalizes. Inject a real `ForwardModel` to activate it.

## 4. Monitoring

`MechMonitor` (a plain rolling-window aggregator) + `make_monitor_callback`
(a lazy-imported `transformers.TrainerCallback`) log mean reward, S_partial,
validity, and completion length each logging step, and warn — or raise with
`--fail-on-validity-collapse` — when windowed validity drops below
`--validity-floor` (default 0.8). `instrument_reward_funcs` feeds the monitor
from the reward pass, so there is no second oMeS computation.

## 5. Curriculum

`--curriculum` orders rows easy→medium→hard by the oMe `level` label and disables
dataset shuffling (SFT uses a `SequentialSampler`; GRPO sets
`shuffle_dataset=False`) so the ordering survives into training.

## 6. SMILES augmentation

`--augment N` emits up to N randomized-SMILES variants of each **train** row's
input reactants/products (RDKit `doRandom`), keeping the assistant target and the
`reference` canonical (LlaSMol's "canonical target, augmented input"). Runs
**after** the val split (train-only) but **before** decontamination, so leaks and
their variants are still caught.

## 7. DoRA / rank / sequence length

`--dora` sets `use_dora=True` in `LoraConfig` (implies LoRA); `--lora-r` is
sweepable. SFT `--max-seq-len` defaults to **6000** to fit long mechanism CoT
(cross-check against the token histogram in `notebooks/train_eval_colab.ipynb`
cell 5 so mechanisms are not silently truncated).

## 8. External elementary-step corpora (license-gated)

`src/rxndata/ingest/{pmechdb,rmechdb}.py` map the Baldi-group elementary-step
CSVs (SMIRKS `reactants>>products` + arrow codes) into the repo's single-step
`mechanism` schema. **Both datasets are CC-BY-NC-ND-4.0** (verified in
`configs/license_inventory.json`): the No-Derivatives + click-through terms
forbid redistributing a derived corpus, so:

- their config verdict is `quarantine` and they ship `enabled: false`;
- Phase 1's gate refuses to ingest them into any release;
- the raw CSVs are **never bundled** — a user must accept the provider license
  and place the file at `data/raw/<path>` themselves;
- every mapped row is flagged `needs_remap` (the source ontology ≠ oMeBench's),
  so Phase 3 remap / Phase 5 validation is the authority on the final label.

If a human clears the license for local research, flip `enabled`/`verdict` in
`configs/pipeline.yaml`; the records still flow through Phase 5 validation and
Phase 6 decontamination like every other source.
