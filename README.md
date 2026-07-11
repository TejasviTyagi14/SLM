# oMeBench API Evaluation Harness

Baseline evaluation of frontier LLMs (Claude Opus, GPT-5.5, ...) on
**organic reaction-mechanism reasoning**, using the
[oMeBench](https://github.com/skylarkie/oMeBench) benchmark
([paper, ACL 2026 Findings](https://arxiv.org/abs/2510.07731)) and its
**oMeS** scoring metric.

The official oMeBench repo only runs local HuggingFace models. This harness
adds **API-based providers** (Anthropic + OpenAI), concurrency, resume, and a
leaderboard — while reusing the *exact* oMeS alignment/scoring algorithm so
results stay comparable to the paper.

## What gets measured (oMeS)

For each reaction the model must output a stepwise mechanism as JSON
(`type`, `subtype`, `intermediate_smiles` per step). Predictions are aligned to
the expert gold mechanism with a weighted Needleman–Wunsch algorithm and scored:

| Metric | Meaning |
| --- | --- |
| **S_partial** | Partial credit: type match + Morgan-fingerprint Tanimoto similarity of intermediates (the headline metric). |
| **S_total** | Strict: exact reaction type + exact canonical structure match. |
| **V** | Validity: fraction of predicted intermediates that are parseable SMILES. |
| **L** | Logical fidelity: fraction of gold steps that align to a predicted step of the same type. |

Reference point from the paper: the best model (Gemini-Pro-2.5) reached
**~37.9 S_partial** on gold — so expect double-digit, not near-100, scores.

## Baseline results (this harness)

Gold set (196 reactions), `--prompt cot`, `--max-tokens 32000`:

| Model | S_partial | S_total | V | L | easy / medium / hard (S_partial) |
| --- | ---: | ---: | ---: | ---: | --- |
| gpt-5.5 (reasoning: medium) | **0.469** | 0.420 | 0.99 | 0.643 | 0.647 / 0.457 / 0.272 |

> Note: reasoning models need a **large output-token budget** — reasoning tokens
> count against `--max-tokens`, and a low budget produces empty answers on hard
> reactions (an initial 4k-token run lost 38/196 reactions and scored 0.419).
> The default is now 16000; raise it further for high-effort reasoning.

## Datasets (bundled in `data/`)

| Split | Reactions | Notes |
| --- | ---: | --- |
| `gold` | 196 | Expert-verified textbook reactions (use this for baselines). |
| `template` | 167 | Named-reaction templates with R-group placeholders. |
| `silver` | 2,493 | LLM-expanded set for large-scale runs. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your API keys
export $(grep -v '^#' .env | xargs)   # or use direnv / your shell of choice
```

## Usage

```bash
# Baseline both models on the gold set with chain-of-thought prompting
python -m omebench_eval.cli run --models opus gpt-5.5 --dataset gold --prompt cot

# Fast smoke test (5 reactions, one model)
python -m omebench_eval.cli run --models opus --dataset gold --limit 5

# Re-score an existing predictions file (no API calls)
python -m omebench_eval.cli score --dataset gold \
  --pred results/gold/opus.predictions.jsonl --label opus

# Print a leaderboard across every model you've evaluated
python -m omebench_eval.cli report --dataset gold
```

Runs are **resumable**: predictions stream to
`results/<dataset>/<model>.predictions.jsonl` and completed reactions are
skipped on re-run. Scores land in `results/<dataset>/<model>.eval.json`.

## Model aliases

Defined in `omebench_eval/providers.py` (`MODEL_REGISTRY`). Defaults:

| Alias | Provider | Default model id |
| --- | --- | --- |
| `opus` | anthropic | `claude-opus-4-6` |
| `opus-thinking` | anthropic | `claude-opus-4-6` (extended thinking) |
| `gpt-5.5` | openai | `gpt-5.5` (reasoning: medium) |
| `gpt-5.5-high` | openai | `gpt-5.5` (reasoning: high) |

The concrete model ids are best-effort defaults — override them to whatever your
account exposes:

```bash
python -m omebench_eval.cli run --models opus \
  --model-id claude-opus-4-6-20260101 --dataset gold
```

Any unlisted model works via raw ids:

```bash
python -m omebench_eval.cli run --models my-model \
  --provider openai --model-id o5-preview --dataset gold
```

## Prompt styles

- `--prompt default`: model returns a JSON mechanism directly.
- `--prompt cot`: model reasons first, then returns JSON between
  `[ANSWER]...[/ANSWER]` tags (recommended for reasoning models).

## Layout

```
data/                 bundled oMeBench splits
prompts/              default.txt / cot.txt (from oMeBench, with the 30 allowed subtypes)
omebench_eval/
  scoring.py          oMeS metric (faithful re-implementation)
  dataset.py          split loading + prompt building
  parsing.py          robust JSON extraction from model output
  providers.py        Anthropic + OpenAI clients + model registry
  runner.py           generate -> score -> report
  cli.py              command-line entrypoint
results/              predictions + eval outputs (gitignored)
```

## Dataset construction pipeline (`src/rxndata/`)

A reproducible pipeline that builds a **training-ready mechanism-step dataset**
optimized for oMeBench/oMeS. The core unit is a typed, balanced elementary
mechanistic step with a valid intermediate SMILES — not an overall
transformation. Run it with `make all`; every phase stops at a gate.

```
make setup            # venv + pinned deps + clone oMeBench into third_party/
make all              # phases 1-9 -> data/final/
make eval-format-check # confirm our SFT targets score at ceiling on real oMeS
make test             # pytest (schema, validators, decon controls, format-check)
```

| Phase | Module | What | Gate |
| --- | --- | --- | --- |
| 0 | `ontology.py` | parse 11 types / 31 subtypes from the benchmark; license findings | ontology + license inventory |
| 1 | `ingest/` | ingest 5 license-cleared sources → interim parquet | row counts per source |
| 2 | `normalize.py` | RDKit canonicalization (charge/radical/map-preserving), OPSIN, InChIKey vocab | % sanitized |
| 3 | `mechanism.py` | typing/remap to ontology, bounded R-group expansion, capped inference | subtype distribution |
| 4 | `atommap.py` | RXNMapper atom mapping (overall + per-step) | confidence histogram |
| 5 | `validate.py` | sanitize/valence/balance/continuity(MCS)/no-op gates | pass/reject + reason codes |
| 6 | `decontaminate.py` | remove oMe-Gold/Template overlap (InChIKey + fp + mech-hash) | **0 gold leaks** |
| 7 | `dedup.py` | near-duplicate removal, keep highest provenance | cluster sizes |
| 8 | `format_tasks.py` | 6 SFT task views + pretrain corpus (Qwen-clean) | oMeS self-check |
| 9 | `splits.py` | stratified train/val (test = held-out oMe-Gold) + data card | `data/final/DATA_CARD.md` |

Outputs land in `data/final/`: `sft_mechanisms.jsonl` (the oMeS-critical product),
`sft_reactions.jsonl`, `sft_named_qa.jsonl`, `pretrain_corpus.txt`,
`DATA_CARD.md`, `decontamination_report.json`, and `splits/`.

Config lives in `configs/pipeline.yaml` (seeds, thresholds, per-source
verdicts). Licensing is enforced: PMechDB/RMechDB (CC-BY-NC-ND) and OpenStax
(no-AI clause) are quarantined; only redistributable sources are ingested.

### Publishing the dataset to Hugging Face

`scripts/export_to_hf.py` packages `data/final/` into a multi-config Hub dataset
(`sft_mechanisms`, `sft_reactions`, `pretrain`) with real train/validation splits
routed from the decontaminated split manifests (no train/val leakage), and a
dataset card whose license tag is **derived from the data** (MIT for the Tier-A
build; CC-BY-SA propagates if you rebuild with ORD).

```bash
pip install datasets huggingface_hub

# 1) Always dry-run first (no network): builds configs + writes a local preview
#    to data/final/hf_export/ and prints the push commands.
python scripts/export_to_hf.py --repo-id <user>/rxndata-mechanisms --dry-run

# 2) Authenticate, then push for real.
huggingface-cli login          # or: export HF_TOKEN=hf_...
python scripts/export_to_hf.py --repo-id <user>/rxndata-mechanisms
```

Then load it back:

```python
from datasets import load_dataset
ds = load_dataset("<user>/rxndata-mechanisms", "sft_mechanisms")
import json
messages = json.loads(ds["train"][0]["messages"])   # chat rows
```

The card carries the license inventory, per-source verdicts, and the
decontamination proof (0 oMe-Gold leaks). Flags: `--subsets`, `--private`.

## Training a specialist model

A pipeline for fine-tuning a small (~1B) model to specialize in organic
mechanisms lives in `training/`. Strategy: **SFT (silver + distilled CoT) → RL
(GRPO/DAPO) against the verifiable oMeS reward** (the paper stopped at SFT; RL is
the main untapped lever). The SOTA levers — distillation, DAPO, reward gating,
validity monitoring, curriculum, SMILES augmentation, DoRA — are documented with
their evidence in [`docs/sota_features.md`](docs/sota_features.md). See the
feasibility notes below for realistic targets.

```bash
pip install -r requirements-train.txt   # heavier: torch/transformers/trl/peft

# 1) Build RDKit-validated chat SFT data from oMe-Silver (2,493 rxns w/ rationales).
#    --augment N adds randomized-SMILES INPUT variants (train only; canonical targets).
python -m training.build_sft_data --dataset silver --style cot --val-frac 0.03 --augment 4

# 1b) (BIGGEST LEVER) Distill long verified CoT from a frontier model: only traces
#     whose final answer scores S_partial>=0.9 on oMeS are kept, then decontaminated.
python -m training.distill_cot --dataset silver --model opus \
  --limit 1000 --keep-threshold 0.9 --out-dir training/data

# 2) Supervised fine-tune a ~1B base (--dora for weight-decomposed LoRA on small GPUs;
#    --max-seq-len defaults to 6000 to fit long mechanism CoT; --curriculum = easy→hard)
python -m training.sft_train \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train training/data/sft_distilled_cot_train.jsonl \
  --val   training/data/sft_distilled_cot_val.jsonl \
  --out   checkpoints/sft-qwen1.5b-cot --epochs 3 --dora

# 3) RL against oMeS. --algo dapo (token-level loss + clip-higher, better for long
#    CoT); --reward omes+roundtrip adds a pluggable forward-model feasibility term.
#    A callback logs reward/S_partial/validity/length and warns on validity collapse.
python -m training.grpo_train \
  --model checkpoints/sft-qwen1.5b-cot \
  --train training/data/sft_silver_cot_train.jsonl \
  --out   checkpoints/grpo-qwen1.5b-cot \
  --algo dapo --reward omes+roundtrip --num-generations 8 --validity-floor 0.8

# 4) Evaluate the checkpoint with the SAME harness (concurrency 1 for a local GPU model)
python -m omebench_eval.cli run --models mymodel \
  --provider local --model-id checkpoints/grpo-qwen1.5b-cot \
  --dataset gold --prompt cot --concurrency 1
python -m omebench_eval.cli report --dataset gold   # vs gpt-5.5, opus, ...
```

Pipeline pieces:

| File | Role |
| --- | --- |
| `training/build_sft_data.py` | Silver → chat SFT JSONL (plain/CoT), every intermediate RDKit-validated; `--augment N` SMILES aug; decontaminated; emits `prompt` + `reference` for RL. |
| `training/distill_cot.py` | Frontier CoT rejection-sampling distillation (answer-conditioned, oMeS-verified, decontaminated). |
| `training/reward.py` | Gated oMeS reward + `validity_reward` + optional round-trip; weights as constants + `grpo_train` flags. |
| `training/forward_model.py` | Pluggable forward model for the round-trip reward (no-op default). |
| `training/algo.py` | GRPO vs DAPO config builder (`--algo`), filtered to the installed TRL's knobs. |
| `training/callbacks.py` | `MechMonitor` + TrainerCallback: logs reward/S_partial/validity/length, enforces a validity floor. |
| `training/curriculum.py` | easy→medium→hard ordering (`--curriculum`). |
| `training/sft_train.py` | TRL `SFTTrainer` (assistant-only loss, `--lora`/`--dora`, `--curriculum`, `--max-seq-len 6000`). |
| `training/grpo_train.py` | TRL `GRPOTrainer` (`--algo`, `--reward`, reward-weight flags, `--curriculum`, `--dora`, validity monitoring). |
| `src/rxndata/ingest/{pmechdb,rmechdb}.py` | External elementary-step corpora (CC-BY-NC-ND → quarantined/disabled; user-supplied data only). |
| `omebench_eval.providers.LocalHFProvider` | Runs the resulting checkpoint through the eval harness. |

### Running on TrueFoundry

The pipeline is packaged as four chained TrueFoundry Jobs (`deploy/truefoundry/deploy.py`),
sharing a persistent Volume mounted at `/data`:

```
build (CPU) -> /data/sft/*.jsonl
sft   (GPU) -> /data/checkpoints/sft
grpo  (GPU) -> /data/checkpoints/grpo
eval  (GPU) -> oMeS leaderboard in job logs
```

**One-time setup**

```bash
pip install truefoundry
tfy login --host <your-truefoundry-host>
```

In the dashboard create: a **Workspace**, a **Volume** (for `/data`), and
optionally an **ML Repo** (for the model registry). Grab their FQNs. If you'll
also benchmark API models or use gated base weights, add **Secrets**
(`OPENAI_API_KEY`, `HF_TOKEN`) and reference them as
`tfy-secret://<org>:<collection>:<key>` (see the commented `env` in `deploy.py`).

**Deploy + run each stage**

```bash
WS=<workspace-fqn>
VOL=tfy-volume://<cluster>:<workspace>:<volume>

python deploy/truefoundry/deploy.py --stage build --workspace-fqn $WS --volume-fqn $VOL
python deploy/truefoundry/deploy.py --stage sft   --workspace-fqn $WS --volume-fqn $VOL --gpu A100_80GB
python deploy/truefoundry/deploy.py --stage grpo  --workspace-fqn $WS --volume-fqn $VOL --gpu A100_80GB
python deploy/truefoundry/deploy.py --stage eval  --workspace-fqn $WS --volume-fqn $VOL --gpu A100_80GB
```

Each command registers/updates the Job; trigger runs from the dashboard or
`tfy trigger job --application-fqn <fqn>`. Run stages **in order** (each reads
the previous stage's output from the volume). SFT hyperparameters
(`base_model`, `epochs`, `batch`, ...) are exposed as Job **params**, so you can
launch sweeps from the dashboard without redeploying.

**Publish the model** (optional, after GRPO):

```bash
python -m training.publish_model --ml-repo organic-mechanism \
  --name mech-specialist-1.5b --path /data/checkpoints/grpo
```

Notes:
- `build` runs CPU-only to save cost; the rest request 1 GPU (default
  `A100_80GB`; use `--gpu A10G`/`T4` + `--lora` for cheaper/smaller runs).
- GRPO uses `--use-vllm` for fast rollouts — keep it on if vLLM is in the image.
- The eval stage runs the local checkpoint through the exact same oMeS harness,
  so its numbers are directly comparable to the GPT-5.5 / Opus baselines.

### Realistic expectations (S_partial, gold, from the paper)

- Frontier bar: GPT-5.5 ≈ 48 (we measured 46.9), Gemini-3.1-Pro ≈ 51.
- Paper's best fine-tuned model (Qwen-3 **4B** SFT) ≈ 20 (simple) / ~30 (ICL) —
  matches Claude-Sonnet-4.6, still well short of GPT-5.5.
- A **1B** SFT+GRPO specialist can realistically beat weak/mid baselines
  (GPT-4o 5, Sonnet-4 18, o3 28) and target Sonnet-4.6 (~30) on in-domain
  reactions; beating GPT-5.5 head-on is a stretch goal. Note in-domain scores
  run ~2× out-of-domain, so leaderboard wins may not mean general chemistry skill.

## Attribution

Benchmark data, prompts, and the oMeS algorithm are from oMeBench
(skylarkie/oMeBench, MIT). This harness is an independent API-runner wrapper.
