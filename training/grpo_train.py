"""GRPO/DAPO RL fine-tuning against the verifiable oMeS reward (TRL).

Run this after SFT. It samples multiple mechanisms per reaction, scores each
with oMeS (see training/reward.py), and optimizes the policy toward higher
partial scores + validity. This is the step the oMeBench paper did NOT do and is
the main lever for a small specialist to punch above its weight.

SOTA knobs wired in (see docs/sota_features.md):
  --algo {grpo,dapo}   DAPO = token-level loss + clip-higher + dynamic sampling,
                       better for long mechanism CoT (RetroDFM-R arXiv:2507.17448).
  --reward {omes,omes+roundtrip}   multiplicative-gated oMeS, optional round-trip
                       feasibility term (RTRL arXiv:2510.01527).
  --curriculum         order reactions easy -> medium -> hard.
  --dora / --lora-r    DoRA (arXiv:2402.09353) + sweepable LoRA rank.
  validity monitoring  a callback logs reward/S_partial/validity/length and warns
                       if validity collapses (PSV-PPO arXiv:2505.00530).

Example (single GPU, DAPO)
--------------------------
  python -m training.grpo_train \
      --model checkpoints/sft-qwen1.5b-cot \
      --train training/data/sft_silver_cot_train.jsonl \
      --out   checkpoints/grpo-qwen1.5b-cot \
      --algo dapo --reward omes+roundtrip --num-generations 8 \
      --lr 1e-6 --batch 8 --grad-accum 4

Notes
-----
- The train JSONL already contains a `prompt` (chat) column and a `reference`
  column, which is exactly what GRPO + omes_reward need.
- Prefer vLLM for generation speed: add --use-vllm (requires `pip install vllm`).
- DAPO knobs require a recent TRL (`loss_type="dapo"`, `epsilon_high`,
  `mask_truncated_completions`); on older TRL they are dropped with a warning and
  the run falls back to GRPO-with-defaults for the missing knobs.
"""

from __future__ import annotations

import argparse
import json

from training.algo import ALGOS, DEFAULT_EPSILON_HIGH


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="SFT checkpoint to start from.")
    ap.add_argument("--train", required=True,
                    help="JSONL with 'prompt' (chat) and 'reference' columns.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-generations", type=int, default=8,
                    help="Samples per prompt (GRPO group size).")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--batch", type=int, default=8, help="Per-device prompt batch.")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-prompt-len", type=int, default=2048)
    ap.add_argument("--max-completion-len", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.02, help="KL coefficient.")
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--use-vllm", action="store_true")

    # --- Algorithm (Feature 2: DAPO) ---
    ap.add_argument("--algo", choices=ALGOS, default="grpo",
                    help="RL objective. dapo = token-level loss + clip-higher + "
                         "dynamic sampling (better for long CoT).")
    ap.add_argument("--epsilon-high", type=float, default=DEFAULT_EPSILON_HIGH,
                    help="DAPO clip-higher upper epsilon (only used with --algo dapo).")
    ap.add_argument("--no-dynamic-sampling", dest="dynamic_sampling",
                    action="store_false", default=True,
                    help="Disable DAPO dynamic sampling / truncated-completion masking.")

    # --- Reward (Feature 3) ---
    ap.add_argument("--reward", choices=["omes", "omes+roundtrip"], default="omes",
                    help="Reward composition. omes+roundtrip adds a forward-model "
                         "feasibility term (no-op unless a forward model is wired).")
    ap.add_argument("--w-spartial", type=float, default=None)
    ap.add_argument("--w-validity", type=float, default=None)
    ap.add_argument("--w-format", type=float, default=None)
    ap.add_argument("--w-roundtrip", type=float, default=None)

    # --- Monitoring (Feature 4) ---
    ap.add_argument("--validity-floor", type=float, default=0.8,
                    help="Warn if mean SMILES validity drops below this over a window.")
    ap.add_argument("--fail-on-validity-collapse", action="store_true",
                    help="Raise (stop training) instead of warning when below floor.")

    # --- Curriculum (Feature 5) ---
    ap.add_argument("--curriculum", action="store_true",
                    help="Order reactions easy -> medium -> hard (disables shuffle).")

    # --- LoRA / DoRA (Feature 7) ---
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--dora", action="store_true",
                    help="Use DoRA (weight-decomposed LoRA); implies --lora.")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    return ap.parse_args()


def _reward_weight_kwargs(args) -> dict:
    from training import reward as R

    return {
        "w_spartial": args.w_spartial if args.w_spartial is not None else R.W_SPARTIAL,
        "w_validity": args.w_validity if args.w_validity is not None else R.W_VALIDITY,
        "w_format": args.w_format if args.w_format is not None else R.W_FORMAT,
        "w_roundtrip": args.w_roundtrip if args.w_roundtrip is not None else R.W_ROUNDTRIP,
    }


def main() -> None:
    args = parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from training.algo import build_algo_config_kwargs, filter_supported_kwargs
    from training.callbacks import (
        MechMonitor,
        instrument_reward_funcs,
        make_monitor_callback,
    )
    from training.curriculum import level_histogram, order_by_difficulty
    from training.reward import build_reward_funcs

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files={"train": args.train})["train"]

    # Curriculum ordering must happen BEFORE we drop the 'level' column, and
    # requires shuffling to be off so the order survives into training.
    shuffle = True
    if args.curriculum:
        rows = order_by_difficulty(list(ds))
        print(f"[curriculum] level histogram: {json.dumps(level_histogram(rows))}")
        from datasets import Dataset

        ds = Dataset.from_list(rows)
        shuffle = False

    # GRPO needs 'prompt' and any reward-func kwargs columns (here 'reference').
    keep = {"prompt", "reference"}
    drop_cols = [c for c in ds.column_names if c not in keep]
    ds = ds.remove_columns(drop_cols)

    peft_config = None
    if args.lora or args.dora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
            use_dora=bool(args.dora),
        )

    # Reward functions (Feature 3) + validity monitor (Feature 4).
    reward_funcs = build_reward_funcs(reward=args.reward, **_reward_weight_kwargs(args))
    monitor = MechMonitor(validity_floor=args.validity_floor)
    reward_funcs = instrument_reward_funcs(reward_funcs, monitor)
    monitor_cb = make_monitor_callback(monitor, raise_on_floor=args.fail_on_validity_collapse)

    base_cfg = dict(
        output_dir=args.out,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
        temperature=args.temperature,
        beta=args.beta,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        use_vllm=args.use_vllm,
        shuffle_dataset=shuffle,
        report_to="none",
    )
    # Algorithm knobs (Feature 2), filtered to what the installed TRL supports.
    algo_kwargs = build_algo_config_kwargs(
        algo=args.algo,
        epsilon_high=args.epsilon_high,
        dynamic_sampling=args.dynamic_sampling,
        mask_truncated=args.dynamic_sampling,
    )
    base_cfg = filter_supported_kwargs({**base_cfg, **algo_kwargs}, GRPOConfig)
    cfg = GRPOConfig(**base_cfg)
    print(f"[grpo] algo={args.algo} reward={args.reward} curriculum={args.curriculum} "
          f"dora={args.dora}")

    trainer = GRPOTrainer(
        model=args.model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        peft_config=peft_config,
        callbacks=[monitor_cb],
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[grpo] saved to {args.out}")


if __name__ == "__main__":
    main()
