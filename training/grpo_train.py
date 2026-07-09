"""GRPO RL fine-tuning against the verifiable oMeS reward (TRL).

Run this after SFT. It samples multiple mechanisms per reaction, scores each
with oMeS (see training/reward.py), and optimizes the policy toward higher
partial scores + validity. This is the step the oMeBench paper did NOT do and is
the main lever for a small specialist to punch above its weight.

Example (single GPU)
--------------------
  python -m training.grpo_train \
      --model checkpoints/sft-qwen1.5b-cot \
      --train training/data/sft_silver_cot_train.jsonl \
      --out   checkpoints/grpo-qwen1.5b-cot \
      --num-generations 8 --lr 1e-6 --batch 8 --grad-accum 4

Notes
-----
- The train JSONL already contains a `prompt` (chat) column and a `reference`
  column, which is exactly what GRPO + omes_reward need.
- Prefer vLLM for generation speed: add --use-vllm (requires `pip install vllm`).
"""

from __future__ import annotations

import argparse


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
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from training.reward import format_reward, omes_reward

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files={"train": args.train})["train"]
    # GRPO needs 'prompt' and any reward-func kwargs columns (here 'reference').
    keep = {"prompt", "reference"}
    drop_cols = [c for c in ds.column_names if c not in keep]
    ds = ds.remove_columns(drop_cols)

    peft_config = None
    if args.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    cfg = GRPOConfig(
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
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        reward_funcs=[omes_reward, format_reward],
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[grpo] saved to {args.out}")


if __name__ == "__main__":
    main()
