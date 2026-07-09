"""Supervised fine-tuning of a small base model on oMeBench SFT data (TRL).

This teaches the output format and mechanistic patterns. Run GRPO afterwards to
push quality against the oMeS reward.

Example (single GPU, full fine-tune of a 1-1.5B model)
------------------------------------------------------
  python -m training.sft_train \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --train training/data/sft_silver_cot_train.jsonl \
      --val   training/data/sft_silver_cot_val.jsonl \
      --out   checkpoints/sft-qwen1.5b-cot \
      --epochs 3 --lr 1e-5 --batch 8 --grad-accum 2

Add --lora for parameter-efficient tuning on smaller GPUs.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="HF base model. Pick a strong ~1B instruct model.")
    ap.add_argument("--train", required=True, help="Train JSONL (chat 'messages').")
    ap.add_argument("--val", default=None, help="Optional validation JSONL.")
    ap.add_argument("--out", required=True, help="Output checkpoint dir.")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8, help="Per-device batch size.")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--lora", action="store_true", help="Use LoRA (PEFT).")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Heavy imports live here so the module stays importable without a GPU stack.
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = {"train": args.train}
    if args.val:
        data_files["validation"] = args.val
    ds = load_dataset("json", data_files=data_files)

    # Keep only the chat 'messages' column for SFT.
    keep = "messages"
    for split in ds:
        drop_cols = [c for c in ds[split].column_names if c != keep]
        ds[split] = ds[split].remove_columns(drop_cols)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="bfloat16" if args.bf16 else "auto",
        trust_remote_code=True,
    )

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

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_len,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="steps" if args.val else "no",
        eval_steps=args.save_steps if args.val else None,
        report_to="none",
        packing=False,
        assistant_only_loss=True,  # only train on assistant tokens
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[sft] saved to {args.out}")


if __name__ == "__main__":
    main()
