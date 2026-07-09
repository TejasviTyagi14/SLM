"""Training pipeline for an organic-mechanism specialist model.

Stages:
1. build_sft_data  -> RDKit-validated chat SFT data from oMe-Silver
2. sft_train       -> supervised fine-tuning (format + patterns)
3. grpo_train      -> RL against the verifiable oMeS reward
Then evaluate with the `omebench_eval` harness.
"""
