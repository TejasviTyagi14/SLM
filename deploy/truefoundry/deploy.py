"""Deploy the mechanism-specialist training pipeline as TrueFoundry Jobs.

The pipeline has four stages, each deployed as its own Job and chained through a
shared persistent Volume mounted at /data:

    build   (CPU)  -> /data/sft/*.jsonl
    sft     (GPU)  -> /data/checkpoints/sft
    grpo    (GPU)  -> /data/checkpoints/grpo
    eval    (GPU)  -> prints oMeS leaderboard to job logs

Prereqs (one-time, see README "Running on TrueFoundry"):
  pip install truefoundry
  tfy login --host <your-truefoundry-host>
  # create a Workspace, a Volume, and (optional) an ML Repo in the dashboard.

Deploy a stage:
  python deploy/truefoundry/deploy.py --stage build \
      --workspace-fqn <ws-fqn> --volume-fqn <vol-fqn>
  python deploy/truefoundry/deploy.py --stage sft  --workspace-fqn ... --volume-fqn ...
  python deploy/truefoundry/deploy.py --stage grpo --workspace-fqn ... --volume-fqn ...
  python deploy/truefoundry/deploy.py --stage eval --workspace-fqn ... --volume-fqn ...

Each `deploy(...)` registers/updates the Job; trigger runs from the dashboard or
with `tfy trigger job --application-fqn <fqn>`.
"""

from __future__ import annotations

import argparse

from truefoundry.deploy import (
    Build,
    GPUType,
    Job,
    LocalSource,
    NodeSelector,
    Param,
    PythonBuild,
    Resources,
    VolumeMount,
)

# Where the shared volume is mounted inside every job.
DATA = "/data"
SFT_DATA = f"{DATA}/sft"
SFT_CKPT = f"{DATA}/checkpoints/sft"
GRPO_CKPT = f"{DATA}/checkpoints/grpo"

# Base image deps. torch/trl/etc come from requirements-train.txt; the eval SDKs
# and rdkit are added explicitly so the same image can run every stage.
PIP_PACKAGES = [
    "rdkit>=2024.3.1",
    "anthropic>=0.40.0",
    "openai>=1.50.0",
]

# --- per-stage command templates (params are filled by TrueFoundry at run time) ---
COMMANDS = {
    "build": (
        "python -m training.build_sft_data "
        "--dataset silver --style cot "
        f"--out-dir {SFT_DATA} --val-frac 0.03"
    ),
    "sft": (
        "python -m training.sft_train "
        "--model {{base_model}} "
        f"--train {SFT_DATA}/sft_silver_cot_train.jsonl "
        f"--val {SFT_DATA}/sft_silver_cot_val.jsonl "
        f"--out {SFT_CKPT} "
        "--epochs {{epochs}} --batch {{batch}} --grad-accum {{grad_accum}}"
    ),
    "grpo": (
        "python -m training.grpo_train "
        f"--model {SFT_CKPT} "
        f"--train {SFT_DATA}/sft_silver_cot_train.jsonl "
        f"--out {GRPO_CKPT} "
        "--num-generations {{num_generations}} --batch {{batch}} --use-vllm"
    ),
    "eval": (
        "python -m omebench_eval.cli run --models specialist "
        f"--provider local --model-id {GRPO_CKPT} "
        "--dataset gold --prompt cot --concurrency 1 --max-tokens 4096 && "
        "python -m omebench_eval.cli report --dataset gold"
    ),
}

# Stages that need a GPU (build is CPU-only).
GPU_STAGES = {"sft", "grpo", "eval"}


def build_job(stage: str, args: argparse.Namespace) -> Job:
    is_gpu = stage in GPU_STAGES

    if is_gpu:
        resources = Resources(
            cpu_request=4,
            cpu_limit=8,
            memory_request=32000,
            memory_limit=64000,
            ephemeral_storage_request=50000,
            ephemeral_storage_limit=100000,
            shared_memory_size=16000,
            gpu_count=1,
            node=NodeSelector(gpu_type=getattr(GPUType, args.gpu)),
        )
    else:
        resources = Resources(
            cpu_request=2,
            cpu_limit=4,
            memory_request=8000,
            memory_limit=16000,
            ephemeral_storage_request=10000,
            ephemeral_storage_limit=20000,
        )

    # Params exposed on the Job so runs can be re-parametrized without redeploying.
    params = []
    if stage == "sft":
        params = [
            Param(name="base_model", default=args.base_model),
            Param(name="epochs", default="3"),
            Param(name="batch", default="8"),
            Param(name="grad_accum", default="2"),
        ]
    elif stage == "grpo":
        params = [
            Param(name="num_generations", default="8"),
            Param(name="batch", default="8"),
        ]

    env = {
        "HF_HOME": f"{DATA}/hf-cache",
        "TOKENIZERS_PARALLELISM": "false",
        # Uncomment if using gated base models / needed for downloads:
        # "HF_TOKEN": "tfy-secret://<org>:<collection>:HF_TOKEN",
    }
    # The eval stage can also benchmark API models in the same run if you add keys:
    # env["OPENAI_API_KEY"] = "tfy-secret://<org>:<collection>:OPENAI_API_KEY"

    return Job(
        name=f"mech-{stage}",
        image=Build(
            build_source=LocalSource(local_build=False),
            build_spec=PythonBuild(
                python_version="3.11",
                cuda_version="12.4.1" if is_gpu else None,
                requirements_path="requirements-train.txt",
                pip_packages=PIP_PACKAGES,
                command=COMMANDS[stage],
            ),
        ),
        params=params,
        env=env,
        resources=resources,
        mounts=[VolumeMount(mount_path=DATA, volume_fqn=args.volume_fqn)],
        retries=0,
        timeout=args.timeout,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=list(COMMANDS))
    ap.add_argument("--workspace-fqn", required=True)
    ap.add_argument("--volume-fqn", required=True,
                    help="e.g. tfy-volume://<cluster>:<workspace>:<volume>")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gpu", default="A100_80GB",
                    help="GPUType name, e.g. A100_80GB, A100_40GB, A10G, T4.")
    ap.add_argument("--timeout", type=int, default=86400)
    args = ap.parse_args()

    job = build_job(args.stage, args)
    job.deploy(workspace_fqn=args.workspace_fqn)


if __name__ == "__main__":
    main()
