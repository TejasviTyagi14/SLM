"""Log a trained checkpoint to the TrueFoundry model registry (ML Repo).

Run this after GRPO (e.g. as a final job stage or locally with the SDK
authenticated) to version the specialist model and get a shareable FQN.

Example:
  python -m training.publish_model \
      --ml-repo organic-mechanism \
      --name mech-specialist-1.5b \
      --path /data/checkpoints/grpo \
      --metadata '{"s_partial_gold": 0.31, "base": "Qwen2.5-1.5B"}'
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ml-repo", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--path", required=True, help="Checkpoint folder to log.")
    ap.add_argument("--description", default="Organic-mechanism specialist (SFT+GRPO).")
    ap.add_argument("--metadata", default="{}", help="JSON string of metadata.")
    args = ap.parse_args()

    from truefoundry.ml import TransformersFramework, get_client

    client = get_client()
    model_version = client.log_model(
        ml_repo=args.ml_repo,
        name=args.name,
        description=args.description,
        model_file_or_folder=args.path,
        framework=TransformersFramework(),
        metadata=json.loads(args.metadata),
    )
    print(f"[publish] logged model version: {model_version.fqn}")


if __name__ == "__main__":
    main()
