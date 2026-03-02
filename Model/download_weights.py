#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


DEFAULT_MODEL_ID  = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "Llama-3.2-11B-Vision-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LLaMA-3.2 Vision Instruct weights from HuggingFace."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model repository ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Local directory to save the downloaded weights.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "HuggingFace access token. If omitted the cached token from "
            "`huggingface-cli login` is used."
        ),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch / tag / commit to pull from (default: main).",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=["*.pt", "*.bin", "original/*"],
        help=(
            "Glob patterns to exclude from the download. "
            "Defaults to ['*.pt', '*.bin', 'original/*'] to save only safetensors."
        ),
    )
    return parser.parse_args()


def download(args: argparse.Namespace) -> Path:
    try:
        from huggingface_hub import snapshot_download, HfApi
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed.\n"
            "Run:  pip install huggingface_hub"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify the model exists and is accessible before downloading
    api = HfApi()
    try:
        info = api.model_info(args.model_id, token=args.token)
        print(f"Model  : {info.modelId}")
        print(f"Revised: {info.lastModified}")
    except Exception as exc:
        sys.exit(
            f"Could not access '{args.model_id}'.\n"
            f"Error: {exc}\n\n"
            "If this is a gated model, run:  huggingface-cli login"
        )

    print(f"\nDownloading to: {output_dir.resolve()}")
    print("This may take a while for large models (≈22 GB for 11B)...\n")

    local_path = snapshot_download(
        repo_id=args.model_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        revision=args.revision,
        token=args.token,
        ignore_patterns=args.ignore_patterns,
    )

    print(f"\nDownload complete.")
    print(f"Weights saved to: {local_path}")
    print()
    print("To load the model:")
    print()
    print("    from Model.utils import load_hf_model")
    print(f'    model, tokenizer = load_hf_model("{local_path}", device="cuda")')
    return Path(local_path)


def main() -> None:
    args = parse_args()
    download(args)


if __name__ == "__main__":
    main()
