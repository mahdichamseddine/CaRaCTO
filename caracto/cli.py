"""Shared argparse setup for scripts that need a dataset path or Hub repo_id."""

import argparse
from pathlib import Path

import huggingface_hub


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    """--dataset_path takes precedence; otherwise download --repo_id from the Hub."""
    if args.dataset_path is not None:
        return args.dataset_path

    return Path(
        huggingface_hub.snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
        ),
    )


def get_main_parser() -> argparse.ArgumentParser:
    """Build the shared --dataset_path/--repo_id/--revision argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=Path,
        help="Path to a local copy of the dataset. Takes precedence over --repo_id.",
    )
    parser.add_argument(
        "--repo_id",
        default="dfki-av/CaRaCTO-3D",
        help="Hugging Face Hub dataset repo to use if --dataset_path is unset.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Hugging Face Hub dataset revision (branch/tag/commit) to use with "
        "--repo_id.",
    )
    return parser
