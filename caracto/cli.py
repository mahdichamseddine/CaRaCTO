import argparse
from pathlib import Path


def get_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=Path,
        help="Path to the dataset.",
    )
    return parser
