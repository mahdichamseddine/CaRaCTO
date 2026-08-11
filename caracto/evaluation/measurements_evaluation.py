"""Evaluates calibration accuracy as the number of measurement positions varies."""

import json

from tqdm import tqdm

from caracto.cli import get_main_parser, resolve_dataset_path
from caracto.common import HD_1080, MAX_EVAL_RUNS, OUTPUT_DIR, X0
from caracto.evaluation.run_evaluation import single_run


def main() -> None:
    """Run calibration on increasing measurement subset sizes and save the errors."""
    parser = get_main_parser()
    args = parser.parse_args()
    calibration_path = resolve_dataset_path(args)

    results_dict = {}

    for s in range(1, 37):
        results_dict[s] = []
        for _ in tqdm(range(MAX_EVAL_RUNS), desc=f"{s:02d} measuremets"):
            try:
                results_dict[s].append(
                    single_run(
                        calibration_path,
                        X0,
                        HD_1080,
                        simulation_std=None,
                        subset=s,
                    ),
                )
            except ValueError:
                break

    with (OUTPUT_DIR / "measurements_evaluation.json").open("w") as f:
        json.dump(results_dict, f)


if __name__ == "__main__":
    main()
