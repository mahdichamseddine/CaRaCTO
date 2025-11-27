import json
from pathlib import Path

from tqdm import tqdm

from caracto.cli import get_main_parser
from caracto.common import HD_1080, MAX_EVAL_RUNS, OUTPUT_DIR, X0
from caracto.evaluation.run_evaluation import single_run


def levels_simulation(
    calibration_path: Path,
    x0: list[float],
    image_dimensions: tuple[int, int],
    n_runs: int,
    n_levels: int = 10,
):
    results_dict = {}
    for level in range(n_levels + 1):
        results_dict[level] = []
        for _ in tqdm(range(n_runs), desc=f"Level - {level:02d}"):
            results_dict[level].append(
                single_run(
                    calibration_path,
                    x0,
                    image_dimensions,
                    simulation_std=(0.05 * level, level / 100, level),
                )
            )

    with open(OUTPUT_DIR / "levels_simulation.json", "w") as f:
        json.dump(results_dict, f)


def range_simulation(
    calibration_path: Path,
    x0: list[float],
    image_dimensions: tuple[int, int],
    n_runs: int,
    n_levels: int = 10,
):
    results_dict = {}
    for level in range(n_levels + 1):
        results_dict[level] = []
        for _ in tqdm(range(n_runs), desc=f"Range - {level:02d}"):
            results_dict[level].append(
                single_run(
                    calibration_path,
                    x0,
                    image_dimensions,
                    simulation_std=(0.05 * level, 0, 0),
                )
            )

    with open(OUTPUT_DIR / "range_simulation.json", "w") as f:
        json.dump(results_dict, f)


def azimuth_simulation(
    calibration_path: Path,
    x0: list[float],
    image_dimensions: tuple[int, int],
    n_runs: int,
    n_levels: int = 10,
):
    results_dict = {}
    for level in range(n_levels + 1):
        results_dict[level] = []
        for _ in tqdm(range(n_runs), desc=f"Azimuth - {level:02d}"):
            results_dict[level].append(
                single_run(
                    calibration_path,
                    x0,
                    image_dimensions,
                    simulation_std=(0, level / 100, 0),
                )
            )

    with open(OUTPUT_DIR / "azimuth_simulation.json", "w") as f:
        json.dump(results_dict, f)


def pixel_simulation(
    calibration_path: Path,
    x0: list[float],
    image_dimensions: tuple[int, int],
    n_runs: int,
    n_levels: int = 10,
):
    results_dict = {}
    for level in range(n_levels + 1):
        results_dict[level] = []
        for _ in tqdm(range(n_runs), desc=f"Pixel - {level:02d}"):
            results_dict[level].append(
                single_run(
                    calibration_path,
                    x0,
                    image_dimensions,
                    simulation_std=(0, 0, level),
                )
            )

    with open(OUTPUT_DIR / "pixel_simulation.json", "w") as f:
        json.dump(results_dict, f)


def main() -> None:
    parser = get_main_parser()
    args = parser.parse_args()
    calibration_path = args.dataset_path

    # Potentially could run each in a separate process
    levels_simulation(calibration_path, X0, HD_1080, MAX_EVAL_RUNS)
    range_simulation(calibration_path, X0, HD_1080, MAX_EVAL_RUNS)
    azimuth_simulation(calibration_path, X0, HD_1080, MAX_EVAL_RUNS)
    pixel_simulation(calibration_path, X0, HD_1080, MAX_EVAL_RUNS)


if __name__ == "__main__":
    main()
