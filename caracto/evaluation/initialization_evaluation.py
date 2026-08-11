"""Evaluates calibration robustness to increasingly perturbed initial guesses."""

import json

import numpy as np
from tqdm import tqdm

from caracto.cli import get_main_parser, resolve_dataset_path
from caracto.common import HD_1080, MAX_EVAL_RUNS, OUTPUT_DIR, X0
from caracto.evaluation.run_evaluation import single_run

_rng = np.random.default_rng()


def main() -> None:
    """Run calibration under best/moderate/bad initial guesses and save the errors."""
    parser = get_main_parser()
    args = parser.parse_args()
    calibration_path = resolve_dataset_path(args)

    results_dict = {}

    # Best initialization
    errors_0 = single_run(calibration_path, X0, HD_1080)
    print("Best initialization:")
    for key, value in errors_0.items():
        print(f"{key} 3D error: {value['3d'][0]:.3f} ± {value['3d'][1]:.3f}")
        print(f"{key} 2D error: {value['2d'][0]:.3f} ± {value['2d'][1]:.3f}")

    errors_1_elnatour = []
    errors_1_caracto_radar = []
    errors_1_caracto_camera = []
    # Moderate initialization
    for _ in tqdm(range(MAX_EVAL_RUNS), desc="Moderate initialization"):
        x0_copy = np.array(X0.copy())
        x0_copy[0:3] += _rng.normal(0, 1, 3)  # ± 1 rad
        x0_copy[3:6] += _rng.normal(0, 0.1, 3)  # ± 0.1 m
        errors = single_run(calibration_path, list(x0_copy), HD_1080)
        errors_1_elnatour.append(errors["elnatour"])
        errors_1_caracto_radar.append(errors["caracto_radar"])
        errors_1_caracto_camera.append(errors["caracto_camera"])

    errors_2_elnatour = []
    errors_2_caracto_radar = []
    errors_2_caracto_camera = []
    # Bad initialization
    for _ in tqdm(range(MAX_EVAL_RUNS), desc="Bad initialization"):
        x0_copy = np.array(X0.copy())
        x0_copy[0:3] += _rng.normal(0, 2, 3)  # ± 2 rad
        x0_copy[3:6] += _rng.normal(0, 0.5, 3)  # ± 0.5 m
        errors = single_run(calibration_path, list(x0_copy), HD_1080)
        errors_2_elnatour.append(errors["elnatour"])
        errors_2_caracto_radar.append(errors["caracto_radar"])
        errors_2_caracto_camera.append(errors["caracto_camera"])

    results_dict["errors_0"] = errors_0
    results_dict["errors_1"] = (
        errors_1_elnatour,
        errors_1_caracto_radar,
        errors_1_caracto_camera,
    )
    results_dict["errors_2"] = (
        errors_2_elnatour,
        errors_2_caracto_radar,
        errors_2_caracto_camera,
    )

    with (OUTPUT_DIR / "initialization_evaluation.json").open("w") as f:
        json.dump(results_dict, f)


if __name__ == "__main__":
    main()
