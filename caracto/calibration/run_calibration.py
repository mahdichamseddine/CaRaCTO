"""Shared entry point for running either calibration baseline end to end."""

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from caracto.calibration.calibration_setup import CalibrationSetup
from caracto.calibration.caracto_calibration import CaRaCTOSetup, RangeMethod
from caracto.calibration.elnatour_calibration import ElNatourSetup


def run_calibration(
    calibration_path: Path,
    image_dimensions: tuple[int, int],
    x0: list[float],
    *,
    frozen_params: list[bool] | None = None,
    range_method: RangeMethod | None = None,
    simulation_std: tuple[float, float, float] | None = None,  # r, theta, px
    subset: int | None = None,
) -> tuple[CalibrationSetup, np.ndarray]:
    """Build the CaRaCTO (if range_method given) or El Natour setup and fit it."""
    if range_method is not None:
        calibration_setup = CaRaCTOSetup(
            calibration_path,
            image_dimensions,
            x0,
            range_method,
            simulation_std=simulation_std,
            subset=subset,
        )
    else:
        calibration_setup = ElNatourSetup(
            calibration_path,
            image_dimensions,
            x0,
            simulation_std=simulation_std,
            subset=subset,
        )

    ls_result = least_squares(
        calibration_setup.compute_residuals,
        calibration_setup.get_initial_guess(frozen_params),
        method="lm",
        verbose=0,
    )
    x_result = calibration_setup.check_frozen_params(ls_result["x"])

    return calibration_setup, x_result
