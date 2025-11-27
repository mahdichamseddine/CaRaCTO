from pathlib import Path

import numpy as np
import numpy.typing as npt

from caracto.calibration.calibration_setup import CalibrationSetup

"""
@article{pervsic2019extrinsic,
    title     = {Extrinsic 6dof calibration of a radar--lidar--camera system enhanced by radar cross section estimates evaluation},
    author    = {Peršić, Juraj and Marković, Ivan and Petrović, Ivan},
    journal   = {Robotics and Autonomous Systems},
    volume    = {114},
    pages     = {217--230},
    year      = {2019},
    publisher = {Elsevier},
}
"""


class PersicSetup(CalibrationSetup):
    def __init__(
        self,
        calibration_path: Path,
        image_dimensions: tuple[int, int],
        x0: list[float] | npt.NDArray[np.float64],
        simulation_std: tuple[float, float, float] | None = None,  # r, theta, px
        subset: int | None = None,
    ) -> None:
        super().__init__(calibration_path, image_dimensions, x0, simulation_std, subset)
        raise NotImplementedError(
            "Requires Radar Cross Section (RCS) which is not available from TinyRad."
        )
