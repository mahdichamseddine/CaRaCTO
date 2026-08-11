"""Shared constants used across calibration, evaluation, and reconstruction."""

from pathlib import Path

import numpy as np

HD_1080 = (1080, 1920)
X0 = [np.pi / 2, -np.pi / 2, 0, 0, 0, 0]

MAX_EVAL_RUNS = 250
OUTPUT_DIR = Path("outputs")
