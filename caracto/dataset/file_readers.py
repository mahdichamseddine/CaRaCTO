"""File readers for each format used by the raw/published dataset."""

import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import yaml


def read_file(file_path: Path) -> dict:
    """Dispatch to the reader matching file_path's suffix."""
    match file_path.suffix:
        case ".yaml":
            return read_yaml_file(file_path)
        case ".json":
            return read_json_file(file_path)
        case ".pickle":
            return read_pickle_file(file_path)
        case ".npz":
            return read_npz_file(file_path)
        case _:
            msg = "File format loader not implemented"
            raise NotImplementedError(msg)


def read_yaml_file(file_path: Path) -> dict:
    """Read a YAML file into a dict."""
    with (file_path).open() as f:
        return yaml.safe_load(f)


def read_json_file(file_path: Path) -> dict:
    """Read a JSON file into a dict."""
    with (file_path).open() as f:
        return json.load(f)


def read_pickle_file(file_path: Path) -> dict:
    """Read a pickle file into a dict."""
    with (file_path).open("rb") as f:
        # Only ever pointed at raw capture files produced by this project's own tooling
        # (e.g. camera_annotation.py annotating a new session), never untrusted input.
        return pickle.load(f)  # noqa: S301


def read_npz_file(file_path: Path) -> dict:
    """Read an NPZ archive into a dict of arrays."""
    with np.load(file_path) as npz_data:
        return dict(npz_data)


def read_image_file(file_path: Path) -> npt.NDArray[np.uint8]:
    """Read an image file into a uint8 array."""
    image = cv2.imread(str(file_path))
    if image is None:
        msg = f"Could not read image file: {file_path}"
        raise FileNotFoundError(msg)
    return image.astype(np.uint8)
