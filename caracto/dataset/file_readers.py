import json
import pickle
from pathlib import Path

import yaml


def read_file(file_path: Path) -> dict:
    match file_path.suffix:
        case ".yaml":
            return read_yaml_file(file_path)
        case ".json":
            return read_json_file(file_path)
        case ".pickle":
            return read_pickle_file(file_path)
        case _:
            raise NotImplementedError("File format loader not implemented")


def read_yaml_file(file_path: Path) -> dict:
    with open(file_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    return yaml_data


def read_json_file(file_path: Path) -> dict:
    with open(file_path, "r") as f:
        json_data = json.load(f)

    return json_data


def read_pickle_file(file_path: Path) -> dict:
    with open(file_path, "rb") as f:
        pickle_data = pickle.load(f)

    return pickle_data
