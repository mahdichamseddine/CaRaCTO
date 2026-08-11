"""Shared model-weight identifiers and device selection for the ML models."""

from enum import StrEnum

import torch


class ModelWeights(StrEnum):
    """Hugging Face Hub model IDs for the depth/segmentation models used."""

    # Depth Anything V2 paths
    DAV2_LARGE = "depth-anything/Depth-Anything-V2-Large-hf"
    # Segment Anything Model paths
    SAM_HUGE = "facebook/sam-vit-huge"


def get_torch_device() -> str:
    """Return the best available torch device: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
