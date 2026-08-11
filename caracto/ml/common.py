from enum import StrEnum

import torch


class ModelWeights(StrEnum):
    # Depth Anything V2 paths
    DAV2_LARGE = "depth-anything/Depth-Anything-V2-Large-hf"
    # Segment Anything Model paths
    SAM_HUGE = "facebook/sam-vit-huge"


def get_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
