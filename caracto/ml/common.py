from enum import Enum

import torch


class ModelWeights(str, Enum):
    # Depth Anything V2 paths
    DAV2_LARGE = "depth-anything/Depth-Anything-V2-Large-hf"
    # Segment Anything Model paths
    SAM_HUGE = "facebook/sam-vit-huge"

    def __str__(self):
        return str(self.value)


def get_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
