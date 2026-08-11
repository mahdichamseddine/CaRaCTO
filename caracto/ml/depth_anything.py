"""Depth estimation via the Depth Anything V2 model."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 (universal PyTorch convention)
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers.utils.generic import TensorType

from caracto.ml.common import ModelWeights, get_torch_device


def depth_estimation(
    image: np.ndarray,
    border_size: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (depth_map, disparity) for image, optionally padded by border_size."""
    model = AutoModelForDepthEstimation.from_pretrained(ModelWeights.DAV2_LARGE)
    image_processor = AutoImageProcessor.from_pretrained(ModelWeights.DAV2_LARGE)
    if border_size > 0:
        augmented_image = cv2.copyMakeBorder(
            image,
            border_size,
            0,
            border_size,
            border_size,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    else:
        augmented_image = image
    inputs = image_processor(images=augmented_image, return_tensors=TensorType.PYTORCH)

    device = get_torch_device()
    model = model.to(device)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Interpolate predicted depth to original size
    predicted_depth = outputs.predicted_depth.cpu()
    disparity = F.interpolate(
        predicted_depth.unsqueeze(1),
        size=augmented_image.shape[0:2],
        mode="bicubic",
        align_corners=False,
    )

    # Normalize the depth
    lower = torch.min(disparity)
    upper = torch.max(disparity)
    normalized_disparity = (disparity - lower) / (upper - lower)
    normalized_disparity = normalized_disparity.numpy(force=True).squeeze()
    depth_map = -1 * normalized_disparity + 1
    h, w = depth_map.shape

    disparity = disparity.numpy(force=True).squeeze()

    return (
        depth_map[border_size:h, border_size : w - border_size],
        disparity[border_size:h, border_size : w - border_size],
    )
