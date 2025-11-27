import numpy as np
import torch
from transformers import SamModel, SamProcessor
from transformers.utils.generic import TensorType

from caracto.ml.common import ModelWeights, get_torch_device


def instance_segmentation(
    image: np.ndarray,
    # transformers.SamProcessor requires input to be lists of float, however the type
    # checking here is for lists of int since we are dealing with pixel positions
    input_points: list[list[list[int]]] | None = None,
    input_boxes: list[list[list[list[int]]]] | None = None,
) -> np.ndarray:
    model = SamModel.from_pretrained(ModelWeights.SAM_HUGE)
    processor = SamProcessor.from_pretrained(ModelWeights.SAM_HUGE)
    inputs = processor(  # type: ignore
        image,
        input_points=input_points,
        input_boxes=input_boxes,
        return_tensors=TensorType.PYTORCH,
    )
    original_sizes = inputs["original_sizes"]
    reshaped_input_sizes = inputs["reshaped_input_sizes"]

    device = get_torch_device()
    model = model.to(device)  # type: ignore
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_masks = outputs.pred_masks.cpu()
    (masks,) = processor.image_processor.post_process_masks(  # type: ignore
        predicted_masks,
        original_sizes,
        reshaped_input_sizes,
    )

    return masks.numpy(force=True).squeeze()
