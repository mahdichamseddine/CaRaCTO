"""Interactive OpenCV crop-patch selection around the radar retroreflector."""

import sys
from typing import Any

import cv2
import numpy as np

from caracto.target_annotation.drawing import draw_rectangle


def select_patch(input_image: np.ndarray) -> np.ndarray | None:
    """Interactively select the patch containing the radar retroreflector.

    Returns the top-left/bottom-right corners of the selected patch, or None
    if cancelled.
    """

    def select_patch_callback(
        event: int,
        x: int,
        y: int,
        _flags: int,
        _param: Any | None,  # noqa: ANN401 (OpenCV callback param, genuinely untyped)
    ) -> None:
        """Handle mouse events for interactively drawing the crop rectangle."""
        # if the left mouse button was DOWN, start RECORDING
        # (x, y) coordinates and indicate that cropping is being
        nonlocal mouse_clicked, x_start, y_start, x_end, y_end

        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            mouse_clicked = True

        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if mouse_clicked:
                x_end, y_end = x, y

        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            x_end, y_end = x, y
            mouse_clicked = False  # cropping is finished

    mouse_clicked = False
    x_start, y_start, x_end, y_end = -1, -1, -1, -1

    window_name = "Crop Window"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_patch_callback)

    visualization = input_image
    visualization_updated = True  # To reduce computation when nothing is changing
    while True:
        if not mouse_clicked:
            if (
                not visualization_updated
                and x_start != -1
                and y_start != -1
                and x_end != -1
                and y_end != -1
            ):
                visualization = draw_rectangle(
                    input_image,
                    (x_start, y_start),
                    (x_end, y_end),
                    (255, 0, 0),
                    2,
                )
                visualization_updated = True
        else:
            visualization = draw_rectangle(
                input_image,
                (x_start, y_start),
                (x_end, y_end),
                (255, 0, 0),
                2,
            )
            visualization_updated = False

        cv2.imshow(window_name, visualization)
        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord("c"):  # cancel
            cv2.destroyWindow(window_name)
            return None
        if key_press == ord("q"):  # exit
            cv2.destroyWindow(window_name)
            sys.exit()
        elif key_press == ord("r"):  # restart
            x_start, y_start, x_end, y_end = -1, -1, -1, -1
            visualization = input_image
            visualization_updated = True
        elif key_press == ord("s"):  # save
            if x_start == -1 and y_start == -1 and x_end == -1 and y_end == -1:
                continue
            cv2.destroyWindow(window_name)
            break

    # Return patch corners
    return np.sort([[x_start, y_start], [x_end, y_end]], axis=0)
