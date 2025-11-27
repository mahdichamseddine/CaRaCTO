import cv2
import numpy as np


def enhance_image(input_image: np.ndarray, area_corners: np.ndarray):  # TODO
    area_mask = np.ones_like(input_image)
    cv2.rectangle(
        img=area_mask,
        pt1=(area_corners[0, 0], area_corners[0, 1]),
        pt2=(area_corners[1, 0], area_corners[1, 1]),
        color=(255, 255, 255),
        thickness=-1,
    )

    output_image = cv2.bilateralFilter(input_image, 10, 15, 30)
    """
    lab = cv2.cvtColor(output_image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    output_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    """
    alpha = 1.0  # Simple contrast control
    beta = 0  # Simple brightness control
    gamma = 1.5
    output_image = cv2.convertScaleAbs(output_image, alpha=alpha, beta=beta)
    look_up_table = np.empty((1, 256), np.uint8)
    for i in range(256):
        look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    output_image = cv2.LUT(output_image, look_up_table)
    output_image = cv2.bitwise_and(output_image, area_mask)
    return output_image
