import numpy as np


def min_max_normalization(
    image: np.ndarray, img_min: float, img_max: float
) -> np.ndarray:
    return (image - img_min) / (img_max - img_min)
