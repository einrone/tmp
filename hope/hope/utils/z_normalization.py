import numpy as np


def z_normalization(image: np.ndarray, img_mean: float, img_std: float) -> np.ndarray:
    eps = 1e-7
    return (image - img_mean + eps) / (img_std + eps)
