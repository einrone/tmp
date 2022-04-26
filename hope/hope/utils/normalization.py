import numpy as np
import pandas as pd
from skimage.exposure import equalize_adapthist as adaptive_histogram_equalizer

from hope.utils.z_normalization import z_normalization
from hope.utils.min_max_normalization import min_max_normalization


def normalization(
    image: np.ndarray,
    image_statistics: pd.DataFrame,
    norm_mode: str = "z_norm",
    sample_mode: str = "population",
) -> np.ndarray:

    if norm_mode == "z_norm":
        if sample_mode == "population":

            std_ = image_statistics["population_std"]
            mean_ = image_statistics["population_mean"]

        elif sample_mode == "image":

            std_ = image_statistics["image_std"]
            mean_ = image_statistics["image_mean"]
            
        elif sample_mode == "brain":

            std_ = image_statistics["image_std_brain"]
            mean_ = image_statistics["image_mean_brain"] 
            
        else:

            raise ValueError(f"Norm_mode is not population brain, or image, got {sample_mode}")
            
        return z_normalization(image, mean_, std_)

    elif norm_mode == "min_max":
        min_ = image_statistics["min"]
        max_ = image_statistics["max"]

        return min_max_normalization(image, min_, max_)
    
    elif norm_mode == "adaptive_histogram":
        min_ = image_statistics["min"]
        max_ = image_statistics["max"]

        image = min_max_normalization(image, min_, max_)
        
        std_ = image_statistics["image_std"]
        mean_ = image_statistics["image_mean"]

        image = adaptive_histogram_equalizer(image)

        return z_normalization(image, mean_, std_)
    else:
        raise ValueError(f"norm_mode is not z_norm/adaptive_histogram/min_max. Got {norm_mode}")
