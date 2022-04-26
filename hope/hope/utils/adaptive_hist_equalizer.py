import numpy as np 
import pandas as pd
from skimage import exposure.equalize_adapthist as adaptive_histogram_equalizer

from hope.utils.normalization import normalization

def adaptive_hist_equalizer(
    image: np.ndarray,
    image_statistics: pd.DataFrame,
    )-> np.ndarray:
    
    return adaptive_histogram_equalizer(image)