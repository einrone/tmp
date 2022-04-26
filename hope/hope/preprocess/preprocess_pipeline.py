from typing import Union

import numpy as np
import pandas as pd
import torch

from hope.dataset.datacontainer import DatacContainer
from hope.utils import 

class PreprocessPipeline:
    def __init__(
        self,
        data_container: DatacContainer,
        normalization_type: str = "z_norm",
        split_images: bool = True,
        remove_empty_slices: bool = True,
    ):

        self.data_container = data_container
        self.normalization_type = normalization_type
        self.split_images = split_images
        self.remove_empty_slices = remove_empty_slices

    def _create_dataframe(self):
        pass

    def _remove_zeros(image: Union[np.ndarray, torch.tensor]):
        pass

    def _normalize(self):
        pass

    def _split_images(self):
        pass

    def __call__(self):
        for data in self.data_container:
            pass


