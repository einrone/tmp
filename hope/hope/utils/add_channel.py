import torch
import numpy as np
from typing import Union


class AddChannel:
    def __init__(self, num_in_channels=5, this_axis=0):
        if not isinstance(this_axis, int):
            raise ValueError("Please give a integer value for the axis")

        if not isinstance(num_in_channels, int):
            raise ValueError("Please give a integer value for num_in_channels")

        self.num_in_channels = num_in_channels
        self.this_axis = this_axis

    def __call__(image: Union[np.ndarray, torch.Tensor]):
        for channel_in in range(-self.num_in_channels, self.num_in_channels):
            x = 0

        if type(image) == torch.tensor:
            image = image.unsqueeze(axis=self.this_axis)
