import numpy as np


def slice_and_concat(image: np.ndarray, mask: np.ndarray, idx: int, num_channel: int) -> np.ndarray:
    """
    This function slices and concatenates
    neighbouring slices. The amount of
    neighbouring slices are determined by
    the number of channels, which
    is a parameter.

    args:
        image: Image to be sliced
        idx: the current index slice
        num_channel: Number of channels

    return:
        An image with reduced number of channels,
        and only contains desired neighbours. Middle
        channel is the focus.
    """

    #fix this to support num_channels, because 3 channels are hardcoded down below
    neighbour_idx = num_channel // 2
    ret = image.copy()
    ret_mask = mask.copy()
    if idx == 0:

        first_slice = ret[idx]
        current_slice = ret[idx]
        next_slice = ret[idx + neighbour_idx]
        current_mask_slice = ret_mask[idx]

        new_image = np.concatenate(
            [
                first_slice[np.newaxis],
                current_slice[np.newaxis],
                next_slice[np.newaxis],
                current_mask_slice[np.newaxis]
            ],
            axis=0,
        )

    elif idx == image.shape[0] - 1:

        previous_slice = ret[idx - neighbour_idx]
        current_slice = ret[idx]
        last_slice = ret[idx]
        current_mask_slice = ret_mask[idx]

        new_image = np.concatenate(
            [
                previous_slice[np.newaxis],
                current_slice[np.newaxis],
                last_slice[np.newaxis],
                current_mask_slice[np.newaxis]
            ],
            axis=0,
        )

    else:

        previous_slice = ret[idx - neighbour_idx]
        current_slice = ret[idx]
        next_slice = ret[idx + neighbour_idx]
        current_mask_slice = mask[idx]

        new_image = np.concatenate(
            [
                previous_slice[np.newaxis],
                current_slice[np.newaxis],
                next_slice[np.newaxis],
                current_mask_slice[np.newaxis]
            ],
            axis=0,
        )
    return new_image
