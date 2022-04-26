import numpy as np


def remove_zeros(image: np.ndarray) -> np.ndarray:
    """
    This function removes empty slices
    in images, i.e images having slices
    only containing zeros.

    args:
        image : The image that is changed

    return:
        a new image with removed slices that
        only contained zeros, and list of indices
        of the nonzero slices.

    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"The image given is not a np.ndarray. Got {type(image)}")

    index_list = []
    for INDEX, slices in enumerate(image):
        if np.sum(slices) != 0:

            index_list.append(INDEX)
        else:
            pass

    image = image[index_list]
    return image, index_list


if __name__ == "__main__":
    import nibabel

    path = "/mnt/HDD16TB/arams/copy_to_crai/Piotr/MS4030/brain_flair.nii.gz"
    image = np.array(nibabel.load(path).get_fdata())
    print(image.shape)
    image, indices = remove_zeros(image)
    print(image.shape, indices)
