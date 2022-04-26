from pathlib import Path
import torch
import numpy as np
"""import matplotlib.pyplot as plt
import numpy as np
import h5py
import copy"""
"""from hope.utils.z_normalization import z_normalization
from hope.utils.remove_zeros import remove_zeros
"""
"""path = Path(
    "/mnt/HDD16TB/arams/hope/hope/dataset/FLAIR_dataset/train/MS4010_flair_57.hdf5"
)
# patient_images = list(path.glob("*"))
# print(len(patient_images))
img_path = path  # / patient_images[0]
# print(img_path)

with h5py.File(img_path, "r") as f:
    image = f.get("flair")[:]
    print(image.shape)
    mask = f.get("pvalue")[:]
    print(mask.shape)
    id_value = f.get("id")[()].decode()
    slice_idx = f.get("slice_number")[()]

print(id_value, slice_idx)
image = np.array(image)
print(np.unique(image[0] - image[1]))
"""
import nibabel

path = "/mnt/HDD16TB/arams/copy_to_crai/Piotr/MS4061/brain_flair.nii.gz"
image = np.array(nibabel.load(path).get_fdata())
image = (image - 0.144983)/0.335272
print(np.max(image))

#model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#    in_channels=3, out_channels=1, init_features=32, pretrained=True)
# image, idx = remove_zeros(image)
# image = z_normalization(image, 0.1561, 0.34)
# print(image[70].max(), image[71].max())
#
# find out how slices are connected to each other to make a test
#
#
"""for idx, img in enumerate(image):
    plt.subplot(211)
    plt.imshow(img)
    plt.imshow(mask, cmap="Reds", alpha=0.5)
    plt.subplot(212)
    plt.imshow(img)
    plt.savefig(f"/mnt/HDD16TB/arams/hope/hope/dataset/test/{idx}")
"""
