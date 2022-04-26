import torch
import torch.nn as nn
import nibabel as nib 
import numpy as np 

from hope.utils.check_axis import CheckAxis
from hope.utils.remove_zeros import remove_zeros

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)

image_path = "/mnt/HDD16TB/arams/copy_to_crai/Piotr/MS4015/brain_flair.nii.gz"
mask_path = "/mnt/HDD16TB/arams/copy_to_crai/Piotr/MS4015/pvalue.nii.gz"

image = np.array(nib.load(image_path).get_fdata())
mask = np.array(nib.load(mask_path).get_fdata())

check_axis = CheckAxis()

image, indices = remove_zeros(check_axis(image))
mask = check_axis(mask)
mask = mask[indices]

valid_idx = image.shape[0] - len(indices)

print(indices, image.shape[0])

image = torch.from_numpy(image)
mask = torch.from_numpy(mask)


mask[mask >= 0.7] = 1.0
mask[mask < 0.7] = 0.0

image = (image - 0.1564)/0.334

img_slice = image[73:76]
img_slice = img_slice.unsqueeze(dim = 0)
img_slice = img_slice.float()
mask_slice = mask[74]

predicted = model(img_slice)
print(predicted.shape)
import matplotlib.pyplot as plt 

fig, axs = plt.subplots(3)
axs[0].imshow(predicted.detach().numpy()[0,0])
axs[1].imshow(mask_slice.detach().numpy())
axs[2].imshow(image.detach().numpy()[74])
plt.savefig("test.png")