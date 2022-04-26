import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hope.dataset.imagecontainer import ImageContainer
from hope.utils.check_axis import CheckAxis
from hope.utils.remove_zeros import remove_zeros
from hope.utils.normalization import normalization
from hope.utils.imagestatistics import ImageStatistics


def analyze_images(PATH: str, container:ImageContainer, statistics: pd.DataFrame)-> None:
    checkaxis = CheckAxis()
    
    for container_idx, data in enumerate(container):
        patient_id = list(data.keys())[0]
        
        mask = checkaxis(data[patient_id]["pvalue"])
        image = checkaxis(data[patient_id]["flair"])

        image, non_zero_slices_idx = remove_zeros(image)
        mask = mask[non_zero_slices_idx]
        zeros_slices_idx = image.shape[0] - len(non_zero_slices_idx)
        valid_idx = image.shape[0] - zeros_slices_idx
        
        image_statistics = statistics.loc[patient_id]

        
        normed_image = normalization(image,image_statistics, "z_norm", "image" )
        print(f"normed_image std: {normed_image.std()} and mean: {normed_image.mean()}")
        if mask.shape == image.shape:
            for idx in range(valid_idx):
                fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
                fig.suptitle(f"{patient_id}: image and mask")

                axs[0,0].imshow(image[idx])#, alpha = 0.5)
                axs[0,0].imshow(mask[idx], alpha = 0.5)
                fig.tight_layout()
                
                axs[0,1].imshow(mask[idx])
                fig.tight_layout()

                axs[1,0].imshow(image[idx])
                fig.tight_layout()

                axs[1,1].imshow(normed_image[idx])
                plt.savefig(PATH + "/image_" + patient_id + "_" + str(idx) + ".png")

                plt.close()
                #exit()
        if container_idx == 10:
            exit()
        else:
            pass 

if __name__ == "__main__":
    generate_img = ImageContainer(
        "/mnt/HDD16TB/arams/copy_to_crai/Piotr/",
        "MS",
        ["flair", "pvalue"],
    )
    img_statistics = ImageStatistics(generate_img, image_type="flair")
    stat_df = img_statistics()
    print(stat_df)
    print(stat_df["max"].to_dict())

    exit()
    PATH = "/mnt/HDD16TB/arams/hope/hope/dataset/check_dataset"

    analyze_images(PATH, generate_img, stat_df)