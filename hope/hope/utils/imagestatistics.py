import numpy as np
import pandas as pd
import copy

from collections import defaultdict
from hope.dataset.imagecontainer import ImageContainer
from hope.utils.remove_zeros import remove_zeros
from hope.utils.stats import sum_of_data
from hope.utils.check_axis import CheckAxis
from hope.utils.swap_brain_plane import swap_brain_plane
from hope.utils.equalize_dimension import equalize_dimension

class ImageStatistics:
    def __init__(
        self,
        container: ImageContainer,
        image_type: str,
        mode: str = "population",
        remove_zero_slices: bool = True,
    ) -> None:

        if mode in ["population", "image", "channel"]:
            self.mode = mode
        else:
            raise ValueError(f"mode is not population/image/channel. Got {mode}")

        self.image_type = image_type
        self.tmp_save = defaultdict(dict)
        self.container = container
        self.remove_zero_slices = remove_zero_slices
        self.checkaxis = CheckAxis()

    def __call__(
        self, 
        brain_axis,  
        equalize_dimension_in_image: bool = False,
        save_statistics: bool = False
        ) -> pd.DataFrame:
        """
        A method calculating the statistics of the image dataset.
        The statistics are mean, std, min, max, expaction values up to
        order 2.

        args:
            save_statistics: (default False) Save the dataframe
            to a csv file

        return:
            a pandas dataframe with the statistics, each
            row are patient id, and each column is a statistical
            variable.
        """
        total_pixel_for_each_image = []

        for files in self.container:

            id_value = list(files.keys())[0]

            image = self.checkaxis(files[id_value][self.image_type])
            
            if equalize_dimension_in_image == True:
                image = equalize_dimension(image)

            image = swap_brain_plane(image, brain_axis)

            if self.remove_zero_slices == True:
                image, indices = remove_zeros(image)
            else:
                pass

            total_pixel_for_each_image.append(np.prod(image.shape))

            self.tmp_save[id_value]["sum_of_data_squared"] = sum_of_data(image, order=2)
            self.tmp_save[id_value]["sum_of_data"] = sum_of_data(image, order=1)

            self.tmp_save[id_value]["image_std"] = image.std()
            self.tmp_save[id_value]["image_mean"] = image.mean()

            self.tmp_save[id_value]["image_std_brain"] = image[image != 0].std()
            self.tmp_save[id_value]["image_mean_brain"] = image[image!=0].mean()

            self.tmp_save[id_value]["max_brain"] = image[image!=0].max()
            self.tmp_save[id_value]["min_brain"] = image[image!=0].min()
            
            self.tmp_save[id_value]["max"] = image.max()
            self.tmp_save[id_value]["min"] = image.min()

        df = pd.DataFrame.from_dict(self.tmp_save, orient="index")

        if self.mode == "population":
            df["population_std"] = np.sqrt(
                (df["sum_of_data_squared"].sum()/sum(total_pixel_for_each_image) - (df["sum_of_data"].sum()/sum(total_pixel_for_each_image))**2) 
            )
            df["population_mean"] = df["sum_of_data"].sum() / sum(total_pixel_for_each_image)  # <- spør gunnar

        else:
            pass

        if save_statistics == True:
            pass
            # future implementation
        return df

    def visualize_statistics(self, save_plot=False) -> None:
        """
        A method that visualizes the statistics of the image dataset

        args:
            save_plot: (default False) save the image plot of the statistics

        return:
            None
        """
        pass


if __name__ == "__main__":
    preprocess = ImageStatistics(
        container=ImageContainer(
            "/mnt/HDD16TB/arams/copy_to_crai/Piotr/", "MS", ["flair", "pvalue"]
        ),
        image_type="flair",
        mode="population",
    )

    print(preprocess())
