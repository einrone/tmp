import copy
from pathlib import Path
from typing import List, Union

import h5py
import pandas as pd
import torch
import numpy as np

from hope.dataset.imagecontainer import ImageContainer
from hope.utils.imagestatistics import ImageStatistics
from hope.utils.normalization import normalization
from hope.utils.remove_zeros import remove_zeros
from hope.utils.slice_and_concat import slice_and_concat
from hope.utils.check_axis import CheckAxis


class DistributeFiles:
    def __init__(
        self,
        path: Path,
        container: ImageContainer,
        statistics: pd.DataFrame,
        files_to_distribute: List[str],
        treshold: int,
        preprocess: Union[list, tuple] = None,
        volume: bool = False,
        norm_mode: str = "z_norm",
        sample_mode: str = "population",
        num_channel: int = 3,
        remove_zero_slices: bool = True,
    ) -> None:

        self.path = Path(path)
        if not self.path.is_dir():
            self.path.mkdir()
        else:
            if list(self.path.glob("*")):
                raise ValueError(
                    "The path of directory given is not empty, exiting program to prevent overwriting files"
                )
            else:
                pass

        self.norm_mode = norm_mode
        self.sample_mode = sample_mode

        self.files_to_distribute = files_to_distribute
        if len(self.files_to_distribute) < 2 or len(self.files_to_distribute) > 2:
            raise ValueError(
                """files to distribute has to be 2, image and its mask. 
                   It has to be same as files to extract in ImageContainer class
                """
            )
        else:
            pass

        self.treshold = treshold

        self.container = container
        self.statistics = statistics
        self.volume = volume
        self.num_channel = num_channel
        self.remove_zero_slices = remove_zero_slices
        self.preprocess = preprocess
        self.checkaxis = CheckAxis()

        print(f"Using normalization mode :{self.norm_mode} and using sample: {self.sample_mode} volumes")

    def split_and_distribute(self) -> None:
        num_slices = 0
        for files in self.container:

            id_value = list(files.keys())[0]

            image = self.checkaxis(files[id_value][self.files_to_distribute[0]])
            mask = self.checkaxis(files[id_value][self.files_to_distribute[1]])

            image_statistics = self.statistics.loc[id_value]

            if self.remove_zero_slices == True:
                image, indices = remove_zeros(image)
                mask = mask[indices]
            else:
                indices = image.shape[0]

            num_slices += image.shape[0]

            image = normalization(
                image, image_statistics, self.norm_mode, self.sample_mode
            )
            
            if self.sample_mode == "image":
                if (image.std() - 1) < 1e-6  and image.mean() < 1e-6:
                    print("here")
                else:
                    raise ValueError(f"something went wrong. std {image.std()} and mean {image.mean()}")
            else:
                pass

            if self.treshold != 0 or self.treshold != None:
                mask[mask >= self.treshold] = 1.0
                mask[mask < self.treshold] = 0.0
            else:
                pass

            if self.preprocess:
                for transform in self.preprocess:
                    image = transform(image)

            else:
                pass

            for idx, slice_idx in enumerate(indices):
                sliced_image = slice_and_concat(image, mask, idx, self.num_channel)

                
                #corresponding_mask_slice = corresponding_mask_slice[np.newaxis]

                #new_image = np.concatenate([corresponding_mask_slice, sliced_image], axis = 0) # last index of self.num_channel is mask_slice

                new_image = torch.from_numpy(sliced_image)

                name = (
                    id_value + "_" + self.files_to_distribute[0] + "_slice_idx_" + str(slice_idx) + ".pt"
                )
                name_and_path = self.path / name
                #print(name_and_path)
                torch.save(new_image, name_and_path)
                #exit()
                
                """
                with h5py.File(f"{name_and_path}.hdf5", "w") as f:
                    f.create_dataset(self.files_to_distribute[0], data=sliced_image)
                    f.create_dataset(
                        self.files_to_distribute[1], data=corresponding_mask_slice
                    )
                    f.create_dataset("id", data=id_value)
                    f.create_dataset("slice_number", data=slice_idx, dtype=int)"""
                # maybe in efficinet solution

        print(f"The images are preproccessed and sliced and distributed to {self.path}")
        print(f"It was {len(self.container)} and it produced {num_slices} slices with {self.num_channel} channels")
        
if __name__ == "__main__":
    datacontainer = ImageContainer(
        "/mnt/HDD16TB/arams/copy_to_crai/Piotr/", "MS", ["flair", "pvalue"]
    )

    img_statistics = ImageStatistics(datacontainer, image_type="flair")
    stat_df = img_statistics()
    print(stat_df)

    distributefile = DistributeFiles(
        path="/mnt/HDD16TB/arams/hope/hope/dataset/FLAIR",
        container=datacontainer,
        statistics=stat_df,
        files_to_distribute=["flair", "pvalue"],
        treshold=0.7,
        remove_zero_slices=True,
        norm_mode="z_norm",
        sample_mode="image",
    )
    distributefile.split_and_distribute()
