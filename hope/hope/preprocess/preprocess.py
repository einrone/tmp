from pathlib import Path 
from typing import Union, List

import numpy as np 
import torch

from hope.dataset.imagecontainer import ImageContainer
from hope.utils.imagestatistics import ImageStatistics
from hope.utils.normalization import normalization
from hope.utils.equalize_dimension import equalize_dimension
from hope.utils.clean_mask import clean_mask
from hope.utils.slice_and_concat import slice_and_concat
from hope.utils.swap_brain_plane import swap_brain_plane
from hope.utils.remove_zeros import remove_zeros

class Preprocess(ImageStatistics):
    def __init__(
        self, 
        path: Union[str, Path],
        container: ImageContainer,
        normalization_mode: str = "z_norm",
        normalize: str = "image",
        mask_treshold: float = 0.4,
        clean_label_in_mask: bool = False,
        min_pixel_to_remove: int = 4,
        volume_images: bool = False, 
        remove_empty_slices: bool = True,
        num_channel: int = 3,
    ) -> None:
    
        super(Preprocess, self).__init__(
            container,
            "flair",
            normalize,
            remove_empty_slices)
        

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

        self.container = container
        self.normalization_mode = normalization_mode
        self.normalize = normalize

        if 0 < mask_treshold < 1.0:
            self.mask_treshold = mask_treshold
        else:
            self.mask_treshold = None

        self.min_pixel_to_remove = min_pixel_to_remove
        self.clean_label_in_mask = clean_label_in_mask
        self.remove_empty_slices = remove_empty_slices
        self.num_channel = num_channel
        self.volume_images = volume_images
    
    def __collect_stats(
        self, 
        brain_plane: list, 
        save_statistic: bool = False, 
        equalize_dimension_in_image = False
        )-> dict:  
        print("calculating statistics")
        stat_df = {}
        for brain_axis in brain_plane:
                
            stat_df[brain_axis] = self.__call__(
                brain_axis, 
                equalize_dimension_in_image, 
                save_statistic
                )

            print(f"Dataframe for {brain_axis}")
            print(stat_df[brain_axis])

        print("calculating done \n \n")
        return stat_df

    def preprocess_images(
        self, 
        brain_plane: List[str], 
        save_statistic: bool = False
        )-> None:

        
        print("Starting preprocessing")
        if isinstance(brain_plane, list):
            if len(brain_plane) > 1:
                equalize_dimension_in_image = True
        else:
            equalize_dimension_in_image = False

        stat_df = self.__collect_stats(
            brain_plane, 
            save_statistic, 
            equalize_dimension_in_image
            )
        num_slices = 0
        for index, data in enumerate(self.container):
            id_value = list(data.keys())[0]
            img_key, mask_key = list(data[id_value])
            
            
            current_path = self.path / id_value
            current_path.mkdir()
            
            image = data[id_value][img_key]
            mask = data[id_value][mask_key]

            if equalize_dimension_in_image == True:
                image = equalize_dimension(image)
                mask = equalize_dimension(mask)


            if self.mask_treshold != None:
                mask[mask >= self.mask_treshold] = 1
                mask[mask < self.mask_treshold] = 0
            else:
                pass
            


            for plane in brain_plane:
                image_statistics = stat_df[plane].loc[id_value]

                current_brain_plane_path = current_path / plane
                print(current_brain_plane_path)
                current_brain_plane_path.mkdir()

                reshaped_image = swap_brain_plane(image, plane)
                reshaped_mask = swap_brain_plane(mask, plane)

                original_reshape_img = reshaped_image.copy()

                reshaped_image, indices = remove_zeros(reshaped_image)
                reshaped_image = normalization(reshaped_image, image_statistics, self.normalization_mode, self.normalize)
                
                reshaped_mask = reshaped_mask[indices]
                
                if self.clean_label_in_mask == True:
                    reshaped_mask = clean_mask(
                        mask = reshaped_mask, 
                        connectivity = 8,
                        min_pixel = self.min_pixel_to_remove)
                    
                    print(np.unique(reshaped_mask))
                else:
                    pass


                
                if self.normalize == "image":
                    if self.normalization_mode == "z_norm":
                        if np.abs(reshaped_image.std() - 1) < 1e-6  and reshaped_image.mean() < 1e-6:
                            pass
                        else:
                            raise ValueError(f"something went wrong. std {reshaped_image.std()} and mean {reshaped_image.mean()}")
                
                elif self.normalize == "brain":
                    if self.normalization_mode == "z_norm":
                        if np.abs(reshaped_image[original_reshape_img != 0].std() - 1) < 1e-6  and reshaped_image[original_reshape_img != 0].mean() < 1e-6:
                            print("here")
                        else:
                            raise ValueError(f"something went wrong. std {image[original_reshape_img != 0].std()} and mean {image[original_reshape_img!= 0].mean()}")
                else:
                    pass


                if self.volume_images == True:
                    pass 
                else:
                    for idx, slice_idx in enumerate(indices):
                        sliced_image = slice_and_concat(reshaped_image, reshaped_mask, idx, self.num_channel)

                        
                        #corresponding_mask_slice = corresponding_mask_slice[np.newaxis]

                        #new_image = np.concatenate([corresponding_mask_slice, sliced_image], axis = 0) # last index of self.num_channel is mask_slice

                        new_image = torch.from_numpy(sliced_image)

                        name = (
                            f"{id_value}_{img_key}_{plane}_slice_idx_{slice_idx}.pt"
                        )
                        name_and_path = current_brain_plane_path / name
                        torch.save(new_image, name_and_path)
                        num_slices += 1



        print(f"The images are preproccessed and sliced and distributed to {self.path}")
        print(f"It was {len(self.container)} patients and it produced {num_slices} slices with {self.num_channel} channels. In the plane {brain_plane}")

if __name__ == "__main__":
    datacontainer = ImageContainer(
        path = "/mnt/HDD16TB/arams/copy_to_crai/Piotr", 
        filterpath = "MS", 
        files_to_extract = ["flair", "pvalue"],
        exclude_files = ["MS4001"]
    )

    preprocess = Preprocess(
        path = "/mnt/HDD16TB/arams/hope/hope/dataset/FLAIR_new_dataset_volume_norm2",
        container = datacontainer,
        normalization_mode = "z_norm",
        normalize = "image",
        mask_treshold = 0.4,
        clean_label_in_mask = True,
        volume_images = False,
        remove_empty_slices = True,
        num_channel = 3,
    )

    preprocess.preprocess_images(
        brain_plane = ["sagittal", "frontal", "horizontal"],
        save_statistic = False
    )
    

