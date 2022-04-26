from pathlib import Path
from typing import Union, List, Optional, Dict
import collections

import numpy as np
import nibabel as nib

from hope.utils.fetchpath import FetchPath


class ImageContainer(FetchPath):
    def __init__(
        self,
        path: Union[str, Path],
        filterpath: str,
        files_to_extract: Optional[List] = None,
    ):
        self.path = Path(path)
        self.filterpath = filterpath
        self.files_to_extract = files_to_extract

        if self.path.is_dir():
            pass
        else:
            raise ValueError(f"The following path does not direct to any directory")

        self.fetched_path = FetchPath()
        self.list_of_path = self.fetched_path(self.path, self.filterpath + "*")

        self.structurized_data = []

        if self.files_to_extract:
            self.__create_structurize_data__()
        else:
            self.structurized_data = self.list_of_path

    @staticmethod
    def load_image(path: Union[str, Path]):
        """
        Method responsible for opening .nii.gz
        files. This method uses nibabel.load()

        args:
            path: path for the given image to be
                  loaded
        return:
            np.ndarray with dimension (C, H, W)

        """

        path = str(path)
        return np.array(nib.load(path).get_fdata())

    def __create_structurize_data__(self) -> List:
        """
        private method which responsible for
        creating structurized data keys and entries,
        These dictornaries is then filled with images and
        their corresponding mask.

        args:
            None
        return:
            Updating self.structurized_data with structurized dictonaries,
            and adds image and label path


        """

        for current_path in self.list_of_path:
            patient_id = current_path.name
            temp_dict = collections.defaultdict(dict)

            if len(self.files_to_extract) > 1:
                for files in self.files_to_extract:

                    temp_dict[patient_id][files] = self.fetched_path(
                        current_path, "*" + files + ".nii.gz"
                    )

                    if not temp_dict[patient_id][files]:
                        del temp_dict[patient_id][files]
                        continue
                    else:
                        pass

                if len(temp_dict[patient_id].keys()) == len(self.files_to_extract):
                    self.structurized_data.append(temp_dict)
                else:
                    pass
            else:
                temp_dict[patient_id] = self.fetched_path(
                    current_path, "*" + self.files_to_extract[0] + ".nii.gz"
                )

                if not temp_dict[patient_id]:
                    del temp_dict[patient_id]
                    continue
                else:
                    pass

                self.structurized_data.append(temp_dict)

            del temp_dict

    def __len__(self):
        return len(self.structurized_data)

    def __iter__(self) -> "ImageGenerator":
        self.index = 0
        return self

    def __next__(self):
        """
        Fetchs image paths from self.structurized_data,
        and returns pair/triplets/etc.. of images and their
        label. This is a form for a generator.

        args:
            index: indexes for fetching data,
                   index = (0, len(self.structurized_data))

        return:
            pair/triplets/etc.. of images and their
            label

        """
        if self.index < len(self.structurized_data):

            image_dict = collections.defaultdict(dict)
            fetced_dict_from_list = self.structurized_data[self.index]
            id_value = list(fetced_dict_from_list.keys())[0]

            image_path = list(fetced_dict_from_list.values())[0]
            if len(self.files_to_extract) > 1:
                for mode, paths in image_path.items():
                    image_dict[id_value][mode] = self.load_image(paths)
            else:
                image_dict[id_value] = self.load_image(image_path)

            self.index += 1
            return image_dict

        else:
            raise StopIteration


if __name__ == "__main__":
    generate_img = ImageContainer(
        "/mnt/HDD16TB/arams/copy_to_crai/Piotr/",
        "MS",
        ["flair", "pvalue"],
    )

    for img in generate_img:
        print(img)
        exit()
