from pathlib import Path
import json

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from typing import Union, List

from hope.utils.split_train_validation import SplitTrainValidation


class ImageLoader(Dataset):
    def __init__(
        self, 
        path: List[Union[Path, list]], 
        augmentation: list = None, 
        mode = "train"
        ) -> None:
        self.mode = mode

        if isinstance(path, list):
            self.all_paths = path
        else:
            path = Path(path)
            self.all_paths = list(path.glob("*"))
        
        self.dataset = []
        for current_path in self.all_paths:
            
            image_path = list(Path(current_path).glob("*"))
            if ".DS_Store" in image_path:
                image_path.remove(".DS_Store")
            else:
                pass
            for paths in image_path:
                if paths.is_dir():
                    fetch_image_path = list(paths.glob("*"))
                    for img in fetch_image_path:
                        self.dataset.append(img)
                else:
                    for img in image_path:
                        self.dataset.append(img)

        
        print(f"Number of files in the {self.mode} dataset: {len(self.dataset)}")

        if ".DS_Store" in self.all_paths:
            self.all_paths.remove(".DS_Store")
        else:
            pass

        self.augmentation = augmentation
        self.patient_slice_index = self.__len__()
        

    def __len__(self) -> None: 
        return len(self.dataset) 

    def __getitem__(self, idx):
        current_image_path = self.dataset[idx]
        image_and_mask = torch.load(current_image_path)
        slice_idx = self.fetch_slice_number(idx)
        id_value = self.fetch_id_value(idx)

        if isinstance(image_and_mask, np.ndarray):
            image_and_mask = torch.from_numpy(image_and_mask)
        else:
            pass 

        if self.augmentation is not None:
            image_and_mask = self.augmentation(image_and_mask)
        else:
            pass
        
        
        image = image_and_mask[0:3].clone()
        mask = image_and_mask[-1].clone()
        
        return {"image": image, "mask": mask, "id" : id_value, "slice_idx": slice_idx}

    def fetch_id_value(self, idx) -> str:
        id_value = str(Path(self.dataset[idx]).stem).split("_")
        return id_value[0]

    def fetch_slice_number(self, idx) -> int:
        slice_idx = str(Path(self.dataset[idx]).stem).split("_")
        return int(slice_idx[-1])


def initiate_dataloader(
    path: Union[str, Path],
    batch_size: int,
    train_ratio: float = None,
    validation_ratio: float = None,
    augmentation: list = None,
    mode: str = "train",
    shuffle: bool = True,
    save_fold: bool = False,
    path_to_save_fold: Union[str, Path] = None,
    use_existing_fold: bool = False,
    ) -> DataLoader:

    if mode == "train":
        if use_existing_fold == True:
            save_fold = False
            path_to_save_fold = None

        if use_existing_fold == False:
            split = SplitTrainValidation(
                path=path, 
                train_ratio=train_ratio, 
                validation_ratio=validation_ratio,
                save_fold = save_fold, 
                save_fold_path = path_to_save_fold
            )

            dataset = split(shuffle)
            
        else:
            if Path(path).is_file():
                dataset = None
                with open(path, "r") as f:
                    loaded_json_file = json.load(f)
                    dataset = loaded_json_file.copy()
            else:
                raise RuntimeError(f"The given path is not a fold file. Got {path}")

        train = ImageLoader(dataset["train"], augmentation = augmentation, mode = "train")
        validation = ImageLoader(dataset["validation"], augmentation = None, mode = "validation")

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
        
        validation_loader = DataLoader(
            validation, batch_size=batch_size, shuffle=shuffle
        )

        return {"train" : train_loader, "validation": validation_loader}
    else:
        test = ImageLoader(path, augmentation = None, mode = "test")
        return DataLoader(test, batch_size=batch_size, augmentation = None, shuffle=shuffle)


if __name__ == "__main__":

    path = "/mnt/HDD16TB/arams/hope/hope/dataset/FLAIR_new_dataset_volume_norm_splitted/train"
    #path = "/mnt/HDD16TB/arams/hope/hope/dataset/fold/train_ratio_0.7_validation_ratio_0.3_fold_8.json"
    import time 
    start = time.time()
    dataloader = initiate_dataloader(
        path, 
        16, 
        0.7, 
        0.3, 
        save_fold = False, 
        use_existing_fold = False, 
        path_to_save_fold= "/mnt/HDD16TB/arams/hope/hope/dataset/fold")

    import matplotlib.pyplot as plt 
    for phase in ["train", "validation"]:
        for idx, img in enumerate(dataloader[phase]):
            print(img["id"])

        
    end = time.time()
    print(end-start)