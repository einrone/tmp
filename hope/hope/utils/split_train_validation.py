from pathlib import Path
from typing import Union, Optional, Dict

import numpy as np
import pandas as pd
import json

class SplitTrainValidation:
    def __init__(
        self, path: Union[str, Path], 
        train_ratio: float, 
        validation_ratio: float,
        save_fold: bool = False,
        save_fold_path: Union[str, Path] = None
    ) -> None:

        self.path = Path(path)

        self.all_paths = list(self.path.glob("*"))
        self.all_paths = [str(current_path) for current_path in self.all_paths]
        
        if "DS_Store" in self.all_paths:
            self.all_paths.remove("DS_Store")
        else:
            pass

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.dataset_length = len(self.all_paths)

        self.save_fold = save_fold

        if self.train_ratio + self.validation_ratio == 1.0:
            pass

        else:
            raise ValueError("the sum of ratio isn't 1.0")

        if ".DS_Store" in self.all_paths:
            self.all_paths.remove(".DS_Store")
        else:
            pass

        self.num_files = self.__calculate_num_files()
        self.__initiate_dataset()

        if self.save_fold == True:
            self.save_fold_path = Path(save_fold_path)
            if self.save_fold_path.is_dir():
                pass 
            else:
                print("The current folder doesnt exist. Creating folder fold")
                self.save_fold_path.mkdir()
                print(f"Folder located at {self.save_fold_path}")
        else:
            pass

    def __initiate_dataset(self) -> Dict:
        """
        insert docs here
        """
        self.dataset = {"train": None, "validation": None}

    def __calculate_num_files(self) -> dict:

        """
        insert docs here
        """
        num_train_files = int(round(self.train_ratio * self.dataset_length))
        num_validation_files = int(round(self.validation_ratio * self.dataset_length))

        total = num_train_files + num_validation_files

        diff = np.abs(total - self.dataset_length)

        if self.dataset_length > total:
            num_train_files += diff

        elif self.dataset_length < total:
            num_train_files -= diff

        else:
            pass

        return {
            "train": num_train_files,
            "validation": num_validation_files,
        }
    def check_files(self):
        files_in_folder = list(self.save_fold_path.glob("*"))
        if "DS_Store" in files_in_folder:
            files_in_folder.remove("DS_Store")
        else:
            pass 
        
        if len(files_in_folder) == 0:
            filename_and_path = self.save_fold_path / f"train_ratio_{self.train_ratio}_validation_ratio_{self.validation_ratio}_fold_0.json"
            return filename_and_path
        else:
            list_of_files = [files.stem for files in files_in_folder]
            list_of_folds = []
            for filenames in list_of_files:
                splitted_name = filenames.split("_")
                fold_number = int(splitted_name[-1])
                list_of_folds.append(fold_number)
            
            list_of_folds = sorted(list_of_folds)
            highest_fold_number = list_of_folds[-1]
            new_fold_number = highest_fold_number + 1
            filename_and_path = self.save_fold_path / f"train_ratio_{self.train_ratio}_validation_ratio_{self.validation_ratio}_fold_{new_fold_number}.json"
            return filename_and_path

    def __call__(self, shuffle: Optional[bool] = False) -> dict:
        if shuffle == True:
            np.random.shuffle(self.all_paths)

        self.dataset["train"] = self.all_paths[0 : self.num_files["train"]]
        self.dataset["validation"] = self.all_paths[
            self.num_files["train"] : self.num_files["train"]
            + self.num_files["validation"]
        ]


        if self.save_fold == True:
            filename_and_path = self.check_files()
            with open(filename_and_path, "w") as f:
                json.dump(self.dataset, f, indent=4)

        return self.dataset


if __name__ == "__main__":
    path = "/mnt/HDD16TB/arams/hope/hope/dataset/FLAIR_dataset/train"
    train_ratio = 0.7
    validation_ratio = 0.3

    split_train_validation = SplitTrainValidation(
        path=path, train_ratio=train_ratio, validation_ratio=validation_ratio
    )
    dataset = split_train_validation(True)
    print(dataset)
