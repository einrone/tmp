from pathlib import Path

import pandas as pd
import numpy as np

from typing import Union, List, Optional, Dict


class SplitData:
    def __init__(
        self,
        path: Union[str, Path],
        train_ratio: float,
        test_ratio: float,
        remainder_ratio: Optional[float] = None,
    ) -> None:
        if not isinstance(path, (str, Path)):
            raise TypeError(
                f"The following path is not str/pathlib.Path. Got: {type(path)}"
            )
        else:
            pass

        self.path = Path(path)
        self.list_of_paths = list(self.path.glob("*"))
        self.dataset_length = len(self.list_of_paths)

        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.remainder_ratio = remainder_ratio

        self.num_files = self.calculate_num_files()
        self.initiate_dataset(self.remainder_ratio)

    def calculate_num_files(
        self,
    ) -> Dict:

        """
        insert docs here
        """

        if self.remainder_ratio is None:
            self.remainder_ratio = 0
        else:
            pass

        num_train_files = int(round(self.train_ratio * self.dataset_length))
        num_test_files = int(round(self.test_ratio * self.dataset_length))
        num_remainder_files = int(round(self.remainder_ratio * self.dataset_length))

        total = num_train_files + num_test_files + num_remainder_files

        diff = np.abs(total - self.dataset_length)

        if self.dataset_length > total:
            num_train_files += diff

        elif self.dataset_length < total:
            num_train_files -= diff

        else:
            pass

        return {
            "train": num_train_files,
            "test": num_test_files,
            "remainder": num_remainder_files,
        }

    def initiate_dataset(self, remainder: float) -> Dict:
        """
        insert docs here
        """
        self.dataset = {"train": None, "test": None}
        if remainder != 0 and remainder is not None:
            self.dataset["remainder"] = None
        else:
            pass

    @property
    def fetch_dataset(self) -> Dict:
        return self.dataset

    def split(self, shuffle: bool = True) -> Dict:
        if shuffle == True:
            np.random.shuffle(self.list_of_paths)
        else:
            pass

        train = self.list_of_paths[0 : self.num_files["train"]]
        test = self.list_of_paths[
            self.num_files["train"] : self.num_files["train"] + self.num_files["test"]
        ]

        self.dataset["train"] = train
        self.dataset["test"] = test

        if self.remainder_ratio != 0 or self.remainder_ratio != None:
            remainder = self.list_of_paths[
                self.num_files["train"]
                + self.num_files["test"] : self.num_files["train"]
                + self.num_files["test"]
                + self.num_files["remainder"]
            ]
            self.dataset["remainder"] = remainder
        else:
            pass

        return self.dataset

    # maybe change name


if __name__ == "__main__":
    path = "/mnt/HDD16TB/arams/hope/hope/dataset/FLAIR"
    dataset = SplitData(path=path, train_ratio=0.8, test_ratio=0.1, remainder_ratio=0.1)
    a = dataset.calculate_num_files()
    print(dataset.split(shuffle=True))
    # dataset.structurize_data("o")
