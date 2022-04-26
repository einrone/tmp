from pathlib import Path

import pandas as pd
import numpy as np

from typing import Union, List, Optional, Dict

# from metainfo import MetaInfo


class CreateDataset:
    pass

    def __init__(self, path: Union[str, Path], testmode=False) -> None:
        if not isinstance(path, (str, Path)):
            raise TypeError(
                f"The following path is not str/pathlib.Path. Got: {type(path)}"
            )
        else:
            pass

        self.path = Path(path)

        if testmode is False:
            if self.path.is_dir() is True:
                pass
            else:
                raise ValueError(
                    f"The path given does not point to a existing directory"
                )
        else:
            pass

        self.dataset = None

    def fetch_filepaths(
        self, filepath: Union[str, Path], filterPath: Optional[str] = None
    ) -> List[Path]:
        """insert doc here"""
        filepath = Path(filepath)

        if filterPath is None:
            filterPath = "*"

        return list(filepath.glob(f"{filterPath}"))

    @staticmethod
    def calculate_num_files(
        dataset_length,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
        remainder_ratio: Optional[float] = None,
    ) -> Dict:

        """
        insert docs here
        """
        if remainder_ratio is None:
            remainder_ratio = 0

        num_train_files = int(round(train_ratio * dataset_length))
        num_validation_files = int(round(validation_ratio * dataset_length))
        num_test_files = int(round(test_ratio * dataset_length))
        num_remainder_files = int(round(remainder_ratio * dataset_length))

        total = (
            num_train_files
            + num_validation_files
            + num_test_files
            + num_remainder_files
        )

        diff = np.abs(total - dataset_length)

        if dataset_length > total:
            num_train_files += diff

        elif dataset_length < total:
            num_train_files -= diff

        else:
            pass

        return {
            "train": num_train_files,
            "validation": num_validation_files,
            "test": num_test_files,
            "remainder": num_remainder_files,
        }

    def initiate_dataset(self, remainder: float) -> Dict:
        """
        insert docs here
        """
        self.dataset = {"train": None, "validation": None, "test": None}
        if remainder != 0 and remainder is not None:
            self.dataset["remainder"] = None
        else:
            pass

    @property
    def fetch_dataset(self):
        return self.dataset

    # maybe change name
    def split(
        self,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
        remainder_ratio: float,
        filtertypes: Optional[str],
        shuffle: Optional[bool] = True,
        seed: Optional[np.random.seed] = None,
    ) -> Dict:

        """
        insert docs here
        """

        if seed is None:
            pass
        else:
            np.random.seed(seed)

        ratio_sum = train_ratio + validation_ratio + test_ratio + remainder_ratio
        if (
            not np.abs(
                train_ratio + validation_ratio + test_ratio + remainder_ratio - 1.0
            )
            <= 1e-7
        ):
            raise ValueError(f"Got total sum of ratios: {ratio_sum}")

        else:
            pass

        self.initiate_dataset(remainder_ratio)
        fetched_paths = self.fetch_filepaths(self.path, filterPath=filtertypes)

        # fetched_file_paths = self.fetch_dataset()

        if shuffle is True:
            np.random.shuffle(fetched_paths)

        dataset_length = len(fetched_paths)

        num_files_of_each_phase = self.calculate_num_files(
            dataset_length, train_ratio, validation_ratio, test_ratio, remainder_ratio
        )

        num_train = num_files_of_each_phase["train"]
        num_validation = num_files_of_each_phase["validation"]
        num_test = num_files_of_each_phase["test"]

        ##################### splitting #####################
        self.dataset["train"] = fetched_paths[0:num_train]
        self.dataset["validation"] = fetched_paths[
            num_train : num_train + num_validation
        ]
        self.dataset["test"] = fetched_paths[
            num_train + num_validation : num_train + num_validation + num_test
        ]
        #####################################################

        # self.dataset = pd.DataFrame.from_dict(self.dataset)
        if remainder_ratio is None or remainder_ratio == 0:
            return self.dataset
        else:

            num_remainder = num_files_of_each_phase["remainder"]

            self.dataset["remainder"] = fetched_paths[
                num_train + num_validation + num_test : :
            ]
            return self.dataset

    # @staticmethod

    def extract_files_from_folders(self, which_file_to_extract: List[str]) -> Dict:

        structurized_data = dict()

        for phase, list_of_paths in self.dataset.items():
            structurized_data[phase] = dict()
            for current_path in list_of_paths:

                foldername = current_path.name
                structurized_data[phase][foldername] = {}

                for file_to_extract in which_file_to_extract:
                    name_of_file_extracted = file_to_extract

                    if "*" in file_to_extract:
                        name_of_file_extracted = list(name_of_file_extracted)
                        distinct = set(name_of_file_extracted)

                        name_of_file_extracted.remove("*")
                        name_of_file_extracted.remove("*")

                        # name_of_file_extracted.remove("*")

                        name_of_file_extracted = "".join(name_of_file_extracted)

                    else:
                        pass

                    structurized_data[phase][foldername][
                        name_of_file_extracted
                    ] = self.fetch_filepaths(current_path, file_to_extract)

        self.dataset = structurized_data
        return self.dataset

    def distribute_files(self, savepath: Union[str, Path] = None):
        """
        insert docs here
        """
        """
        if savepath is None:
            savepath = (Path.cwd().parent).parent
            savepath.mkdir(parents=True)
        else:
            savepath = Path(savepath)
            if savepath.is_dir() is True:
                pass
            else:
                raise ValueError(f"savepath does not point to a existing directory.")"""

        print(self.dataset)


if __name__ == "__main__":
    path = "/mnt/HDD16TB/arams/copy_to_crai/Piotr"

    dataset = CreateDataset(path)
    a = dataset.split(0.7, 0.1, 0.1, 0.1, "MS*/")
    dataset.extract_files_from_folders(["*flair*", "*t1*", "*pvalue*"])
    dataset.distribute_files()
    # dataset.structurize_data("o")
