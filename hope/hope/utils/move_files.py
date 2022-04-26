import shutil
from pathlib import Path

from typing import Union
from hope.dataset.splitdata import SplitData


class MoveFiles:
    def __init__(self, savepath: Union[str, Path], dataset_split: dict) -> None:
        self.savepath = Path(savepath)
        self.train_path = self.savepath / "train"
        self.test_path = self.savepath / "test"
        self.remainder_path = self.savepath / "remainder"
        self.dataset_split = dataset_split

        if self.dataset_split["remainder"] != None:
            self.remainder_path = self.savepath / "remainder"
        else:
            pass

        if not self.savepath.is_dir():
            self.savepath.mkdir()

            self.test_path.mkdir()
            self.train_path.mkdir()
            self.remainder_path.mkdir()
        else:
            self.test_path.mkdir()
            self.train_path.mkdir()
            self.remainder_path.mkdir()

        self.mode_path_dict = {
            "train": self.train_path,
            "test": self.test_path,
            "remainder": self.remainder_path,
        }

    def __call__(self) -> None:
        print("Moving file from given source to destination")
        for k in self.dataset_split:
            for v in self.dataset_split[k]:
                filename = v.name
                shutil.move(v, self.mode_path_dict[k] / filename)

        print(
            f"Moving of files are complete. The dataset is ready to use at path {self.savepath}"
        )


if __name__ == "__main__":
    path = "/mnt/HDD16TB/arams/hope/hope/dataset/FLAIR_new_dataset_volume_norm2"
    FLAIR = SplitData(path=path, train_ratio=0.9, test_ratio=0.1, remainder_ratio=0)
    # FLAIR.calculate_num_files()
    dataset = FLAIR.split(shuffle=True)

    move_flair = MoveFiles(path + "_splitted", dataset)
    move_flair()
