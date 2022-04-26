import os
from glob import glob
import unittest
from unittest import mock
from pathlib import Path
from typing import Union, List, Optional, Dict

import numpy as np
import random

from hope.dataset.createdataset import CreateDataset


"""def mock_split_no_remainder(path, train_ratio, val_ratio, test_ratio):

    train_list = [path/"test1.nifti.gzz"]
    return {
        "train": [],
        "validation": [],
        "test": [],
    }
"""


import random


def mock_fetch_filepaths(filterPath: Optional[str] = None):
    path = Path("basepath")
    pathObj = [
        path / "test1.nii.gz",
        path / "test2.nii.gz",
        path / "test3.nii.gz",
        path / "test4.nii.gz",
        path / "test5.nii.gz",
    ]
    empty_pathObj = []

    if filterPath is None:  # 1
        return empty_pathObj
    elif ".*" in filterPath:  # 2
        return pathObj
    elif "test*.*" in filterPath == "test*.*":  # 3
        return pathObj

    elif "test*" in filterPath == "test*":  # 4
        return pathObj
    else:  # 5
        return empty_pathObj


def mock_split_no_remainder(
    dataset_length: float,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    remainder_ratio: float,
    filtertypes: Optional[str] = None,
    shuffle: Optional[bool] = True,
    seed: Optional[np.random.seed] = None,
):
    path = Path("basepath")
    return {
        "train": [path / "test1.nii.gz", path / "test2.nii.gz", path / "test3.nii.gz"],
        "validation": [path / "test4.nii.gz"],
        "test": [path / "test5.nii.gz"],
    }


def mock_split_with_remainder(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    remainder_ratio: float,
    filtertypes: Optional[str] = None,
    shuffle: Optional[bool] = True,
    seed: Optional[np.random.seed] = None,
):
    path = Path("basepath")
    return {
        "train": [path / "test1.nii.gz", path / "test2.nii.gz"],
        "validation": [path / "test3.nii.gz"],
        "test": [path / "test4.nii.gz"],
        "remainder": [path / "test5.nii.gz"],
    }


def mock_calcule_num_files(
    dataset_length,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    remainder_ratio: Optional[float] = None,
):

    if remainder_ratio is None or remainder_ratio == 0:
        dataset_length = 77
        train = 47
        validation = 15
        test = 15
        remainder = 0
        num_files_tuple_expected = {
            "train": train,
            "validation": validation,
            "tets": test,
            "remainder": remainder,
        }
    else:

        dataset_length = 77
        train = 53
        validation = 8
        test = 8
        remainder = 8
        num_files_tuple_expected = {
            "train": train,
            "validation": validation,
            "tets": test,
            "remainder": remainder,
        }


def mock_distribute_files(savepath: Union[str, Path]):
    savepath = Path(savepath)

    return None


class TestCreateDataset(unittest.TestCase):
    @mock.patch("pathlib.Path.glob", side_effect=mock_fetch_filepaths)
    def test_fetch_paths(self, mock_fetch_filepaths):
        basepath = Path("basepath")
        dataset = CreateDataset(basepath, True)

        # test1
        filterPath1 = None
        fetchedpaths1 = dataset.fetch_filepaths(filterPath1)
        self.assertEqual(fetchedpaths1, mock_fetch_filepaths(filterPath1))

        # test2
        filterPath2 = ".*"
        fetchedpaths2 = dataset.fetch_filepaths(filterPath2)
        self.assertEqual(fetchedpaths2, mock_fetch_filepaths(filterPath2))

        # test3
        filterPath3 = "test*.*"
        fetchedpaths3 = dataset.fetch_filepaths(filterPath3)

        self.assertEqual(fetchedpaths3, mock_fetch_filepaths(filterPath3))

        # test4
        filterPath3 = "test*"

        fetchedpaths4 = dataset.fetch_filepaths(filterPath3)

        self.assertEqual(fetchedpaths4, mock_fetch_filepaths(filterPath3))

        # test5
        filterPath5 = "testings*"

        fetchedpaths5 = dataset.fetch_filepaths(filterPath5)

        self.assertEqual(fetchedpaths5, mock_fetch_filepaths(filterPath5))

    @mock.patch(
        "hope.dataset.createdataset.CreateDataset.split",
        side_effect=mock_split_no_remainder,
    )
    def test_split_no_remainder(self, mock_split_no_remainder):
        basepath = Path("basepath")
        dataset = CreateDataset(basepath, True)
        split_path = dataset.split(5, 0.6, 0.2, 0.2, 0)

        dictkey = split_path.keys()
        dictvalues = split_path.values()

        self.assertEqual(dictkey, mock_split_no_remainder(5, 0.6, 0.2, 0.2, 0).keys())

        self.assertEqual(
            list(dictvalues),
            list(mock_split_no_remainder(5, 0.6, 0.2, 0.2, 0).values()),
        )

    @mock.patch(
        "hope.dataset.createdataset.CreateDataset.split",
        side_effect=mock_split_with_remainder,
    )
    def test_split_with_remainder(self, mock_split_with_remainder):
        basepath = Path("basepath")
        dataset = CreateDataset(basepath, True)
        split_path = dataset.split(0.4, 0.2, 0.2, 0.2)

        dictkey = split_path.keys()
        dictvalues = split_path.values()

        self.assertEqual(dictkey, mock_split_with_remainder(0.4, 0.2, 0.2, 0.2).keys())
        self.assertEqual(
            list(dictvalues),
            list(mock_split_with_remainder(0.4, 0.2, 0.2, 0.2).values()),
        )

    @mock.patch(
        "hope.dataset.createdataset.CreateDataset.calculate_num_files",
        side_effect=mock_calcule_num_files,
    )
    def test_calculate_num_files_no_remainder(self, mock_calcule_num_files):

        basepath = Path("basepath")
        dataset = CreateDataset(basepath, True)

        num_files_tuple = dataset.calculate_num_files(77, 0.6, 0.2, 0.2, 0)
        self.assertEqual(
            num_files_tuple, num_files_tuple_expected(77, 0.6, 0.2, 0.2, 0)
        )

    @mock.patch(
        "hope.dataset.createdataset.CreateDataset.calculate_num_files",
        side_effect=mock_calcule_num_files,
    )
    def test_calculate_num_files_no_remainder(self, mock_calcule_num_files):

        basepath = Path("basepath")
        dataset = CreateDataset(basepath, True)
        num_files_tuple = dataset.calculate_num_files(77, 0.7, 0.1, 0.1, 0.1)
        self.assertEqual(
            num_files_tuple, mock_calcule_num_files(77, 0.7, 0.1, 0.1, 0.1)
        )

    """@mock.patch("pathlib.Path.mkdir", side_effect=mock_distribute_files)
    def test_distribute_files(self):
        pass"""


if __name__ == "__main__":
    test = TestCreateDataset()
    test.test_fetch_paths()
    test.test_split_no_remainder()

    """

      @staticmethod
    def fetch_filepaths(
        path: [str, Path],
        filename: Optional[str] = None,
        filetypes: Optional[str] = None,
    ) -> List[Path]:
        insert doc here

        print(path.glob("*.test*"))
        thisFile = None

        if filetypes is None and filename is None:
            filename = "*"
            filetypes = "*"

            thisFile = "{}.{}".format(filename, filetypes)

        elif filetypes is not None and filename is None:
            filename = "*"
            thisFile = "{}.{}".format(filename, filetypes)

        elif filetypes is None and filename is not None:
            filetypes = "*"
            thisFile = "{}.{}".format(filename, filetypes)

        elif filetypes is not None and filename is not None:
            thisFile = "{}.{}".format(filename, filetypes)

        else:
            raise ValueError(f"Please choose a filetype or select None. Got {filetype}")

        return [currentpath for currentpath in path.glob(f"{thisFile}")]
    """
