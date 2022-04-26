import argparse
from pathlib import path

from hope.dataset.imagecontainer import ImageContainer
from hope.dataset.distribute_files import DistributeFiles
from hope.dataset.splitdata import SplitData

from hope.utils.move_files import MoveFiles
from hope.utils.imagestatistics import ImageStatistics
from hope.utils.split_train_validation import SplitTrainValidation

from hope.train.imageloader import ImageLoader, initiate_dataloader
from hope.train.train import TrainModel


def main(args) -> None:
    img_container = ImageContainer(
        args.image_directory_path, args.filter_keyword, args.files_to_extract
    )

    statistics = ImageStatistics(
        container=img_container,
        image_type=args.files_to_extract[0],
        save_statistics=args.save_patient_statistics_df,
        mode=args.sample_mode,
        remove_zeros=args.remove_zeros,
    )

    if path(args.savepath_for_images).is_dir() and len(
        list(path(args.savepath_for_images).glob("*"))
    ):
        # checks if folder exist and if the folder containing files
        pass

    else:
        distributefile = DistributeFiles(
            path=args.savepath_for_images,
            container=img_container,
            statistics=statistics,
            files_to_distribute=args.files_to_extract,
            treshold=args.mask_treshold,
            remove_zero_slices=args.remove_zeros,
            norm_mode=args.normalization_mode,
            sample_mode=args.sample_mode,
        )
        distributefile.split_and_distribute()
        dataset = SplitData(
            path=path,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            remainder_ratio=args.remainder_ratio,
        )
        splitdataset = dataset.split(shuffle=args.shuffle)

        move_file = MoveFiles(args.savepath_for_images, splitdataset)
        move_files()

    if args.mode == "train":
        train_loader, validation_loader = initiate_dataloader(
            new_train_ratio = 1 - args.validation_ratio
            path=args.savepath,
            batch_size=args.batch_size,
            train_ratio=new_train_ratio,
            validation_ratio=args.validation_ratio,
            augmentation=args.augmentation,
            mode=args.mode,
            shuffle=args.shuffle,
        )

        train = TrainModel(model, optimizer, loss_function, train_loader, resume_checkpoint)
        train(save_model, epoch)
    

    elif args.mode == "test":
        pass
    else:
        raise ValueError(
            f"Please give a valid mode=train or test, remember validation is included in train. Got mode: {args.mode}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="segmentation for MRI images with multiple sceleriosis using different CNN architectures"
    )

    parser.add_argument(
        "--image_directory_path",
        help="The path for directory with image files",
        type=str,
        default="/mnt/HDD16TB/arams/copy_to_crai/Piotr/",
    )

    parser.add_argument(
        "--savepath_for_images",
        help="A path where the images are being saved to",
        type=str,
        default="/mnt/HDD16TB/arams/hope/hope/dataset/FLAIR",
    )

    parser.add_argument(
        "--filter_keyword",
        help="The keyword is being used to find and include files having this keyword value",
        type=str,
        default="MS",
    )

    parser.add_argument(
        "--files_to_extract",
        help="A list containing names that is going to be extracted to the container",
        type=list,
        default=["flair", "pvalue"],
    )

    parser.add_argument(
        "--save_patient_statistics_df",
        help="a boolean value wether to save the dataframe as a csv file for later analysis",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--sample_mode",
        help="Looking at the population or single volumes when calculating statistics",
        type=str,
        default="population",
    )

    parser.add_argument(
        "--remove_zeros",
        help="Removing empty slices only containing zeros.",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--mask_treshold",
        help="Treshold value to convert mask to binary mask",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--train_ratio",
        help="A ratio value of much files is going to be included in the train set",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--validation_ratio",
        help="A ratio value of much files is going to be included in the validation set",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--test_ratio",
        help="A ratio value of much files is going to be included in the test set",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--remainder_ratio",
        help="""A ratio value of much files is going to be included in the remainder set. 
                This is solely for adding more files to other sets if 
                needed, these files are unseen thus can be used for test.""",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--normalization_mode",
        help="Type of normalization applied on the image. Default z-norm, you can choose from min-max and z-norm",
        type=str,
        default="z_norm",
    )

    parser.add_argument(
        "--network_save_path",
        help="the path for the saved network",
        type=str,
        default="/mnt/HDD16TB/arams/MSseg_logs/saved_network",
    )

    parser.add_argument(
        "--lr", help="value for learning rate", type=float, default=5e-5
    )

    parser.add_argument(
        "--epoch", help="value for number of epochs", type=int, default=250
    )

    parser.add_argument(
        "--batc_size",
        help="value for for size of a single batch",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--shuffle", help="Wether to shuffle or not", type=bool, default=True
    )

    args = parser.parse_args()

    main(args=args)


    ######
    #spør daniel om hjelp for denne main funksjonen
    ####