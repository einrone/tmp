from pathlib import Path
from typing import Dict, Union
import numpy as np 
import torch 
from collections import defaultdict
from itertools import product
import json
from torchvision.transforms.functional import InterpolationMode
import tqdm

from hope.train.imageloader import ImageLoader, initiate_dataloader
from hope.augmentation.initialize_augmentation import initialize_augmentation
from hope.train.train import TrainModel
from hope.model.attention_unet import AttentionUNet
from hope.loss.hybridloss import HybridLoss
from hope.loss.focaltverskyloss import FocalTverskyLoss

def permute(hyperparameter_dict: dict) -> list:
    return [dict(zip(hyperparameter_dict, v)) for v in product(*hyperparameter_dict.values())]

def grid_search(
    model: torch.nn.Module,
    optimizer_name: str,
    loss_function : torch.nn.Module,
    dataset_path: Union[str, Path],
    hyperparameter_dict: dict,
    epoch: int = 10,
    specific_device: str = None,
    ) -> None:

    augmentation = initialize_augmentation(
            probability = 0.5,
            angle = [-15,15], #not using rotation due to shifting the values and the image is not normalized
            interpolation_mode = InterpolationMode.BILINEAR
            )
    
    list_of_perumations_of_parameters = permute(hyperparam_dict)
    for config in list_of_perumations_of_parameters:
        print(f"Using config: {config}")
        dataloader = initiate_dataloader(
                path = dataset_path,
                batch_size = config["batch_size"],
                train_ratio = 0.7,
                validation_ratio = 0.3,
                augmentation = augmentation,
                mode = "train",
                shuffle = False,
            )

        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr = config["lr"], momentum = config["momentum"], nesterov = config["nesterov"])
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"])
        elif optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr = config["lr"])
        else:
            raise RuntimeError("The given optimizer name is not valid, expcted SGD, Adam or AdamW")
        
        train = TrainModel(
                            model = model, 
                            optimizer = optimizer, 
                            loss_function = loss_function, 
                            dataloader = dataloader, 
                            epoch = epoch,
                            learning_rate = config["lr"],
                            scheduler_name = "StepLr",
                            specific_device = specific_device
                        )

        best_dsc_value, best_loss_value, path = train(
            save_model = True, 
            save_result_to_csv = False,
            show_progress = True,
            )
        
        print(f"best dice score {best_dsc_value} and loss {best_loss_value}")

        config["best_dice"] = best_dsc_value
        config["best_loss"] = best_loss_value
        config["path"] = path

        
    current_path = Path(__file__).parent / "grid_search_result"
    filename_with_path = current_path / f"{type(model).__name__ + '_' + optimizer_name + '_FocalTverskyLoss.json'}"
    
    with open(filename_with_path, "w") as f:
        json.dump(list_of_perumations_of_parameters,f, indent=4)
            
    


if __name__ == "__main__":
    lr = [0.0001, 0.001, 0.01, 0.01]
    momentum = [0.9, 0.85, 0.8]
    batch_size = [64, 32, 16,8,4]
    
    hyperparam_dict = {"batch_size": batch_size, "lr" : lr, "momentum": momentum, "nesterov": [False, True]}
    list_of_perumations_of_parameters = permute(hyperparam_dict)
    """for run, dict_of_params in enumerate(list_of_perumations_of_parameters):
        print(dict_of_params["batch_size"], dict_of_params["lr"], dict_of_params["momentum"], dict_of_params["nesterov"],run)"""
    grid_search(
        model = AttentionUNet(3,1,32),
        optimizer_name = "SGD",
        loss_function = FocalTverskyLoss,
        dataset_path = Path("/mnt/HDD16TB/arams/hope/hope/dataset/FLAIR_dataset/train"),
        hyperparameter_dict = hyperparam_dict,
        epoch = 250, 
        specific_device = "cuda:1"
    )