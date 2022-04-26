from pathlib import Path
from typing import Dict, Union
import warnings
from collections import defaultdict
import time 

from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hope.train.imageloader import ImageLoader
from hope.metrics.calculate_metrics import calculate_metrics


random_seed = 1 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TrainModel:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim,
        loss_function: torch.nn.Module,
        dataloader : Dict[ImageLoader, ImageLoader],
        ) -> None:

        self.device = "cuda:1"
        self.model = model 
        self.model.to(self.device)
        self.optimizer = optimizer
        self.loss_function = loss_function()
        self.dataloader = dataloader
        self.train_dataset_length = len(self.dataloader["train"])
        self.validation_dataset_length = len(self.dataloader["validation"])
        self.initiate_metrics_dict()

    def initiate_metrics_dict(self) -> None:
        self.metrics_dict = {
            "train":
                {
                    "recall": [],
                    "precision": [],
                    "accuracy": [],
                    "dice_score": [],
                    "loss": []
                },
            "validation":
                {
                    "recall": [],
                    "precision": [],
                    "accuracy": [],
                    "dice_score": [],
                    "loss": []
                }
        }
    def __train(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        epoch: int,
        ) -> dict:
        """
        THIS METHOD IS NOT DONE
        """
        self.model.train()
       
        loss_value = 0

        tmp_metrics_dict = {
            "recall": 0,
            "precision": 0,
            "accuracy": 0,
            "dice_score": 0,
        }

        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            for dataloader_idx, data in enumerate(train_loader):

                image = data["image"]
                mask = data["mask"]

                image, mask = (image.to(self.device)).float(), mask.to(self.device)

                image_predicted = self.model(image)
                loss = self.loss_function(image_predicted, mask)

                loss.backward()
                self.optimizer.step()


                loss_value += loss.item()
                metric_values = calculate_metrics(image_predicted, mask, 0.7)

                for k,v in zip(tmp_metrics_dict.keys(), metric_values):
                    tmp_metrics_dict[k] += v 
                #release binding, since data is 
                #not binded somewhere else, the 
                #data is deleted from cache 
                del data, image, mask

        
        tmp_metrics_dict["loss"] = loss_value
        for k,v in tmp_metrics_dict.items():
            tmp_metrics_dict[k] = v/self.train_dataset_length

        return tmp_metrics_dict

    def __validate(
        self, 
        validation_loader: torch.utils.data.DataLoader, 
        epoch: int,  
        save_model: bool
        ) -> dict:
        """
        THIS METHOD IS NOT DONE
        """
        self.model.eval()

        loss_value = 0
        tmp_metrics_dict = {
            "recall": 0,
            "precision": 0,
            "accuracy": 0,
            "dice_score": 0,
        }

        save_image_predictions = True

        with torch.torch.no_grad():
            for dataloader_idx, data in enumerate(validation_loader):
                image = data["image"]
                mask = data["mask"]

                self.optimizer.zero_grad()

                image, mask = (image.to(self.device)).float(), mask.to(self.device)
                image_predicted = self.model(image)
                loss = self.loss_function(image_predicted, mask)
                #print("validation", metric_dict)
                
                
                prev_loss_value = loss_value
                loss_value += loss.item()
                
                metric_values = calculate_metrics(image_predicted, mask, 0.7)


                for k,v in zip(tmp_metrics_dict.keys(), metric_values):
                    tmp_metrics_dict[k] += v 
                #release binding, since data is 
                #not binded somewhere else, the 
                #data is deleted from cache 
                del data, image, mask
            
        tmp_metrics_dict["loss"] = loss_value
        for k,v in tmp_metrics_dict.items():
            tmp_metrics_dict[k] = v/self.validation_dataset_length
        
       
        return tmp_metrics_dict
    def __call__(self, total_epoch, save_model = False) -> None:

        

        for current_epoch in tqdm(range(total_epoch)):
            for phase in ["train", "validation"]:
                dataloader_phase = self.dataloader[phase]
                if phase == "train":
                    metrics_value_for_this_epoch = self.__train(dataloader_phase, current_epoch)
                else:
                    metrics_value_for_this_epoch = self.__validate(dataloader_phase, current_epoch, save_model)
                
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"current learning rate : {lr}")
                print(f"Phase: {phase}. Metrics for current epoch: {current_epoch}:")
                for k, v in metrics_value_for_this_epoch.items():
                    self.metrics_dict[phase][k].append(v)
                    print(f"current epoch {current_epoch} -- {k} : {v}")  
                

                 

def dice_score(predicted, target):
    smooth = 1.

    iflat = predicted.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    
    return ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

if __name__ == "__main__":
    from hope.loss.tverskyloss import TverskyLoss
    from hope.model.unet import Unet
    from hope.model.attention_unet import AttentionUNet
    from hope.train.imageloader import ImageLoader, initiate_dataloader
    from hope.train.testloader.artifical_dataset_loader import (ArtificialDatasetLoader,    
                                                        initiate_dataloader as test_initiate_dataloader)

    path = Path("/mnt/HDD16TB/arams/hope/hope/dataset/FLAIR_dataset/train")
    batch_size = 16
    train_ratio = 0.5
    validation_ratio = 0.5
    augmentation = None
    mode = "train"
    shuffle = True

    dataloader = initiate_dataloader(
            path = path,
            batch_size = batch_size,
            train_ratio = train_ratio,
            validation_ratio = validation_ratio,
            augmentation = augmentation,
            mode = mode,
            shuffle = shuffle,
    )

    #dataloader = test_initiate_dataloader((3,192,192))
    in_channel = 3
    out_channel = 1
    init_features = 32

    model = AttentionUNet(
            in_channels = in_channel, 
            out_channels = out_channel, 
            init_features = init_features
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    train = TrainModel(
        model, 
        optimizer, 
        TverskyLoss, 
        dataloader
        )
    train(total_epoch = 20)