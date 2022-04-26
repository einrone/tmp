from pathlib import Path
from typing import Dict, Union


from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hope.train.imageloader import initiate_dataloader, ImageLoader
from hope.train.scheduler import RangeFinder
from hope.model.attention_unet import AttentionUNet
from hope.loss.hybridloss import HybridLoss

def estimate_lr(
    model: nn.Module,
    optimizer: torch.optim,
    loss_function: nn.Module,
    data: Dict[torch.tensor, torch.tensor],
    epoch: int = 100,
    lr: float = 0.01,
    momentum: float = 0.9,
    device : str = "cpu"):

    model.to(device)
    range_finder = RangeFinder(optimizer, epoch)

    losses = []
    for current_epoch in range(0, epoch):
        
        current_lr = [g['lr'] for g in optimizer.param_groups]
        current_mom = [g['momentum'] for g in optimizer.param_groups]
        print('LR: {}, Momentum: {}'.format(current_lr, current_mom))

        image = data["image"]
        mask = data["mask"]

        image, mask = (image.to(device)).float(), mask.to(device)
        image_predicted = model(image)
        loss = loss_function(image_predicted, mask)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        range_finder.step()
        losses.append(loss.item())
        print(f"Epoch: {current_epoch}, loss value: {losses[-1]}, best loss value {np.min(losses)}")

    return losses 

if __name__ == "__main__":
    lr = 0.01
    model = AttentionUNet(3,1,32)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum = 0.9)
   
    loss_function = HybridLoss()
    dataloader = initiate_dataloader(
            path = "/mnt/HDD16TB/arams/hope/hope/dataset/FLAIR_dataset/train",
            batch_size = 64,
            train_ratio = 0.7,
            validation_ratio = 0.3,
            )
    data = iter(dataloader["train"]).__next__()
    
    estimate_lr(
        model= model,
        optimizer = optimizer,
        loss_function = loss_function,
        data = data,
        device = "cuda:0"
    )