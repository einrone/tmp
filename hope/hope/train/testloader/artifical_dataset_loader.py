from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from typing import Union, List

class ArtificialDatasetLoader(Dataset):
    def __init__(self, image_shape: Union[list, tuple]) -> None:
        self.image_shape = tuple(image_shape)
        self.channel = self.image_shape[0]
        self.width = self.image_shape[1]
        self.height = self.image_shape[2]

        img1 = np.ones(self.image_shape)
        
        img2 = np.ones(self.image_shape)
        #img2[:, int(self.height/2), int(self.width/2)] = 0

        img3 = np.zeros(self.image_shape)
    
        self.images = [img1, img2, img3]
        self.mask = [img1[1], img2[1], img3[1]]

    def __len__(self) -> None:
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        ground_truth = self.mask[idx]

        return {"image" : torch.from_numpy(img), "mask" : torch.from_numpy(ground_truth)}

def initiate_dataloader(image_shape):
    train = ArtificialDatasetLoader(image_shape)
    validation = ArtificialDatasetLoader(image_shape)

    train_loader = DataLoader(train, batch_size = 2, shuffle = True)
    validation_loader = DataLoader(validation, batch_size = 2, shuffle = True)

    return {"train" : train_loader, "validation" : validation_loader}
