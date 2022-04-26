from pathlib import Path
from typing import Dict, Union, Optional
import warnings
from collections import defaultdict
import time

from tqdm import tqdm

import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.morphology import binary_opening, binary_closing

from hope.train.imageloader import ImageLoader
from hope.train.modelutils import ModelUtils
from hope.metrics.calculate_metrics import calculate_metrics


seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class TrainModel(ModelUtils):
    def __init__(
        self, 
        model : torch.nn.Module, 
        optimizer : torch.optim.Optimizer, 
        loss_function : nn.Module, 
        dataloader : Dict[ImageLoader, ImageLoader], 
        epoch : int,
        learning_rate: float = 5e-4,
        treshold: float = 0.7,
        scheduler_name: str = "OneCycleLR",
        resume_from_checkpoint : bool = False, 
        resume_epoch : bool  = False,
        model_path: Union[str, Path] = None,
        specific_device: Optional[str] = None
        ) -> None:
        
        super(TrainModel,self).__init__(
            model_name = type(model).__name__,
            prediction_treshold = treshold
        )
        """
            make docstring
        """
        # internal definition of variables
        if specific_device != None:
            print(f"Using device: {specific_device}")
            self.device = specific_device
        else:
            self.device = self.prepare_and_set_GPU()
        
        self.state_dict = None
        self.scheduler_name = scheduler_name
        self.dataloader = dataloader

        #dataset length when having batch_size = size
        self.train_dataset_length = len(self.dataloader["train"])
        self.validation_dataset_length = len(self.dataloader["validation"])
        
        #float and integer values
        self.learning_rate = learning_rate
        self.start_epoch = 0
        self.treshold = treshold
        self.epoch = epoch

        #initiate model
        self.model = model#model(self.in_channel, self.out_channel, self.init_features)
        self.model.to(self.device)

        #initiate optimizer
        self.optimizer = optimizer 
        
        #fetch scheduler
        if self.scheduler_name != None:
            self.scheduler = self.initiate_scheduler(scheduler_name)
        else:
            self.scheduler = None

        if resume_from_checkpoint == True:
            model_path = Path(model_path)
            if model_path.is_file():
                self.resume_checkpoint(
                    model_path = model_path,
                    resume = resume_from_checkpoint, 
                    resume_epoch = resume_epoch,
                )
            else:
                warnings.warn(f"The following saved model path is not a file. Not resuming checkpoint!")
                print(f"Got path {model_path}")
                
        else:
            pass
        
        #initiate the loss function class
        self.loss_function = loss_function() 

    
        self.initiate_metrics_dict()
        self.best_dsc_value = 0
        self.best_loss = 0
 
    
    def initiate_metrics_dict(self) -> None:
        self.metrics_dict = {
            "train":
                {
                    "recall": [],
                    "precision": [],
                    "IoU": [],
                    "dice_score": [],
                    "loss": [],
                    "TP": [],
                    "FP":[],
                    "FN":[],
                    "TN": []
                },
            "validation":
                {
                    "recall": [],
                    "precision": [],
                    "IoU": [],
                    "dice_score": [],
                    "loss": [],
                    "TP": [],
                    "FP":[],
                    "FN":[],
                    "TN": []
                }
        }
    
    """def failure_analysis(
        self,
        image,  
        prediction, 
        ground_truth, 
        id_value, 
        slice_idx,
        epoch
        ) -> None:

        for batch in (image.shape[0]):
            metric_values = calculate_metrics(prediction, ground_truth)
            recall = metric_values[0]
            dsc = metric_values[-1]"""
    
    def __opening_closing(
        self, 
        prediction: torch.tensor
        )-> np.ndarray:

        prediction = prediction.detach().cpu().numpy()
        
        
        for batch_idx in range(prediction.shape[0]):
            img = prediction[batch_idx][0]
            img = binary_opening(img)
            img = binary_closing(img)
            prediction[batch_idx][0] = img
        return prediction

    def __train(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        epoch: int,
        ) -> None:
        """
        THIS METHOD IS NOT DONE
        """
        self.model.train()
       
        loss_value = 0

        tmp_metrics_dict = {
            "recall": 0,
            "precision": 0,
            "IoU": 0,
            "dice_score": 0,
            "TP": 0,
            "FP":0,
            "FN":0,
            "TN":0
        }

        with torch.set_grad_enabled(True):
            for dataloader_idx, data in enumerate(train_loader):
                self.optimizer.zero_grad()

                image = data["image"]
                mask = data["mask"]

                image, mask = (image.to(self.device)).float(), mask.to(self.device)
                image_predicted = self.model(image)
                loss = self.loss_function(image_predicted, mask)

                loss.backward()
                self.optimizer.step()

                if self.scheduler != None and self.scheduler_name == "OneCycleLR":
                    self.scheduler.step()
                      
                else:
                    pass
                

                loss_value += loss.item()
                #image_predicted = self.__opening_closing(image_predicted)

                metric_values = calculate_metrics(image_predicted, mask, self.treshold, return_metrics=True)

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
        ) -> None:
        """
        THIS METHOD IS NOT DONE
        """
        self.model.eval()

        loss_value = 0
        tmp_metrics_dict = {
            "recall": 0,
            "precision": 0,
            "IoU": 0,
            "dice_score": 0,
            "TP": 0,
            "FP":0,
            "FN":0,
            "TN":0
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
                if save_image_predictions == True:
                    if epoch % 10 == 0:

                        # saves the mask, prediction and original image
                        # for the sake of analysis
                        #assert image.shape() == mask.shape()
                        TP, FP, FN, TN = calculate_metrics(image_predicted, mask, self.treshold, return_metrics = False)
                        id_ = data["id"]
                        slice_idx = data["slice_idx"]
                        rates = (TP, FP, FN, TN)
                        self.plot(epoch, image, mask, image_predicted, rates, id_, slice_idx)
                    
                        save_image_predictions = False 
                    else:
                        pass 
                else:
                    pass
                    
                #image_predicted = self.__opening_closing(image_predicted)
                metric_values = calculate_metrics(image_predicted, mask, self.treshold, return_metrics = True)


                for k,v in zip(tmp_metrics_dict.keys(), metric_values):
                    tmp_metrics_dict[k] += v 
                #release binding, since data is 
                #not binded somewhere else, the 
                #data is deleted from cache 
                del data, image, mask
            
        tmp_metrics_dict["loss"] = loss_value
        for k,v in tmp_metrics_dict.items():
            tmp_metrics_dict[k] = v/self.validation_dataset_length
        
        if self.scheduler != None:
            if self.scheduler_name == "ReduceLROnPlateau":
                self.scheduler.step(tmp_metrics_dict["loss"])
            else:
                self.scheduler.step()
        else:
            pass

        if save_model == True:
            if self.best_dsc_value < tmp_metrics_dict["dice_score"]:
                self.save_model(epoch = epoch, lr = None)
                self.best_dsc_value = tmp_metrics_dict["dice_score"]
                self.best_loss = tmp_metrics_dict["loss"]
                print(f"Best median dice score: {self.best_dsc_value} at epoch {epoch}")
                print(f"Model saved")
            else:
                pass
        else:
            pass
        
        return tmp_metrics_dict

    def __call__(
        self, 
        save_model : bool = True, 
        save_result_to_csv = False,
        show_progress = True,
        ) -> None:

        """
            make docstring
        """
        print(f"Starting training of the model: {type(self.model).__name__}")

        start = time.time()
        for current_epoch in tqdm(range(self.start_epoch, self.epoch)):
            
            for phase in ["train", "validation"]:
                dataloader_phase = self.dataloader[phase]
                
                
                if phase == "train":
                    metrics_value_for_this_epoch = self.__train(dataloader_phase, current_epoch)
                else:
                    metrics_value_for_this_epoch = self.__validate(dataloader_phase, current_epoch, save_model)
                
                
                for k, v in metrics_value_for_this_epoch.items():
                    self.metrics_dict[phase][k].append(v)
                
                if show_progress == True:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"current learning rate : {lr}")
                    print(f"Phase: {phase}. Metrics for current epoch: {current_epoch}:")
                    self.print_performance(phase, current_epoch, self.metrics_dict)
                else:
                    pass


            self.save_metrics(self.metrics_dict, save_result_to_csv)
        end = time.time()
        print("The model has been trained success_fully")
        print(f"validation image prediction saved at {self.plot_image_path}")
        print(f"Metrics plots saved at {self.plot_metric_path}")
        print(f"The best dice score: {self.best_dsc_value}")
        print(f"The best loss value: {self.best_loss}")
        print(f"Total time spent during training {(end - start)/60} minutes")
        print("train script exited")

        return self.best_dsc_value, self.best_loss, self.experiment_path
        
if __name__ == "__main__":
    from torchvision.transforms.functional import InterpolationMode
    from hope.model.nested_unet import NestedUNet
    from hope.loss.haussdorff_dt_loss import HausdorffDTLoss
    from hope.loss.focaltverskyloss import FocalTverskyLoss
    from hope.loss.hybridloss import HybridLoss
    from hope.model.unet import Unet
    from hope.model.test import UNet
    from hope.model.attention_unet import AttentionUNet
    from hope.train.imageloader import ImageLoader, initiate_dataloader
    from hope.augmentation.initialize_augmentation import initialize_augmentation
    from hope.train.testloader.artifical_dataset_loader import (ArtificialDatasetLoader,    
                                                        initiate_dataloader as test_initiate_dataloader)

    for i in [64]: #it was 64
        print(f"using batch size {i}")
        #path = Path("/mnt/HDD16TB/arams/hope/hope/dataset/FLAIR_new_dataset_volume_norm2_splitted/train")
        path = Path("/mnt/HDD16TB/arams/hope/hope/dataset/fold/train_ratio_0.85_validation_ratio_0.15_fold_26.json")
        batch_size = i
        train_ratio = 0.9
        validation_ratio = 0.1
        augmentation = initialize_augmentation(
            probability = 0.5,
            angle = [-15,15],
            interpolation_mode = InterpolationMode.BILINEAR
            )
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
            save_fold = False,
            path_to_save_fold = None,#,"/mnt/HDD16TB/arams/hope/hope/dataset/fold",
            use_existing_fold = True
        )

        """dataloader = test_initiate_dataloader(
            (3,192,192)
        )"""
        

        in_channel = 3
        out_channel = 1
        init_features = 32

        """model = Unet(
            in_channels = in_channel, 
            out_channels = out_channel, 
            init_features = init_features, 
            pretrained_model_torch_model = False
            )"""
        
        """model = NestedUNet(
            in_ch = in_channel, 
            out_ch = out_channel 
        )"""
        

        model = AttentionUNet(
            in_channels = in_channel, 
            out_channels = out_channel, 
            init_features = init_features
        )
        #0.001
        learning_rate = 0.005 #it was 0.02 for run 46
        #treshold was 0.5 for run44 and 46
        #optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.90, nesterov = False) #it was 0.8 for run 46
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        train = TrainModel(
            model = model, 
            optimizer = optimizer,
            loss_function = FocalTverskyLoss,
            dataloader = dataloader,
            epoch = 150,
            learning_rate = learning_rate,
            treshold = 0.5,
            scheduler_name = "StepLR",
            resume_from_checkpoint = False,
            resume_epoch = False,
            model_path = None #"/mnt/HDD16TB/arams/hope/hope/train/predictions/AttentionUNet/experiment/60/best_model.pt"
            )

        train(
            save_model = True,
            save_result_to_csv = True
        )