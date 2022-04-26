from pathlib import Path
from typing import Dict, Union

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from hope.train import TrainModel
from hope.metrics.metrics import recall, precision, accuracy, dicescore

class ModelPerformance(TrainModel):
    def __init__(
        self, 
        model_name: str, 
        path: Union[str, Path], 
        prediction_treshold: float
        ) -> None:

        self.path = Path(path)

        if 0 < prediction_treshold < 1.0:
            self.prediction_treshold = prediction_treshold
        else:
            raise ValueError(f"Prediction treshold is not in the interval [0,1], got {prediction_treshold}")
        plot_image_path, plot_metric_path = experiments(model_name, path)

        self.plot_image_path = plot_image_path
        self.plot_metric_path = plot_metric_path
    
    def print_metric(self, epoch, metric_dict) -> None:

        for metric_name, metric_value in metric_dict.items():
            diff = np.abs(metric_value[-1] - metric_value[-2])
            if diff > 0:
                arrow = u'\u2191'
            elif diff < 0:
                arrow = u'\u2193'
            else: 
                arrow = "no change"
            print("---------------------------------------------------------------")
            print(f"Current epoch {epoch}: {metric_name}, current value {metric_value[-1]}. Difference: {diff} --- {arrow}")


    def plot(
        self,
        epoch: int, 
        image: torch.tensor,
        mask : torch.tensor,
        image_predicted : torch.tensor,
    ) -> None:

        """
            Add docstrings later,
            and fix this for num_channels.
            hardcoded 3 channels.

        """
        image = image.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        image_predicted = image_predicted.detach().cpu().numpy()
        
        #tresholding all pixel values over 0.5 = 1 and vice versa (image prediction cleaning...?)
        image_predicted[image_predicted >= self.treshold] = 1
        image_predicted[image_predicted < self.treshold] = 0

        for batch_idx in range(image.shape[0]):
            plt.title(f"epoch: {epoch}, batch number: {batch_idx}")
            fig,axs = plt.subplots(3)
            
            axs[0].set_title("Original image")
            axs[0].imshow(image[batch_idx][1])
            fig.tight_layout()

            axs[1].set_title("predicted mask")
            axs[1].imshow(image_predicted[batch_idx][0])
            fig.tight_layout()

            axs[2].set_title("Original mask")
            axs[2].imshow(mask[batch_idx])
            fig.tight_layout()

            plt.savefig(f"{str(self.plot_image_path)}/image_{epoch}_batch_{batch_idx}_lesion_size_{str(mask[batch_idx].sum())}.png")
            plt.close()
    
    def save_metrics(self, metrics_dict: dict, save_to_csv) -> None:
        train_values = metrics_dict["train"]
        validation_values = metrics_dict["validation"]

        if list(train_values.keys()) == list(validation_values.keys())
            metric_keys = list(train_values)
        else:
            raise ValueError("The keys in metric_dict for train and validation does not correspond")
        
        
        for metric_name in metric_keys:
            length = np.arange(train_values[metric_name])

            plt.title(f"Median of {metric_name} vs epoch")
            plt.plot(length, train_values[metric_name])
            plt.plot(length, validation_values[metric_name])
            plt.legend(["train", "validation"],loc="upper right")
            plt.savefig(self.plot_metric_path / f"{metric_name}.png")
            plt.close()
        

def experiments(modelname: str) -> Path:
        CURRENT_WORKING_DIR = Path(__file__).parent
        ROOT_TRAIN_FOLDER = CURRENT_WORKING_DIR / "predictions"
        
        list_of_paths = list(ROOT_TRAIN_FOLDER.glob("*"))
        if ".DS_Store" in list_of_paths:
            list_of_paths.remove(".DS_Store")
        else:
            pass

        MODELNAME_PREDICTION_FOLDER = ROOT_TRAIN_FOLDER / modelname
        if not MODELNAME_PREDICTION_FOLDER in list_of_paths:

            EXPERIMENT_PATH = MODELNAME_PREDICTION_FOLDER / "experiment/1"
            EXPERIMENT_PATH.mkdir(parents=True)

            EXPERIMENT_PATH_IMAGES = EXPERIMENT_PATH / "image_predictions"
            EXPERIMENT_PATH_IMAGES.mkdir()

            EXPERIMENT_PATH_STATISTICS = EXPERIMENT_PATH / "metrics"
            EXPERIMENT_PATH_STATISTICS.mkdir()



        else:

            #the model image prediction exist
            #no need to create a folder with name
            #MODELNAME_PREDICTION_FOLDER
            #creating folder for number of model is runned

            EXPERIMENT_PATH = MODELNAME_PREDICTION_FOLDER / "experiment"
            list_of_experiments_runs = list(EXPERIMENT_PATH.glob("*"))
            
            if ".DS_Store" in list_of_experiments_runs:
                list_of_experiments_runs.remove(".DS_Store")
            else:
                
                list_of_experiments_runs = [int(idx.name) for idx in list_of_experiments_runs]
                list_of_experiments_runs = np.array(list_of_experiments_runs)

            largest_experiment_number = np.max(list_of_experiments_runs)
            next_experiment_number = largest_experiment_number + 1

            EXPERIMENT_PATH = EXPERIMENT_PATH / str(next_experiment_number)
            
            EXPERIMENT_PATH_IMAGES = EXPERIMENT_PATH / "image_predictions"
            EXPERIMENT_PATH_IMAGES.mkdir(parents=True)

            EXPERIMENT_PATH_STATISTICS = EXPERIMENT_PATH / "metrics"
            EXPERIMENT_PATH_STATISTICS.mkdir(parents=True)
        
        return EXPERIMENT_PATH_IMAGES, EXPERIMENT_PATH_STATISTICS
        
