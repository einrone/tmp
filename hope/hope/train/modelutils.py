from typing import Union
from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import py3nvml
import torch 


class ModelUtils:
    def __init__(
        self, 
        model_name: str, 
        prediction_treshold: float
        ) -> None:
        if prediction_treshold != None:
            if 0 < prediction_treshold < 1.0:
                self.prediction_treshold = prediction_treshold
            else:
                raise ValueError(f"Prediction treshold is not in the interval [0,1], got {prediction_treshold}")
            
        plot_image_path, plot_metric_path, experiment_path = ModelUtils.__experiments(model_name)

        self.plot_image_path = plot_image_path
        self.plot_metric_path = plot_metric_path
        self.experiment_path = experiment_path


    
    def print_performance(self, phase, epoch, metric_dict) -> None:

        for metric_name, metric_value in metric_dict[phase].items():
            if epoch != 0:
                diff = metric_value[-1] - metric_value[-2]
            else:
                diff = metric_value[-1]

            if diff > 0:
                arrow = u'\u2191'
            elif diff < 0:
                arrow = u'\u2193'
            else: 
                arrow = "no change"

            print("---------------------------------------------------------------------------------------------------------------------------")
            print(f"Phase: {phase} --- Current epoch {epoch}: {metric_name}, current value {metric_value[-1]}. Difference: {diff} --- {arrow}. Best value {np.max(metric_value)}")



    def plot(
        self,
        epoch: int, 
        image: torch.tensor,
        mask : torch.tensor,
        image_predicted : torch.tensor,
        rates: tuple,
        id_: str,
        slice_idx: int,
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
        if self.treshold != None:
            image_predicted[image_predicted >= self.prediction_treshold] = 1
            image_predicted[image_predicted < self.prediction_treshold] = 0

        for batch_idx in range(image.shape[0]):
            plt.title(f"epoch: {epoch}, batch number: {batch_idx}")
            fig,axs = plt.subplots(nrows = 2, ncols=2)
            
            axs[0,0].set_title(f"Original image.\n Patient: {id_[batch_idx]} slice {slice_idx[batch_idx]}")
            axs[0,0].imshow(image[batch_idx][1], cmap = "gray", vmin =image[batch_idx][1].min(), vmax = image[batch_idx][1].max() + 2)
            fig.tight_layout()

            axs[0,1].set_title(f"Pred (red) and GT (green)")
            axs[0,1].imshow(image_predicted[batch_idx][0], cmap = "Reds")
            axs[0,1].imshow(mask[batch_idx], cmap = "Greens")
            fig.tight_layout()

            axs[1,0].set_title(f"Pred. TP: {rates[0][batch_idx]:.2f}, FP: {rates[1][batch_idx]:.2f}, \n FN: {rates[2][batch_idx]:.2f}, TN: {rates[3][batch_idx]:.2f}")
            axs[1,0].imshow(image_predicted[batch_idx][0])
            fig.tight_layout()

            axs[1,1].set_title("Original mask")
            axs[1,1].imshow(mask[batch_idx])
            fig.tight_layout()

            plt.savefig(f"{str(self.plot_image_path)}/image_{epoch}_batch_{batch_idx}_lesion_size_{str(mask[batch_idx].sum())}.png")
            plt.close()
    def plot_ROC_and_PR(self, metrics_dict : dict)-> None:
        pass

    def save_metrics(self, metrics_dict: dict, save_to_csv = True, PR_and_ROC_curve = True) -> None:
        train_values = metrics_dict["train"]
        validation_values = metrics_dict["validation"]

        if list(train_values.keys()) == list(validation_values.keys()):
            metric_keys = list(train_values)
        else:
            raise ValueError("The keys in metric_dict for train and validation does not correspond")
        
        for idx, metric_name in enumerate(metric_keys):
            length = np.arange(len(train_values[metric_name]))

            plt.title(f"Median of {metric_name} vs epoch")
            plt.plot(length, train_values[metric_name])
            plt.plot(length, validation_values[metric_name])
            plt.xlabel("epoch")
            plt.ylabel(f"{metric_name}")
            plt.legend(["train", "validation"],loc="upper right")

            plt.savefig(self.plot_metric_path / f"{metric_name}.png")
            plt.close()
        
        if PR_and_ROC_curve == True:
            pass

        if save_to_csv == True:
            df_train = pd.DataFrame.from_dict(metrics_dict["train"])
            df_train.to_csv(self.plot_metric_path / "train_metrics.csv")

            df_valid = pd.DataFrame.from_dict(metrics_dict["validation"])
            df_valid.to_csv(self.plot_metric_path / "validation_metrics.csv")

        
    @staticmethod
    def __experiments(modelname: str) -> Path:
            CURRENT_WORKING_DIR = Path(__file__).parent
            ROOT_TRAIN_FOLDER = CURRENT_WORKING_DIR / "predictions4"
            
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
            
            return EXPERIMENT_PATH_IMAGES, EXPERIMENT_PATH_STATISTICS, EXPERIMENT_PATH
    

    def initiate_scheduler(self, scheduler_name : str) -> torch.optim.lr_scheduler:

        if scheduler_name == "OneCycleLR":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr = self.learning_rate*10,
                steps_per_epoch = self.train_dataset_length,
                epochs = self.epoch,
                anneal_strategy = "cos",
                cycle_momentum = True,
                base_momentum = 0.87,
                max_momentum = 0.95,
                )

        elif scheduler_name == "StepLR":
            exponent = np.floor(np.log10(np.abs(self.learning_rate))).astype(int)
            if exponent <= -3:
                gamma_value = 1.1
            else:
                gamma_value = 0.9
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size = 25,
                gamma = gamma_value
            )

        elif scheduler_name  == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode = "min",
                factor = 0.9,
                patience = 5,
                threshold = 1e-3,
                threshold_mode = "rel"
            )
        elif scheduler_name == "LambdaLR":
            lambda_function = lambda epoch: 0.9 if epoch <= 200 == 0 else 1.1
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda_function,
            )
        else:
            warnings.warn("The given scheduler name is not ReduceLROnPlateau, StepLR or OneCycleLR. Using None")
            return None

    def prepare_and_set_GPU(self) -> torch.device:
        """
        Input: None
        Return:
            torch.cuda.device. If gpu is available a CUDA device is used, if not
            it will automatically be set to cpu
        """
        gpu_index = None
        if torch.cuda.is_available():
            number_of_gpu = torch.cuda.device_count()
            if number_of_gpu == 1:
                print("Using device: cuda:0")
                return torch.device("cuda:0")
            else:
                free_gpus = py3nvml.get_free_gpus()
                indices_for_free_gpu = [i for i, x in enumerate(free_gpus) if x == True]
        else:
            indices_for_free_gpu = [] #no gpu on system, return empty list to turn on cpu


        if not indices_for_free_gpu:
            print("Using device: cpu")
            return torch.device("cpu")
        else:
            gpu_index = indices_for_free_gpu[0]
            print(f"Using device: cuda {gpu_index}")
            # print(("cuda:{}".format(gpu_index) if gpu_index!= None else "cpu")

            return torch.device("cuda:{}".format(gpu_index))

    def save_model(
        self, 
        epoch: int, 
        lr: float = None
        ) -> torch.save:
        """
            make docstring
        """

        state_dict = {
            "epoch": epoch,
            "model_name" : type(self.model).__name__,
            "optimizer_name": type(self.optimizer).__name__,
            "model_state_dict": self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'validation_loss_value': self.best_loss,
            'best_dice': self.best_dsc_value,
            'loss_function_name': type(self.loss_function).__name__,
            'learning_rate' : "N/A" if lr == None else lr 
            }
        torch.save(state_dict, self.experiment_path / "best_model.pt")
    
    def resume_checkpoint(
        self, 
        model_path: Union[str, Path],
        resume: bool = False, 
        resume_epoch: bool = False,
        ):

        if resume == True:
            checkpoint = torch.load(model_path, map_location=self.device)
            if resume_epoch == True:
                print(f"Resuming from epoch: {checkpoint['epoch']}")
                self.start_epoch = checkpoint["epoch"]

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.model.eval()

        else:
            pass
    
            

    
if __name__ == "__main__":
    utils = ModelUtils(22,0.7)
    


    print(torch.version.cuda)