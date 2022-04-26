import torch
import numpy as np 
from hope.metrics.metrics import recall, precision, accuracy, dicescore, IoU

def calculate_metrics(
    predicted_image : torch.tensor, 
    ground_truth: torch.tensor,
    treshold: float,
    return_metrics: bool = True
    ) -> dict:
    """
        This function calculates statistical
        metrics by using tp,fp,fn, tn.

        args:
            predicted_image: An image predicted from an arbitrary cnn
            ground_truth: The ground truth from the training dataset.
        return:
            A dictonary containing recall, precision, accuracy and
            dice score.
    """
    
    predicted = predicted_image.detach().cpu().squeeze(dim = 1).numpy()
    target = ground_truth.detach().cpu().clone().numpy()
  
    #tresholding the predicted image to be 0 and 1
    if treshold != None:
        if 0 < treshold < 1.0:
            predicted[predicted >= treshold] = 1.0
            predicted[predicted < treshold] = 0.0
        else:
            pass
    else:
        pass

    if return_metrics == True:
        dim_to_sum = (0,1,2)
    else:
        dim_to_sum = (1,2)

    TP = (predicted * target).sum(axis = dim_to_sum)
    FP = ((1-target) * predicted).sum(axis = dim_to_sum)
    FN = (target * (1-predicted)).sum(axis = dim_to_sum)
    TN = ((1- predicted)*(1-target)).sum(axis = dim_to_sum)

    if return_metrics == True:
        return (
            recall(TP, FN),
            precision(TP, FP),
            IoU(TP, FN, FP),
            dicescore(TP, FP, FN),
            TP,
            FP,
            FN,
            TN
            )
    else:
        return (
            TP,
            FP,
            FN,
            TN
        )

    


if __name__ == "__main__":
    a = np.array([[1,0,1],[0,1,0], [0,0,0]])
    b = np.array([[1,0,1],[0,0,0], [0,1,0]])

    a = torch.from_numpy(a)
    b = torch.from_numpy(b)

    print(calculate_metrics(a,b))