import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
# remember weight false positive, thus this more crucial. 
#It says that there is no disease when it is.
#Thus, is better to have more FP than FN
# run 44 had  these values


ALPHA1 = 0.9 #used to be 0.3
ALPHA2 = 0.6
BETA = 0.1 # used to be 0.7
GAMMA1 = 4/3
GAMMA2 = 2.5
LMBDA = 0.4


"""ALPHA1 = 0.85
ALPHA2 = 0.25
BETA = 0.15
GAMMA1 = 4/3
GAMMA2 = 2
LMBDA = 0.5
"""

class HybridLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(HybridLoss, self).__init__()

    def forward(
        self, 
        inputs, 
        targets, 
        smooth=1, 
        alpha1 = ALPHA1, 
        alpha2 = ALPHA2, 
        beta = BETA, 
        gamma1 = GAMMA1, 
        gamma2=GAMMA2, 
        lmbda=LMBDA
        ):

        
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha1 * FP + beta * FN + smooth)
        focaltversky = (1 - Tversky) ** gamma1

        BCE = F.binary_cross_entropy(inputs, targets.float(), reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha2 * (1 - BCE_EXP) ** gamma2 * BCE

        return focal_loss * lmbda + (1 - lmbda) * focaltversky