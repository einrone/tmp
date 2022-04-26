import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.3
BETA = 0.7
DELTA = 0.7
GAMMA = 4/3

class TverskyBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA,delta = DELTA, gamma = GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + beta*FP + (1-beta)*FN + smooth) 
        Tverskyloss = (1 - Tversky)**gamma

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
        
        return Bce*delta + (1 - delta)*Tversky 