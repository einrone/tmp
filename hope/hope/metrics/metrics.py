import numpy 
import torch 
def IoU(tp, fn, fp):
    smooth = 1.0
    return (tp + smooth)/(tp + fp + fn + smooth)
def recall(tp, fn):
    eps = 1.0
    return (tp + eps)/(tp + fn + eps)

def precision(tp, fp):
    eps = 1.0
    return (tp + eps)/(tp + fp + eps)

def accuracy(tp, tn, fn, fp):
    """ 
        This function uses balanced
        accuracy formula, so it also
        inbalanced dataset

        args:
            tp: true positive
            tn: true negative
            fn: false negative
            fp: false positive
        
        return:
            the balanced accuracy value
    """
    eps = 1.0
    TPR = (tp + eps)/(tp + fn + eps) #true positive rate, also called recall
    TNR = (tn + eps)/(tn + fp + eps) #true negative rate, also called specificity

    return (TPR + TNR)/2.0

def dicescore(TP, FP, FN):
    smooth = 1.0
    return (2*TP + smooth) / (2*TP + FP + FN + smooth)
    