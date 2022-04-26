import numpy as np
import scipy


def population_normalize(arr: np.ndarray, population_size):
    standard_div = _population_std(arr, population_size)
    mean_value = _population_mean(arr, population_size)

    return (arr - mean_value) / standard_div


def volume_normalize(arr: np.ndarray):
    standard_div = arr.std()
    mean_value = arr.mean()

    return (arr - mean_value) / standard_div


# def channel_normalize(arr: np.ndarray):
#    normalization_values = np.zeros(arr.shape[0])
