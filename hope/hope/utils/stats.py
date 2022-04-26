import numpy as np


def sum_of_data(arr: np.ndarray, order: int = 1) -> float:
    """
    This function calculates the n'th order
    of the expectation value. Keep in mind that
    the default value is order = 1

    args:
        arr: The array used for finding expectation value
        order: The order of the expectation value

    return:
        returns the expectation value of order n

    """
    ret = (arr ** order).sum()
    return ret
