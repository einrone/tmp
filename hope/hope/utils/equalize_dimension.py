import numpy as np

from hope.utils.numpy_check_axis import numpy_check_axis

def equalize_dimension(image: np.ndarray)->np.ndarray:
    image = numpy_check_axis(image) 

    dim = image.shape 
    diff = np.abs(dim[0] - dim[1])
    a = int(np.ceil(diff/2))
    b = int(np.floor(diff/2))

    return np.pad(image, ((a,b), (0,0), (0,0)), mode = "constant")
