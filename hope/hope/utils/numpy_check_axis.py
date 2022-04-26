import numpy as np

def numpy_check_axis(image: np.ndarray):
        axis1, axis2, axis3 = image.shape
        # 0,    1,      2
        new_image = image.copy()

        crit1 = np.abs(axis2 - axis3) < 1e-8
        crit2 = np.abs(axis1 - axis3) < 1e-8
        crit3 = np.abs(axis1 - axis2) < 1e-8

        if crit1 == True:
            pass

        elif crit2 == True:
            H = 2
            W = 0
            D = 1

            new_image = new_image.transpose(D, H, W)
        else:
            H = 0
            W = 1
            D = 2

            new_image = new_image.transpose(D, H, W)

        return new_image