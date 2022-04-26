import numpy as np

def swap_brain_plane(
    image: np.ndarray, 
    brain_plane: str
    )-> np.ndarray:

    if brain_plane == "sagittal":
        reshaped_image = image.copy() 
    elif brain_plane == "frontal":
        reshaped_image =np.transpose(image.copy(), (1,0,2))
    elif brain_plane == "horizontal":
        reshaped_image =np.transpose(image.copy(), (2,1,0))
    else:
        raise ValueError(f"The plane given is not sagittal/frontal/horizontal. Got {brain_plane}")
    
    return reshaped_image