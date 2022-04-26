import numpy as np 
import cc3d

def clean_mask(
    mask: np.ndarray,
    connectivity: int, 
    min_pixel: int = 4,
    dtype: np.dtype = np.int32
):
    dim = mask.shape
    assert len(dim) != 2, f"The mask must be a 2D slice, got {dim}"

    mask = mask.astype(dtype = dtype)
    for idx, mask_slice in enumerate(mask):
        labels_out, num = cc3d.connected_components(
                mask_slice, connectivity=connectivity, return_N=True
            )
        stats = cc3d.statistics(labels_out)

        for label, count in enumerate(stats["voxel_counts"]):
            if count == 0:
                continue
            else:
                if count < min_pixel:
                    labels_out[labels_out == label] = 0
                else:
                    pass
        mask[idx] = labels_out
    
    mask[mask != 0] = 1

    return mask
