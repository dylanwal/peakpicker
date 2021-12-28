"""
Algorithms for 2D peak detection.

"""
import numpy as np
from numba import njit
from numba.pycc import CC

from peakpicker.utils import get_neighbors_circle, get_neighbors_rectangle
from peakpicker.core_math import unravel_index_single

cc = CC('max_intensity')
cc.verbose = True


@njit
@cc.export('apply_mask_uint', 'u8[:,:](u8[:,:],i8[:,:],u8)')
@cc.export('apply_mask_float', 'f8[:,:](f8[:,:],i8[:,:],f8)')
def apply_mask(mat, mask: np.ndarray, new_value: int = 0) -> np.ndarray:
    for entry in mask:
        mat[entry[0], entry[1]] = new_value

    return mat


@njit
@cc.export('max_intensity_uint64', 'u8[:,:](u8[:,:],u8,u8,f8,f8,f8)')
@cc.export('max_intensity_float64', 'u8[:,:](f8[:,:],u8,u8,f8,f8,f8)')
def max_intensity(
        mat: np.ndarray,
        n: int = 10,
        mask_type: int = 0,
        d1: int = 1,
        d2: int = 1,
        cut_off: float = 0,
) -> np.ndarray:
    """

    Parameters
    ----------
    mat: np.ndarray[:,:] [uint64, float64]
        2D data
    n: float64
        number of peaks expected
    mask_type: int64
        type of mask
            0: None (default)
            1: circle; d1 = radius, d2 = unused
            2: rectangle; d1 = x length, d2 = y length
    d1: float64
        mask dimension
    d2: float64
        mask dimension
    cut_off: float64
        lowest value allowed to be considered a peak

    Returns
    -------
    peaks: np.ndarray[:,:]
        x,y position of peaks found

    """
    mat = np.copy(mat)
    pos = np.empty((n, 2), dtype="uint64")  # x, y
    for i in range(n):
        # get max peak index
        max_ = np.argmax(mat, axis=None)
        xy = unravel_index_single(int(max_), mat.shape)

        # check cutoff
        if mat[xy[0], xy[1]] < cut_off:
            pos = pos[:i, :]
            break

        # add to position list
        pos[i] = xy

        # apply mask to avoid picking same peak again
        if mask_type != 0:
            if mask_type == 1:
                mask = get_neighbors_circle(xy, d1)
                mat = apply_mask(mat, mask)
            elif mask_type == 2:
                mask = get_neighbors_rectangle(xy, d1, d2)
                mat = apply_mask(mat, mask)

    return pos


if __name__ == "__main__":
    cc.compile()
