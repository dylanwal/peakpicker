import numpy as np

from .max_intensity import max_intensity_uint64, max_intensity_float64


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
    n = float(n)
    mask_type = int(mask_type)
    d1 = float(d1)
    d2 = float(d2)
    cut_off = float(cut_off)
    if np.issubdtype(mat.dtype, np.unsignedinteger):
        mat = mat.astype("uint64")
        peaks = max_intensity_uint64(mat, n, mask_type, d1, d2, cut_off)
    else:
        mat = mat.astype("float64")
        peaks = max_intensity_float64(mat, n, mask_type, d1, d2, cut_off)

    return peaks