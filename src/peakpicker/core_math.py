"""
Numpy functions re-written to play nice with numba.py

"""

import numpy as np
from numba import njit
from numba.pycc import CC

cc = CC('core_math')
cc.verbose = True


@njit
@cc.export('dot', 'f8(u8[:],u8[:])')
def dot(x, y):
    """ Dot Product for indexes. """
    out = 0
    for i in range(x.size):
        out += x[i]*y[i]
    return out


@njit
@cc.export('unravel_index_single', 'u8[:](u8,u8[:])')
def unravel_index_single(flat_index, shape: tuple) -> np.ndarray:
    """ Unravel index

    Convert flatten index into tuple index.

    Parameters
    ----------
    flat_index: uint64
        flatten index
    shape: tuple
        shape of matrix the flatten index is about

    Returns
    -------
    row: uint64
        row index
    col: uint64
        column index
    """
    out = np.empty(2, dtype="uint64")
    out[0] = int(flat_index / shape[1])
    out[1] = flat_index - out[0] * shape[1]

    return out


@njit
@cc.export('unravel_index', 'u8[:,:](u8[:],u8[:])')
def unravel_index(flat_index: np.ndarray, shape: tuple) -> np.ndarray:
    """ Unravel index

    Convert flatten index into tuple index.

    Parameters
    ----------
    flat_index: uint64
        flatten index
    shape: tuple
        shape of matrix the flatten index is about

    Returns
    -------
    row: uint64
        row index
    col: uint64
        column index
    """
    out = np.empty((2, flat_index.size), dtype="uint64")
    for i, index in enumerate(flat_index):
        out[i, 0] = int(index/shape[1])
        out[i, 1] = index - out[i, 0] * shape[1]

    return out


@njit
@cc.export('flatten_index', 'u8(u8[:],u8[:])')
def flatten_index(index: tuple, shape: tuple) -> int:
    """ Unravel index

    Convert flatten index into tuple index.

    Parameters
    ----------
    index: tuple
        index (x, y)
    shape: tuple
        shape of matrix the flatten index is about

    Returns
    -------
    index: int64
        flatten index
    """
    return index[0] * shape[0] + index[1]


@njit
@cc.export('in1d', 'b1[:](u8[:], u8[:])')
def in1d(a, b):
    return np.array([item in b for item in a])


@njit
@cc.export('flat', 'u8[:](u8[:,:], u8[:])')
def flat(a, b):
    out = np.empty(b.size, dtype=a.dtype)
    for i, index in enumerate(b):
        out[i] = a.flat[index]
    return out


if __name__ == "__main__":
    cc.compile()
