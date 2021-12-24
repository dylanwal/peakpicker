"""
Algorithms for 2D peak detection.

"""
import numpy as np
from numba import njit
from numba.pycc import CC

cc = CC('max_intensity')
cc.verbose = True


@njit
@cc.export('unravel_index', 'u8[:](u8,u8[:])')
def unravel_index(flat_index: int, shape: tuple) -> np.ndarray:
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
    out[0] = int(flat_index/shape[1])
    out[1] = flat_index - out[0] * shape[1]

    return out


@njit
@cc.export('create_grid', 'u8[:,:](u8,u8)')
def create_grid(x_len: int = 10, y_len: int = 10) -> np.ndarray:
    """ Grid indexes

    Generates a column matrix of x,y index for a square grid around (0,0).

    Parameters
    ----------
    x_len: int
        x size of grid
    y_len: int
        y size of grid

    Returns
    -------
    xy: np.ndarray
        xy indexes of grid
    """
    x_len = x_len+2
    y_len = y_len+2
    xy = np.empty((x_len * y_len, 2), dtype="uint64")
    for i in range(x_len):
        for ii in range(y_len):
            xy[i*y_len+ii, 0] = i
            xy[i*y_len+ii, 1] = ii

    return xy


@njit
@cc.export('dot', 'f8(u8[:],u8[:])')
def dot(x, y):
    """ Dot Product for indexes. """
    out = 0
    for i in range(x.size):
        out += x[i]*y[i]
    return out


@njit
@cc.export('norm_row_axis', 'f8[:](u8[:,:])')
def norm_row_axis(xy):
    """ Frobenius norm across row"""
    distance = np.empty(xy.shape[0], dtype="float64")
    for i, row in enumerate(xy):
        distance[i] = np.sqrt(dot(row, row))

    return distance


@njit
@cc.export('get_circle_mask', 'i8[:,:](u8[:],f8)')
def get_circle_mask(xy: tuple, r: float = 10) -> np.ndarray:
    """ Get circle mask

    Generates indexes that are within a circle of radius "r" from the point xy

    Parameters
    ----------
    xy: Tuple
        center of circle
    r: float
        radius of circle

    Returns
    -------
    xy: np.ndarray
        xy indexes within circle
    """
    # create grid (only first quadrant)
    dim = int(r)
    grid = create_grid(dim, dim)
    distance = norm_row_axis(grid)
    mask = distance < r
    grid = grid[mask]

    # mirror grid into 4 quadrants
    out = np.empty((grid.shape[0]*4, 2), dtype="int64")
    neg_x = np.ones_like(grid, dtype="int8")
    neg_x[:, 0] = -1
    neg_y = np.ones_like(grid, dtype="int8")
    neg_y[:, 1] = -1
    out[:grid.shape[0], :] = grid
    out[grid.shape[0]:2*grid.shape[0], :] = grid * neg_x
    out[2*grid.shape[0]:3*grid.shape[0], :] = grid * neg_y
    out[3*grid.shape[0]:, :] = grid * neg_x * neg_y

    out[:, 0] = out[:, 0] + xy[0]
    out[:, 1] = out[:, 1] + xy[1]

    # remove negative index (happens if point is near edge of matrix)
    mask = out[:, 0] > 0
    out = out[mask]
    mask = out[:, 1] > 0
    out = out[mask]

    return out


@njit
@cc.export('get_square_mask', 'i8[:,:](u8[:],u8,u8)')
def get_square_mask(xy: tuple, x_len: int = 10, y_len: int = 10) -> np.ndarray:
    """ get square mask

    Generates indexes that are within a square of x_len x y_len around point xy

    Parameters
    ----------
    xy: Tuple
        center of square
    x_len: int64
        x length of rectangle
    y_len: int64
        y length of rectangle

    Returns
    -------
    xy: np.ndarray
        xy indexes within square
    """
    # create grid (only first quadrant)
    grid = create_grid(int(x_len), int(y_len))

    out = np.empty_like(grid, dtype="int64")
    out[:, 0] = grid[:, 0] + xy[0] - np.round(x_len)
    out[:, 1] = grid[:, 1] + xy[1] - np.round(y_len)

    # remove negative index (happens if point is near edge of matrix)
    mask = out[:, 0] > 0
    out = out[mask]
    mask = out[:, 1] > 0
    out = out[mask]

    return out


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
        xy = unravel_index(int(max_), mat.shape)

        # check cutoff
        if mat[xy[0], xy[1]] < cut_off:
            pos = pos[:i, :]
            break

        # add to position list
        pos[i] = xy

        # apply mask to avoid picking same peak again
        if mask_type != 0:
            if mask_type == 1:
                mask = get_circle_mask(xy, d1)
                mat = apply_mask(mat, mask)
            elif mask_type == 2:
                mask = get_square_mask(xy, d1, d2)
                mat = apply_mask(mat, mask)

    return pos


if __name__ == "__main__":
    cc.compile()
