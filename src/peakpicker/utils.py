import numpy as np
from numba import njit
from numba.pycc import CC

from peakpicker.core_math import dot

cc = CC('utils')
cc.verbose = True


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
    xy: np.ndarray[:, 2]
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
@cc.export('norm_row_axis', 'f8[:](u8[:,:])')
def norm_row_axis(xy):
    """ Frobenius norm across row"""
    distance = np.empty(xy.shape[0], dtype="float64")
    for i, row in enumerate(xy):
        distance[i] = np.sqrt(dot(row, row))

    return distance


@njit
@cc.export('get_neighbors_circle', 'i8[:,:](u8[:],f8)')
def get_neighbors_circle(xy: tuple, r: float = 10) -> np.ndarray:
    """ Get neighbors within a circle around a point

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
@cc.export('get_neighbors_rectangle', 'i8[:,:](u8[:],u8,u8,u8,u8)')
def get_neighbors_rectangle(xy: tuple, x_len: int = 10, y_len: int = 10, x_lim: int = None, y_lim: int = None) \
        -> np.ndarray:
    """ Get neighbors within rectangle around a point

    Generates indexes that are within a square of x_len x y_len around point xy

    Parameters
    ----------
    xy: Tuple
        center of square
    x_len: int64
        x length of rectangle
    y_len: int64
        y length of rectangle
    x_lim: int64
        x limit
    y_lim: int64
        y limit

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

    # remove out of bound indexes
    mask = out[:, 0] > 0
    out = out[mask]
    mask = out[:, 1] > 0
    out = out[mask]
    if x_lim is not None:
        mask = out[:, 0] < x_lim
        out = out[mask]
    if y_lim is not None:
        mask = out[:, 1] < y_lim
        out = out[mask]

    return out


@njit
@cc.export('get_neighbors_rectangle_flat', 'u8[:](u8,u8[:],u8,u8)')
def get_neighbors_rectangle_flat(center: int, shape: tuple, x_len: int = 2, y_len: int = 2) -> np.ndarray:
    """ Get neighbors within rectangle around a point

    Generates indexes that are within a square of x_len x y_len around point xy

    Parameters
    ----------
    center: int64
        center of square
    shape: Tuple (x,y)
        shape of larger matrix
    x_len: int64
        x length of rectangle
    y_len: int64
        y length of rectangle

    Returns
    -------
    index: np.ndarray[:]
        flattened index
    """
    space_x_low = int(center % shape[1] - 1)
    space_x_high = shape[1] - space_x_low - 1
    space_y_low = int((center - space_x_low) / shape[1])
    space_y_high = shape[0] - space_y_low - 1

    x_low = int(np.round_(x_len / 2))
    x_high = int(x_len - x_low - 1)
    y_low = int(np.round_(y_len / 2))
    y_high = int(y_len - y_low - 1)

    if x_low > space_x_low:
        x_low = space_x_low
    if x_high > space_x_high:
        x_high = space_x_high
    if y_low > space_y_low:
        y_low = space_y_low
    if y_high > space_y_high:
        y_high = space_y_high

    index = np.empty((x_high + x_low + 1) * (y_high + y_low + 1), dtype="uint64")
    first_row = np.linspace(
        center - x_low - shape[1] * y_low-1,
        center + x_high - shape[1] * y_low-1,
        x_high + x_low + 1)

    for i in range(y_high + y_low + 1):
        index[i*first_row.size: (1+i)*first_row.size] = first_row + shape[1] * i

    return index


if __name__ == "__main__":
    cc.compile()
