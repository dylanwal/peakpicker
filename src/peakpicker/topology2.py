import numpy as np
from numba import njit
from numba.pycc import CC

from .utils import get_neighbors_rectangle_flat
from .core_math import in1d, flat

cc = CC('topology')
cc.verbose = True


def get_index_sort_by_max(mat: np.ndarray) -> np.ndarray:
    """

    Returns x,y indexes of matrix in order of maximum to minimum.

    Parameters
    ----------
    mat

    Returns
    -------

    """
    flat_indices = np.linspace(0, mat.size-1, mat.size, dtype="uint64")
    values = mat.flat[flat_indices]
    sorted_index = np.argsort(values)

    return flat_indices[np.flip(sorted_index)]


def get_peaks_by_topology(mat: np.ndarray, n_neighbors: int = 5, r_min: float = 10, delta_h: float = 5):
    """ Determine peaks by topology

    """
    flat_indices = get_index_sort_by_max(mat)
    data = np.ones((mat.size, 3), dtype="int64") * -1  # index, peak index, weight
    fill_level = 0
    data = topology_main_loop(flat_indices, data, fill_level, mat, n_neighbors, r_min, delta_h)

    return data


@njit
@cc.export('get_higher_neighbors', 'u8[:](u8, u8, u8[:,:], u8[:,:], u8)')
def get_higher_neighbors(pixel, n_neighbors, mat, data, fill_level):
    """

    Find if neighbors that are in data (higher value).

    Parameters
    ----------
    pixel
    n_neighbors
    mat
    data
    fill_level

    Returns
    -------

    """
    neighbor_indexes = get_neighbors_rectangle_flat(pixel, mat.shape, x_len=n_neighbors, y_len=n_neighbors)
    mask = in1d(neighbor_indexes, data[:fill_level, 0])
    return neighbor_indexes[mask]


@njit
@cc.export('get_peak_associated_neighbor', 'u8(u8[:], u8[:,:], u8[:,:], u8)')
def get_peak_associated_neighbor(neighbors, mat, data, fill_level):
    neighbor_values = flat(mat, neighbors)
    max_neighbor_index = np.argmax(neighbor_values)
    nearest_peak_index = np.where(data[:fill_level, 0] == neighbors[max_neighbor_index])[0]
    return data[nearest_peak_index, 0][0]


@njit
@cc.export('get_peaks_by_topology', 'u8[:,](u8[:,:], u8, f8, f8)')
def topology_main_loop(flat_indices, data, fill_level, mat, n_neighbors, r_min, delta_h):
    for i, pixel in enumerate(flat_indices):
        higher_neighbors = get_higher_neighbors(pixel, n_neighbors, mat, data, fill_level)
        if higher_neighbors.size > 0:
            # if there are neighbors we have seen already (they should be higher), get peak they are associated with
            peak_associated_neighbor = get_peak_associated_neighbor(higher_neighbors, mat, data, fill_level)
            data[i, 1] = peak_associated_neighbor

        data[i, 0] = pixel
        data[i, 2] = i
        fill_level += 1

    return data


def process_data(data):
    peaks = np.where(data[:20, 1] == -1)

    return data[peaks[0], 0]


if __name__ == "__main__":
    cc.compile()