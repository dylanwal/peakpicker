import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

from .data_structures import DynamicArray


def get_potential_peaks(mat: np.ndarray, axis: int = 0, **kwargs) -> np.ndarray:
    """ Get Potential Peaks

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    Parameters
    ----------
    mat: np.ndarray
        2D data to find peaks in (Image)
    axis: int [0,1,2]
        0 = find peaks across rows
        1 = find peaks across columns
        2 = find peaks across both rows and columns
    kwargs:
        key word arguments for peak picker

    Returns
    -------
    peaks: np.ndarray[:,2]
        x,y position of peaks
    """
    peaks = DynamicArray((100, 2))
    if axis == 1:
        get_peaks_by_col(peaks, mat, **kwargs)
    elif axis == 2:
        get_peaks_by_row(peaks, mat, **kwargs)
        get_peaks_by_col(peaks, mat, **kwargs)
    else:
        get_peaks_by_row(peaks, mat, **kwargs)

    return peaks.data


def get_peaks_by_row(peaks: DynamicArray, mat: np.array, **kwargs):
    for i, row in enumerate(mat):
        peaks_, _ = find_peaks(row, **kwargs)
        if peaks_.size:
            peaks_to_add = np.ones((peaks_.size, 2)) * i
            peaks_to_add[:, 1] = peaks_
            peaks.add(peaks_to_add)


def get_peaks_by_col(peaks: DynamicArray, mat: np.array, **kwargs):
    peaks_col = DynamicArray((100, 2))
    mat = np.copy(mat)
    get_peaks_by_row(peaks_col, mat.T, **kwargs)
    peaks.add(peaks_col.data)


def remove_outliers(peaks: np.ndarray, **kwargs):
    """

    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
    https://scikit-learn.org/stable/modules/outlier_detection.html
    Parameters
    ----------
    peaks
    kwargs

    Returns
    -------

    """
    clf = LocalOutlierFactor(**kwargs)
    is_inlier = clf.fit_predict(peaks)  # 1 inliers, -1 is outliers
    mask = is_inlier == 1

    return peaks[mask], peaks[np.invert(mask)]


def get_cluster_center(potential_peaks, labels):
    num_labels = np.max(labels)
    peaks = np.empty((num_labels, 2))
    for label in range(num_labels):
        mask = labels == label
        cluster = potential_peaks[mask]
        peaks[label, :] = np.mean(cluster, axis=0)

    return peaks


def peak_by_clustering(potential_peaks: np.ndarray, kwargs_cluster: dict = None):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
    Parameters
    ----------
    potential_peaks
    kwargs_cluster

    Returns
    -------

    """
    clustering = DBSCAN(**kwargs_cluster).fit(potential_peaks)

    meta_data = {}
    meta_data["n_peaks"] = np.max(clustering.labels_)
    meta_data["peaks_removed"] = list(clustering.labels_).count(-1)
    meta_data["potential_peaks"] = potential_peaks
    meta_data["labels"] = clustering.labels_
    peaks = get_cluster_center(potential_peaks, clustering.labels_)

    return peaks, meta_data

