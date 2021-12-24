"""
Example 1: Simple numpy array with a few peak added.

"""
import time

import numpy as np
from numba import config
config.DISABLE_JIT = True  # must be above peakpicker import

import peakpicker
import peakpicker.compiled as peakpicker2

# create test matrix
matrix = np.linspace(0, 10, 10000).reshape((100, 100))
# add 3 peaks
matrix[20:25, 20:25] = 20
matrix[80, 50] = 30
matrix[24:26, 70:73] = 25

# Run python version
peaks = peakpicker.max_intensity(mat=matrix, n=3, mask_type=1, d1=5, cut_off=11)
print(peaks)

# Run pre-compiled version
peaks = peakpicker2.max_intensity(mat=matrix, n=3, mask_type=1, d1=5, cut_off=11)
print(peaks)
