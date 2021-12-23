import time

from numba import config
config.DISABLE_JIT = True

import numpy as np

import peakpicker
import peakpicker.compiled as peakpicker2

# create test matrix
matrix = np.linspace(0, 10, 10000).reshape((100, 100))
matrix[20:25, 20:25] = 20
matrix[80, 50] = 30
matrix[24:26, 70:73] = 25


start = time.time()
for i in range(1000):
    peaks = peakpicker.max_intensity(mat=matrix, n=3, mask_type=1, d1=5, cut_off=11)
print(time.time()-start)
print(peaks)


start = time.time()
for i in range(1000):
    peaks = peakpicker2.max_intensity(mat=matrix, n=3, mask_type=1, d1=5, cut_off=11)
print(time.time()-start)
print(peaks)
