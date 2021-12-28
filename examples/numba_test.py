import numpy as np
from numba import njit
from numba.pycc import CC

cc = CC('topology')
cc.verbose = True


@njit
@cc.export('flat', 'u8[:](u8[:,:], u8[:])')
def flat(a, b):
    out = np.empty(b.size, dtype=a.dtype)
    for i, index in enumerate(b):
        out[i] = a.flat[index]
    return out


a = np.linspace(1, 100, 100, dtype="float64").reshape(10,10)
b = np.array([1,5,76,4], dtype="uint64")
c = flat(a, b)
print(c)
