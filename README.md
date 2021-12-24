# Peak Picker / Peak Detection / Peak Finder

Need to find peaks in your 2D data (or images)? 

Here we have a few methods to get the job done.

## Installation

`pip install peakpicker`


## 3 ways to run the code
Some codes can be really slow and have been written to take advantage of the speed gains from Numba. However, the 
Python (using numpy) is provided for easy modification of the code.

* Python (default)
```python
import peakpicker

peaks = peakpicker.max_intensity(...)
```

* Numba
Numba can significantly speed up calculation on large images, or large sets of images. However, the python code is 
must be complied each time it's run, so it adds a ~10 sec. overhead.
```python
from numba import config
config.DISABLE_JIT = True  # must be above peakpicker import

import peakpicker

peaks = peakpicker.max_intensity(...)
```
* Pre-compiled Numba
To eliminate the ~10 sec. overhead of compiling the code, a pre-complied code is also available. However, it may only 
  work for Windows users. The code is also setup, so you can compile the code yourself.
```python
import peakpicker.compiled as peakpicker

peaks = peakpicker.max_intensity(...)
```

## Examples
Check out the examples folder in this repository!

