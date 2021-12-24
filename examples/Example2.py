"""
Example 2: Finding peaks in a PNG image.
    The image has a lot of artifacts which makes this a challenging peak picking analysis.

Notes
-----
* We will use PIL to import the PNG and then convert it to a gray scale image. The gray scale image is then converted
into a numpy array for analysis.
* To view the image and the peaks, we will use plotly. A helper plotting function can be found in plotting_utils.py file.

"""


from PIL import Image, ImageOps  # pip install Pillow
import numpy as np
from numba import config
config.DISABLE_JIT = True  # must be above peakpicker import
import plotly.graph_objs as go  # pip install plotly

import peakpicker
from utils import image_to_fig, transform_points


img = Image.open("./data/example2.png")
img_gray = ImageOps.grayscale(img)
matrix = np.array(img_gray)

# 3D surface plot of image
# fig = go.Figure(go.Surface(z=matrix[::3,::3]))
# fig.write_html("temp.html", auto_open=True)

# Get peaks with a circle mask of radius of 30 pixels (change d1 to 60 for perfect peak picking on this image)
peaks = peakpicker.max_intensity(matrix, n=46, mask_type=1, d1=30, cut_off=20)
print(f"number of peaks found: {peaks.shape[0]}")
print(peaks)
peaks = transform_points(peaks, matrix)
fig = image_to_fig(img, scale=1, op_show=False)
fig.add_trace(go.Scatter(x=peaks[:, 0], y=peaks[:, 1], mode="markers",
                         marker=dict(color='Blue', line=dict(color="White", width=0.5))))
fig.write_html("temp2.html", auto_open=True)
