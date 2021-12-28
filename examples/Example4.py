"""
Example 3: Finding peaks in a PNG image.
    The image has a lot of artifacts which makes this a challenging peak picking analysis.

Notes
-----
* We will use PIL to import the PNG and then convert it to a gray scale image. The gray scale image is then converted
into a numpy array for analysis.
* To view the image and the peaks, we will use plotly. A helper plotting function can be found in data_structures.py file.

"""
import time

from PIL import Image, ImageOps  # pip install Pillow
import numpy as np
import plotly.graph_objs as go  # pip install plotly
from numba import config
config.DISABLE_JIT = False  # must be above peakpicker import


import peakpicker
from plotting_utils import image_to_fig, transform_points

img = Image.open("./data/example2.png")
img_gray = ImageOps.grayscale(img)
matrix = np.array(img_gray)[500:800, 500:800]  # [550:620, 450:600]

# 3D surface plot of image
# fig = go.Figure(go.Surface(z=matrix[::3,::3]))
# fig.write_html("temp.html", auto_open=True)


# Get potential peaks
start = time.time()
data = peakpicker.get_peaks_by_topology(matrix)
print(time.time()-start)
peaks = peakpicker.process_data(data)
peaks = peakpicker.core_math.unravel_index(peaks, matrix.shape)


# plotting potential peaks
peaks_t = transform_points(peaks, matrix)
small_img = Image.fromarray(matrix)
fig = image_to_fig(small_img, scale=1, op_show=False)
fig.add_trace(go.Scatter(x=peaks_t[:, 0], y=peaks_t[:, 1], mode="markers",
                         marker=dict(color='red', opacity=1)))
fig.write_html("temp2.html", auto_open=True)


# Remove outlier peaks
# kwargs_outliers = {
#     "n_neighbors": 5,
#     "contamination": "auto"  # auto or [0,0.5)
# }
# potential_peaks, removed_peaks = peakpicker.remove_outliers(potential_peaks, **kwargs_outliers)
#
# # Plotting
# peaks_t = transform_points(potential_peaks, matrix)
# removed_peaks_t = transform_points(removed_peaks, matrix)
# fig = image_to_fig(img, scale=1, op_show=False)
# fig.add_trace(go.Scatter(x=peaks_t[:, 0], y=peaks_t[:, 1], mode="markers",
#                          marker=dict(color='Gray', opacity=0.5)))
# fig.add_trace(go.Scatter(x=removed_peaks_t[:, 0], y=removed_peaks_t[:, 1], mode="markers",
#                          marker=dict(color='Blue', opacity=0.3)))
# fig.write_html("temp3.html", auto_open=True)

# kwargs_cluster = {
#     "eps": 5,
#     "min_samples": 5
# }
#
# ## Plotting
# peaks = transform_points(peaks, matrix)
# potential_peaks = transform_points(meta_data["potential_peaks"], matrix)
# fig = image_to_fig(img, scale=1, op_show=False)
# fig.add_trace(go.Scatter(x=potential_peaks[:, 0], y=potential_peaks[:, 1], mode="markers",
#                          marker=dict(color='Gray', opacity=0.5)))
# fig.add_trace(go.Scatter(x=peaks[:, 0], y=peaks[:, 1], mode="markers",
#                          marker=dict(color='Blue', line=dict(color="White", width=0.5))))
# fig.write_html("temp2.html", auto_open=True)
#
# print(f"Potential Peaks: {meta_data['potential_peaks'].shape[0]}")
# print(f"Potential peaks removed: {meta_data['peaks_removed']}")
# print(f"Peaks found: {meta_data['n_peaks']}")
