import numpy as np
import PIL
import plotly.graph_objects as go


config = {
    'doubleClick': 'reset',
    "displaylogo": False,
    "showTips": True,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect',
                            'eraseshape'
                            ]
}

_counter = 0


def image_to_fig(
        img: PIL.Image,
        scale: float = 1,
        op_gridlines: bool = False,
        op_show: bool = True
) -> go.Figure:
    """ Image to Figure

    Create Plotly Figure of PIL Image.

    Parameters
    ----------
    img: PIL.Image
        Image
    scale: float
        How much to scale the overall size of the figure
    op_gridlines: bool
        Show grid lines (default: False)
    op_show: bool
        write to html and open it (default: True)

    Returns
    -------
    fig: Plotly Figure

    """
    img_width = img.width
    img_height = img.height

    fig = go.Figure()

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale],
            y=[0, img_height * scale],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=op_gridlines,
        range=[0, img_width * scale]
    )

    fig.update_yaxes(
        visible=op_gridlines,
        range=[0, img_height * scale],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale,
            y=img_height * scale,
            sizey=img_height * scale,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=img)
    )

    # Configure other layout
    fig.update_layout(
        width=img_width * scale,
        height=img_height * scale,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False
    )

    if op_show:
        global _counter
        fig.write_html(f"temp_{_counter}.html", auto_open=True, config=config)
        _counter += 1

    return fig


def transform_points(xy: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """ Transform points

    PIL puts (0,0) at the upper, left corner; so we need to transform out peak positions for plotting.

    Parameters
    ----------
    xy: np.ndarray[:,2]
        peak position
    mat: np.ndarray
        Matrix on which peak detection was preformed on

    Returns
    -------
    pos: np.ndarray
        new peak positions that can be plotted with PIL images
    """
    pos = np.copy(xy)
    pos[:, 0] = xy[:, 1]
    pos[:, 1] = mat.shape[0] - xy[:, 0]

    return pos
