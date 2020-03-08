import numpy as np

def set_bounds(fig, ax, handle_zero_zero=True):
    """
    Change the limit of axis spines so that they match the first and last tick marks

    Parameters
    -----------
    fig:
    ax:
    handle_zero_zero : bool
        whether to join axis if both have minimum take mark value of zero
        useful for creating unity plots / plots where (0, 0) has a special meaning.
    :return:
    """

    xmin, xmax = ax.get_xlim()
    all_xtick_loc = ax.get_xticks()
    visible_xtick = [t for t in all_xtick_loc if (t >= xmin) & (t <= xmax)]
    min_visible_xtick_loc = min(visible_xtick)
    max_visible_xtick_loc = max(visible_xtick)
    ax.spines['bottom'].set_bounds(min_visible_xtick_loc, max_visible_xtick_loc)

    ymin, ymax = ax.get_ylim()
    all_ytick_loc = ax.get_yticks()
    visible_ytick = [t for t in all_ytick_loc if (t >= ymin) & (t <= ymax)]
    min_visible_ytick_loc = min(visible_ytick)
    max_visible_ytick_loc = max(visible_ytick)
    ax.spines['left'].set_bounds(min_visible_ytick_loc, max_visible_ytick_loc)

    if handle_zero_zero:
        if min_visible_xtick_loc == 0 and min_visible_ytick_loc == 0:
            ax.set_xlim([0, xmax])
            ax.set_ylim([0, ymax])

    return fig, ax


def apply_gradient(ax, extent, direction=0.3, cmap_range=(0, 1),
                   aspect='auto', **kwargs):
    """
    Draw a gradient image based on a colormap.

    Parameters
    ----------
    ax : Axes
        The axes to draw on.
    extent
        The extent of the image as (xmin, xmax, ymin, ymax).
        By default, this is in Axes coordinates but may be
        changed using the *transform* kwarg.
    direction : float
        The direction of the gradient. This is a number in
        range 0 (=vertical) to 1 (=horizontal).
    cmap_range : float, float
        The fraction (cmin, cmax) of the colormap that should be
        used for the gradient, where the complete colormap is (0, 1).
    **kwargs
        Other parameters are passed on to `.Axes.imshow()`.
        In particular useful is *cmap*.
    """
    if extent is None:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        extent = (xmin, xmax, ymin, ymax)

    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    X = np.array([[v @ [1, 0], v @ [1, 1]],
                  [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    X = a + (b - a) / X.max() * X
    im = ax.imshow(X, extent=extent, interpolation='bicubic',
                   vmin=0, vmax=1, aspect=aspect, **kwargs)
    return im
