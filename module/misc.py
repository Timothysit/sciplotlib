

# Functions for the economist theme

def add_economist_rectangle(fig, ax, xloc=0.1, yloc=1.04, width=0.05, height=0.04):
    """


    References
    ---------
    economist color scheme: http://pattern-library.economist.com/color.html
    """

    economist_red = '#e3120b'

    fig.patches.extend([plt.Rectangle((xloc, yloc), width=width, height=height,
                                      fill=True, color=economist_red, alpha=1, zorder=1000,
                                      transform=fig.transFigure, figure=fig)])

    return fig, ax


def add_datasource(fig, ax, s='Source: IMF', xloc=0.1, yloc=-0.1, fontsize=10, weight='light', alpha=0.8):
    """
    Add datasource text to plots imitating the style of The Economist.
    Parameters
    ----------
    fig
    ax
    s
    xloc
    yloc
    fontsize
    weight
    alpha

    Returns
    -------

    """

    fig.text(xloc, yloc, s=s, fontsize=fontsize, alpha=alpha,
             weight=weight)

    return fig, ax