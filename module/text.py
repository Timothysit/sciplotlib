import string


def add_lettering(axs, size=10, xloc=-0.1, yloc=1.1, weight='bold'):
    """
    Add lettering to each subplot
    
    Parameters
    ----------
    axs  : (Axes)
        matplotlib Axes object containing multiple subplots
    size : (float)
        font size of the lettering
    xloc : (float)
        location relative to the x axis of the subplot to add lettering
    yloc : (float)
        location relative to the y axis of the subplot to add lettering
    weight : (str)
        font weight of the lettering (eg. italic, bold)

    Returns
    -------

    axs : (Axes)
        updated matplolib Axes object with lettering

    References
    ------
    Based on code provided here:
    https://stackoverflow.com/questions/25543978/matplotlib-annotate-subplots-in-a-figure-with-a-b-c
    """

    for n, ax in enumerate(axs.flatten()):
        ax.text(xloc, yloc, string.ascii_uppercase[n],
                transform=ax.transAxes,
                size=size, weight=weight)

    return axs
