import string
import numpy as np


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


def get_axes_object_max(ax, x_loc=1, object_type='line', verbose=False):
    """
    Obtains the maximum height of any matplotlib object given a specific x location.

    Parameters
    ----------
    ax
    x_loc
    object_type
    verbose

    Returns
    -------

    """
    if object_type == 'line':
        axes_objects = ax.lines
    else:
        axes_objects = ax.get_children()

    y_data_store = list()

    for ax_obj in axes_objects:

        if x_loc in ax_obj.get_xdata():

            if verbose:
                print(ax_obj.get_xdata())
                print(ax_obj.get_ydata())
            y_data_store.extend(ax_obj.get_ydata())

    return np.max(y_data_store)


def add_stat_annot(fig, ax, x_start_list, x_end_list,
                   y_start_list=None, y_end_list=None,
                   line_height=2, stat_list=['*'],
                   text_y_offset=0.2, text_x_offset=-0.01):
    """
    Add annotation indicating statistical significance (mainly to be used in boxplots, but can be generalised
    to any boxplot-like figurse, such as stripplots)

    Parameters
    -----------
    
    """

    if type(x_start_list) is not list:
        x_start_list = [x_start_list]

    for x_start, x_end, y_start, y_end, stat in zip(x_start_list, x_end_list,
                                                    y_start_list, y_end_list, stat_list):

        if y_start is None:
            y_start = get_axes_object_max(ax, x_loc=x_start, object_type='line') + line_height
        if y_end is None:
            max_at_x_end = get_axes_object_max(ax, x_loc=x_end, object_type='line')
            print(max_at_x_end)
            y_end = max_at_x_end + line_height

        y_start_end_max = np.max([y_start, y_end])

        sig_line = ax.plot([x_start, x_start, x_end, x_end],
                           [y_start, y_start_end_max + line_height, y_start_end_max + line_height, y_end],
                           linewidth=1, color='k')

        ax.text(x=(x_start + x_end) / 2 + text_x_offset,
                y=y_start_end_max + line_height + text_y_offset,
                s=stat, horizontalalignment='center')

    return fig, ax