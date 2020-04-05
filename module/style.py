import sciplotlib as spl
import os
import matplotlib as mpl
from collections import OrderedDict


def get_style(stylesheet_name='nature'):
    """
    Function to get stylesheet to be used to apply a style to matplotlib plots.

    Example:
    >> import matplotlib.pyplot as plt
    >> from sciplotlib import style as spstyle
    >> nature_style = spstyle.get_style('nature')
    >> with plt.style.context(nature_style):
    >>      plt.plot([1, 2, 3, 4, 5])

    Parameters
    ----------
    stylesheet_name : (str)
        name of stylesheet to get

    Returns
    -------
    stylesheet_path : (str)
        path of stylesheet
    """

    package_dir = spl.__path__[0]
    dirname = os.path.join(package_dir, 'stylesheets')
    stylesheet_path = os.path.join(dirname, stylesheet_name + '.mplstyle')

    return stylesheet_path


def use_sans_maths(return_type='permenant', fig=None, ax=None, verbose=False):
    """
    Uses LaTeX computer modern *SANS* mathematics font.
    Note that you can reset using: plt.rcdefaults()

    Parameters
    ----------
    return_type : (str)
    fig : (matplotlib fig object)
    ax : (matplotlib ax object)
    verbose : (bool)
    """

    if return_type is 'temp_dict' and (fig is None):
        # usage is with mpl.rc_context(use_sans_maths())
        # note that there is currenlty an issue with matplotli that prevents this from working without
        # calling plt.show() or fig.savefig()
        # see: https://github.com/matplotlib/matplotlib/issues/13431
        if verbose:
            print("""Note you must use plt.show() so that the plot will render properly. 
                    You can turn off this warning by setting verbose=False.""")
        rc_param_dict = {'text.usetex': True,
                         'text.latex.preamble': [r'\usepackage[cm]{sfmath}'],
                         'font.family': 'sans-serif',
                         'font.sans-serif': 'cm',
                         'text.latex.preamble': [
                             r'\usepackage{siunitx}',
                             r'\sisetup{detect-all}',
                             r'\usepackage{sansmath}',
                             r'\sansmath'
                         ]}

        return OrderedDict(rc_param_dict.items())

    elif return_type is 'permenant':
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = 'cm'
        mpl.rcParams['text.latex.preamble'] = [
            r'\usepackage{siunitx}',  # i need upright \micro symbols, but you need...
            r'\sisetup{detect-all}',  # ...this to force siunitx to actually use your fonts
            r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
            r'\sansmath'  # <- tricky! -- gotta actually tell tex to use!
        ]

    if (fig is not None) and (ax is not None):
        # TODO: see if there is a way to redraw (fig.canvas.draw()?) to circumvent the context manager
        fig = fig
        ax = ax

        return fig, ax

