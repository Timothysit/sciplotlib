import sciplotlib as spl
import os

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

