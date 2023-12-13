def savefig(fig, fig_savepath, dpi=300, fig_exts=['.png', '.svg']):
    """
    Utility function for saving figure with better defaults.
    Parameters
    ----------
    fig
    fig_savepath
    dpi
    fig_exts

    Returns
    -------

    """

    for ext in fig_exts:
        fig.savefig(fig_savepath + ext, dpi=dpi, bbox_inches='tight', transparent=True)