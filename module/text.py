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


def pval_to_stars(p, thresholds=((1e-4, '****'), (1e-3, '***'),
                                   (1e-2, '**'), (5e-2, '*')),
                  ns='n.s.'):
    """Convert a p-value to a significance string."""
    for thresh, star in thresholds:
        if p < thresh:
            return star
    return ns


def _span_data_top(ax, lo, hi, tol):
    """Highest y of plotted data whose x falls within [lo-tol, hi+tol]."""
    ys = []
    for line in ax.lines:
        xd = np.asarray(line.get_xdata(), dtype=float)
        yd = np.asarray(line.get_ydata(), dtype=float)
        if xd.size != yd.size or xd.size == 0:
            continue
        m = (xd >= lo - tol) & (xd <= hi + tol)
        if m.any():
            ys.append(yd[m])
    for coll in ax.collections:
        offs = np.asarray(coll.get_offsets(), dtype=float)
        if offs.ndim == 2 and offs.size:
            m = (offs[:, 0] >= lo - tol) & (offs[:, 0] <= hi + tol)
            if m.any():
                ys.append(offs[m, 1])
    if not ys:
        return None
    allys = np.concatenate(ys)
    allys = allys[np.isfinite(allys)]
    return np.nanmax(allys) if allys.size else None


def add_significance_bars(ax, pairs, pvalues=None, labels=None,
                          pad=None, tick_height=None, text_pt_offset=1.0,
                          color='black', linewidth=1.0, fontsize=10,
                          show_ns=True, ns_label='n.s.', allow_above_ylim=False,
                          star_thresholds=((1e-4, '****'), (1e-3, '***'),
                                           (1e-2, '**'), (5e-2, '*')),
                          expand_ylim=True):
    """Draw significance bracket(s) between pairs of x-positions.

    Each bracket sits above the tallest data point within its x-span.
    Overlapping brackets are stacked automatically (narrow spans first).

    Parameters
    ----------
    ax : matplotlib Axes
    pairs : list of (x1, x2)
        X-positions in data coordinates.
    pvalues : list of float, optional
        One p-value per pair, converted to stars via *star_thresholds*.
    labels : list of str, optional
        Explicit text per pair (overrides pvalues).
    pad : float, optional
        Vertical gap above data (data units). Default ~6% of y-range.
    tick_height : float, optional
        Length of bracket end-ticks. Default ~30% of pad.
    text_pt_offset : float
        Gap in points between bracket line and label text.
    color, linewidth, fontsize : styling
    show_ns : bool
        If False, skip non-significant pairs.
    ns_label : str
        Text used for non-significant results.
    star_thresholds : tuple of (threshold, label)
        P-value thresholds for star labels.
    expand_ylim : bool
        If True, expand ylim to fit brackets.
    allow_above_ylim : bool
        Set True when the brackets are *meant* to sit above the axes -- e.g. in a
        panel's top margin, level with the titles of its neighbours -- which keeps
        the data band full height.  Suppresses the warning that otherwise fires
        when ``expand_ylim`` is False and a bracket ends up outside the view.
        Only safe where nothing later switches clipping on for the bracket lines
        (in a composed figure, plot the panel after ``compose()``).

    Returns
    -------
    top_line : float
        Y-position of the highest bracket drawn.
    """
    y0, y1 = ax.get_ylim()
    yrange = (y1 - y0) or 1.0
    x0, x1 = ax.get_xlim()
    tol = 0.01 * (abs(x1 - x0) or 1.0)

    if pad is None:
        pad = 0.06 * yrange
    if tick_height is None:
        tick_height = 0.3 * pad

    label_list, keep = [], []
    for i, _ in enumerate(pairs):
        if labels is not None:
            txt = labels[i]
        elif pvalues is not None:
            txt = pval_to_stars(pvalues[i], star_thresholds, ns_label)
        else:
            txt = ''
        keep.append(not (txt == ns_label and not show_ns))
        label_list.append(txt)

    pairs_k = [(min(p), max(p)) for p, k in zip(pairs, keep) if k]
    labels_k = [t for t, k in zip(label_list, keep) if k]

    order = sorted(range(len(pairs_k)), key=lambda i: pairs_k[i][1] - pairs_k[i][0])
    line_y = [None] * len(pairs_k)
    placed = []

    for i in order:
        lo, hi = pairs_k[i]
        top = _span_data_top(ax, lo, hi, tol)
        if top is None:
            top = y0
        base = top + pad

        for (plo, phi, py) in placed:
            strict_overlap = (lo < phi) and (plo < hi)
            touch = (abs(lo - phi) <= tol) or (abs(hi - plo) <= tol)
            if strict_overlap:
                base = max(base, py + 2 * pad)
            elif touch and abs(base - py) < 0.6 * pad:
                base = max(base, py + pad)

        placed.append((lo, hi, base))
        line_y[i] = base

    top_line = y0
    for (lo, hi), y, txt in zip(pairs_k, line_y, labels_k):
        ax.plot([lo, lo, hi, hi],
                [y - tick_height, y, y, y - tick_height],
                color=color, lw=linewidth, clip_on=False, zorder=10)
        is_star = bool(txt) and set(txt) <= {'*'}
        # annotation_clip=False matters as much as clip_on: annotate() defaults to
        # dropping the label entirely when its xy lies outside the axes, so a
        # bracket parked above the view limits loses its stars silently -- the
        # bracket line still draws (clip_on=False), leaving a bare bracket.
        ax.annotate(txt, xy=((lo + hi) / 2.0, y),
                    xytext=(0, text_pt_offset), textcoords='offset points',
                    ha='center', va='center_baseline' if is_star else 'bottom',
                    color=color, fontsize=fontsize, clip_on=False,
                    annotation_clip=False, zorder=10)
        top_line = max(top_line, y)

    if expand_ylim and (top_line + 2 * pad) > y1:
        ax.set_ylim(y0, top_line + 2 * pad)
    elif not expand_ylim and not allow_above_ylim and top_line > y1:
        # The caller took responsibility for the y-range and left the top bracket
        # *line* above the visible axes, where a composed figure clips it away
        # (clip_all_axes clips lines, not text) leaving a stranded label -- or, with
        # the label clipped too, nothing at all.  Nothing downstream can see this,
        # so say it here.  A label sitting a line above the topmost bracket is not
        # flagged: spilling that into a panel's top margin is a legitimate choice.
        import warnings
        warnings.warn(
            f'add_significance_bars: highest bracket line is at y={top_line:.3g} but '
            f'ylim ends at {y1:.3g}, so the top bracket(s) fall outside the axes. '
            f'Raise the upper ylim to at least {top_line:.3g} (plus about one text '
            f'line for the label), reduce pad, or pass expand_ylim=True.',
            stacklevel=2)

    return top_line


def add_stat_annot(fig, ax, x_start_list, x_end_list,
                   y_start_list=None, y_end_list=None,
                   line_height=2, stat_list=['*'],
                   text_y_offset=0.2, text_x_offset=-0.01):
    """Legacy interface — use :func:`add_significance_bars` instead."""

    if type(x_start_list) is not list:
        x_start_list = [x_start_list]

    for x_start, x_end, y_start, y_end, stat in zip(x_start_list, x_end_list,
                                                    y_start_list, y_end_list, stat_list):

        if y_start is None:
            y_start = get_axes_object_max(ax, x_loc=x_start, object_type='line') + line_height
        if y_end is None:
            max_at_x_end = get_axes_object_max(ax, x_loc=x_end, object_type='line')
            y_end = max_at_x_end + line_height

        y_start_end_max = np.max([y_start, y_end])

        ax.plot([x_start, x_start, x_end, x_end],
                [y_start, y_start_end_max + line_height, y_start_end_max + line_height, y_end],
                linewidth=1, color='k')

        ax.text(x=(x_start + x_end) / 2 + text_x_offset,
                y=y_start_end_max + line_height + text_y_offset,
                s=stat, horizontalalignment='center')

    return fig, ax