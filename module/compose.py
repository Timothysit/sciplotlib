"""Core figure composition functionality for sciplotlib.

Provides the shared rendering logic used by both the GUI/CLI in layout.py
and the programmatic Python API (FigureComposer).
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle
from pathlib import Path
import pickle
import io
import json
import yaml
from copy import copy


_UNREGISTERED = object()   # internal sentinel: register_stats was never called

NOT_APPLICABLE = object()
"""Sentinel for :meth:`FigureComposer.register_stats`.

Pass this instead of ``None`` to mark a panel as intentionally having no
statistical tests (e.g. schematics, count plots).  ``None`` is accepted as
an alias for backwards compatibility.

Example::

    composer.register_stats('b', splcompose.NOT_APPLICABLE)
"""

NO_FONT_NORMALIZE = 'spl-no-font-normalize'
"""gid marking a text artist that :meth:`FigureComposer.normalize_fonts` skips.

:meth:`~FigureComposer.normalize_fonts` rewrites every text size in the figure
to the composer's ``font_size`` so labels can never drift out of sync.  That is
wrong for text drawn *inside* a schematic -- a string in a cartoon monitor, a
symbol in a flow diagram -- which is sized to fit a hand-drawn box, not to match
the axis labels.  Forcing such text to ``font_size`` overflows the box it lives
in.  Tag it instead, via :func:`exempt_from_font_normalization`, and it keeps
whatever size it was created with.
"""


def exempt_from_font_normalization(artist):
    """Tag an artist so :meth:`FigureComposer.normalize_fonts` leaves its size alone.

    Use for schematic text whose size is dictated by surrounding drawn
    geometry rather than by the figure's label size.

    Accepts either a text artist or an Axes.  Tagging an Axes exempts that
    axes wholesale -- its ticks, axis labels, title and any child axes --
    which is what you want for a small inset that is part of a cartoon.

    Example::

        txt = ax.annotate('LRLR', xy=(0, 0), fontsize=7)
        splcompose.exempt_from_font_normalization(txt)

        inset = ax.inset_axes([0.5, 0.1, 0.08, 0.08])   # sigmoid in a schematic
        splcompose.exempt_from_font_normalization(inset)

    Returns the artist, so it can be used inline.
    """
    artist.set_gid(NO_FONT_NORMALIZE)
    return artist

PAPER_DIMENSIONS = {
    'a4': (21.0, 29.7),
    'a4_half_portrait': (10.5, 29.7),
    'a0_portrait': (84.1, 118.9),
    'a0_landscape': (118.9, 84.1),
    '16:9_monitor': (59.7, 33.6),
}


def cm_to_px(cm, dpi=96):
    return int(cm * dpi / 2.54)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def copy_axes_content(src_ax, dest_ax, scale_factor=1.0):
    """Copy data artists and properties from a source axes to a destination axes.

    Creates new artist instances to avoid reuse errors when transferring
    content between figures (e.g. from a pickled figure into a composed layout).
    """
    import matplotlib.text as mtext
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    for line in src_ax.lines:
        new_line = Line2D(
            xdata=line.get_xdata(), ydata=line.get_ydata(),
            color=line.get_color(), linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth() * scale_factor, marker=line.get_marker(),
            markersize=line.get_markersize() * scale_factor, label=line.get_label(),
        )
        dest_ax.add_line(new_line)

    for collection in src_ax.collections:
        if isinstance(collection, PathCollection):
            offsets = collection.get_offsets()
            dest_ax.scatter(
                offsets[:, 0], offsets[:, 1],
                s=collection.get_sizes() * (scale_factor ** 2),
                c=collection.get_facecolors(),
                marker=collection.get_paths()[0] if collection.get_paths() else 'o',
                alpha=collection.get_alpha(),
                linewidths=collection.get_linewidths() * scale_factor,
                edgecolors=collection.get_edgecolors(),
                label=collection.get_label(),
            )
            if collection.get_array() is not None:
                new_coll = dest_ax.collections[-1]
                new_coll.set_array(collection.get_array())
                new_coll.set_cmap(collection.get_cmap())
                new_coll.set_norm(collection.get_norm())

    for patch in src_ax.patches:
        if isinstance(patch, Rectangle):
            new_patch = Rectangle(
                xy=patch.get_xy(), width=patch.get_width(),
                height=patch.get_height(), angle=patch.get_angle(),
                facecolor=patch.get_facecolor(), edgecolor=patch.get_edgecolor(),
                linewidth=patch.get_linewidth() * scale_factor, linestyle=patch.get_linestyle(),
                alpha=patch.get_alpha(), label=patch.get_label(),
            )
            dest_ax.add_patch(new_patch)
        else:
            new_patch = copy(patch)
            if hasattr(new_patch, 'set_linewidth'):
                new_patch.set_linewidth(patch.get_linewidth() * scale_factor)
            dest_ax.add_patch(new_patch)

    for image in src_ax.images:
        new_img = dest_ax.imshow(
            image.get_array(), extent=image.get_extent(),
            cmap=image.get_cmap(), norm=image.norm,
            origin=getattr(image, '_origin', 'upper'),
            interpolation=image.get_interpolation(),
        )
        new_img.set_clim(image.get_clim())

    for text in src_ax.texts:
        if isinstance(text, mtext.Annotation):
            xytext = text.xytext
            if text.textcoords == 'offset points':
                xytext = (xytext[0] * scale_factor, xytext[1] * scale_factor)
            arrowprops = copy(text.arrowprops) if text.arrowprops else None
            if arrowprops:
                for k in ['width', 'headwidth', 'headlength', 'shrink']:
                    if k in arrowprops:
                        arrowprops[k] = arrowprops[k] * scale_factor
            dest_ax.annotate(
                text.get_text(), xy=text.xy, xytext=xytext,
                textcoords=text.textcoords, xycoords=text.xycoords,
                arrowprops=arrowprops, ha=text.get_ha(), va=text.get_va(),
                fontsize=text.get_fontsize() * scale_factor, color=text.get_color(),
                fontweight=text.get_fontweight(), fontstyle=text.get_fontstyle(),
                fontfamily=text.get_fontfamily(), zorder=text.get_zorder(),
            )
        else:
            dest_ax.text(
                *text.get_position(), text.get_text(),
                transform=dest_ax.transData,
                ha=text.get_ha(), va=text.get_va(),
                fontsize=text.get_fontsize() * scale_factor, color=text.get_color(),
                fontweight=text.get_fontweight(), fontstyle=text.get_fontstyle(),
                fontfamily=text.get_fontfamily(), zorder=text.get_zorder(),
            )

    for artist in src_ax.artists:
        if isinstance(artist, AnnotationBbox):
            offsetbox = artist.offsetbox
            if isinstance(offsetbox, OffsetImage):
                new_zoom = offsetbox.get_zoom() * scale_factor
                new_oi = OffsetImage(offsetbox.get_data(), zoom=new_zoom)
                xybox = artist.xybox
                if artist.boxcoords == 'offset points':
                    xybox = (xybox[0] * scale_factor, xybox[1] * scale_factor)
                new_ab = AnnotationBbox(
                    new_oi, artist.xy, xybox=xybox,
                    boxcoords=artist.boxcoords,
                    frameon=artist.patch.get_visible(),
                    zorder=artist.get_zorder(),
                )
                dest_ax.add_artist(new_ab)

    dest_ax.set_xlim(src_ax.get_xlim())
    dest_ax.set_ylim(src_ax.get_ylim())
    dest_ax.set_xscale(src_ax.get_xscale())
    dest_ax.set_yscale(src_ax.get_yscale())
    dest_ax.set_aspect(src_ax.get_aspect(),
                       adjustable=src_ax.get_adjustable(),
                       anchor=src_ax.get_anchor())
    dest_ax.set_title(src_ax.get_title())
    dest_ax.set_xlabel(src_ax.get_xlabel())
    dest_ax.set_ylabel(src_ax.get_ylabel())
    dest_ax.set_xticks(src_ax.get_xticks())
    dest_ax.set_xticklabels([l.get_text() for l in src_ax.get_xticklabels()])
    dest_ax.set_yticks(src_ax.get_yticks())
    dest_ax.set_yticklabels([l.get_text() for l in src_ax.get_yticklabels()])
    if src_ax.get_xgridlines():
        dest_ax.grid(src_ax.get_xgridlines()[0].get_visible())


def render_svg(filepath, scale=4, output_width=None):
    """Render an SVG file to a numpy RGBA array.

    Parameters
    ----------
    filepath : str or Path
        Path to the SVG file.
    scale : float
        Resolution multiplier. Ignored when *output_width* is set.
    output_width : int, optional
        Render the SVG to exactly this many pixels wide (height scales
        to preserve aspect ratio).  Much more memory-efficient than a
        large *scale* when the display size is known.

    Returns
    -------
    img : numpy array (H, W, 4) with values in [0, 1].
    """
    import cairosvg
    filepath = Path(filepath)
    kwargs = {'url': str(filepath)}
    if output_width is not None:
        kwargs['output_width'] = output_width
    else:
        kwargs['scale'] = scale
    png_bytes = cairosvg.svg2png(**kwargs)
    buf = io.BytesIO(png_bytes)
    return mpimg.imread(buf, format='png')


def tint_image(img, color):
    """Recolor an RGBA image array by replacing RGB values on opaque pixels.

    Parameters
    ----------
    img : numpy array
        An RGBA image array (H, W, 4) with values in [0, 1] or [0, 255].
    color : str or tuple
        Matplotlib color description.

    Returns
    -------
    out : numpy array
        The recolored image.
    """
    from matplotlib.colors import to_rgba
    r, g, b, _ = to_rgba(color)
    out = img.copy()
    if out.dtype.kind in 'iu':
        r_val, g_val, b_val = int(r * 255), int(g * 255), int(b * 255)
        mask = out[:, :, 3] > 12
    else:
        r_val, g_val, b_val = r, g, b
        mask = out[:, :, 3] > 0.05
    out[mask, 0] = r_val
    out[mask, 1] = g_val
    out[mask, 2] = b_val
    return out


def place_image(ax, img, x, y, zoom, zorder=5, frameon=False, **kwargs):
    """Place an image onto a matplotlib Axes using OffsetImage and AnnotationBbox.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw into.
    img : numpy array
        The image array to place.
    x, y : float
        Coordinates in axes data space to place the image center.
    zoom : float
        Zoom factor for the image.
    zorder : int, default 5
        Z-order for the artist.
    frameon : bool, default False
        Whether to draw a frame around the image.
    **kwargs
        Passed to matplotlib.offsetbox.AnnotationBbox.

    Returns
    -------
    ab : matplotlib.offsetbox.AnnotationBbox
        The created AnnotationBbox artist.
    """
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    im = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(im, (x, y), frameon=frameon, zorder=zorder, **kwargs)
    ax.add_artist(ab)
    return ab


def create_inset_grid(ax, nrows, ncols, sharex=False, sharey=False,
                      wspace=0.1, hspace=0.1, left=0.0, bottom=0.0,
                      width=1.0, height=1.0,
                      keep_aspect=False, source_figsize=None):
    """Create a grid of inset axes within a parent axes.

    Returns a numpy array of axes shaped (nrows, ncols), matching the
    output of ``plt.subplots()``.  Supports ``sharex`` and ``sharey``
    so that generic plotting functions that expect ``plt.subplots()``
    output work without modification.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Parent axes (typically with ``ax.axis('off')``).
    nrows, ncols : int
        Grid dimensions.
    sharex, sharey : bool
        Share x/y axes across the grid.
    wspace, hspace : float
        Horizontal/vertical gap between subplots, as a fraction of
        the subplot width/height.
    left, bottom, width, height : float
        Bounding box of the grid within the parent axes, in
        axes-fraction coordinates.  Defaults to the full axes.
    keep_aspect : bool
        If True, preserve the aspect ratio of the original figure
        and center the grid within the panel, leaving whitespace.
        Requires *source_figsize* or falls back to a square aspect.
    source_figsize : tuple of (width, height), optional
        The (width, height) in inches of the original standalone
        figure.  Used with *keep_aspect* to compute the target
        aspect ratio.  E.g. ``source_figsize=(2, 2)`` for a square.

    Returns
    -------
    axes : numpy.ndarray of Axes, shape (nrows, ncols)

    Notes
    -----
    Because the parent axes typically has ``axis('off')``,
    ``fit_axes_to_cells`` will skip it — meaning axis labels on the
    inset axes (e.g. ylabel text like "Win" / "Loss") can overflow
    the panel boundary.  Use the *left* parameter to reserve space::

        create_inset_grid(ax, 2, 2, left=0.12, width=0.88, ...)

    This is needed whenever inset subplots have y-axis labels or
    tick labels on the left edge.  The same applies to *bottom* for
    x-axis labels on the bottom edge.
    """
    import numpy as np

    if keep_aspect:
        fig = ax.figure
        pos = ax.get_position()
        fig_w, fig_h = fig.get_size_inches()
        panel_w = pos.width * fig_w
        panel_h = pos.height * fig_h

        src_w, src_h = source_figsize if source_figsize else (1.0, 1.0)
        src_aspect = src_w / src_h

        panel_aspect = panel_w / panel_h
        if src_aspect > panel_aspect:
            used_w = width
            used_h = width * (panel_w / panel_h) / src_aspect * (height / width)
            left_off = left
            bottom_off = bottom + (height - used_h) / 2
        else:
            used_h = height
            used_w = height * (panel_h / panel_w) * src_aspect * (width / height)
            left_off = left + (width - used_w) / 2
            bottom_off = bottom

        left, bottom, width, height = left_off, bottom_off, used_w, used_h

    cell_w = (width - wspace * (ncols - 1)) / ncols
    cell_h = (height - hspace * (nrows - 1)) / nrows

    axes = np.empty((nrows, ncols), dtype=object)
    ref_x = None
    ref_y = None

    for r in range(nrows):
        for c in range(ncols):
            x0 = left + c * (cell_w + wspace)
            y0 = bottom + height - (r + 1) * cell_h - r * hspace
            sa = ax.inset_axes([x0, y0, cell_w, cell_h])
            if sharex and ref_x is not None:
                sa.sharex(ref_x)
            if sharey and ref_y is not None:
                sa.sharey(ref_y)
            if ref_x is None:
                ref_x = sa
            if ref_y is None:
                ref_y = sa
            axes[r, c] = sa

    return axes


def figure_to_image(fig, width=None, dpi=None, **kwargs):
    """Convert a matplotlib Figure to a marimo-compatible HTML image.

    Prevents aspect-ratio squishing in the marimo UI by rendering the figure
    to PNG bytes with its native physical aspect ratio.
    """
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi or fig.dpi)
    plt.close(fig)
    buf.seek(0)
    if width is None:
        # Default to 96 CSS pixels per inch
        width = int(fig.get_size_inches()[0] * 96)
    try:
        import marimo as mo
        return mo.image(src=buf, width=width, **kwargs)
    except ImportError:
        return buf.read()


def load_panel_content(ax, filepath, fig=None):
    """Load an image or pickled matplotlib figure into an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw into.
    filepath : str or Path
        Path to an image (.png/.jpg/.jpeg), SVG placeholder, or pickled
        figure (.pkl).
    fig : matplotlib.figure.Figure, optional
        Parent figure, needed when a .pkl contains multiple sub-axes so
        that sub-gridspecs can be created.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in ('.png', '.jpg', '.jpeg'):
        img = mpimg.imread(str(filepath))
        ax.imshow(img)
        ax.set_box_aspect(img.shape[0] / img.shape[1])
        ax.axis('off')
    elif suffix == '.svg':
        img = render_svg(filepath)
        ax.imshow(img)
        ax.set_box_aspect(img.shape[0] / img.shape[1])
        ax.axis('off')
    elif suffix == '.pkl':
        try:
            with open(filepath, 'rb') as f:
                original_fig = pickle.load(f)
            buf = io.BytesIO()
            pickle.dump(original_fig, buf)
            buf.seek(0)
            fig_copy = pickle.load(buf)
            source_axes = fig_copy.get_axes()

            src_fig_w, src_fig_h = fig_copy.get_size_inches()
            parent_fig = fig or ax.figure
            dest_fig_w, dest_fig_h = parent_fig.get_size_inches()

            if len(source_axes) == 1:
                src_ax = source_axes[0]
                src_pos = src_ax.get_position()
                src_w = src_pos.width * src_fig_w
                src_h = src_pos.height * src_fig_h
                
                dest_pos = ax.get_position()
                dest_w = dest_pos.width * dest_fig_w
                scale_factor = dest_w / src_w if src_w > 0 else 1.0

                ax.set_box_aspect(src_h / src_w)
                copy_axes_content(src_ax, ax, scale_factor=scale_factor)
            elif fig is not None:
                ax.axis('off')
                inner_gs = ax.get_subplotspec().subgridspec(
                    1, len(source_axes), wspace=0.3)
                for i, src_ax in enumerate(source_axes):
                    sub_ax = fig.add_subplot(inner_gs[0, i])
                    
                    src_pos = src_ax.get_position()
                    src_w = src_pos.width * src_fig_w
                    src_h = src_pos.height * src_fig_h
                    
                    sub_pos = sub_ax.get_position()
                    sub_w = sub_pos.width * dest_fig_w
                    sub_scale = sub_w / src_w if src_w > 0 else 1.0

                    sub_ax.set_box_aspect(src_h / src_w)
                    copy_axes_content(src_ax, sub_ax, scale_factor=sub_scale)
            plt.close(fig_copy)
        except Exception as e:
            ax.text(0.5, 0.5, f'Failed to load:\n{filepath.name}',
                    ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, f'Unsupported: {suffix}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_facecolor('#f0f0f0')


def get_axes_scale_factor(ax, ref_width_inches=3.5):
    """Calculate a scale factor for scaling plot elements (fonts, line widths, offsets)
    based on the physical width of the axes relative to a reference width.
    """
    fig = ax.figure
    fig_w, _ = fig.get_size_inches()
    pos = ax.get_position()
    current_w = pos.width * fig_w
    return current_w / ref_width_inches


# ---------------------------------------------------------------------------
# Layout file parsing
# ---------------------------------------------------------------------------

def parse_layout_file(filepath):
    """Parse a YAML or JSON layout file into a normalized dict.

    Returns
    -------
    dict with keys:
        paper_w_cm, paper_h_cm : float
        grid_rows, grid_cols   : int
        style                  : dict (stylesheet, font, font_size, …)
        panels                 : list of dict (label, row, col, rowspan, colspan, file)
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    with open(filepath, 'r') as f:
        if suffix in ('.yaml', '.yml'):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    # --- Legacy JSON format (from the GUI's _save_json) ---
    if 'grid_settings' in data:
        grid_rows = data['grid_settings']['rows']
        grid_cols = data['grid_settings']['cols']
        paper_size = data['paper_settings']['size_name']
        custom_w = data['paper_settings']['custom_width_cm']
        custom_h = data['paper_settings']['custom_height_cm']
        style_info = {}

        if paper_size == 'custom':
            pw, ph = custom_w, custom_h
        else:
            pw, ph = PAPER_DIMENSIONS.get(paper_size, (21.0, 29.7))
        true_w = cm_to_px(pw, 96)
        true_h = cm_to_px(ph, 96)
        canvas_w, canvas_h = 700, 1200
        scale = 1.0
        if true_w > canvas_w or true_h > canvas_h:
            scale = min(canvas_w / true_w, canvas_h / true_h)
        disp_w = true_w * scale
        disp_h = true_h * scale

        panels = []
        for p in data['panels']:
            x0, y0, x1, y1 = p['bbox']
            col0 = int(round((x0 - 50) / disp_w * grid_cols))
            row0 = int(round((y0 - 50) / disp_h * grid_rows))
            col1 = int(round((x1 - 50) / disp_w * grid_cols))
            row1 = int(round((y1 - 50) / disp_h * grid_rows))
            panels.append({
                'label': p.get('label', ''),
                'row': row0, 'col': col0,
                'rowspan': row1 - row0, 'colspan': col1 - col0,
                'file': p.get('filepath'),
            })

    # --- YAML format (from the GUI's _save_yaml) ---
    else:
        grid_rows = data.get('grid', {}).get('rows', 20)
        grid_cols = data.get('grid', {}).get('cols', 10)
        paper_size = data.get('paper', {}).get('size', 'a4')
        custom_w = data.get('paper', {}).get('width_cm', 21.0)
        custom_h = data.get('paper', {}).get('height_cm', 29.7)
        style_info = data.get('style', {})
        panels = []
        for p in data.get('panels', []):
            panels.append({
                'label': p.get('label', ''),
                'row': p.get('row', 0),
                'col': p.get('col', 0),
                'rowspan': p.get('rowspan', 2),
                'colspan': p.get('colspan', 2),
                'file': p.get('file'),
            })

    if paper_size == 'custom':
        paper_w_cm, paper_h_cm = custom_w, custom_h
    else:
        paper_w_cm, paper_h_cm = PAPER_DIMENSIONS.get(paper_size, (21.0, 29.7))

    return {
        'paper_w_cm': paper_w_cm,
        'paper_h_cm': paper_h_cm,
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
        'style': style_info,
        'panels': panels,
    }


# ---------------------------------------------------------------------------
# Mid-level rendering
# ---------------------------------------------------------------------------

def clip_all_axes(fig):
    """Clip non-text artists in every axes of *fig* to the axes bounding box.

    Prevents lines, markers, patches, and images from overflowing into
    neighbouring panels.  The following are **never** clipped:

    - **Text** objects — panel labels, axis labels, and annotations are
      placed intentionally and may legitimately sit outside the axes box.
    - **Spine** objects — spines sit exactly at the axes boundary.
      Applying a clip-path to a spine cuts off the outer half of its
      stroke, making it render at half the specified ``stroke-width``.
      For example, a 0.7 pt spine appears as 0.35 pt when clipped.
      This is a subtle SVG/PDF rendering issue: the ``stroke-width``
      attribute is correct, but the ``clip-path`` silently halves the
      visible line.  Inset axes (child_axes) are not returned by
      ``fig.get_axes()`` and therefore escape this clipping, which
      causes inconsistent spine widths across panels.
    """
    from matplotlib.text import Text
    from matplotlib.spines import Spine
    for ax in fig.get_axes():
        bbox = ax.get_clip_box() or ax.bbox
        for artist in ax.get_children():
            if isinstance(artist, (Text, Spine)):
                continue
            artist.set_clip_on(True)
            artist.set_clip_box(bbox)


def render_panels_to_figure(panels, grid_rows, grid_cols, fig,
                            label_font_size=14, label_weight='bold',
                            label_x=0.0, label_y=0.01, label_ha='left',
                            wspace=None, hspace=None, margins=None,
                            axes_pad=None):
    """Render a list of panel dicts onto a matplotlib figure.

    Parameters
    ----------
    panels : list of dict
        Each dict must have keys: label, row, col, rowspan, colspan.
        Optional key: file (path to image or .pkl).
    grid_rows, grid_cols : int
        Grid dimensions for the GridSpec.
    fig : matplotlib.figure.Figure
        Target figure.
    label_font_size : float
        Font size for panel labels.
    label_weight : str
        Font weight for panel labels.
    label_x, label_y : float
        Offset of the panel label from the gridspec cell's top-left corner,
        in figure-fraction coordinates.
    wspace, hspace : float or None
        Spacing passed to GridSpec.
    margins : dict, optional
        GridSpec margins with keys 'left', 'right', 'top', 'bottom'
        (figure-fraction). Use to reserve space for axis labels on
        edge panels.
    axes_pad : dict, optional
        Inset each axes from its gridspec cell to leave room for axis
        decorations (tick labels, axis labels). Keys 'left', 'right',
        'top', 'bottom' in figure-fraction units. The panel label is
        placed at the cell boundary, and the axes is shrunk inward.

    Returns
    -------
    axes : dict
        Mapping of panel label → matplotlib Axes.
    """
    gs_kwargs = {}
    if wspace is not None:
        gs_kwargs['wspace'] = wspace
    if hspace is not None:
        gs_kwargs['hspace'] = hspace
    if margins is not None:
        for key in ('left', 'right', 'top', 'bottom'):
            if key in margins:
                gs_kwargs[key] = margins[key]
    gs = GridSpec(grid_rows, grid_cols, figure=fig, **gs_kwargs)

    axes = {}
    for p in panels:
        r0, c0 = p['row'], p['col']
        r1, c1 = r0 + p['rowspan'], c0 + p['colspan']
        r0 = max(0, min(r0, grid_rows - 1))
        c0 = max(0, min(c0, grid_cols - 1))
        r1 = max(r0 + 1, min(r1, grid_rows))
        c1 = max(c0 + 1, min(c1, grid_cols))

        ax = fig.add_subplot(gs[r0:r1, c0:c1])

        pad = p.get('axes_pad', axes_pad)
        if pad is not None:
            cell = ax.get_position()
            pl = pad.get('left', 0)
            pb = pad.get('bottom', 0)
            pr = pad.get('right', 0)
            pt = pad.get('top', 0)
            ax.set_position([
                cell.x0 + pl, cell.y0 + pb,
                cell.width - pl - pr, cell.height - pb - pt,
            ])

        axes[p.get('label', '')] = ax
        # Stamp the panel label so tools like the drag editor can name an
        # axes (and its child colorbars/insets) without any manual tagging.
        ax._sciplotlib_panel = p.get('label', '') or None

        label = p.get('label', '')
        if label:
            pos = gs[r0:r1, c0:c1].get_position(fig)
            p_lx = p.get('label_x')
            p_ly = p.get('label_y')
            lx = label_x if p_lx is None else p_lx
            ly = label_y if p_ly is None else p_ly
            fig.text(pos.x0 + lx, pos.y1 + ly, label,
                     fontsize=label_font_size, fontweight=label_weight,
                     va='bottom', ha=label_ha)

        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

        fp = p.get('file')
        if fp:
            load_panel_content(ax, fp, fig)

    return axes


# ---------------------------------------------------------------------------
# Stats report PDF helpers
# ---------------------------------------------------------------------------

def _write_stats_pdf(md_text, pdf_path):
    """Write *md_text* as a PDF, trying the best available backend."""
    # 1. weasyprint (markdown → HTML → PDF, best quality)
    try:
        import markdown as _md
        import weasyprint as _wp
        _html_body = _md.markdown(md_text, extensions=['tables'])
        _css = (
            'body{font-family:Arial,sans-serif;margin:2cm;font-size:10pt}'
            'h1{font-size:16pt;border-bottom:1px solid #555}'
            'h2{font-size:12pt;border-bottom:1px solid #aaa;margin-top:1.5em}'
            'li{margin:3px 0}em{color:#555}'
        )
        _full = f'<html><head><style>{_css}</style></head><body>{_html_body}</body></html>'
        _wp.HTML(string=_full).write_pdf(str(pdf_path))
        return
    except ImportError:
        pass

    # 2. pandoc (subprocess, second-best quality)
    try:
        import subprocess as _sp
        _r = _sp.run(
            ['pandoc', '--from=markdown', '--to=pdf', '-o', str(pdf_path)],
            input=md_text.encode(),
            capture_output=True,
        )
        if _r.returncode == 0:
            return
    except FileNotFoundError:
        pass

    # 3. matplotlib fallback — always available
    _render_stats_pdf_matplotlib(md_text, pdf_path)


def _render_stats_pdf_matplotlib(md_text, pdf_path):
    """Render a Markdown stats report to PDF using matplotlib's PDF backend."""
    import re
    from matplotlib.backends.backend_pdf import PdfPages

    PAGE_W, PAGE_H = 8.27, 11.69   # A4 inches
    LX = 0.07                        # left margin in axes fraction
    TOP_Y = 0.95
    LINE_H = 0.026                   # normal line height (axes fraction)
    H2_EXTRA = 0.008                 # extra gap before a section heading
    BULLET_INDENT = 0.025

    _STRIP_INLINE = re.compile(r'\*\*(.*?)\*\*|\*(.*?)\*|`(.*?)`')

    def _clean(text):
        return _STRIP_INLINE.sub(lambda m: m.group(1) or m.group(2) or m.group(3), text)

    lines = md_text.split('\n')
    # Simple word-wrap at ~95 chars
    wrapped = []
    for raw in lines:
        if raw.startswith('#') or raw.startswith('-') or raw.startswith('*') or not raw.strip():
            wrapped.append(raw)
        elif len(raw) > 95:
            words = raw.split()
            buf = ''
            for w in words:
                if len(buf) + len(w) + 1 <= 95:
                    buf = (buf + ' ' + w).lstrip()
                else:
                    wrapped.append(buf)
                    buf = w
            if buf:
                wrapped.append(buf)
        else:
            wrapped.append(raw)

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(PAGE_W, PAGE_H))
        ax.axis('off')
        y = TOP_Y

        def _flush():
            nonlocal fig, ax, y
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(PAGE_W, PAGE_H))
            ax.axis('off')
            y = TOP_Y

        for raw in wrapped:
            if y < 0.05:
                _flush()

            if raw.startswith('# '):
                text = _clean(raw[2:])
                ax.text(LX, y, text, transform=ax.transAxes,
                        fontsize=15, fontweight='bold', va='top', color='#111111')
                y -= LINE_H * 1.8

            elif raw.startswith('## '):
                y -= H2_EXTRA
                text = _clean(raw[3:])
                ax.text(LX, y, text, transform=ax.transAxes,
                        fontsize=11, fontweight='bold', va='top', color='#222222')
                y -= LINE_H * 1.5

            elif raw.startswith('- ') or raw.startswith('* '):
                text = _clean(raw[2:])
                ax.text(LX + BULLET_INDENT, y, f'•  {text}',
                        transform=ax.transAxes,
                        fontsize=9, va='top', color='#333333')
                y -= LINE_H

            elif not raw.strip():
                y -= LINE_H * 0.45

            else:
                text = _clean(raw)
                italic = raw.startswith('*') and raw.rstrip().endswith('*')
                ax.text(LX, y, text.strip('*').strip('_'),
                        transform=ax.transAxes,
                        fontsize=9, va='top', color='#444444',
                        style='italic' if italic else 'normal')
                y -= LINE_H

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


# ---------------------------------------------------------------------------
# High-level Python API
# ---------------------------------------------------------------------------

class FigureComposer:
    """Compose multi-panel scientific figures using a grid layout.

    Example
    -------
    >>> composer = FigureComposer(width_cm=18, height_cm=21, grid_rows=24, grid_cols=20)
    >>> composer.add_panel('a', row=0, col=0, rowspan=3, colspan=10)
    >>> composer.add_panel('b', row=0, col=10, rowspan=3, colspan=10)
    >>> fig, axes = composer.compose()
    >>> axes['a'].plot([1, 2, 3], [1, 4, 9])
    >>> composer.save('my_figure')
    """

    def __init__(self, width_cm=18, height_cm=24, grid_rows=24, grid_cols=20,
                 label_font_size=14, label_weight='bold',
                 label_x=0.0, label_y=0.01, label_ha='left', dpi=300,
                 stylesheet=None, rc_params=None,
                 font_size=None, axis_label_font_size=None, title_font_size=None,
                 margins=None, axes_pad=None,
                 wspace=0.4, hspace=0.5,
                 spine_linewidth=None, tick_linewidth=None, tick_length=None,
                 tick_pad=None, axis_label_pad=None, line_linewidth=None):
        self.width_cm = width_cm
        self.height_cm = height_cm
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.label_font_size = label_font_size
        self.label_weight = label_weight
        self.label_x = label_x
        self.label_y = label_y
        self.label_ha = label_ha
        self.dpi = dpi
        self.stylesheet = stylesheet
        self.rc_params = rc_params or {}
        self.font_size = font_size
        self.axis_label_font_size = axis_label_font_size
        self.title_font_size = title_font_size
        self.margins = margins
        self.axes_pad = axes_pad
        self.wspace = wspace
        self.hspace = hspace
        self.spine_linewidth = spine_linewidth
        self.tick_linewidth = tick_linewidth
        self.tick_length = tick_length
        self.tick_pad = tick_pad
        self.axis_label_pad = axis_label_pad
        self.line_linewidth = line_linewidth
        self.panels = []
        self._fig = None
        self._axes = None
        self._stats = {}

    def apply_style(self):
        """Apply the composer's stylesheet and rc_params globally.

        Call once at the start of a notebook so all subsequent plotting
        uses the same fonts, tick sizes, etc.::

            composer = FigureComposer(..., stylesheet='mp-paper',
                                     rc_params={'axes.labelsize': 9})
            composer.apply_style()
        """
        if self.stylesheet:
            import sciplotlib.style as splstyle
            plt.style.use(splstyle.get_style(self.stylesheet))
        
        # Apply custom font sizes to rcParams
        if self.font_size is not None:
            plt.rcParams['font.size'] = self.font_size
            plt.rcParams['xtick.labelsize'] = self.font_size
            plt.rcParams['ytick.labelsize'] = self.font_size
            plt.rcParams['legend.fontsize'] = self.font_size

        # Apply axis label font size (default to general font_size)
        ax_label_sz = self.axis_label_font_size
        if ax_label_sz is None:
            ax_label_sz = self.font_size
        if ax_label_sz is not None:
            plt.rcParams['axes.labelsize'] = ax_label_sz

        # Apply title font size (default to general font_size)
        title_sz = self.title_font_size
        if title_sz is None:
            title_sz = self.font_size
        if title_sz is not None:
            plt.rcParams['axes.titlesize'] = title_sz

        if self.rc_params:
            plt.rcParams.update(self.rc_params)

    def add_panel(self, label, row, col, rowspan, colspan, file=None,
                  no_axis=False, axes_pad=None, plot_func=None,
                  label_x=None, label_y=None, tick_pad=None):
        """Add a panel to the layout.

        Parameters
        ----------
        axes_pad : dict or None
            Per-panel override for axes inset padding. Set to ``{}``
            to disable the global axes_pad for this panel. ``None``
            (default) inherits the composer's global axes_pad.
        label_x, label_y : float or None
            Per-panel override for the panel label's offset from the cell
            corner, in figure-fraction units. ``None`` (default) inherits
            the composer's global ``label_x`` / ``label_y``. Use when one
            panel's label collides with its own axis decorations.
        tick_pad : float or None
            Per-panel override for the tick-label pad, in points. ``None``
            (default) inherits the composer's global ``tick_pad``. Applies
            to the panel's own axes only, not to child axes such as
            colorbars.
        plot_func : callable, optional
            A function with signature ``plot_func(ax)`` that draws content
            onto the panel axes.  Called during :meth:`compose` after the
            axes is created.  Takes priority over *file* if both are given.

            Example::

                def draw_panel_a(ax):
                    ax.scatter(x, y, color='steelblue')
                    ax.set_xlabel('Time (s)')

                composer.add_panel('A', row=0, col=0, rowspan=4, colspan=5,
                                   plot_func=draw_panel_a)

        Returns self for method chaining.
        """
        self.panels.append({
            'label': label,
            'row': row,
            'col': col,
            'rowspan': rowspan,
            'colspan': colspan,
            'file': str(file) if file else None,
            'no_axis': no_axis,
            'axes_pad': axes_pad,
            'plot_func': plot_func,
            'label_x': label_x,
            'label_y': label_y,
            'tick_pad': tick_pad,
        })
        return self

    def register_stats(self, label, stats):
        """Register statistical results for a panel.

        Parameters
        ----------
        label : str
            Panel label (e.g. ``'d'``).
        stats : list of dict or None
            A list of stat entries, each a dict with any subset of:
            ``description``, ``test``, ``statistic``, ``p_value``,
            ``n``, ``effect_size``, ``ci``, ``note``.
            Pass ``None`` to explicitly mark the panel as having no
            applicable statistical tests (e.g. schematics).

        Examples
        --------
        >>> composer.register_stats('d', [
        ...     {'description': 'GLM-HMM vs LR log-likelihood',
        ...      'test': 'Wilcoxon signed-rank',
        ...      'statistic': 28.0, 'p_value': 0.031, 'n': 3},
        ... ])
        >>> composer.register_stats('b', None)   # schematic — no tests
        """
        self._stats[label] = stats
        return self

    def to_stats_markdown(self, title=''):
        """Return a Markdown string summarising stats for all panels.

        Panels are listed in the order they were added via ``add_panel``.
        Three states are distinguished:

        * **registered list** -- one subsection per stat entry.
        * **``NOT_APPLICABLE`` / ``None``** -- intentionally no tests (schematic etc.).
        * **not registered** -- panel not yet assessed.
        """
        from datetime import date

        def _fmt_stat(x):
            if x is None:
                return None
            try:
                return f'{float(x):.3g}'
            except (TypeError, ValueError):
                return str(x)

        def _fmt_pval(x):
            if x is None:
                return None
            try:
                p = float(x)
                if p < 0.001:
                    return f'{p:.2e}'
                return f'{p:.3f}'
            except (TypeError, ValueError):
                return str(x)

        lines = []
        header = f'Statistics — {title}' if title else 'Statistics Report'
        lines.append(f'# {header}')
        lines.append('')
        lines.append(f'Generated: {date.today().isoformat()}')
        lines.append('')

        for p in self.panels:
            lbl = p.get('label', '')
            if not lbl:
                continue
            lines.append(f'## Panel {lbl}')
            lines.append('')

            val = self._stats.get(lbl, _UNREGISTERED)
            if val is _UNREGISTERED:
                lines.append('*(stats not yet registered)*')
            elif val is NOT_APPLICABLE or val is None:
                lines.append('*Not applicable -- no statistical tests for this panel type.*')
            else:
                entries = val
                if not entries:
                    lines.append('*(no entries)*')
                for entry in entries:
                    desc = entry.get('description', '')
                    test = entry.get('test', '')
                    stat = _fmt_stat(entry.get('statistic'))
                    pval = _fmt_pval(entry.get('p_value'))
                    n    = entry.get('n')
                    eff  = _fmt_stat(entry.get('effect_size'))
                    ci   = entry.get('ci')
                    note = entry.get('note', '')

                    if desc:
                        lines.append(f'**{desc}**')
                    if test:
                        lines.append(f'- Test: {test}')

                    detail_parts = []
                    if stat is not None:
                        detail_parts.append(f'statistic = {stat}')
                    if pval is not None:
                        detail_parts.append(f'p = {pval}')
                    if n is not None:
                        detail_parts.append(f'n = {n}')
                    if eff is not None:
                        detail_parts.append(f'effect size = {eff}')
                    if ci is not None:
                        detail_parts.append(f'95 % CI = {ci}')
                    if detail_parts:
                        lines.append(f'- {", ".join(detail_parts)}')
                    if note:
                        lines.append(f'- Note: {note}')
                    lines.append('')

            lines.append('')

        return '\n'.join(lines)

    def save_stats_report(self, path, title=''):
        """Write a stats report as both ``.md`` and ``.pdf``.

        The Markdown file is always written.  The PDF is generated by the
        best available backend in order: ``weasyprint``, ``pandoc``
        (subprocess), then a pure-matplotlib fallback that requires no
        extra dependencies.

        Parameters
        ----------
        path : str or Path
            Output path, with or without extension.  Both
            ``path.md`` and ``path.pdf`` are created.
        title : str
            Optional figure title shown at the top of the report.

        Returns
        -------
        md_path, pdf_path : Path
        """
        p = Path(path).with_suffix('')
        p.parent.mkdir(parents=True, exist_ok=True)

        md_text = self.to_stats_markdown(title=title)

        md_path = p.with_suffix('.md')
        md_path.write_text(md_text, encoding='utf-8')
        print(f'Saved: {md_path}')

        pdf_path = p.with_suffix('.pdf')
        _write_stats_pdf(md_text, pdf_path)
        print(f'Saved: {pdf_path}')

        return md_path, pdf_path

    def panel_figsize(self, label, wspace=None, hspace=None):
        """Return (width_inches, height_inches) for a panel, matching
        the exact physical size it occupies in the composed figure.

        Uses matplotlib's own GridSpec positioning internally so the
        result is pixel-accurate.  Defaults to the composer's wspace/hspace.
        """
        for p in self.panels:
            if p['label'] == label:
                fig_w = self.width_cm / 2.54
                fig_h = self.height_cm / 2.54
                gs_kwargs = dict(
                    wspace=wspace if wspace is not None else self.wspace,
                    hspace=hspace if hspace is not None else self.hspace,
                )
                if self.margins is not None:
                    for key in ('left', 'right', 'top', 'bottom'):
                        if key in self.margins:
                            gs_kwargs[key] = self.margins[key]
                gs = GridSpec(self.grid_rows, self.grid_cols, **gs_kwargs)
                r0, c0 = p['row'], p['col']
                ss = gs[r0:r0 + p['rowspan'], c0:c0 + p['colspan']]
                tmp_fig = plt.figure(figsize=(fig_w, fig_h))
                pos = ss.get_position(tmp_fig)
                plt.close(tmp_fig)
                w_in = (pos.x1 - pos.x0) * fig_w
                h_in = (pos.y1 - pos.y0) * fig_h
                if self.axes_pad is not None:
                    w_in -= (self.axes_pad.get('left', 0) + self.axes_pad.get('right', 0)) * fig_w
                    h_in -= (self.axes_pad.get('bottom', 0) + self.axes_pad.get('top', 0)) * fig_h
                return (w_in, h_in)
        raise KeyError(f"No panel with label '{label}'")

    def preview(self, label, wspace=None, hspace=None, pad_inches=(2.0, 1.0)):
        """Create a preview figure+axes that matches the physical size, DPI,
        and style of a panel in the composed figure.

        Includes optional padding (width, height) in inches around the axes to prevent
        clipping of elements (like annotations or OffsetImages) that bleed outside
        the formal gridspec boundaries.  Defaults to the composer's wspace/hspace.
        """
        size = self.panel_figsize(label, wspace=wspace, hspace=hspace)
        if pad_inches:
            pad_w, pad_h = pad_inches
            fig = plt.figure(figsize=(size[0] + pad_w, size[1] + pad_h), dpi=self.dpi)
            ax = fig.add_axes([
                (pad_w / 2) / (size[0] + pad_w),
                (pad_h / 2) / (size[1] + pad_h),
                size[0] / (size[0] + pad_w),
                size[1] / (size[1] + pad_h)
            ])
        else:
            fig, ax = plt.subplots(figsize=size, dpi=self.dpi)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return fig, ax

    def preview_image(self, label, wspace=None, hspace=None, plot_func=None,
                      normalize=True, show_label=True,
                      width=None, pad_inches=(2.0, 1.0), **kwargs):
        """Create a preview of a panel and return it as a marimo-compatible HTML image.

        This avoids aspect-ratio squishing in the marimo UI by rendering the panel
        to PNG bytes with its native physical aspect ratio.

        Parameters
        ----------
        label : str
            The panel label.
        wspace, hspace : float or None
            Grid spacing parameters.  Defaults to the composer's wspace/hspace.
        plot_func : callable, optional
            A function that takes an Axes object and plots the panel content.
            If provided, it will be executed on the preview axes before rendering.
            When *normalize* is True this falls back to the ``plot_func``
            registered via :meth:`add_panel` if not supplied here.
        normalize : bool, optional
            If True, compose the full figure (drawing only this panel), run the
            same normalisation passes as :meth:`to_image` and :meth:`save`
            (``normalize_fonts``, ``fit_axes_to_cells``, ``normalize_spines``),
            then crop to only the target panel's content.  The result matches
            the final PDF exactly.  Defaults to True.
        show_label : bool, optional
            When *normalize* is True, whether to show the panel letter label
            (e.g. "A", "B") in the preview.  Defaults to True.  Has no effect
            when *normalize* is False (standalone preview figures have no label).
        width : int or str, optional
            The display width of the image. Defaults to the calculated physical size
            in CSS pixels (96 pixels per inch).
        pad_inches : tuple of float, optional
            Width and height padding in inches to add around the axes to prevent
            clipping of overflowing content (ignored when *normalize* is True).
            Defaults to (2.0, 1.0).
        **kwargs
            Passed to marimo.image.
        """
        if normalize:
            return self._normalized_preview_image(
                label, plot_func=plot_func, show_label=show_label, width=width, **kwargs)
        fig, ax = self.preview(label, wspace=wspace, hspace=hspace, pad_inches=pad_inches)
        if plot_func is not None:
            plot_func(ax)
        if width is None:
            size = self.panel_figsize(label, wspace=wspace, hspace=hspace)
            width = int(size[0] * 96)
        return figure_to_image(fig, width=width, dpi=self.dpi, **kwargs)

    def _normalized_preview_image(self, label, plot_func=None, show_label=True,
                                   width=None, **kwargs):
        """Compose the full figure, normalize, and crop to the target panel's content.

        This is the engine behind ``preview_image(..., normalize=True)``.
        Only the target panel is drawn; all other panels are left empty so
        that normalization is computed in the same GridSpec context as the
        final figure, giving an exact match with the PDF output.
        """
        p_def = next((p for p in self.panels if p['label'] == label), None)
        if p_def is None:
            raise ValueError(f"Panel '{label}' not found.")

        _plot_func = plot_func if plot_func is not None else p_def.get('plot_func')

        # Snapshot all plot_funcs, then suppress every panel except the target
        # so compose() only draws this one panel in its correct GridSpec slot.
        snapshot = {p['label']: p.get('plot_func') for p in self.panels}
        for p in self.panels:
            if p['label'] != label:
                p.pop('plot_func', None)
            elif _plot_func is not None:
                p['plot_func'] = _plot_func

        try:
            fig, axes = self.compose()
            self.normalize_fonts()
            self.fit_axes_to_cells()
            self.normalize_spines()
            self.normalize_linewidths()

            # Hide every non-target axes so bbox_inches='tight' crops to just
            # this panel's content (colorbars, titles, tick labels included).
            for p in self.panels:
                lbl = p.get('label', '')
                if lbl and lbl != label and lbl in axes:
                    axes[lbl].set_visible(False)

            # Panel letters are figure-level text objects.  Hide the ones that
            # belong to other panels; optionally hide the target panel's own letter.
            all_labels = {p.get('label', '') for p in self.panels if p.get('label')}
            for text_obj in fig.texts:
                txt = text_obj.get_text()
                if txt in all_labels:
                    if txt != label:
                        text_obj.set_visible(False)
                    elif not show_label:
                        text_obj.set_visible(False)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
            buf.seek(0)
            data = buf.read()
        finally:
            plt.close(self._fig) if self._fig is not None else None
            self._fig = None
            for p in self.panels:
                saved = snapshot.get(p['label'])
                if saved is not None:
                    p['plot_func'] = saved
                else:
                    p.pop('plot_func', None)

        if width is None:
            size = self.panel_figsize(label)
            width = int(size[0] * 96)

        import marimo as mo
        return mo.image(data, width=width, **kwargs)

    def compose(self, wspace=None, hspace=None, clip_panels=True):
        """Create the composed figure with all panels on a grid.

        Parameters
        ----------
        wspace, hspace : float or None
            Spacing passed to GridSpec.  Defaults to the composer's wspace/hspace.
        clip_panels : bool
            If True (default), clip all panel contents to the axes
            bounding box so plots cannot overflow into neighbouring panels.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : dict mapping panel label → Axes
        """
        wspace = wspace if wspace is not None else self.wspace
        hspace = hspace if hspace is not None else self.hspace

        fig_w = self.width_cm / 2.54
        fig_h = self.height_cm / 2.54

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=self.dpi)
        axes = render_panels_to_figure(
            self.panels, self.grid_rows, self.grid_cols, fig,
            label_font_size=self.label_font_size,
            label_weight=self.label_weight,
            label_x=self.label_x,
            label_y=self.label_y,
            label_ha=self.label_ha,
            wspace=wspace, hspace=hspace,
            margins=self.margins,
            axes_pad=self.axes_pad,
        )

        # Call plot_func for panels that provide one
        for p in self.panels:
            label = p.get('label', '')
            plot_func = p.get('plot_func')
            if plot_func is not None and label in axes:
                plot_func(axes[label])

        # For panels with no_axis=True, turn off axes after render
        for p in self.panels:
            if p.get('no_axis') and p['label'] in axes:
                axes[p['label']].axis('off')

        if clip_panels:
            clip_all_axes(fig)

        self._fig = fig
        self._axes = axes
        self._wspace = wspace
        self._hspace = hspace
        self._panel_label_ids = {id(t) for t in fig.texts}
        return fig, axes

    def normalize_fonts(self):
        """Normalize all text sizes in the composed figure to the composer's settings.

        Called automatically by :meth:`to_image` and :meth:`save`.

        Mapping:
        - Tick labels → font_size
        - Axis labels (xlabel / ylabel) → axis_label_font_size
        - Axes titles → title_font_size
        - Free text above axes (transAxes y > 1) → title_font_size
        - Free text inside axes → font_size
        - Legend entries + legend title → font_size (overrides any per-call
          ``ax.legend(fontsize=...)`` so legends can never be left inconsistent
          with the rest of the figure)
        - Panel labels → label_font_size (unchanged)

        Text tagged with :func:`exempt_from_font_normalization` is skipped and
        keeps the size it was created with -- use that for schematic text sized
        to fit drawn geometry.
        """
        if self._fig is None:
            return

        font_sz = self.font_size or plt.rcParams.get('font.size', 10)
        label_sz = self.axis_label_font_size or font_sz
        title_sz = self.title_font_size or font_sz

        def _exempt(text):
            return text.get_gid() == NO_FONT_NORMALIZE

        label_pad = self.axis_label_pad

        def _apply_fonts(ax):
            # an exempt axes keeps its authored sizes wholesale, children included
            if _exempt(ax):
                return
            ax.tick_params(labelsize=font_sz)
            ax.xaxis.label.set_fontsize(label_sz)
            ax.yaxis.label.set_fontsize(label_sz)
            if label_pad is not None:
                _default_pad = plt.rcParams.get('axes.labelpad', 4.0)
                if ax.xaxis.labelpad == _default_pad:
                    ax.xaxis.labelpad = label_pad
                if ax.yaxis.labelpad == _default_pad:
                    ax.yaxis.labelpad = label_pad
            if ax.get_title():
                ax.title.set_fontsize(title_sz)

            for text in ax.texts:
                if _exempt(text):
                    continue
                if text.get_transform() is ax.transAxes:
                    _, y = text.get_position()
                    text.set_fontsize(title_sz if y > 1.0 else font_sz)
                else:
                    text.set_fontsize(font_sz)

            # Legend entries (and legend title) are normalized to font_size so a
            # per-call `fontsize=` override cannot leave a legend inconsistent with
            # the rest of the figure.
            legend = ax.get_legend()
            if legend is not None:
                for legend_text in legend.get_texts():
                    legend_text.set_fontsize(font_sz)
                legend_title = legend.get_title()
                if legend_title is not None and legend_title.get_text():
                    legend_title.set_fontsize(font_sz)

            for child in getattr(ax, 'child_axes', []):
                _apply_fonts(child)

        for ax in self._fig.get_axes():
            _apply_fonts(ax)

        panel_ids = getattr(self, '_panel_label_ids', set())
        for text in self._fig.texts:
            if id(text) in panel_ids or _exempt(text):
                continue
            text.set_fontsize(font_sz)

    def normalize_spines(self):
        """Normalize spine linewidths and tick sizes across all axes.

        Called automatically by :meth:`to_image` and :meth:`save`.
        Uses the composer's spine_linewidth, tick_linewidth, and tick_length
        settings. If not set, reads from current rcParams as defaults.

        Also sets rcParams so that any subsequent redraw (e.g. by savefig
        with bbox_inches='tight') preserves the normalized values.
        """
        if self._fig is None:
            return

        spine_lw = self.spine_linewidth
        tick_lw = self.tick_linewidth
        if spine_lw is not None and tick_lw is None:
            tick_lw = spine_lw
        elif tick_lw is not None and spine_lw is None:
            spine_lw = tick_lw
        elif spine_lw is None and tick_lw is None:
            spine_lw = plt.rcParams.get('axes.linewidth', 0.8)
            tick_lw = spine_lw
        tick_len = self.tick_length
        if tick_len is None:
            tick_len = plt.rcParams.get('xtick.major.size', 3.5)
        tick_pad = self.tick_pad

        plt.rcParams['axes.linewidth'] = spine_lw
        plt.rcParams['xtick.major.width'] = tick_lw
        plt.rcParams['ytick.major.width'] = tick_lw
        plt.rcParams['xtick.minor.width'] = tick_lw
        plt.rcParams['ytick.minor.width'] = tick_lw
        plt.rcParams['xtick.major.size'] = tick_len
        plt.rcParams['ytick.major.size'] = tick_len
        if tick_pad is not None:
            plt.rcParams['xtick.major.pad'] = tick_pad
            plt.rcParams['ytick.major.pad'] = tick_pad

        panel_tick_pads = {p['label']: p.get('tick_pad') for p in self.panels
                           if p.get('label')}

        def _apply(ax, pad=tick_pad):
            for spine in ax.spines.values():
                spine.set_linewidth(spine_lw)

            tp_kwargs = {'width': tick_lw}
            if pad is not None:
                tp_kwargs['pad'] = pad

            x_params = ax.xaxis.get_tick_params('major')
            y_params = ax.yaxis.get_tick_params('major')
            x_has_ticks = x_params.get('length', tick_len) > 0
            y_has_ticks = y_params.get('length', tick_len) > 0

            if x_has_ticks:
                ax.tick_params(axis='x', length=tick_len, **tp_kwargs)
            else:
                ax.tick_params(axis='x', length=0, **tp_kwargs)
            if y_has_ticks:
                ax.tick_params(axis='y', length=tick_len, **tp_kwargs)
            else:
                ax.tick_params(axis='y', length=0, **tp_kwargs)

            # children (colorbars, insets) keep the global pad
            for child in getattr(ax, 'child_axes', []):
                _apply(child)

        for ax in self._fig.get_axes():
            _lbl = getattr(ax, '_sciplotlib_panel', None)
            _pad = panel_tick_pads.get(_lbl) if _lbl else None
            _apply(ax, tick_pad if _pad is None else _pad)

    def normalize_linewidths(self):
        """Normalize plot line widths across all axes to a consistent value.

        Sets the linewidth of every ``Line2D`` data line to
        ``self.line_linewidth``.  Tick lines, spines, and patch edges are
        excluded (those are controlled by :meth:`normalize_spines`).

        Only acts when *line_linewidth* was supplied to the constructor;
        if it is ``None`` this method is a no-op so existing figures are
        unaffected.

        Called automatically by :meth:`to_image`, :meth:`save`, and the
        preview/editor methods when *line_linewidth* is set.

        Example
        -------
        ::

            composer = FigureComposer(..., line_linewidth=1.0)
            # … add panels, compose, plot …
            composer.normalize_linewidths()   # or called automatically
        """
        if self._fig is None or self.line_linewidth is None:
            return

        lw = self.line_linewidth

        def _apply(ax):
            for line in ax.get_lines():
                line.set_linewidth(lw)
            for child in getattr(ax, 'child_axes', []):
                _apply(child)

        for ax in self._fig.get_axes():
            _apply(ax)

    @staticmethod
    def _fit_single_ax(fig, ax, cell_bounds, n_iterations=3):
        """Apply the fit_axes_to_cells logic to a single axes.

        ``cell_bounds`` is a 4-tuple (x0, y0, x1, y1) in figure-fraction
        coordinates defining the boundary the axes must stay within.
        """
        from matplotlib.transforms import Bbox
        cell = Bbox([[cell_bounds[0], cell_bounds[1]],
                     [cell_bounds[2], cell_bounds[3]]])
        inv_fig = fig.transFigure.inverted()
        for _ in range(n_iterations):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            tight = ax.get_tightbbox(renderer)
            if tight is None:
                break
            tb = tight.transformed(inv_fig)
            pos = ax.get_position()
            dl = max(0.0, cell.x0 - tb.x0)
            db = max(0.0, cell.y0 - tb.y0)
            dr = max(0.0, tb.x1 - cell.x1)
            if dl + db + dr < 1e-6:
                break
            ax.set_position([
                pos.x0 + dl, pos.y0 + db,
                pos.width - dl - dr, pos.height - db,
            ])

    def fit_axes_to_cells(self, n_iterations=3, constrain_top=False):
        """Shrink each axes so its decorations fit within the gridspec cell.

        Uses matplotlib's tight bounding box to measure how much tick labels,
        axis labels, etc. overflow the cell boundary, then insets the axes
        accordingly. Called automatically by :meth:`to_image` and :meth:`save`.

        Panels whose main axes has ``axis('off')`` (i.e. ``axison == False``)
        are **skipped**, because they have no standard decorations to
        constrain.  This means that panels using ``ax.axis('off')`` with
        inset axes (via :func:`create_inset_grid`) need to manage their
        own margins — use the *left*, *bottom* parameters of
        :func:`create_inset_grid` to reserve space for axis labels on
        the inset subplots.

        Parameters
        ----------
        n_iterations : int
            Number of measure-and-adjust passes (2-3 is usually enough).
        constrain_top : bool
            If False (default), don't shrink axes to fit content above
            the cell (e.g. titles, algorithm labels placed above the axes).
        """
        if self._fig is None:
            return

        fig = self._fig
        gs_kwargs = dict(wspace=self._wspace, hspace=self._hspace)
        if self.margins is not None:
            for key in ('left', 'right', 'top', 'bottom'):
                if key in self.margins:
                    gs_kwargs[key] = self.margins[key]
        gs = GridSpec(self.grid_rows, self.grid_cols, **gs_kwargs)

        inv_fig = fig.transFigure.inverted()

        for _ in range(n_iterations):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            for p in self.panels:
                label = p.get('label', '')
                if not label or label not in self._axes:
                    continue
                if p.get('axes_pad') is not None and p['axes_pad'] == {}:
                    continue

                ax = self._axes[label]
                if not ax.axison:
                    continue
                r0, c0 = p['row'], p['col']
                r1 = r0 + p['rowspan']
                c1 = c0 + p['colspan']
                cell = gs[r0:r1, c0:c1].get_position(fig)

                tight = ax.get_tightbbox(renderer)
                if tight is None:
                    continue
                tb = tight.transformed(inv_fig)

                pos = ax.get_position()
                dl = max(0, cell.x0 - tb.x0)
                db = max(0, cell.y0 - tb.y0)
                dr = max(0, tb.x1 - cell.x1)
                dt = max(0, tb.y1 - cell.y1) if constrain_top else 0

                if dl + db + dr + dt < 1e-6:
                    continue

                ax.set_position([
                    pos.x0 + dl, pos.y0 + db,
                    pos.width - dl - dr, pos.height - db - dt,
                ])

        # Align panels that share the same grid row extent to the same y0 and height.
        # Group by (start_row, end_row) so panels with different rowspans that happen
        # to share a starting row are not incorrectly resized to match each other.
        from collections import defaultdict
        row_groups = defaultdict(list)
        for p in self.panels:
            label = p.get('label', '')
            if not label or label not in self._axes:
                continue
            row_groups[(p['row'], p['row'] + p['rowspan'])].append(label)

        for row_labels in row_groups.values():
            if len(row_labels) < 2:
                continue
            positions = [self._axes[l].get_position() for l in row_labels]
            max_y0 = max(p.y0 for p in positions)
            min_y1 = min(p.y0 + p.height for p in positions)
            for l in row_labels:
                pos = self._axes[l].get_position()
                self._axes[l].set_position([pos.x0, max_y0, pos.width, min_y1 - max_y0])

        # Align panels that share the same grid column extent to the same x0.
        # Group by (start_col, end_col) so panels with different colspans that happen
        # to share a starting column are not incorrectly shifted against each other.
        col_groups = defaultdict(list)
        for p in self.panels:
            label = p.get('label', '')
            if not label or label not in self._axes:
                continue
            ax = self._axes[label]
            if not ax.axison:
                continue
            col_groups[(p['col'], p['col'] + p['colspan'])].append(label)

        for col_labels in col_groups.values():
            if len(col_labels) < 2:
                continue
            positions = [self._axes[l].get_position() for l in col_labels]
            max_x0 = max(p.x0 for p in positions)
            for l in col_labels:
                pos = self._axes[l].get_position()
                dx = max_x0 - pos.x0
                self._axes[l].set_position([max_x0, pos.y0, pos.width - dx, pos.height])

    def compose_image(self, wspace=None, hspace=None, clip_panels=True, width=None, **kwargs):
        """Compose the figure and return it as a marimo-compatible HTML image.

        Prevents aspect-ratio squishing in the marimo UI.
        """
        fig, axes = self.compose(wspace=wspace, hspace=hspace, clip_panels=clip_panels)
        return figure_to_image(fig, width=width, dpi=self.dpi, **kwargs)

    def to_image(self, width=None, **kwargs):
        """Return the current composed figure as a marimo-compatible HTML image.

        Useful when you want to compose and plot on the axes manually, and then
        render the final result as an image in marimo.
        """
        if self._fig is None:
            raise RuntimeError("Call compose() and plot on axes before calling to_image().")
        self.normalize_fonts()
        self.fit_axes_to_cells()
        self.normalize_spines()
        return figure_to_image(self._fig, width=width, dpi=self.dpi, **kwargs)

    def apply_overrides(self, path, verbose=True):
        """Apply hand-tuned position overrides written by the drag editor.

        Call *after* :meth:`compose` + plotting all panels (so colorbars/labels
        exist) and *before* :meth:`save`.  Reapplies positions keyed by stable
        panel-role addresses (``panel:<label>/xlabel|ylabel|colorbar``), so
        manual tweaks survive re-runs with **no code edits and no manual
        tagging**.  A missing file is a no-op.

        Returns ``(n_applied, warnings)``.

        Typical loop::

            fig, axes = composer.compose()
            # ... plot all panels ...
            composer.apply_overrides('figure-3.overrides.json')
            composer.save('figures/figure-3')

        To edit interactively, drag in ``composer.launch_editor(
        overrides_path='figure-3.overrides.json')`` and close the window; the
        colorbar/axis-label moves are written to that file automatically.
        """
        if self._fig is None:
            raise RuntimeError("Call compose() and plot all panels before apply_overrides().")
        from sciplotlib import overrides as _ov
        return _ov.apply_overrides(self._fig, path, verbose=verbose)

    def print_overrides_as_code(self, path, axes_var='axes'):
        """Print the overrides in *path* as explicit matplotlib calls, so you can
        paste them into the compose cell and delete the JSON ("bake" to code).

        Once a figure is final this collapses the two sources (code + JSON) back
        into one — paste the printed lines where ``fig, axes = composer.compose()``
        is in scope (replacing the ``apply_overrides`` call), then remove the JSON.

        Returns the code string.
        """
        from sciplotlib import overrides as _ov
        code = _ov.overrides_as_code(_ov.read_overrides(path), axes_var=axes_var)
        print(code)
        return code

    def launch_editor(self, patch_types=None, overrides_path=None, screen_dpi=100):
        """Normalize the figure and open the interactive drag-position editor.

        Applies :meth:`normalize_fonts`, :meth:`fit_axes_to_cells`, and
        :meth:`normalize_spines` before opening the window, so what you see
        in the editor is pixel-identical to the final PDF/SVG output.
        Blocks until the editor window is closed.

        Call this *after* composing and plotting all panels::

            fig, axes = composer.compose()
            plot_panel_a(axes['a'])
            plot_panel_j(axes['j'])   # ... all panels
            composer.launch_editor()  # adjust, close → paste coordinates back

        Updated coordinates are printed to the terminal where marimo was
        started.  Paste the ``.set_position`` / ``.xy`` / ``.set_xy`` values
        back into your plotting functions and re-run.

        Parameters
        ----------
        patch_types : tuple of type, optional
            Patch subclasses to make draggable.  Defaults to
            ``(Rectangle,)``.  Pass ``None`` to disable patch dragging.
        """
        if self._fig is None:
            raise RuntimeError("Call compose() and plot all panels before launch_editor().")

        self.normalize_fonts()
        self.fit_axes_to_cells()
        self.normalize_spines()

        from sciplotlib.drag_editor import launch_editor as _launch
        kw = {} if patch_types is None else {'patch_types': patch_types}
        _launch(self._fig, overrides_path=overrides_path, screen_dpi=screen_dpi, **kw)

    def launch_editor_panel(self, label, plot_func=None, patch_types=None,
                            screen_dpi=96):
        """Launch the drag editor for a single panel, matching ``normalize=True``.

        Mirrors ``preview_image(label, normalize=True)`` exactly:

        1. Composes the full figure with only *label* drawn (so
           ``fit_axes_to_cells`` runs in the correct GridSpec context).
        2. Applies all three normalisation passes (fonts, cell-fit, spines).
        3. Crops the figure to just the panel's tight bounding box — the same
           crop ``bbox_inches='tight'`` produces when saving the preview PNG.
        4. Opens the interactive editor on the cropped figure.

        You do **not** need to have plotted any other panels first.

        Parameters
        ----------
        label : str
            Panel label to edit, e.g. ``'j'``.
        plot_func : callable, optional
            ``plot_func(ax)`` to draw the panel.  Falls back to the
            ``plot_func`` registered via :meth:`add_panel` if not given.
        patch_types : tuple of type, optional
            Patch subclasses to make draggable.  Defaults to
            ``(Rectangle,)``.  Pass ``None`` to disable patch dragging.
        screen_dpi : int, optional
            DPI for the interactive window (default 96).  The composer's
            ``dpi`` (e.g. 300) is used for layout, then lowered to this
            value before opening so the window fits comfortably on screen.

        Usage::

            composer.launch_editor_panel('j', plot_func=plot_panel_j)
            # or, if plot_func was registered with add_panel:
            composer.launch_editor_panel('j')
        """
        from matplotlib.transforms import Bbox

        p_def = next((p for p in self.panels if p['label'] == label), None)
        if p_def is None:
            raise ValueError(f"No panel with label '{label}'")

        _plot_func = plot_func if plot_func is not None else p_def.get('plot_func')
        if _plot_func is None:
            raise ValueError(
                f"No plot_func for panel '{label}'. "
                f"Pass plot_func=... or register it via add_panel(plot_func=...)."
            )

        # ── Step 1: compose full figure with only the target panel drawn ──────
        # (same as _normalized_preview_image so fit_axes_to_cells has full context)
        snapshot = {p['label']: p.get('plot_func') for p in self.panels}
        for p in self.panels:
            if p['label'] != label:
                p.pop('plot_func', None)
            elif _plot_func is not None:
                p['plot_func'] = _plot_func

        try:
            fig, axes = self.compose()
            self.normalize_fonts()
            self.fit_axes_to_cells()
            self.normalize_spines()
            self.normalize_linewidths()

            # ── Step 2: hide non-target panels ────────────────────────────────
            all_labels = {p.get('label', '') for p in self.panels if p.get('label')}
            for lbl, ax in axes.items():
                if lbl != label:
                    ax.set_visible(False)
            for text_obj in fig.texts:
                if text_obj.get_text() in all_labels - {label}:
                    text_obj.set_visible(False)

            # ── Step 3: compute tight bbox in display (pixel) coordinates ───────
            # We deliberately avoid fig.get_tightbbox(), which returns a
            # TransformedBbox in inches rather than display pixels, causing a
            # unit mismatch when unioned with ax.get_tightbbox() values.
            # Instead, collect ax.get_tightbbox() (always display pixels) from
            # every visible axes, recursing into child_axes.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            target_ax = axes[label]
            bboxes = []
            seen_ids: set[int] = set()

            def _collect_bboxes(a):
                if id(a) in seen_ids or not a.get_visible():
                    return
                seen_ids.add(id(a))
                bb = a.get_tightbbox(renderer)
                if bb is not None:
                    bboxes.append(bb.frozen())
                for child in getattr(a, 'child_axes', []):
                    _collect_bboxes(child)

            for ax in fig.get_axes():
                _collect_bboxes(ax)

            # Include visible figure-level text (panel letter label)
            for text_obj in fig.texts:
                if text_obj.get_visible():
                    bb = text_obj.get_window_extent(renderer)
                    if bb is not None:
                        bboxes.append(bb.frozen())

            # ── Step 4: crop figure to the tight bbox ─────────────────────────
            if bboxes:
                crop = Bbox.union(bboxes)
                fig_w, fig_h = fig.get_size_inches()
                fig_dpi = fig.get_dpi()

                # Use the same padding as bbox_inches='tight' so that
                # element positions match the preview image exactly.
                pad_px = plt.rcParams.get('savefig.pad_inches', 0.1) * fig_dpi
                x0f = max(0.0, (crop.x0 - pad_px) / (fig_w * fig_dpi))
                y0f = max(0.0, (crop.y0 - pad_px) / (fig_h * fig_dpi))
                x1f = min(1.0, (crop.x1 + pad_px) / (fig_w * fig_dpi))
                y1f = min(1.0, (crop.y1 + pad_px) / (fig_h * fig_dpi))

                if x1f > x0f and y1f > y0f:
                    def _remap(a):
                        # get_position() returns the locator-computed position
                        # after the canvas.draw() call above — use that.
                        pos = a.get_position()
                        # Detach any InsetPosition locator BEFORE set_position so
                        # it cannot re-fire on the next draw (window resize etc.)
                        # and double-remap the child relative to the already-
                        # remapped parent.
                        try:
                            a.set_axes_locator(None)
                        except Exception:
                            pass
                        a.set_position([
                            (pos.x0 - x0f) / (x1f - x0f),
                            (pos.y0 - y0f) / (y1f - y0f),
                            pos.width  / (x1f - x0f),
                            pos.height / (y1f - y0f),
                        ])
                        for child in getattr(a, 'child_axes', []):
                            _remap(child)
                    _remap(target_ax)

                    for text_obj in fig.texts:
                        if text_obj.get_visible():
                            tx, ty = text_obj.get_position()
                            text_obj.set_position((
                                (tx - x0f) / (x1f - x0f),
                                (ty - y0f) / (y1f - y0f),
                            ))

                    new_w_in = (x1f - x0f) * fig_w
                    new_h_in = (y1f - y0f) * fig_h

                    # Scale the figure up in inches (DPI stays fixed so
                    # fonts/spines stay at print size) so the GTK window
                    # is comfortably large on screen.
                    longest_in = max(new_w_in, new_h_in)
                    target_px = max(screen_dpi * 14, 1200)  # ≥14" or 1200px
                    scale = max(1.0, target_px / (longest_in * fig_dpi))
                    fig.set_size_inches(new_w_in * scale, new_h_in * scale)
                    fig.set_dpi(fig_dpi)
                else:
                    fig.set_dpi(screen_dpi)
            else:
                fig.set_dpi(screen_dpi)

            from sciplotlib.drag_editor import launch_editor as _launch
            kw = {} if patch_types is None else {'patch_types': patch_types}
            _launch(fig, **kw)

        finally:
            plt.close(self._fig) if self._fig is not None else None
            self._fig = None
            for p in self.panels:
                saved = snapshot.get(p['label'])
                if saved is not None:
                    p['plot_func'] = saved
                else:
                    p.pop('plot_func', None)

    def save(self, path, formats=('pdf', 'svg'), dpi=None, transparent=True,
             bbox_inches='standard'):
        """Save the composed figure to one or more file formats.

        Parameters
        ----------
        bbox_inches : 'standard' or 'tight'
            'standard' (the default) emits exactly ``width_cm`` x
            ``height_cm``, so the figure is the size you asked for and a
            typesetter placing it at a fixed width will not rescale it --
            which would silently rescale every font in it too.
            'tight' re-crops to the ink bounding box, so the saved size no
            longer matches ``width_cm``/``height_cm``.  Note that 'tight'
            also *expands* the canvas to include artists drawn outside it;
            'standard' clips them instead, so check the figure edges when
            switching a layout that was built under 'tight'.
        """
        if self._fig is None:
            raise RuntimeError("Call compose() before save().")
        self.normalize_fonts()
        self.fit_axes_to_cells()
        self.normalize_spines()

        dpi = dpi or self.dpi
        p = Path(path).with_suffix('')
        p.parent.mkdir(parents=True, exist_ok=True)

        # savefig(bbox_inches=None) falls back to rcParams['savefig.bbox'],
        # which stylesheets may set to 'tight'; force it through the rcParam
        # so 'standard' cannot be silently overridden.
        with plt.rc_context({'savefig.bbox': bbox_inches}):
            for fmt in formats:
                save_path = p.with_suffix(f'.{fmt}')
                kwargs = {'dpi': dpi, 'bbox_inches': None}
                if fmt in ('svg', 'png'):
                    kwargs['transparent'] = transparent
                self._fig.savefig(save_path, **kwargs)
                print(f"Saved: {save_path}")

    @classmethod
    def from_yaml(cls, filepath, **overrides):
        """Create a FigureComposer from a saved YAML/JSON layout file.

        Any keyword arguments override the values read from the file.
        """
        parsed = parse_layout_file(filepath)

        init_kwargs = {
            'width_cm': parsed['paper_w_cm'],
            'height_cm': parsed['paper_h_cm'],
            'grid_rows': parsed['grid_rows'],
            'grid_cols': parsed['grid_cols'],
        }
        init_kwargs.update(overrides)

        composer = cls(**init_kwargs)
        for p in parsed['panels']:
            composer.add_panel(
                label=p['label'],
                row=p['row'], col=p['col'],
                rowspan=p['rowspan'], colspan=p['colspan'],
                file=p.get('file'),
            )
        return composer

    @property
    def fig(self):
        return self._fig

    @property
    def axes(self):
        return self._axes
