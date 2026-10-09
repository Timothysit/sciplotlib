"""Poster composition -- :class:`PosterComposer`, a :class:`~sciplotlib.compose.FigureComposer`
for conference posters.

A poster is a figure, so everything :class:`FigureComposer` does (grid panels,
font/spine normalisation, the position editor, overrides, the layout checker)
works unchanged.  What a poster adds on top is *chrome* and *scale*:

* **Scale.**  An A0 poster is ~6x the width of a journal figure and is read from
  a metre away, so 9 pt body text is illegible.  Type sizes are derived from
  ``type_scale`` (1.0 at A0), not carried over from the paper defaults.
* **Sections.**  Posters are read as titled blocks -- Introduction, Task,
  Results, Conclusions -- each holding several panels.  :meth:`add_section`
  draws the block and its header, and panels can then be placed *relative to a
  section* so moving a section moves its contents.
* **Prose.**  Posters carry running text: a takeaway line under a result, a
  bulleted conclusion.  :meth:`add_text_block` lays out wrapped body text as a
  first-class grid citizen instead of a hand-placed ``fig.text``.
* **A title bar.**  :meth:`add_title` draws title, authors, affiliations and
  logos across the top.

Minimal poster::

    import sciplotlib.poster as splposter

    poster = splposter.PosterComposer(paper='a0_landscape',
                                      grid_rows=48, grid_cols=72)
    poster.apply_style()
    poster.add_title('Distinct neuromodulatory contributions to strategic decisions',
                     authors='Aloor, Sit, Geuter, Tutt & Duan',
                     affiliations='Sainsbury Wellcome Centre, UCL',
                     rowspan=5)
    poster.add_section('intro', row=5, col=0, rowspan=43, colspan=18,
                       title='Background')
    poster.add_panel('a', row=0, col=1, rowspan=12, colspan=16,
                     section='intro', plot_func=draw_task)
    poster.add_text_block('intro-take', row=13, col=1, rowspan=4, colspan=16,
                          section='intro',
                          bullets=['Mice play matching pennies against an adversary.'])
    fig, axes = poster.compose()
    poster.save('figures/poster')

Sizes and colours all come from the theme (see :data:`POSTER_THEMES`); pass
``theme='dark'`` or a dict to override.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath

from . import collide
from .compose import (
    PAPER_DIMENSIONS,
    FigureComposer,
    exempt_from_font_normalization,
    render_svg,
    set_redraw_hook,
    tint_image,
)


def _is_unfilled(color):
    """True when a face colour paints nothing (``'none'``, or fully transparent)."""
    if color is None:
        return True
    if isinstance(color, str):
        return color.lower() in ('none', 'transparent')
    try:
        from matplotlib.colors import to_rgba
        return to_rgba(color)[3] == 0.0
    except (ValueError, TypeError):
        return False


def relative_luminance(color):
    """Perceived brightness of *color* in [0, 1] (ITU-R BT.709 coefficients)."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def is_dark(color):
    """True if text on *color* should be light. ``'none'`` counts as not dark."""
    if color is None or color == 'none':
        return False
    try:
        return relative_luminance(color) < 0.5
    except ValueError:
        return False

__all__ = [
    'PosterComposer',
    'POSTER_THEMES',
    'POSTER_TYPE_SIZES',
    'REFERENCE_WIDTH_CM',
    'draw_text_block',
    'wrap_text_to_width',
]


#: Width, in cm, at which ``type_scale`` is 1.0 (A0 landscape).  A poster half
#: this wide gets half-size type by default, which keeps the *relative* weight
#: of text to panels constant across paper sizes.
REFERENCE_WIDTH_CM = 118.9


#: Type sizes in points at ``type_scale == 1.0``.  Everything the composer sets
#: is derived from these, so a single ``type_scale`` re-sizes a whole poster.
POSTER_TYPE_SIZES = {
    'font_size': 20.0,             # tick labels, in-panel text
    'axis_label_font_size': 22.0,
    'title_font_size': 24.0,       # axes titles
    'label_font_size': 32.0,       # panel letters
    'section_title_size': 40.0,
    'body_size': 26.0,             # add_text_block
    'poster_title_size': 76.0,
    'authors_size': 34.0,
    'affiliations_size': 26.0,
}


#: Line weights in points at ``type_scale == 1.0``.
POSTER_LINE_SIZES = {
    'spine_linewidth': 1.8,
    'tick_linewidth': 1.8,
    'tick_length': 6.0,
    'tick_pad': 4.0,
    'axis_label_pad': 6.0,
    'line_linewidth': 2.5,
    'section_linewidth': 1.5,
}


#: Colour themes.  ``section_cycle`` tints successive sections when a section is
#: added without an explicit ``facecolor``; give it a single-entry list for a
#: uniform look.
POSTER_THEMES = {
    'light': {
        'background': 'white',
        'text': '#111111',
        'muted_text': '#444444',
        'section_face': '#f4f5f7',
        'section_edge': '#d6d9de',
        'section_header_face': '#e8eaee',
        'section_title': '#111111',
        'title_band_face': 'none',
        'title_text': '#111111',
        'section_cycle': ['#f4f5f7'],
    },
    'dark': {
        'background': '#12141a',
        'text': '#f0f2f5',
        'muted_text': '#b8bec9',
        'section_face': '#1c1f27',
        'section_edge': '#333945',
        'section_header_face': '#262b35',
        'section_title': '#f0f2f5',
        'title_band_face': 'none',
        'title_text': '#f0f2f5',
        'section_cycle': ['#1c1f27'],
    },
    'plain': {           # no blocks, just rules -- the classic academic look
        'background': 'white',
        'text': '#000000',
        'muted_text': '#333333',
        'section_face': 'none',
        'section_edge': 'none',
        'section_header_face': 'none',
        'section_title': '#000000',
        'title_band_face': 'none',
        'title_text': '#000000',
        'section_cycle': ['none'],
    },
}


# ---------------------------------------------------------------------------
# Text measurement and wrapping
# ---------------------------------------------------------------------------

def _string_width_pt(text, fontsize, fontproperties=None):
    """Width of *text* in points when set at *fontsize*.

    Measured with :class:`~matplotlib.textpath.TextPath`, which needs no
    renderer and so works before the figure is drawn (and headless).
    """
    if not text:
        return 0.0
    prop = FontProperties() if fontproperties is None else fontproperties
    prop = prop.copy()
    prop.set_size(fontsize)
    try:
        return TextPath((0, 0), text, size=fontsize, prop=prop).get_extents().width
    except Exception:
        # Any font backend hiccup: fall back to an average-advance estimate
        # rather than failing the whole compose.
        return 0.5 * fontsize * len(text)


def wrap_text_to_width(text, fontsize, width_pt, fontproperties=None,
                       subsequent_indent=''):
    """Greedily wrap *text* to lines no wider than *width_pt* points.

    Existing newlines in *text* are kept as hard breaks.  Returns a list of
    lines.  A single word longer than *width_pt* is left on its own long line
    rather than being hyphenated.
    """
    if width_pt <= 0:
        return text.split('\n')

    lines = []
    for paragraph in text.split('\n'):
        words = paragraph.split()
        if not words:
            lines.append('')
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f'{current} {word}'
            indent = subsequent_indent if lines and current is not words[0] else ''
            if _string_width_pt(indent + candidate, fontsize, fontproperties) <= width_pt:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_text_block(ax, text=None, bullets=None, fontsize=26.0, color='black',
                    weight=None, ha='left', va='top', linespacing=1.35,
                    bullet_char='•', bullet_gap_pt=None, bullet_spacing=0.0,
                    wrap=True,
                    family=None, fig_width_cm=REFERENCE_WIDTH_CM):
    """Draw wrapped body text filling *ax*, replacing whatever was there.

    A module-level function taking only plain arguments, so it can serve as the
    redraw hook registered by :meth:`PosterComposer.add_text_block` -- the
    editor re-runs it on resize and the text re-wraps to the new width instead
    of overflowing.  See :func:`~sciplotlib.compose.set_redraw_hook`.
    """
    ax.clear()
    ax.set_axis_off()
    exempt_from_font_normalization(ax)

    prop = FontProperties(family=family) if family else None
    gap_pt = (fontsize * 0.75) if bullet_gap_pt is None else bullet_gap_pt
    width_pt = ax.get_position().width * (fig_width_cm / 2.54) * 72.0

    x0 = {'left': 0.0, 'center': 0.5, 'right': 1.0}[ha]
    y0 = {'top': 1.0, 'center': 0.5, 'bottom': 0.0}[va]
    common = dict(transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
                  color=color, fontweight=weight, linespacing=linespacing,
                  fontproperties=prop)

    if bullets:
        # Bullets hang: a wrapped continuation line starts under its first
        # line's text, not under the bullet mark. Each line is drawn as its own
        # pair of texts -- mark and body -- at an explicit pitch.
        #
        # The obvious implementation is two multi-line texts, one of marks and
        # one of bodies, and it is what this did until 2026-08-18. It breaks the
        # moment any line contains mathtext (a superscript reference marker, in
        # the case that found it): matplotlib measures each line of a multi-line
        # text from its own content, a math line is taller than a plain one, and
        # the two texts then step down the block at different rates -- so every
        # mark below the math line sits off its own line. Placing each line
        # explicitly at `fontsize * linespacing` cannot drift. That pitch is
        # matplotlib's own multiline spacing, so blocks without mathtext are
        # unchanged.
        lines = []
        for b_i, item in enumerate(bullets):
            wrapped = (wrap_text_to_width(item, fontsize, width_pt - gap_pt, prop)
                       if wrap else item.split('\n'))
            for i, line in enumerate(wrapped):
                # third field: does this line START a bullet after the first?
                # `bullet_spacing` opens a gap there, so a list of multi-line
                # points reads as points rather than as one paragraph.
                lines.append((bullet_char if i == 0 else '', line,
                              i == 0 and b_i > 0))
        gap_frac = gap_pt / max(width_pt, 1e-9)
        height_pt = ax.get_position().height * ax.get_figure().get_size_inches()[1] * 72.0
        pitch = fontsize * linespacing / max(height_pt, 1e-9)
        extra = pitch * float(bullet_spacing)
        # Offsets down the block, so the extra gaps are part of the height the
        # `va` anchor is computed from.
        offsets, acc = [], 0.0
        for _mark, _line, starts_bullet in lines:
            if starts_bullet:
                acc += extra
            offsets.append(acc)
            acc += pitch
        span = acc
        # Where the first line's top goes, so the block as a whole still sits
        # where `va` asks for it.
        top = {'top': 1.0, 'center': 0.5 + span / 2, 'bottom': span}[va]
        line_common = dict(common, va='top')
        for (mark, line, _), off in zip(lines, offsets):
            y = top - off
            if mark:
                ax.text(x0, y, mark, **line_common)
            ax.text(x0 + gap_frac, y, line, **line_common)
    else:
        content = ('\n'.join(wrap_text_to_width(text, fontsize, width_pt, prop))
                   if wrap else text)
        ax.text(x0, y0, content, **common)
    return ax


# ---------------------------------------------------------------------------
# PosterComposer
# ---------------------------------------------------------------------------

class PosterComposer(FigureComposer):
    """A :class:`FigureComposer` sized and styled for a conference poster.

    Parameters
    ----------
    paper : str, default 'a0_landscape'
        Key into :data:`~sciplotlib.compose.PAPER_DIMENSIONS`.  Ignored if
        both *width_cm* and *height_cm* are given.
    width_cm, height_cm : float, optional
        Explicit canvas size, overriding *paper*.
    grid_rows, grid_cols : int
        The virtual grid.  Posters want a fine grid -- the default 48 x 72 on
        A0 landscape is a ~1.75 cm cell, small enough to place section headers
        and text blocks precisely.
    type_scale : float, optional
        Multiplier on every type size and line weight.  Defaults to
        ``width_cm / REFERENCE_WIDTH_CM``, i.e. 1.0 on A0 landscape, so a
        smaller poster gets proportionally smaller type.  Pass a number to
        override (e.g. 1.15 for a text-heavy poster read from further away).
    theme : str or dict, default 'light'
        A key into :data:`POSTER_THEMES`, or a dict of the same shape.  A dict
        is merged over the ``'light'`` theme, so partial overrides are fine.
    background : bool, default True
        Paint the canvas with the theme background.  Turn off for a
        transparent export.
    panel_letters : bool, default True
        Draw a letter above each panel.  Set False for a poster that navigates
        by section headings instead -- lettering exists so a caption can refer
        to a panel, and a poster has no caption.  Individual panels can still
        opt in or out with ``add_panel(..., no_label=...)``.

    Any other keyword is passed through to :class:`FigureComposer`, and wins
    over the theme-derived default -- so ``font_size=24`` fixes body text at
    24 pt regardless of *type_scale*.
    """

    def __init__(self, paper='a0_landscape', width_cm=None, height_cm=None,
                 grid_rows=48, grid_cols=72, type_scale=None, theme='light',
                 background=True, panel_letters=True, **kwargs):
        if width_cm is None or height_cm is None:
            if paper not in PAPER_DIMENSIONS:
                raise ValueError(
                    f'unknown paper {paper!r}; known sizes: '
                    f'{", ".join(sorted(PAPER_DIMENSIONS))}. '
                    'Pass width_cm and height_cm for a custom size.')
            paper_w, paper_h = PAPER_DIMENSIONS[paper]
            width_cm = paper_w if width_cm is None else width_cm
            height_cm = paper_h if height_cm is None else height_cm

        self.paper = paper
        self.type_scale = (width_cm / REFERENCE_WIDTH_CM if type_scale is None
                           else float(type_scale))
        self.theme = self._resolve_theme(theme)
        self.background = background
        self.panel_letters = panel_letters

        scaled = {k: v * self.type_scale for k, v in POSTER_TYPE_SIZES.items()}
        scaled.update({k: v * self.type_scale for k, v in POSTER_LINE_SIZES.items()})

        # Sizes that FigureComposer itself understands; anything the caller
        # passed explicitly wins.
        for key in ('font_size', 'axis_label_font_size', 'title_font_size',
                    'label_font_size', 'spine_linewidth', 'tick_linewidth',
                    'tick_length', 'tick_pad', 'axis_label_pad', 'line_linewidth'):
            kwargs.setdefault(key, scaled[key])
        kwargs.setdefault('margins', {'left': 0.012, 'right': 0.988,
                                      'bottom': 0.012, 'top': 0.988})
        kwargs.setdefault('label_weight', 'bold')

        # Poster-only sizes, kept as attributes.
        self.section_title_size = kwargs.pop('section_title_size',
                                             scaled['section_title_size'])
        self.body_size = kwargs.pop('body_size', scaled['body_size'])
        self.poster_title_size = kwargs.pop('poster_title_size',
                                            scaled['poster_title_size'])
        self.authors_size = kwargs.pop('authors_size', scaled['authors_size'])
        self.affiliations_size = kwargs.pop('affiliations_size',
                                            scaled['affiliations_size'])
        self.section_linewidth = kwargs.pop('section_linewidth',
                                            scaled['section_linewidth'])

        super().__init__(width_cm=width_cm, height_cm=height_cm,
                         grid_rows=grid_rows, grid_cols=grid_cols, **kwargs)

        self.sections = []
        self._section_index = {}
        self._title = None
        self._section_artists = {}
        self._title_artists = []

    # -- theme ------------------------------------------------------------

    @staticmethod
    def _resolve_theme(theme):
        if isinstance(theme, str):
            if theme not in POSTER_THEMES:
                raise ValueError(f'unknown theme {theme!r}; known themes: '
                                 f'{", ".join(sorted(POSTER_THEMES))}')
            return dict(POSTER_THEMES[theme])
        merged = dict(POSTER_THEMES['light'])
        merged.update(theme)
        return merged

    # -- geometry ---------------------------------------------------------

    def cell_rect(self, row, col, rowspan, colspan, fig=None):
        """Figure-fraction ``(x0, y0, width, height)`` of a block of grid cells.

        Spans the *outer* edges of the cells -- the gridspec's wspace/hspace
        gutters around them are included, which is what a section block wants
        (its panels sit inside those gutters).
        """
        fig = fig or self._fig
        if fig is None:
            raise RuntimeError('call compose() before cell_rect()')
        gs = getattr(fig, '_sciplotlib_gridspec', None)
        if gs is None:
            raise RuntimeError('no gridspec on the figure; compose() first')

        r0 = max(0, min(row, self.grid_rows - 1))
        c0 = max(0, min(col, self.grid_cols - 1))
        r1 = max(r0 + 1, min(row + rowspan, self.grid_rows))
        c1 = max(c0 + 1, min(col + colspan, self.grid_cols))

        left = gs[r0:r1, c0:c1].get_position(fig)
        # get_position on a slice already returns the union rect, but it is
        # inset by half a gutter on each side; widen it back out so adjacent
        # sections touch rather than leaving a double gutter between them.
        wpad = _half_gutter(gs, fig, axis='w')
        hpad = _half_gutter(gs, fig, axis='h')
        return (left.x0 - wpad, left.y0 - hpad,
                left.width + 2 * wpad, left.height + 2 * hpad)

    def _fig_frac_per_pt(self, axis='x'):
        """Figure fraction corresponding to one point, along *axis*."""
        inches = self.width_cm / 2.54 if axis == 'x' else self.height_cm / 2.54
        return 1.0 / (inches * 72.0)

    def cm_to_frac(self, cm, axis='x'):
        """Convert centimetres to a figure fraction along *axis*."""
        total = self.width_cm if axis == 'x' else self.height_cm
        return cm / total

    # -- sections ---------------------------------------------------------

    def add_section(self, name, row, col, rowspan, colspan, title=None,
                    header_rows=None, facecolor=None, edgecolor=None,
                    linewidth=None, title_color=None, title_size=None,
                    header_style='bar', corner_radius_cm=0.5, pad_cm=0.0,
                    title_ha='left', zorder=-10):
        """Add a titled block that panels can be placed inside.

        The section occupies grid cells ``row..row+rowspan``, and its *body* --
        what :meth:`add_panel` positions against when given ``section=name`` --
        is everything below the header.

        Parameters
        ----------
        name : str
            Key used by ``section=`` on :meth:`add_panel` and
            :meth:`add_text_block`, and by :meth:`section_body`.
        header_rows : int, optional
            Grid rows the header occupies.  Defaults to enough rows to fit
            *title_size* type, or 0 when there is no title.
        header_style : {'bar', 'plain', 'none'}
            ``'bar'`` fills the header strip with ``section_header_face`` and
            sets the title in it; ``'plain'`` draws the title on the section
            background with a rule beneath it; ``'none'`` draws no header
            decoration but still reserves *header_rows*.
        pad_cm : float
            Grow the block outward by this much on every side.  Use a small
            negative value to leave an explicit gutter between sections.
        corner_radius_cm : float
            Rounding of the block corners.  0 gives square corners.

        Returns self for chaining.
        """
        if name in self._section_index:
            raise ValueError(f'section {name!r} already exists')

        title_size = self.section_title_size if title_size is None else title_size
        if header_rows is None:
            if title:
                # Header height = title type plus half a line of padding above
                # and below, converted to whole grid rows (rounded up).
                row_h_pt = (self.height_cm / 2.54) * 72.0 / self.grid_rows
                header_rows = max(1, int(-(-title_size * 2.0 // row_h_pt)))
            else:
                header_rows = 0
        if header_rows >= rowspan:
            raise ValueError(
                f'section {name!r}: header_rows={header_rows} leaves no body '
                f'in a {rowspan}-row section')

        cycle = self.theme.get('section_cycle') or [self.theme['section_face']]
        section = {
            'name': name,
            'row': row, 'col': col, 'rowspan': rowspan, 'colspan': colspan,
            'title': title,
            'header_rows': header_rows,
            'facecolor': facecolor if facecolor is not None
                         else cycle[len(self.sections) % len(cycle)],
            'edgecolor': edgecolor if edgecolor is not None
                         else self.theme['section_edge'],
            'linewidth': self.section_linewidth if linewidth is None else linewidth,
            'title_color': title_color or self.theme['section_title'],
            'title_size': title_size,
            'header_style': header_style,
            'corner_radius_cm': corner_radius_cm,
            'pad_cm': pad_cm,
            'title_ha': title_ha,
            'zorder': zorder,
        }
        self.sections.append(section)
        self._section_index[name] = section
        return self

    def section_body(self, name):
        """``(row, col, rowspan, colspan)`` of a section's body, in grid cells.

        The body excludes the header, so ``row=0`` for a panel placed with
        ``section=name`` is the first row under the section title.
        """
        s = self._require_section(name)
        return (s['row'] + s['header_rows'], s['col'],
                s['rowspan'] - s['header_rows'], s['colspan'])

    def _require_section(self, name):
        try:
            return self._section_index[name]
        except KeyError:
            known = ', '.join(self._section_index) or '(none defined)'
            raise KeyError(f'unknown section {name!r}; defined: {known}') from None

    def _resolve_placement(self, row, col, rowspan, colspan, section, label):
        """Translate section-relative grid coordinates to absolute ones."""
        if section is None:
            return row, col, rowspan, colspan
        b_row, b_col, b_rowspan, b_colspan = self.section_body(section)
        if row + rowspan > b_rowspan or col + colspan > b_colspan:
            warnings.warn(
                f'panel {label!r} overflows section {section!r}: needs '
                f'{row + rowspan}x{col + colspan} cells of a '
                f'{b_rowspan}x{b_colspan} body. It will be clipped to the grid, '
                f'and will overlap whatever is below or to the right.',
                stacklevel=3)
        return b_row + row, b_col + col, rowspan, colspan

    # -- panels and text --------------------------------------------------

    def add_panel(self, label, row, col, rowspan, colspan, section=None, **kwargs):
        """Add a panel, optionally positioned relative to a section's body.

        With ``section=name``, *row* and *col* are counted from the top-left of
        that section's body (i.e. under its header) rather than from the top-left
        of the poster.  Everything else matches
        :meth:`FigureComposer.add_panel`.
        """
        row, col, rowspan, colspan = self._resolve_placement(
            row, col, rowspan, colspan, section, label)
        if not self.panel_letters:
            kwargs.setdefault('no_label', True)
        super().add_panel(label, row, col, rowspan, colspan, **kwargs)
        self.panels[-1]['section'] = section
        return self

    def add_text_block(self, label, row, col, rowspan, colspan, text=None,
                       bullets=None, section=None, fontsize=None, color=None,
                       weight=None, ha='left', va='top', linespacing=1.35,
                       bullet_char='•', bullet_gap_pt=None, bullet_spacing=0.0,
                       wrap=True, family=None):
        """Lay out wrapped body text as a grid panel.

        Either *text* (a paragraph; ``\\n`` is a hard break) or *bullets* (a
        list of strings, each hanging-indented under its bullet).  The text is
        wrapped to the panel width using real font metrics, so it does not
        overflow into the neighbouring panel.

        The panel is created with ``no_label=True`` (no panel letter),
        ``no_axis=True`` and ``fit_exempt=True``, so ``fit_axes_to_cells``
        leaves its geometry exactly where the grid put it -- text positioned in
        axes fractions must not be rescaled.

        Returns self for chaining.
        """
        if text is None and not bullets:
            raise ValueError('add_text_block needs text= or bullets=')
        if text is not None and bullets:
            raise ValueError('give text= or bullets=, not both')

        spec = {
            'text': text,
            'bullets': list(bullets) if bullets else None,
            'fontsize': self.body_size if fontsize is None else fontsize,
            'color': color or self.theme['text'],
            'weight': weight,
            'ha': ha, 'va': va,
            'linespacing': linespacing,
            'bullet_char': bullet_char,
            'bullet_gap_pt': bullet_gap_pt,
            'bullet_spacing': bullet_spacing,
            'wrap': wrap,
            'family': family,
            'fig_width_cm': self.width_cm,
        }

        def draw(ax):
            draw_text_block(ax, **spec)
            # So the editor can re-wrap this block when the panel is resized.
            # The hook is a module-level function and *spec* is plain data, so
            # both survive the pickle trip into the editor subprocess.
            set_redraw_hook(ax, draw_text_block, **spec)

        self.add_panel(label, row, col, rowspan, colspan, section=section,
                       plot_func=draw, no_axis=True, no_label=True,
                       fit_exempt=True, axes_pad={})
        self.panels[-1]['is_text_block'] = True
        return self

    # -- title ------------------------------------------------------------

    def add_title(self, title, authors=None, affiliations=None, row=0,
                  rowspan=5, col=0, colspan=None, ha='center',
                  title_size=None, authors_size=None, affiliations_size=None,
                  color=None, affiliations_color=None, weight='bold',
                  left_logo=None, right_logo=None, left_logo_tint=None,
                  right_logo_tint=None, logo_height_frac=0.72,
                  logo_pad_cm=1.0, band_facecolor=None, full_bleed=None,
                  rule=True, wrap=True):
        """Draw the poster title bar across grid rows ``row..row+rowspan``.

        Parameters
        ----------
        band_facecolor : colour, optional
            Fill behind the band.  Defaults to the theme's ``title_band_face``
            (``'none'`` in the shipped themes).  Set a dark colour for a
            reversed banner -- the text colours follow automatically, see
            *color* and *affiliations_color*.
        color : colour, optional
            Title and author colour.  Defaults to white on a dark
            *band_facecolor*, and to the theme's title colour otherwise.
        affiliations_color : colour, optional
            Defaults to a *legible* muted tone for the band: a light grey on a
            dark band, the theme's muted text on a light one.  Without this the
            theme's dark muted grey would be invisible on a black banner.
        left_logo, right_logo : path-like, optional
            Image placed at that end of the band.  ``.png``/``.jpg`` are read
            directly; ``.svg`` is rasterised via
            :func:`~sciplotlib.compose.render_svg` (needs cairosvg).  A missing
            or unreadable file warns and is skipped rather than failing the
            render, so check the log before printing.
        left_logo_tint, right_logo_tint : colour, optional
            Recolour the logo's opaque pixels, for placing a dark logo on a
            dark banner.  ``'white'`` gives the usual reversed-mono treatment.
            Only meaningful for a logo with a transparent background; it flattens
            a multi-colour mark to one colour, so check it against the
            institution's own reversed asset before printing.
        logo_height_frac : float
            Logo height as a fraction of the band height.
        logo_pad_cm : float
            Gap between the canvas edge and the logo.
        full_bleed : bool, optional
            Run the band to the paper edges instead of stopping at the grid
            margins.  Defaults to True whenever *band_facecolor* is set: a
            coloured banner with a thin white border around it reads as a
            mistake, while an unfilled title area should stay aligned with the
            columns beneath it.

        Only one title bar is supported; calling this again replaces it.
        """
        band = (self.theme['title_band_face'] if band_facecolor is None
                else band_facecolor)
        dark_band = is_dark(band)
        if color is None:
            color = 'white' if dark_band else self.theme['title_text']
        if affiliations_color is None:
            affiliations_color = ('#d5d8dd' if dark_band
                                  else self.theme['muted_text'])
        self._title = {
            'title': title,
            'authors': authors,
            'affiliations': affiliations,
            'row': row, 'rowspan': rowspan,
            'col': col,
            'colspan': self.grid_cols if colspan is None else colspan,
            'ha': ha,
            'title_size': self.poster_title_size if title_size is None else title_size,
            'authors_size': self.authors_size if authors_size is None else authors_size,
            'affiliations_size': (self.affiliations_size if affiliations_size is None
                                  else affiliations_size),
            'color': color,
            'affiliations_color': affiliations_color,
            'weight': weight,
            'left_logo': str(left_logo) if left_logo else None,
            'right_logo': str(right_logo) if right_logo else None,
            'left_logo_tint': left_logo_tint,
            'right_logo_tint': right_logo_tint,
            'logo_height_frac': logo_height_frac,
            'logo_pad_cm': logo_pad_cm,
            'band_facecolor': band,
            'full_bleed': (band not in (None, 'none') if full_bleed is None
                           else full_bleed),
            'rule': rule,
            'wrap': wrap,
        }
        return self

    # -- compose ----------------------------------------------------------

    def compose(self, wspace=None, hspace=None, clip_panels=True):
        """Compose the poster: grid panels, then section blocks and title bar.

        Section chrome is drawn at negative zorder so it sits behind the
        panels, and every string it adds is exempted from
        :meth:`normalize_fonts` -- section titles are meant to be bigger than
        body text, which is exactly what normalisation would undo.
        """
        fig, axes = super().compose(wspace=wspace, hspace=hspace,
                                    clip_panels=clip_panels)
        if self.background:
            fig.patch.set_facecolor(self.theme['background'])
        self._draw_sections(fig)
        self._draw_title(fig)
        return fig, axes

    def _draw_sections(self, fig):
        self._section_artists = {}
        for s in self.sections:
            artists = []
            pad_x = self.cm_to_frac(s['pad_cm'], 'x')
            pad_y = self.cm_to_frac(s['pad_cm'], 'y')
            x0, y0, w, h = self.cell_rect(s['row'], s['col'],
                                          s['rowspan'], s['colspan'], fig)
            x0, y0 = x0 - pad_x, y0 - pad_y
            w, h = w + 2 * pad_x, h + 2 * pad_y

            radius = self.cm_to_frac(s['corner_radius_cm'], 'x')
            box = FancyBboxPatch(
                (x0 + radius, y0 + radius), w - 2 * radius, h - 2 * radius,
                boxstyle=f'round,pad={radius}',
                transform=fig.transFigure,
                facecolor=s['facecolor'], edgecolor=s['edgecolor'],
                linewidth=s['linewidth'], zorder=s['zorder'], clip_on=False,
                mutation_aspect=(self.width_cm / self.height_cm),
            )
            # A frame is a boundary, not decoration: panels and text belong
            # inside it and must not cross its stroke, so hand it to the layout
            # checker (kind 'section', in DEFAULT_KINDS). Only when it is
            # *unfilled* — the checks are ink-based, and a filled frame inks its
            # whole interior, so everything inside it would read as a collision.
            if _is_unfilled(s['facecolor']):
                setattr(box, collide.POSTER_SECTION_ATTR, s['name'])
            fig.add_artist(box)
            artists.append(box)

            if s['header_rows'] and s['title']:
                hx0, hy0, hw, hh = self.cell_rect(
                    s['row'], s['col'], s['header_rows'], s['colspan'], fig)
                hx0 -= pad_x
                hw += 2 * pad_x
                hy1 = y0 + h            # header top == section top
                hy0 = hy1 - hh - pad_y

                if s['header_style'] == 'bar':
                    # Rounded box for the top corners, then a square patch over
                    # its lower edge so the header reads as a strip flush
                    # against the body rather than a floating pill.
                    bar = FancyBboxPatch(
                        (hx0 + radius, hy0 + radius), hw - 2 * radius,
                        (hy1 - hy0) - 2 * radius,
                        boxstyle=f'round,pad={radius}',
                        transform=fig.transFigure,
                        facecolor=self.theme['section_header_face'],
                        edgecolor='none', zorder=s['zorder'] + 1, clip_on=False,
                        mutation_aspect=(self.width_cm / self.height_cm),
                    )
                    fig.add_artist(bar)
                    artists.append(bar)
                    radius_y = radius * (self.width_cm / self.height_cm)
                    square_h = min(radius_y * 1.5, (hy1 - hy0) / 2)
                    foot = Rectangle(
                        (hx0, hy0), hw, square_h,
                        transform=fig.transFigure,
                        facecolor=self.theme['section_header_face'],
                        edgecolor='none', zorder=s['zorder'] + 1, clip_on=False)
                    fig.add_artist(foot)
                    artists.append(foot)
                elif s['header_style'] == 'plain':
                    rule = plt.Line2D(
                        [hx0 + radius, hx0 + hw - radius], [hy0, hy0],
                        transform=fig.transFigure, color=s['title_color'],
                        linewidth=s['linewidth'], zorder=s['zorder'] + 1,
                        clip_on=False)
                    rule.set_gid('sciplotlib-poster-rule')
                    fig.add_artist(rule)
                    artists.append(rule)

                inset = self.cm_to_frac(0.6, 'x')
                if s['title_ha'] == 'center':
                    tx = hx0 + hw / 2
                elif s['title_ha'] == 'right':
                    tx = hx0 + hw - inset
                else:
                    tx = hx0 + inset
                txt = fig.text(tx, (hy0 + hy1) / 2, s['title'],
                               ha=s['title_ha'], va='center',
                               fontsize=s['title_size'], fontweight='bold',
                               color=s['title_color'], zorder=s['zorder'] + 2)
                exempt_from_font_normalization(txt)
                artists.append(txt)

            self._section_artists[s['name']] = artists

    def _draw_title(self, fig):
        self._title_artists = []
        t = self._title
        if t is None:
            return

        x0, y0, w, h = self.cell_rect(t['row'], t['col'],
                                      t['rowspan'], t['colspan'], fig)

        if t['full_bleed']:
            # Stretch to the paper edges. Only the left/right/top edges move:
            # the bottom stays on the grid so the sections below still butt up
            # against the band.
            x0, w = 0.0, 1.0
            h = 1.0 - y0

        if t['band_facecolor'] not in (None, 'none'):
            band = FancyBboxPatch(
                (x0, y0), w, h, boxstyle='square,pad=0',
                transform=fig.transFigure, facecolor=t['band_facecolor'],
                edgecolor='none', zorder=-20, clip_on=False)
            fig.add_artist(band)
            self._title_artists.append(band)

        # Reserve horizontal room for the logos so a long title cannot run
        # underneath them.
        logo_w = 0.0
        for side in ('left_logo', 'right_logo'):
            if t[side]:
                logo_w = max(logo_w, self._place_logo(fig, t, side, x0, y0, w, h))
        text_x0 = x0 + logo_w
        text_w = w - 2 * logo_w
        tx = {'left': text_x0, 'center': text_x0 + text_w / 2,
              'right': text_x0 + text_w}[t['ha']]

        lines = [(t['title'], t['title_size'], t['weight'], t['color'])]
        if t['authors']:
            lines.append((t['authors'], t['authors_size'], 'normal', t['color']))
        if t['affiliations']:
            lines.append((t['affiliations'], t['affiliations_size'], 'normal',
                          t['affiliations_color']))

        # Wrap each line to the available width, then stack the whole block
        # vertically centred in the band.
        width_pt = text_w * (self.width_cm / 2.54) * 72.0
        rendered = []
        for text, size, weight, color in lines:
            wrapped = (wrap_text_to_width(text, size, width_pt) if t['wrap']
                       else text.split('\n'))
            rendered.append(('\n'.join(wrapped), size, weight, color,
                             len(wrapped)))

        pt_frac = self._fig_frac_per_pt('y')
        gap_frac = 0.45     # inter-block gap, as a fraction of a line height
        heights = [n * size * 1.2 * pt_frac for _, size, _, _, n in rendered]
        gaps = [gap_frac * size * 1.2 * pt_frac
                for _, size, _, _, _ in rendered[1:]]
        total_h = sum(heights) + sum(gaps)

        # Centre the block in the band when it fits; when it doesn't, top-align
        # it so the overflow all happens at the bottom, where it is visible,
        # rather than half of it running off the top of the poster.
        inset = self.cm_to_frac(0.2, 'y')
        y1 = y0 + h
        if total_h > h - 2 * inset:
            warnings.warn(
                f'poster title needs {total_h / pt_frac:.0f} pt of height but the '
                f'title band is only {h / pt_frac:.0f} pt '
                f'(rowspan={t["rowspan"]}). Give add_title a larger rowspan, a '
                f'smaller title_size, or a shorter title.', stacklevel=3)
            cursor = y1 - inset
        else:
            cursor = y0 + h / 2 + total_h / 2

        for i, (text, size, weight, color, n) in enumerate(rendered):
            txt = fig.text(tx, cursor, text, ha=t['ha'], va='top',
                           fontsize=size, fontweight=weight, color=color,
                           linespacing=1.2, zorder=10)
            exempt_from_font_normalization(txt)
            self._title_artists.append(txt)
            cursor -= heights[i]
            if i < len(gaps):
                cursor -= gaps[i]

        if t['rule']:
            line = plt.Line2D([x0, x0 + w], [y0, y0], transform=fig.transFigure,
                              color=self.theme['section_edge'],
                              linewidth=self.section_linewidth, clip_on=False,
                              zorder=-5)
            fig.add_artist(line)
            self._title_artists.append(line)

    def _place_logo(self, fig, t, side, x0, y0, w, h):
        """Draw one logo; return the horizontal room it takes, in fig fraction."""
        path = Path(t[side])
        if not path.is_file():
            warnings.warn(
                f'{side} not found: {path} -- the banner will be drawn without '
                'it. Put the file there and re-render.', stacklevel=3)
            return 0.0
        try:
            if path.suffix.lower() == '.svg':
                # Rasterise at the size it will actually be printed at. The
                # default scale=4 is tuned for a journal figure; on an A0 poster
                # at 300 dpi the same logo lands ~4x bigger and looks soft.
                # Cheap probe render first, only to learn the aspect ratio.
                probe = render_svg(str(path), scale=1)
                aspect = probe.shape[1] / probe.shape[0]
                target_h_px = (h * t['logo_height_frac']
                               * (self.height_cm / 2.54) * self.dpi)
                target_w_px = int(min(max(target_h_px * aspect, 64), 8000))
                img = render_svg(str(path), output_width=target_w_px)
            else:
                img = plt.imread(str(path))
        except Exception as exc:
            warnings.warn(f'could not read {side} {path}: {exc}', stacklevel=3)
            return 0.0

        tint = t.get(f'{side}_tint')
        if tint is not None:
            if img.ndim != 3 or img.shape[2] != 4:
                warnings.warn(
                    f'{side} {path.name} has no alpha channel, so tinting it '
                    f'{tint!r} would flood the whole rectangle. Skipping the '
                    'tint -- use a transparent PNG or SVG.', stacklevel=3)
            else:
                img = tint_image(img, tint)

        img_h, img_w = img.shape[0], img.shape[1]
        logo_h = h * t['logo_height_frac']
        # Keep the aspect ratio in *physical* units, not figure fractions.
        aspect = img_w / img_h
        logo_w = logo_h * aspect * (self.height_cm / self.width_cm)

        pad = self.cm_to_frac(t.get('logo_pad_cm', 1.0), 'x')
        lx = x0 + pad if side == 'left_logo' else x0 + w - pad - logo_w
        ax = fig.add_axes([lx, y0 + (h - logo_h) / 2, logo_w, logo_h],
                          zorder=20)
        ax.imshow(img)
        ax.set_axis_off()
        exempt_from_font_normalization(ax)
        ax._sciplotlib_poster_logo = side
        self._title_artists.append(ax)
        return logo_w + 2 * pad

    # -- introspection ----------------------------------------------------

    def describe(self):
        """A short human-readable summary of the poster's structure.

        Useful in a notebook when the poster is large enough that the layout is
        hard to hold in your head, and in a terminal when checking a render.
        """
        lines = [
            f'{self.paper} poster {self.width_cm:.1f} x {self.height_cm:.1f} cm, '
            f'grid {self.grid_rows} x {self.grid_cols}, type_scale '
            f'{self.type_scale:.2f}',
        ]
        if self._title:
            lines.append(f'  title  rows {self._title["row"]}-'
                         f'{self._title["row"] + self._title["rowspan"]}: '
                         f'{self._title["title"][:60]}')
        by_section = {}
        for p in self.panels:
            by_section.setdefault(p.get('section'), []).append(p)
        for s in self.sections:
            b_row, b_col, b_rowspan, b_colspan = self.section_body(s['name'])
            lines.append(
                f'  section {s["name"]!r} rows {s["row"]}-'
                f'{s["row"] + s["rowspan"]} cols {s["col"]}-'
                f'{s["col"] + s["colspan"]}  body {b_rowspan}x{b_colspan}'
                f'  "{s["title"] or ""}"')
            for p in by_section.get(s['name'], []):
                kind = 'text' if p.get('is_text_block') else 'panel'
                lines.append(f'      {kind} {p["label"]!r} at r{p["row"]} c{p["col"]}'
                             f' ({p["rowspan"]}x{p["colspan"]})')
        loose = by_section.get(None, [])
        if loose:
            lines.append('  unsectioned:')
            for p in loose:
                lines.append(f'      panel {p["label"]!r} at r{p["row"]} c{p["col"]}'
                             f' ({p["rowspan"]}x{p["colspan"]})')
        return '\n'.join(lines)


def _half_gutter(gs, fig, axis='w'):
    """Half the gridspec gutter, in figure fraction, along one axis."""
    if axis == 'w':
        n = gs.ncols
        space = gs.wspace if gs.wspace is not None else plt.rcParams['figure.subplot.wspace']
        left = gs.left if gs.left is not None else plt.rcParams['figure.subplot.left']
        right = gs.right if gs.right is not None else plt.rcParams['figure.subplot.right']
    else:
        n = gs.nrows
        space = gs.hspace if gs.hspace is not None else plt.rcParams['figure.subplot.hspace']
        left = gs.bottom if gs.bottom is not None else plt.rcParams['figure.subplot.bottom']
        right = gs.top if gs.top is not None else plt.rcParams['figure.subplot.top']
    total = right - left
    # cell + gutter widths satisfy: n*cell + (n-1)*space*cell = total
    cell = total / (n + (n - 1) * space)
    return 0.5 * space * cell
