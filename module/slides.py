"""Slide composition -- :class:`SlideComposer` and :class:`SlideDeck`, for
building a talk as a multi-page PDF.

A slide is a figure, so :class:`SlideComposer` **subclasses**
:class:`~sciplotlib.poster.PosterComposer`: the grid, the normalisation passes,
the position editor, overrides and the layout checker all work unchanged, and so
do ``add_text_block`` and the colour themes.  What a deck adds on top is:

* **A slide-sized canvas.**  ``paper='slide_16x9'`` is 33.867 x 19.05 cm, i.e.
  exactly 13.333 x 7.5 in -- PowerPoint's widescreen default.  A deck exported
  here therefore concatenates with a collaborator's PowerPoint export without
  either being rescaled.
* **Slide-scaled type.**  Between a journal figure (9 pt, read at 30 cm) and an
  A0 poster (20 pt, read at a metre) sits a projected slide, read at 10 m from
  the back row.  Sizes come from :data:`SLIDE_TYPE_SIZES`, not from either.
* **Chrome.**  A title band reserved out of the canvas (so panels never collide
  with the title), and an optional footer with a slide number.
* **Builds.**  A slide declared with ``stages=3`` is rendered three times, with
  ``stage`` counting 0, 1, 2 -- one PDF page each.  That is how a PDF deck does
  progressive reveal, and how Beamer overlays work underneath.
* **Multi-page output.**  :meth:`SlideDeck.save` writes every page into one PDF.

Minimal deck::

    import sciplotlib.slides as splslides

    deck = splslides.SlideDeck(footer='S4SN 2026')

    @deck.slide(title='Dopamine tracks reward, noradrenaline does not',
                stages=2, notes='Land the timing difference here.')
    def encoding(slide, stage):
        slide.add_image('model', row=0, col=0, rowspan=10, colspan=15,
                        file='parts/regression-model.png')
        if stage >= 1:
            slide.add_panel('ev', row=0, col=16, rowspan=10, colspan=15,
                            plot_func=plot_delta_ev)

    deck.save('figures/talk')          # -> figures/talk.pdf, 2 pages

The builder is called once per stage and must be **idempotent** -- it gets a
fresh :class:`SlideComposer` each time, so build the whole slide every call and
gate the parts that appear later behind ``stage``.
"""

from __future__ import annotations

import functools
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

from .compose import PAPER_DIMENSIONS, exempt_from_font_normalization
from .poster import PosterComposer, wrap_text_to_width

#: Canvas width the type sizes below are calibrated for (the 16:9 slide).
REFERENCE_SLIDE_WIDTH_CM = 33.867

#: Type sizes in points at ``type_scale == 1.0``.  Calibrated for a projected
#: slide: the smallest of these (tick labels) is still legible from the back of
#: a lecture theatre, which is the constraint that makes slide figures different
#: from paper figures rather than just bigger.
SLIDE_TYPE_SIZES = {
    'font_size': 14.0,             # tick labels, in-panel text
    'axis_label_font_size': 16.0,
    'title_font_size': 16.0,       # axes titles
    'label_font_size': 20.0,       # panel letters (off by default on slides)
    'section_title_size': 22.0,
    'body_size': 20.0,             # add_text_block / add_bullets
    'slide_title_size': 28.0,
    'subtitle_size': 18.0,
    'footer_size': 11.0,
}

#: Line weights in points at ``type_scale == 1.0``.
SLIDE_LINE_SIZES = {
    'spine_linewidth': 1.2,
    'tick_linewidth': 1.2,
    'tick_length': 4.0,
    'tick_pad': 3.0,
    'axis_label_pad': 4.0,
    'line_linewidth': 1.8,
    'section_linewidth': 1.2,
    'rule_linewidth': 1.0,
}

#: Sentinel for "not given", where None is itself a meaningful value (a slide
#: overriding the deck's footer with no footer at all).
_UNSET = object()


def _with_ticks(ax, plot_func=None):
    """Turn ticks and tick labels on, then run the panel's plot function.

    ``render_panels_to_figure`` gives every fresh panel
    ``tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)``,
    so a paper-figure panel function has to re-enable them by hand.  A deck is
    dozens of small panel functions, and forgetting the incantation costs a
    silent axis, so :class:`SlideComposer` re-enables first and lets the panel
    function override -- calling ``tick_params(labelbottom=False)`` inside
    *plot_func* still wins, because it runs second.

    Module-level and wrapped with ``functools.partial`` rather than a closure,
    so it survives the pickle trip into the position editor's subprocess.
    """
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
    if plot_func is not None:
        plot_func(ax)


# ---------------------------------------------------------------------------
# SlideComposer
# ---------------------------------------------------------------------------

class SlideComposer(PosterComposer):
    """A :class:`~sciplotlib.poster.PosterComposer` sized and styled for one slide.

    Parameters
    ----------
    paper : str, default ``'slide_16x9'``
        Key into :data:`~sciplotlib.compose.PAPER_DIMENSIONS`.  The three slide
        sizes are ``slide_16x9`` (PowerPoint widescreen), ``slide_16x10``
        (Keynote) and ``slide_4x3``.  Ignored if both *width_cm* and
        *height_cm* are given.
    width_cm, height_cm : float, optional
        Explicit canvas size, overriding *paper*.
    grid_rows, grid_cols : int
        The virtual grid, covering the **content area only** -- the title band
        and footer are reserved outside it (see *title_cm*).  So ``row=0`` is
        the top of the content, never underneath the title.
    type_scale : float, optional
        Multiplier on every type size and line weight.  Defaults to
        ``width_cm / 33.867``, i.e. 1.0 on a 16:9 slide.
    title, subtitle : str, optional
        Slide title, drawn in the reserved band.  Passing *title* here rather
        than calling :meth:`add_title` is what reserves the band, because the
        reservation has to happen before the grid margins are computed.
    title_cm : float, optional
        Height of the reserved title band.  Defaults to enough for a title
        (plus a subtitle if given), or 0 when there is no title.
    footer, page_number : str / int, optional
        Footer text (bottom left) and slide number (bottom right).
        :class:`SlideDeck` fills both in.
    theme : str or dict, default ``'light'``
        As :class:`~sciplotlib.poster.PosterComposer` -- ``'light'``, ``'dark'``,
        ``'plain'``, or a dict merged over ``'light'``.
    panel_letters : bool, default False
        Off by default: a slide has no caption to refer to a panel by letter,
        and the letters cost space that a projected figure needs.

    Any other keyword goes to :class:`~sciplotlib.compose.FigureComposer` and
    wins over the scale-derived default.
    """

    def __init__(self, paper='slide_16x9', width_cm=None, height_cm=None,
                 grid_rows=18, grid_cols=32, type_scale=None, theme='light',
                 background=True, panel_letters=False,
                 title=None, subtitle=None, title_cm=None, title_align='left',
                 title_rule=True, footer=None, page_number=None,
                 footer_cm=None, margin_cm=1.4, **kwargs):
        if width_cm is None or height_cm is None:
            if paper not in PAPER_DIMENSIONS:
                raise ValueError(
                    f'unknown paper {paper!r}; known sizes: '
                    f'{", ".join(sorted(PAPER_DIMENSIONS))}. '
                    'Pass width_cm and height_cm for a custom size.')
            paper_w, paper_h = PAPER_DIMENSIONS[paper]
            width_cm = paper_w if width_cm is None else width_cm
            height_cm = paper_h if height_cm is None else height_cm

        scale = (width_cm / REFERENCE_SLIDE_WIDTH_CM if type_scale is None
                 else float(type_scale))
        sizes = {k: v * scale for k, v in SLIDE_TYPE_SIZES.items()}
        sizes.update({k: v * scale for k, v in SLIDE_LINE_SIZES.items()})

        # Slide-only sizes, kept as attributes.  Popped before super() so they
        # do not reach FigureComposer, which would not know them.
        self.slide_title_size = kwargs.pop('slide_title_size',
                                           sizes['slide_title_size'])
        self.subtitle_size = kwargs.pop('subtitle_size', sizes['subtitle_size'])
        self.footer_size = kwargs.pop('footer_size', sizes['footer_size'])
        self.rule_linewidth = kwargs.pop('rule_linewidth', sizes['rule_linewidth'])

        # Reserve the chrome bands *out of the canvas* rather than out of the
        # grid.  Panels then address a grid that is exactly the content area,
        # so no panel can ever be placed under the title.
        if title_cm is None:
            if title is None:
                title_cm = 0.0
            else:
                pt_cm = 2.54 / 72.0          # points -> cm
                title_cm = 1.5 * self.slide_title_size * pt_cm
                if subtitle is not None:
                    title_cm += 1.4 * self.subtitle_size * pt_cm
        if footer_cm is None:
            has_footer = footer is not None or page_number is not None
            footer_cm = (0.85 * scale) if has_footer else 0.0
        self.title_cm = float(title_cm)
        self.footer_cm = float(footer_cm)
        self.margin_cm = float(margin_cm)

        side = margin_cm / width_cm
        if kwargs.get('margins') is None:
            # setdefault would not fire on an explicit margins=None, and
            # FigureComposer reads the four keys unconditionally.
            kwargs['margins'] = {
                'left': side,
                'right': 1.0 - side,
                'bottom': (self.footer_cm + 0.45 * margin_cm) / height_cm,
                'top': 1.0 - (self.title_cm + 0.45 * margin_cm) / height_cm,
            }

        # Sizes FigureComposer understands; anything explicit still wins.
        for key in ('font_size', 'axis_label_font_size', 'title_font_size',
                    'label_font_size', 'spine_linewidth', 'tick_linewidth',
                    'tick_length', 'tick_pad', 'axis_label_pad',
                    'line_linewidth'):
            kwargs.setdefault(key, sizes[key])
        # PosterComposer's own chrome sizes, remapped to the slide scale so its
        # A0-calibrated defaults (76 pt titles) never leak into a slide.
        kwargs.setdefault('section_title_size', sizes['section_title_size'])
        kwargs.setdefault('body_size', sizes['body_size'])
        kwargs.setdefault('poster_title_size', sizes['slide_title_size'])
        kwargs.setdefault('authors_size', sizes['subtitle_size'])
        kwargs.setdefault('affiliations_size', sizes['footer_size'])
        kwargs.setdefault('section_linewidth', sizes['section_linewidth'])
        kwargs.setdefault('wspace', 0.35)
        kwargs.setdefault('hspace', 0.7)
        kwargs.setdefault('label_weight', 'bold')

        # type_scale=1.0 because every size above is already slide-scaled; the
        # real scale is restored onto the attribute straight after.
        super().__init__(paper=paper, width_cm=width_cm, height_cm=height_cm,
                         grid_rows=grid_rows, grid_cols=grid_cols,
                         type_scale=1.0, theme=theme, background=background,
                         panel_letters=panel_letters, **kwargs)
        self.type_scale = scale

        self.title_align = title_align
        self.title_rule = title_rule
        self.footer = footer
        self.page_number = page_number
        self._slide_title = None
        self._chrome_artists = []
        if title is not None:
            self.add_title(title, subtitle=subtitle)

    # -- title ------------------------------------------------------------

    def add_title(self, title, subtitle=None, size=None, subtitle_size=None,
                  color=None, subtitle_color=None, weight='bold', align=None,
                  rule=None, wrap=True):
        """Set the slide title (and optional subtitle) drawn in the title band.

        This **replaces** :meth:`~sciplotlib.poster.PosterComposer.add_title`'s
        poster banner -- a slide has no authors, affiliations or logo strip.

        A long title is wrapped to the content width using real font metrics,
        and then shrunk if the wrapped block is taller than the reserved band.
        The band's height is fixed at construction (``title_cm``) because the
        grid margins derive from it, so shrinking is what keeps a long title
        from growing down over the first row of panels.  Pass a bigger
        ``title_cm`` to the constructor if you would rather have the room.

        Returns self for chaining.
        """
        self._slide_title = {
            'title': title,
            'subtitle': subtitle,
            'size': self.slide_title_size if size is None else size,
            'subtitle_size': (self.subtitle_size if subtitle_size is None
                              else subtitle_size),
            'color': color or self.theme['title_text'],
            'subtitle_color': subtitle_color or self.theme['muted_text'],
            'weight': weight,
            'align': self.title_align if align is None else align,
            'rule': self.title_rule if rule is None else rule,
            'wrap': wrap,
        }
        return self

    def _fit_title(self, spec):
        """Wrap the title to the content width, shrinking it to fit the band.

        Returns ``(lines, fontsize)``.  Without this a long title silently runs
        off the right edge of the canvas -- the commonest way a slide breaks,
        and one that no amount of care at the call site prevents, because the
        author cannot see the wrap point while typing the string.
        """
        m = self.margins
        pt_cm = 2.54 / 72.0
        width_pt = (m['right'] - m['left']) * (self.width_cm / 2.54) * 72.0
        # Height the title text may use: the band, less the subtitle's line.
        avail_cm = self.title_cm
        if spec['subtitle']:
            avail_cm -= 1.4 * spec['subtitle_size'] * pt_cm

        # Measure in the weight the title is actually drawn in.  A bold string
        # is ~10% wider than its regular counterpart, which is enough for a
        # title wrapped against the regular metrics to still run off the page.
        prop = FontProperties(weight=spec['weight'] or 'normal')

        size = spec['size']
        for _ in range(6):
            lines = (wrap_text_to_width(spec['title'], size, width_pt, prop)
                     if spec['wrap'] else spec['title'].split('\n'))
            needed_cm = len(lines) * 1.25 * size * pt_cm
            if needed_cm <= avail_cm or size <= 8.0:
                return lines, size
            size *= max(0.72, avail_cm / needed_cm)
        return lines, size

    # -- panels -----------------------------------------------------------

    def add_panel(self, label, row, col, rowspan, colspan, section=None,
                  ticks=True, **kwargs):
        """Add a panel.  As
        :meth:`~sciplotlib.poster.PosterComposer.add_panel`, plus *ticks*.

        Parameters
        ----------
        ticks : bool, default True
            Restore ticks and tick labels before the panel's plot function
            runs.  **This differs from every other composer**, where a fresh
            panel has them switched off and the plot function must turn them
            back on.  A slide deck is many small panel functions and a silently
            unlabelled axis is easy to miss on a projector, so the default is
            flipped here.  The plot function still has the last word.

            Pass ``ticks=False`` for the paper-figure behaviour.  Panels that
            call ``ax.axis('off')`` need neither -- that hides the ticks again.
        """
        if ticks and not kwargs.get('no_axis'):
            kwargs['plot_func'] = functools.partial(
                _with_ticks, plot_func=kwargs.get('plot_func'))
        return super().add_panel(label, row, col, rowspan, colspan,
                                 section=section, **kwargs)

    # -- convenience placement -------------------------------------------

    def add_bullets(self, label, row, col, rowspan, colspan, bullets, **kwargs):
        """Bulleted list as a grid panel -- :meth:`add_text_block` with bullets."""
        return self.add_text_block(label, row, col, rowspan, colspan,
                                   bullets=bullets, **kwargs)

    def add_prose(self, label, row, col, rowspan, colspan, text, **kwargs):
        """Wrapped paragraph as a grid panel -- :meth:`add_text_block` with text."""
        return self.add_text_block(label, row, col, rowspan, colspan,
                                   text=text, **kwargs)

    def add_image(self, label, row, col, rowspan, colspan, file, **kwargs):
        """Place an image file (``.png``/``.jpg``/``.svg``/``.pkl``) as a panel.

        Shorthand for ``add_panel(..., no_axis=True, no_label=True,
        fit_exempt=True)`` -- an image positions its own content in axes
        fractions, so ``fit_axes_to_cells`` must leave the geometry alone.

        There is no PDF reader: export vector artwork to ``.svg`` (rasterised
        via ``cairosvg``) or a high-dpi ``.png``.
        """
        kwargs.setdefault('no_axis', True)
        kwargs.setdefault('no_label', True)
        kwargs.setdefault('fit_exempt', True)
        return self.add_panel(label, row, col, rowspan, colspan, file=str(file),
                              **kwargs)

    # -- compose ----------------------------------------------------------

    def compose(self, wspace=None, hspace=None, clip_panels=True):
        """Compose the slide: grid panels, then title band and footer."""
        fig, axes = super().compose(wspace=wspace, hspace=hspace,
                                    clip_panels=clip_panels)
        self._draw_footer(fig)
        return fig, axes

    def _chrome_text(self, fig, x, y, s, **kwargs):
        """fig.text that survives normalize_fonts (chrome sets its own sizes)."""
        t = fig.text(x, y, s, **kwargs)
        exempt_from_font_normalization(t)
        self._chrome_artists.append(t)
        return t

    def _draw_title(self, fig):
        """Draw the slide title band.

        Overrides :class:`~sciplotlib.poster.PosterComposer`'s poster banner --
        same hook, called from ``PosterComposer.compose``.
        """
        self._title_artists = []
        spec = self._slide_title
        if spec is None:
            return

        m = self.margins
        align = spec['align']
        x = {'left': m['left'], 'center': 0.5,
             'right': m['right']}[align]
        ha = {'left': 'left', 'center': 'center', 'right': 'right'}[align]

        pt_y = self._fig_frac_per_pt('y')
        top = 1.0 - (0.42 * self.margin_cm) / self.height_cm

        lines, size = self._fit_title(spec)
        t = self._chrome_text(fig, x, top, '\n'.join(lines), ha=ha, va='top',
                              fontsize=size, color=spec['color'],
                              fontweight=spec['weight'], linespacing=1.25)
        self._title_artists.append(t)

        if spec['subtitle']:
            sub_y = top - len(lines) * size * 1.25 * pt_y
            t = self._chrome_text(fig, x, sub_y, spec['subtitle'], ha=ha,
                                  va='top', fontsize=spec['subtitle_size'],
                                  color=spec['subtitle_color'])
            self._title_artists.append(t)

        if spec['rule']:
            # Sit the rule just above the content grid, not just under the
            # title text -- a rule that tracks the text jumps around between
            # one-line and two-line titles.
            y = m['top'] + 0.35 * self.margin_cm / self.height_cm
            line = Line2D([m['left'], m['right']], [y, y],
                          transform=fig.transFigure,
                          color=self.theme['section_edge'],
                          linewidth=self.rule_linewidth, zorder=-5)
            fig.add_artist(line)
            self._title_artists.append(line)
            self._chrome_artists.append(line)

    def _draw_footer(self, fig):
        if self.footer is None and self.page_number is None:
            return
        m = self.margins
        y = 0.42 * self.margin_cm / self.height_cm
        color = self.theme['muted_text']
        if self.footer:
            self._chrome_text(fig, m['left'], y, str(self.footer), ha='left',
                              va='center', fontsize=self.footer_size,
                              color=color)
        if self.page_number is not None:
            self._chrome_text(fig, m['right'], y, str(self.page_number),
                              ha='right', va='center',
                              fontsize=self.footer_size, color=color)

    def describe(self):
        """Print the slide's structure -- title, panels, reserved bands."""
        title = self._slide_title['title'] if self._slide_title else '(untitled)'
        print(f'{self.width_cm:.2f} x {self.height_cm:.2f} cm  '
              f'grid {self.grid_rows}x{self.grid_cols}  '
              f'type_scale {self.type_scale:.2f}')
        print(f'  title: {title!r}  band {self.title_cm:.2f} cm, '
              f'footer {self.footer_cm:.2f} cm')
        for p in self.panels:
            kind = 'text' if p.get('is_text_block') else (
                'image' if p.get('file') else 'axes')
            print(f"  [{p['label']}] {kind:5s} "
                  f"r{p['row']}+{p['rowspan']} c{p['col']}+{p['colspan']}")


# ---------------------------------------------------------------------------
# SlideDeck
# ---------------------------------------------------------------------------

class SlideDeck:
    """An ordered set of slides rendered into one multi-page PDF.

    Each slide is declared with a *builder* -- a function that receives a fresh
    :class:`SlideComposer` and populates it.  The deck owns the lifecycle:
    construct, build, compose, normalise, write a page, close the figure.

    ::

        deck = SlideDeck(footer='S4SN 2026', theme='light')

        @deck.slide(title='The task', notes='30 s, they have seen this already')
        def task(slide):
            slide.add_image('a', 0, 0, 14, 20, file='parts/task.png')

        deck.save('figures/talk')

    Parameters
    ----------
    paper, theme, stylesheet : as :class:`SlideComposer`
        Defaults for every slide.
    footer : str, optional
        Footer text on every slide.
    number_slides : bool, default True
        Draw a slide number bottom-right.  Build stages of one slide share a
        number, so the numbering matches what the audience sees as "a slide".
    first_number : int, default 1
        Number of the first slide.  Set this when your slides sit partway
        through a deck someone else opens -- the number then matches the
        merged PDF rather than restarting at 1.
    **slide_kwargs
        Any other :class:`SlideComposer` keyword, applied to every slide.
    """

    def __init__(self, paper='slide_16x9', theme='light', stylesheet=None,
                 footer=None, number_slides=True, first_number=1,
                 **slide_kwargs):
        self.paper = paper
        self.theme = theme
        self.stylesheet = stylesheet
        self.footer = footer
        self.number_slides = number_slides
        self.first_number = int(first_number)
        self.slide_kwargs = slide_kwargs
        self._specs = []

    # -- declaration ------------------------------------------------------

    def add(self, build, title=None, subtitle=None, stages=1, notes=None,
            name=None, **kwargs):
        """Register a slide.  *build* is ``f(slide)`` or ``f(slide, stage)``.

        Parameters
        ----------
        stages : int, default 1
            Render the slide this many times, with ``stage`` counting
            ``0..stages-1``, one PDF page each.  This is how a PDF deck does a
            progressive reveal.  The builder must draw the whole slide every
            call and gate later content behind ``stage``.
        notes : str, optional
            Speaker notes, written out by :meth:`save_notes`.  Never drawn.
        name : str, optional
            Identifier for per-slide filenames; defaults to the builder's name.
        **kwargs
            Per-slide :class:`SlideComposer` overrides.

        Returns self for chaining.
        """
        if stages < 1:
            raise ValueError(f'stages must be >= 1, got {stages}')
        # Deck-level settings a slide may override individually.  Pulled out of
        # kwargs here because the rest of kwargs goes to SlideComposer, which
        # would reject them.
        self._specs.append({
            'build': build,
            'title': title,
            'subtitle': subtitle,
            'stages': int(stages),
            'notes': notes,
            'name': name or getattr(build, '__name__', f'slide{len(self._specs)}'),
            'number_slides': kwargs.pop('number_slides', None),
            'footer': kwargs.pop('footer', _UNSET),
            'kwargs': kwargs,
        })
        return self

    def slide(self, title=None, subtitle=None, stages=1, notes=None, name=None,
              **kwargs):
        """Decorator form of :meth:`add`::

            @deck.slide(title='Results', stages=2)
            def results(slide, stage):
                ...
        """
        def decorator(build):
            self.add(build, title=title, subtitle=subtitle, stages=stages,
                     notes=notes, name=name, **kwargs)
            return build
        return decorator

    # -- rendering --------------------------------------------------------

    def __len__(self):
        return len(self._specs)

    @property
    def n_pages(self):
        """Total PDF pages -- the sum of every slide's stage count."""
        return sum(s['stages'] for s in self._specs)

    def _make(self, spec, stage, number):
        kwargs = dict(self.slide_kwargs)
        kwargs.update(spec['kwargs'])
        numbered = (self.number_slides if spec['number_slides'] is None
                    else spec['number_slides'])
        footer = self.footer if spec['footer'] is _UNSET else spec['footer']
        slide = SlideComposer(
            paper=self.paper, theme=self.theme, stylesheet=self.stylesheet,
            title=spec['title'], subtitle=spec['subtitle'],
            footer=footer,
            page_number=number if numbered else None,
            **kwargs)
        slide.apply_style()

        build = spec['build']
        try:
            n_params = len(inspect.signature(build).parameters)
        except (TypeError, ValueError):     # builtins, C callables
            n_params = 1
        if n_params >= 2:
            build(slide, stage)
        else:
            build(slide)

        # A builder may compose itself (the marimo pattern: compose, then plot
        # into the axes dict).  Only compose here if it did not.
        if slide.fig is None:
            slide.compose()
        return slide

    def _apply_deck_style(self):
        """Apply the deck's stylesheet globally, without building a figure."""
        probe = SlideComposer(paper=self.paper, theme=self.theme,
                              stylesheet=self.stylesheet, **self.slide_kwargs)
        probe.apply_style()
        return probe

    def render(self, indices=None):
        """Yield ``(spec, stage, number, slide)`` for each page, in order.

        The caller owns the figures and must close them.  :meth:`save` uses
        this and closes as it goes, so a long deck never holds more than one
        figure open.
        """
        number = self.first_number
        for i, spec in enumerate(self._specs):
            if indices is None or i in indices:
                for stage in range(spec['stages']):
                    yield spec, stage, number, self._make(spec, stage, number)
            number += 1

    def preview(self, index, stage=None, width=None):
        """Render one slide and return a PIL image (for marimo/notebooks).

        *stage* defaults to the slide's last stage -- the finished state.
        """
        spec = self._specs[index]
        stage = spec['stages'] - 1 if stage is None else stage
        slide = self._make(spec, stage, self.first_number + index)
        img = slide.to_image(width=width)
        plt.close(slide.fig)
        return img

    # -- output -----------------------------------------------------------

    def save(self, path, formats=('pdf',), dpi=None, check_layout=True,
             min_gap_pt=1.0, normalize_linewidths=False, transparent=False,
             verbose=True):
        """Render every page and write the deck.

        ``'pdf'`` writes a single multi-page ``<path>.pdf``.  ``'png'`` and
        ``'svg'`` have no multi-page form, so they are written per page as
        ``<path>-01.png`` and so on.

        Parameters
        ----------
        check_layout : bool, default True
            Run the ink-based layout check on every page and print findings.
            Advisory only -- pages are written either way.
        normalize_linewidths : bool, default False
            Off by default, matching
            :meth:`~sciplotlib.compose.FigureComposer.save`: flattening every
            ``Line2D`` to one width is wrong for rasters and hand-drawn
            cartoons.  Turn it on for a deck of ordinary line plots.
        transparent : bool, default False
            Slides want their background painted -- a transparent page shows
            whatever the projector or PDF viewer puts behind it.
        """
        stem = Path(path).with_suffix('')
        stem.parent.mkdir(parents=True, exist_ok=True)
        formats = tuple(formats)
        want_pdf = 'pdf' in formats
        others = [f for f in formats if f != 'pdf']

        # Apply the deck's stylesheet BEFORE the PDF file is opened.
        # matplotlib reads pdf.fonttype once, when PdfPages is constructed --
        # not per page.  Applying the stylesheet later (as each slide is built)
        # leaves the file on the default Type 3, and subsetting an OTF/CFF face
        # such as TeX Gyre Heros into Type 3 produces glyph procedures that
        # readers reject ('Missing or bad Type3 CharProc'), so the text renders
        # with wrong advance widths -- letter-spaced and overflowing the page.
        self._apply_deck_style()

        pdf = PdfPages(str(stem.with_suffix('.pdf'))) if want_pdf else None
        n_written = 0
        findings = 0
        try:
            # savefig.bbox='standard' so every page is exactly width_cm x
            # height_cm; a stylesheet setting 'tight' would otherwise re-crop
            # each page to its ink and give the deck ragged page sizes.
            with plt.rc_context({'savefig.bbox': 'standard'}):
                for spec, stage, number, slide in self.render():
                    slide.normalize_fonts()
                    slide.fit_axes_to_cells()
                    slide.normalize_spines()
                    if normalize_linewidths:
                        slide.normalize_linewidths()

                    if check_layout:
                        try:
                            hits = slide.check_layout(min_gap_pt=min_gap_pt,
                                                      verbose=False)
                            if hits:
                                findings += len(hits)
                                label = spec['title'] or spec['name']
                                print(f'[slide {number}.{stage} {label!r}] '
                                      f'{len(hits)} layout finding(s):')
                                for c in hits[:5]:
                                    print(f'    {c}')
                        except Exception as exc:
                            print(f'Layout check skipped: {exc}')

                    page_dpi = dpi or slide.dpi
                    if pdf is not None:
                        pdf.savefig(slide.fig, dpi=page_dpi, bbox_inches=None,
                                    facecolor=slide.fig.get_facecolor())
                    for fmt in others:
                        out = stem.parent / f'{stem.name}-{n_written + 1:02d}.{fmt}'
                        slide.fig.savefig(out, dpi=page_dpi, bbox_inches=None,
                                          transparent=transparent,
                                          facecolor=slide.fig.get_facecolor())
                        if verbose:
                            print(f'Saved: {out}')
                    plt.close(slide.fig)
                    n_written += 1
        finally:
            if pdf is not None:
                pdf.close()

        if verbose:
            if want_pdf:
                print(f'Saved: {stem.with_suffix(".pdf")} '
                      f'({n_written} pages, {len(self._specs)} slides)')
            if check_layout and findings:
                print(f'Layout check: {findings} finding(s) across the deck '
                      '(advisory)')
        return n_written

    def save_notes(self, path, title=None):
        """Write speaker notes to Markdown, one section per slide.

        Notes never appear in the deck PDF, so print this or keep it open on a
        second screen -- a PDF has nowhere to put them.
        """
        p = Path(path).with_suffix('.md')
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [f'# {title}' if title else '# Speaker notes', '']
        number = self.first_number
        for spec in self._specs:
            heading = spec['title'] or spec['name']
            pages = (f' ({spec["stages"]} build steps)' if spec['stages'] > 1
                     else '')
            lines.append(f'## {number}. {heading}{pages}')
            lines.append('')
            lines.append(spec['notes'] or '_No notes._')
            lines.append('')
            number += 1
        p.write_text('\n'.join(lines))
        print(f'Saved: {p}')
        return p

    def describe(self):
        """Print the deck: slide numbers, titles, stage counts, page numbers."""
        print(f'{len(self._specs)} slides, {self.n_pages} pages, '
              f'{self.paper}, theme {self.theme!r}')
        page = 1
        number = self.first_number
        for spec in self._specs:
            span = (f'p{page}' if spec['stages'] == 1
                    else f'p{page}-{page + spec["stages"] - 1}')
            title = spec['title'] or f'({spec["name"]})'
            notes = ' [notes]' if spec['notes'] else ''
            print(f'  {number:>3}. {span:>9}  {title}{notes}')
            page += spec['stages']
            number += 1
