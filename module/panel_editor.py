"""
Tk panel editor — a real application window for adjusting a composed figure.

This is the GUI counterpart to :mod:`sciplotlib.drag_editor`. That editor is a
bare matplotlib window: the figure *is* the interface, and every control is a
mouse gesture or a key press. This one is a proper app — an element tree, numeric
position fields, buttons, a status line — built on the same toolkit as the
``make-layout`` designer in :mod:`sciplotlib.layout`, so the two feel related.

    ┌──────────────────────────────────────────────────────────────┐
    │ Save  Re-render  Undo  Reset all  ☑snap ☑reflow  zoom [Fit]  │
    ├──────────────────┬───────────────────────────────────────────┤
    │ Elements         │                                           │
    │  ▾ panel a       │      figure, with the selection outlined  │
    │      image       │      drag to move, grab an edge to resize │
    │      text: …     │      ctrl-click / lasso for many at once  │
    │  ▾ panel b       │      middle-drag pans, ctrl+wheel zooms  ▓│
    ├──────────────────┤                                          ▓│
    │ Selected         │                                           │
    │  x0 [ 0.0352 ]   │                                           │
    │  y0 [ 0.3699 ]   │                                           │
    │  w  [ 0.1869 ]   │                                           │
    │  h  [ 0.2800 ]   │                                           │
    │  [Apply] [Reset] │  ▓▓▓▓▓▓▓▓▓▓▓                              │
    ├──────────────────┴───────────────────────────────────────────┤
    │ status                                                       │
    └──────────────────────────────────────────────────────────────┘

Poster-scale work (:class:`~sciplotlib.poster.PosterComposer`) drove three of
those controls. **zoom** defaults to ``Fit``, which sizes the backdrop to the
window — an A0 canvas at the old fixed 150 dpi is a 7000 x 5000 px image, so
there was nothing to look at. Above Fit the canvas becomes a scrolling viewport
(wheel scrolls, shift+wheel scrolls sideways, ctrl+wheel or ctrl +/- zooms,
middle- or right-drag pans, ctrl+0 refits). **reflow** re-runs the redraw hooks
registered by :func:`~sciplotlib.compose.set_redraw_hook`, so content laid out
against the axes size — wrapped body text above all — is rebuilt when a panel is
resized rather than overflowing it.

What it edits, what a move means, and how positions are written back are all
shared with the matplotlib editor: both call :func:`drag_editor.collect_items`
and store through :mod:`sciplotlib.overrides`, so a panel move is a delta against
the fitted layout in either one.

**Selecting more than one element.** Ctrl- or shift-click adds an element to the
selection (click it again to drop it), and dragging an empty patch of canvas
lassoes everything the rubber-band touches. Dragging any member of a selection
then moves the whole group by the same offset, as do the arrow keys; Undo
reverts a group move in one step. Grouping a panel together with something drawn
*inside* it moves the panel once — the contained element is carried, not shifted
twice. The numeric position fields stay bound to a single element, so they blank
out while a group is selected. This is the piece the tk editor has that the
matplotlib one does not.

**Editing text.** Double-click a text element (or select it and type in the
*Text* box on the left), then press Ctrl+Enter or *Apply text*. A bare Enter
inserts a newline, so multi-line headings and labels edit as one block. The new
string is written into the overrides JSON as a ``text`` field on the element's
entry and replayed by :func:`~sciplotlib.overrides.apply_overrides` on every
render, so a re-render keeps the edit. Undo and Reset restore the old words.
This is the only edit the editor makes to *content* rather than position; it is
matched by fingerprint against the artist the plotting code still produces, so
it survives so long as that code emits the original string.

Why a rendered image rather than an embedded matplotlib canvas: the backdrop is
drawn once with Agg, and the overlay is drawn with native canvas items that can
carry handles, guides and hover feedback.

**Why edits do not re-render the figure.** matplotlib has no scene graph and no
dirty-region tracking, so every ``draw()`` re-rasterises the whole document —
about 0.8 s for an A0 poster, whatever the dpi, because the cost is in the tens
of thousands of vector segments. Redrawing one *element* costs 0.1 ms for a text
and a few ms for a panel. So a move never redraws the figure. The editor keeps a
**clean plate**: the figure rendered with the elements you are working on left
out (``Artist.set_animated``). A move restores that plate and draws just those
elements back — the blitting trick matplotlib's own animations use, and roughly
what a real vector editor does when it repaints only the region that changed.

The plate hides every text, label and image on the figure plus the few axes
moved most recently, so one plate serves a whole editing session. It is built on
your first edit (one full draw) and rebuilt only when you first touch a new
panel. The consequence to know about: blitted elements are composited on top, so
a move that should slide *under* a neighbour shows correctly only at the next
full render — any zoom change, or *Re-render*. Nothing is saved from the
preview, so this cannot reach the final figure.

Entry points
------------
    composer.launch_editor(overrides_path='fig.overrides.json')   # this editor
    composer.launch_editor(..., editor='mpl')                     # the other one

    uv run python -m sciplotlib.panel_editor /tmp/fig.pkl --overrides o.json

Requires tkinter (stdlib) and Pillow. If ``ttkbootstrap`` is installed it is used
for theming, but it is not required.
"""

from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np

import matplotlib
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg

from sciplotlib import overrides as _overrides
from sciplotlib.drag_editor import (HANDLE_PX, SNAP_PX, _AxesItem, _Item,
                                    _norm_bbox, collect_items, snap_axes_item)

# Selection / handle styling
SEL_COLOUR = '#2e7bf6'
HOVER_COLOUR = '#9bbcf0'
GUIDE_COLOUR = '#f6a02e'
HANDLE_SIZE = 4


class _ShimEvent:
    """Stands in for a matplotlib mouse event where the item wrappers want one.

    ``_AxesItem.drag`` consults ``event.key`` to decide whether to lock the
    aspect ratio; nothing else about the event is used.
    """

    def __init__(self, key=None):
        self.key = key


def _group_of(item):
    """Panel label an item belongs to, for grouping in the tree."""
    if isinstance(item, _AxesItem):
        return item.parent_label
    return getattr(item.ax, '_sciplotlib_panel', None)


def _is_attached(artist):
    """Is *artist* still part of a figure, and so drawable on its own?

    A redraw hook can replace the artists it owns, orphaning the ones the
    editor still holds; asking an orphan to draw raises on its missing figure.
    """
    try:
        return artist.get_figure() is not None
    except Exception:                                 # pragma: no cover
        return False


def _display_name(item):
    """Short label for the element tree."""
    prefix = '[deleted] ' if getattr(item, 'deleted', False) else ''
    return prefix + _display_name_of(item)


def _display_name_of(item):
    addr = getattr(item, 'override_address', None)
    if addr:
        role = addr.split('/', 1)[1] if '/' in addr else 'panel'
        if role.startswith('text'):
            txt = item.artist.get_text().replace('\n', ' ')
            return f'{role}: {txt[:24]}' + ('…' if len(txt) > 24 else '')
        return role
    return item.label


class PanelEditor:
    """The application window.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        A composed figure. Panels are recognised by the ``_sciplotlib_panel``
        label :class:`~sciplotlib.compose.FigureComposer` stamps on each axes.
    overrides_path : str or Path, optional
        Where **Save** writes. Without it the editor still works, but moves can
        only be copied out of the printed summary.
    view_dpi : float or 'fit'
        Render dpi for the on-screen backdrop. Independent of the figure's print
        dpi; all stored positions are relative, so this changes nothing but the
        pixel size of the preview. ``'fit'`` picks the dpi that shows the whole
        figure at once, which is the only workable default for a poster -- an A0
        canvas at 150 dpi is a 7000 x 5000 px backdrop.
    snap : bool
        Snap a dragged axes edge onto a neighbouring panel's edge.
    """

    #: Never render a backdrop larger than this many pixels. A poster at a
    #: careless dpi would otherwise ask for a multi-gigabyte image.
    MAX_BACKDROP_PX = 40_000_000

    #: How many rendered zoom levels to keep. Each is a full RGBA image
    #: (~10 MB for a poster), so this trades memory for instant zoom changes.
    MAX_CACHED_RENDERS = 4

    #: How many clean plates to keep (see :meth:`_clean_plate`). Each is a
    #: full-size Agg buffer, so this is deliberately tiny; two is enough to
    #: keep the last element and the one before it instant to move.
    MAX_CACHED_PLATES = 2

    #: Fit never renders below this many dpi. Fitting a whole A0 poster into a
    #: window works out at ~35 dpi, where 15 pt body text is 7 px tall and
    #: nothing can be read or aimed at. Past this point Fit stops shrinking and
    #: the canvas scrolls instead; type a smaller dpi for a true whole-figure
    #: overview. Print-sized figures fit far above this, so they are unaffected.
    MIN_FIT_DPI = 60.0

    #: How long to wait after the last resize before redrawing crisply.
    SHARPEN_DELAY_MS = 400

    #: Redraw budget, in seconds, for the axes kept movable without rebuilding
    #: the plate (see :meth:`_blit_plan`). Every incremental repaint pays it,
    #: so it is set well under a frame.
    HOT_AXES_BUDGET = 0.08

    def __init__(self, fig, overrides_path=None, view_dpi='fit', snap=True,
                 fit_min_dpi=None):
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                'The panel editor needs tkinter, which is not available in this '
                'Python. Either install it (python3-tk on Debian/Ubuntu) or use '
                "the matplotlib editor instead: launch_editor(editor='mpl')."
            ) from exc
        try:
            from PIL import Image, ImageTk
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                'The panel editor needs Pillow to show the figure. Install it '
                "with `uv add pillow`, or use launch_editor(editor='mpl')."
            ) from exc

        self._tk, self._ttk = tk, ttk
        self._Image, self._ImageTk = Image, ImageTk

        self.fig = fig
        self.overrides_path = overrides_path
        self.snap = snap
        self._fit_mode = (isinstance(view_dpi, str)
                          and view_dpi.strip().lower() == 'fit')
        self.view_dpi = 150.0 if self._fit_mode else float(view_dpi)

        # Agg only: the backdrop is rendered to an image, never shown by a
        # matplotlib GUI backend, so no interactive backend is needed at all.
        matplotlib.use('Agg', force=True)
        self._agg = FigureCanvasAgg(fig)
        self._print_dpi = fig.get_dpi()

        self.items = collect_items(fig)
        # Selection is a list; `selected` is the primary (the last one added),
        # which is what the numeric fields edit. Single-selection code and the
        # tests read `self.selected`, so it stays the canonical "one item".
        self.selection = []
        self.selected = None
        self._drag = None            # list of items being dragged, or None
        self._band = None            # (x0, y0) canvas coords of a rubber-band
        self._band_additive = False  # rubber-band adds to the selection (ctrl)
        self._photo = None
        self._img_size = (0, 0)
        # Cache of the last full Agg render, kept so a window resize can show a
        # cheap PIL-scaled preview instantly instead of a white gap while the
        # (slow, on a poster) real re-render is debounced.
        self._backdrop_flat = None      # last flat RGBA PIL image
        self._rendered_dpi = None       # dpi it was rendered at
        self._rendering = False         # guard: a full Agg draw is in progress
        self._sharpen_after = None      # pending debounced crisp redraw
        self._zoom_was_clamped = False  # last zoom request exceeded the px cap
        self.fit_min_dpi = (self.MIN_FIT_DPI if fit_min_dpi is None
                            else float(fit_min_dpi))
        self._fit_was_floored = False   # Fit hit fit_min_dpi and now scrolls
        self._zoom_shown = None         # last value the zoom box was acted on
        self._render_cache = {}         # dpi -> rendered PIL image
        # Clean plates for incremental redraw: an Agg snapshot of the figure
        # with a chosen set of elements left out, so moving those elements
        # costs a redraw of them alone rather than of the whole figure.
        self._plate_cache = {}          # (artist ids, dpi) -> Agg region
        self._hot_axes = []             # recently moved axes, newest last
        self._draw_cost = {}            # id(axes artist) -> measured seconds
        # Item bounding boxes are cached against this counter, which every
        # edit bumps. Hit-testing walks all ~200 items and runs on every mouse
        # motion, so recomputing the boxes each time made the cursor lag.
        self._geom_serial = 0
        self._box_cache = (-1, {})      # (serial, {id(item): display bbox})
        # Displayed px per rendered px. 1.0 after a real render; a resize shows
        # a cheap PIL-scaled preview at some other ratio, and the coordinate
        # maps fold it in so hit-testing and the overlay stay right on the
        # preview without paying for a ~2 s poster redraw on every resize.
        self._view_scale = 1.0
        self._overlay_ids = []
        self._guide_ids = []
        # Undo stack of *groups*: each entry is a list of items changed
        # together, so one Undo reverts a whole group move at once.
        self._history = []
        self._reflow_error = None    # last redraw-hook failure, shown in status

        self._build_ui()
        # Let tk compute widget geometry before the first render: 'fit' needs a
        # real viewport size, and before the first layout pass every widget
        # still reports 1x1.
        self.root.update_idletasks()
        self._render_backdrop()
        self._populate_tree()
        self._status(f'{sum(1 for i in self.items if isinstance(i, _AxesItem) and i.role == "panel")} '
                     f'panels, {len(self.items)} editable elements. '
                     f'Click one, or pick it from the list.')

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        tk, ttk = self._tk, self._ttk
        # ttkbootstrap gives the same modern look as the make-layout designer
        # (:mod:`sciplotlib.layout`, which themes with 'lumen'); match it. Plain
        # tk is the fallback, but the dated default theme is the giveaway that
        # ttkbootstrap is missing from the interpreter running the editor.
        try:                                   # optional, purely cosmetic
            import ttkbootstrap
            self.root = ttkbootstrap.Window(themename='lumen')
        except Exception:
            self.root = tk.Tk()
        self.root.title('sciplotlib — panel editor')
        # Test hook: build the whole app but never map the window, so the GUI
        # can be exercised on a workstation without a window flashing up.
        if os.environ.get('SPL_EDITOR_HEADLESS'):
            self.root.withdraw()

        # -- toolbar ---------------------------------------------------------
        bar = ttk.Frame(self.root, padding=(6, 4))
        bar.grid(row=0, column=0, columnspan=2, sticky='ew')
        ttk.Button(bar, text='Save', command=self._save).pack(side='left')
        ttk.Button(bar, text='Re-render', command=self._rerender_all
                   ).pack(side='left', padx=(4, 0))
        ttk.Button(bar, text='Undo', command=self._undo).pack(side='left', padx=(4, 0))
        ttk.Button(bar, text='Delete', command=self._delete_selected
                   ).pack(side='left', padx=(4, 0))
        ttk.Button(bar, text='Reset all', command=self._reset_all
                   ).pack(side='left', padx=(4, 0))

        self.snap_var = tk.BooleanVar(value=self.snap)
        ttk.Checkbutton(bar, text='snap', variable=self.snap_var
                        ).pack(side='left', padx=(12, 0))

        self.reflow_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text='reflow', variable=self.reflow_var
                        ).pack(side='left', padx=(6, 0))

        ttk.Label(bar, text='zoom').pack(side='left', padx=(12, 2))
        # 'Fit' first, then a range that spans poster work (a whole A0 at 12 dpi)
        # through print-figure work (a 10 cm panel at 300).
        self.zoom_var = tk.StringVar(
            value='Fit' if self._fit_mode else f'{self.view_dpi:g}')
        # Editable, not readonly: the presets cannot suit every figure, and on
        # a poster the difference between 60 and 90 dpi is the difference
        # between guessing and reading. Type a dpi and press Enter.
        zoom = ttk.Combobox(bar, textvariable=self.zoom_var, width=6,
                            values=('Fit', '12', '25', '50', '75', '100',
                                    '150', '200', '300'))
        zoom.pack(side='left')
        zoom.bind('<<ComboboxSelected>>', self._on_zoom)
        zoom.bind('<Return>', self._on_zoom)
        zoom.bind('<FocusOut>', self._on_zoom)
        self.zoom_lbl = ttk.Label(bar, text='', foreground='#777')
        self.zoom_lbl.pack(side='left', padx=(6, 0))

        # -- left column: element tree + properties --------------------------
        left = ttk.Frame(self.root, padding=(6, 6))
        left.grid(row=1, column=0, sticky='ns')

        ttk.Label(left, text='Elements', font=('TkDefaultFont', 10, 'bold')
                  ).pack(anchor='w')
        self.tree = ttk.Treeview(left, height=18, show='tree', selectmode='extended')
        self.tree.pack(fill='y', expand=True, pady=(2, 8))
        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)

        self.prop = ttk.LabelFrame(left, text='Selected', padding=(6, 4))
        self.prop.pack(fill='x')
        self.addr_lbl = ttk.Label(self.prop, text='(nothing selected)',
                                  wraplength=190, foreground='#555')
        self.addr_lbl.grid(row=0, column=0, columnspan=4, sticky='w', pady=(0, 4))

        self.fields = {}
        for i, name in enumerate(('x0', 'y0', 'w', 'h')):
            ttk.Label(self.prop, text=name).grid(row=1 + i // 2, column=(i % 2) * 2,
                                                 sticky='e', padx=(0, 3))
            var = tk.StringVar()
            ent = ttk.Entry(self.prop, textvariable=var, width=8)
            ent.grid(row=1 + i // 2, column=(i % 2) * 2 + 1, sticky='w', pady=1)
            ent.bind('<Return>', self._apply_fields)
            self.fields[name] = (var, ent)

        ttk.Label(self.prop, text='scale').grid(row=3, column=0, sticky='e',
                                                padx=(0, 3))
        self.zoom_field = tk.StringVar()
        self.zoom_entry = ttk.Entry(self.prop, textvariable=self.zoom_field, width=8)
        self.zoom_entry.grid(row=3, column=1, sticky='w', pady=1)
        self.zoom_entry.bind('<Return>', self._apply_fields)

        btns = ttk.Frame(self.prop)
        btns.grid(row=4, column=0, columnspan=4, sticky='w', pady=(6, 0))
        ttk.Button(btns, text='Apply', command=self._apply_fields).pack(side='left')
        ttk.Button(btns, text='Reset', command=self._reset_selected
                   ).pack(side='left', padx=(4, 0))

        # -- text content editor (single free-text selection only) -----------
        self.text_frame = ttk.LabelFrame(left, text='Text', padding=(6, 4))
        self.text_frame.pack(fill='x', pady=(8, 0))
        self.text_edit = tk.Text(self.text_frame, width=26, height=4,
                                 wrap='word', font=('TkDefaultFont', 9),
                                 undo=True)
        self.text_edit.pack(fill='x')
        # Ctrl+Enter applies; a bare Enter inserts a newline (labels are multi-line).
        self.text_edit.bind('<Control-Return>', self._apply_text)
        trow = ttk.Frame(self.text_frame)
        trow.grid_columnconfigure(0, weight=1)
        trow.pack(fill='x', pady=(4, 0))
        ttk.Button(trow, text='Apply text', command=self._apply_text
                   ).pack(side='left')
        ttk.Label(trow, text='⏎ = newline · Ctrl+⏎ applies',
                  foreground='#999').pack(side='left', padx=(6, 0))
        self._text_target = None      # the item the text box currently edits

        ttk.Label(left, text='drag to move · grab an edge to resize\n'
                            'ctrl/shift-click adds to the selection\n'
                            'drag empty space to lasso · drag any one to\n'
                            'move the group · arrows nudge (shift = 10 px)\n'
                            'double-click text to edit its wording',
                  foreground='#777', justify='left').pack(anchor='w', pady=(8, 0))

        # -- figure canvas ---------------------------------------------------
        # Scrollable: at anything above 'Fit' a poster backdrop is far larger
        # than the window, so the canvas is a viewport onto it rather than
        # being sized to the image.
        right = ttk.Frame(self.root, padding=(0, 6, 6, 6))
        right.grid(row=1, column=1, sticky='nsew')
        # Ask for a *modest* viewport. Tk's grid never shrinks a widget below
        # its requested size, so requesting a poster-sized canvas pinned the
        # viewport open at ~1200 px tall however small the window got: the
        # backdrop was then fitted to an area far larger than what was on
        # screen (so only a corner showed, and clicks landed on nothing), and
        # resizing the window changed no reported size, so it looked stuck.
        # A small request lets the canvas shrink and grow with the window; the
        # window itself is given a sensible opening size below instead.
        self.canvas = tk.Canvas(right, background='#f4f4f4',
                                highlightthickness=0,
                                width=480, height=360)
        self.hbar = ttk.Scrollbar(right, orient='horizontal',
                                  command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(right, orient='vertical',
                                  command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set,
                              yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.vbar.grid(row=0, column=1, sticky='ns')
        self.hbar.grid(row=1, column=0, sticky='ew')
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.canvas.bind('<Button-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Double-Button-1>', self._on_double)
        self.canvas.bind('<Motion>', self._on_hover)
        # In 'fit' the backdrop is sized to the viewport, so a window resize
        # has to re-render it. Debounced: a drag-resize fires Configure
        # continuously and each render is a full Agg draw.
        self._resize_after = None
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        # Coming back from another window can leave the canvas showing nothing
        # (the image item lost, or a stale fit from a transient size). Re-assert
        # the backdrop when the window is exposed or re-mapped, so switching
        # away and back never leaves a blank editor.
        self._expose_after = None
        self.canvas.bind('<Expose>', self._schedule_expose)
        self.canvas.bind('<Visibility>', self._schedule_expose)
        self.root.bind('<Map>', self._schedule_expose)
        self.root.bind('<FocusIn>', self._schedule_expose)

        # Middle-drag pans. So does space-drag, for mice without a middle
        # button; the space bar only pans while the pointer is over the canvas.
        self.canvas.bind('<Button-2>', self._pan_start)
        self.canvas.bind('<B2-Motion>', self._pan_move)
        self.canvas.bind('<Button-3>', self._pan_start)
        self.canvas.bind('<B3-Motion>', self._pan_move)

        # Wheel scrolls vertically, shift+wheel horizontally, ctrl+wheel zooms.
        # X11 sends wheel as Button-4/5; other platforms send <MouseWheel>.
        for seq in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            self.canvas.bind(seq, self._on_wheel)
        self.root.bind('<Control-plus>', lambda e: self._step_zoom(+1))
        self.root.bind('<Control-equal>', lambda e: self._step_zoom(+1))
        self.root.bind('<Control-minus>', lambda e: self._step_zoom(-1))
        self.root.bind('<Control-0>', lambda e: self._set_zoom('Fit'))

        for seq, d in (('<Left>', (-1, 0)), ('<Right>', (1, 0)),
                       ('<Up>', (0, 1)), ('<Down>', (0, -1))):
            self.root.bind(seq, lambda e, d=d: self._nudge(*d, step=1))
            self.root.bind(seq.replace('<', '<Shift-'),
                           lambda e, d=d: self._nudge(*d, step=10))
        self.root.bind('<Control-s>', lambda e: self._save())
        self.root.bind('<Control-z>', lambda e: self._undo())
        # Delete/BackSpace hide the selection; Ctrl+D brings it back. Bound on
        # the canvas and the tree rather than the root so they do not fire
        # while a text-edit box has focus, where BackSpace has to mean
        # backspace.
        for _w in (self.canvas, self.tree):
            _w.bind('<Delete>', lambda e: self._delete_selected())
            _w.bind('<BackSpace>', lambda e: self._delete_selected())
            _w.bind('<Control-d>', lambda e: self._undelete_selected())

        # -- status ----------------------------------------------------------
        self.status = ttk.Label(self.root, text='', anchor='w',
                                padding=(8, 3), foreground='#333')
        self.status.grid(row=2, column=0, columnspan=2, sticky='ew')

        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(1, weight=1)

        # Open at a size that certainly fits on the screen. Without this the
        # window sizes itself to its contents, which for a poster ran past the
        # screen edge and pushed the status bar (and part of the canvas) out of
        # sight. A minimum keeps the controls usable if it is dragged smaller.
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f'{int(sw * 0.82)}x{int(sh * 0.82)}+40+40')
        self.root.minsize(760, 520)

    def _status(self, msg):
        self.status.configure(text=msg)

    def _diag(self, tag):
        """Print live geometry when SPL_EDITOR_DIAG=1.

        Tk's reported sizes depend on the window manager, so when the editor
        misbehaves on a machine that cannot be reproduced elsewhere this is the
        only way to see what it actually got.
        """
        if not os.environ.get('SPL_EDITOR_DIAG'):
            return
        try:
            print(f'[diag/{tag}] screen='
                  f'{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}'
                  f' root={self.root.winfo_width()}x{self.root.winfo_height()}'
                  f'+{self.root.winfo_x()}+{self.root.winfo_y()}'
                  f' canvas={self.canvas.winfo_width()}x{self.canvas.winfo_height()}'
                  f' viewport={self._viewport_size()}'
                  f' img={self._img_size} scale={self._view_scale:.3f}'
                  f' dpi={self.view_dpi:.1f} fit={self._fit_mode}'
                  f' backdrop={bool(self.canvas.find_withtag("backdrop"))}',
                  flush=True)
        except Exception as exc:                      # pragma: no cover
            print(f'[diag/{tag}] failed: {exc!r}', flush=True)

    # ── rendering ────────────────────────────────────────────────────────────

    def _viewport_size(self):
        """Pixel size of the canvas viewport, falling back to the screen."""
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w > 1 and h > 1:
            # Never trust a viewport bigger than the screen. Tk can report the
            # requested rather than the on-screen size, and fitting the
            # backdrop to that would push most of the figure out of view.
            return (min(w, self.root.winfo_screenwidth()),
                    min(h, self.root.winfo_screenheight()))
        # Before the first map the canvas has no size yet; estimate from the
        # screen, leaving room for the element tree and window furniture.
        return (max(400, int(self.root.winfo_screenwidth() * 0.78)),
                max(300, int(self.root.winfo_screenheight() * 0.80)))

    def _fit_dpi(self):
        """Render dpi at which the whole figure fits in the viewport."""
        fig_w_in, fig_h_in = self.fig.get_size_inches()
        vw, vh = self._viewport_size()
        pad = 12                     # so the edges are not flush with the frame
        dpi = min((vw - pad) / max(fig_w_in, 1e-6),
                  (vh - pad) / max(fig_h_in, 1e-6))
        self._fit_was_floored = dpi < self.fit_min_dpi
        return max(4.0, self.fit_min_dpi, dpi)

    def _clamp_dpi(self, dpi):
        """Hold the backdrop under MAX_BACKDROP_PX, whatever dpi was asked for."""
        fig_w_in, fig_h_in = self.fig.get_size_inches()
        px = (fig_w_in * dpi) * (fig_h_in * dpi)
        if px <= self.MAX_BACKDROP_PX:
            return dpi, False
        scale = (self.MAX_BACKDROP_PX / px) ** 0.5
        return dpi * scale, True

    def _busy(self, msg):
        """Show the window as working. A full Agg draw blocks the tk loop for
        seconds, during which nothing repaints and no click lands; saying so
        makes the freeze read as work in progress rather than a dead window."""
        try:
            self.root.configure(cursor='watch')
            self.canvas.configure(cursor='watch')
            self._status(msg)
            self.root.update_idletasks()
        except Exception:                             # pragma: no cover
            pass

    def _unbusy(self):
        try:
            self.root.configure(cursor='')
            self.canvas.configure(cursor='')
        except Exception:                             # pragma: no cover
            pass

    def _render_backdrop(self):
        """Re-draw the whole figure to an image and show it.

        Slow — see the note on incremental redraw in the module docstring. This
        is for a zoom change, *Re-render*, or an edit the incremental path
        cannot express; a plain move must not come through here.

        Guarded against re-entrancy: on a heavy figure a single Agg draw can run
        for a second or more, blocking the tk loop, so a stray event that lands
        mid-draw must not launch a second, overlapping render."""
        if self._rendering:
            return
        self._rendering = True
        self._busy('Rendering… the window is busy for a moment.')
        try:
            self._render_backdrop_locked()
        finally:
            self._rendering = False
            self._unbusy()

    def _render_backdrop_locked(self):
        if self._fit_mode:
            self.view_dpi = self._fit_dpi()
        dpi, clamped = self._clamp_dpi(self.view_dpi)
        self.view_dpi = dpi
        # Set the dpi before the cache check, not only on the draw path: hit
        # testing measures artists through the renderer, so the figure's dpi
        # has to match the image on screen even when that image came from the
        # cache and no draw happened.
        self.fig.set_dpi(self.view_dpi)

        # Re-use an identical earlier render. Going back to a zoom level you
        # have already visited is then instant instead of costing another
        # multi-second draw. Keyed on dpi and invalidated whenever the figure
        # changes (see _invalidate_render_cache).
        key = round(dpi, 3)
        cached = self._render_cache.get(key)
        if cached is not None:
            self._show_rendered(cached, clamped)
            return

        # The draw below blocks the UI on a big poster; say so, and flush the
        # message to screen first, so the freeze reads as work rather than a hang.
        self.zoom_lbl.configure(text='rendering…')
        self.zoom_lbl.update_idletasks()

        self._agg.draw()
        flat = self._flat_image()
        # Keep a bounded number of rendered sizes; a poster image is ~10 MB, so
        # an unbounded cache would grow into the hundreds of MB over a session.
        if len(self._render_cache) >= self.MAX_CACHED_RENDERS:
            self._render_cache.pop(next(iter(self._render_cache)))
        self._render_cache[key] = flat
        self._show_rendered(flat, clamped)

    def _flat_image(self):
        """The current Agg buffer as an opaque RGB image, detached from Agg.

        RGB throughout, not RGBA: the buffer has to be copied anyway (Agg
        reuses it on the next draw), and dropping the alpha channel makes that
        copy, every later rescale, and the upload to tk all markedly cheaper,
        while cutting a poster backdrop from 62 MB to 46 MB. A figure with a
        see-through facecolor is composited onto white first -- white being
        what the page behind the saved PDF would be.
        """
        w, h = self._agg.get_width_height()
        img = self._Image.frombuffer(
            'RGBA', (w, h), self._agg.buffer_rgba(), 'raw', 'RGBA', 0, 1)
        if self.fig.get_facecolor()[3] < 1.0:
            flat = self._Image.new('RGBA', (w, h), (255, 255, 255, 255))
            flat.alpha_composite(img)
            img = flat
        return img.convert('RGB')

    def _put_image(self, img):
        """Show *img* as the backdrop, re-using the PhotoImage where possible.

        Uploading into the existing photo is about twice as fast as building a
        new one and avoids destroying and recreating the canvas item, which on
        a large image shows as a flash.
        """
        w, h = img.size
        if self._photo is not None and self._img_size == (w, h):
            self._photo.paste(img)
            return
        self._photo = self._ImageTk.PhotoImage(img)
        self._img_size = (w, h)
        self.canvas.delete('backdrop')
        self.canvas.create_image(0, 0, image=self._photo, anchor='nw',
                                 tags='backdrop')
        self.canvas.tag_lower('backdrop')
        # Scroll over the image; the canvas keeps whatever size the window
        # gives it rather than growing to the backdrop, which on a poster
        # would be several times the screen.
        self.canvas.configure(scrollregion=(0, 0, w, h))

    def _show_rendered(self, flat, clamped=False):
        """Put an already-rendered image on the canvas at 1:1."""
        self._geom_changed()      # a render can change fig dpi, hence the boxes
        w, h = flat.size
        self._backdrop_flat = flat
        self._rendered_dpi = self.view_dpi
        self._view_scale = 1.0          # displayed image now matches the render
        self._put_image(flat)
        note = ' (clamped)' if clamped else ''
        self.zoom_lbl.configure(
            text=f'{self.view_dpi:.0f} dpi · {w}x{h} px{note}')
        self._refresh_overlay()
        self._diag('render')

    def _invalidate_render_cache(self):
        """Drop cached renders — the figure itself changed, so they are stale."""
        self._render_cache.clear()

    # ── incremental (blit) redraw ────────────────────────────────────────────
    # A full Agg draw of a poster costs ~0.8 s, because matplotlib has no scene
    # graph and no dirty-region tracking: every draw re-rasterises all tens of
    # thousands of segments from scratch. Redrawing one *element* costs 0.1 ms
    # (a text) to ~15 ms (a whole panel with its data).
    #
    # So a move must not redraw the figure. Instead we keep a "clean plate" --
    # the figure rendered with the moving elements left out -- restore it, and
    # draw just those elements back at their new position. That is the same
    # blitting trick matplotlib's own animations use, and it is roughly what
    # makes a real vector editor (Inkscape) feel instant: it, too, repaints
    # only the region that changed rather than re-rasterising the document.

    def _blit_artists(self, items):
        """The artists that a move of *items* actually repaints, or ``[]``.

        An axes carries its insets: ``_AxesItem._set_bounds`` translates
        ``child_axes`` with the parent, so they have to be lifted out of the
        plate and redrawn too, or the inset is left behind at the old spot.
        """
        artists = []
        for it in items:
            art = getattr(it, 'artist', None)
            if art is None or not _is_attached(art):
                return []
            artists.append(art)
            artists.extend(getattr(art, 'child_axes', []))
        return artists

    def _leaf_artists(self):
        """Every editable artist that is not an axes: text, labels, images.

        Detached artists are skipped. A panel's redraw hook may *replace* the
        artists it owns -- a text block rebuilds its lines -- which leaves the
        originals in ``self.items`` no longer attached to any figure. Drawing
        one raises, and would take the whole incremental repaint down with it.
        """
        return [it.artist for it in self.items
                if not isinstance(it, _AxesItem)
                and getattr(it, 'artist', None) is not None
                and _is_attached(it.artist)]

    def _blit_plan(self, items):
        """``(hide, draw)`` artist lists for repainting *items*, or ``(None, None)``.

        The plate hides one set: every *leaf* (text, axis label, placed image)
        plus the axes moved most recently. Everything hidden is redrawn, so the
        set is chosen to be cheap to redraw and broad enough that consecutive
        edits reuse the same plate:

        * Leaves cost ~0.1 ms each, so all of them go in unconditionally. Any
          text you move is then free after the first plate is built.
        * An axes costs milliseconds to tens of milliseconds, so they are kept
          only while their *measured* redraw cost fits ``HOT_AXES_BUDGET``.
          That is what lets you go back and forth between a few panels without
          paying for a full figure draw on every switch, while stopping one
          expensive panel from slowing every later edit down.
        """
        if not items or any(getattr(it, 'artist', None) is None for it in items):
            return None, None

        moved_axes = [it for it in items if isinstance(it, _AxesItem)]
        for it in moved_axes:                    # most recently moved goes last
            if it in self._hot_axes:
                self._hot_axes.remove(it)
            self._hot_axes.append(it)
        hot, spent = [], 0.0
        for it in reversed(self._hot_axes):      # newest first
            cost = self._draw_cost.get(id(it.artist), 0.0)
            if it not in moved_axes and spent + cost > self.HOT_AXES_BUDGET:
                break                            # this one and all older go
            hot.append(it)
            spent += cost
        self._hot_axes = list(reversed(hot))

        artists = self._leaf_artists()
        for it in self._hot_axes:
            if not _is_attached(it.artist):
                continue
            artists.append(it.artist)
            artists.extend(getattr(it.artist, 'child_axes', []))
        # Everything that moved has to be hidden by the plate, or its old
        # position stays painted in it and the move shows as a duplicate.
        ids = {id(a) for a in artists}
        moved = self._blit_artists(items)
        if not moved or any(id(a) not in ids for a in moved):
            return None, None
        # Painting order matters where elements overlap; follow zorder.
        return artists, sorted(artists, key=lambda a: a.get_zorder())

    def _plate_key(self, artists):
        """Cache key for the plate hiding *artists* at the current render dpi."""
        return (frozenset(id(a) for a in artists), round(self._rendered_dpi, 3))

    def _clean_plate(self, artists):
        """``(key, plate)`` for the figure drawn *without* ``artists``.

        Building one costs a full draw, so plates are cached: the first move of
        an element pays for the draw and every later move of it is free.
        ``set_animated`` is what excludes an artist from ``Figure.draw``.
        """
        key = self._plate_key(artists)
        plate = self._plate_cache.get(key)
        if plate is not None:
            return key, plate
        for art in artists:
            art.set_animated(True)
        try:
            self.fig.set_dpi(self._rendered_dpi)
            self._agg.draw()
            plate = self._agg.copy_from_bbox(self.fig.bbox)
        finally:
            for art in artists:
                art.set_animated(False)
        if len(self._plate_cache) >= self.MAX_CACHED_PLATES:
            self._plate_cache.pop(next(iter(self._plate_cache)))
        self._plate_cache[key] = plate
        return key, plate

    def _blit_change(self, items, must_hide=()):
        """Repaint only *items* over their clean plate. True if that worked.

        The caller falls back to a full render when this returns False.

        What you get is a faithful preview with one caveat: the moved elements
        are composited on top, so a move that should slide *underneath* a
        neighbour will not show that until the next full render (any zoom
        change, or Re-render). Nothing is saved from the preview -- overrides
        record positions -- so this cannot affect the final figure.
        """
        if (self._backdrop_flat is None or self._rendering
                or not self._rendered_dpi):
            return False
        hide, draw = self._blit_plan(items)
        if not hide:
            return False
        # A redraw hook rebuilds the contents of its axes. That is fine as long
        # as the plate hides that axes -- it gets redrawn from scratch here
        # anyway. If the plate holds it, the plate now shows the old contents.
        if must_hide:
            ids = {id(a) for a in hide}
            if any(id(a) not in ids for a in must_hide):
                return False
        # Hold the on-screen size across the repaint: the backdrop is rebuilt
        # at its rendered dpi, but the view may be showing a scaled preview.
        displayed_dpi = self._rendered_dpi * self._view_scale
        # Building a plate costs one full draw. It happens on the first edit,
        # and again the first time a new panel is moved; every edit in between
        # is instant. Say which one this is, so the wait is not a mystery.
        building = self._plate_key(hide) not in self._plate_cache
        if building:
            self._busy('Preparing a fast redraw — a moment, once. '
                       'Edits after this are instant.')
        try:
            key, plate = self._clean_plate(hide)
            self._agg.restore_region(plate)
            for art in draw:
                started = time.perf_counter()
                self.fig.draw_artist(art)
                if isinstance(art, Axes):
                    # Measured, not guessed: panels differ by 20x in redraw
                    # cost, and _blit_plan spends this to decide what it can
                    # afford to keep hot.
                    self._draw_cost[id(art)] = time.perf_counter() - started
            flat = self._flat_image()
        except Exception as exc:                      # pragma: no cover
            print(f'[panel_editor] incremental redraw failed, falling back to '
                  f'a full render: {exc!r}')
            if os.environ.get('SPL_EDITOR_DIAG'):
                import traceback
                traceback.print_exc()
            self._plate_cache.clear()
            return False
        finally:
            if building:
                self._unbusy()
        # A plate that hides set H stays valid only if everything that moved
        # was inside H — anything else moved *within* the plate, so the plate
        # now shows it at its old place. The plate we just used is the only one
        # that can satisfy that, so the rest go.
        self._plate_cache = {key: plate}
        self._backdrop_flat = flat
        self._render_cache.clear()     # cached zoom levels show the old layout
        if self._view_scale == 1.0:
            # Common case: the view is showing the render 1:1, so the new image
            # can go straight up without a rescale.
            self._put_image(flat)
            self._refresh_overlay()
        else:
            self._preview_to_dpi(displayed_dpi)
        self._diag('blit')
        return True

    # ── geometry cache ───────────────────────────────────────────────────────

    def _geom_changed(self):
        """Something moved: item boxes cached for hit-testing are now stale."""
        self._geom_serial += 1

    # ── zoom and pan ─────────────────────────────────────────────────────────

    ZOOM_STEPS = (12, 25, 50, 75, 100, 150, 200, 300)

    def _set_zoom(self, value):
        """Apply a zoom setting: the string 'Fit', or a dpi as a number."""
        self.zoom_var.set(str(value))
        self._zoom_shown = None          # force it through, even if unchanged
        self._on_zoom()

    def _on_zoom(self, _event=None):
        value = self.zoom_var.get().strip()
        # The box is editable and also fires on focus-out, so most calls carry
        # a value that has not changed. Redrawing for those would mean a stray
        # render every time focus moved away from the toolbar.
        if value == getattr(self, '_zoom_shown', None):
            return
        self._fit_mode = value.lower() == 'fit'
        if not self._fit_mode:
            try:
                dpi = float(value.rstrip('dpi ').strip() or 'x')
            except ValueError:
                self._status(f'Not a zoom level: {value!r} — '
                             'type a number of dpi, or Fit.')
                return
            if dpi <= 0:
                self._status('A zoom level has to be greater than zero.')
                return
            self.view_dpi = dpi
            # Clamp now, not when the deferred render runs: the requested dpi
            # is the editor's state as soon as it is picked, and a reckless
            # value must never be left standing even for a moment.
            requested = self.view_dpi
            self.view_dpi, _ = self._clamp_dpi(self.view_dpi)
            # The clamp is reported here because by the time the preview runs
            # the dpi is already capped and looks unremarkable.
            self._zoom_was_clamped = self.view_dpi < requested
        else:
            self._zoom_was_clamped = False
        # Show the new zoom immediately by scaling the cached render, then
        # sharpen. Zooming in is how a poster becomes legible at all, so it has
        # to feel instant rather than costing a ~2 s Agg redraw per step.
        self._preview_to_dpi(self.view_dpi if not self._fit_mode
                             else self._fit_dpi())
        self._zoom_shown = value
        self._status('zooming… sharpening')
        self._schedule_sharpen(350)

    def _schedule_sharpen(self, delay=None):
        """Queue the crisp redraw that follows an instant preview.

        Zooming and resizing both show a cheap rescale of the last render
        first, so they never block. Without this the preview is where it
        stopped: the window opens showing the startup render — sized from an
        estimate, since the canvas has no size before it is mapped — stretched
        to whatever size the window turned out to be, i.e. permanently blurred.
        """
        if self._sharpen_after is not None:
            self.root.after_cancel(self._sharpen_after)
        self._sharpen_after = self.root.after(
            self.SHARPEN_DELAY_MS if delay is None else delay, self._sharpen_view)

    def _sharpen_view(self):
        """The debounced crisp render behind a zoom or a resize."""
        self._sharpen_after = None
        try:
            self._render_backdrop()
            self._status(f'{self.view_dpi:.0f} dpi'
                         + ('  ·  Fit floored at '
                            f'{self.fit_min_dpi:g} dpi so it stays legible — '
                            'scroll, or type a smaller dpi to see it all'
                            if self._fit_mode and self._fit_was_floored else ''))
        except Exception as exc:                      # pragma: no cover
            self._status(f'render failed: {exc}')
            print(f'[panel_editor] sharpen render failed: {exc!r}')

    def _step_zoom(self, direction):
        """Move one stop up or down the zoom ladder from wherever we are."""
        current = self.view_dpi
        steps = self.ZOOM_STEPS
        if direction > 0:
            nxt = next((s for s in steps if s > current + 0.5), steps[-1])
        else:
            nxt = next((s for s in reversed(steps) if s < current - 0.5), steps[0])
        self._set_zoom(nxt)

    def _on_wheel(self, event):
        # X11 reports the wheel as buttons 4/5 with no delta; everyone else
        # sends a signed delta on <MouseWheel>.
        if getattr(event, 'num', None) in (4, 5):
            delta = 1 if event.num == 4 else -1
        else:
            delta = 1 if getattr(event, 'delta', 0) > 0 else -1

        if event.state & 0x0004:          # ctrl → zoom
            self._step_zoom(delta)
        elif event.state & 0x0001:        # shift → horizontal
            self.canvas.xview_scroll(-delta, 'units')
        else:
            self.canvas.yview_scroll(-delta, 'units')
        return 'break'

    def _preview_rescale(self):
        """Cheaply stretch the cached backdrop to the current fit size.

        A full Agg redraw of a poster takes ~2 s (tens of thousands of raster
        line segments), so doing it on every window resize froze the editor.
        Instead we rescale the last rendered image with PIL — instant — and
        record the scale so the coordinate maps still place the overlay and
        resolve clicks correctly on the stretched preview. It is softer than a
        true render until the next crisp redraw (any edit, or the Re-render /
        Fit buttons), but the editor never blocks.
        """
        self._preview_to_dpi(self._fit_dpi())

    def _preview_to_dpi(self, target_dpi):
        """Show the cached render scaled to *target_dpi*, without redrawing."""
        if self._backdrop_flat is None or not self._rendered_dpi:
            return
        target, clamped = self._clamp_dpi(target_dpi)
        scale = target / self._rendered_dpi
        w0, h0 = self._backdrop_flat.size
        w, h = max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))
        resample = getattr(self._Image, 'Resampling', self._Image).BILINEAR
        img = (self._backdrop_flat if (w, h) == (w0, h0)
               else self._backdrop_flat.resize((w, h), resample))
        self._view_scale = scale
        self._put_image(img)
        # Keep the readout truthful on the preview too — it is what tells you a
        # reckless zoom was capped, and the crisp render may be moments away.
        self.zoom_lbl.configure(
            text=f'{target:.0f} dpi · {w}x{h} px'
                 + (' (clamped)'
                    if (clamped or self._zoom_was_clamped) else ''))
        self._refresh_overlay()           # coords fold in _view_scale, so valid

    def _schedule_expose(self, _event=None):
        """Coalesce a burst of Expose/Visibility/Map events into one repaint."""
        if self._expose_after is not None:
            self.root.after_cancel(self._expose_after)
        self._expose_after = self.root.after(30, self._run_expose)

    def _run_expose(self):
        self._expose_after = None
        try:
            self._on_expose()
        except Exception as exc:                      # pragma: no cover
            print(f'[panel_editor] repaint failed: {exc!r}')

    def _on_expose(self, _event=None):
        """Re-assert the backdrop when the window comes back into view.

        A tiling WM (i3) unmaps the window on a workspace switch, and a large
        canvas image does not reliably repaint itself on the following Expose —
        the editor comes back blank. Re-creating the image item from the
        PhotoImage we already hold forces the repaint. It costs nothing (no Agg
        render), so it is safe to do on every expose rather than trying to
        detect the bad case.
        """
        if self._rendering or self._photo is None:
            return
        self.canvas.delete('backdrop')
        self.canvas.create_image(0, 0, image=self._photo, anchor='nw',
                                 tags='backdrop')
        self.canvas.tag_lower('backdrop')
        self.canvas.configure(scrollregion=(0, 0, *self._img_size))
        self._refresh_overlay()

    def _on_canvas_resize(self, _event=None):
        """Re-fit the backdrop to the window, in fit mode only.

        Resize shows only the instant PIL-scaled preview — never the ~2 s Agg
        redraw that made the editor feel stuck. The crisp render is left to the
        Re-render / Fit buttons and to the automatic redraw after any edit, so
        the window stays responsive no matter how heavy the figure is.
        """
        if not self._fit_mode or self._rendering:
            return
        self._preview_rescale()
        self._diag('resize')
        self._status('resizing… sharpening')
        # A resize storm (dragging a window edge, or the first map) coalesces
        # into one crisp redraw once it settles.
        self._schedule_sharpen()

    def _pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self._refresh_overlay()

    # ── coordinate mapping ───────────────────────────────────────────────────
    # matplotlib display coords have their origin bottom-left, tk's is top-left;
    # the backdrop is drawn 1:1 at view_dpi so only the y flip is needed.

    def _to_canvas(self, mx, my):
        s = self._view_scale
        return mx * s, self._img_size[1] - my * s

    def _to_mpl(self, cx, cy):
        s = self._view_scale
        return cx / s, (self._img_size[1] - cy) / s

    def _event_xy(self, event):
        """Canvas coords of a mouse event.

        The canvas scrolls, so ``event.x``/``event.y`` are viewport-relative
        and must be translated before they mean anything against the backdrop.
        """
        return self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

    def _item_box(self, item):
        """Item bounding box in canvas coords, or None.

        The display bbox is cached until something moves. Asking every item for
        its bbox costs a renderer text-layout pass each time, which at ~200
        items is milliseconds -- affordable once, but not on every mouse-motion
        event, where it made hover and selection feel like the window had hung.
        """
        serial, boxes = self._box_cache
        if serial != self._geom_serial:
            boxes = {}
            self._box_cache = (self._geom_serial, boxes)
        if id(item) in boxes:
            bbox = boxes[id(item)]
        else:
            bbox = item.bbox_display(self._agg.get_renderer())
            boxes[id(item)] = bbox
        if bbox is None:
            return None
        x0, y0, x1, y1 = _norm_bbox(bbox)
        cx0, cy0 = self._to_canvas(x0, y1)     # top-left
        cx1, cy1 = self._to_canvas(x1, y0)     # bottom-right
        return cx0, cy0, cx1, cy1

    # ── overlay ──────────────────────────────────────────────────────────────

    def _refresh_overlay(self):
        self.canvas.delete('overlay')
        self._overlay_ids = []
        if not self.selection:
            return
        single = len(self.selection) == 1
        for item in self.selection:
            box = self._item_box(item)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            # The primary is drawn solid; the rest of a group dashed, so it is
            # clear which one the numeric fields refer to.
            primary = item is self.selected
            # A whole poster fits the window at ~40 dpi, where a label is a few
            # pixels tall and a hairline outline around it is invisible — the
            # selection reads as "the click did nothing". Pad the box out and
            # draw it thick, over a white halo, so a hit is unmistakable at any
            # zoom. Small items get the most padding, since they need it most.
            pad = 4 if min(x1 - x0, y1 - y0) < 30 else 2
            bx0, by0, bx1, by1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
            self.canvas.create_rectangle(
                bx0, by0, bx1, by1, outline='white', width=5, tags='overlay')
            self.canvas.create_rectangle(
                bx0, by0, bx1, by1, outline=SEL_COLOUR,
                width=2.5, dash=() if (primary and not single) else (4, 3),
                tags='overlay')
            # Resize handles only for a lone axes: a group move never resizes,
            # and handles on every box would say otherwise.
            if single and isinstance(item, _AxesItem):
                for hx, hy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1),
                               ((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1),
                               (x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2)):
                    self.canvas.create_rectangle(
                        hx - HANDLE_SIZE, hy - HANDLE_SIZE,
                        hx + HANDLE_SIZE, hy + HANDLE_SIZE,
                        outline=SEL_COLOUR, fill='white', width=1, tags='overlay')

    def _clear_guides(self):
        for gid in self._guide_ids:
            self.canvas.delete(gid)
        self._guide_ids = []

    def _draw_guides(self, hit_x, hit_y):
        self._clear_guides()
        w, h = self._img_size
        if hit_x is not None:
            cx = hit_x * w
            self._guide_ids.append(self.canvas.create_line(
                cx, 0, cx, h, fill=GUIDE_COLOUR, dash=(2, 2), tags='overlay'))
        if hit_y is not None:
            cy = h - hit_y * h
            self._guide_ids.append(self.canvas.create_line(
                0, cy, w, cy, fill=GUIDE_COLOUR, dash=(2, 2), tags='overlay'))

    # ── element tree ─────────────────────────────────────────────────────────

    def _populate_tree(self):
        self._tree_items = {}
        groups = {}
        for it in self.items:
            groups.setdefault(_group_of(it) or '(unlabelled)', []).append(it)

        for label in sorted(groups, key=lambda s: (s == '(unlabelled)', s)):
            parent = self.tree.insert('', 'end', text=f'panel {label}'
                                      if label != '(unlabelled)' else label,
                                      open=True)
            # the panel axes itself first, then its contents
            members = sorted(groups[label],
                             key=lambda i: 0 if getattr(i, 'role', '') == 'panel' else 1)
            for it in members:
                node = self.tree.insert(parent, 'end', text='  ' + _display_name(it))
                self._tree_items[node] = it

    def _on_tree_select(self, _event=None):
        """Selection made in the element tree — mirror it onto the canvas.

        Deliberately does *not* write the selection back to the tree: it came
        from there. That is what stops the two from driving each other. See
        :meth:`_sync_tree_to_selection` for why a flag cannot do this job.
        """
        items = [self._tree_items[n] for n in self.tree.selection()
                 if n in self._tree_items]
        if not items or items == self.selection:
            return
        self._set_selection(items, sync_tree=False)

    def _sync_tree_to_selection(self):
        """Reflect the current selection in the element tree.

        ``<<TreeviewSelect>>`` is a *virtual* event: tk queues it and delivers
        it from the event loop, after the code that changed the selection has
        long since returned. A flag raised around ``selection_set`` is
        therefore no guard at all — it is always back down by the time the
        handler runs. Clicking the canvas used to sync the tree, the queued
        event bounced back into ``_on_tree_select``, which re-selected and
        synced again: an endless loop that pinned a core and left the window
        unable to respond to anything.

        Two things break it, neither depending on timing: the handler above
        does not sync back, and this skips the call entirely when the tree
        already shows the right rows, so no event is queued in the first place.
        """
        nodes = [node for node, it in self._tree_items.items()
                 if it in self.selection]
        if list(self.tree.selection()) == nodes:
            return
        self.tree.selection_set(nodes)
        if nodes:
            self.tree.see(nodes[-1])

    # ── selection ────────────────────────────────────────────────────────────

    def _select(self, item):
        """Select exactly one item (clearing any group)."""
        self._set_selection([item])

    def _set_selection(self, items, sync_tree=True):
        """Replace the selection with *items* (order preserved, de-duplicated).

        The last item becomes the primary — the one the numeric fields edit
        and the one drawn with a solid outline.
        """
        seen, ordered = set(), []
        for it in items:
            if it is not None and id(it) not in seen:
                seen.add(id(it))
                ordered.append(it)
        self.selection = ordered
        self.selected = ordered[-1] if ordered else None
        self._describe_selection()
        self._sync_fields()
        self._refresh_overlay()
        if sync_tree:
            self._sync_tree_to_selection()

    def _toggle_item(self, item):
        """Add *item* to the selection, or remove it if already there."""
        if item in self.selection:
            rest = [it for it in self.selection if it is not item]
            self._set_selection(rest)
            self._status(f'{len(self.selection)} selected'
                         if self.selection else 'Selection cleared')
        else:
            self._set_selection(self.selection + [item])
            self._status(f'{len(self.selection)} selected')

    def _describe_selection(self):
        n = len(self.selection)
        if n == 0:
            self.addr_lbl.configure(text='(nothing selected)')
        elif n == 1:
            item = self.selected
            addr = getattr(item, 'override_address', None)
            self.addr_lbl.configure(
                text=f'{item.label}\n'
                     f'{addr or "(not addressable — prints a snippet)"}')
        else:
            self.addr_lbl.configure(
                text=f'{n} elements selected\nprimary: {self.selected.label}')

    def _independent_movers(self):
        """Selected items to actually translate on a group move.

        Drop any selected item that lives inside another selected axes: moving
        a panel already carries its inset children and its in-axes text, so
        moving those again would double their shift.
        """
        panel_axes = [it.ax for it in self.selection
                      if isinstance(it, _AxesItem)]
        movers = []
        for it in self.selection:
            ax = getattr(it, 'ax', None)
            contained = False
            for pax in panel_axes:
                if pax is ax and not isinstance(it, _AxesItem):
                    contained = True          # text drawn in a selected axes
                    break
                if ax is not pax and ax in getattr(pax, 'child_axes', []):
                    contained = True          # inset child of a selected axes
                    break
            if not contained:
                movers.append(it)
        return movers

    def _sync_text_editor(self):
        """Show the selected item's string in the text box, or clear it.

        The box binds to a single free-text artist; anything else (a panel, an
        image, a group) empties and disables it.
        """
        item = self.selected
        editable = (item is not None and len(self.selection) == 1
                    and getattr(item, 'is_text', False))
        self.text_edit.delete('1.0', 'end')
        if editable:
            self.text_edit.insert('1.0', item.artist.get_text())
            self.text_edit.configure(state='normal')
            self._text_target = item
        else:
            self.text_edit.configure(state='disabled')
            self._text_target = None

    def _sync_fields(self):
        self._sync_text_editor()
        item = self.selected
        # The numeric fields edit one item. With a group selected they would
        # be ambiguous, so blank and disable them until it is a single pick.
        if item is None or len(self.selection) > 1:
            for name in ('x0', 'y0', 'w', 'h'):
                var, ent = self.fields[name]
                var.set('')
                ent.state(['disabled'])
            self.zoom_field.set('')
            self.zoom_entry.state(['disabled'])
            return
        pos = item.pos()
        names = ('x0', 'y0', 'w', 'h') if len(pos) == 4 else ('x0', 'y0')
        for i, name in enumerate(('x0', 'y0', 'w', 'h')):
            var, ent = self.fields[name]
            if i < len(pos):
                var.set(f'{pos[i]:.4f}')
                ent.state(['!disabled'])
            else:
                var.set('')
                ent.state(['disabled'])
        if getattr(item, 'is_image', False):
            self.zoom_field.set(f'{_overrides.image_zoom(item.artist):.4f}')
            self.zoom_entry.state(['!disabled'])
        else:
            self.zoom_field.set('')
            self.zoom_entry.state(['disabled'])

    def _apply_fields(self, _event=None):
        item = self.selected
        if item is None:
            return
        if len(self.selection) > 1:
            self._status('The position fields edit one element — '
                         'select a single one to type a position.')
            return
        try:
            vals = [float(self.fields[n][0].get())
                    for n in ('x0', 'y0', 'w', 'h')
                    if self.fields[n][0].get().strip()]
        except ValueError:
            self._status('Could not read those numbers.')
            return
        pos = item.pos()
        target = np.array(vals, dtype=float)
        # The fields are shown to four decimals, so "unchanged" means within
        # half of the smallest difference they can express — comparing exactly
        # would treat every Enter on an untouched field as an edit.
        moves = not (len(target) == len(pos)
                     and np.allclose(target, pos, atol=5e-5, rtol=0))

        new_zoom, rescales = None, False
        if getattr(item, 'is_image', False) and self.zoom_field.get().strip():
            try:
                new_zoom = float(self.zoom_field.get())
            except ValueError:
                self._status('Could not read that scale.')
                return
            rescales = abs(new_zoom - _overrides.image_zoom(item.artist)) > 1e-9

        if not (moves or rescales):
            self._status('Those are already its numbers.')
            return

        self._push_history(item)
        if moves:
            # Record on the item too: Undo and Save both read the item's own
            # history, so a typed position that skips this applies to the
            # screen and is then quietly lost.
            item.record()
            if isinstance(item, _AxesItem) and len(vals) == 4:
                item._set_bounds(vals)
            elif len(vals) >= 2:
                from sciplotlib.drag_editor import _set_pos
                _set_pos(item.artist, vals[:2])
        if rescales:
            _overrides.set_image_zoom(item.artist, new_zoom)
            item._zoom_changed = True
        self._after_change('Applied')

    def _apply_text(self, _event=None):
        """Replace the selected text artist's string with the box's contents."""
        item = self._text_target
        if item is None or not getattr(item, 'is_text', False):
            self._status('Select a single text element to edit its wording.')
            return 'break'
        # Text widgets keep a trailing newline; drop it so the string matches
        # what was typed rather than gaining a blank last line each apply.
        new_text = self.text_edit.get('1.0', 'end-1c')
        if new_text == item.artist.get_text():
            self._status('Text unchanged.')
            return 'break'
        self._push_history(item, kind='text')
        item.set_text(new_text)
        self._geom_changed()              # new string, new bounding box
        self._refresh_tree_label(item)
        # Deliberately no reflow: a text-block panel's redraw hook rebuilds its
        # string from the block's source and would wipe the edit out. Just
        # redraw the backdrop so the new words show -- incrementally, since the
        # plate for this text is exactly the figure without it.
        if not self._blit_change([item]):
            self._invalidate_render_cache()  # the words changed; zooms stale
            self._plate_cache.clear()
            self._render_backdrop()
        self._sync_fields()
        self._status('Edited text')
        return 'break'          # a Ctrl+Return must not also insert a newline

    def _refresh_tree_label(self, item):
        """Update one element-tree row after its text changed."""
        for node, it in self._tree_items.items():
            if it is item:
                self.tree.item(node, text='  ' + _display_name(item))
                break

    # ── mouse ────────────────────────────────────────────────────────────────

    #: no artist is harder to grab than this many screen pixels across. A
    #: small arrow can be 4 x 5 px at editor zoom -- selectable in principle,
    #: unhittable in practice, and it usually sits on top of something large
    #: and resizable that swallows the click instead.
    MIN_GRAB_PX = 14

    def _hit_test(self, cx, cy, pad=6):
        """Topmost item under the point. Items are ordered biggest-first, so the
        reversed walk reaches text and images before the panels holding them."""
        for item in reversed(self.items):
            box = self._item_box(item)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            # Inflate a too-small box about its centre, so tiny artists can be
            # clicked. Only tiny ones grow, so this cannot make a big artist
            # steal a click from a small one in front of it.
            grow_x = max(0.0, self.MIN_GRAB_PX - (x1 - x0)) / 2
            grow_y = max(0.0, self.MIN_GRAB_PX - (y1 - y0)) / 2
            x0, x1 = x0 - grow_x, x1 + grow_x
            y0, y1 = y0 - grow_y, y1 + grow_y
            if x0 - pad <= cx <= x1 + pad and y0 - pad <= cy <= y1 + pad:
                return item
        return None

    def _on_hover(self, event):
        ex, ey = self._event_xy(event)
        item = self._hit_test(ex, ey)
        cursor = ''
        if item is not None and isinstance(item, _AxesItem):
            box = self._item_box(item)
            if box:
                x0, y0, x1, y1 = box
                near_x = abs(ex - x0) <= HANDLE_PX or abs(ex - x1) <= HANDLE_PX
                near_y = abs(ey - y0) <= HANDLE_PX or abs(ey - y1) <= HANDLE_PX
                if near_x and near_y:
                    cursor = 'sizing'
                elif near_x:
                    cursor = 'sb_h_double_arrow'
                elif near_y:
                    cursor = 'sb_v_double_arrow'
                else:
                    cursor = 'fleur'
        elif item is not None:
            cursor = 'fleur'
        self.canvas.configure(cursor=cursor)

    @staticmethod
    def _modifier(event):
        """True if ctrl or shift is held — the 'add to selection' modifier."""
        return bool(event.state & 0x0004) or bool(event.state & 0x0001)

    def _on_press(self, event):
        ex, ey = self._event_xy(event)
        item = self._hit_test(ex, ey)
        modifier = self._modifier(event)
        if os.environ.get('SPL_EDITOR_DIAG'):
            print(f'[diag/click] event=({event.x},{event.y}) '
                  f'canvas=({ex:.0f},{ey:.0f}) '
                  f'mpl={tuple(round(v) for v in self._to_mpl(ex, ey))} '
                  f'hit={item.label if item else None}', flush=True)

        # Empty space: begin a rubber-band. A plain click that never moves
        # clears the selection on release; a ctrl/shift-drag adds to it.
        if item is None:
            self._band = (ex, ey)
            self._band_additive = modifier
            return

        # Ctrl/shift on an element toggles it in the selection; no drag starts,
        # so the click cannot also nudge the group.
        if modifier:
            self._toggle_item(item)
            return

        # Plain click on an already-selected member of a group keeps the group
        # and moves all of it; otherwise it selects just this element.
        if item in self.selection and len(self.selection) > 1:
            self.selected = item
            self._describe_selection()
            self._refresh_overlay()
            self._sync_tree_to_selection()
        else:
            self._set_selection([item])
            # Say what was hit. At poster zoom the outline alone is easy to
            # miss, and silence after a click reads as an unresponsive window.
            self._status(f'Selected {_display_name(item)}'
                         + ('  ·  double-click to edit its words'
                            if getattr(item, 'is_text', False) else ''))

        self._begin_drag(ex, ey)

    def _on_double(self, event):
        """Double-click a text element to jump straight into editing its words.

        Handles the press first, exactly as a single click would. Tk delivers
        ``<Double-Button-1>`` *instead of* ``<Button-1>`` for the second click
        of a pair, so a handler that swallows it loses that press entirely:
        clicking an element to select it and then pressing again to drag did
        nothing at all, which reads as an editor that ignores the mouse. Every
        press must still select and arm a drag; opening the text box is added
        on top, and is harmless if the user turns out to be dragging.
        """
        result = self._on_press(event)
        item = self.selected
        if item is not None and getattr(item, 'is_text', False):
            self.text_edit.focus_set()
            self.text_edit.tag_add('sel', '1.0', 'end-1c')
            self._status(f'Editing text: {item.label}. '
                         'Type, then Ctrl+Enter or "Apply text".')
        return result

    def _begin_drag(self, ex, ey):
        """Start dragging every independent mover in the current selection."""
        movers = self._independent_movers()
        if not movers:
            return
        mx, my = self._to_mpl(ex, ey)
        group = len(movers) > 1
        for it in movers:
            it.start(mx, my, _ShimEvent())
            if group and isinstance(it, _AxesItem):
                it._mode = 'move'      # a group drag translates, never resizes
        self._push_history(*movers)
        self._drag = movers

    def _on_drag(self, event):
        if self._band is not None:
            self._update_band(event)
            return
        if not self._drag:
            return
        mx, my = self._to_mpl(*self._event_xy(event))
        shift = bool(event.state & 0x0001)
        ev = _ShimEvent('shift' if shift else None)
        for it in self._drag:
            it.drag(mx, my, ev)
        self._geom_changed()          # the dragged boxes are moving under us
        # Snap only a lone axes: a group has no single edge to align.
        if (self.snap_var.get() and len(self._drag) == 1
                and isinstance(self._drag[0], _AxesItem)):
            dragged = self._drag[0]
            others = [i for i in self.items
                      if isinstance(i, _AxesItem) and i is not dragged]
            hx, hy = snap_axes_item(dragged, others, self.fig)
            self._draw_guides(hx, hy)
        # Move the outline live; the backdrop catches up on release.
        self._refresh_overlay()
        self._sync_fields()

    def _on_release(self, event):
        if self._band is not None:
            self._finish_band(event)
            return
        if not self._drag:
            return
        moved_any = False
        for it in self._drag:
            it.end()
            # A press records a history entry *before* any motion. If the item
            # did not actually move, drop that no-op: a plain click (to select,
            # or to open the text editor) must not count as a change, leave an
            # Undo step, or — the expensive part on a heavy figure — trigger a
            # full ~2 s backdrop re-render. Only a real move pays for a redraw.
            if it._history and np.allclose(it._history[-1], it.pos()):
                it._history.pop()
            elif it._history:
                moved_any = True
        drag = self._drag
        self._drag = None
        self._clear_guides()
        if not moved_any:
            # Undo the editor-level history group pushed for this press, so the
            # no-op click leaves nothing on the Undo stack either.
            if self._history and self._history[-1][1] == drag:
                self._history.pop()
            self._refresh_overlay()
            return
        n = 1 if self.selected is None else len(self._independent_movers())
        self._after_change('Moved' if n <= 1 else f'Moved {n} elements')

    # ── rubber-band selection ────────────────────────────────────────────────

    def _update_band(self, event):
        ex, ey = self._event_xy(event)
        bx, by = self._band
        self.canvas.delete('band')
        self.canvas.create_rectangle(
            bx, by, ex, ey, outline=SEL_COLOUR, width=1.0, dash=(3, 2),
            tags='band')

    def _finish_band(self, event):
        ex, ey = self._event_xy(event)
        bx, by = self._band
        self._band = None
        self.canvas.delete('band')
        # A click that barely moved is a plain click on empty space: clear,
        # unless it was a modifier-click (which leaves the selection alone).
        if abs(ex - bx) < 3 and abs(ey - by) < 3:
            if not self._band_additive:
                self._set_selection([])
                self._status('Selection cleared')
            return
        rx0, rx1 = sorted((bx, ex))
        ry0, ry1 = sorted((by, ey))
        caught = [it for it in self.items if self._box_intersects(
            self._item_box(it), (rx0, ry0, rx1, ry1))]
        base = self.selection if self._band_additive else []
        self._set_selection(base + caught)
        self._status(f'{len(self.selection)} selected'
                     if self.selection else 'Nothing in the lasso')

    @staticmethod
    def _box_intersects(box, rect):
        if box is None:
            return False
        x0, y0, x1, y1 = box
        rx0, ry0, rx1, ry1 = rect
        return not (x1 < rx0 or x0 > rx1 or y1 < ry0 or y0 > ry1)

    def _nudge(self, dx, dy, step=1):
        movers = self._independent_movers()
        if not movers:
            return
        self._push_history(*movers)
        for it in movers:
            it.nudge(dx * step, dy * step)
        tail = '' if len(movers) == 1 else f' ({len(movers)} elements)'
        self._after_change(f'Nudged {dx * step:+d}, {dy * step:+d} px{tail}')

    # ── history / reset / save ───────────────────────────────────────────────

    def _push_history(self, *items, kind='pos'):
        """Record a group of items changed together, so one Undo reverts all.

        *kind* is ``'pos'`` for a move/resize or ``'text'`` for a string edit;
        the two use different undo stacks on the item, so Undo must know which.
        """
        if items:
            self._history.append((kind, list(items)))

    def _undo(self):
        while self._history:
            kind, group = self._history.pop()
            if kind == 'hide':
                # Symmetric: the step recorded which items changed visibility,
                # so undoing is just flipping them back.
                done = list(group)
                for it in done:
                    it.set_hidden(not it.hidden)
                    self._refresh_tree_label(it)
                verb = 'delete' if done and not done[0].hidden else 'restore'
            elif kind == 'text':
                done = [it for it in group if it.undo_text()]
                verb = 'text edit'
            else:
                done = [it for it in group if it._history]
                for it in done:
                    it.undo()
                verb = 'change'
            if done:
                self._set_selection(group)
                self._after_change(
                    f'Undid a {verb} to {done[0].label}' if len(done) == 1
                    else f'Undid a group move of {len(done)} elements')
                return
        self._status('Nothing to undo.')

    def _set_hidden_on_selection(self, hidden):
        """Hide or unhide every selected element, as one undoable step.

        Hidden, not removed: the artist stays in the figure so its address
        keeps resolving, `_save` can write the flag against it, and un-deleting
        restores it exactly. A panel takes its contents with it.
        """
        targets = [it for it in self.selection
                   if getattr(it, 'set_hidden', None) is not None
                   and it.hidden != hidden]
        if not targets:
            self._status('Nothing to delete.' if hidden
                         else 'Nothing to restore.')
            return
        for it in targets:
            it.set_hidden(hidden)
        self._push_history(*targets, kind='hide')
        verb = 'Deleted' if hidden else 'Restored'
        for it in targets:
            self._refresh_tree_label(it)
        self._after_change(
            f'{verb} {targets[0].label}  (Ctrl+Z to undo)' if len(targets) == 1
            else f'{verb} {len(targets)} elements  (Ctrl+Z to undo)')

    def _delete_selected(self):
        self._set_hidden_on_selection(True)

    def _undelete_selected(self):
        self._set_hidden_on_selection(False)

    def _reset_selected(self):
        if not self.selection:
            return
        for it in self.selection:
            it.reset()
        self._after_change(f'Reset {self.selected.label}'
                           if len(self.selection) == 1
                           else f'Reset {len(self.selection)} elements')

    def _reset_all(self):
        for it in self.items:
            if getattr(it, 'set_hidden', None) is not None:
                it.set_hidden(not getattr(it, '_initially_visible', True))
            it.reset()
        self._history.clear()
        self._reflow()          # every panel moved, so refresh every hook
        # Everything moved at once, so no plate describes the new figure.
        self._after_change('Reset every element to its composed position',
                           blit=False)

    def _reflow(self, item=None):
        """Re-run redraw hooks so size-dependent content matches its new box.

        Data artists redraw at whatever size their axes ends up, but content
        laid out *against* the axes size -- wrapped body text, most obviously --
        has to be rebuilt.  Panels that registered a hook with
        :func:`~sciplotlib.compose.set_redraw_hook` get it run here.

        With *item*, only that item's axes is refreshed; without one, every
        panel is (used after Reset all).  Returns the number of hooks run.
        """
        if not self.reflow_var.get():
            return 0
        from sciplotlib.compose import run_redraw_hook

        axes = [getattr(item, 'ax', None)] if item is not None else self.fig.get_axes()
        ran = 0
        self._reflow_error = None
        for ax in axes:
            if ax is None:
                continue
            try:
                ran += bool(run_redraw_hook(ax))
            except Exception as exc:      # a bad hook must not kill the editor
                self._reflow_error = f'redraw hook failed: {exc}'
                print(f'[panel_editor] redraw hook failed on {ax}: {exc!r}')
        if ran:
            self._geom_changed()       # a hook may have re-laid-out its panel
        return ran

    def _reflow_note(self, ran):
        if self._reflow_error:
            return f' · {self._reflow_error}'
        return ' · reflowed' if ran else ''

    def _rerender_all(self):
        """Toolbar 'Re-render': reflow every hooked panel, then redraw."""
        ran = self._reflow()
        self._render_backdrop()
        self._status(f'Re-rendered ({ran} panel(s) reflowed)'
                     + self._reflow_note(0))

    def _after_change(self, msg, blit=True):
        self._geom_changed()
        # Reflow every distinct axes in the selection: a group move can carry
        # several text blocks, each needing its redraw hook re-run.
        reflowed, seen, rebuilt = 0, set(), []
        for it in self.selection:
            ax = getattr(it, 'ax', None)
            if ax is not None and id(ax) not in seen:
                seen.add(id(ax))
                ran = self._reflow(it)
                reflowed += ran
                if ran:
                    rebuilt.append(ax)
        # Redraw only what moved, so long as the plate also hides every axes a
        # redraw hook has just rebuilt.
        if not (blit and self._blit_change(self.selection, must_hide=rebuilt)):
            # The figure moved, so every cached zoom level, and every plate,
            # now describes the old layout.
            self._invalidate_render_cache()
            self._plate_cache.clear()
            self._render_backdrop()
        self._sync_fields()
        self._status(msg + self._reflow_note(reflowed))

    def _save(self):
        moved = [it for it in self.items
                 if it.moved and getattr(it, 'override_address', None)]
        loose = [it for it in self.items
                 if it.moved and not getattr(it, 'override_address', None)]
        if not self.overrides_path:
            self._status('No overrides file was given — nothing written. '
                         'Positions are printed to the terminal on close.')
            return
        data = _overrides.read_overrides(self.overrides_path)   # merge, don't clobber
        for it in moved:
            data[it.override_address] = it.override_entry(it.override_kind)
        _overrides.write_overrides(self.overrides_path, data)
        extra = f'  ({len(loose)} unaddressable — see terminal)' if loose else ''
        self._status(f'Wrote {len(moved)} position(s) to '
                     f'{os.path.basename(str(self.overrides_path))}{extra}')
        print(f'[panel_editor] wrote {len(moved)} position(s) to '
              f'{self.overrides_path}')

    def print_positions(self):
        moved = [it for it in self.items if it.moved]
        if not moved:
            print('(No elements were moved.)')
            return
        addressed = [it for it in moved if getattr(it, 'override_address', None)]
        loose = [it for it in moved if not getattr(it, 'override_address', None)]
        if addressed:
            print('\n# ── Addressable — saved with the Save button')
            for it in addressed:
                vals = [round(float(v), 4) for v in it.pos()]
                extra = ('  delta=' + str([round(float(v), 4) for v in it.delta()])
                         if it.override_kind == 'panel' else '')
                print(f'#   {it.override_address}  ->  {vals}{extra}')
        if loose:
            print('\n# ── Paste these back into your plotting code ' + '─' * 20)
            for it in loose:
                print(it.code_snippet())

    # ── run ──────────────────────────────────────────────────────────────────

    def run(self):
        # No render here. __init__ has already drawn the backdrop, and on a
        # heavy figure a second full draw costs seconds of a frozen window at
        # exactly the moment the user first tries to click — which reads as an
        # editor that does not respond at all. The initial <Configure> refits
        # the cached image instantly instead; the crisp redraw follows from any
        # edit, or from Re-render / a zoom change on demand.
        self.root.mainloop()
        # restore the print dpi so a figure edited in-process still saves right
        self.fig.set_dpi(self._print_dpi)
        self.print_positions()


# ── launcher ─────────────────────────────────────────────────────────────────

def _rc_snapshot():
    """This process's rcParams, minus the ones that must not follow the figure."""
    skip = {'backend', 'backend_fallback', 'interactive'}
    rc = {}
    for key, value in matplotlib.rcParams.items():
        if key in skip:
            continue
        try:
            pickle.dumps(value)
        except Exception:                             # pragma: no cover
            continue                                  # not transportable; skip
        rc[key] = value
    return rc


def _apply_rc(rc):
    """Restore a snapshot from :func:`_rc_snapshot`, key by key.

    One at a time on purpose: a key this matplotlib no longer accepts must
    cost that one setting, not the whole stylesheet.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for key, value in (rc or {}).items():
            try:
                matplotlib.rcParams[key] = value
            except Exception:
                pass


def launch_panel_editor(fig, overrides_path=None, view_dpi='fit', snap=True,
                        fit_min_dpi=None):
    """Open the panel editor on *fig* in a subprocess. Blocks until it closes.

    The subprocess keeps the editor's Agg rendering away from whatever backend
    the caller (marimo, a script, a notebook) has already configured.

    *view_dpi* defaults to ``'fit'``, which sizes the backdrop to the window --
    the only sane default once figures can be A0.
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pkl_path = f.name
        # The rcParams travel with the figure. Text artists store *generic*
        # family names ('sans-serif'), resolved against rcParams at draw time,
        # and a composer applies its stylesheet by mutating rcParams globally
        # in this process. Without them the child falls back to matplotlib's
        # defaults and lays every string out in DejaVu Sans -- ~18% wider than
        # a Helvetica-like paper font -- so text overflowed its panel on
        # screen while the PDF was fine, and the editor lied about the layout.
        pickle.dump({'figure': fig, 'rcParams': _rc_snapshot()}, f)

    env = dict(os.environ)
    env['MPLBACKEND'] = 'Agg'          # the editor renders, tk does the window

    # The child resolves `sciplotlib` through its own sys.path, which may be a
    # different install from the one that composed this figure. Pin it.
    import sciplotlib as _spl
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(_spl.__file__)))
    env['PYTHONPATH'] = os.pathsep.join(
        [pkg_root] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))

    if not (env.get('DISPLAY') or env.get('WAYLAND_DISPLAY')
            or sys.platform in ('darwin', 'win32')):
        print('[panel_editor] No display detected ($DISPLAY unset) — a window '
              'cannot open on a headless host.')

    cmd = [sys.executable, '-m', 'sciplotlib.panel_editor', pkl_path,
           '--view-dpi', str(view_dpi)]
    if fit_min_dpi is not None:
        cmd += ['--fit-min-dpi', str(fit_min_dpi)]
    if overrides_path is not None:
        cmd += ['--overrides', str(overrides_path)]
    if not snap:
        cmd += ['--no-snap']
    print(f'[panel_editor] opening editor (sciplotlib={pkg_root})')
    try:
        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode not in (0, None):
            # Tk aborts outright on some uv-managed CPythons (it is a hard
            # SIGABRT inside the Tcl library, not a catchable exception), so the
            # failure can only be seen from out here.
            print(f'\n[panel_editor] the editor exited with code '
                  f'{result.returncode}. If that was a Tk crash, this Python\'s '
                  f'tkinter is broken — use the matplotlib editor instead:\n'
                  f"    composer.launch_editor(..., editor='mpl')")
    finally:
        try:
            os.unlink(pkl_path)
        except OSError:
            pass


def _cli():
    import argparse
    parser = argparse.ArgumentParser(
        prog='python -m sciplotlib.panel_editor',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pkl', help='Path to a pickled matplotlib Figure')
    parser.add_argument('--overrides', default=None,
                        help='Overrides JSON that Save writes into')
    parser.add_argument('--view-dpi', default='fit',
                        help="Backdrop render dpi, or 'fit' (default) to size "
                             'it to the window')
    parser.add_argument('--fit-min-dpi', type=float, default=None,
                        help='Lowest dpi Fit may use before it stops shrinking '
                             'and lets the canvas scroll (default '
                             f'{PanelEditor.MIN_FIT_DPI:g})')
    parser.add_argument('--no-snap', action='store_true')
    args = parser.parse_args()

    with open(args.pkl, 'rb') as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and 'figure' in payload:
        _apply_rc(payload.get('rcParams'))
        fig = payload['figure']
    else:
        # A figure pickled by hand — the manual workflow in the module
        # docstring. It carries no rcParams, so fonts resolve to whatever this
        # interpreter defaults to.
        fig = payload
    matplotlib.use('Agg', force=True)

    editor = PanelEditor(fig, overrides_path=args.overrides,
                         view_dpi=args.view_dpi, snap=not args.no_snap,
                         fit_min_dpi=args.fit_min_dpi)
    print(f'[panel_editor] running {__file__}')
    editor.run()


if __name__ == '__main__':
    _cli()
