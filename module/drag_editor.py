"""
Interactive drag-and-drop position editor for matplotlib artists.

Supported artist types
----------------------
- Text / Annotation (respects transAxes, transData, transFigure)
- Axis labels (x/y label) — emit ``set_label_coords`` so the move survives redraws
- AnnotationBbox  (SVG/image overlays created with place_image()) — move, and
  scroll / ``+`` / ``-`` to rescale
- Axes — **panel axes**, colorbars and insets alike.  Grab the interior to move,
  grab within a few pixels of an edge or corner to **resize**.
- Rectangle and other Patch subclasses

Mouse / keyboard
----------------
======================  ====================================================
drag interior           move the artist under the cursor
drag edge / corner      resize (Axes only); hold **shift** to keep the aspect
arrow keys              nudge the selection 1 px (**shift** = 10 px)
scroll / ``+`` / ``-``  rescale the selected image overlay
``r``                   reset the selection to where it started
``u``                   undo the last move
``p``                   print current coordinates
close window            save overrides / print paste-back snippets
======================  ====================================================

Recommended entry point (FigureComposer)
----------------------------------------
    composer.launch_editor(overrides_path='figure.overrides.json')

opens the window; on close, every move of an *addressable* artist (panel axes,
axis labels, colorbars, insets, ``place_image`` overlays and ``ax.text``) is
written to the overrides JSON keyed by a stable auto-address with no manual
tagging, and ``composer.apply_overrides(path)`` re-applies them on every render.
Artists outside a labelled panel print paste-back snippets instead.

Panel axes are stored as a **delta** against the position
``fit_axes_to_cells`` computed, so the tweak still means the right thing after
the grid or figure size changes.  See :mod:`sciplotlib.overrides`.

Requires an interactive backend + a display: under marimo/uv install Qt
(``uv add pyside6``); ``launch_editor`` auto-selects it and warns otherwise.
The figure is shown at ``screen_dpi`` (default 100), not its print dpi.

Manual workflow (any figure, no composer)
-----------------------------------------
1. Pickle the figure::

       import pickle
       with open('/tmp/fig.pkl', 'wb') as f:
           pickle.dump(fig, f)

2. Run the editor (add ``--overrides path.json`` to save addressable moves)::

       uv run python -m sciplotlib.drag_editor /tmp/fig.pkl

3. Drag artists, then close the window.
4. Paste the printed ``set_position`` / ``set_label_coords`` / ``.xy`` /
   ``set_xy`` calls into your notebook.

Importing as a library
----------------------
    from sciplotlib.drag_editor import PositionEditor
    editor = PositionEditor(fig)
    editor.run()
"""

from __future__ import annotations

import sys
import pickle
import numpy as np

from matplotlib.text import Text
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.patches import Patch, Rectangle
import matplotlib.pyplot as plt

from sciplotlib import overrides as _overrides


_CONTROLS = """\
  ┌─ drag editor ──────────────────────────────────────────────────────────┐
  │  This is a plain matplotlib window — the controls are the mouse and    │
  │  keyboard, there are no on-screen buttons.                             │
  │                                                                        │
  │    drag inside a panel .... move it                                    │
  │    drag near its edge ..... resize it   (shift = keep aspect)          │
  │    click a label / image .. move just that artist                      │
  │    arrow keys ............. nudge the selection 1 px (shift = 10 px)   │
  │    scroll, + / - .......... rescale a selected image                   │
  │    r / u / p .............. reset selection / undo / print positions   │
  │                                                                        │
  │  A dashed blue outline marks the selection. CLOSE the window to save.  │
  └────────────────────────────────────────────────────────────────────────┘"""

# Grab within this many pixels of an Axes edge to resize rather than move.
HANDLE_PX = 8.0
# Snap a dragged Axes edge to a neighbour's edge within this many pixels.
SNAP_PX = 6.0
# Smallest Axes size (figure fraction) a resize may produce.
MIN_SIZE = 0.01


# ── coordinate helpers ────────────────────────────────────────────────────────

def _text_transform(text: Text, ax):
    """Return the coordinate transform that *text* uses for its position."""
    return text.get_transform()


def _annotation_bbox_transform(ab: AnnotationBbox, ax):
    """Return the coordinate transform for an AnnotationBbox's *xy* anchor."""
    coords = getattr(ab, 'xycoords', 'data')
    if coords == 'data':
        return ax.transData
    if coords == 'axes fraction':
        return ax.transAxes
    if coords == 'figure fraction':
        return ax.figure.transFigure
    # xycoords can also be a Transform object
    if callable(getattr(coords, 'transform', None)):
        return coords
    return ax.transData


def _patch_transform(patch: Patch, ax):
    return ax.transData  # patch coordinates are always in data space


def _get_transform(artist, ax):
    if isinstance(artist, Text):
        return _text_transform(artist, ax)
    if isinstance(artist, AnnotationBbox):
        return _annotation_bbox_transform(artist, ax)
    if isinstance(artist, Patch):
        return _patch_transform(artist, ax)
    raise TypeError(f"Unsupported artist: {type(artist)}")


def _transform_name(artist, ax, transform=None):
    t = transform if transform is not None else _get_transform(artist, ax)
    if t is ax.transAxes:
        return 'axes fraction'
    if t is ax.transData:
        return 'data'
    if t is ax.figure.transFigure:
        return 'figure fraction'
    return 'custom'


def _panel_label(ax):
    """Best-effort panel label for *ax* (set by FigureComposer). None otherwise."""
    return getattr(ax, '_sciplotlib_panel', None)


# ── native position get/set ───────────────────────────────────────────────────

def _get_pos(artist) -> np.ndarray:
    if isinstance(artist, Text):
        return np.array(artist.get_position(), dtype=float)
    if isinstance(artist, AnnotationBbox):
        return np.array(artist.xy, dtype=float)
    if isinstance(artist, Patch):
        return np.array(artist.get_xy(), dtype=float)
    raise TypeError(f"Unsupported artist: {type(artist)}")


def _set_pos(artist, pos):
    if isinstance(artist, Text):
        artist.set_position(tuple(pos))
    elif isinstance(artist, AnnotationBbox):
        artist.xy = tuple(pos)
    elif isinstance(artist, Patch):
        artist.set_xy(tuple(pos))


# ── hit detection ─────────────────────────────────────────────────────────────

def _renderer_for(fig):
    """A renderer for *fig*, whichever matplotlib version/backend is in play."""
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        try:
            return fig._get_renderer()
        except Exception:
            return None


def _get_window_extent(artist, renderer):
    try:
        return artist.get_window_extent(renderer)
    except Exception:
        return None


def _norm_bbox(bbox):
    """(x0, y0, x1, y1) with x0<x1, y0<y1 (AnnotationBbox may be inverted)."""
    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _hits(artist, event, renderer, pad: float = 6.0) -> bool:
    """Return True if the mouse event is within the artist's bounding box."""
    bbox = _get_window_extent(artist, renderer)
    if bbox is None:
        return False
    x0, y0, x1, y1 = _norm_bbox(bbox)
    ex, ey = event.x, event.y
    return (x0 - pad) <= ex <= (x1 + pad) and (y0 - pad) <= ey <= (y1 + pad)


def _edge_zone(bbox, ex, ey, pad=HANDLE_PX):
    """Which edges of *bbox* the point is near: a string of 'l','r','b','t'."""
    x0, y0, x1, y1 = _norm_bbox(bbox)
    zone = ''
    if abs(ex - x0) <= pad:
        zone += 'l'
    elif abs(ex - x1) <= pad:
        zone += 'r'
    if abs(ey - y0) <= pad:
        zone += 'b'
    elif abs(ey - y1) <= pad:
        zone += 't'
    return zone


# ── artist description ────────────────────────────────────────────────────────

def _artist_label(artist, role='text') -> str:
    if role == 'xlabel':
        return f'x-axis label "{artist.get_text()}"'
    if role == 'ylabel':
        return f'y-axis label "{artist.get_text()}"'
    if isinstance(artist, Text):
        txt = artist.get_text().replace('\n', '\\n')
        return f'Text("{txt}")'
    if isinstance(artist, AnnotationBbox):
        return 'Image overlay'
    if isinstance(artist, Rectangle):
        return f'Rectangle'
    if isinstance(artist, Patch):
        return type(artist).__name__
    return type(artist).__name__


# ── draggable wrapper (text / annotation / patch / axis label / image) ─────────

class _Item:
    """Draggable wrapper for a point-positioned artist.

    ``role`` is ``'text'`` for ordinary artists, or ``'xlabel'`` / ``'ylabel'``
    for axis labels — those are pinned with ``set_label_coords`` on first grab so
    the move survives matplotlib's automatic re-placement, and their code snippet
    emits ``set_label_coords`` rather than ``set_position``.

    Image overlays (``AnnotationBbox``) additionally support rescaling, which is
    tracked separately from position so a pure zoom still counts as "changed".
    """

    resizable = False

    def __init__(self, artist, ax, role='text'):
        self.artist = artist
        self.ax = ax
        self.role = role
        self.label = _artist_label(artist, role)

        self._explicit_transform = None
        self._pinned = False
        self._history: list[np.ndarray] = []  # positions before each move
        self._press_display: np.ndarray | None = None
        self._press_native: np.ndarray | None = None
        self._press_display_of_native: np.ndarray | None = None

        self._initial_pos = _get_pos(artist).copy()
        self.is_image = isinstance(artist, AnnotationBbox)
        self._initial_zoom = (_overrides.image_zoom(artist)
                              if self.is_image else None)
        self._zoom_changed = False

    @property
    def moved(self) -> bool:
        return bool(self._history) or self._zoom_changed

    def _tf(self):
        if self._explicit_transform is not None:
            return self._explicit_transform
        return _get_transform(self.artist, self.ax)

    def _pin_axis_label(self):
        """Freeze an axis label at its current spot in axes-fraction coords so it
        stops being auto-positioned (and drags therefore persist)."""
        try:
            cur_disp = self.artist.get_transform().transform(self.artist.get_position())
            fx, fy = self.ax.transAxes.inverted().transform(cur_disp)
            axis = self.ax.xaxis if self.role == 'xlabel' else self.ax.yaxis
            axis.set_label_coords(float(fx), float(fy))
            self._explicit_transform = self.ax.transAxes
            self._initial_pos = np.array([fx, fy], dtype=float)
        except Exception:
            pass
        self._pinned = True

    def start(self, ex: float, ey: float, event=None):
        if self.role in ('xlabel', 'ylabel') and not self._pinned:
            self._pin_axis_label()
        native = _get_pos(self.artist)
        t = self._tf()
        self._press_display = np.array([ex, ey], dtype=float)
        self._press_native = native.copy()
        self._press_display_of_native = t.transform(native)
        self._history.append(native.copy())

    def drag(self, ex: float, ey: float, event=None):
        if self._press_display is None:
            return
        delta = np.array([ex, ey]) - self._press_display
        new_display = self._press_display_of_native + delta
        t = self._tf()
        new_native = t.inverted().transform(new_display)
        _set_pos(self.artist, new_native)

    def nudge(self, dx_px: float, dy_px: float):
        """Shift by a display-pixel offset (arrow keys)."""
        if self.role in ('xlabel', 'ylabel') and not self._pinned:
            self._pin_axis_label()
        t = self._tf()
        native = _get_pos(self.artist)
        self._history.append(native.copy())
        disp = t.transform(native) + np.array([dx_px, dy_px])
        _set_pos(self.artist, t.inverted().transform(disp))

    def scale(self, factor: float):
        """Rescale an image overlay about its anchor point."""
        if not self.is_image:
            return False
        z = _overrides.image_zoom(self.artist)
        if z is None:
            return False
        _overrides.set_image_zoom(self.artist, max(z * factor, 1e-4))
        self._zoom_changed = True
        return True

    def end(self):
        self._press_display = None
        self._press_native = None
        self._press_display_of_native = None

    def undo(self):
        if self._history:
            _set_pos(self.artist, self._history.pop())

    def reset(self):
        _set_pos(self.artist, self._initial_pos)
        if self.is_image and self._initial_zoom is not None:
            _overrides.set_image_zoom(self.artist, self._initial_zoom)
        self._history.clear()
        self._zoom_changed = False

    def pos(self) -> np.ndarray:
        return _get_pos(self.artist)

    def bbox_display(self, renderer):
        return _get_window_extent(self.artist, renderer)

    def coord_system(self) -> str:
        return _transform_name(self.artist, self.ax, self._explicit_transform)

    def override_entry(self, kind) -> dict:
        entry = {'kind': kind,
                 'value': [float(v) for v in self.pos()],
                 'fingerprint': getattr(self, 'override_fingerprint', None)}
        if kind == 'image':
            z = _overrides.image_zoom(self.artist)
            if z is not None:
                entry['zoom'] = z
        return entry

    def code_snippet(self) -> str:
        x, y = self.pos()
        a = self.artist
        cs = self.coord_system()
        if self.role == 'xlabel':
            return (f'# {self.label}\n'
                    f'ax.xaxis.set_label_coords({x:.4f}, {y:.4f})')
        if self.role == 'ylabel':
            return (f'# {self.label}\n'
                    f'ax.yaxis.set_label_coords({x:.4f}, {y:.4f})')
        if isinstance(a, Text):
            return (
                f'# {self.label}  [{cs}]\n'
                f'.set_position(({x:.4f}, {y:.4f}))'
            )
        if isinstance(a, AnnotationBbox):
            zoom = _overrides.image_zoom(a)
            lines = [f'# {self.label}  [{cs}]', f'.xy = ({x:.4f}, {y:.4f})']
            if self._zoom_changed and zoom is not None:
                lines.append(f'.get_children()[0].set_zoom({zoom:.4f})')
            return '\n'.join(lines)
        if isinstance(a, Patch):
            return (
                f'# {self.label}  [{cs}]\n'
                f'.set_xy(({x:.4f}, {y:.4f}))'
            )
        return f'# {self.label}: ({x:.4f}, {y:.4f})'


# ── draggable wrapper (Axes: panels, colorbars, inset axes) ───────────────────

class _AxesItem:
    """Draggable / resizable wrapper for an Axes.

    Grabbing the interior translates the position box; grabbing within
    ``HANDLE_PX`` of an edge or corner resizes it.  Positions are in
    figure-fraction coordinates.

    ``role='panel'`` marks a composer panel axes.  Those are recorded as a
    *delta* against ``_initial_bounds`` (the position ``fit_axes_to_cells``
    produced), so the saved tweak survives a change of grid or figure size.
    """

    resizable = True

    def __init__(self, ax, parent_label=None, role='axes'):
        self.artist = ax
        self.ax = ax
        self.role = role
        self.parent_label = parent_label
        if role == 'panel':
            self.label = f"Panel '{parent_label}'" if parent_label else 'Panel axes'
        else:
            loc = f" in panel '{parent_label}'" if parent_label else ''
            kind = ('colorbar' if getattr(ax, '_colorbar', None) is not None
                    else 'inset')
            self.label = f'Axes ({kind}){loc}'

        self._history: list[np.ndarray] = []
        self._press_display: np.ndarray | None = None
        self._press_bounds: np.ndarray | None = None
        self._mode = 'move'          # 'move' or an edge zone like 'lb'
        # For a panel that already carries an override, the baseline is the
        # position *before* that override was applied (stamped by
        # overrides._apply_panel) — so a further nudge reports the total offset
        # from the fitted layout rather than an increment on the previous one.
        base = getattr(ax, '_sciplotlib_panel_base', None) if role == 'panel' else None
        self._initial_bounds = np.array(
            base if base is not None else ax.get_position().bounds, dtype=float)

    @property
    def moved(self) -> bool:
        return len(self._history) > 0

    # -- internals ----------------------------------------------------------

    def _detach_locator(self):
        # Inset axes (e.g. colorbars made with ax.inset_axes) carry a locator
        # that re-pins them to the parent on every redraw, silently overriding
        # set_position. Clear it so the drag actually moves (and sticks).
        if self.ax.get_axes_locator() is not None:
            self.ax.set_axes_locator(None)

    def _set_bounds(self, bounds, move_children=True):
        """Apply *bounds*, translating un-pinned children by the same offset."""
        x0, y0, w, h = bounds
        w = max(float(w), MIN_SIZE)
        h = max(float(h), MIN_SIZE)
        old = self.ax.get_position()
        dx, dy = float(x0) - old.x0, float(y0) - old.y0
        self.ax.set_position([float(x0), float(y0), w, h])
        if move_children and (dx or dy):
            for child in getattr(self.ax, 'child_axes', []):
                if child.get_axes_locator() is not None:
                    continue  # still pinned to the parent, follows on its own
                cpos = child.get_position()
                child.set_position([cpos.x0 + dx, cpos.y0 + dy,
                                    cpos.width, cpos.height])

    # -- interaction --------------------------------------------------------

    def start(self, ex: float, ey: float, event=None):
        self._detach_locator()
        bbox = _get_window_extent(self.ax, _renderer_for(self.ax.figure))
        self._mode = (_edge_zone(bbox, ex, ey) or 'move') if bbox else 'move'
        self._press_display = np.array([ex, ey], dtype=float)
        self._press_bounds = np.array(self.ax.get_position().bounds, dtype=float)
        self._history.append(self._press_bounds.copy())

    def drag(self, ex: float, ey: float, event=None):
        if self._press_display is None:
            return
        fig = self.ax.figure
        dx = (ex - self._press_display[0]) / fig.bbox.width
        dy = (ey - self._press_display[1]) / fig.bbox.height
        x0, y0, w, h = self._press_bounds

        if self._mode == 'move':
            self._set_bounds([x0 + dx, y0 + dy, w, h])
            return

        # Resize: each grabbed edge moves, the opposite edge stays put.
        nx0, ny0, nw, nh = x0, y0, w, h
        if 'l' in self._mode:
            nx0, nw = x0 + dx, w - dx
        elif 'r' in self._mode:
            nw = w + dx
        if 'b' in self._mode:
            ny0, nh = y0 + dy, h - dy
        elif 't' in self._mode:
            nh = h + dy

        if event is not None and getattr(event, 'key', None) == 'shift' \
                and w > 0 and h > 0:
            # Lock the aspect ratio to the larger relative change.
            sx, sy = nw / w, nh / h
            s = sx if abs(sx - 1) > abs(sy - 1) else sy
            nw, nh = w * s, h * s
            if 'l' in self._mode:
                nx0 = x0 + w - nw
            if 'b' in self._mode:
                ny0 = y0 + h - nh

        self._set_bounds([nx0, ny0, nw, nh], move_children=False)

    def nudge(self, dx_px: float, dy_px: float):
        self._detach_locator()
        fig = self.ax.figure
        bounds = np.array(self.ax.get_position().bounds, dtype=float)
        self._history.append(bounds.copy())
        x0, y0, w, h = bounds
        self._set_bounds([x0 + dx_px / fig.bbox.width,
                          y0 + dy_px / fig.bbox.height, w, h])

    def scale(self, factor: float):
        return False

    def end(self):
        self._press_display = None
        self._press_bounds = None
        self._mode = 'move'

    def undo(self):
        if self._history:
            self._set_bounds(self._history.pop())

    def reset(self):
        self._set_bounds(self._initial_bounds)
        self._history.clear()

    def pos(self) -> np.ndarray:
        return np.array(self.ax.get_position().bounds, dtype=float)

    def bbox_display(self, renderer):
        return _get_window_extent(self.ax, renderer)

    def coord_system(self) -> str:
        return 'figure fraction'

    def delta(self) -> np.ndarray:
        return self.pos() - self._initial_bounds

    def override_entry(self, kind) -> dict:
        entry = {'kind': kind, 'value': [float(v) for v in self.pos()],
                 'fingerprint': None}
        if kind == 'panel':
            entry['delta'] = [float(v) for v in self.delta()]
        return entry

    def code_snippet(self) -> str:
        x0, y0, w, h = self.pos()
        return (f'# {self.label}  [figure fraction; x0, y0, w, h]\n'
                f'.set_position([{x0:.4f}, {y0:.4f}, {w:.4f}, {h:.4f}])')



# ── item collection (shared by both editors) ─────────────────────────────────

def collect_items(fig, patch_types=(Rectangle,), include_axes=True,
                  include_panels=True, include_axis_labels=True):
    """Wrap every editable artist in *fig* as a draggable item.

    Returned in hit-test order for a ``reversed()`` walk: panels first (biggest),
    then colorbars/insets, then text/images/patches last — so a small artist
    inside a panel wins the click over the panel box containing it.

    Shared by the matplotlib-event editor and the tk panel editor so both agree
    on what is editable and on what a move means.
    """
    patch_types = patch_types or ()
    seen: set[int] = set()
    panel_items: list = []
    child_items: list = []
    leaf_items: list = []

    def _collect_ax(ax):
        if id(ax) in seen:
            return
        seen.add(id(ax))
        for t in ax.texts:
            leaf_items.append(_Item(t, ax))
        for a in ax.artists:
            if isinstance(a, AnnotationBbox):
                leaf_items.append(_Item(a, ax))
        for p in ax.patches:
            if patch_types and isinstance(p, patch_types):
                leaf_items.append(_Item(p, ax))
        if include_axis_labels:
            if ax.xaxis.label.get_text():
                leaf_items.append(_Item(ax.xaxis.label, ax, role='xlabel'))
            if ax.yaxis.label.get_text():
                leaf_items.append(_Item(ax.yaxis.label, ax, role='ylabel'))
        for child in getattr(ax, 'child_axes', []):
            if include_axes:
                child_items.append(_AxesItem(child, parent_label=_panel_label(ax)))
            _collect_ax(child)

    for ax in fig.get_axes():
        _collect_ax(ax)
        label = _panel_label(ax)
        if include_panels and label:
            panel_items.append(_AxesItem(ax, parent_label=label, role='panel'))
        elif include_axes and getattr(ax, '_colorbar', None) is not None:
            child_items.append(_AxesItem(ax, parent_label=label))

    items = panel_items + child_items + leaf_items
    tag_overridable(fig, items)
    return items


def tag_overridable(fig, items):
    """Attach stable auto-addresses (see :mod:`sciplotlib.overrides`) to *items*."""
    amap = _overrides.address_map(fig)
    for it in items:
        addr, kind, fp = amap.get(id(it.artist), (None, None, None))
        it.override_address = addr
        it.override_kind = kind
        it.override_fingerprint = fp
    return items


def snap_axes_item(item, others, fig, snap_px=SNAP_PX):
    """Snap *item*'s edges onto any of *others*' edges. Returns (x_hit, y_hit)
    in figure-fraction coords for drawing guides, either may be None."""
    tol_x = snap_px / fig.bbox.width
    tol_y = snap_px / fig.bbox.height
    xs, ys = [], []
    for o in others:
        b = o.pos()
        xs += [b[0], b[0] + b[2]]
        ys += [b[1], b[1] + b[3]]
    if not xs:
        return None, None

    x0, y0, w, h = item.pos()
    nx0, ny0 = x0, y0
    hit_x = hit_y = None
    for edge, cur in (('x0', x0), ('x1', x0 + w)):
        for cand in xs:
            if abs(cur - cand) <= tol_x:
                nx0 = cand if edge == 'x0' else cand - w
                hit_x = cand
                break
        if hit_x is not None:
            break
    for edge, cur in (('y0', y0), ('y1', y0 + h)):
        for cand in ys:
            if abs(cur - cand) <= tol_y:
                ny0 = cand if edge == 'y0' else cand - h
                hit_y = cand
                break
        if hit_y is not None:
            break
    if hit_x is not None or hit_y is not None:
        item._set_bounds([nx0, ny0, w, h])
    return hit_x, hit_y


# ── editor ────────────────────────────────────────────────────────────────────

class PositionEditor:
    """Interactive drag-and-drop position editor for a matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    patch_types : tuple of type, optional
        Patch subclasses to make draggable.  Defaults to ``(Rectangle,)``.
        Pass ``None`` to disable patch dragging entirely, or a tuple of
        types (e.g. ``(Rectangle, FancyArrow)``) to extend coverage.
    include_axes : bool, optional
        Make inset/colorbar Axes draggable. Default True.
    include_panels : bool, optional
        Make composer panel axes (those carrying ``_sciplotlib_panel``)
        draggable and resizable. Default True.  Their moves are saved as deltas
        against the fitted layout and re-applied after ``fit_axes_to_cells``.
    include_axis_labels : bool, optional
        Make x/y axis labels draggable. Default True.
    snap : bool, optional
        Snap dragged Axes edges to neighbouring panel edges. Default True.
    """

    def __init__(self, fig, patch_types=(Rectangle,), include_axes=True,
                 include_axis_labels=True, overrides_path=None, screen_dpi=100,
                 include_panels=True, snap=True):
        self.fig = fig
        self._patch_types = patch_types or ()
        self._include_axes = include_axes
        self._include_panels = include_panels
        self._include_axis_labels = include_axis_labels
        self._snap = snap
        self.overrides_path = overrides_path
        self._screen_dpi = screen_dpi
        self._items: list = []
        self._active = None
        self._selected = None
        self._renderer = None
        self._highlight = None
        self._guides: list = []
        self._ensure_managed()
        self._collect()
        self._tag_overridable()
        self._connect()
        self._set_title('Click to select  |  drag edges to resize  |  '
                        'arrows nudge  |  r reset  |  u undo  |  p print')

    # ── collection ────────────────────────────────────────────────────────────

    def _collect(self):
        self._items = collect_items(
            self.fig, patch_types=self._patch_types,
            include_axes=self._include_axes,
            include_panels=self._include_panels,
            include_axis_labels=self._include_axis_labels)

    def _tag_overridable(self):
        tag_overridable(self.fig, self._items)

    def _write_overrides(self):
        if not self.overrides_path:
            return
        moved = [it for it in self._items
                 if it.moved and getattr(it, 'override_address', None)]
        if not moved:
            return
        data = _overrides.read_overrides(self.overrides_path)  # merge, don't clobber
        for it in moved:
            data[it.override_address] = it.override_entry(it.override_kind)
        _overrides.write_overrides(self.overrides_path, data)
        print(f'[overrides] wrote {len(moved)} position(s) to {self.overrides_path}')

    # ── event plumbing ────────────────────────────────────────────────────────

    def _connect(self):
        c = self.fig.canvas
        self._cids = [
            c.mpl_connect('button_press_event',   self._on_press),
            c.mpl_connect('button_release_event', self._on_release),
            c.mpl_connect('motion_notify_event',  self._on_motion),
            c.mpl_connect('key_press_event',       self._on_key),
            c.mpl_connect('scroll_event',          self._on_scroll),
            c.mpl_connect('close_event',           self._on_close),
        ]

    def _disconnect(self):
        for cid in self._cids:
            self.fig.canvas.mpl_disconnect(cid)

    # ── renderer cache ────────────────────────────────────────────────────────

    def _get_renderer(self):
        if self._renderer is None:
            self.fig.canvas.draw()
            try:
                self._renderer = self.fig.canvas.get_renderer()
            except AttributeError:
                self._renderer = self.fig._get_renderer()
        return self._renderer

    def _invalidate_renderer(self):
        self._renderer = None

    # ── selection highlight + snap guides ─────────────────────────────────────

    def _clear_guides(self):
        for g in self._guides:
            try:
                g.remove()
            except Exception:
                pass
        self._guides = []

    def _draw_highlight(self):
        """Outline the selection so an accidental grab is visible."""
        if self._highlight is not None:
            try:
                self._highlight.remove()
            except Exception:
                pass
            self._highlight = None
        if self._selected is None:
            return
        bbox = self._selected.bbox_display(self._get_renderer())
        if bbox is None:
            return
        x0, y0, x1, y1 = _norm_bbox(bbox)
        inv = self.fig.transFigure.inverted()
        (fx0, fy0), (fx1, fy1) = inv.transform([(x0, y0), (x1, y1)])
        self._highlight = Rectangle(
            (fx0, fy0), fx1 - fx0, fy1 - fy0, transform=self.fig.transFigure,
            fill=False, ec='#2e7bf6', lw=1.2, ls=(0, (4, 3)), zorder=10_000)
        self.fig.add_artist(self._highlight)

    def _snap_axes(self, item):
        """Nudge *item*'s edges onto nearby panel edges; draw guides for hits."""
        if not self._snap or not isinstance(item, _AxesItem):
            return
        self._clear_guides()
        fig = self.fig
        tol_x = SNAP_PX / fig.bbox.width
        tol_y = SNAP_PX / fig.bbox.height

        others = [it for it in self._items
                  if isinstance(it, _AxesItem) and it is not item]
        xs, ys = [], []
        for o in others:
            b = o.pos()
            xs += [b[0], b[0] + b[2]]
            ys += [b[1], b[1] + b[3]]
        if not xs:
            return

        x0, y0, w, h = item.pos()
        nx0, ny0 = x0, y0
        hit_x = hit_y = None
        for edge, cur in (('x0', x0), ('x1', x0 + w)):
            for cand in xs:
                if abs(cur - cand) <= tol_x:
                    nx0 = cand if edge == 'x0' else cand - w
                    hit_x = cand
                    break
            if hit_x is not None:
                break
        for edge, cur in (('y0', y0), ('y1', y0 + h)):
            for cand in ys:
                if abs(cur - cand) <= tol_y:
                    ny0 = cand if edge == 'y0' else cand - h
                    hit_y = cand
                    break
            if hit_y is not None:
                break

        if hit_x is None and hit_y is None:
            return
        item._set_bounds([nx0, ny0, w, h])
        for val, vertical in ((hit_x, True), (hit_y, False)):
            if val is None:
                continue
            xdata = [val, val] if vertical else [0, 1]
            ydata = [0, 1] if vertical else [val, val]
            ln = Line2D(xdata, ydata, transform=fig.transFigure,
                        color='#f6a02e', lw=0.8, ls=':', zorder=10_001)
            fig.add_artist(ln)
            self._guides.append(ln)

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.button != 1 or event.x is None:
            return
        renderer = self._get_renderer()
        for item in reversed(self._items):  # reversed → topmost first
            if _hits(item.artist, event, renderer):
                self._active = item
                self._selected = item
                item.start(event.x, event.y, event)
                mode = getattr(item, '_mode', 'move')
                verb = 'Resizing' if mode != 'move' else 'Dragging'
                self._set_title(f'{verb} {item.label}  [{item.coord_system()}]  '
                                f'|  r = reset  |  u = undo')
                self._draw_highlight()
                self.fig.canvas.draw_idle()
                return
        # Clicked empty space: drop the selection.
        self._selected = None
        self._draw_highlight()
        self.fig.canvas.draw_idle()

    def _on_motion(self, event):
        if self._active is None or event.x is None:
            return
        self._active.drag(event.x, event.y, event)
        self._snap_axes(self._active)
        self._invalidate_renderer()
        self._draw_highlight()
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self._active is None:
            return
        _p = self._active.pos()
        self._active.end()
        self._clear_guides()
        self._set_title(
            f'{self._active.label} → '
            f'({", ".join(f"{v:.4f}" for v in _p)})  '
            f'[{self._active.coord_system()}]  |  p = print  |  u = undo'
        )
        self._active = None
        self._invalidate_renderer()
        self._draw_highlight()
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if self._selected is None:
            return
        factor = 1.1 ** (event.step if event.step else
                         (1 if event.button == 'up' else -1))
        if self._selected.scale(factor):
            self._invalidate_renderer()
            self._draw_highlight()
            self.fig.canvas.draw_idle()
            self._set_title(f'Rescaled {self._selected.label}  |  r = reset')

    _ARROWS = {'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1)}

    def _on_key(self, event):
        key = event.key or ''
        if key in ('p', 'P'):
            self.print_positions()
            return
        if key in ('u', 'U'):
            self._undo_last()
            return
        if key in ('r', 'R'):
            self._reset_selected()
            return
        if key in ('+', '=', '-') and self._selected is not None:
            if self._selected.scale(1.1 if key in ('+', '=') else 1 / 1.1):
                self._invalidate_renderer()
                self._draw_highlight()
                self.fig.canvas.draw_idle()
            return

        step, arrow_key = 1.0, key
        if key.startswith('shift+'):
            step, arrow_key = 10.0, key[len('shift+'):]
        if arrow_key in self._ARROWS and self._selected is not None:
            dx, dy = self._ARROWS[arrow_key]
            self._selected.nudge(dx * step, dy * step)
            self._invalidate_renderer()
            self._draw_highlight()
            self.fig.canvas.draw_idle()
            vals = ', '.join(f'{v:.4f}' for v in self._selected.pos())
            self._set_title(f'{self._selected.label} → ({vals})  |  r = reset')

    def _on_close(self, event):
        # The highlight and guides are editor chrome, not figure content.
        self._selected = None
        self._draw_highlight()
        self._clear_guides()
        self._write_overrides()
        self.print_positions()

    # ── undo / reset ──────────────────────────────────────────────────────────

    def _undo_last(self):
        candidates = [it for it in self._items if it._history]
        if not candidates:
            self._set_title('Nothing to undo.')
            return
        # Undo the one with the most recent history entry
        target = max(candidates, key=lambda it: len(it._history))
        target.undo()
        self._invalidate_renderer()
        self._draw_highlight()
        self.fig.canvas.draw_idle()
        self._set_title(f'Undid last move of {target.label}  |  p = print')

    def _reset_selected(self):
        if self._selected is None:
            self._set_title('Nothing selected to reset.')
            return
        self._selected.reset()
        self._invalidate_renderer()
        self._draw_highlight()
        self.fig.canvas.draw_idle()
        self._set_title(f'Reset {self._selected.label} to its original position')

    # ── output ────────────────────────────────────────────────────────────────

    def print_positions(self):
        moved = [it for it in self._items if it.moved]
        if not moved:
            print('(No artists were moved.)')
            return
        addr = [it for it in moved if getattr(it, 'override_address', None)]
        other = [it for it in moved if not getattr(it, 'override_address', None)]
        if addr:
            dest = self.overrides_path or '(pass overrides_path to auto-save these)'
            print(f'\n# ── Overridable — {"saved to " + dest if self.overrides_path else dest}')
            for it in addr:
                vals = [round(float(v), 4) for v in it.pos()]
                extra = ''
                if it.override_kind == 'panel':
                    extra = ('  delta=' +
                             str([round(float(v), 4) for v in it.delta()]))
                print(f'#   {it.override_address}  ->  {vals}{extra}')
        if other:
            print('\n# ── Paste these back into your plotting code ' + '─' * 20)
            for it in other:
                print(it.code_snippet())
            print('# ' + '─' * 60)

    # ── window title ──────────────────────────────────────────────────────────

    def _set_title(self, msg: str):
        try:
            self.fig.canvas.manager.set_window_title(f'DragEditor — {msg}')
        except Exception:
            pass

    # ── run ───────────────────────────────────────────────────────────────────

    def _ensure_managed(self):
        """Attach the figure to a pyplot figure manager for the active backend.

        A figure that was detached/closed from pyplot's registry before pickling
        (marimo does this after every cell) unpickles *without* a manager, so
        ``plt.show()`` would have nothing to display and return immediately. Give
        it a fresh manager so the editor window actually opens.  Runs before
        event connection so handlers bind to the correct (new) canvas.
        """
        from matplotlib._pylab_helpers import Gcf
        fig = self.fig
        # Composer figures use dpi=600 for print; at that dpi the on-screen window
        # would be thousands of pixels and overflow the display. Render at a
        # screen dpi instead — all positions are relative (figure/axes fraction),
        # so this does NOT change any override values, only the pixel size.
        if self._screen_dpi:
            fig.set_dpi(self._screen_dpi)
        if any(m.canvas.figure is fig for m in Gcf.get_all_fig_managers()):
            return
        try:
            num = max(Gcf.figs, default=0) + 1
            manager = plt._backend_mod.new_figure_manager_given_figure(num, fig)
            Gcf._set_new_active_manager(manager)
        except Exception:
            # Fallback: adopt our figure onto a fresh pyplot-managed canvas.
            mgr = plt.figure().canvas.manager
            mgr.canvas.figure = fig
            fig.set_canvas(mgr.canvas)

    def run(self):
        """Open the interactive window (blocks until closed)."""
        plt.show(block=True)


# ── Convenience launcher (no manual pickle) ───────────────────────────────────

def launch_editor(fig, patch_types=(Rectangle,), overrides_path=None,
                  screen_dpi=100, include_panels=True, snap=True):
    """Launch the drag editor for *fig* without any manual pickle step.

    Serialises *fig* to a temporary file, opens the interactive window in a
    subprocess (so the GUI backend is always available regardless of the
    calling environment), and cleans up on exit.  Blocks until the window is
    closed, then prints updated coordinates to the terminal where marimo/your
    script was started.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    patch_types : tuple of type, optional
        Patch subclasses to make draggable. Defaults to ``(Rectangle,)``.
    include_panels : bool, optional
        Allow panel axes to be moved/resized (default True).
    snap : bool, optional
        Snap dragged Axes edges to neighbouring panel edges (default True).

    Usage (from a marimo cell)::

        from sciplotlib.drag_editor import launch_editor
        launch_editor(fig)
    """
    import os
    import subprocess
    import tempfile
    import importlib.util as _u

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pkl_path = f.name
        pickle.dump(fig, f)

    # The editor opens a native window, which needs an *interactive* matplotlib
    # backend + a display.  The default is often non-interactive Agg (esp. under
    # marimo / uv), which silently shows nothing — so pick a working one and warn
    # clearly if we can't.  Prefer Qt: it is self-contained, whereas uv's managed
    # Python frequently ships a broken Tk.
    env = dict(os.environ)
    if env.get('MPLBACKEND', '').lower() in ('', 'agg'):
        if any(_u.find_spec(m) for m in ('PyQt6', 'PySide6', 'PyQt5', 'PySide2')):
            env['MPLBACKEND'] = 'QtAgg'
        elif _u.find_spec('tkinter'):
            env['MPLBACKEND'] = 'TkAgg'
        else:
            print('[drag_editor] No interactive matplotlib backend is installed, '
                  'so the editor window cannot open.\n'
                  '              Install one, e.g.:  uv add pyside6')
    if not (env.get('DISPLAY') or env.get('WAYLAND_DISPLAY')
            or sys.platform in ('darwin', 'win32')):
        print('[drag_editor] No display detected ($DISPLAY unset) — a native '
              'window cannot be shown on a headless/remote host (e.g. marimo on '
              'a server). Run where you have a desktop, or use X-forwarding.')

    # The child resolves `sciplotlib` through its OWN sys.path, which can pick a
    # different installation from the one that just composed this figure — e.g. a
    # pinned wheel in the project venv shadowing the working checkout the caller
    # imported. That silently runs an older editor with fewer features. Pin the
    # child to whichever copy the parent is actually using.
    import sciplotlib as _spl
    _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(_spl.__file__)))
    env['PYTHONPATH'] = os.pathsep.join(
        [_pkg_root] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))

    print(f'[drag_editor] opening editor (backend={env.get("MPLBACKEND", "default")}, '
          f'sciplotlib={_pkg_root})')
    print(_CONTROLS)

    cmd = [sys.executable, '-m', 'sciplotlib.drag_editor', pkl_path,
           '--screen-dpi', str(screen_dpi)]
    if overrides_path is not None:
        cmd += ['--overrides', str(overrides_path)]
    if not include_panels:
        cmd += ['--no-panels']
    if not snap:
        cmd += ['--no-snap']
    try:
        subprocess.run(cmd, env=env, check=False)
    finally:
        try:
            os.unlink(pkl_path)
        except OSError:
            pass


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli():
    import argparse
    parser = argparse.ArgumentParser(
        prog='python -m sciplotlib.drag_editor',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('pkl', help='Path to a pickled matplotlib Figure (.pkl)')
    parser.add_argument('--overrides', default=None,
                        help='Path to an overrides JSON to write dragged '
                             'artists (panels, labels, colorbars, images, '
                             'text) into on close.')
    parser.add_argument('--screen-dpi', type=float, default=100.0,
                        help='On-screen render dpi (default 100). Lower it if the '
                             'window is too big; layout and overrides are unaffected.')
    parser.add_argument('--no-panels', action='store_true',
                        help='Do not make composer panel axes draggable.')
    parser.add_argument('--no-snap', action='store_true',
                        help='Disable edge snapping while dragging axes.')
    args = parser.parse_args()

    try:
        with open(args.pkl, 'rb') as f:
            fig = pickle.load(f)
    except Exception as e:
        print(f'Error loading {args.pkl}: {e}', file=sys.stderr)
        sys.exit(1)

    editor = PositionEditor(fig, overrides_path=args.overrides,
                            screen_dpi=args.screen_dpi,
                            include_panels=not args.no_panels,
                            snap=not args.no_snap)
    n_panels = sum(1 for it in editor._items
                   if getattr(it, 'role', None) == 'panel')
    n_addr = sum(1 for it in editor._items
                 if getattr(it, 'override_address', None))
    print(f'[drag_editor] running {__file__}\n'
          f'[drag_editor] {n_panels} draggable panel(s), '
          f'{n_addr} addressable artist(s)')
    if n_panels == 0:
        print('[drag_editor] NOTE: no panel axes found. Either the figure was '
              'not built by FigureComposer, or an older sciplotlib is being '
              'imported here than the one that composed it.')
    editor.run()


if __name__ == '__main__':
    _cli()
