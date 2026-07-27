"""
Interactive drag-and-drop position editor for matplotlib artists.

Supported artist types
----------------------
- Text / Annotation (respects transAxes, transData, transFigure)
- Axis labels (x/y label) — emit ``set_label_coords`` so the move survives redraws
- AnnotationBbox  (SVG/image overlays created with place_image())
- Axes (colorbars, inset axes) — translate their position box (the inset
  locator is detached so the move sticks)
- Rectangle and other Patch subclasses

Recommended entry point (FigureComposer)
----------------------------------------
    composer.launch_editor(overrides_path='figure.overrides.json')

opens the window; on close, moves of *named panel roles* (axis labels,
colorbars) are written to the overrides JSON keyed by stable auto-addresses
(``panel:<label>/xlabel|ylabel|colorbar``) with no manual tagging, and
``composer.apply_overrides(path)`` re-applies them on every render. Other
artists (arbitrary text/images/patches) print paste-back snippets instead.

Requires an interactive backend + a display: under marimo/uv install Qt
(``uv add pyside6``); ``launch_editor`` auto-selects it and warns otherwise.
The figure is shown at ``screen_dpi`` (default 100), not its print dpi.

Manual workflow (any figure, no composer)
-----------------------------------------
1. Pickle the figure::

       import pickle
       with open('/tmp/fig.pkl', 'wb') as f:
           pickle.dump(fig, f)

2. Run the editor (add ``--overrides path.json`` to save panel roles)::

       uv run python -m sciplotlib.drag_editor /tmp/fig.pkl

3. Drag artists.  **p** = print coordinates, **u** = undo, **close** = summary.
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
import traceback
import numpy as np

import matplotlib
from matplotlib.text import Text, Annotation
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.patches import Patch, Rectangle
import matplotlib.pyplot as plt

from sciplotlib import overrides as _overrides


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

def _get_window_extent(artist, renderer):
    try:
        return artist.get_window_extent(renderer)
    except Exception:
        return None


def _hits(artist, event, renderer, pad: float = 6.0) -> bool:
    """Return True if the mouse event is within the artist's bounding box."""
    bbox = _get_window_extent(artist, renderer)
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    # Ensure x0 < x1 and y0 < y1 (AnnotationBbox bbox may be inverted)
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    ex, ey = event.x, event.y
    return (x0 - pad) <= ex <= (x1 + pad) and (y0 - pad) <= ey <= (y1 + pad)


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
        return 'AnnotationBbox'
    if isinstance(artist, Rectangle):
        return f'Rectangle'
    if isinstance(artist, Patch):
        return type(artist).__name__
    return type(artist).__name__


# ── draggable wrapper (text / annotation / patch / axis label) ─────────────────

class _Item:
    """Draggable wrapper for a point-positioned artist.

    ``role`` is ``'text'`` for ordinary artists, or ``'xlabel'`` / ``'ylabel'``
    for axis labels — those are pinned with ``set_label_coords`` on first grab so
    the move survives matplotlib's automatic re-placement, and their code snippet
    emits ``set_label_coords`` rather than ``set_position``.
    """

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

    @property
    def moved(self) -> bool:
        return len(self._history) > 0

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
        except Exception:
            pass
        self._pinned = True

    def start(self, ex: float, ey: float):
        if self.role in ('xlabel', 'ylabel') and not self._pinned:
            self._pin_axis_label()
        native = _get_pos(self.artist)
        t = self._tf()
        self._press_display = np.array([ex, ey], dtype=float)
        self._press_native = native.copy()
        self._press_display_of_native = t.transform(native)
        self._history.append(native.copy())

    def drag(self, ex: float, ey: float):
        if self._press_display is None:
            return
        delta = np.array([ex, ey]) - self._press_display
        new_display = self._press_display_of_native + delta
        t = self._tf()
        new_native = t.inverted().transform(new_display)
        _set_pos(self.artist, new_native)

    def end(self):
        self._press_display = None
        self._press_native = None
        self._press_display_of_native = None

    def undo(self):
        if self._history:
            _set_pos(self.artist, self._history.pop())

    def pos(self) -> np.ndarray:
        return _get_pos(self.artist)

    def coord_system(self) -> str:
        return _transform_name(self.artist, self.ax, self._explicit_transform)

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
            return (
                f'# {self.label}  [{cs}]\n'
                f'.xy = ({x:.4f}, {y:.4f})'
            )
        if isinstance(a, Patch):
            return (
                f'# {self.label}  [{cs}]\n'
                f'.set_xy(({x:.4f}, {y:.4f}))'
            )
        return f'# {self.label}: ({x:.4f}, {y:.4f})'


# ── draggable wrapper (Axes: colorbars, inset axes) ────────────────────────────

class _AxesItem:
    """Draggable wrapper for an Axes (colorbar / inset). Translates its position
    box (x0, y0) in figure-fraction coords; width/height are preserved."""

    def __init__(self, ax, parent_label=None):
        self.artist = ax
        self.ax = ax
        self.parent_label = parent_label
        loc = f" in panel '{parent_label}'" if parent_label else ''
        self.label = f'Axes (colorbar/inset){loc}'

        self._history: list[np.ndarray] = []
        self._press_display: np.ndarray | None = None
        self._press_bounds: np.ndarray | None = None

    @property
    def moved(self) -> bool:
        return len(self._history) > 0

    def start(self, ex: float, ey: float):
        # Inset axes (e.g. colorbars made with ax.inset_axes) carry a locator
        # that re-pins them to the parent on every redraw, silently overriding
        # set_position. Clear it so the drag actually moves (and sticks).
        if self.ax.get_axes_locator() is not None:
            self.ax.set_axes_locator(None)
        self._press_display = np.array([ex, ey], dtype=float)
        self._press_bounds = np.array(self.ax.get_position().bounds, dtype=float)
        self._history.append(self._press_bounds.copy())

    def drag(self, ex: float, ey: float):
        if self._press_display is None:
            return
        fig = self.ax.figure
        dx = (ex - self._press_display[0]) / fig.bbox.width
        dy = (ey - self._press_display[1]) / fig.bbox.height
        x0, y0, w, h = self._press_bounds
        self.ax.set_position([x0 + dx, y0 + dy, w, h])

    def end(self):
        self._press_display = None
        self._press_bounds = None

    def undo(self):
        if self._history:
            self.ax.set_position(list(self._history.pop()))

    def pos(self) -> np.ndarray:
        return np.array(self.ax.get_position().bounds, dtype=float)

    def coord_system(self) -> str:
        return 'figure fraction'

    def code_snippet(self) -> str:
        x0, y0, w, h = self.pos()
        return (f'# {self.label}  [figure fraction; x0, y0, w, h]\n'
                f'.set_position([{x0:.4f}, {y0:.4f}, {w:.4f}, {h:.4f}])')


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
        Make inset/colorbar Axes draggable (translated as a whole). Default True.
        Top-level panel axes are never draggable (they are re-placed by the
        composer's ``fit_axes_to_cells``, so a manual move would not persist).
    include_axis_labels : bool, optional
        Make x/y axis labels draggable. Default True.
    """

    def __init__(self, fig, patch_types=(Rectangle,), include_axes=True,
                 include_axis_labels=True, overrides_path=None, screen_dpi=100):
        self.fig = fig
        self._patch_types = patch_types or ()
        self._include_axes = include_axes
        self._include_axis_labels = include_axis_labels
        self.overrides_path = overrides_path
        self._screen_dpi = screen_dpi
        self._items: list = []
        self._active = None
        self._renderer = None
        self._ensure_managed()
        self._collect()
        self._tag_overridable()
        self._connect()
        self._set_title('Click an artist to drag it  |  p = print  |  u = undo')

    # ── collection ────────────────────────────────────────────────────────────

    def _collect(self):
        # Inset axes (create_inset_grid, ax.inset_axes) may live in
        # ax.child_axes without appearing in fig.get_axes(), so recurse.
        seen: set[int] = set()
        axes_items: list = []

        def _collect_ax(ax):
            if id(ax) in seen:
                return
            seen.add(id(ax))
            for t in ax.texts:
                self._items.append(_Item(t, ax))
            for a in ax.artists:
                if isinstance(a, AnnotationBbox):
                    self._items.append(_Item(a, ax))
            if self._patch_types:
                for p in ax.patches:
                    if isinstance(p, self._patch_types):
                        self._items.append(_Item(p, ax))
            if self._include_axis_labels:
                if ax.xaxis.label.get_text():
                    self._items.append(_Item(ax.xaxis.label, ax, role='xlabel'))
                if ax.yaxis.label.get_text():
                    self._items.append(_Item(ax.yaxis.label, ax, role='ylabel'))
            for child in getattr(ax, 'child_axes', []):
                if self._include_axes:
                    axes_items.append(_AxesItem(child, parent_label=_panel_label(ax)))
                _collect_ax(child)

        for ax in self.fig.get_axes():
            _collect_ax(ax)
            # Figure-level colorbars are top-level axes flagged with ._colorbar.
            if self._include_axes and getattr(ax, '_colorbar', None) is not None:
                axes_items.append(_AxesItem(ax, parent_label=_panel_label(ax)))

        # Axes are checked LAST (lowest priority) so text/patches inside them win
        # hit-testing; put them at the front (reversed() checks the tail first).
        self._items = axes_items + self._items

    def _tag_overridable(self):
        """Attach a stable auto-address to items that map to a named panel role
        (xlabel/ylabel/colorbar), so drops can be written to an overrides file
        with no manual tagging."""
        amap = _overrides.address_map(self.fig)
        for it in self._items:
            addr, kind, fp = amap.get(id(it.artist), (None, None, None))
            it.override_address = addr
            it.override_kind = kind
            it.override_fingerprint = fp

    def _write_overrides(self):
        if not self.overrides_path:
            return
        moved = [it for it in self._items
                 if it.moved and getattr(it, 'override_address', None)]
        if not moved:
            return
        data = _overrides.read_overrides(self.overrides_path)  # merge, don't clobber
        for it in moved:
            data[it.override_address] = {
                'kind': it.override_kind,
                'value': [float(v) for v in it.pos()],
                'fingerprint': it.override_fingerprint,
            }
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

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.button != 1 or event.x is None:
            return
        renderer = self._get_renderer()
        for item in reversed(self._items):  # reversed → topmost first
            if _hits(item.artist, event, renderer):
                self._active = item
                item.start(event.x, event.y)
                self._set_title(f'Dragging {item.label}  [{item.coord_system()}]  |  u = undo')
                return

    def _on_motion(self, event):
        if self._active is None or event.x is None:
            return
        self._active.drag(event.x, event.y)
        self._invalidate_renderer()
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self._active is None:
            return
        _p = self._active.pos()
        x, y = _p[0], _p[1]
        self._active.end()
        self._set_title(
            f'Released {self._active.label} → ({x:.4f}, {y:.4f})  '
            f'[{self._active.coord_system()}]  |  p = print  |  u = undo'
        )
        self._active = None
        self._invalidate_renderer()

    def _on_key(self, event):
        if event.key in ('p', 'P'):
            self.print_positions()
        elif event.key in ('u', 'U'):
            self._undo_last()

    def _on_close(self, event):
        self._write_overrides()
        self.print_positions()

    # ── undo ──────────────────────────────────────────────────────────────────

    def _undo_last(self):
        # Find the most recently moved item (last entry in history)
        candidates = [it for it in self._items if it.moved]
        if not candidates:
            self._set_title('Nothing to undo.')
            return
        # Undo the one with the most recent history entry
        target = max(candidates, key=lambda it: len(it._history))
        target.undo()
        self._invalidate_renderer()
        self.fig.canvas.draw_idle()
        self._set_title(f'Undid last move of {target.label}  |  p = print')

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
                print(f'#   {it.override_address}  ->  {vals}')
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

def launch_editor(fig, patch_types=(Rectangle,), overrides_path=None, screen_dpi=100):
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
    print(f'[drag_editor] opening editor (backend={env.get("MPLBACKEND", "default")}) — '
          'drag artists, then CLOSE the window to save/print positions…')

    cmd = [sys.executable, '-m', 'sciplotlib.drag_editor', pkl_path,
           '--screen-dpi', str(screen_dpi)]
    if overrides_path is not None:
        cmd += ['--overrides', str(overrides_path)]
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
                             'panel roles (colorbar/axis labels) into on close.')
    parser.add_argument('--screen-dpi', type=float, default=100.0,
                        help='On-screen render dpi (default 100). Lower it if the '
                             'window is too big; layout and overrides are unaffected.')
    args = parser.parse_args()

    try:
        with open(args.pkl, 'rb') as f:
            fig = pickle.load(f)
    except Exception as e:
        print(f'Error loading {args.pkl}: {e}', file=sys.stderr)
        sys.exit(1)

    editor = PositionEditor(fig, overrides_path=args.overrides,
                            screen_dpi=args.screen_dpi)
    editor.run()


if __name__ == '__main__':
    _cli()
