"""
Tk panel editor — a real application window for adjusting a composed figure.

This is the GUI counterpart to :mod:`sciplotlib.drag_editor`. That editor is a
bare matplotlib window: the figure *is* the interface, and every control is a
mouse gesture or a key press. This one is a proper app — an element tree, numeric
position fields, buttons, a status line — built on the same toolkit as the
``make-layout`` designer in :mod:`sciplotlib.layout`, so the two feel related.

    ┌──────────────────────────────────────────────────────────────┐
    │ Save   Re-render   Undo   Reset all   ☑ snap   zoom [100 %]  │
    ├──────────────────┬───────────────────────────────────────────┤
    │ Elements         │                                           │
    │  ▾ panel a       │      figure, with the selection outlined  │
    │      image       │      drag to move, grab an edge to resize │
    │      text: …     │                                           │
    │  ▾ panel b       │                                           │
    ├──────────────────┤                                           │
    │ Selected         │                                           │
    │  x0 [ 0.0352 ]   │                                           │
    │  y0 [ 0.3699 ]   │                                           │
    │  w  [ 0.1869 ]   │                                           │
    │  h  [ 0.2800 ]   │                                           │
    │  [Apply] [Reset] │                                           │
    ├──────────────────┴───────────────────────────────────────────┤
    │ status                                                       │
    └──────────────────────────────────────────────────────────────┘

What it edits, what a move means, and how positions are written back are all
shared with the matplotlib editor: both call :func:`drag_editor.collect_items`
and store through :mod:`sciplotlib.overrides`, so a panel move is a delta against
the fitted layout in either one.

Why a rendered image rather than an embedded matplotlib canvas: the backdrop is
drawn once with Agg and only re-rendered when a drag ends, so dragging stays
responsive on a 600-dpi print figure, and the overlay is drawn with native
canvas items that can carry handles, guides and hover feedback.

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

import numpy as np

import matplotlib
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


def _display_name(item):
    """Short label for the element tree."""
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
    view_dpi : float
        Render dpi for the on-screen backdrop. Independent of the figure's print
        dpi; all stored positions are relative, so this changes nothing but the
        pixel size of the preview.
    snap : bool
        Snap a dragged axes edge onto a neighbouring panel's edge.
    """

    def __init__(self, fig, overrides_path=None, view_dpi=150, snap=True):
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
        self.view_dpi = view_dpi
        self.snap = snap

        # Agg only: the backdrop is rendered to an image, never shown by a
        # matplotlib GUI backend, so no interactive backend is needed at all.
        matplotlib.use('Agg', force=True)
        self._agg = FigureCanvasAgg(fig)
        self._print_dpi = fig.get_dpi()

        self.items = collect_items(fig)
        self.selected = None
        self._drag = None            # {'item', 'x0', 'y0'} while dragging
        self._photo = None
        self._img_size = (0, 0)
        self._overlay_ids = []
        self._guide_ids = []
        self._history = []           # (item, kind) for undo across items

        self._build_ui()
        self._render_backdrop()
        self._populate_tree()
        self._status(f'{sum(1 for i in self.items if isinstance(i, _AxesItem) and i.role == "panel")} '
                     f'panels, {len(self.items)} editable elements. '
                     f'Click one, or pick it from the list.')

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        tk, ttk = self._tk, self._ttk
        try:                                   # optional, purely cosmetic
            import ttkbootstrap
            self.root = ttkbootstrap.Window(themename='cosmo')
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
        ttk.Button(bar, text='Re-render', command=self._render_backdrop
                   ).pack(side='left', padx=(4, 0))
        ttk.Button(bar, text='Undo', command=self._undo).pack(side='left', padx=(4, 0))
        ttk.Button(bar, text='Reset all', command=self._reset_all
                   ).pack(side='left', padx=(4, 0))

        self.snap_var = tk.BooleanVar(value=self.snap)
        ttk.Checkbutton(bar, text='snap', variable=self.snap_var
                        ).pack(side='left', padx=(12, 0))

        ttk.Label(bar, text='zoom').pack(side='left', padx=(12, 2))
        self.zoom_var = tk.StringVar(value='150')
        zoom = ttk.Combobox(bar, textvariable=self.zoom_var, width=5,
                            state='readonly',
                            values=('75', '100', '150', '200', '300'))
        zoom.pack(side='left')
        zoom.bind('<<ComboboxSelected>>', self._on_zoom)

        # -- left column: element tree + properties --------------------------
        left = ttk.Frame(self.root, padding=(6, 6))
        left.grid(row=1, column=0, sticky='ns')

        ttk.Label(left, text='Elements', font=('TkDefaultFont', 10, 'bold')
                  ).pack(anchor='w')
        self.tree = ttk.Treeview(left, height=18, show='tree', selectmode='browse')
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

        ttk.Label(left, text='drag to move · grab an edge to resize\n'
                            'arrows nudge (shift = 10 px)',
                  foreground='#777', justify='left').pack(anchor='w', pady=(8, 0))

        # -- figure canvas ---------------------------------------------------
        right = ttk.Frame(self.root, padding=(0, 6, 6, 6))
        right.grid(row=1, column=1, sticky='nsew')
        self.canvas = tk.Canvas(right, background='#f4f4f4',
                                highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Button-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Motion>', self._on_hover)

        for seq, d in (('<Left>', (-1, 0)), ('<Right>', (1, 0)),
                       ('<Up>', (0, 1)), ('<Down>', (0, -1))):
            self.root.bind(seq, lambda e, d=d: self._nudge(*d, step=1))
            self.root.bind(seq.replace('<', '<Shift-'),
                           lambda e, d=d: self._nudge(*d, step=10))
        self.root.bind('<Control-s>', lambda e: self._save())
        self.root.bind('<Control-z>', lambda e: self._undo())

        # -- status ----------------------------------------------------------
        self.status = ttk.Label(self.root, text='', anchor='w',
                                padding=(8, 3), foreground='#333')
        self.status.grid(row=2, column=0, columnspan=2, sticky='ew')

        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(1, weight=1)

    def _status(self, msg):
        self.status.configure(text=msg)

    # ── rendering ────────────────────────────────────────────────────────────

    def _render_backdrop(self):
        """Re-draw the figure to an image and show it. Cheap enough to call on
        every drag release; too slow to call on every motion event."""
        self.fig.set_dpi(self.view_dpi)
        self._agg.draw()
        w, h = self._agg.get_width_height()
        img = self._Image.frombuffer(
            'RGBA', (w, h), self._agg.buffer_rgba(), 'raw', 'RGBA', 0, 1)
        flat = self._Image.new('RGBA', (w, h), (255, 255, 255, 255))
        flat.alpha_composite(img)
        self._photo = self._ImageTk.PhotoImage(flat)
        self._img_size = (w, h)

        self.canvas.delete('backdrop')
        self.canvas.create_image(0, 0, image=self._photo, anchor='nw',
                                 tags='backdrop')
        self.canvas.tag_lower('backdrop')
        self.canvas.configure(width=w, height=h, scrollregion=(0, 0, w, h))
        self._refresh_overlay()

    def _on_zoom(self, _event=None):
        self.view_dpi = float(self.zoom_var.get())
        self._render_backdrop()

    # ── coordinate mapping ───────────────────────────────────────────────────
    # matplotlib display coords have their origin bottom-left, tk's is top-left;
    # the backdrop is drawn 1:1 at view_dpi so only the y flip is needed.

    def _to_canvas(self, mx, my):
        return mx, self._img_size[1] - my

    def _to_mpl(self, cx, cy):
        return cx, self._img_size[1] - cy

    def _item_box(self, item):
        """Item bounding box in canvas coords, or None."""
        bbox = item.bbox_display(self._agg.get_renderer())
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
        if self.selected is None:
            return
        box = self._item_box(self.selected)
        if box is None:
            return
        x0, y0, x1, y1 = box
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=SEL_COLOUR,
                                     width=1.5, dash=(4, 3), tags='overlay')
        if isinstance(self.selected, _AxesItem):
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
        sel = self.tree.selection()
        if not sel:
            return
        item = self._tree_items.get(sel[0])
        if item is not None:
            self._select(item)

    def _sync_tree_to(self, item):
        for node, it in self._tree_items.items():
            if it is item:
                self.tree.selection_set(node)
                self.tree.see(node)
                return

    # ── selection ────────────────────────────────────────────────────────────

    def _select(self, item):
        self.selected = item
        addr = getattr(item, 'override_address', None)
        self.addr_lbl.configure(
            text=f'{item.label}\n{addr or "(not addressable — prints a snippet)"}')
        self._sync_fields()
        self._refresh_overlay()
        self._status(f'Selected {item.label}')

    def _sync_fields(self):
        item = self.selected
        if item is None:
            for var, _ in self.fields.values():
                var.set('')
            self.zoom_field.set('')
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
        try:
            vals = [float(self.fields[n][0].get())
                    for n in ('x0', 'y0', 'w', 'h')
                    if self.fields[n][0].get().strip()]
        except ValueError:
            self._status('Could not read those numbers.')
            return
        self._push_history(item)
        if isinstance(item, _AxesItem) and len(vals) == 4:
            item._set_bounds(vals)
        elif len(vals) >= 2:
            from sciplotlib.drag_editor import _set_pos
            _set_pos(item.artist, vals[:2])
        if getattr(item, 'is_image', False) and self.zoom_field.get().strip():
            try:
                _overrides.set_image_zoom(item.artist, float(self.zoom_field.get()))
                item._zoom_changed = True
            except ValueError:
                pass
        self._after_change('Applied')

    # ── mouse ────────────────────────────────────────────────────────────────

    def _hit_test(self, cx, cy, pad=6):
        """Topmost item under the point. Items are ordered biggest-first, so the
        reversed walk reaches text and images before the panels holding them."""
        for item in reversed(self.items):
            box = self._item_box(item)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            if x0 - pad <= cx <= x1 + pad and y0 - pad <= cy <= y1 + pad:
                return item
        return None

    def _on_hover(self, event):
        item = self._hit_test(event.x, event.y)
        cursor = ''
        if item is not None and isinstance(item, _AxesItem):
            box = self._item_box(item)
            if box:
                x0, y0, x1, y1 = box
                near_x = abs(event.x - x0) <= HANDLE_PX or abs(event.x - x1) <= HANDLE_PX
                near_y = abs(event.y - y0) <= HANDLE_PX or abs(event.y - y1) <= HANDLE_PX
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

    def _on_press(self, event):
        item = self._hit_test(event.x, event.y)
        if item is None:
            self.selected = None
            self._refresh_overlay()
            self._sync_fields()
            return
        self._select(item)
        self._sync_tree_to(item)
        self._push_history(item)
        mx, my = self._to_mpl(event.x, event.y)
        item.start(mx, my, _ShimEvent())
        self._drag = item

    def _on_drag(self, event):
        if self._drag is None:
            return
        mx, my = self._to_mpl(event.x, event.y)
        shift = bool(event.state & 0x0001)
        self._drag.drag(mx, my, _ShimEvent('shift' if shift else None))
        if self.snap_var.get() and isinstance(self._drag, _AxesItem):
            others = [i for i in self.items
                      if isinstance(i, _AxesItem) and i is not self._drag]
            hx, hy = snap_axes_item(self._drag, others, self.fig)
            self._draw_guides(hx, hy)
        # Move the outline live; the backdrop catches up on release.
        self._refresh_overlay()
        self._sync_fields()

    def _on_release(self, _event):
        if self._drag is None:
            return
        self._drag.end()
        self._drag = None
        self._clear_guides()
        self._after_change('Moved')

    def _nudge(self, dx, dy, step=1):
        if self.selected is None:
            return
        self._push_history(self.selected)
        self.selected.nudge(dx * step, dy * step)
        self._after_change(f'Nudged {dx * step:+d}, {dy * step:+d} px')

    # ── history / reset / save ───────────────────────────────────────────────

    def _push_history(self, item):
        self._history.append(item)

    def _undo(self):
        while self._history:
            item = self._history.pop()
            if item._history:
                item.undo()
                self._after_change(f'Undid a change to {item.label}')
                return
        self._status('Nothing to undo.')

    def _reset_selected(self):
        if self.selected is None:
            return
        self.selected.reset()
        self._after_change(f'Reset {self.selected.label}')

    def _reset_all(self):
        for it in self.items:
            it.reset()
        self._history.clear()
        self._after_change('Reset every element to its composed position')

    def _after_change(self, msg):
        self._render_backdrop()
        self._sync_fields()
        self._status(msg)

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
        self.root.mainloop()
        # restore the print dpi so a figure edited in-process still saves right
        self.fig.set_dpi(self._print_dpi)
        self.print_positions()


# ── launcher ─────────────────────────────────────────────────────────────────

def launch_panel_editor(fig, overrides_path=None, view_dpi=150, snap=True):
    """Open the panel editor on *fig* in a subprocess. Blocks until it closes.

    The subprocess keeps the editor's Agg rendering away from whatever backend
    the caller (marimo, a script, a notebook) has already configured.
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pkl_path = f.name
        pickle.dump(fig, f)

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
    parser.add_argument('--view-dpi', type=float, default=150.0)
    parser.add_argument('--no-snap', action='store_true')
    args = parser.parse_args()

    with open(args.pkl, 'rb') as f:
        fig = pickle.load(f)

    editor = PanelEditor(fig, overrides_path=args.overrides,
                         view_dpi=args.view_dpi, snap=not args.no_snap)
    print(f'[panel_editor] running {__file__}')
    editor.run()


if __name__ == '__main__':
    _cli()
