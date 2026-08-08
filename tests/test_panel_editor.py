"""Tests for the tk panel editor. Skipped where no display is available."""
import os
import pytest

pytest.importorskip('tkinter')
pytest.importorskip('PIL')
if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
    pytest.skip('no display', allow_module_level=True)

# Tk aborts with SIGABRT on some uv-managed CPythons — a hard abort inside the
# Tcl library, not a catchable exception, so it would take the whole pytest run
# down. Probe by building the real editor in a subprocess and skip if it dies.
# A cheaper probe is not enough: on uv's CPython 3.13 a bare Tk() + Combobox
# succeeds and only ttk.Entry aborts, and only once matplotlib and PIL are in.
import subprocess as _sp
import sys as _sys

_PROBE = """
import os
os.environ['SPL_EDITOR_HEADLESS'] = '1'
import matplotlib; matplotlib.use('Agg')
from PIL import Image
from sciplotlib.compose import FigureComposer
from sciplotlib.panel_editor import PanelEditor
c = FigureComposer(width_cm=6, height_cm=4, grid_rows=1, grid_cols=1, dpi=100)
c.add_panel('a', 0, 0, 1, 1)
fig, axes = c.compose()
axes['a'].plot([0, 1], [0, 1])
e = PanelEditor(fig, view_dpi=60)
e.root.destroy()
"""
if _sp.run([_sys.executable, '-c', _PROBE], capture_output=True).returncode != 0:
    pytest.skip('tkinter cannot build the editor in this interpreter',
                allow_module_level=True)

os.environ['SPL_EDITOR_HEADLESS'] = '1'

import matplotlib
matplotlib.use('Agg')
import numpy as np

from sciplotlib import overrides as ov
from sciplotlib.compose import FigureComposer, place_image
from sciplotlib.drag_editor import _AxesItem
from sciplotlib.panel_editor import PanelEditor

IMG = np.zeros((4, 6, 4), dtype=float)


def _fig():
    c = FigureComposer(width_cm=10, height_cm=6, grid_rows=2, grid_cols=2, dpi=600)
    c.add_panel('a', 0, 0, 1, 1)
    c.add_panel('b', 0, 1, 1, 1)
    fig, axes = c.compose()
    axes['a'].plot([0, 1], [0, 1])
    axes['a'].set_xlabel('trials')
    axes['a'].text(0.2, 0.8, 'hello')
    place_image(axes['a'], IMG, 0.5, 0.2, zoom=0.5)
    axes['b'].plot([0, 1], [1, 0])
    c.fit_axes_to_cells()
    return c, fig, axes


@pytest.fixture
def ed():
    c, fig, axes = _fig()
    e = PanelEditor(fig, view_dpi=100)
    yield e
    e.root.destroy()


def test_builds_and_finds_panels(ed):
    panels = [i for i in ed.items if getattr(i, 'role', None) == 'panel']
    assert len(panels) == 2


def test_backdrop_rendered_at_view_dpi(ed):
    w, h = ed._img_size
    assert w == pytest.approx(10 / 2.54 * 100, abs=2)
    assert h == pytest.approx(6 / 2.54 * 100, abs=2)


def test_view_dpi_does_not_disturb_the_print_dpi(ed):
    assert ed._print_dpi == 600
    ed.fig.set_dpi(ed._print_dpi)
    assert ed.fig.get_dpi() == 600


def test_tree_groups_elements_under_their_panel(ed):
    labels = [ed.tree.item(n, 'text') for n in ed.tree.get_children('')]
    assert 'panel a' in labels and 'panel b' in labels


def test_coordinate_roundtrip(ed):
    for pt in ((0, 0), (37, 91), (ed._img_size[0], ed._img_size[1])):
        assert ed._to_mpl(*ed._to_canvas(*pt)) == pt


def test_hit_test_prefers_small_artists_over_the_panel(ed):
    text_item = next(i for i in ed.items
                     if getattr(i, 'override_address', '') and
                     'text' in (i.override_address or ''))
    box = ed._item_box(text_item)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    assert ed._hit_test(cx, cy) is text_item


def test_press_drag_release_moves_a_panel(ed):
    panel = next(i for i in ed.items if getattr(i, 'role', None) == 'panel'
                 and i.parent_label == 'a')
    before = panel.pos().copy()
    box = ed._item_box(panel)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    class E:
        def __init__(self, x, y, state=0):
            self.x, self.y, self.state = x, y, state

    ed._on_press(E(cx, cy))
    assert ed.selected is panel
    ed._on_drag(E(cx + 20, cy))
    ed._on_release(E(cx + 20, cy))
    assert panel.pos()[0] > before[0]
    assert panel.moved


def test_numeric_fields_apply(ed):
    panel = next(i for i in ed.items if getattr(i, 'role', None) == 'panel')
    ed._select(panel)
    ed.fields['x0'][0].set('0.2000')
    ed._apply_fields()
    assert panel.pos()[0] == pytest.approx(0.20, abs=1e-6)


def test_nudge_and_undo(ed):
    panel = next(i for i in ed.items if getattr(i, 'role', None) == 'panel')
    ed._select(panel)
    start = panel.pos().copy()
    ed._nudge(1, 0, step=10)
    assert panel.pos()[0] > start[0]
    ed._undo()
    assert panel.pos() == pytest.approx(start)


def test_reset_all_restores_everything(ed):
    panel = next(i for i in ed.items if getattr(i, 'role', None) == 'panel')
    start = panel.pos().copy()
    ed._select(panel)
    ed._nudge(1, 1, step=20)
    ed._reset_all()
    assert panel.pos() == pytest.approx(start)
    assert not panel.moved


def test_save_writes_overrides(ed, tmp_path):
    path = tmp_path / 'o.json'
    ed.overrides_path = str(path)
    panel = next(i for i in ed.items if getattr(i, 'role', None) == 'panel'
                 and i.parent_label == 'a')
    ed._select(panel)
    ed._nudge(15, 0)
    ed._save()
    data = ov.read_overrides(path)
    assert data['panel:a']['kind'] == 'panel'
    assert 'delta' in data['panel:a']


def test_image_scale_field(ed):
    img = next(i for i in ed.items if getattr(i, 'is_image', False))
    ed._select(img)
    assert ed.zoom_field.get()
    ed.zoom_field.set('0.9')
    ed._apply_fields()
    assert ov.image_zoom(img.artist) == pytest.approx(0.9)
