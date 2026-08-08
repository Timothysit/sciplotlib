"""Tests for the drag editor's item wrappers and collection.

The GUI itself needs a display, so these drive the same code paths the mouse
handlers do (start/drag/end, nudge, scale, reset) without opening a window.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

from sciplotlib import overrides as ov
from sciplotlib.compose import FigureComposer, place_image
from sciplotlib.drag_editor import (PositionEditor, _AxesItem, _Item,
                                    _edge_zone, _norm_bbox)

IMG = np.zeros((4, 6, 4), dtype=float)


def _fig():
    c = FigureComposer(width_cm=10, height_cm=6, grid_rows=2, grid_cols=2, dpi=100)
    c.add_panel('a', 0, 0, 1, 1)
    c.add_panel('b', 0, 1, 1, 1)
    fig, axes = c.compose()
    axes['a'].plot([0, 1], [0, 1])
    axes['a'].set_xlabel('trials')
    axes['a'].text(0.2, 0.8, 'MP computer')
    place_image(axes['a'], IMG, 0.5, 0.2, zoom=0.5)
    axes['b'].plot([0, 1], [1, 0])
    c.fit_axes_to_cells()
    return c, fig, axes


def _editor(fig, **kw):
    return PositionEditor(fig, screen_dpi=None, **kw)


# ── hit zones ──────────────────────────────────────────────────────────────

class _Box:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1


@pytest.mark.parametrize('pt,expected', [
    ((50, 50), ''),        # interior -> move
    ((0, 50), 'l'),
    ((100, 50), 'r'),
    ((50, 0), 'b'),
    ((50, 100), 't'),
    ((0, 0), 'lb'),
    ((100, 100), 'rt'),
])
def test_edge_zone(pt, expected):
    assert _edge_zone(_Box(0, 0, 100, 100), *pt) == expected


def test_norm_bbox_flips_inverted_boxes():
    assert _norm_bbox(_Box(10, 20, 5, 8)) == (5, 8, 10, 20)


# ── collection ─────────────────────────────────────────────────────────────

def test_panels_are_collected_and_addressed():
    c, fig, axes = _fig()
    ed = _editor(fig)
    panels = [it for it in ed._items if getattr(it, 'role', None) == 'panel']
    assert {it.parent_label for it in panels} == {'a', 'b'}
    assert {it.override_address for it in panels} == {'panel:a', 'panel:b'}
    assert {it.override_kind for it in panels} == {'panel'}


def test_include_panels_false_drops_them():
    c, fig, axes = _fig()
    ed = _editor(fig, include_panels=False)
    assert not [it for it in ed._items if getattr(it, 'role', None) == 'panel']


def test_images_and_texts_get_addresses():
    c, fig, axes = _fig()
    ed = _editor(fig)
    got = {it.override_address for it in ed._items if it.override_address}
    assert 'panel:a/image' in got
    assert 'panel:a/text' in got
    assert 'panel:a/xlabel' in got


def test_panels_are_hit_tested_last():
    """Small artists inside a panel must win the click over the panel box."""
    c, fig, axes = _fig()
    ed = _editor(fig)
    roles = [getattr(it, 'role', None) for it in ed._items]
    last_panel = max(i for i, r in enumerate(roles) if r == 'panel')
    first_other = min(i for i, r in enumerate(roles) if r != 'panel')
    assert last_panel < first_other   # reversed() reaches non-panels first


# ── panel item behaviour ───────────────────────────────────────────────────

def test_panel_move_translates_position():
    c, fig, axes = _fig()
    ed = _editor(fig)
    item = next(it for it in ed._items if getattr(it, 'role', None) == 'panel'
                and it.parent_label == 'a')
    before = item.pos().copy()

    bbox = item.bbox_display(fig.canvas.get_renderer())
    cx = (bbox.x0 + bbox.x1) / 2
    cy = (bbox.y0 + bbox.y1) / 2
    item.start(cx, cy)
    assert item._mode == 'move'
    item.drag(cx + 30, cy + 20)
    item.end()

    after = item.pos()
    assert after[0] > before[0] and after[1] > before[1]
    assert after[2:] == pytest.approx(before[2:])      # size unchanged


def test_panel_edge_grab_resizes_not_moves():
    c, fig, axes = _fig()
    ed = _editor(fig)
    item = next(it for it in ed._items if getattr(it, 'role', None) == 'panel'
                and it.parent_label == 'a')
    before = item.pos().copy()

    bbox = item.bbox_display(fig.canvas.get_renderer())
    # grab the right edge, mid-height
    item.start(bbox.x1, (bbox.y0 + bbox.y1) / 2)
    assert 'r' in item._mode
    item.drag(bbox.x1 + 25, (bbox.y0 + bbox.y1) / 2)
    item.end()

    after = item.pos()
    assert after[0] == pytest.approx(before[0])        # left edge pinned
    assert after[2] > before[2]                        # got wider


def test_panel_delta_is_measured_against_the_fitted_layout():
    c, fig, axes = _fig()
    ed = _editor(fig)
    item = next(it for it in ed._items if getattr(it, 'role', None) == 'panel'
                and it.parent_label == 'a')
    start = item.pos().copy()
    item.nudge(10, 0)
    assert item.delta()[0] == pytest.approx(item.pos()[0] - start[0])
    entry = item.override_entry('panel')
    assert entry['kind'] == 'panel'
    assert entry['delta'][0] == pytest.approx(item.delta()[0])
    assert len(entry['value']) == 4


def test_panel_reset_returns_to_the_fitted_position():
    c, fig, axes = _fig()
    ed = _editor(fig)
    item = next(it for it in ed._items if getattr(it, 'role', None) == 'panel')
    start = item.pos().copy()
    item.nudge(30, 30)
    assert not np.allclose(item.pos(), start)
    item.reset()
    assert item.pos() == pytest.approx(start)
    assert not item.moved


def test_moving_a_panel_carries_unpinned_children():
    """A colorbar detached by an earlier drag must not be left behind."""
    c, fig, axes = _fig()
    cax = axes['a'].inset_axes([1.02, 0, 0.04, 1])
    cax.set_axes_locator(None)                 # simulate a previous drag
    before = cax.get_position().bounds

    item = _AxesItem(axes['a'], parent_label='a', role='panel')
    item.nudge(20, 0)
    after = cax.get_position().bounds
    assert after[0] > before[0]
    assert after[1] == pytest.approx(before[1])


def test_resizing_a_panel_leaves_children_alone():
    c, fig, axes = _fig()
    cax = axes['a'].inset_axes([1.02, 0, 0.04, 1])
    cax.set_axes_locator(None)
    before = cax.get_position().bounds

    item = _AxesItem(axes['a'], parent_label='a', role='panel')
    bbox = item.bbox_display(fig.canvas.get_renderer())
    item.start(bbox.x1, (bbox.y0 + bbox.y1) / 2)
    item.drag(bbox.x1 + 20, (bbox.y0 + bbox.y1) / 2)
    item.end()
    assert cax.get_position().bounds == pytest.approx(before)


def test_resize_cannot_collapse_the_axes():
    c, fig, axes = _fig()
    item = _AxesItem(axes['a'], parent_label='a', role='panel')
    bbox = item.bbox_display(fig.canvas.get_renderer())
    item.start(bbox.x1, (bbox.y0 + bbox.y1) / 2)
    item.drag(bbox.x0 - 500, (bbox.y0 + bbox.y1) / 2)   # drag way past the left
    item.end()
    assert item.pos()[2] > 0


# ── image item behaviour ───────────────────────────────────────────────────

def test_image_scale_marks_the_item_changed_and_round_trips():
    c, fig, axes = _fig()
    ed = _editor(fig)
    item = next(it for it in ed._items if getattr(it, 'is_image', False))
    assert not item.moved

    assert item.scale(2.0) is True
    assert item.moved                       # a pure zoom counts as a change
    assert ov.image_zoom(item.artist) == pytest.approx(1.0)

    entry = item.override_entry('image')
    assert entry['zoom'] == pytest.approx(1.0)
    assert entry['value'] == pytest.approx([0.5, 0.2])


def test_image_reset_restores_position_and_zoom():
    c, fig, axes = _fig()
    ed = _editor(fig)
    item = next(it for it in ed._items if getattr(it, 'is_image', False))
    item.scale(3.0)
    item.nudge(20, 20)
    item.reset()
    assert ov.image_zoom(item.artist) == pytest.approx(0.5)
    assert tuple(item.artist.xy) == pytest.approx((0.5, 0.2))
    assert not item.moved


def test_scale_is_a_noop_for_non_images():
    c, fig, axes = _fig()
    ed = _editor(fig)
    text_item = next(it for it in ed._items
                     if isinstance(it.artist, matplotlib.text.Text)
                     and it.artist.get_text() == 'MP computer')
    assert text_item.scale(2.0) is False
    panel = next(it for it in ed._items if getattr(it, 'role', None) == 'panel')
    assert panel.scale(2.0) is False


# ── text item behaviour ────────────────────────────────────────────────────

def test_text_nudge_and_undo():
    c, fig, axes = _fig()
    ed = _editor(fig)
    item = next(it for it in ed._items
                if isinstance(it.artist, matplotlib.text.Text)
                and it.artist.get_text() == 'MP computer')
    start = item.pos().copy()
    item.nudge(10, 0)
    assert item.pos()[0] > start[0]
    item.undo()
    assert item.pos() == pytest.approx(start)


def test_axis_label_pins_itself_on_nudge():
    """An unpinned axis label is re-placed on every draw, so the editor must
    call set_label_coords before moving it."""
    c, fig, axes = _fig()
    ed = _editor(fig)
    item = next(it for it in ed._items if it.role == 'xlabel')
    assert not item._pinned
    item.nudge(0, -5)
    assert item._pinned
    assert 'set_label_coords' in item.code_snippet()


# ── writing ────────────────────────────────────────────────────────────────

def test_write_overrides_merges_and_keys_by_address(tmp_path):
    c, fig, axes = _fig()
    path = tmp_path / 'o.json'
    ov.write_overrides(path, {'panel:zzz': {'kind': 'panel', 'value': [0, 0, 1, 1]}})

    ed = _editor(fig, overrides_path=str(path))
    panel = next(it for it in ed._items if getattr(it, 'role', None) == 'panel'
                 and it.parent_label == 'a')
    panel.nudge(15, 0)
    img = next(it for it in ed._items if getattr(it, 'is_image', False))
    img.nudge(0, 10)
    ed._write_overrides()

    data = ov.read_overrides(path)
    assert 'panel:zzz' in data                 # pre-existing entry preserved
    assert data['panel:a']['kind'] == 'panel'
    assert 'delta' in data['panel:a']
    assert data['panel:a/image']['kind'] == 'image'
    assert data['panel:a/image']['zoom'] == pytest.approx(0.5)


def test_write_overrides_skips_untouched_artists(tmp_path):
    c, fig, axes = _fig()
    path = tmp_path / 'o.json'
    ed = _editor(fig, overrides_path=str(path))
    ed._write_overrides()
    assert ov.read_overrides(path) == {}


def test_editor_write_then_apply_reproduces_the_drag(tmp_path):
    """End to end: drag in the editor, save, re-render, re-apply."""
    c, fig, axes = _fig()
    path = tmp_path / 'o.json'
    ed = _editor(fig, overrides_path=str(path))
    panel = next(it for it in ed._items if getattr(it, 'role', None) == 'panel'
                 and it.parent_label == 'a')
    panel.nudge(20, -10)
    dragged = panel.pos().copy()
    ed._write_overrides()

    # fresh render of the same figure spec
    c2 = FigureComposer(width_cm=10, height_cm=6, grid_rows=2, grid_cols=2, dpi=100)
    c2.add_panel('a', 0, 0, 1, 1)
    c2.add_panel('b', 0, 1, 1, 1)
    fig2, axes2 = c2.compose()
    axes2['a'].plot([0, 1], [0, 1])
    axes2['a'].set_xlabel('trials')
    axes2['b'].plot([0, 1], [1, 0])
    c2.apply_overrides(str(path), verbose=False)
    c2.fit_axes_to_cells()

    assert axes2['a'].get_position().bounds == pytest.approx(dragged, abs=1e-6)


# ── launcher: the child must run the SAME sciplotlib as the parent ─────────

def test_launch_editor_pins_the_child_to_the_parent_sciplotlib(tmp_path,
                                                               monkeypatch):
    """The editor runs in a subprocess, which resolves `sciplotlib` through its
    own sys.path. Without help that can be a *different* install from the one
    that composed the figure (a pinned wheel in a project venv shadowing the
    working checkout), silently giving an older editor with no panel support."""
    import os
    import sciplotlib
    from sciplotlib import drag_editor as de

    seen = {}

    def fake_run(cmd, env=None, **kw):
        seen['cmd'] = cmd
        seen['env'] = env
        class R:
            returncode = 0
        return R()

    # launch_editor imports subprocess inside the function, so patch the module
    import subprocess as _sp
    monkeypatch.setattr(_sp, 'run', fake_run)

    c, fig, axes = _fig()
    de.launch_editor(fig, overrides_path=str(tmp_path / 'o.json'))

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(sciplotlib.__file__)))
    assert 'PYTHONPATH' in seen['env']
    assert seen['env']['PYTHONPATH'].split(os.pathsep)[0] == pkg_root


def test_launch_editor_prepends_without_dropping_existing_pythonpath(
        tmp_path, monkeypatch):
    import os
    from sciplotlib import drag_editor as de

    seen = {}

    def fake_run(cmd, env=None, **kw):
        seen['env'] = env
        class R:
            returncode = 0
        return R()

    import subprocess as _sp
    monkeypatch.setattr(_sp, 'run', fake_run)
    monkeypatch.setenv('PYTHONPATH', '/some/existing/path')

    c, fig, axes = _fig()
    de.launch_editor(fig)
    parts = seen['env']['PYTHONPATH'].split(os.pathsep)
    assert '/some/existing/path' in parts
    assert len(parts) >= 2


def test_launch_editor_forwards_panel_and_snap_flags(tmp_path, monkeypatch):
    from sciplotlib import drag_editor as de

    seen = {}

    def fake_run(cmd, env=None, **kw):
        seen['cmd'] = cmd
        class R:
            returncode = 0
        return R()

    import subprocess as _sp
    monkeypatch.setattr(_sp, 'run', fake_run)

    c, fig, axes = _fig()
    de.launch_editor(fig, include_panels=False, snap=False)
    assert '--no-panels' in seen['cmd']
    assert '--no-snap' in seen['cmd']

    de.launch_editor(fig)
    assert '--no-panels' not in seen['cmd']
    assert '--no-snap' not in seen['cmd']
