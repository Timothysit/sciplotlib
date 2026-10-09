"""Tests for the tk panel editor. Skipped where no display is available."""
import json
import os
import pickle
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


# ── multi-selection and group move ─────────────────────────────────────────


class _E:
    """Synthetic tk mouse event: position and a modifier state mask."""
    CTRL = 0x0004
    SHIFT = 0x0001

    def __init__(self, x, y, state=0):
        self.x, self.y, self.state = x, y, state


def _panels(ed):
    a = next(i for i in ed.items if getattr(i, 'role', None) == 'panel'
             and i.parent_label == 'a')
    b = next(i for i in ed.items if getattr(i, 'role', None) == 'panel'
             and i.parent_label == 'b')
    return a, b


def _centre(ed, item):
    x0, y0, x1, y1 = ed._item_box(item)
    return (x0 + x1) / 2, (y0 + y1) / 2


def test_ctrl_click_adds_a_second_element_to_the_selection(ed):
    a, b = _panels(ed)
    ed._on_press(_E(*_centre(ed, a)))                 # plain click selects a
    assert ed.selection == [a]
    ed._on_press(_E(*_centre(ed, b), state=_E.CTRL))  # ctrl-click adds b
    assert set(ed.selection) == {a, b}
    assert ed.selected is b                            # primary is the last


def test_ctrl_click_again_toggles_an_element_back_out(ed):
    a, b = _panels(ed)
    ed._set_selection([a, b])
    ed._on_press(_E(*_centre(ed, b), state=_E.CTRL))
    assert ed.selection == [a]


def test_group_drag_moves_every_selected_panel_by_the_same_delta(ed):
    a, b = _panels(ed)
    ed._set_selection([a, b])
    ax0, bx0 = a.pos()[0], b.pos()[0]
    cx, cy = _centre(ed, a)                            # grab one of the group
    ed._on_press(_E(cx, cy))                           # a is in a >1 selection
    assert set(ed.selection) == {a, b}                 # group is preserved
    ed._on_drag(_E(cx + 25, cy))
    ed._on_release(_E(cx + 25, cy))
    da, db = a.pos()[0] - ax0, b.pos()[0] - bx0
    assert da > 0 and db > 0
    assert da == pytest.approx(db, abs=1e-6)           # moved together


def test_group_move_then_undo_reverts_both(ed):
    a, b = _panels(ed)
    a0, b0 = a.pos().copy(), b.pos().copy()
    ed._set_selection([a, b])
    ed._nudge(1, 0, step=10)
    assert a.pos()[0] > a0[0] and b.pos()[0] > b0[0]
    ed._undo()
    assert a.pos() == pytest.approx(a0)
    assert b.pos() == pytest.approx(b0)


def test_rubber_band_over_the_whole_figure_selects_every_panel(ed):
    ed._band = (0.0, 0.0)
    ed._band_additive = False
    w, h = ed._img_size
    ed._on_release(_E(w, h))
    labels = {getattr(i, 'parent_label', None) for i in ed.selection}
    assert 'a' in labels and 'b' in labels


def test_a_plain_click_on_empty_space_clears_the_selection(ed):
    a, _ = _panels(ed)
    ed._set_selection([a])
    ed._band = (5.0, 5.0)                              # a press that…
    ed._band_additive = False
    ed._on_release(_E(6, 6))                           # …barely moved
    assert ed.selection == []


def test_selecting_a_panel_and_its_own_text_moves_the_panel_once(ed):
    """The panel carries its in-axes text, so the text must not move twice."""
    a, _ = _panels(ed)
    text = next(i for i in ed.items if getattr(i, 'override_address', '')
                and 'text' in (i.override_address or '')
                and getattr(i, 'ax', None) is a.ax)
    ed._set_selection([a, text])
    movers = ed._independent_movers()
    assert a in movers and text not in movers


# ── poster-scale viewport ──────────────────────────────────────────────────
# A poster is ~6x the width of a print figure: at the old fixed 150 dpi an A0
# backdrop is ~7000 x 5000 px, far past the window and past what a PhotoImage
# should be asked to hold. These cover the fit/clamp/scroll behaviour that
# makes such a figure editable.

def _poster_editor(**kwargs):
    from sciplotlib.poster import PosterComposer
    p = PosterComposer(paper='a0_landscape', grid_rows=12, grid_cols=12, dpi=300)
    p.add_title('T', authors='A', row=0, rowspan=2)
    p.add_section('s', row=2, col=0, rowspan=10, colspan=12, title='Results')
    p.add_panel('a', row=0, col=0, rowspan=6, colspan=6, section='s')
    p.add_text_block('t', row=0, col=6, rowspan=6, colspan=6, section='s',
                     text='running text that should re-wrap when resized ' * 4)
    fig, axes = p.compose()
    axes['a'].plot([0, 1], [0, 1])
    return p, PanelEditor(fig, **kwargs)


def test_fit_with_no_floor_shows_the_whole_poster_within_the_viewport():
    _, e = _poster_editor(view_dpi='fit', fit_min_dpi=0)
    try:
        vw, vh = e._viewport_size()
        w, h = e._img_size
        assert w <= vw and h <= vh
        # ... and it uses the room it has, rather than showing a thumbnail
        assert max(w / vw, h / vh) > 0.5
    finally:
        e.root.destroy()


def test_fit_stops_at_the_floor_so_a_poster_stays_legible():
    """Fitting a whole A0 lands near 35 dpi, where nothing can be read."""
    _, e = _poster_editor(view_dpi='fit')
    try:
        assert e._fit_was_floored
        assert e._rendered_dpi == pytest.approx(e.fit_min_dpi)
        vw, vh = e._viewport_size()
        w, h = e._img_size
        assert w > vw or h > vh          # bigger than the window: it scrolls
    finally:
        e.root.destroy()


def test_a_print_sized_figure_is_not_floored(ed):
    """The floor must only bite on canvases too big to fit legibly."""
    ed._set_zoom('Fit')
    ed._sharpen_view()
    assert not ed._fit_was_floored
    vw, vh = ed._viewport_size()
    w, h = ed._img_size
    assert w <= vw and h <= vh


def test_fit_mode_is_the_default():
    _, e = _poster_editor()
    try:
        assert e._fit_mode
        assert e.zoom_var.get() == 'Fit'
    finally:
        e.root.destroy()


def test_an_explicit_dpi_turns_fit_off_and_is_used_verbatim():
    _, e = _poster_editor(view_dpi=25)
    try:
        assert not e._fit_mode
        assert e.view_dpi == pytest.approx(25)
        w, _ = e._img_size
        assert w == pytest.approx(118.9 / 2.54 * 25, abs=2)
    finally:
        e.root.destroy()


def test_a_reckless_dpi_is_clamped_rather_than_allocating_a_huge_image():
    _, e = _poster_editor(view_dpi=25)
    try:
        e._set_zoom(300)                  # ~7e7 px on A0, over the cap
        w, h = e._img_size
        assert w * h <= PanelEditor.MAX_BACKDROP_PX
        assert e.view_dpi < 300
        assert 'clamped' in e.zoom_lbl.cget('text')
    finally:
        e.root.destroy()


def test_the_canvas_is_a_scrollable_viewport_not_sized_to_the_backdrop():
    _, e = _poster_editor(view_dpi=50)
    try:
        w, h = e._img_size
        region = [float(v) for v in e.canvas.cget('scrollregion').split()]
        assert region == [0, 0, w, h]
        # the canvas widget itself must not have been grown to the image
        assert e.canvas.winfo_reqwidth() < w
    finally:
        e.root.destroy()


def test_zoom_steps_walk_the_ladder_in_both_directions():
    _, e = _poster_editor(view_dpi=50)
    try:
        e._step_zoom(+1)
        assert e.view_dpi == pytest.approx(75)
        e._step_zoom(-1)
        assert e.view_dpi == pytest.approx(50)
    finally:
        e.root.destroy()


def test_events_are_translated_through_the_scroll_offset():
    """Mouse events are viewport-relative; hit testing happens in image space."""
    _, e = _poster_editor(view_dpi=50)
    try:
        class E:
            def __init__(self, x, y):
                self.x, self.y, self.state = x, y, 0
        # With no scrolling the two spaces coincide ...
        assert e._event_xy(E(40, 30)) == (40, 30)
        # ... and _event_xy is what every handler uses, so scrolling cannot
        # desync the overlay from the pointer.
        assert e._event_xy(E(0, 0)) == (e.canvas.canvasx(0), e.canvas.canvasy(0))
    finally:
        e.root.destroy()


# ── live reflow ────────────────────────────────────────────────────────────

def test_resizing_a_text_block_reflows_it_in_the_editor():
    _, e = _poster_editor(view_dpi=30)
    try:
        item = next(i for i in e.items if getattr(i, 'role', None) == 'panel'
                    and i.parent_label == 't')
        ax = item.ax
        before = ax.texts[0].get_text()
        pos = ax.get_position()
        item._set_bounds([pos.x0, pos.y0, pos.width / 3, pos.height])
        e._select(item)
        e._after_change('Moved')
        assert ax.texts[0].get_text() != before
        assert 'reflowed' in e.status.cget('text')
    finally:
        e.root.destroy()


def test_reflow_can_be_switched_off():
    _, e = _poster_editor(view_dpi=30)
    try:
        item = next(i for i in e.items if getattr(i, 'role', None) == 'panel'
                    and i.parent_label == 't')
        ax = item.ax
        before = ax.texts[0].get_text()
        e.reflow_var.set(False)
        pos = ax.get_position()
        item._set_bounds([pos.x0, pos.y0, pos.width / 3, pos.height])
        e._select(item)
        e._after_change('Moved')
        assert ax.texts[0].get_text() == before
    finally:
        e.root.destroy()


def _boom(ax):
    raise RuntimeError('nope')


def test_a_failing_hook_reports_but_does_not_kill_the_editor():
    from sciplotlib.compose import set_redraw_hook
    _, e = _poster_editor(view_dpi=30)
    try:
        item = next(i for i in e.items if getattr(i, 'role', None) == 'panel'
                    and i.parent_label == 'a')
        set_redraw_hook(item.ax, _boom)
        e._select(item)
        e._after_change('Moved')          # must not raise
        assert 'failed' in e.status.cget('text').lower()
    finally:
        e.root.destroy()


# ── editing text content ────────────────────────────────────────────────────
# The panel editor moves elements; it also edits the *words* of a free-text
# artist (headings, labels) and writes the new string into the overrides file
# so it survives a re-render.

def _hello(ed):
    """The 'hello' text placed in panel a by the fixture."""
    return next(i for i in ed.items
                if getattr(i, 'is_text', False)
                and i.artist.get_text() == 'hello')


def test_selecting_a_text_item_populates_the_text_box(ed):
    t = _hello(ed)
    ed._select(t)
    assert ed.text_edit.get('1.0', 'end-1c') == 'hello'
    assert ed._text_target is t


def test_a_panel_selection_disables_the_text_box(ed):
    panel = next(i for i in ed.items if getattr(i, 'role', None) == 'panel')
    ed._select(panel)
    assert ed._text_target is None
    assert str(ed.text_edit.cget('state')) == 'disabled'


def test_apply_text_changes_the_artist_and_marks_it_moved(ed):
    t = _hello(ed)
    ed._select(t)
    ed.text_edit.delete('1.0', 'end')
    ed.text_edit.insert('1.0', 'goodbye')
    ed._apply_text()
    assert t.artist.get_text() == 'goodbye'
    assert t.text_changed and t.moved


def test_editing_text_then_undo_restores_the_old_words(ed):
    t = _hello(ed)
    ed._select(t)
    ed.text_edit.delete('1.0', 'end')
    ed.text_edit.insert('1.0', 'goodbye')
    ed._apply_text()
    ed._undo()
    assert t.artist.get_text() == 'hello'
    assert not t.text_changed


def test_multiline_text_is_kept_without_a_trailing_newline(ed):
    t = _hello(ed)
    ed._select(t)
    ed.text_edit.delete('1.0', 'end')
    ed.text_edit.insert('1.0', 'line one\nline two')
    ed._apply_text()
    assert t.artist.get_text() == 'line one\nline two'


def test_edited_text_is_saved_into_the_overrides_file(ed, tmp_path):
    path = tmp_path / 'o.json'
    ed.overrides_path = str(path)
    t = _hello(ed)
    ed._select(t)
    ed.text_edit.delete('1.0', 'end')
    ed.text_edit.insert('1.0', 'goodbye')
    ed._apply_text()
    ed._save()
    data = ov.read_overrides(path)
    entry = next(v for v in data.values()
                 if v.get('kind') == 'text' and v.get('text') == 'goodbye')
    assert entry['fingerprint'] == 'hello'      # resolves the un-edited artist


def test_saved_text_override_reapplies_to_a_fresh_figure(ed, tmp_path):
    path = tmp_path / 'o.json'
    ed.overrides_path = str(path)
    t = _hello(ed)
    ed._select(t)
    ed.text_edit.delete('1.0', 'end')
    ed.text_edit.insert('1.0', 'goodbye')
    ed._apply_text()
    ed._save()
    # A brand-new figure (text still 'hello' from code) should pick up the edit.
    _, fig2, axes2 = _fig()
    n, warnings = ov.apply_overrides(fig2, str(path), verbose=False)
    hits = [x for x in axes2['a'].texts if x.get_text() == 'goodbye']
    assert len(hits) == 1


def test_double_click_on_text_focuses_the_editor(ed):
    t = _hello(ed)
    box = ed._item_box(t)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    class _E:
        def __init__(self, x, y):
            self.x, self.y, self.state, self.num = x, y, 0, 1
    ed._on_double(_E(cx, cy))
    assert ed._text_target is t


def test_resize_preview_keeps_a_backdrop_without_a_full_render():
    _, e = _poster_editor(view_dpi='fit')
    try:
        assert e._backdrop_flat is not None      # cached by the first render
        before = e._rendered_dpi
        e._preview_rescale()                     # cheap path: no Agg redraw
        assert e._rendered_dpi == before         # dpi cache untouched
        assert e._photo is not None              # something is on screen
    finally:
        e.root.destroy()


def test_resize_preview_keeps_click_coordinates_accurate():
    """After a preview-only resize, a click still resolves to the right item.

    The preview folds its scale into the coordinate maps, so hit-testing on the
    stretched image lands where the eye expects without a full re-render.
    """
    _, e = _poster_editor(view_dpi='fit')
    try:
        panel = next(i for i in e.items if getattr(i, 'role', None) == 'panel'
                     and i.parent_label == 'a')
        box = e._item_box(panel)                     # canvas coords, scale 1
        e._view_scale = 1.0
        # Pretend the window grew: rescale the cached backdrop bigger.
        e._backdrop_flat = e._backdrop_flat.resize(
            (e._img_size[0] * 2, e._img_size[1] * 2))
        e._img_size = (e._img_size[0] * 2, e._img_size[1] * 2)
        e._view_scale = 2.0
        box2 = e._item_box(panel)                    # canvas coords at scale 2
        # The box should have scaled with the image, so its centre still lands
        # on the panel — the hit test agrees.
        cx = (box2[0] + box2[2]) / 2
        cy = (box2[1] + box2[3]) / 2
        assert e._hit_test(cx, cy) is not None
        # And roughly twice the size/offset of the un-scaled box.
        assert box2[2] == pytest.approx(box[2] * 2, rel=0.02)
    finally:
        e.root.destroy()


# ── a plain click must be free (the "can't click / stuck" fix) ───────────────
# On a heavy poster a full backdrop render is ~2 s. A click that only selects
# must not pay that cost, leave an Undo step, or mark the element moved.

class _Click:
    def __init__(self, x, y, state=0):
        self.x, self.y, self.state, self.num = x, y, state, 1


def _panel_centre(ed, label='a'):
    panel = next(i for i in ed.items if getattr(i, 'role', None) == 'panel'
                 and i.parent_label == label)
    box = ed._item_box(panel)
    return panel, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def test_a_plain_click_triggers_no_render(ed, monkeypatch):
    panel, cx, cy = _panel_centre(ed)
    calls = {'n': 0}
    monkeypatch.setattr(ed, '_render_backdrop', lambda: calls.__setitem__('n', calls['n'] + 1))
    ed._on_press(_Click(cx, cy))
    ed._on_release(_Click(cx, cy))          # released without moving
    assert calls['n'] == 0                  # no expensive redraw
    assert not panel.moved                  # not recorded as a change
    assert panel is ed.selected            # but it did get selected


def test_a_plain_click_leaves_nothing_to_undo(ed):
    _, cx, cy = _panel_centre(ed)
    before = len(ed._history)
    ed._on_press(_Click(cx, cy))
    ed._on_release(_Click(cx, cy))
    assert len(ed._history) == before       # no Undo step from a mere click


def test_a_real_drag_repaints_incrementally_not_with_a_full_render(ed, monkeypatch):
    panel, cx, cy = _panel_centre(ed)
    full = {'n': 0}
    blits = {'n': 0}
    monkeypatch.setattr(ed, '_render_backdrop', lambda: full.__setitem__('n', full['n'] + 1))
    real_blit = ed._blit_change
    monkeypatch.setattr(ed, '_blit_change',
                        lambda items, **kw: (blits.__setitem__('n', blits['n'] + 1),
                                             real_blit(items, **kw))[1])
    ed._on_press(_Click(cx, cy))
    ed._on_drag(_Click(cx + 40, cy))        # actually move
    ed._on_release(_Click(cx + 40, cy))
    assert panel.moved
    assert blits['n'] == 1                  # one repaint, on release
    assert full['n'] == 0                   # and it did *not* redraw the figure


# ── incremental redraw ───────────────────────────────────────────────────────
# A full Agg draw of a poster is ~0.8 s. Moving an element must repaint only
# that element over a cached "clean plate" of everything else.


def _a_text(ed):
    return next(i for i in ed.items if getattr(i, 'is_text', False))


def _a_panel(ed):
    return next(i for i in ed.items if getattr(i, 'role', None) == 'panel')


def test_blit_matches_a_full_render(ed):
    """The cheap path must show what the expensive path would have shown."""
    txt = _a_text(ed)
    ed._set_selection([txt])
    txt.nudge(20, 15)
    assert ed._blit_change([txt])
    blit = np.asarray(ed._backdrop_flat, dtype=float)

    ed._plate_cache.clear()
    ed._invalidate_render_cache()
    ed._render_backdrop()
    full = np.asarray(ed._backdrop_flat, dtype=float)

    assert blit.shape == full.shape
    # Not bit-identical: blitted elements are composited on top, so z-order
    # against a neighbour can differ. It must still be the same picture.
    assert np.abs(blit - full).mean() < 2.0
    assert (np.abs(blit - full).max(axis=2) > 8).mean() < 0.05


def test_one_plate_serves_every_text(ed):
    """Moving a second text must not pay for another full draw."""
    texts = [i for i in ed.items if getattr(i, 'is_text', False)]
    if len(texts) < 2:
        pytest.skip('fixture has only one text')
    draws = {'n': 0}
    real = ed._agg.draw
    ed._agg.draw = lambda *a, **k: (draws.__setitem__('n', draws['n'] + 1), real())[1]
    try:
        for t in texts[:2]:
            ed._set_selection([t])
            t.nudge(3, 3)
            assert ed._blit_change([t])
        assert draws['n'] == 1          # one plate built, reused for both
    finally:
        ed._agg.draw = real


def test_moving_a_panel_invalidates_the_text_plate(ed):
    """A plate is only valid while nothing outside its hidden set has moved."""
    txt, panel = _a_text(ed), _a_panel(ed)
    ed._set_selection([txt])
    txt.nudge(2, 2)
    ed._blit_change([txt])
    assert ed._plate_cache
    text_key = next(iter(ed._plate_cache))
    assert id(panel.artist) not in text_key[0]   # the panel is *in* that plate

    ed._set_selection([panel])
    panel.nudge(2, 2)
    ed._blit_change([panel])
    # That plate held the panel at its old position, so it cannot be reused.
    assert len(ed._plate_cache) == 1
    key = next(iter(ed._plate_cache))
    assert key != text_key
    assert id(panel.artist) in key[0]            # now the panel is hidden too


def test_a_reflow_of_an_unhidden_axes_forces_a_full_render(ed, monkeypatch):
    """A redraw hook rebuilds artists the plate still holds at their old state."""
    txt = _a_text(ed)
    # Moving a text does not make its panel hot, so the plate keeps that panel
    # -- including whatever the hook has just replaced inside it.
    assert not ed._blit_change([txt], must_hide=[txt.ax])

    monkeypatch.setattr(ed, '_reflow', lambda item=None: 1)
    full = {'n': 0}
    monkeypatch.setattr(ed, '_render_backdrop', lambda: full.__setitem__('n', full['n'] + 1))
    ed._set_selection([txt])
    txt.nudge(2, 2)
    ed._after_change('moved')
    assert full['n'] == 1


def test_reset_all_does_not_blit(ed, monkeypatch):
    """Everything moved at once, so no plate describes the new figure."""
    monkeypatch.setattr(ed, '_blit_change',
                        lambda items, **kw: pytest.fail('must not blit'))
    ed._reset_all()


def test_editing_text_repaints_incrementally(ed, monkeypatch):
    txt = _a_text(ed)
    ed._set_selection([txt])
    monkeypatch.setattr(ed, '_render_backdrop', lambda: pytest.fail('must not redraw all'))
    ed.text_edit.delete('1.0', 'end')
    ed.text_edit.insert('1.0', 'a much longer replacement string')
    ed._apply_text()
    assert txt.artist.get_text() == 'a much longer replacement string'


def test_item_boxes_are_cached_but_follow_a_move(ed):
    """The cache is what makes hover cheap; it must never go stale."""
    panel = _a_panel(ed)
    box = ed._item_box(panel)
    calls = {'n': 0}
    real = panel.bbox_display
    panel.bbox_display = lambda r: (calls.__setitem__('n', calls['n'] + 1), real(r))[1]
    try:
        for _ in range(5):
            ed._item_box(panel)
        assert calls['n'] == 0                 # served from the cache
        ed._set_selection([panel])
        panel.nudge(30, 0)
        ed._after_change('moved')
        moved = ed._item_box(panel)
        assert calls['n'] > 0                  # recomputed after the move
        assert moved[0] > box[0] + 5           # and it actually followed
    finally:
        panel.bbox_display = real


def test_flat_image_is_rgb_and_detached_from_the_agg_buffer(ed):
    """Agg reuses its buffer, so a retained image has to be a real copy."""
    ed._agg.draw()
    first = ed._flat_image()
    assert first.mode == 'RGB'
    before = np.asarray(first).copy()
    _a_panel(ed).nudge(60, 40)
    ed._agg.draw()                             # overwrites the Agg buffer
    assert np.array_equal(np.asarray(first), before)


def test_detached_artists_are_skipped(ed):
    """A redraw hook can replace the artists the editor still holds."""
    orphan = _a_text(ed)                        # a text the panel owns
    assert orphan.artist in ed._leaf_artists()
    orphan.artist.remove()                      # as a redraw hook would
    assert orphan.artist not in ed._leaf_artists()

    panel = _a_panel(ed)
    ed._set_selection([panel])
    panel.nudge(4, 4)
    assert ed._blit_change([panel])             # still repaints, no exception


def test_a_hooked_panel_still_blits_when_the_plate_hides_it(ed):
    """Rebuilt contents are fine as long as that axes is redrawn anyway."""
    panel = _a_panel(ed)
    ed._set_selection([panel])
    panel.nudge(3, 3)
    ed._blit_change([panel])                    # makes it hot
    assert ed._blit_change([panel], must_hide=[panel.artist])
    # An axes the plate does *not* hide must force the full render instead.
    other = next(i for i in ed.items
                 if getattr(i, 'role', None) == 'panel' and i is not panel)
    assert not ed._blit_change([panel], must_hide=[other.artist])


def test_an_expensive_axes_does_not_stay_hot(ed):
    """One slow panel must not tax every later edit."""
    a, b = _panels(ed)
    ed._set_selection([a])
    a.nudge(2, 2)
    ed._blit_change([a])
    ed._draw_cost[id(a.artist)] = 10 * ed.HOT_AXES_BUDGET   # pretend it is slow
    ed._set_selection([b])
    b.nudge(2, 2)
    ed._blit_change([b])
    assert a not in ed._hot_axes                # dropped: it blew the budget
    assert b in ed._hot_axes


# ── event-layer regressions ─────────────────────────────────────────────────
# These two froze the editor outright, and no direct-call test could see them:
# both live in how tk *delivers* events, not in the handlers.


def test_a_tree_selection_does_not_sync_back_to_the_tree(ed):
    """<<TreeviewSelect>> is queued and delivered late.

    A flag raised around ``selection_set`` is down again by then, so the only
    thing that can break the canvas->tree->canvas cycle is the handler not
    writing back.
    """
    a, b = _panels(ed)
    ed._set_selection([a])                      # canvas selection syncs the tree
    # Point the tree at a *different* row, so the handler has real work to do
    # and cannot simply no-op its way out.
    node = next(n for n, it in ed._tree_items.items() if it is b)
    ed.tree.selection_set([node])
    calls = {'n': 0}
    real = ed._sync_tree_to_selection
    ed._sync_tree_to_selection = lambda: (calls.__setitem__('n', calls['n'] + 1),
                                          real())[1]
    ed._on_tree_select()                        # the queued event, delivered late
    assert ed.selected is b                     # it did act on the tree's choice
    assert calls['n'] == 0                      # but did not write it back


def test_syncing_a_tree_already_in_sync_queues_no_event(ed):
    panel = _a_panel(ed)
    ed._set_selection([panel])
    sets = {'n': 0}
    real = ed.tree.selection_set
    ed.tree.selection_set = lambda *a: (sets.__setitem__('n', sets['n'] + 1),
                                        real(*a))[1]
    try:
        ed._sync_tree_to_selection()
        assert sets['n'] == 0
    finally:
        ed.tree.selection_set = real


def test_a_canvas_selection_settles_instead_of_looping(ed):
    """Re-delivering the queued tree event must not keep re-selecting."""
    panel = _a_panel(ed)
    ed._set_selection([panel])
    seen = {'n': 0}
    real = ed._set_selection
    ed._set_selection = lambda items, **kw: (seen.__setitem__('n', seen['n'] + 1),
                                             real(items, **kw))[1]
    for _ in range(20):
        ed._on_tree_select()      # stands in for the queued <<TreeviewSelect>>
    assert seen['n'] == 0         # each one used to queue another: a CPU spin


def test_a_double_click_still_selects_and_can_drag(ed):
    """Tk sends <Double-Button-1> *instead of* <Button-1> for the second click.

    Swallowing it lost the press: click to select, click again to drag, and
    nothing moved.
    """
    panel, cx, cy = _panel_centre(ed)
    before = panel.pos().copy()
    ed._on_double(_Click(cx, cy))
    assert ed.selected is panel
    assert ed._drag                             # a drag is armed, not swallowed
    ed._on_drag(_Click(cx + 30, cy))
    ed._on_release(_Click(cx + 30, cy))
    assert panel.pos()[0] > before[0]


def test_a_double_click_on_text_opens_the_box_and_still_arms_a_drag(ed):
    txt = _a_text(ed)
    box = ed._item_box(txt)
    ed._on_double(_Click((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
    assert ed.selected is txt
    assert ed._text_target is txt
    assert ed._drag


def test_a_typed_position_is_undoable_and_saved(ed, tmp_path):
    """Undo and Save both read the item's own history, not the editor's."""
    path = tmp_path / 'o.json'
    ed.overrides_path = str(path)
    panel = next(i for i in ed.items if getattr(i, 'role', None) == 'panel'
                 and i.parent_label == 'a')
    ed._select(panel)
    start = panel.pos().copy()
    ed.fields['x0'][0].set('0.2000')
    ed._apply_fields()
    assert panel.pos()[0] == pytest.approx(0.20, abs=1e-6)
    assert panel.moved                       # the change is visible to Save
    ed._save()
    assert 'panel:a' in ov.read_overrides(path)
    ed._undo()
    assert panel.pos() == pytest.approx(start)


def test_retyping_the_same_numbers_is_not_a_change(ed):
    panel = _a_panel(ed)
    ed._select(panel)
    ed._apply_fields()                       # fields still hold its own values
    assert not panel.moved
    assert not ed._history


# ── the child must render what the parent composed ──────────────────────────


def test_rc_snapshot_round_trips_through_apply():
    """Font resolution is the point: the family on an artist is generic."""
    import matplotlib
    from sciplotlib.panel_editor import _rc_snapshot, _apply_rc

    with matplotlib.rc_context({'font.sans-serif': ['Nimbus Sans'],
                                'font.family': 'sans-serif'}):
        snap = _rc_snapshot()
    assert snap['font.sans-serif'] == ['Nimbus Sans']
    assert 'backend' not in snap              # the child pins its own

    # _apply_rc writes the *whole* snapshot to the global rcParams, which is
    # the point of it — so it has to be contained, or it leaks into every
    # later test in the session.
    with matplotlib.rc_context():
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        _apply_rc(snap)
        assert matplotlib.rcParams['font.sans-serif'] == ['Nimbus Sans']


def test_apply_rc_survives_a_key_this_matplotlib_rejects():
    import matplotlib
    from sciplotlib.panel_editor import _apply_rc

    with matplotlib.rc_context():
        _apply_rc({'a.key.that.never.existed': 1,
                   'font.sans-serif': ['Nimbus Sans']})
        assert matplotlib.rcParams['font.sans-serif'] == ['Nimbus Sans']


def test_the_editor_subprocess_is_sent_the_parents_rcparams(tmp_path, monkeypatch):
    """Without this the child lays text out in a different, wider font."""
    import matplotlib
    import subprocess as _sp
    from sciplotlib import panel_editor as pe

    seen = {}

    def fake_run(cmd, env=None, **kw):
        seen['pkl'] = cmd[3]
        with open(seen['pkl'], 'rb') as fh:
            seen['payload'] = pickle.load(fh)

        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(_sp, 'run', fake_run)
    c, fig, axes = _fig()
    with matplotlib.rc_context({'font.sans-serif': ['Nimbus Sans']}):
        pe.launch_panel_editor(fig, overrides_path=str(tmp_path / 'o.json'))

    payload = seen['payload']
    assert isinstance(payload, dict) and 'figure' in payload
    assert payload['rcParams']['font.sans-serif'] == ['Nimbus Sans']


def test_a_resize_queues_a_crisp_redraw(ed):
    """Resize shows an instant rescale; without a follow-up it stays blurred."""
    ed._fit_mode = True
    ed._sharpen_after = None
    ed._on_canvas_resize()
    assert ed._sharpen_after is not None
    ed.root.after_cancel(ed._sharpen_after)
    ed._sharpen_after = None


def test_a_typed_dpi_is_accepted(ed):
    """The zoom box is editable, so any dpi can be asked for."""
    ed.zoom_var.set('87')
    ed._on_zoom()
    assert not ed._fit_mode
    assert ed.view_dpi == pytest.approx(87)
    if ed._sharpen_after is not None:
        ed.root.after_cancel(ed._sharpen_after)
        ed._sharpen_after = None


def test_a_nonsense_zoom_is_refused_without_changing_the_view(ed):
    before = ed.view_dpi
    ed.zoom_var.set('wide please')
    ed._on_zoom()
    assert ed.view_dpi == before
    ed.zoom_var.set('-40')
    ed._on_zoom()
    assert ed.view_dpi == before


# ── delete ─────────────────────────────────────────────────────────────────

def _text_item(e):
    return next(it for it in e.items
                if getattr(it, 'is_text', False) and it.artist.get_text())


def test_delete_hides_the_selection_and_undo_restores_it():
    _, e = _poster_editor()
    try:
        item = _text_item(e)
        e._set_selection([item])
        e._delete_selected()
        assert item.hidden and item.deleted and item.moved
        e._undo()
        assert not item.hidden and not item.deleted
    finally:
        e.root.destroy()


def test_delete_writes_a_deleted_flag_and_restore_clears_it(tmp_path):
    path = tmp_path / 'ov.json'
    _, e = _poster_editor(overrides_path=str(path))
    try:
        item = _text_item(e)
        e._set_selection([item])
        e._delete_selected()
        e._save()
        data = json.loads(path.read_text())
        entry = data[item.override_address]
        assert entry['deleted'] is True

        e._undelete_selected()
        assert not item.deleted
        # a restored item is no longer "moved", so _save stops writing it
        assert not item.moved
    finally:
        e.root.destroy()


def test_reset_all_restores_a_deleted_element():
    _, e = _poster_editor()
    try:
        item = _text_item(e)
        e._set_selection([item])
        e._delete_selected()
        assert item.hidden
        e._reset_all()
        assert not item.hidden
    finally:
        e.root.destroy()


def test_tiny_artist_can_be_clicked():
    """A 4x5 px arrow must be grabbable, not just theoretically selectable."""
    import matplotlib.pyplot as plt
    _, e = _poster_editor()
    try:
        ax = e.fig.get_axes()[0]
        ax.annotate('', xy=(0.50, 0.52), xytext=(0.50, 0.50),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle='-|>', lw=1))
        e.items = __import__('sciplotlib.drag_editor', fromlist=['x']).collect_items(e.fig)
        e.fig.canvas.draw()
        item = next(it for it in e.items if it.label == 'Arrow')
        box = e._item_box(item)
        assert box is not None
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        # a click a few pixels off centre still lands on it
        assert e._hit_test(cx + 4, cy) is item
    finally:
        e.root.destroy()
