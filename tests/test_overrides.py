"""Tests for tagless position overrides (sciplotlib.overrides).

Covers the addressing scheme, fingerprint-before-index resolution, and the
deferred-panel round trip: a panel nudge must survive fit_axes_to_cells(),
which re-places every panel axes and would otherwise erase it.
"""
import json

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

from sciplotlib import overrides as ov
from sciplotlib.compose import FigureComposer, place_image


IMG = np.zeros((4, 6, 4), dtype=float)


def _composer():
    c = FigureComposer(width_cm=10, height_cm=6, grid_rows=2, grid_cols=2, dpi=100)
    c.add_panel('a', 0, 0, 1, 1)
    c.add_panel('b', 0, 1, 1, 1)
    c.add_panel('c', 1, 0, 1, 2)
    return c


def _draw(c):
    fig, axes = c.compose()
    axes['a'].plot([0, 1], [0, 1])
    axes['a'].set_xlabel('trials')
    axes['a'].set_ylabel('reward rate')
    axes['a'].text(0.2, 0.8, 'MP computer')
    axes['a'].text(0.5, 0.5, 'bandit')
    place_image(axes['a'], IMG, 0.5, 0.2, zoom=0.5)
    axes['b'].plot([0, 1], [1, 0])
    axes['c'].plot([0, 1], [0, 1])
    return fig, axes


# ── addressing ─────────────────────────────────────────────────────────────

def test_addresses_cover_panels_labels_text_and_images():
    c = _composer()
    fig, axes = _draw(c)
    addrs = {o['address']: o['kind'] for o in ov.iter_overridable(fig)}

    assert addrs['panel:a'] == 'panel'
    assert addrs['panel:b'] == 'panel'
    assert addrs['panel:a/xlabel'] == 'xlabel'
    assert addrs['panel:a/ylabel'] == 'ylabel'
    # two texts -> indexed; one image -> unindexed
    assert addrs['panel:a/text:0'] == 'text'
    assert addrs['panel:a/text:1'] == 'text'
    assert addrs['panel:a/image'] == 'image'


def test_single_artist_addresses_are_unindexed():
    c = _composer()
    fig, axes = _draw(c)
    axes['b'].text(0.1, 0.1, 'only one')
    addrs = {o['address'] for o in ov.iter_overridable(fig)}
    assert 'panel:b/text' in addrs
    assert 'panel:b/text:0' not in addrs


def test_panel_letters_are_not_addressable():
    """Panel letters live in fig.texts, so they must not be picked up as
    panel content (moving them is the composer's job, not an override's)."""
    c = _composer()
    fig, axes = _draw(c)
    texts = [o for o in ov.iter_overridable(fig) if o['kind'] == 'text']
    assert {t['fingerprint'] for t in texts} == {'MP computer', 'bandit'}


# ── resolution ─────────────────────────────────────────────────────────────

def test_text_resolves_by_fingerprint_not_index():
    """Inserting a text ahead of the target must not shift the override onto
    the wrong artist — the fingerprint wins over the recorded index."""
    c = _composer()
    fig, axes = _draw(c)
    ax = axes['a']
    # override was recorded when 'bandit' was at index 1
    ax.texts[0].remove()          # now 'bandit' sits at index 0
    kind, target, warn = ov.resolve(fig, 'panel:a/text:1', fingerprint='bandit')
    assert kind == 'text'
    assert target.get_text() == 'bandit'
    assert warn is None


def test_text_falls_back_to_index_and_warns_when_edited():
    c = _composer()
    fig, axes = _draw(c)
    kind, target, warn = ov.resolve(fig, 'panel:a/text:1',
                                    fingerprint='text that no longer exists')
    assert kind == 'text'
    assert target.get_text() == 'bandit'   # index 1
    assert 'fell back to index' in warn


def test_resolve_reports_missing_panel():
    c = _composer()
    fig, axes = _draw(c)
    kind, target, warn = ov.resolve(fig, 'panel:zzz/xlabel')
    assert target is None
    assert 'no panel' in warn


# ── application ────────────────────────────────────────────────────────────

def test_image_position_and_zoom_round_trip(tmp_path):
    c = _composer()
    fig, axes = _draw(c)
    ab = ov._images(axes['a'])[0]

    path = tmp_path / 'o.json'
    ov.write_overrides(path, {
        'panel:a/image': {'kind': 'image', 'value': [0.75, 0.25], 'zoom': 0.9,
                          'fingerprint': ov.image_fingerprint(ab)},
    })
    applied, warns = ov.apply_overrides(fig, path, verbose=False)

    assert applied == 1 and warns == []
    assert tuple(ab.xy) == (0.75, 0.25)
    assert ov.image_zoom(ab) == pytest.approx(0.9)


def test_text_position_round_trip(tmp_path):
    c = _composer()
    fig, axes = _draw(c)
    path = tmp_path / 'o.json'
    ov.write_overrides(path, {
        'panel:a/text:0': {'kind': 'text', 'value': [0.11, 0.22],
                           'fingerprint': 'MP computer'},
    })
    ov.apply_overrides(fig, path, verbose=False)
    assert axes['a'].texts[0].get_position() == (0.11, 0.22)


# ── the deferred-panel round trip ──────────────────────────────────────────

def test_panel_delta_survives_fit_axes_to_cells(tmp_path):
    """The whole point: fit_axes_to_cells() re-places panels, so a panel
    override applied before it would be wiped. It must land after."""
    c = _composer()
    fig, axes = _draw(c)
    c.fit_axes_to_cells()
    fitted = axes['a'].get_position().bounds

    path = tmp_path / 'o.json'
    ov.write_overrides(path, {
        'panel:a': {'kind': 'panel',
                    'value': list(fitted),
                    'delta': [0.05, -0.02, 0.0, 0.0]},
    })
    c.apply_overrides(path, verbose=False)

    # apply_overrides must NOT have moved it yet (fit would undo that)
    assert axes['a'].get_position().bounds == pytest.approx(fitted)

    c.fit_axes_to_cells()   # re-fit, then replay the deferred panel delta
    moved = axes['a'].get_position().bounds
    assert moved[0] == pytest.approx(fitted[0] + 0.05)
    assert moved[1] == pytest.approx(fitted[1] - 0.02)
    assert moved[2:] == pytest.approx(fitted[2:])


def test_repeated_fit_does_not_compound_the_delta(tmp_path):
    """Calling save()/to_image() twice must not drift the panel further each
    time — fit rewinds to its recorded baseline before re-measuring."""
    c = _composer()
    fig, axes = _draw(c)
    c.fit_axes_to_cells()
    fitted = axes['a'].get_position().bounds

    path = tmp_path / 'o.json'
    ov.write_overrides(path, {
        'panel:a': {'kind': 'panel', 'value': list(fitted),
                    'delta': [0.05, 0.0, 0.0, 0.0]},
    })
    c.apply_overrides(path, verbose=False)

    c.fit_axes_to_cells()
    once = axes['a'].get_position().bounds
    c.fit_axes_to_cells()
    twice = axes['a'].get_position().bounds
    c.fit_axes_to_cells()
    thrice = axes['a'].get_position().bounds

    assert once == pytest.approx(twice)
    assert twice == pytest.approx(thrice)
    assert once[0] == pytest.approx(fitted[0] + 0.05)


def test_panel_resize_delta_applies():
    c = _composer()
    fig, axes = _draw(c)
    c.fit_axes_to_cells()
    fitted = axes['b'].get_position().bounds
    ov.apply_value('panel', axes['b'], list(fitted),
                   {'kind': 'panel', 'delta': [0.0, 0.0, -0.03, 0.04]})
    got = axes['b'].get_position().bounds
    assert got[2] == pytest.approx(fitted[2] - 0.03)
    assert got[3] == pytest.approx(fitted[3] + 0.04)


def test_panel_base_is_stamped_for_the_editor():
    """The editor needs the pre-override position so a second round of nudging
    reports a total delta, not an increment."""
    c = _composer()
    fig, axes = _draw(c)
    c.fit_axes_to_cells()
    fitted = axes['a'].get_position().bounds
    ov.apply_value('panel', axes['a'], list(fitted),
                   {'kind': 'panel', 'delta': [0.05, 0.0, 0.0, 0.0]})
    assert axes['a']._sciplotlib_panel_base == pytest.approx(fitted)


def test_absolute_mode_ignores_delta():
    c = _composer()
    fig, axes = _draw(c)
    c.fit_axes_to_cells()
    ov.apply_value('panel', axes['a'], [0.1, 0.2, 0.3, 0.4],
                   {'kind': 'panel', 'mode': 'absolute',
                    'delta': [9.0, 9.0, 9.0, 9.0]})
    assert axes['a'].get_position().bounds == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_apply_overrides_kind_filtering(tmp_path):
    c = _composer()
    fig, axes = _draw(c)
    path = tmp_path / 'o.json'
    ov.write_overrides(path, {
        'panel:a': {'kind': 'panel', 'value': [0, 0, 1, 1], 'delta': [0.1, 0, 0, 0]},
        'panel:a/text:0': {'kind': 'text', 'value': [0.9, 0.9],
                           'fingerprint': 'MP computer'},
    })
    n, _ = ov.apply_overrides(fig, path, verbose=False, skip_kinds=ov.DEFERRED_KINDS)
    assert n == 1
    assert axes['a'].texts[0].get_position() == (0.9, 0.9)

    n, _ = ov.apply_overrides(fig, path, verbose=False, kinds=ov.DEFERRED_KINDS)
    assert n == 1


def test_unknown_address_is_skipped_not_raised(tmp_path):
    c = _composer()
    fig, axes = _draw(c)
    path = tmp_path / 'o.json'
    ov.write_overrides(path, {
        'panel:nope/xlabel': {'kind': 'xlabel', 'value': [0, 0]},
    })
    n, warns = ov.apply_overrides(fig, path, verbose=False)
    assert n == 0
    assert len(warns) == 1


def test_overrides_as_code_emits_every_kind():
    code = ov.overrides_as_code({
        'panel:a': {'kind': 'panel', 'value': [0.1, 0.2, 0.3, 0.4]},
        'panel:a/xlabel': {'kind': 'xlabel', 'value': [0.5, -0.1]},
        'panel:a/image': {'kind': 'image', 'value': [1.0, 2.0], 'zoom': 0.3},
        'panel:a/text:0': {'kind': 'text', 'value': [0.1, 0.1],
                           'fingerprint': 'hello'},
        'panel:j/colorbar': {'kind': 'axes', 'value': [0.1, 0.2, 0.3, 0.4]},
    })
    assert 'set_position([0.1000, 0.2000, 0.3000, 0.4000])' in code
    assert 'set_label_coords(0.5000, -0.1000)' in code
    assert 'set_zoom(0.3000)' in code
    assert "texts[0].set_position((0.1000, 0.1000))" in code
    assert '_colorbar' in code
