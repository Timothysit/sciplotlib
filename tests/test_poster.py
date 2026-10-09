"""Tests for poster composition (sciplotlib.poster).

Covers the geometry that is easy to get subtly wrong -- section body
arithmetic, section-relative panel placement, metric-based text wrapping --
and the two interactions with FigureComposer that a poster depends on:
poster chrome must survive ``normalize_fonts`` (section titles are meant to be
bigger than body text) and text blocks must survive ``fit_axes_to_cells``
(text positioned in axes fractions must not be rescaled).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches
import numpy as np
import pytest

import pickle

from sciplotlib.compose import (NO_FONT_NORMALIZE, REDRAW_HOOK_ATTR,
                                exempt_from_linewidth_normalization,
                                run_redraw_hook, set_redraw_hook)
from sciplotlib.poster import (
    POSTER_THEMES,
    PosterComposer,
    REFERENCE_WIDTH_CM,
    draw_text_block,
    is_dark,
    relative_luminance,
    wrap_text_to_width,
)


def _poster(**kwargs):
    kwargs.setdefault('grid_rows', 24)
    kwargs.setdefault('grid_cols', 24)
    kwargs.setdefault('dpi', 60)
    return PosterComposer(**kwargs)


# ── canvas and scale ───────────────────────────────────────────────────────

def test_a0_landscape_is_the_default_and_sets_type_scale_to_one():
    p = _poster()
    assert (p.width_cm, p.height_cm) == (118.9, 84.1)
    assert p.type_scale == pytest.approx(1.0)
    assert p.font_size == pytest.approx(20.0)


def test_type_scale_follows_width_so_a_smaller_poster_gets_smaller_type():
    p = _poster(paper='a1_landscape')
    assert p.type_scale == pytest.approx(84.1 / REFERENCE_WIDTH_CM)
    assert p.font_size == pytest.approx(20.0 * p.type_scale)
    assert p.spine_linewidth == pytest.approx(1.8 * p.type_scale)


def test_explicit_type_scale_and_explicit_sizes_both_win():
    p = _poster(type_scale=2.0)
    assert p.font_size == pytest.approx(40.0)
    p2 = _poster(type_scale=2.0, font_size=11.0)
    assert p2.font_size == pytest.approx(11.0)   # explicit beats the scale


def test_custom_size_bypasses_the_paper_table():
    p = _poster(width_cm=60.0, height_cm=40.0)
    assert (p.width_cm, p.height_cm) == (60.0, 40.0)


def test_unknown_paper_size_is_an_error_naming_the_known_ones():
    with pytest.raises(ValueError, match='a0_landscape'):
        _poster(paper='a0_diagonal')


# ── themes ─────────────────────────────────────────────────────────────────

def test_theme_dict_is_merged_over_light_so_partial_overrides_work():
    p = _poster(theme={'background': '#ff0000'})
    assert p.theme['background'] == '#ff0000'
    assert p.theme['section_edge'] == POSTER_THEMES['light']['section_edge']


def test_unknown_theme_is_an_error():
    with pytest.raises(ValueError, match='unknown theme'):
        _poster(theme='chartreuse')


# ── text wrapping ──────────────────────────────────────────────────────────

def test_wrapping_respects_the_width_and_keeps_hard_newlines():
    text = 'one two three four five six seven eight nine ten'
    wide = wrap_text_to_width(text, 10, 10_000)
    narrow = wrap_text_to_width(text, 10, 60)
    assert wide == [text]
    assert len(narrow) > 1
    assert ' '.join(narrow).split() == text.split()   # no words lost

    hard = wrap_text_to_width('alpha\nbeta', 10, 10_000)
    assert hard == ['alpha', 'beta']


def test_a_word_longer_than_the_line_is_left_alone_not_dropped():
    lines = wrap_text_to_width('supercalifragilistic x', 40, 10)
    assert 'supercalifragilistic' in lines


# ── sections ───────────────────────────────────────────────────────────────

def test_section_body_starts_under_the_header():
    p = _poster()
    p.add_section('s', row=4, col=2, rowspan=10, colspan=6, title='T',
                  header_rows=2)
    assert p.section_body('s') == (6, 2, 8, 6)


def test_a_section_with_no_title_has_no_header_and_a_full_body():
    p = _poster()
    p.add_section('s', row=0, col=0, rowspan=10, colspan=6)
    assert p.section_body('s') == (0, 0, 10, 6)


def test_header_rows_are_derived_from_the_title_size_when_not_given():
    p = _poster()
    p.add_section('s', row=0, col=0, rowspan=12, colspan=6, title='T')
    header = p._section_index['s']['header_rows']
    assert header >= 1
    assert header < 12


def test_a_header_that_swallows_the_whole_section_is_an_error():
    p = _poster()
    with pytest.raises(ValueError, match='no body'):
        p.add_section('s', row=0, col=0, rowspan=2, colspan=6, title='T',
                      header_rows=2)


def test_duplicate_section_names_are_rejected():
    p = _poster()
    p.add_section('s', row=0, col=0, rowspan=8, colspan=6)
    with pytest.raises(ValueError, match='already exists'):
        p.add_section('s', row=8, col=0, rowspan=8, colspan=6)


def test_unknown_section_names_the_defined_ones():
    p = _poster()
    p.add_section('intro', row=0, col=0, rowspan=8, colspan=6)
    with pytest.raises(KeyError, match='intro'):
        p.section_body('outro')


# ── section-relative placement ─────────────────────────────────────────────

def test_panels_are_placed_relative_to_the_section_body():
    p = _poster()
    p.add_section('s', row=4, col=2, rowspan=12, colspan=8, title='T',
                  header_rows=2)
    p.add_panel('a', row=1, col=1, rowspan=4, colspan=4, section='s')
    panel = p.panels[-1]
    assert (panel['row'], panel['col']) == (7, 3)     # 4 + 2 + 1, 2 + 1
    assert (panel['rowspan'], panel['colspan']) == (4, 4)
    assert panel['section'] == 's'


def test_unsectioned_panels_keep_absolute_coordinates():
    p = _poster()
    p.add_panel('a', row=3, col=5, rowspan=4, colspan=4)
    panel = p.panels[-1]
    assert (panel['row'], panel['col']) == (3, 5)
    assert panel['section'] is None


def test_a_panel_overflowing_its_section_warns_with_both_sizes():
    p = _poster()
    p.add_section('s', row=0, col=0, rowspan=10, colspan=6, title='T',
                  header_rows=2)
    with pytest.warns(UserWarning, match='overflows section'):
        p.add_panel('a', row=0, col=0, rowspan=20, colspan=4, section='s')


# ── text blocks ────────────────────────────────────────────────────────────

def test_text_block_is_an_unlettered_fit_exempt_axis_free_panel():
    p = _poster()
    p.add_text_block('t', row=0, col=0, rowspan=4, colspan=6,
                     text='hello world')
    panel = p.panels[-1]
    assert panel['no_label'] and panel['no_axis'] and panel['fit_exempt']
    assert panel['is_text_block']


def test_text_block_needs_exactly_one_of_text_or_bullets():
    p = _poster()
    with pytest.raises(ValueError, match='text= or bullets='):
        p.add_text_block('t', 0, 0, 4, 6)
    with pytest.raises(ValueError, match='not both'):
        p.add_text_block('t', 0, 0, 4, 6, text='a', bullets=['b'])


def test_bullets_render_as_one_mark_and_body_text_per_line():
    p = _poster()
    p.add_text_block('t', row=0, col=0, rowspan=6, colspan=12,
                     bullets=['first point', 'second point'])
    _, axes = p.compose()
    texts = [t.get_text() for t in axes['t'].texts]
    assert texts == ['•', 'first point', '•', 'second point']


# ── interaction with FigureComposer's normalisation passes ─────────────────

def test_poster_chrome_survives_normalize_fonts():
    p = _poster()
    p.add_section('s', row=6, col=0, rowspan=18, colspan=24, title='Results')
    p.add_title('A Poster', authors='Someone', row=0, rowspan=6)
    p.add_panel('a', row=1, col=1, rowspan=8, colspan=8, section='s')
    fig, _ = p.compose()

    chrome = [t for t in fig.texts
              if t.get_text() in ('Results', 'A Poster', 'Someone')]
    assert len(chrome) == 3
    before = [t.get_fontsize() for t in chrome]
    assert all(t.get_gid() == NO_FONT_NORMALIZE for t in chrome)

    p.normalize_fonts()
    assert [t.get_fontsize() for t in chrome] == before
    # ... and the section title is still bigger than body text, which is the
    # whole point of exempting it.
    section_title = next(t for t in chrome if t.get_text() == 'Results')
    assert section_title.get_fontsize() > p.font_size


def test_fit_exempt_panels_are_not_moved_by_fit_axes_to_cells():
    """The row-alignment pass resizes every panel sharing a row extent.

    A text block shares its rows with data panels but positions its content in
    axes fractions, so being resized would move the text off its intended spot.
    """
    p = _poster()
    p.add_panel('a', row=0, col=0, rowspan=8, colspan=12)
    p.add_text_block('t', row=0, col=12, rowspan=8, colspan=12,
                     text='some running text that must not be rescaled')
    fig, axes = p.compose()
    axes['a'].plot([0, 1], [0, 1])
    axes['a'].set_xlabel('a long axis label forcing a shrink')

    before = tuple(axes['t'].get_position().bounds)
    p.fit_axes_to_cells()
    assert tuple(axes['t'].get_position().bounds) == pytest.approx(before)
    # the ordinary panel *was* fitted, so the test is not vacuous
    assert tuple(axes['a'].get_position().bounds) != pytest.approx(
        tuple(axes['a']._sciplotlib_fit_base))


def test_no_label_suppresses_the_panel_letter_but_keeps_the_axes_key():
    p = _poster()
    p.add_panel('a', row=0, col=0, rowspan=8, colspan=12)
    p.add_panel('quiet', row=0, col=12, rowspan=8, colspan=12, no_label=True)
    fig, axes = p.compose()
    assert set(axes) == {'a', 'quiet'}
    letters = {t.get_text() for t in fig.texts}
    assert 'a' in letters and 'quiet' not in letters


# ── compose ────────────────────────────────────────────────────────────────

def test_compose_draws_section_and_title_artists_inside_the_canvas():
    p = _poster()
    p.add_title('T', authors='A', affiliations='Aff', row=0, rowspan=6)
    p.add_section('s', row=6, col=0, rowspan=18, colspan=24, title='Results')
    fig, _ = p.compose()

    assert p._section_artists['s']
    assert p._title_artists

    x0, y0, w, h = p.cell_rect(6, 0, 18, 24)
    assert 0.0 <= x0 < 1.0 and 0.0 <= y0 < 1.0
    assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def test_a_title_too_tall_for_its_band_warns_rather_than_running_off_the_page():
    p = _poster()
    with pytest.warns(UserWarning, match='title band'):
        p.add_title('A very long poster title ' * 6, authors='A',
                    affiliations='Aff', row=0, rowspan=2)
        p.compose()


# ── redraw hooks ───────────────────────────────────────────────────────────

def test_a_text_block_registers_a_redraw_hook():
    p = _poster()
    p.add_text_block('t', row=0, col=0, rowspan=6, colspan=12,
                     text='some text')
    _, axes = p.compose()
    hook = getattr(axes['t'], REDRAW_HOOK_ATTR, None)
    assert hook is not None
    func, spec = hook
    assert func is draw_text_block
    assert spec['text'] == 'some text'


def test_the_hook_and_its_spec_survive_pickling():
    """The editor runs in a subprocess on a pickled figure.

    A closure would not make that trip, which is why the hook is a
    module-level function taking plain data.
    """
    p = _poster()
    p.add_text_block('t', row=0, col=0, rowspan=6, colspan=12, text='hello')
    fig, axes = p.compose()
    func, spec = pickle.loads(pickle.dumps(getattr(axes['t'], REDRAW_HOOK_ATTR)))
    assert func is draw_text_block
    assert spec['text'] == 'hello'


def test_resizing_a_text_block_rewraps_it_when_the_hook_runs():
    long_text = ('a fairly long sentence that has to wrap differently once the '
                 'panel it lives in is made much narrower than it started')
    p = _poster()
    p.add_text_block('t', row=0, col=0, rowspan=8, colspan=24, text=long_text)
    _, axes = p.compose()
    ax = axes['t']
    wide_lines = ax.texts[0].get_text().count('\n')

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width / 4, pos.height])
    assert run_redraw_hook(ax) is True
    narrow_lines = ax.texts[0].get_text().count('\n')

    assert narrow_lines > wide_lines
    # no words lost in the re-wrap
    assert ax.texts[0].get_text().split() == long_text.split()


def test_fit_axes_to_cells_reflows_hooked_panels_so_saves_match_the_editor():
    p = _poster()
    p.add_text_block('t', row=0, col=0, rowspan=8, colspan=24,
                     text='some words that will need to be re-wrapped ' * 3)
    _, axes = p.compose()
    ax = axes['t']

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width / 5, pos.height])
    ax._sciplotlib_fit_base = tuple(ax.get_position().bounds)
    before = ax.texts[0].get_text()

    p.fit_axes_to_cells()
    assert ax.texts[0].get_text() != before   # re-wrapped to the narrow box


# ── linewidth-normalisation exemptions ─────────────────────────────────────
# A poster's house linewidth is thick (2.5 pt at A0). Anything drawn as many
# short strokes -- a choice raster with one Line2D per trial, a cartoon --
# fills in solid at that width, so it has to be able to opt out.

def test_normalize_linewidths_still_sets_ordinary_lines():
    p = _poster(line_linewidth=2.5)
    p.add_panel('a', row=0, col=0, rowspan=8, colspan=12)
    _, axes = p.compose()
    line, = axes['a'].plot([0, 1], [0, 1], linewidth=0.3)
    p.normalize_linewidths()
    assert line.get_linewidth() == pytest.approx(2.5)


def test_an_exempt_line_keeps_its_own_width():
    p = _poster(line_linewidth=2.5)
    p.add_panel('a', row=0, col=0, rowspan=8, colspan=12)
    _, axes = p.compose()
    normal, = axes['a'].plot([0, 1], [0, 1], linewidth=0.3)
    raster, = axes['a'].plot([0, 1], [1, 0], linewidth=0.9)
    exempt_from_linewidth_normalization(raster)
    p.normalize_linewidths()
    assert raster.get_linewidth() == pytest.approx(0.9)
    assert normal.get_linewidth() == pytest.approx(2.5)   # not vacuous


def test_an_exempt_axes_covers_every_line_in_it():
    p = _poster(line_linewidth=2.5)
    p.add_panel('a', row=0, col=0, rowspan=8, colspan=12)
    _, axes = p.compose()
    inset = axes['a'].inset_axes([0.1, 0.1, 0.5, 0.5])
    lines = [inset.plot([0, 1], [i, i], linewidth=0.9)[0] for i in range(5)]
    exempt_from_linewidth_normalization(inset)
    p.normalize_linewidths()
    assert all(ln.get_linewidth() == pytest.approx(0.9) for ln in lines)


def test_exempting_an_axes_also_covers_its_child_axes():
    p = _poster(line_linewidth=2.5)
    p.add_panel('a', row=0, col=0, rowspan=8, colspan=12)
    _, axes = p.compose()
    exempt_from_linewidth_normalization(axes['a'])
    child = axes['a'].inset_axes([0.1, 0.1, 0.4, 0.4])
    line, = child.plot([0, 1], [0, 1], linewidth=0.4)
    p.normalize_linewidths()
    assert line.get_linewidth() == pytest.approx(0.4)


def test_run_redraw_hook_is_a_no_op_without_a_hook():
    p = _poster()
    p.add_panel('a', row=0, col=0, rowspan=8, colspan=12)
    _, axes = p.compose()
    assert run_redraw_hook(axes['a']) is False


def test_a_hook_survives_being_run_repeatedly():
    """The hook clears the axes, which must not clear the hook itself."""
    p = _poster()
    p.add_text_block('t', row=0, col=0, rowspan=8, colspan=24, text='hello there')
    _, axes = p.compose()
    for _ in range(3):
        assert run_redraw_hook(axes['t']) is True
    assert len(axes['t'].texts) == 1


def _draw_marker(ax, label):
    ax.clear()
    ax.text(0.5, 0.5, label)


def test_set_redraw_hook_takes_any_module_level_function():
    p = _poster()
    p.add_panel('a', row=0, col=0, rowspan=8, colspan=12)
    _, axes = p.compose()
    set_redraw_hook(axes['a'], _draw_marker, label='hi')
    assert run_redraw_hook(axes['a']) is True
    assert axes['a'].texts[0].get_text() == 'hi'


# ── title banner: dark bands and logos ─────────────────────────────────────

def _logo_png(tmp_path, name='logo.png', alpha=True, color=(0.0, 0.2, 0.6)):
    """A small logo-shaped image: an opaque blob on a transparent ground."""
    import matplotlib.pyplot as plt
    img = np.zeros((20, 60, 4), dtype=float)
    img[5:15, 10:50, :3] = color
    img[5:15, 10:50, 3] = 1.0
    if not alpha:
        img = img[:, :, :3]
    path = tmp_path / name
    plt.imsave(path, img)
    return path


def test_a_dark_band_flips_the_text_to_white_without_being_told():
    p = _poster()
    p.add_title('T', authors='A', affiliations='Aff', row=0, rowspan=6,
                band_facecolor='black')
    t = p._title
    assert t['color'] == 'white'
    # the theme's muted grey would be invisible on black, so it must not be used
    assert t['affiliations_color'] != POSTER_THEMES['light']['muted_text']
    assert relative_luminance(t['affiliations_color']) > 0.5


def test_a_light_band_keeps_the_theme_colours():
    p = _poster()
    p.add_title('T', affiliations='Aff', row=0, rowspan=6,
                band_facecolor='#eeeeee')
    assert p._title['color'] == POSTER_THEMES['light']['title_text']
    assert p._title['affiliations_color'] == POSTER_THEMES['light']['muted_text']


def test_explicit_colours_still_win_over_the_band():
    p = _poster()
    p.add_title('T', affiliations='Aff', row=0, rowspan=6,
                band_facecolor='black', color='#ff0000',
                affiliations_color='#00ff00')
    assert p._title['color'] == '#ff0000'
    assert p._title['affiliations_color'] == '#00ff00'


def test_is_dark_handles_none_and_named_colours():
    assert is_dark('black') and is_dark('#101010')
    assert not is_dark('white')
    assert not is_dark('none') and not is_dark(None)


def test_a_filled_band_bleeds_to_the_paper_edge_by_default():
    p = _poster()
    p.add_title('T', row=0, rowspan=6, band_facecolor='black')
    assert p._title['full_bleed'] is True
    fig, _ = p.compose()
    band = next(a for a in p._title_artists
                if isinstance(a, matplotlib.patches.FancyBboxPatch))
    bb = band.get_window_extent().transformed(fig.transFigure.inverted())
    assert bb.x0 == pytest.approx(0.0, abs=1e-6)
    assert bb.x1 == pytest.approx(1.0, abs=1e-6)
    assert bb.y1 == pytest.approx(1.0, abs=1e-6)


def test_an_unfilled_title_stays_on_the_grid():
    p = _poster()
    p.add_title('T', row=0, rowspan=6)
    assert p._title['full_bleed'] is False


def test_a_missing_logo_warns_and_draws_the_banner_anyway():
    p = _poster()
    p.add_title('T', row=0, rowspan=6, band_facecolor='black',
                left_logo='/nonexistent/ucl-logo.png')
    with pytest.warns(UserWarning, match='left_logo not found'):
        fig, _ = p.compose()
    assert p._title_artists          # the band and text were still drawn


def test_a_logo_is_placed_and_tinted(tmp_path):
    path = _logo_png(tmp_path)
    p = _poster()
    p.add_title('T', row=0, rowspan=6, band_facecolor='black',
                left_logo=path, left_logo_tint='white')
    fig, _ = p.compose()
    logo_axes = [a for a in p._title_artists
                 if getattr(a, '_sciplotlib_poster_logo', None) == 'left_logo']
    assert len(logo_axes) == 1
    img = logo_axes[0].images[0].get_array()
    opaque = img[..., 3] > 0.5
    assert opaque.any()
    assert img[opaque][:, :3].min() == pytest.approx(1.0)   # tinted white
    assert not img[~opaque].any()                            # ground untouched


def test_tinting_an_image_without_alpha_is_refused_not_silently_wrong(tmp_path):
    """Tinting an opaque rectangle would flood the whole box with the colour."""
    path = _logo_png(tmp_path, name='flat.jpg', alpha=False)
    p = _poster()
    p.add_title('T', row=0, rowspan=6, band_facecolor='black',
                left_logo=path, left_logo_tint='white')
    with pytest.warns(UserWarning, match='no alpha channel'):
        p.compose()


# ── section frames are checked by the layout checker ───────────────────────
# A frame is a boundary: content belongs inside it and must not cross its
# stroke. Ordinary patches are opt-in for the checks, so without this the
# border can run straight through a text block and the report still says clean.

def _framed_poster(facecolor, reps):
    p = _poster(grid_rows=24, grid_cols=24)
    p.add_section('s', row=0, col=0, rowspan=12, colspan=12, title='Sec',
                  facecolor=facecolor, edgecolor='#6E7B8B', linewidth=2.2,
                  header_style='plain')
    p.add_text_block('t', row=6, col=0, rowspan=3, colspan=12, section='s',
                     text='overflowing text ' * reps)
    fig, _ = p.compose()
    return p, fig


def _frame_hits(fig):
    from sciplotlib import collide
    return [c for c in collide.find_collisions(fig, min_gap_pt=1.0)
            if 'section frame' in (c.a_desc + c.b_desc)]


def test_an_unfilled_frame_is_tagged_for_the_layout_checker():
    from sciplotlib import collide
    _, fig = _framed_poster('none', 5)
    tagged = [a for a in fig.artists
              if getattr(a, collide.POSTER_SECTION_ATTR, None) == 's']
    assert len(tagged) == 1
    assert collide._artist_kind(tagged[0]) == 'section'
    assert 'section' in collide.DEFAULT_KINDS


def test_text_bursting_out_of_its_section_is_reported():
    _, fig = _framed_poster('none', 200)
    hits = _frame_hits(fig)
    assert hits, 'text crossing the section frame should be reported'
    assert hits[0].kind == 'overlap'
    assert hits[0].overlap_pt2 > 0


def test_text_merely_inside_its_section_is_not_reported():
    """The frame surrounds the content; only crossing its stroke is a problem."""
    _, fig = _framed_poster('none', 20)
    assert _frame_hits(fig) == []


def test_a_filled_frame_is_left_untagged():
    """Ink-based checks + a filled frame = every panel inside it 'collides'."""
    from sciplotlib import collide
    _, fig = _framed_poster('#f4f5f7', 200)
    tagged = [a for a in fig.artists
              if getattr(a, collide.POSTER_SECTION_ATTR, None) is not None]
    assert tagged == []
    assert _frame_hits(fig) == []


def test_describe_lists_sections_with_their_panels():
    p = _poster()
    p.add_section('s', row=0, col=0, rowspan=12, colspan=24, title='Results')
    p.add_panel('a', row=0, col=0, rowspan=6, colspan=10, section='s')
    p.add_panel('loose', row=14, col=0, rowspan=6, colspan=10)
    out = p.describe()
    assert "section 's'" in out
    assert "panel 'a'" in out
    assert 'unsectioned' in out and "'loose'" in out


# ── panel lettering ────────────────────────────────────────────────────────
# Lettering exists so a caption can point at a panel. A poster has no caption,
# so it navigates by section heading instead and the letters are noise.

def test_panel_letters_are_drawn_by_default():
    p = _poster()
    p.add_panel('a', row=0, col=0, rowspan=8, colspan=12)
    fig, _ = p.compose()
    assert 'a' in {t.get_text() for t in fig.texts}


def test_panel_letters_can_be_switched_off_for_the_whole_poster():
    p = _poster(panel_letters=False)
    p.add_panel('a', row=0, col=0, rowspan=8, colspan=12)
    fig, axes = p.compose()
    assert 'a' not in {t.get_text() for t in fig.texts}
    assert set(axes) == {'a'}          # still the axes dict key


def test_a_single_panel_can_opt_back_in():
    p = _poster(panel_letters=False)
    p.add_panel('a', row=0, col=0, rowspan=8, colspan=12)
    p.add_panel('b', row=0, col=12, rowspan=8, colspan=12, no_label=False)
    fig, _ = p.compose()
    letters = {t.get_text() for t in fig.texts}
    assert 'b' in letters and 'a' not in letters


def test_bullet_spacing_opens_a_gap_between_points_only():
    """Extra space goes between bullets, not between a bullet's own lines."""
    from sciplotlib.poster import PosterComposer

    def _ys(spacing):
        p = PosterComposer(paper='a0_landscape', grid_rows=12, grid_cols=12,
                           dpi=100)
        p.add_text_block('t', row=0, col=0, rowspan=12, colspan=12,
                         fontsize=26, bullet_spacing=spacing, wrap=False,
                         bullets=['one\nwrapped', 'two', 'three'])
        _, axes = p.compose()
        # body texts only (the bullet marks share their line's y)
        return sorted({round(t.get_position()[1], 6)
                       for t in axes['t'].texts}, reverse=True)

    tight, loose = _ys(0.0), _ys(1.0)
    assert len(tight) == len(loose) == 4
    # within a bullet the pitch is unchanged...
    assert abs((tight[0] - tight[1]) - (loose[0] - loose[1])) < 1e-9
    # ...but the step into the next bullet is bigger
    assert (loose[1] - loose[2]) > (tight[1] - tight[2]) + 1e-6
