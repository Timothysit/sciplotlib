"""Tests for slide composition (sciplotlib.slides).

Covers what a deck adds on top of a poster and what is easy to get subtly
wrong: the exact page geometry (a deck is concatenated with a PowerPoint
export, so 13.333 x 7.5 in has to be exact), the chrome bands being reserved
*out of the canvas* so no panel can land under the title, builds producing one
page per stage, and the two behaviours that differ from every other composer --
ticks defaulting to on, and chrome surviving ``normalize_fonts``.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from sciplotlib.compose import NO_FONT_NORMALIZE, PAPER_DIMENSIONS
from sciplotlib.slides import (
    REFERENCE_SLIDE_WIDTH_CM,
    SLIDE_TYPE_SIZES,
    SlideComposer,
    SlideDeck,
)


def _slide(**kwargs):
    kwargs.setdefault('dpi', 60)
    return SlideComposer(**kwargs)


def _plot(ax):
    ax.plot([0, 1], [0, 1])


# ── canvas geometry ────────────────────────────────────────────────────────

def test_default_paper_is_powerpoint_widescreen():
    # 13.333 x 7.5 in exactly -- a deck merged with a collaborator's PowerPoint
    # export must not be rescaled, which would rescale every font in it.
    s = _slide()
    assert s.width_cm / 2.54 == pytest.approx(13.333, abs=1e-3)
    assert s.height_cm / 2.54 == pytest.approx(7.5, abs=1e-3)


@pytest.mark.parametrize('paper', ['slide_16x9', 'slide_16x10', 'slide_4x3'])
def test_slide_papers_are_registered(paper):
    assert paper in PAPER_DIMENSIONS
    s = _slide(paper=paper)
    w, h = PAPER_DIMENSIONS[paper]
    assert (s.width_cm, s.height_cm) == (w, h)


def test_unknown_paper_raises():
    with pytest.raises(ValueError, match='unknown paper'):
        _slide(paper='a5_landscape_with_go_faster_stripes')


def test_explicit_size_overrides_paper():
    s = _slide(width_cm=40, height_cm=20)
    assert (s.width_cm, s.height_cm) == (40, 20)


# ── type scale ─────────────────────────────────────────────────────────────

def test_type_scale_is_one_on_the_reference_slide():
    s = _slide()
    assert s.type_scale == pytest.approx(1.0)
    assert s.font_size == pytest.approx(SLIDE_TYPE_SIZES['font_size'])
    assert s.slide_title_size == pytest.approx(SLIDE_TYPE_SIZES['slide_title_size'])


def test_type_scale_tracks_width():
    s = _slide(width_cm=REFERENCE_SLIDE_WIDTH_CM / 2, height_cm=10)
    assert s.type_scale == pytest.approx(0.5)
    assert s.font_size == pytest.approx(SLIDE_TYPE_SIZES['font_size'] / 2)


def test_poster_type_sizes_do_not_leak_in():
    # PosterComposer is the base class and its defaults are calibrated for A0
    # (76 pt titles, 20 pt body).  On a slide those would be absurd.
    s = _slide(title='t')
    assert s.body_size == pytest.approx(SLIDE_TYPE_SIZES['body_size'])
    assert s.poster_title_size == pytest.approx(SLIDE_TYPE_SIZES['slide_title_size'])


def test_explicit_font_size_wins():
    s = _slide(font_size=9)
    assert s.font_size == 9


# ── chrome bands are reserved out of the canvas ────────────────────────────

def test_title_band_is_reserved_so_row_zero_clears_it():
    without = _slide()
    with_title = _slide(title='A title')
    assert with_title.title_cm > 0
    assert without.title_cm == 0
    # The grid top margin drops by at least the band height.
    assert with_title.margins['top'] < without.margins['top']
    drop = (without.margins['top'] - with_title.margins['top']) * with_title.height_cm
    assert drop == pytest.approx(with_title.title_cm, abs=0.01)


def test_subtitle_makes_the_band_taller():
    one_line = _slide(title='A title')
    two_line = _slide(title='A title', subtitle='and a subtitle')
    assert two_line.title_cm > one_line.title_cm


def test_footer_band_is_reserved_only_when_there_is_a_footer():
    assert _slide().footer_cm == 0
    assert _slide(footer='S4SN').footer_cm > 0
    assert _slide(page_number=3).footer_cm > 0


def test_panel_at_row_zero_sits_below_the_title_band():
    s = _slide(title='A title')
    s.add_panel('a', 0, 0, 6, 10, plot_func=_plot)
    fig, axes = s.compose()
    top_of_panel = axes['a'].get_position().y1
    band_bottom = 1.0 - s.title_cm / s.height_cm
    assert top_of_panel <= band_bottom + 1e-6
    plt.close(fig)


# ── ticks default to on, unlike every other composer ───────────────────────

def test_ticks_are_on_by_default():
    s = _slide()
    s.add_panel('a', 0, 0, 10, 10, plot_func=_plot)
    fig, axes = s.compose()
    fig.canvas.draw()
    assert [t.get_text() for t in axes['a'].get_xticklabels()]
    plt.close(fig)


def test_ticks_false_restores_paper_figure_behaviour():
    s = _slide()
    s.add_panel('a', 0, 0, 10, 10, ticks=False, plot_func=_plot)
    fig, axes = s.compose()
    fig.canvas.draw()
    assert not [t.get_text() for t in axes['a'].get_xticklabels()]
    plt.close(fig)


def test_plot_func_can_still_turn_ticks_off():
    # The wrapper runs first so the panel function has the last word.
    def hide(ax):
        ax.plot([0, 1], [0, 1])
        ax.tick_params(labelbottom=False)

    s = _slide()
    s.add_panel('a', 0, 0, 10, 10, plot_func=hide)
    fig, axes = s.compose()
    fig.canvas.draw()
    assert not [t.get_text() for t in axes['a'].get_xticklabels()]
    plt.close(fig)


def test_tick_wrapper_runs_a_none_plot_func():
    s = _slide()
    s.add_panel('a', 0, 0, 10, 10)          # no plot_func at all
    fig, axes = s.compose()
    axes['a'].plot([0, 1], [0, 1])
    fig.canvas.draw()
    assert [t.get_text() for t in axes['a'].get_xticklabels()]
    plt.close(fig)


# ── title and footer chrome ────────────────────────────────────────────────

def test_title_and_footer_are_drawn():
    s = _slide(title='Encoding model', subtitle='DA vs NE',
               footer='S4SN 2026', page_number=7)
    fig, _ = s.compose()
    drawn = {t.get_text() for t in fig.texts}
    assert {'Encoding model', 'DA vs NE', 'S4SN 2026', '7'} <= drawn
    plt.close(fig)


def test_chrome_survives_normalize_fonts():
    # A slide title is meant to be bigger than body text; normalisation would
    # flatten it to font_size otherwise.
    s = _slide(title='Encoding model', footer='S4SN 2026')
    fig, _ = s.compose()
    s.normalize_fonts()
    title = next(t for t in fig.texts if t.get_text() == 'Encoding model')
    footer = next(t for t in fig.texts if t.get_text() == 'S4SN 2026')
    assert title.get_gid() == NO_FONT_NORMALIZE
    assert title.get_fontsize() == pytest.approx(s.slide_title_size)
    assert footer.get_fontsize() == pytest.approx(s.footer_size)
    assert s.slide_title_size != s.font_size     # normalisation had teeth
    plt.close(fig)


def test_long_title_wraps_inside_the_canvas():
    long = ('Strategy helps noradrenaline before the choice and dopamine '
            'after it, which is the whole result of this section')
    s = _slide(title=long)
    fig, _ = s.compose()
    title = next(t for t in fig.texts if t.get_text().startswith('Strategy'))
    assert '\n' in title.get_text()          # it wrapped
    fig.canvas.draw()
    box = title.get_window_extent().transformed(fig.transFigure.inverted())
    assert box.x1 <= s.margins['right'] + 1e-3
    plt.close(fig)


def test_wrapped_title_shrinks_to_stay_inside_its_band():
    long = ' '.join(['a really quite unreasonably long slide title'] * 4)
    s = _slide(title=long)
    fig, _ = s.compose()
    title = next(t for t in fig.texts if t.get_text().startswith('a really'))
    assert title.get_fontsize() < s.slide_title_size
    fig.canvas.draw()
    box = title.get_window_extent().transformed(fig.transFigure.inverted())
    # Must not grow down over the first row of panels.
    assert box.y0 >= s.margins['top'] - 0.02
    plt.close(fig)


def test_wrap_false_leaves_the_title_alone():
    long = 'x' * 200
    s = _slide(title=long)
    s.add_title(long, wrap=False)
    fig, _ = s.compose()
    assert any(t.get_text() == long for t in fig.texts)
    plt.close(fig)


def test_no_title_draws_no_chrome():
    s = _slide()
    fig, _ = s.compose()
    assert not [t for t in fig.texts]
    plt.close(fig)


def test_add_title_after_construction():
    s = _slide(title='placeholder')
    s.add_title('replacement', subtitle='sub')
    fig, _ = s.compose()
    drawn = {t.get_text() for t in fig.texts}
    assert 'replacement' in drawn and 'placeholder' not in drawn
    plt.close(fig)


# ── deck ───────────────────────────────────────────────────────────────────

def test_deck_counts_pages_by_stage():
    deck = SlideDeck()
    deck.add(lambda s: None, title='one')
    deck.add(lambda s, stage: None, title='two', stages=3)
    assert len(deck) == 2
    assert deck.n_pages == 4


def test_stages_must_be_positive():
    with pytest.raises(ValueError, match='stages must be'):
        SlideDeck().add(lambda s: None, stages=0)


def test_builder_arity_is_detected():
    seen = []
    deck = SlideDeck(number_slides=False)
    deck.add(lambda s: seen.append('one-arg'), title='a')
    deck.add(lambda s, stage: seen.append(f'two-arg{stage}'), title='b', stages=2)
    for _, _, _, slide in deck.render():
        plt.close(slide.fig)
    assert seen == ['one-arg', 'two-arg0', 'two-arg1']


def test_stage_reaches_the_builder_and_pages_differ():
    deck = SlideDeck()

    def build(slide, stage):
        for i in range(stage + 1):
            slide.add_panel(f'p{i}', 0, 4 * i, 4, 4, plot_func=_plot)

    deck.add(build, title='builds', stages=3)
    counts = []
    for _, _, _, slide in deck.render():
        counts.append(len(slide.axes))
        plt.close(slide.fig)
    assert counts == [1, 2, 3]


def test_build_stages_share_one_slide_number():
    deck = SlideDeck(first_number=5)
    deck.add(lambda s, stage: None, title='a', stages=3)
    deck.add(lambda s: None, title='b')
    numbers = []
    for _, _, number, slide in deck.render():
        numbers.append(number)
        plt.close(slide.fig)
    assert numbers == [5, 5, 5, 6]


def test_first_number_offsets_the_deck():
    # The user's slides sit partway through a collaborator's deck.
    deck = SlideDeck(first_number=12)
    deck.add(lambda s: None, title='a')
    _, _, number, slide = next(deck.render())
    assert number == 12
    assert '12' in {t.get_text() for t in slide.fig.texts}
    plt.close(slide.fig)


def test_decorator_registers_and_returns_the_function():
    deck = SlideDeck()

    @deck.slide(title='decorated', notes='hello')
    def built(slide):
        return 'kept'

    assert built(None) == 'kept'
    assert len(deck) == 1
    assert deck._specs[0]['notes'] == 'hello'
    assert deck._specs[0]['name'] == 'built'


def test_per_slide_footer_and_numbering_override_the_deck():
    deck = SlideDeck(footer='deck footer', number_slides=True)
    deck.add(lambda s: None, title='a', footer=None, number_slides=False)
    _, _, _, slide = next(deck.render())
    drawn = {t.get_text() for t in slide.fig.texts}
    assert drawn == {'a'}          # no footer, no number
    plt.close(slide.fig)


def test_builder_may_compose_itself():
    # The marimo pattern: compose, then plot into the axes dict.
    def build(slide):
        slide.add_panel('a', 0, 0, 8, 8)
        fig, axes = slide.compose()
        axes['a'].plot([0, 1], [0, 1])

    deck = SlideDeck()
    deck.add(build, title='self-composed')
    _, _, _, slide = next(deck.render())
    assert slide.fig is not None
    assert len(slide.axes['a'].lines) == 1
    plt.close(slide.fig)


def test_unknown_slide_kwarg_is_rejected():
    deck = SlideDeck()
    deck.add(lambda s: None, title='a', nonsense=True)
    with pytest.raises(TypeError):
        next(deck.render())


# ── output ─────────────────────────────────────────────────────────────────

def test_save_writes_one_pdf_with_a_page_per_stage(tmp_path):
    deck = SlideDeck(footer='S4SN')
    deck.add(lambda s: s.add_panel('a', 0, 0, 8, 8, plot_func=_plot), title='a')
    deck.add(lambda s, stage: None, title='b', stages=2)

    n = deck.save(tmp_path / 'talk', formats=('pdf',), check_layout=False,
                  verbose=False)
    out = tmp_path / 'talk.pdf'
    assert n == 3
    assert out.exists()
    # matplotlib writes one /Pages node carrying the page count.
    assert b'/Type /Pages' in out.read_bytes()
    assert b'/Count 3' in out.read_bytes()


def test_saved_pages_are_exactly_the_paper_size(tmp_path):
    deck = SlideDeck()
    deck.add(lambda s: s.add_panel('a', 0, 0, 8, 8, plot_func=_plot), title='a')
    deck.save(tmp_path / 'talk', formats=('pdf',), check_layout=False,
              verbose=False)
    body = (tmp_path / 'talk.pdf').read_bytes()
    # 13.333 x 7.5 in at 72 pt/in, as matplotlib writes the MediaBox.
    assert b'/MediaBox [ 0 0 960' in body or b'/MediaBox [0 0 960' in body


def test_stylesheet_font_settings_reach_the_pdf(tmp_path):
    # matplotlib reads pdf.fonttype when PdfPages is constructed, so a
    # stylesheet applied per-slide would arrive too late and the file would be
    # written with Type 3 fonts -- which readers reject for OTF/CFF faces.
    deck = SlideDeck(stylesheet='mp-paper')
    deck.add(lambda s: s.add_panel('a', 0, 0, 8, 8, plot_func=_plot),
             title='Encoding model')
    deck.save(tmp_path / 'talk', formats=('pdf',), check_layout=False,
              verbose=False)
    body = (tmp_path / 'talk.pdf').read_bytes()
    assert b'/Subtype /Type3' not in body
    assert plt.rcParams['pdf.fonttype'] == 42


def test_save_closes_every_figure(tmp_path):
    deck = SlideDeck()
    for i in range(3):
        deck.add(lambda s: None, title=f'slide {i}')
    before = len(plt.get_fignums())
    deck.save(tmp_path / 'talk', formats=('pdf',), check_layout=False,
              verbose=False)
    assert len(plt.get_fignums()) == before


def test_png_format_writes_one_file_per_page(tmp_path):
    deck = SlideDeck()
    deck.add(lambda s, stage: None, title='a', stages=2)
    deck.save(tmp_path / 'talk', formats=('png',), check_layout=False,
              verbose=False, dpi=40)
    assert sorted(p.name for p in tmp_path.glob('*.png')) == [
        'talk-01.png', 'talk-02.png']
    assert not (tmp_path / 'talk.pdf').exists()


def test_save_notes_writes_markdown(tmp_path):
    deck = SlideDeck(first_number=3)
    deck.add(lambda s: None, title='Encoding model', notes='Land the timing.')
    deck.add(lambda s, stage: None, title='RPE', stages=2)
    p = deck.save_notes(tmp_path / 'notes')
    text = p.read_text()
    assert '## 3. Encoding model' in text
    assert 'Land the timing.' in text
    assert '## 4. RPE (2 build steps)' in text
    assert '_No notes._' in text


def test_preview_defaults_to_the_last_stage():
    deck = SlideDeck()
    seen = []
    deck.add(lambda s, stage: seen.append(stage), title='a', stages=4)
    deck.preview(0)
    assert seen == [3]


def test_describe_runs(capsys):
    deck = SlideDeck()
    deck.add(lambda s, stage: None, title='a', stages=2, notes='n')
    deck.describe()
    out = capsys.readouterr().out
    assert '2 pages' in out and 'p1-2' in out and '[notes]' in out
