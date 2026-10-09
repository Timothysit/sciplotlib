"""Checks for sciplotlib.collide.

Runs under pytest, or standalone: ``python tests/test_collide.py``.

The cases here are the ones that distinguish ink-based checking from
bounding-box checking, plus the two classes of artist that lie about their
geometry (culled tick labels, tick/axis labels under ``axis('off')``).
"""
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sciplotlib.collide as splcollide
import sciplotlib.compose as splcompose


def hollow_image(n=200, border=12):
    """RGBA square ring: opaque border, fully transparent middle."""
    img = np.zeros((n, n, 4), dtype=np.uint8)
    img[..., :3] = 120
    img[:border, :, 3] = 255
    img[-border:, :, 3] = 255
    img[:, :border, 3] = 255
    img[:, -border:, 3] = 255
    return img


def make_figure():
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax._sciplotlib_panel = 'a'

    art = {}
    art['image'] = splcompose.place_image(ax, hollow_image(), 2.0, 5.0, zoom=0.30)
    # descender of the 'g' reaches down onto the ring's top edge
    art['clash'] = ax.text(2.0, 6.6, 'Algorithm', ha='center', va='center', fontsize=12)
    # inside the hollow middle: overlaps the image's box, touches no ink
    art['inside'] = ax.text(2.0, 5.0, 'Choice', ha='center', va='center', fontsize=8)
    art['clear'] = ax.text(2.0, 9.0, 'Clear', ha='center', va='center', fontsize=8)
    art['left'] = ax.text(6.0, 3.0, 'left', ha='right', va='center', fontsize=10)
    art['right'] = ax.text(6.02, 3.0, 'right', ha='left', va='center', fontsize=10)
    art['offcanvas'] = ax.text(-2.6, 1.0, 'offcanvas', ha='center', va='center', fontsize=10)
    return fig, ax, art


def involving(collisions, artist):
    return [c for c in collisions if c.a is artist or c.b is artist]


def test_ink_finds_descender_overlap():
    fig, ax, art = make_figure()
    cols = splcollide.find_collisions(fig, min_gap_pt=0.0)
    hits = involving(cols, art['clash'])
    assert hits, 'descender overlapping the image border was not reported'
    assert hits[0].kind == 'overlap'
    assert hits[0].panel == 'a'
    assert hits[0].occluded, 'the image is drawn on top, so the text is hidden'
    assert 'move up' in hits[0].suggestion
    plt.close(fig)


def test_ink_ignores_text_inside_hollow_image():
    fig, ax, art = make_figure()
    cols = splcollide.find_collisions(fig, min_gap_pt=0.0)
    assert not involving(cols, art['inside']), \
        'text in the transparent middle of an image is not a collision'
    plt.close(fig)


def test_one_report_per_pair():
    """place_image nests an image inside an AnnotationBbox; report it once."""
    fig, ax, art = make_figure()
    cols = splcollide.find_collisions(fig, min_gap_pt=0.0)
    assert len(involving(cols, art['clash'])) == 1
    plt.close(fig)


def test_min_gap_reports_near_misses_only_when_asked():
    fig, ax, art = make_figure()
    pair = lambda cols: [c for c in cols
                         if {id(c.a), id(c.b)} == {id(art['left']), id(art['right'])}]
    assert not pair(splcollide.find_collisions(fig, min_gap_pt=0.0))
    close = pair(splcollide.find_collisions(fig, min_gap_pt=2.0))
    assert close and close[0].kind == 'too-close'
    assert 0 < close[0].gap_pt < 2.0
    plt.close(fig)


def test_clear_text_is_never_reported():
    fig, ax, art = make_figure()
    cols = splcollide.find_collisions(fig, min_gap_pt=2.0)
    assert not involving(cols, art['clear'])
    plt.close(fig)


def test_off_canvas_text():
    fig, ax, art = make_figure()
    cols = splcollide.find_collisions(fig, min_gap_pt=0.0)
    hits = [c for c in cols if c.kind == 'outside-figure' and c.a is art['offcanvas']]
    assert hits, 'text placed off the canvas was not reported'
    plt.close(fig)


def test_exemptions():
    fig, ax, art = make_figure()
    splcollide.allow_overlap(art['clash'], art['image'])
    assert not involving(splcollide.find_collisions(fig), art['clash'])

    splcollide.exempt_from_collision_check(art['offcanvas'])
    cols = splcollide.find_collisions(fig)
    assert not [c for c in cols if c.a is art['offcanvas']]
    plt.close(fig)


def test_hidden_axis_labels_are_not_collisions():
    """ax.axis('off') keeps tick labels visible=True but never draws them."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.axis('off')                                   # ticks 0.0 ... 1.0 remain
    ax.set_xlabel('hidden xlabel')
    txt = ax.text(0.0, 0.0, 'content', ha='left', va='bottom', fontsize=20)
    cols = splcollide.find_collisions(fig, min_gap_pt=2.0)
    assert not involving(cols, txt), [str(c) for c in cols]
    plt.close(fig)


def test_out_of_view_tick_labels_are_not_collisions():
    """A locator tick outside the view limits is kept, parked, and not drawn."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([-2, -1, 0, 1, 2])                 # -2 and 2 fall out of view
    ax.set_xticks([0.5])
    ax.set_xticklabels(['label'])
    cols = splcollide.find_collisions(fig, min_gap_pt=0.0)
    texts = {c.a_desc for c in cols} | {c.b_desc for c in cols}
    assert not any('2' in t for t in texts), [str(c) for c in cols]
    plt.close(fig)


def test_bbox_mode_runs_and_is_pessimistic():
    fig, ax, art = make_figure()
    ink = splcollide.find_collisions(fig, min_gap_pt=0.0, precision='ink')
    box = splcollide.find_collisions(fig, min_gap_pt=0.0, precision='bbox')
    ink_area = sum(c.overlap_pt2 for c in involving(ink, art['clash']))
    box_area = sum(c.overlap_pt2 for c in involving(box, art['clash']))
    assert box_area > ink_area, 'font-metric boxes should overstate the overlap'
    plt.close(fig)


def test_dpi_and_state_are_restored():
    fig, ax, art = make_figure()
    before = (fig.dpi, art['clash'].get_position(), art['image'].get_clip_on())
    splcollide.find_collisions(fig, min_gap_pt=1.0, check_dpi=250)
    after = (fig.dpi, art['clash'].get_position(), art['image'].get_clip_on())
    assert before == after
    # draw wrappers must be gone, not left shadowing the class method
    assert 'draw' not in vars(art['clash'])
    plt.close(fig)


def test_report_and_overlay(tmp_path=None):
    import tempfile
    from pathlib import Path
    out = Path(tmp_path or tempfile.mkdtemp())
    fig, ax, art = make_figure()
    cols = splcollide.find_collisions(fig, min_gap_pt=1.0)
    text = splcollide.format_collisions(cols, min_gap_pt=1.0)
    assert 'Layout check:' in text and '1.' in text
    n_artists = len(fig.findobj())
    splcollide.save_collision_overlay(fig, cols, out / 'overlay.png', dpi=90)
    assert (out / 'overlay.png').exists()
    assert len(fig.findobj()) == n_artists, 'overlay boxes were not removed'
    plt.close(fig)


def test_imshow_cropped_by_view_limits_is_not_clipping():
    """set_xlim/set_ylim around part of an image is a crop, not a mistake."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(np.random.RandomState(0).rand(60, 60), cmap='viridis')
    ax.set_xlim(10, 40)          # crop to the middle
    ax.set_ylim(40, 10)
    cols = splcollide.find_collisions(fig, min_gap_pt=1.0)
    assert not [c for c in cols if c.kind == 'clipped'], [str(c) for c in cols]
    plt.close(fig)


def test_clean_figure_reports_nothing():
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100, layout='constrained')
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    cols = splcollide.find_collisions(fig, min_gap_pt=0.0)
    assert cols == [], [str(c) for c in cols]
    assert 'clean' in splcollide.format_collisions(cols)
    plt.close(fig)


def test_label_box_over_the_edge_without_lost_ink_is_quiet():
    """A text box spans the font's whole band, so it can overshoot the canvas
    while every painted pixel is still on the page.  Only cut ink counts."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)   # default margins: tight
    ax.plot([0, 1], [0, 1])
    ax.set_ylabel('y')                                # box pokes past the left
    cols = splcollide.find_collisions(fig, min_gap_pt=0.0)
    assert not involving(cols, ax.yaxis.label), [str(c) for c in cols]
    plt.close(fig)


def test_cropped_label_is_reported():
    """Matplotlib's default margins really do cut the xlabel off a small figure."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel('x')
    cols = splcollide.find_collisions(fig, min_gap_pt=0.0)
    hits = [c for c in cols if c.a is ax.xaxis.label]
    assert hits and hits[0].kind == 'outside-figure'
    assert 'bottom' in hits[0].b_desc
    plt.close(fig)


def _line_figure():
    """Flat lines and fixed limits, so gaps in points are predictable.

    (Autoscale margins are what make a hand-computed data coordinate lie: with
    the default 5% padding, data x=0.002 is 10 pt from the spine, not 0.5.)
    """
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([0, 1], [0.5, 0.5], color='black', lw=2)
    ax.plot([0, 1], [0.2, 0.2], color='grey', lw=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def test_label_too_close_to_a_curve():
    fig, ax = _line_figure()
    # no descenders in the string, so its ink bottom is the baseline: ~1 pt
    # above the line's top edge
    close = ax.text(0.1, 0.512, 'close to the line', fontsize=8)
    clear = ax.text(0.1, 0.80, 'well clear', fontsize=8)
    cols = splcollide.find_collisions(fig, min_gap_pt=3.0)
    hits = involving(cols, close)
    assert hits, [str(c) for c in cols]
    assert hits[0].kind == 'too-close', str(hits[0])
    assert 0 < hits[0].gap_pt < 3.0, str(hits[0])
    assert not involving(cols, clear), [str(c) for c in cols]
    assert not involving(splcollide.find_collisions(fig, min_gap_pt=0.5), close)
    plt.close(fig)


def test_label_too_close_to_a_spine():
    fig, ax = _line_figure()
    on_spine = ax.text(0.003, 0.35, 'on the spine', fontsize=8)
    cols = splcollide.find_collisions(fig, min_gap_pt=3.0)
    hits = [c for c in involving(cols, on_spine) if 'spine' in (c.a_desc + c.b_desc)]
    assert hits, [str(c) for c in cols]
    assert 'move right' in hits[0].suggestion
    plt.close(fig)


def test_curves_crossing_each_other_are_not_collisions():
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, color='black', lw=2)
    ax.plot(x, 1 - x, color='red', lw=2)       # crosses in the middle
    cols = splcollide.find_collisions(fig, min_gap_pt=3.0)
    assert cols == [], [str(c) for c in cols]
    plt.close(fig)


def test_tick_labels_are_allowed_near_their_own_spine():
    """tick_pad puts them there on purpose; a big min_gap must not flag them."""
    fig, ax = _line_figure()
    cols = splcollide.find_collisions(fig, min_gap_pt=4.0)
    descs = [str(c) for c in cols]
    assert not any('spine' in d for d in descs), descs
    plt.close(fig)


def test_axes_background_is_not_an_artist_to_avoid():
    """ax.patch covers the data area; text inside the axes must stay quiet."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_facecolor('white')
    txt = ax.text(0.5, 0.5, 'in the middle', ha='center', fontsize=9)
    cols = splcollide.find_collisions(fig, min_gap_pt=0.0,
                                      kinds=('text', 'image', 'line', 'spine', 'patch'))
    assert not involving(cols, txt), [str(c) for c in cols]
    plt.close(fig)


def test_suggestion_is_local_not_bounding_box():
    """A label above a dipping curve should be told to move a little, not below
    the curve's lowest point anywhere in the panel."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    x = np.linspace(0, 1, 200)
    y = np.where(x < 0.7, 0.5, 0.05)               # flat, then a cliff
    ax.plot(x, y, color='black', lw=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    txt = ax.text(0.05, 0.512, 'label', fontsize=8)
    cols = involving(splcollide.find_collisions(fig, min_gap_pt=3.0), txt)
    assert cols, 'label just above the flat part should be reported'
    amount = float(cols[0].suggestion.split('>=')[1].split('pt')[0])
    assert amount < 10, f'suggestion should be local, got {cols[0].suggestion}'
    plt.close(fig)


# ---------------------------------------------------------------------------
# Annotations: the text and its leader are checked as two things
# ---------------------------------------------------------------------------

def _marker_figure():
    """A marker at (0.5, 0.5) with a grey line leaving it to the right."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0.5, 0.9], [0.5, 0.1], color='0.78', lw=0.5)
    ax.plot([0.5], [0.5], linestyle='none', marker='x', markersize=6,
            markeredgewidth=1.2, color='C0')
    return fig, ax


def test_leader_reaching_its_marker_is_not_blamed_on_the_text():
    """The +0.76 case: the glyphs are clear, only the leader meets its target."""
    fig, ax = _marker_figure()
    ann = ax.annotate('+0.76', xy=(0.5, 0.5), xytext=(0.3, 0.5),
                      ha='right', va='center', fontsize=8,
                      arrowprops=dict(arrowstyle='-', shrinkA=1.5, shrinkB=0))
    cols = splcollide.find_collisions(fig, min_gap_pt=1.0)
    assert not involving(cols, ann), [str(c) for c in cols]
    plt.close(fig)


def test_annotation_text_over_a_line_is_still_reported():
    fig, ax = _marker_figure()
    ann = ax.annotate('over the line', xy=(0.2, 0.2), xytext=(0.62, 0.34),
                      ha='center', va='center', fontsize=8,
                      arrowprops=dict(arrowstyle='-'))
    hits = involving(splcollide.find_collisions(fig, min_gap_pt=0.0), ann)
    assert hits and hits[0].kind == 'overlap', [str(c) for c in hits]
    assert hits[0].a_desc.startswith('text'), str(hits[0])
    plt.close(fig)


def test_leader_crossing_another_label_is_reported_as_a_leader():
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    other = ax.text(0.5, 0.5, 'in the way', ha='center', va='center', fontsize=9)
    ann = ax.annotate('source', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                      ha='center', va='center', fontsize=9,
                      arrowprops=dict(arrowstyle='->'))
    hits = involving(splcollide.find_collisions(fig, min_gap_pt=0.0), other)
    assert hits, 'a leader drawn through a label must be reported'
    assert any('leader of "source"' in (c.a_desc + c.b_desc) for c in hits), \
        [str(c) for c in hits]
    assert all(c.a is ann or c.b is ann for c in hits)
    plt.close(fig)


def test_leader_touching_its_artist_target_is_exempt():
    """Like the cross-panel link: an empty-text arrow pointing AT a label."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    target = ax.text(0.7, 0.5, 'target label', ha='center', va='center', fontsize=9)
    link = ax.annotate('', xy=(0.0, 0.5), xycoords=target,
                       xytext=(0.1, 0.5), textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='-|>', shrinkA=0, shrinkB=0))
    cols = splcollide.find_collisions(fig, min_gap_pt=1.0)
    assert not involving(cols, link), [str(c) for c in cols]
    plt.close(fig)


def test_leader_touching_text_at_its_target_point_is_exempt():
    """Same, but targeted by coordinates rather than by the artist."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.7, 0.5, 'x', ha='center', va='center', fontsize=9)
    link = ax.annotate('from here', xy=(0.7, 0.5), xytext=(0.2, 0.5),
                       ha='center', va='center', fontsize=9,
                       arrowprops=dict(arrowstyle='-', shrinkA=2, shrinkB=0))
    cols = splcollide.find_collisions(fig, min_gap_pt=1.0)
    assert not involving(cols, link), [str(c) for c in cols]
    plt.close(fig)


def test_leader_too_short_for_its_shrink_is_reported():
    """matplotlib silently drops shrinkB when the path is shorter than
    shrinkA + shrinkB, and draws the leader right onto its target."""
    fig, ax = _marker_figure()
    short = ax.annotate('+0.76', xy=(0.5, 0.5), xytext=(0.49, 0.5),
                        ha='right', va='center', fontsize=8,
                        arrowprops=dict(arrowstyle='-', shrinkA=1.5, shrinkB=5))
    long = ax.annotate('fine', xy=(0.5, 0.5), xytext=(0.5, 0.9),
                       ha='center', va='center', fontsize=8,
                       arrowprops=dict(arrowstyle='-', shrinkA=1.5, shrinkB=5))
    cols = splcollide.find_collisions(fig, min_gap_pt=1.0)
    hits = [c for c in cols if c.kind == 'short-leader']
    assert [c.a for c in hits] == [short], [str(c) for c in cols]
    assert 'shrinkB' in str(hits[0])
    assert not involving(cols, long), [str(c) for c in cols]
    header = splcollide.format_collisions(hits).splitlines()[0]
    assert '1 leader(s) too short' in header and 'clipped' not in header, header
    plt.close(fig)


def test_exempting_an_annotation_exempts_its_leader():
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, 'in the way', ha='center', va='center', fontsize=9)
    ann = ax.annotate('source', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                      ha='center', va='center', fontsize=9,
                      arrowprops=dict(arrowstyle='->'))
    splcollide.exempt_from_collision_check(ann)
    assert not involving(splcollide.find_collisions(fig), ann)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Composer: the check's findings have to survive a batch render
# ---------------------------------------------------------------------------

def _composer_with_line():
    composer = splcompose.FigureComposer(width_cm=8, height_cm=6, grid_rows=4,
                                         grid_cols=4, dpi=100)
    composer.add_panel('a', row=0, col=0, rowspan=4, colspan=4)
    fig, axes = composer.compose()
    ax = axes['a']
    ax.plot([0, 1], [0.5, 0.5], color='black', lw=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return composer, ax


def test_composer_figure_ignores_autolayout():
    """A stylesheet's figure.autolayout must not re-lay-out composed axes --
    including after a tight-bbox save, which is what to_image() does: savefig
    swaps the engine out and 'restores' it, and restoring a figure that had
    none re-reads figure.autolayout and installs tight_layout after all."""
    import io
    import warnings
    from matplotlib.layout_engine import TightLayoutEngine
    with plt.rc_context({'figure.autolayout': True}):
        composer, ax = _composer_with_line()
        composer.fig.savefig(io.BytesIO(), format='png', bbox_inches='tight')
        assert not isinstance(composer.fig.get_layout_engine(), TightLayoutEngine)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            composer.fig.canvas.draw()
        plt.close(composer.fig)


def test_save_writes_layout_report_and_overlay(tmp_path=None):
    import tempfile
    from pathlib import Path
    out = Path(tmp_path or tempfile.mkdtemp())
    composer, ax = _composer_with_line()
    ax.text(0.2, 0.49, 'on the line', fontsize=8)
    composer.save(out / 'fig', formats=('png',))
    report = (out / 'fig-layout.txt').read_text()
    assert '1 overlap' in report and 'on the line' in report, report
    assert (out / 'fig-layout.png').exists()
    assert len(composer.layout_findings) == 1
    plt.close(composer.fig)


def test_clean_save_says_so_and_removes_a_stale_overlay(tmp_path=None):
    import tempfile
    from pathlib import Path
    out = Path(tmp_path or tempfile.mkdtemp())
    (out / 'fig-layout.png').write_bytes(b'stale')
    composer, ax = _composer_with_line()
    composer.save(out / 'fig', formats=('png',))
    assert 'clean' in (out / 'fig-layout.txt').read_text()
    assert not (out / 'fig-layout.png').exists(), \
        'an overlay from an earlier, dirty save would contradict the report'
    assert composer.layout_findings == []
    plt.close(composer.fig)


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except AssertionError as exc:
                failures += 1
                print(f'FAIL {name}: {exc}')
    print(f'\n{failures} failure(s)')
    raise SystemExit(1 if failures else 0)


# ---------------------------------------------------------------------------
# find_clipped_data
# ---------------------------------------------------------------------------

def _clip_fig():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    return fig, ax


def test_find_clipped_data_reports_points_above_ylim():
    from sciplotlib.collide import find_clipped_data
    fig, ax = _clip_fig()
    ax.plot([0.1, 0.5, 0.9], [0.2, 1.4, 0.3])      # middle point is off the top
    found = find_clipped_data(fig, {'a': ax})
    assert len(found) == 1
    assert found[0].axes_label == 'a'
    assert found[0].axis == 'y'
    assert found[0].n_points == 1
    assert found[0].high == pytest.approx(1.4)
    assert found[0].low is None


def test_find_clipped_data_clean_when_inside():
    from sciplotlib.collide import find_clipped_data
    fig, ax = _clip_fig()
    ax.plot([0.1, 0.5, 0.9], [0.2, 0.8, 0.3])
    assert find_clipped_data(fig, {'a': ax}) == []


def test_find_clipped_data_ignores_axes_anchored_artists():
    """A label pinned to transAxes cannot be clipped by the view limits."""
    from sciplotlib.collide import find_clipped_data
    fig, ax = _clip_fig()
    ax.plot([0.1, 0.9], [0.2, 0.3])
    ax.text(0.5, 1.5, 'title-ish', transform=ax.transAxes)
    ax.axhline(0.5)
    assert find_clipped_data(fig, {'a': ax}) == []


def test_find_clipped_data_respects_exemption():
    from sciplotlib.collide import find_clipped_data, exempt_from_clip_check
    fig, ax = _clip_fig()
    line, = ax.plot([0.1, 0.5], [0.2, 9.0])
    exempt_from_clip_check(line)
    assert find_clipped_data(fig, {'a': ax}) == []


def test_find_clipped_data_tolerance_ignores_points_on_the_limit():
    from sciplotlib.collide import find_clipped_data
    fig, ax = _clip_fig()
    ax.plot([0.1, 0.5], [0.2, 1.0])                # exactly on ylim
    assert find_clipped_data(fig, {'a': ax}) == []


def test_find_clipped_data_sorts_worst_first():
    from sciplotlib.collide import find_clipped_data
    fig, ax = _clip_fig()
    ax.plot([0.1, 0.2], [0.5, 1.1])                # just over
    ax.plot([0.3, 0.4], [0.5, 3.0])                # far over
    found = find_clipped_data(fig, {'a': ax})
    assert [f.high for f in found] == [pytest.approx(3.0), pytest.approx(1.1)]


# ---------------------------------------------------------------------------
# find_crowded_data
# ---------------------------------------------------------------------------

def test_find_crowded_data_flags_marker_against_the_spine():
    """A point just above ylim=0 paints over the axis: inside, but unreadable."""
    from sciplotlib.collide import find_crowded_data
    fig, ax = _clip_fig()
    ax.plot([0.5], [0.001], marker='o', ms=6)
    found = find_crowded_data(fig, {'a': ax})
    assert [f.edge for f in found] == ['bottom']
    assert found[0].n_points == 1
    assert found[0].clearance_pt < 0            # ink crosses the limit


def test_find_crowded_data_clean_with_room_below():
    """The set_bounds fix: view drops below zero, so the marker has clearance."""
    from sciplotlib.collide import find_crowded_data
    fig, ax = _clip_fig()
    ax.set_ylim(-0.1, 1)
    ax.spines['left'].set_bounds(0, 1)
    ax.plot([0.5], [0.001], marker='o', ms=6)
    assert find_crowded_data(fig, {'a': ax}) == []


def test_find_crowded_data_ignores_already_clipped_points():
    """Points outside the view belong to find_clipped_data, not here."""
    from sciplotlib.collide import find_crowded_data
    fig, ax = _clip_fig()
    ax.plot([0.5], [1.6], marker='o', ms=6)
    assert find_crowded_data(fig, {'a': ax}) == []


def test_find_crowded_data_accounts_for_marker_size():
    """The same coordinate is fine small and crowded large."""
    from sciplotlib.collide import find_crowded_data
    fig, ax = _clip_fig()
    small, = ax.plot([0.3], [0.02], marker='o', ms=1)
    assert find_crowded_data(fig, {'a': ax}) == []
    small.set_markersize(40)
    assert len(find_crowded_data(fig, {'a': ax})) == 1


def test_find_crowded_data_respects_the_clip_exemption():
    from sciplotlib.collide import find_crowded_data, exempt_from_clip_check
    fig, ax = _clip_fig()
    line, = ax.plot([0.5], [0.001], marker='o', ms=6)
    exempt_from_clip_check(line)
    assert find_crowded_data(fig, {'a': ax}) == []


def test_find_crowded_data_sorts_worst_first():
    from sciplotlib.collide import find_crowded_data
    fig, ax = _clip_fig()
    ax.plot([0.2], [0.02], marker='o', ms=6)     # almost on the axis
    ax.plot([0.4], [0.12], marker='o', ms=6)     # nearby, but not as close
    found = [f for f in find_crowded_data(fig, {'a': ax}, margin_pt=40)
             if f.edge == 'bottom']
    assert len(found) == 2
    assert found[0].clearance_pt < found[1].clearance_pt


def test_find_crowded_data_ignores_x_edges_by_default():
    """A trace spanning the full x range touches both x limits by construction."""
    from sciplotlib.collide import find_crowded_data
    fig, ax = _clip_fig()
    ax.plot(np.linspace(0, 1, 50), np.full(50, 0.5))
    assert find_crowded_data(fig, {'a': ax}) == []
    widened = find_crowded_data(fig, {'a': ax},
                                edges=('bottom', 'top', 'left', 'right'))
    assert {f.edge for f in widened} == {'left', 'right'}


def test_find_crowded_data_rejects_an_unknown_edge():
    from sciplotlib.collide import find_crowded_data
    fig, ax = _clip_fig()
    with pytest.raises(ValueError, match='unknown edge'):
        find_crowded_data(fig, {'a': ax}, edges=('bottom', 'middle'))
