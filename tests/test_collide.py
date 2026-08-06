"""Checks for sciplotlib.collide.

Runs under pytest, or standalone: ``python tests/test_collide.py``.

The cases here are the ones that distinguish ink-based checking from
bounding-box checking, plus the two classes of artist that lie about their
geometry (culled tick labels, tick/axis labels under ``axis('off')``).
"""
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
