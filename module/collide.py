"""Collision and clipping checks for composed figures.

Hand-positioned figure elements -- a title nudged above a cartoon, a panel
letter beside a y-label, a legend dropped into a corner -- drift out of place
whenever fonts, panel sizes or the grid change.  The failure is silent: a
descender disappears under an image, two labels touch, a text runs off the
canvas.  This module finds those cases automatically.

The core question is "does artist A's *ink* touch artist B's ink, and if not,
how far apart are they?".  Bounding boxes answer that badly for the two artist
types that matter most here:

* a text's bbox spans the font's full ascent-to-descent band, so boxes touch
  long before the glyphs do;
* an image rendered from an SVG is mostly transparent, so its bbox covers a
  large empty region -- text placed *inside* a cartoon monitor overlaps the
  image's box while overlapping none of its strokes.

So the default precision mode is ``'ink'``: each candidate artist is drawn on
its own into an Agg buffer and reduced to a boolean mask of the pixels it
actually paints.  Overlaps and gaps are then measured between masks.  That
makes deliberate arrangements (text inside a hollow cartoon, an icon in the
empty middle of a screen) come out clean, while a clipped descender or two
labels grazing each other are reported.

Typical use::

    import sciplotlib.collide as splcollide

    collisions = splcollide.check_layout(fig, min_gap_pt=1.0)   # prints a report

or, through the composer::

    composer.check_layout(min_gap_pt=1.0)
    composer.save('figures/figure-3')       # warns about overlaps by default

To silence a deliberate arrangement::

    splcollide.exempt_from_collision_check(txt)   # never report this artist
    splcollide.allow_overlap(txt, image)          # allow just this one pair
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    'Collision',
    'find_collisions',
    'format_collisions',
    'check_layout',
    'exempt_from_collision_check',
    'allow_overlap',
    'save_collision_overlay',
    'NO_COLLISION_CHECK',
]


NO_COLLISION_CHECK = '_spl_no_collision_check'
"""Attribute name set on artists that :func:`find_collisions` skips."""

_ALLOWED_PAIRS = '_spl_overlap_allowed'


def exempt_from_collision_check(artist):
    """Tag an artist so the layout checks ignore it entirely.

    Use for artists whose overlap is the whole point -- a value printed on top
    of a heatmap cell, a label written across a shaded band.

    Returns the artist, so it can be used inline.
    """
    setattr(artist, NO_COLLISION_CHECK, True)
    return artist


def allow_overlap(a, b):
    """Whitelist one specific pair of artists, leaving both checked otherwise.

    Preferable to :func:`exempt_from_collision_check` when a text is meant to
    sit on one particular image but should still be checked against everything
    else.
    """
    for x, y in ((a, b), (b, a)):
        allowed = getattr(x, _ALLOWED_PAIRS, None)
        if allowed is None:
            allowed = set()
            setattr(x, _ALLOWED_PAIRS, allowed)
        allowed.add(id(y))
    return a, b


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class Collision:
    """One reported layout problem.

    Attributes
    ----------
    kind : str
        ``'overlap'``     -- the two artists' ink intersects;
        ``'too-close'``   -- ink is separated by less than ``min_gap_pt``;
        ``'outside-figure'`` -- the artist's ink extends past the canvas, so it
        is cropped in a fixed-size save;
        ``'clipped'``     -- the artist is clipped by its own axes clip box.
    gap_pt : float
        Distance between the two artists' ink, in points.  ``0.0`` for an
        overlap.
    overlap_pt2 : float
        Area of ink intersection in square points (``0.0`` when not
        overlapping).
    suggestion : str
        Smallest axis-aligned move of ``a`` that would clear the problem, e.g.
        ``"move up >= 2.3 pt"``.
    """

    kind: str
    a: object = field(repr=False, default=None)
    b: object = field(repr=False, default=None)
    a_desc: str = ''
    b_desc: str = ''
    panel: str = ''
    gap_pt: float = 0.0
    overlap_pt2: float = 0.0
    occluded: bool = False
    suggestion: str = ''
    xy_fig: tuple = (0.0, 0.0)
    bbox_fig: tuple = (0.0, 0.0, 0.0, 0.0)
    bbox_px: tuple = (0.0, 0.0, 0.0, 0.0)

    @property
    def severity(self):
        """Sort key: overlaps first (largest area), then the tightest gaps."""
        if self.kind == 'overlap':
            return (0, -self.overlap_pt2)
        if self.kind in ('outside-figure', 'clipped'):
            return (1, -self.overlap_pt2)
        return (2, self.gap_pt)

    def __str__(self):
        where = f"[{self.panel}] " if self.panel else ''
        if self.kind == 'overlap':
            what = (f"{self.a_desc} overlaps {self.b_desc}"
                    f" ({self.overlap_pt2:.2f} pt^2"
                    + (", hidden underneath" if self.occluded else "") + ")")
        elif self.kind == 'too-close':
            what = (f"{self.a_desc} is {self.gap_pt:.2f} pt from {self.b_desc}")
        elif self.kind == 'outside-figure':
            what = f"{self.a_desc} extends outside the figure ({self.b_desc})"
        else:
            what = f"{self.a_desc} is clipped by {self.b_desc}"
        tail = f"  ->  {self.suggestion}" if self.suggestion else ''
        return f"{where}{self.kind}: {what}{tail}"


# ---------------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------------

@dataclass
class _Ink:
    artist: object
    kind: str
    desc: str
    zorder: float
    draw_index: int
    bbox: tuple                      # ink bbox in display px, (x0, y0, x1, y1), y up
    raw_bbox: tuple = None           # get_window_extent, before ink tightening
    mask: object = None              # bool array, rows top-down in canvas px
    r0: int = 0                      # canvas row of mask[0, 0]
    c0: int = 0                      # canvas col of mask[0, 0]
    panel: str = ''
    descendants: frozenset = frozenset()
    clipped_px: float = 0.0          # ink lost to the artist's own clip box
    clipped_side: str = ''
    edge_side: str = ''              # canvas edge the painted ink runs into


def _shorten(text, n=28):
    text = ' '.join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + '…'


def _artist_kind(artist):
    import matplotlib.collections as mcoll
    import matplotlib.image as mimage
    import matplotlib.text as mtext
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.patches import Patch
    from matplotlib.spines import Spine

    if isinstance(artist, mtext.Text):
        return 'text'
    if isinstance(artist, (mimage.BboxImage, OffsetImage)):
        # internals of an AnnotationBbox / OffsetBox, which is reported instead
        return None
    if isinstance(artist, mimage._ImageBase):
        return 'image'
    if isinstance(artist, AnnotationBbox):
        # place_image() overlays: an AnnotationBbox wrapping an OffsetImage.
        # AnnotationBoxes holding text are covered by their child Text artists.
        if any(isinstance(c, OffsetImage) for c in artist.findobj(OffsetImage)):
            return 'image'
        return None
    if isinstance(artist, Legend):
        return 'legend'
    if isinstance(artist, Spine):
        return 'spine'
    if isinstance(artist, Line2D):
        return 'line'
    if isinstance(artist, (mcoll.LineCollection, mcoll.PathCollection,
                           mcoll.EllipseCollection)):
        return 'line'
    if isinstance(artist, (mcoll.QuadMesh, mcoll.PolyCollection)):
        # Shaded regions and meshes: writing a label across a pale error band or
        # a heatmap cell is ordinary practice, so these are opt-in only.
        return 'fill'
    if isinstance(artist, mcoll.Collection):
        return 'line'
    if isinstance(artist, Patch):
        return 'patch'
    return None


def _describe(artist, kind):
    if kind == 'text':
        return f'text "{_shorten(artist.get_text())}"'
    if kind == 'image':
        from matplotlib.offsetbox import OffsetImage
        imgs = artist.findobj(OffsetImage)
        arr = imgs[0].get_data() if imgs else getattr(artist, '_A', None)
        if arr is not None:
            return f'image {arr.shape[1]}x{arr.shape[0]}'
        return 'image'
    if kind == 'legend':
        return 'legend'
    if kind == 'spine':
        return f'{getattr(artist, "spine_type", "?")} spine'
    if kind == 'line':
        from matplotlib.lines import Line2D
        label = artist.get_label() or ''
        named = f' "{_shorten(label)}"' if label and not label.startswith('_') else ''
        if isinstance(artist, Line2D):
            return f'line{named} ({_colour_name(artist.get_color())})'
        n = len(getattr(artist, 'get_offsets', lambda: [])() or [])
        return f'{type(artist).__name__}{named}' + (f' (n={n})' if n else '')
    return type(artist).__name__


def _stroke_width(artist):
    """Line width of an artist in points, or 0 if it has none."""
    for getter in ('get_linewidth', 'get_linewidths'):
        try:
            value = getattr(artist, getter)()
        except Exception:
            continue
        if value is None:
            continue
        if np.ndim(value) > 0:
            value = max(value) if len(value) else 0
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _colour_name(colour):
    import matplotlib.colors as mcolors
    try:
        rgba = mcolors.to_rgba(colour)
    except Exception:
        return str(colour)
    for name, value in mcolors.CSS4_COLORS.items():
        if mcolors.to_rgba(value) == rgba:
            return name
    return mcolors.to_hex(rgba)


def _is_skippable(artist, kind):
    if not artist.get_visible():
        return True
    if getattr(artist, NO_COLLISION_CHECK, False):
        return True
    if kind == 'text' and not artist.get_text().strip():
        return True
    if kind == 'patch':
        # the axes background patch and figure patch are not content
        ax = getattr(artist, 'axes', None)
        if ax is not None and artist is ax.patch:
            return True
        if artist is getattr(artist.get_figure(), 'patch', None):
            return True
    return False


def _panel_boxes(fig, renderer):
    """(label, bbox) for every axes the composer stamped with a panel label."""
    boxes = []
    for ax in fig.axes:
        label = getattr(ax, '_sciplotlib_panel', None)
        if label:
            bb = ax.get_window_extent(renderer)
            boxes.append((label, (bb.x0, bb.y0, bb.x1, bb.y1)))
    return boxes


def _inset_parents(fig):
    """child axes -> parent axes, so an inset's panel can be looked up."""
    parents = {}
    stack = list(fig.axes)
    while stack:
        ax = stack.pop()
        for child in getattr(ax, 'child_axes', ()) or ():
            parents[child] = ax
            stack.append(child)
    return parents


def _panel_for(item, panel_boxes, inset_parents):
    """Panel an artist belongs to.

    Prefers the owning axes (walking up through inset axes, whose panel label
    lives on the parent), and falls back to the panel whose box contains the
    artist -- or, failing that, the nearest one -- for figure-level artists.
    """
    ax = getattr(item.artist, 'axes', None)
    seen = 0
    while ax is not None and seen < 10:
        label = getattr(ax, '_sciplotlib_panel', None)
        if label:
            return label
        ax = inset_parents.get(ax)
        seen += 1

    bbox = item.raw_bbox
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    best, best_d = '', None
    for label, (x0, y0, x1, y1) in panel_boxes:
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return label
        dx = max(x0 - cx, 0, cx - x1)
        dy = max(y0 - cy, 0, cy - y1)
        d = dx * dx + dy * dy
        if best_d is None or d < best_d:
            best, best_d = label, d
    return best


def _axis_furniture(fig):
    """Map out each axes' own tick labels, tick marks, gridlines and spines.

    A tick label is *meant* to sit a point or two from its tick and spine -- that
    is what ``tick_pad`` is for -- so those pairs must not be reported as being
    too close, while a hand-placed label brushing the same spine must be.
    Telling the two apart needs to know which axes each artist belongs to.

    Returns ``(tick_labels, furniture, skip)``: the first two map artist id ->
    owning axes id, the third is a set of ids to leave out altogether
    (gridlines, and the axes/figure background patches).
    """
    tick_labels, furniture, gridlines = {}, {}, set()
    backgrounds = {id(getattr(fig, 'patch', None))}
    stack = list(fig.axes)
    seen = set()
    while stack:
        ax = stack.pop()
        if id(ax) in seen:
            continue
        seen.add(id(ax))
        stack.extend(getattr(ax, 'child_axes', ()) or ())
        # The axes background fills the whole data area, so every label inside
        # the axes would "overlap" it.  Its `.axes` attribute is not reliably
        # set, so collect it by identity from the axes instead.
        backgrounds.add(id(getattr(ax, 'patch', None)))
        for spine in getattr(ax, 'spines', {}).values():
            furniture[id(spine)] = id(ax)
        for axis in (getattr(ax, 'xaxis', None), getattr(ax, 'yaxis', None)):
            if axis is None:
                continue
            offset = getattr(axis, 'offsetText', None)
            if offset is not None:
                tick_labels[id(offset)] = id(ax)
            try:
                ticks = list(axis.get_major_ticks()) + list(axis.get_minor_ticks())
            except Exception:
                continue
            for tick in ticks:
                for label in (getattr(tick, 'label1', None), getattr(tick, 'label2', None)):
                    if label is not None:
                        tick_labels[id(label)] = id(ax)
                for line in (getattr(tick, 'tick1line', None),
                             getattr(tick, 'tick2line', None)):
                    if line is not None:
                        furniture[id(line)] = id(ax)
                grid = getattr(tick, 'gridline', None)
                if grid is not None:
                    gridlines.add(id(grid))
    gridlines |= backgrounds
    gridlines.discard(id(None))
    return tick_labels, furniture, gridlines


def _record_drawn_ids(fig):
    """ids of the artists a real draw of ``fig`` actually paints.

    Whether an artist ends up on the page is not something its own state
    reveals.  An axis keeps a Tick for every position its locator produced and
    silently skips the ones outside the view limits; ``ax.axis('off')`` leaves
    ``get_visible()`` True on every tick label and axis label while removing
    the whole Axis from the draw list.  Any of those artists will cheerfully
    paint itself if asked directly, which is exactly what the ink pass does --
    so it would measure labels the reader never sees.

    Rather than trying to predict matplotlib's rules, wrap every artist's
    ``draw`` for one real pass and note which ones get called.
    """
    import matplotlib.artist as martist

    recorded = set()
    patched = []

    def wrap(artist):
        original = artist.draw

        def recording_draw(renderer, *args, **kwargs):
            recorded.add(id(artist))
            return original(renderer, *args, **kwargs)

        artist.draw = recording_draw
        patched.append(artist)

    for artist in fig.findobj(martist.Artist):
        try:
            wrap(artist)
        except Exception:
            pass
    try:
        fig.canvas.draw()
    finally:
        for artist in patched:
            try:
                del artist.draw           # unshadow the class method
            except AttributeError:
                pass
    return recorded


def _image_ink_bbox(artist, renderer):
    """Alpha-tight display bbox of an image artist, from its own array.

    An SVG rendered to RGBA is mostly transparent, and an ``AnnotationBbox``
    pads its extent by the frame padding whether or not a frame is drawn -- so
    the reported window extent can overstate an icon by several points in every
    direction.  Reading the array's alpha channel gives the real outline, and
    unlike a rasterised mask it stays valid for ink that falls off the canvas.
    """
    import matplotlib.image as mimage
    from matplotlib.offsetbox import OffsetImage

    children = ([artist] if isinstance(artist, mimage._ImageBase)
                else artist.findobj(OffsetImage))
    boxes = []
    for child in children:
        arr = child.get_data() if isinstance(child, OffsetImage) else child.get_array()
        if arr is None or getattr(arr, 'ndim', 0) != 3 or arr.shape[2] != 4:
            continue
        alpha = np.asarray(arr[..., 3])
        if alpha.dtype.kind == 'f':
            solid = alpha > 0.05
        else:
            solid = alpha > 12
        rows = np.flatnonzero(solid.any(axis=1))
        cols = np.flatnonzero(solid.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            continue
        try:
            bb = child.get_window_extent(renderer)
        except Exception:
            continue
        ny, nx = solid.shape
        # row 0 of the array is drawn at the top of the extent
        boxes.append((
            bb.x0 + bb.width * cols[0] / nx,
            bb.y1 - bb.height * (rows[-1] + 1) / ny,
            bb.x0 + bb.width * (cols[-1] + 1) / nx,
            bb.y1 - bb.height * rows[0] / ny,
        ))
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _collect(fig, renderer, kinds, include, drawn_ids, skip_ids=()):
    """Candidate artists with their bboxes, in draw order.

    ``skip_ids`` drops artists that span a whole region by design -- gridlines
    and axes/figure backgrounds -- which every label inside the axes would
    otherwise be reported as overlapping.
    """
    import matplotlib.artist as martist

    out = []
    seen = set()
    for idx, artist in enumerate(fig.findobj(martist.Artist)):
        if id(artist) in seen or id(artist) not in drawn_ids:
            continue
        if id(artist) in skip_ids:
            continue
        seen.add(id(artist))
        kind = _artist_kind(artist)
        if kind is None or kind not in kinds:
            continue
        if _is_skippable(artist, kind):
            continue
        if include is not None and not include(artist):
            continue
        try:
            bb = artist.get_window_extent(renderer)
        except Exception:
            continue
        if bb is None or not np.isfinite([bb.x0, bb.y0, bb.x1, bb.y1]).all():
            continue
        raw = (bb.x0, bb.y0, bb.x1, bb.y1)
        if kind == 'image':
            tight = _image_ink_bbox(artist, renderer)
            if tight is not None:
                raw = tight
        elif kind in ('line', 'spine', 'patch', 'fill'):
            # A stroke's extent covers its vertices only, so a horizontal line
            # (axhline, a flat segment, an error bar cap) reports zero height and
            # would be dropped as empty -- and it is exactly the kind of element
            # an in-plot label ends up sitting on.  Grow the box by the stroke.
            pad = 0.5 * _stroke_width(artist) * fig.dpi / 72.0
            raw = (raw[0] - pad, raw[1] - pad, raw[2] + pad, raw[3] + pad)
        if raw[2] <= raw[0] or raw[3] <= raw[1]:
            continue                      # nothing with an area to collide with
        out.append(_Ink(
            artist=artist, kind=kind, desc=_describe(artist, kind),
            zorder=float(artist.get_zorder()), draw_index=idx,
            bbox=raw, raw_bbox=raw,
            descendants=frozenset(id(child) for child in artist.findobj()),
        ))

    # An artist wholly contained in another candidate (a legend's own texts,
    # a composite's parts) would otherwise be reported twice, once per level.
    nested = set()
    for item in out:
        for other in out:
            if other is not item and id(item.artist) in other.descendants:
                nested.add(id(item.artist))
                break
    return [item for item in out if id(item.artist) not in nested]


# ---------------------------------------------------------------------------
# Ink masks
# ---------------------------------------------------------------------------

def _ink_mask(renderer, artist, alpha_threshold):
    """Draw one artist alone and return ``(mask, r0, c0)`` of its ink.

    ``mask`` is a cropped boolean array (rows top-down in canvas pixels) of the
    pixels whose alpha clears ``alpha_threshold``; ``r0``/``c0`` locate its
    top-left corner on the canvas.  Returns ``None`` if the artist paints
    nothing visible.
    """
    renderer.clear()
    artist.draw(renderer)
    alpha = np.asarray(renderer.buffer_rgba())[:, :, 3]
    rows = np.flatnonzero(alpha.any(axis=1))
    if rows.size == 0:
        return None
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    band = alpha[r0:r1]
    cols = np.flatnonzero(band.any(axis=0))
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    mask = band[:, c0:c1] >= alpha_threshold
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None                       # only anti-aliasing fringe
    cols = np.flatnonzero(mask.any(axis=0))
    mask = np.ascontiguousarray(mask[rows[0]:rows[-1] + 1,
                                     cols[0]:cols[-1] + 1])
    return mask, r0 + int(rows[0]), c0 + int(cols[0])


def _render_masks(fig, items, alpha_threshold):
    """Fill in ``mask``/``r0``/``c0``/``bbox`` by drawing each artist alone.

    One scratch Agg renderer the size of the figure is cleared and reused, so
    peak memory is a single RGBA buffer regardless of artist count.  Artists
    that refuse to draw in isolation keep their bounding box and are compared
    in bbox terms.

    Labels and images whose bounding box sticks out of their own clip box are
    re-drawn unclipped, which tells us how much ink the clip actually removed --
    the difference between an image with transparent padding hanging over a
    panel edge (harmless) and one whose strokes are cut off (not).
    """
    from matplotlib.backends.backend_agg import RendererAgg

    width = int(np.ceil(fig.bbox.width))
    height = int(np.ceil(fig.bbox.height))
    renderer = RendererAgg(width, height, fig.dpi)

    kept = []
    for item in items:
        try:
            ink = _ink_mask(renderer, item.artist, alpha_threshold)
        except Exception:
            kept.append(item)             # keep it, compared by bbox
            continue
        if ink is None:
            continue                      # paints nothing -> nothing to hit
        item.mask, item.r0, item.c0 = ink
        h, w = item.mask.shape
        item.bbox = (float(item.c0), float(height - (item.r0 + h)),
                     float(item.c0 + w), float(height - item.r0))
        # Ink reaching the outermost pixel row/column was almost certainly cut
        # off there -- the only evidence of cropping available, since a mask can
        # never extend past the buffer it was drawn into.
        for side, hit in (('left', item.c0 <= 0), ('right', item.c0 + w >= width),
                          ('bottom', item.r0 + h >= height), ('top', item.r0 <= 0)):
            if hit:
                item.edge_side = side
                break
        if _clip_check_applies(item):
            _measure_clipping(renderer, item, alpha_threshold, height)
        kept.append(item)
    return kept


def _clip_check_applies(item):
    """Whether losing ink to a clip box is worth reporting for this artist.

    Only labels and *placed* images (an AnnotationBbox from ``place_image``)
    qualify.  Clipping is the intended behaviour for the others:

    * a curve running to the edge of its panel loses ink there by design --
      that is what clipping the data area is for;
    * an ``imshow`` image is cropped by the axes' view limits, so
      ``set_xlim``/``set_ylim`` around a region of a brain map is a *crop*, not
      an accident.
    """
    from matplotlib.image import _ImageBase
    if item.kind == 'text':
        return True
    if item.kind != 'image':
        return False
    return not isinstance(item.artist, _ImageBase)


def _clip_extents(artist):
    """Display-space box an artist's clipping confines it to, or None."""
    boxes = []
    box = artist.get_clip_box()
    if box is not None:
        boxes.append((box.x0, box.y0, box.x1, box.y1))
    path = artist.get_clip_path()
    if path is not None:
        try:
            ext = path.get_extents()
            boxes.append((ext.x0, ext.y0, ext.x1, ext.y1))
        except Exception:
            pass
    if not boxes:
        return None
    return (max(b[0] for b in boxes), max(b[1] for b in boxes),
            min(b[2] for b in boxes), min(b[3] for b in boxes))


def _measure_clipping(renderer, item, alpha_threshold, height):
    """Record how much of an artist's ink its own clipping removes.

    Measured rather than inferred: the artist is re-drawn with clipping off and
    the two ink boxes are compared.  Inferring from the clip box would cry wolf
    on every ``AnnotationBbox``, whose clip path applies to its frame but not to
    the image inside it, and on transparent padding that is clipped away
    without costing any visible ink.
    """
    artist = item.artist
    if not artist.get_clip_on() or item.mask is None:
        return
    clip = _clip_extents(artist)
    slack = 1.0
    if clip is not None:
        raw = item.raw_bbox
        if (raw[0] >= clip[0] - slack and raw[1] >= clip[1] - slack
                and raw[2] <= clip[2] + slack and raw[3] <= clip[3] + slack):
            return                        # nothing sticks out to be clipped
    artist.set_clip_on(False)
    try:
        ink = _ink_mask(renderer, artist, alpha_threshold)
    except Exception:
        return
    finally:
        artist.set_clip_on(True)
    if ink is None:
        return
    mask, r0, c0 = ink
    h, w = mask.shape
    full = (float(c0), float(height - (r0 + h)), float(c0 + w), float(height - r0))
    drawn = item.bbox
    overs = [drawn[0] - full[0], drawn[1] - full[1], full[2] - drawn[2], full[3] - drawn[3]]
    worst = int(np.argmax(overs))
    if overs[worst] > slack:
        item.clipped_px = float(overs[worst])
        item.clipped_side = ['left', 'bottom', 'right', 'top'][worst]


def _mask_window(item, r0, c0, nrows, ncols):
    """Place a cropped mask into a window of the canvas, as a bool array."""
    out = np.zeros((nrows, ncols), dtype=bool)
    if item.mask is None:
        return out
    h, w = item.mask.shape
    ra, rb = item.r0 - r0, item.r0 - r0 + h
    ca, cb = item.c0 - c0, item.c0 - c0 + w
    sra, srb = max(ra, 0), min(rb, nrows)
    sca, scb = max(ca, 0), min(cb, ncols)
    if sra >= srb or sca >= scb:
        return out
    out[sra:srb, sca:scb] = item.mask[sra - ra:srb - ra, sca - ca:scb - ca]
    return out


def _min_distance_px(a, b, max_px):
    """Smallest pixel distance between two boolean masks (``inf`` if > max_px)."""
    try:
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(~a)
        d = dist[b].min() if b.any() else np.inf
        return float(d)
    except ImportError:
        pass
    # Chebyshev fallback: dilate a until it reaches b (no scipy needed).
    grown = a
    for k in range(1, int(np.ceil(max_px)) + 2):
        grown = (grown
                 | np.roll(grown, 1, 0) | np.roll(grown, -1, 0)
                 | np.roll(grown, 1, 1) | np.roll(grown, -1, 1))
        if (grown & b).any():
            return float(k)
    return np.inf


def _distance_field(mask):
    """Distance in pixels from every cell to the nearest True cell of ``mask``."""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return None
    return distance_transform_edt(~mask)


def _mask_suggestion(mover, other, gap_px, px_per_pt, max_px):
    """Smallest axis-aligned shift of ``mover``'s ink that clears ``other``'s.

    Measured by sliding the mask, because the axis-aligned distance between two
    bounding boxes is badly wrong for anything non-convex: to clear a *curve's*
    box a label would have to move below the curve's lowest point anywhere in the
    panel, when a point or two of local clearance is all it needs.

    ``mover`` and ``other`` are boolean arrays on a common window, already padded
    by ``max_px``.  Returns a description, or '' if no direction works.
    """
    field = _distance_field(other)
    if field is None:
        return ''
    rows, cols = np.nonzero(mover)
    if rows.size == 0:
        return ''
    # distance 0 means the pixel sits on the other artist, so even "no overlap"
    # asks for at least one pixel of daylight
    gap_px = max(gap_px, 1.0)
    best = None
    for name, (dr, dc) in (('up', (-1, 0)), ('down', (1, 0)),
                           ('left', (0, -1)), ('right', (0, 1))):
        for k in range(1, int(max_px) + 1):
            r, c = rows + dr * k, cols + dc * k
            if r.min() < 0 or c.min() < 0 or r.max() >= field.shape[0] or c.max() >= field.shape[1]:
                break
            if field[r, c].min() >= gap_px:
                if best is None or k < best[1]:
                    best = (name, k)
                break
    if best is None:
        return ''
    return f'move {best[0]} >= {max(best[1] / px_per_pt, 0.1):.1f} pt'


def _suggest_from_masks(mover, other, gap_px, px_per_pt, max_pt=24.0):
    """Build a common padded window for two _Ink items and slide one clear."""
    max_px = int(np.ceil(max_pt * px_per_pt))
    pad = max_px + 2
    r0 = int(min(mover.r0, other.r0)) - pad
    c0 = int(min(mover.c0, other.c0)) - pad
    r1 = int(max(mover.r0 + mover.mask.shape[0],
                 other.r0 + other.mask.shape[0])) + pad
    c1 = int(max(mover.c0 + mover.mask.shape[1],
                 other.c0 + other.mask.shape[1])) + pad
    ma = _mask_window(mover, r0, c0, r1 - r0, c1 - c0)
    mb = _mask_window(other, r0, c0, r1 - r0, c1 - c0)
    return _mask_suggestion(ma, mb, gap_px, px_per_pt, max_px)


def _bbox_gap_px(a, b):
    """Axis-aligned gap between two bboxes (0 if they intersect)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return float(np.hypot(dx, dy))


def _suggest(a_bbox, b_bbox, gap_px, px_per_pt):
    """Smallest axis-aligned shift of ``a`` that clears ``b`` by ``gap_px``."""
    ax0, ay0, ax1, ay1 = a_bbox
    bx0, by0, bx1, by1 = b_bbox
    moves = {
        'up': by1 - ay0 + gap_px,
        'down': ay1 - by0 + gap_px,
        'left': ax1 - bx0 + gap_px,
        'right': bx1 - ax0 + gap_px,
    }
    moves = {k: v for k, v in moves.items() if v > 0}
    if not moves:
        return ''
    direction = min(moves, key=moves.get)
    return f'move {direction} >= {moves[direction] / px_per_pt:.1f} pt'


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

DEFAULT_KINDS = ('text', 'image', 'line', 'spine')
"""Artist types checked by default: labels against each other and against the
drawing -- curves, markers, axis spines.

Drawn shapes (``'patch'``), area fills and meshes (``'fill'``) are opt-in.
Writing a label across a shape, a pale error band or a heatmap cell is ordinary
practice -- a '?' on a cartoon monitor, a value in a matrix cell -- so checking
them reports mostly deliberate work."""


def find_collisions(fig, min_gap_pt=1.0, kinds=DEFAULT_KINDS,
                    precision='ink', check_dpi=200, alpha_threshold=16,
                    ignore_contained=None, check_figure_bounds=True,
                    require_text=None, include=None, panel_boxes=None):
    """Find artists in ``fig`` that overlap or sit closer than ``min_gap_pt``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to check.  It is drawn (and, in ``'ink'`` mode, temporarily
        re-drawn at ``check_dpi``); nothing about it is permanently changed.
    min_gap_pt : float, default 1.0
        Required clear space between two artists' ink, in points.  Pass ``0.0``
        to report overlaps only.

        A point is a deliberately small ask, and it catches the mistake that
        overlap-only checking cannot see: a label anchored *on* something --
        ``ax.text(0, y, ..., transform=ax.transAxes)`` sits on the y-axis spine
        and clears it only by the first glyph's side bearing, which is well
        under a point.  Measured across four full-page paper figures, a 1 pt
        requirement produced four findings, all real.
    kinds : tuple of str
        Which artist types take part: any of ``'text'``, ``'image'``,
        ``'line'`` (curves, markers), ``'spine'``, ``'patch'``, ``'legend'``,
        ``'fill'`` (shaded bands, meshes).  Defaults to
        :data:`DEFAULT_KINDS`, which is everything except ``'fill'``.
        Gridlines are never included: one spans the whole plot area, so every
        label inside the axes is near one by construction.
    precision : {'ink', 'bbox'}
        ``'ink'`` (default) measures between the pixels each artist actually
        paints -- the only mode that gets transparent images and font
        descenders right.  ``'bbox'`` measures between bounding boxes: faster,
        and deliberately pessimistic about text (a bbox spans the font's full
        ascent-to-descent band whether or not the string has tall or hanging
        glyphs).
    check_dpi : int, default 200
        Resolution of the ink masks.  Higher is more precise (1 pt is
        ``check_dpi / 72`` px) and slower.
    alpha_threshold : int, default 16
        Alpha (0-255) at which a pixel counts as ink.  Above 0 so that
        anti-aliasing fringe does not inflate every artist by a pixel.
    ignore_contained : bool or None
        Skip pairs where one artist's ink box lies entirely inside the
        other's -- text written across a heatmap, for instance.  Defaults to
        ``False`` in ``'ink'`` mode (masks already exclude the common
        deliberate cases) and ``True`` in ``'bbox'`` mode.
    check_figure_bounds : bool, default True
        Also report ink that extends past the canvas, which a fixed-size save
        crops.
    require_text : bool or None
        Only report a pair if one of the two is text.  Defaults to True
        whenever any drawing kind (``'line'``, ``'spine'``, ``'patch'``,
        ``'fill'``) is checked, because plot elements are *supposed* to touch
        each other -- curves cross, markers pile up, a spine meets its ticks --
        while a label touching any of them is a defect.  With only text and
        images in play it defaults to False, so image-to-image spacing in a
        schematic is still measured.
    include : callable, optional
        ``include(artist) -> bool`` filter applied to every candidate.
    panel_boxes : list, optional
        Pre-computed ``[(label, bbox), ...]`` used to name panels in the
        report.  Derived from the figure's axes when omitted.

    Returns
    -------
    list of Collision
        Sorted worst-first.
    """
    if precision not in ('ink', 'bbox'):
        raise ValueError("precision must be 'ink' or 'bbox'")
    if ignore_contained is None:
        ignore_contained = (precision == 'bbox')

    kinds = tuple(kinds)
    if require_text is None:
        require_text = any(k in kinds for k in ('line', 'spine', 'patch', 'fill'))
    orig_dpi = fig.dpi
    use_ink = precision == 'ink'
    try:
        if use_ink and check_dpi and abs(check_dpi - orig_dpi) > 1e-9:
            # Points-based geometry is dpi-invariant, so measuring at a lower
            # dpi keeps the mask buffers small without moving anything.
            fig.set_dpi(check_dpi)
        # first draw settles the layout and creates lazily-built artists (ticks
        # appear during draw), the second records what is really painted
        fig.canvas.draw()
        drawn_ids = _record_drawn_ids(fig)
        renderer = fig.canvas.get_renderer()
        px_per_pt = fig.dpi / 72.0
        min_gap_px = min_gap_pt * px_per_pt

        if panel_boxes is None:
            panel_boxes = _panel_boxes(fig, renderer)

        tick_labels, furniture, skip_ids = _axis_furniture(fig)
        collected = _collect(fig, renderer, kinds, include, drawn_ids, skip_ids)
        items = _render_masks(fig, collected, alpha_threshold) if use_ink else collected
        inset_parents = _inset_parents(fig)
        for item in collected:
            item.panel = _panel_for(item, panel_boxes, inset_parents)

        collisions = []
        fig_w, fig_h = fig.bbox.width, fig.bbox.height

        if check_figure_bounds:
            # Measured on bounding boxes, not ink: ink is whatever landed on the
            # canvas, so by definition it never reaches past the edge.  Tolerate
            # a point of overshoot, since a text box spans the full font band.
            box_tol = max(2.0, 0.5 * px_per_pt)
            for item in collected:
                # Some artists cannot report an extent until they are drawn --
                # mplot3d's Text3D projects its position inside draw(), so
                # get_window_extent beforehand describes a point in the wrong
                # place entirely.  Where the painted ink does not sit inside the
                # reported box, the box is not to be trusted; the ink already
                # told us the artist is on the canvas.
                if item.mask is not None and not _contains(item.raw_bbox, item.bbox,
                                                           tol=box_tol):
                    continue
                # For an artist that painted something, the box overshooting the
                # canvas proves nothing -- a text box spans the font's full
                # ascent-to-descent band, so an 'x' label sitting a few points
                # above the bottom edge overshoots while every pixel of it is on
                # the page.  Ink running into the outermost pixel is the tell.
                if item.mask is not None and not item.edge_side:
                    continue
                x0, y0, x1, y1 = item.raw_bbox
                over = max(-x0, -y0, x1 - fig_w, y1 - fig_h)
                if over > px_per_pt:
                    side = item.edge_side or ['left', 'bottom', 'right', 'top'][
                        int(np.argmax([-x0, -y0, x1 - fig_w, y1 - fig_h]))]
                    collisions.append(Collision(
                        kind='outside-figure', a=item.artist, b=None,
                        a_desc=item.desc, b_desc=f'up to {over / px_per_pt:.1f} pt past {side}',
                        panel=item.panel, gap_pt=0.0,
                        overlap_pt2=over / px_per_pt,
                        suggestion=(f'move {"right" if side == "left" else "left" if side == "right" else "up" if side == "bottom" else "down"}'
                                    f' >= {over / px_per_pt:.1f} pt'),
                        xy_fig=((x0 + x1) / 2 / fig_w, (y0 + y1) / 2 / fig_h),
                        bbox_fig=(x0 / fig_w, y0 / fig_h, x1 / fig_w, y1 / fig_h),
                        bbox_px=item.raw_bbox,
                    ))

        for item in items:
            if item.clipped_px > 0:
                collisions.append(Collision(
                    kind='clipped', a=item.artist, b=None,
                    a_desc=item.desc,
                    b_desc=(f'its clip box, losing {item.clipped_px / px_per_pt:.1f} pt '
                            f'on the {item.clipped_side}'),
                    panel=item.panel,
                    overlap_pt2=item.clipped_px / px_per_pt,
                    xy_fig=((item.bbox[0] + item.bbox[2]) / 2 / fig_w,
                            (item.bbox[1] + item.bbox[3]) / 2 / fig_h),
                    bbox_fig=(item.bbox[0] / fig_w, item.bbox[1] / fig_h,
                              item.bbox[2] / fig_w, item.bbox[3] / fig_h),
                    bbox_px=item.bbox,
                ))

        search_px = min_gap_px + 2.0
        canvas_h = int(np.ceil(fig.bbox.height))
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                if require_text and 'text' not in (a.kind, b.kind):
                    continue
                col = _pair_collision(a, b, min_gap_px, search_px, px_per_pt,
                                      use_ink, ignore_contained,
                                      fig_w, fig_h, canvas_h,
                                      own_furniture=_is_own_furniture(
                                          a, b, tick_labels, furniture))
                if col is not None:
                    collisions.append(col)

        collisions.sort(key=lambda c: c.severity)
        return collisions
    finally:
        if fig.dpi != orig_dpi:
            fig.set_dpi(orig_dpi)
            fig.canvas.draw()


def _is_own_furniture(a, b, tick_labels, furniture):
    """True if this is a tick label beside its own axes' spine or tick marks."""
    for x, y in ((a, b), (b, a)):
        owner = tick_labels.get(id(x.artist))
        if owner is not None and furniture.get(id(y.artist)) == owner:
            return True
    return False


def _pair_collision(a, b, min_gap_px, search_px, px_per_pt, use_ink,
                    ignore_contained, fig_w, fig_h, canvas_h=None,
                    own_furniture=False):
    # never compare an artist with something it contains, or with itself
    if id(b.artist) in a.descendants or id(a.artist) in b.descendants:
        return None
    if id(b.artist) in getattr(a.artist, _ALLOWED_PAIRS, ()):
        return None
    # cheap reject on boxes before touching masks
    if _bbox_gap_px(a.bbox, b.bbox) > search_px:
        return None
    if ignore_contained and (_contains(a.bbox, b.bbox) or _contains(b.bbox, a.bbox)):
        return None

    overlap_px2 = 0.0
    hit_box = None                        # bbox of the ink that actually clashes
    if use_ink and a.mask is not None and b.mask is not None:
        pad = int(np.ceil(search_px)) + 1
        r0 = int(min(a.r0, b.r0)) - pad
        c0 = int(min(a.c0, b.c0)) - pad
        r1 = int(max(a.r0 + a.mask.shape[0], b.r0 + b.mask.shape[0])) + pad
        c1 = int(max(a.c0 + a.mask.shape[1], b.c0 + b.mask.shape[1])) + pad
        ma = _mask_window(a, r0, c0, r1 - r0, c1 - c0)
        mb = _mask_window(b, r0, c0, r1 - r0, c1 - c0)
        both = ma & mb
        overlap_px2 = float(both.sum())
        if overlap_px2 > 0:
            gap_px = 0.0
            if canvas_h is not None:
                rows = np.flatnonzero(both.any(axis=1))
                cols = np.flatnonzero(both.any(axis=0))
                hit_box = (float(c0 + cols[0]),
                           float(canvas_h - (r0 + rows[-1] + 1)),
                           float(c0 + cols[-1] + 1),
                           float(canvas_h - (r0 + rows[0])))
        else:
            gap_px = _min_distance_px(ma, mb, search_px)
    else:
        gap_px = _bbox_gap_px(a.bbox, b.bbox)
        if gap_px == 0.0:
            ox = min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0])
            oy = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
            overlap_px2 = max(ox, 0.0) * max(oy, 0.0)

    if overlap_px2 <= 0 and gap_px >= min_gap_px:
        return None
    # A tick label sits a point or two from its own spine and tick marks because
    # tick_pad says so; only an actual overlap there is news.
    if own_furniture and overlap_px2 <= 0:
        return None

    # report the more movable artist first: text over image, smaller over larger
    mover, other = a, b
    if a.kind != 'text' and b.kind == 'text':
        mover, other = b, a
    elif a.kind == b.kind:
        area_a = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1])
        area_b = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1])
        if area_b < area_a:
            mover, other = b, a

    overlapping = overlap_px2 > 0
    occluded = overlapping and (
        (other.zorder, other.draw_index) > (mover.zorder, mover.draw_index))
    x0, y0, x1, y1 = mover.bbox
    # Clear the ink that actually clashes, not the other artist's whole box:
    # for a hollow cartoon the latter would advise moving a label right across
    # the drawing when a fraction of a point away from one stroke will do.
    clear_of = hit_box if hit_box is not None else other.bbox
    suggestion = ''
    if use_ink and mover.mask is not None and other.mask is not None:
        suggestion = _suggest_from_masks(mover, other, min_gap_px, px_per_pt)
    if not suggestion:
        suggestion = _suggest(mover.bbox, clear_of, min_gap_px, px_per_pt)
    return Collision(
        kind='overlap' if overlapping else 'too-close',
        a=mover.artist, b=other.artist,
        a_desc=mover.desc, b_desc=other.desc,
        panel=mover.panel or other.panel,
        gap_pt=0.0 if overlapping else gap_px / px_per_pt,
        overlap_pt2=overlap_px2 / (px_per_pt ** 2),
        occluded=occluded,
        suggestion=suggestion,
        xy_fig=((x0 + x1) / 2 / fig_w, (y0 + y1) / 2 / fig_h),
        bbox_fig=(x0 / fig_w, y0 / fig_h, x1 / fig_w, y1 / fig_h),
        bbox_px=mover.bbox,
    )


def _contains(outer, inner, tol=0.5):
    return (outer[0] - tol <= inner[0] and outer[1] - tol <= inner[1]
            and outer[2] + tol >= inner[2] and outer[3] + tol >= inner[3])


def format_collisions(collisions, min_gap_pt=0.0, limit=None, header=True):
    """Render a list of :class:`Collision` as a readable report."""
    lines = []
    if header:
        n_over = sum(1 for c in collisions if c.kind == 'overlap')
        n_close = sum(1 for c in collisions if c.kind == 'too-close')
        n_out = len(collisions) - n_over - n_close
        if not collisions:
            return (f'Layout check: clean (no overlaps'
                    + (f', nothing closer than {min_gap_pt:g} pt)'
                       if min_gap_pt else ')'))
        parts = [f'{n_over} overlap(s)']
        if min_gap_pt:
            parts.append(f'{n_close} pair(s) closer than {min_gap_pt:g} pt')
        if n_out:
            parts.append(f'{n_out} artist(s) off-canvas or clipped')
        lines.append('Layout check: ' + ', '.join(parts))
    shown = collisions if limit is None else collisions[:limit]
    for i, c in enumerate(shown, start=1):
        # numbered to match the boxes drawn by save_collision_overlay
        lines.append(f'  {i:>2}. {c}')
    if limit is not None and len(collisions) > limit:
        lines.append(f'  ... and {len(collisions) - limit} more')
    return '\n'.join(lines)


def check_layout(fig, min_gap_pt=1.0, verbose=True, limit=None, **kwargs):
    """Run :func:`find_collisions` and print a report.

    Returns the list of collisions, so it can also be used as an assertion in
    a build script::

        assert not splcollide.check_layout(fig, min_gap_pt=1.0)
    """
    collisions = find_collisions(fig, min_gap_pt=min_gap_pt, **kwargs)
    if verbose:
        print(format_collisions(collisions, min_gap_pt=min_gap_pt, limit=limit))
    return collisions


def save_collision_overlay(fig, collisions, path, color='#e1261c', dpi=150,
                           limit=None, pad_pt=1.5, label=True):
    """Save a copy of the figure with each reported collision boxed in red.

    Boxes are placed in figure fractions (``Collision.bbox_fig``), not display
    pixels, so they land correctly even though the checks measure at a lower
    dpi than the figure's own.  Everything added is removed again afterwards,
    leaving the figure exactly as it was.
    """
    from matplotlib.patches import Rectangle

    added = []
    shown = collisions if limit is None else collisions[:limit]
    px = pad_pt / 72.0
    for i, c in enumerate(shown, start=1):
        fx0, fy0, fx1, fy1 = c.bbox_fig
        dx = px / fig.get_figwidth()
        dy = px / fig.get_figheight()
        rect = Rectangle((fx0 - dx, fy0 - dy),
                         (fx1 - fx0) + 2 * dx, (fy1 - fy0) + 2 * dy,
                         transform=fig.transFigure, facecolor='none',
                         edgecolor=color, linewidth=0.6, zorder=1e6,
                         clip_on=False)
        fig.add_artist(rect)
        added.append(rect)
        if label:
            txt = fig.text(fx1 + dx, fy1 + dy, str(i), color=color,
                           fontsize=4.5, ha='left', va='bottom', zorder=1e6)
            added.append(txt)
    try:
        fig.savefig(path, dpi=dpi)
        print(f'Saved: {path}')
    finally:
        for artist in added:
            artist.remove()
    return path
