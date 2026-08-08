"""
Tagless position overrides for FigureComposer figures.

The drag editor writes an *overrides* JSON keyed by stable, auto-derived
addresses (no manual gids), and :meth:`FigureComposer.apply_overrides` re-applies
them on every render — so hand-tuned positions survive re-runs without editing
plot code.

Addresses (stable, zero user tagging)
-------------------------------------
    panel:<label>               # the panel axes itself -> set_position (delta)
    panel:<label>/xlabel        # x-axis label           -> set_label_coords
    panel:<label>/ylabel        # y-axis label           -> set_label_coords
    panel:<label>/colorbar[:i]  # colorbar axes          -> set_position
    panel:<label>/inset[:i]     # inset axes             -> set_position
    panel:<label>/image[:i]     # place_image() overlay  -> .xy (+ zoom)
    panel:<label>/text[:i]      # ax.text() artist       -> set_position

These rely only on (a) the panel label the composer already stamps on each
axes (``ax._sciplotlib_panel``) and (b) matplotlib's own named roles (axis
labels; colorbar axes carry a ``._colorbar`` back-reference) or stable creation
order within the panel.  The common single-artist case needs no index.

Fingerprints
------------
Indexed kinds carry a *fingerprint* (the text string for ``text``, the image
shape for ``image``) that is matched **before** the index on apply, so inserting
another text or image into a panel does not silently shuffle every override onto
the wrong artist.  A fingerprint that no longer matches anything falls back to
the index and warns.

Panel positions are deltas
--------------------------
``panel:<label>`` stores the drag as a **delta** ``[dx0, dy0, dw, dh]`` against
the position the composer's ``fit_axes_to_cells`` computed, not as absolute
bounds.  That way the hand-tweak still means "nudge this panel left a bit" after
you change the figure size, grid, or a neighbouring panel's content.  The
absolute bounds at the time of the drag are recorded alongside as ``value`` for
reference, and ``"mode": "absolute"`` in an entry switches to using them.

Because ``fit_axes_to_cells`` re-places panel axes, panel deltas must be applied
*after* it — :meth:`FigureComposer.apply_overrides` therefore defers them, and
``to_image``/``save``/``launch_editor`` apply them at the end of the
normalisation chain.

JSON format
-----------
    {
      "panel:b":          {"kind": "panel", "value": [x0, y0, w, h],
                           "delta": [dx0, dy0, dw, dh]},
      "panel:j/colorbar": {"kind": "axes",  "value": [x0, y0, w, h]},
      "panel:m/xlabel":   {"kind": "xlabel","value": [x, y],
                           "fingerprint": "Trials in session"},
      "panel:a/image:0":  {"kind": "image", "value": [x, y], "zoom": 0.12,
                           "fingerprint": "image 512x512x4"},
      "panel:a/text:1":   {"kind": "text",  "value": [x, y],
                           "fingerprint": "MP computer"}
    }
"""

from __future__ import annotations

import json

from matplotlib.offsetbox import AnnotationBbox

# Kinds whose stored ``value`` is meaningful only relative to a freshly laid-out
# figure, so they must be applied after FigureComposer.fit_axes_to_cells().
DEFERRED_KINDS = ('panel',)


def _panel_of(ax):
    return getattr(ax, '_sciplotlib_panel', None)


def _iter_panels(fig):
    for ax in fig.get_axes():
        label = _panel_of(ax)
        if label:
            yield label, ax


def _panel_axes(fig, label):
    return next((a for lbl, a in _iter_panels(fig) if lbl == label), None)


def _colorbar_children(ax):
    """Child axes of *ax* that are colorbars (matplotlib sets ._colorbar)."""
    return [c for c in getattr(ax, 'child_axes', [])
            if getattr(c, '_colorbar', None) is not None]


def _inset_children(ax):
    """Child axes of *ax* that are insets (i.e. everything but colorbars)."""
    return [c for c in getattr(ax, 'child_axes', [])
            if getattr(c, '_colorbar', None) is None]


def _images(ax):
    """place_image() overlays on *ax*, in creation order."""
    return [a for a in ax.artists if isinstance(a, AnnotationBbox)]


def _texts(ax):
    """ax.text() artists on *ax*, in creation order.

    Panel letters live in ``fig.texts``, and axis labels are ``ax.xaxis.label``
    / ``ax.yaxis.label``, so neither shows up here.
    """
    return list(ax.texts)


def image_fingerprint(ab):
    """Stable-ish identity for an image overlay: the underlying array's shape."""
    try:
        arr = ab.get_children()[0].get_data()
        return 'image ' + 'x'.join(str(n) for n in arr.shape)
    except Exception:
        return None


def image_zoom(ab):
    try:
        return float(ab.get_children()[0].get_zoom())
    except Exception:
        return None


def set_image_zoom(ab, zoom):
    ab.get_children()[0].set_zoom(float(zoom))


def _addr(label, role, i, n):
    """``panel:<label>/<role>`` when there is only one, else ``…:<i>``."""
    base = f'panel:{label}/{role}'
    return base if n == 1 else f'{base}:{i}'


def iter_overridable(fig):
    """Yield addressable artists as dicts.

    Each item::

        {address, kind, artist, target, fingerprint}

    - ``artist`` is the thing the drag editor moves (used to match dragged items)
    - ``target`` is the object :func:`apply_value` operates on
    - ``kind``   is ``'panel'`` | ``'xlabel'`` | ``'ylabel'`` | ``'axes'``
                 | ``'image'`` | ``'text'``
    """
    for label, ax in _iter_panels(fig):
        yield {'address': f'panel:{label}', 'kind': 'panel',
               'artist': ax, 'target': ax, 'fingerprint': None}

        if ax.xaxis.label.get_text():
            yield {'address': f'panel:{label}/xlabel', 'kind': 'xlabel',
                   'artist': ax.xaxis.label, 'target': ax,
                   'fingerprint': ax.xaxis.label.get_text()}
        if ax.yaxis.label.get_text():
            yield {'address': f'panel:{label}/ylabel', 'kind': 'ylabel',
                   'artist': ax.yaxis.label, 'target': ax,
                   'fingerprint': ax.yaxis.label.get_text()}

        cbars = _colorbar_children(ax)
        for i, cax in enumerate(cbars):
            yield {'address': _addr(label, 'colorbar', i, len(cbars)),
                   'kind': 'axes', 'artist': cax, 'target': cax,
                   'fingerprint': None}

        insets = _inset_children(ax)
        for i, iax in enumerate(insets):
            yield {'address': _addr(label, 'inset', i, len(insets)),
                   'kind': 'axes', 'artist': iax, 'target': iax,
                   'fingerprint': None}

        imgs = _images(ax)
        for i, ab in enumerate(imgs):
            yield {'address': _addr(label, 'image', i, len(imgs)),
                   'kind': 'image', 'artist': ab, 'target': ab,
                   'fingerprint': image_fingerprint(ab)}

        txts = _texts(ax)
        for i, t in enumerate(txts):
            yield {'address': _addr(label, 'text', i, len(txts)),
                   'kind': 'text', 'artist': t, 'target': t,
                   'fingerprint': t.get_text()}


def address_map(fig):
    """{id(artist): (address, kind, fingerprint)} for all overridable artists."""
    return {id(o['artist']): (o['address'], o['kind'], o['fingerprint'])
            for o in iter_overridable(fig)}


def current_value(kind, artist, target):
    if kind in ('xlabel', 'ylabel'):
        return [float(v) for v in artist.get_position()]
    if kind in ('axes', 'panel'):
        return [float(v) for v in target.get_position().bounds]
    if kind == 'image':
        return [float(v) for v in target.xy]
    if kind == 'text':
        return [float(v) for v in target.get_position()]
    raise ValueError(f'unknown kind: {kind}')


# ── address resolution ─────────────────────────────────────────────────────

def _resolve_indexed(items, idx, fingerprint, describe):
    """Pick from *items* by *fingerprint* first, then by *idx*.

    Returns ``(item, warning_or_None)``.  Matching on the fingerprint means an
    override survives another artist being inserted ahead of it in the panel;
    falling back to the index means an *edited* label still lands somewhere
    sensible (with a warning) rather than being dropped.
    """
    if not items:
        return None, None
    if fingerprint is not None:
        hits = [it for it in items if describe(it) == fingerprint]
        if len(hits) == 1:
            return hits[0], None
        if len(hits) > 1:
            # Ambiguous: prefer the one at the recorded index if it matches.
            if idx < len(items) and describe(items[idx]) == fingerprint:
                return items[idx], None
            return hits[0], (f'{len(hits)} artists match fingerprint '
                             f'{fingerprint!r}; used the first')
    if idx < len(items):
        warn = None
        if fingerprint is not None:
            warn = (f'no artist matches fingerprint {fingerprint!r}; '
                    f'fell back to index {idx} '
                    f'({describe(items[idx])!r})')
        return items[idx], warn
    return None, (f'index {idx} out of range ({len(items)} artist(s))')


def resolve(fig, address, fingerprint=None):
    """Return ``(kind, target, warning)`` for *address*.

    ``(None, None, reason)`` if it cannot be resolved.
    """
    if not address.startswith('panel:'):
        return None, None, 'address does not start with "panel:"'
    body = address[len('panel:'):]
    label, _, role = body.partition('/')
    role, _, idx = role.partition(':')
    idx = int(idx) if idx else 0

    ax = _panel_axes(fig, label)
    if ax is None:
        return None, None, f'no panel labelled {label!r}'

    if role == '':
        return 'panel', ax, None
    if role == 'xlabel':
        return 'xlabel', ax, None
    if role == 'ylabel':
        return 'ylabel', ax, None
    if role == 'colorbar':
        cbars = _colorbar_children(ax)
        return ('axes', cbars[idx], None) if idx < len(cbars) else \
               (None, None, f'panel {label!r} has {len(cbars)} colorbar(s)')
    if role == 'inset':
        insets = _inset_children(ax)
        return ('axes', insets[idx], None) if idx < len(insets) else \
               (None, None, f'panel {label!r} has {len(insets)} inset(s)')
    if role == 'image':
        item, warn = _resolve_indexed(_images(ax), idx, fingerprint,
                                      image_fingerprint)
        return ('image', item, warn) if item is not None else (None, None, warn)
    if role == 'text':
        item, warn = _resolve_indexed(_texts(ax), idx, fingerprint,
                                      lambda t: t.get_text())
        return ('text', item, warn) if item is not None else (None, None, warn)
    return None, None, f'unknown role {role!r}'


# ── application ────────────────────────────────────────────────────────────

def apply_value(kind, target, value, entry=None):
    entry = entry or {}
    if kind == 'xlabel':
        target.xaxis.set_label_coords(float(value[0]), float(value[1]))
    elif kind == 'ylabel':
        target.yaxis.set_label_coords(float(value[0]), float(value[1]))
    elif kind == 'axes':
        # Detach any inset locator (colorbars and insets made with
        # ax.inset_axes re-pin themselves on redraw, which would override us).
        if target.get_axes_locator() is not None:
            target.set_axes_locator(None)
        target.set_position([float(v) for v in value])
    elif kind == 'panel':
        _apply_panel(target, value, entry)
    elif kind == 'image':
        target.xy = (float(value[0]), float(value[1]))
        if entry.get('zoom') is not None:
            set_image_zoom(target, entry['zoom'])
    elif kind == 'text':
        target.set_position((float(value[0]), float(value[1])))
    else:
        raise ValueError(f'unknown kind: {kind}')


def _apply_panel(ax, value, entry):
    """Move/resize a panel axes, by delta (default) or to absolute bounds.

    Child axes without their own locator are translated with the panel, so
    colorbars and insets that were detached by an earlier drag do not get left
    behind.
    """
    # Remember the un-nudged (freshly fitted) position. The drag editor reads
    # it back as the baseline, so re-opening the editor on a figure that
    # already has an override and nudging again yields the *total* delta from
    # the fitted layout, not an increment on top of the previous one.
    ax._sciplotlib_panel_base = tuple(ax.get_position().bounds)

    if entry.get('mode') == 'absolute' or entry.get('delta') is None:
        x0, y0, w, h = (float(v) for v in value)
        pos = ax.get_position()
        dx, dy = x0 - pos.x0, y0 - pos.y0
    else:
        dx0, dy0, dw, dh = (float(v) for v in entry['delta'])
        pos = ax.get_position()
        x0, y0 = pos.x0 + dx0, pos.y0 + dy0
        w, h = pos.width + dw, pos.height + dh
        dx, dy = dx0, dy0

    ax.set_position([x0, y0, max(w, 1e-4), max(h, 1e-4)])

    for child in getattr(ax, 'child_axes', []):
        if child.get_axes_locator() is not None:
            continue  # still pinned to the parent; it follows automatically
        cpos = child.get_position()
        child.set_position([cpos.x0 + dx, cpos.y0 + dy, cpos.width, cpos.height])


# ── file IO ────────────────────────────────────────────────────────────────

def read_overrides(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def write_overrides(path, overrides):
    with open(path, 'w') as f:
        json.dump(overrides, f, indent=2, sort_keys=True)


def overrides_as_code(overrides, axes_var='axes'):
    """Render an overrides dict as explicit matplotlib calls ("bake" to code).

    Lets you fold a hand-tuned ``overrides.json`` back into the plotting code so
    the figure has a single source of truth again (then delete the JSON).  Paste
    the output into the compose cell, where the composer axes dict is in scope.

    Note that ``panel:`` entries bake to **absolute** ``set_position`` calls
    (there is no fitted position to offset from once the JSON is gone), and must
    run *after* the final ``fit_axes_to_cells`` — i.e. after ``save()``'s
    normalisation, so pass ``composer.save(..., )`` a figure you have already
    positioned, or keep the JSON for panels.

    Parameters
    ----------
    overrides : dict
        As returned by :func:`read_overrides`.
    axes_var : str
        Name of the composer axes dict in scope (``fig, axes = composer.compose()``).
    """
    lines = [f'# --- baked position overrides '
             f'(equivalent to composer.apply_overrides) ---']
    for address in sorted(overrides):
        entry = overrides[address]
        kind = entry.get('kind')
        val = entry.get('value')
        body = address[len('panel:'):] if address.startswith('panel:') else address
        label, _, role = body.partition('/')
        role_name, _, idx = role.partition(':')
        i = int(idx) if idx else 0
        ax = f'{axes_var}[{label!r}]'
        lines.append(f'# {address}')
        if kind == 'xlabel':
            lines.append(f'{ax}.xaxis.set_label_coords({val[0]:.4f}, {val[1]:.4f})')
        elif kind == 'ylabel':
            lines.append(f'{ax}.yaxis.set_label_coords({val[0]:.4f}, {val[1]:.4f})')
        elif kind == 'panel':
            lines.append(f'# (absolute; must run AFTER fit_axes_to_cells)')
            lines.append(
                f'{ax}.set_position([{val[0]:.4f}, {val[1]:.4f}, '
                f'{val[2]:.4f}, {val[3]:.4f}])')
        elif kind == 'axes':
            attr = ('getattr(c, "_colorbar", None) is not None'
                    if role_name == 'colorbar'
                    else 'getattr(c, "_colorbar", None) is None')
            lines.append(f'_cax = [c for c in {ax}.child_axes if {attr}][{i}]')
            lines.append('_cax.set_axes_locator(None)')
            lines.append(
                f'_cax.set_position([{val[0]:.4f}, {val[1]:.4f}, '
                f'{val[2]:.4f}, {val[3]:.4f}])')
        elif kind == 'image':
            lines.append(
                f'_ab = [a for a in {ax}.artists '
                f'if isinstance(a, AnnotationBbox)][{i}]')
            lines.append(f'_ab.xy = ({val[0]:.4f}, {val[1]:.4f})')
            if entry.get('zoom') is not None:
                lines.append(
                    f'_ab.get_children()[0].set_zoom({entry["zoom"]:.4f})')
        elif kind == 'text':
            fp = entry.get('fingerprint')
            lines.append(f'# text {fp!r}')
            lines.append(
                f'{ax}.texts[{i}].set_position(({val[0]:.4f}, {val[1]:.4f}))')
        else:
            lines.append(f'# (unknown kind {kind!r}: {val})')
    return '\n'.join(lines)


def apply_overrides(fig, path, verbose=True, kinds=None, skip_kinds=None):
    """Apply every override in *path* to *fig*.

    Parameters
    ----------
    kinds : tuple of str, optional
        Only apply entries of these kinds.
    skip_kinds : tuple of str, optional
        Skip entries of these kinds.  :meth:`FigureComposer.apply_overrides`
        passes ``DEFERRED_KINDS`` here so panel deltas are held back until
        after ``fit_axes_to_cells``.

    Returns ``(n_applied, warnings)``.  Unresolvable addresses are skipped with
    a warning; a changed fingerprint warns but still applies.
    """
    overrides = read_overrides(path)
    applied, warnings = 0, []
    for address, entry in overrides.items():
        entry_kind = entry.get('kind')
        if kinds is not None and entry_kind not in kinds:
            continue
        if skip_kinds is not None and entry_kind in skip_kinds:
            continue

        fp = entry.get('fingerprint')
        kind, target, warn = resolve(fig, address, fingerprint=fp)
        if target is None:
            warnings.append(f'{address}: {warn or "no matching artist"} (skipped)')
            continue
        if warn:
            warnings.append(f'{address}: {warn}')

        if kind in ('xlabel', 'ylabel') and fp is not None:
            axis = target.xaxis if kind == 'xlabel' else target.yaxis
            if axis.label.get_text() != fp:
                warnings.append(
                    f'{address}: label text changed '
                    f'({fp!r} -> {axis.label.get_text()!r}); applied anyway')
        try:
            apply_value(kind, target, entry['value'], entry)
            applied += 1
        except Exception as e:  # pragma: no cover - defensive
            warnings.append(f'{address}: failed to apply ({e})')
    if verbose:
        if applied:
            print(f'[overrides] applied {applied} position override(s) from {path}')
        for w in warnings:
            print(f'[overrides] WARNING {w}')
    return applied, warnings
