"""
Tagless position overrides for FigureComposer figures.

The drag editor writes an *overrides* JSON keyed by stable, auto-derived
addresses (no manual gids), and :meth:`FigureComposer.apply_overrides` re-applies
them on every render — so hand-tuned positions survive re-runs without editing
plot code.

Addresses (layers 1–2, stable, zero user tagging)
-------------------------------------------------
    panel:<label>/xlabel        # x-axis label  -> set_label_coords
    panel:<label>/ylabel        # y-axis label  -> set_label_coords
    panel:<label>/colorbar[:i]  # colorbar axes -> set_position

These rely only on (a) the panel label the composer already stamps on each
axes (``ax._sciplotlib_panel``) and (b) matplotlib's own named roles (axis
labels; colorbar axes carry a ``._colorbar`` back-reference).  The common
single-colorbar case needs no index.

JSON format
-----------
    {
      "panel:j/colorbar": {"kind": "axes",   "value": [x0, y0, w, h], "fingerprint": null},
      "panel:m/xlabel":   {"kind": "xlabel", "value": [x, y],         "fingerprint": "Trials in session"}
    }
"""

from __future__ import annotations

import json


def _panel_of(ax):
    return getattr(ax, '_sciplotlib_panel', None)


def _iter_panels(fig):
    for ax in fig.get_axes():
        label = _panel_of(ax)
        if label:
            yield label, ax


def _colorbar_children(ax):
    """Child axes of *ax* that are colorbars (matplotlib sets ._colorbar)."""
    return [c for c in getattr(ax, 'child_axes', [])
            if getattr(c, '_colorbar', None) is not None]


def iter_overridable(fig):
    """Yield addressable artists as dicts.

    Each item::

        {address, kind, artist, target, fingerprint}

    - ``artist`` is the thing the drag editor moves (used to match dragged items)
    - ``target`` is the object :func:`apply_value` operates on
    - ``kind``   is ``'xlabel'`` | ``'ylabel'`` | ``'axes'``
    """
    for label, ax in _iter_panels(fig):
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
            addr = (f'panel:{label}/colorbar' if len(cbars) == 1
                    else f'panel:{label}/colorbar:{i}')
            yield {'address': addr, 'kind': 'axes',
                   'artist': cax, 'target': cax, 'fingerprint': None}


def address_map(fig):
    """{id(artist): (address, kind, fingerprint)} for all overridable artists."""
    return {id(o['artist']): (o['address'], o['kind'], o['fingerprint'])
            for o in iter_overridable(fig)}


def current_value(kind, artist, target):
    if kind in ('xlabel', 'ylabel'):
        return [float(v) for v in artist.get_position()]
    if kind == 'axes':
        return [float(v) for v in target.get_position().bounds]
    raise ValueError(f'unknown kind: {kind}')


def resolve(fig, address):
    """Return ``(kind, target)`` for *address*, or ``(None, None)`` if not found."""
    if not address.startswith('panel:'):
        return None, None
    body = address[len('panel:'):]
    label, _, role = body.partition('/')
    role, _, idx = role.partition(':')
    idx = int(idx) if idx else 0
    ax = next((a for lbl, a in _iter_panels(fig) if lbl == label), None)
    if ax is None:
        return None, None
    if role == 'xlabel':
        return 'xlabel', ax
    if role == 'ylabel':
        return 'ylabel', ax
    if role == 'colorbar':
        cbars = _colorbar_children(ax)
        if idx < len(cbars):
            return 'axes', cbars[idx]
    return None, None


def apply_value(kind, target, value):
    if kind == 'xlabel':
        target.xaxis.set_label_coords(float(value[0]), float(value[1]))
    elif kind == 'ylabel':
        target.yaxis.set_label_coords(float(value[0]), float(value[1]))
    elif kind == 'axes':
        # Detach any inset locator (colorbars made with ax.inset_axes re-pin
        # themselves on redraw, which would override our position).
        if target.get_axes_locator() is not None:
            target.set_axes_locator(None)
        target.set_position([float(v) for v in value])
    else:
        raise ValueError(f'unknown kind: {kind}')


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
        ax = f'{axes_var}[{label!r}]'
        lines.append(f'# {address}')
        if kind == 'xlabel':
            lines.append(f'{ax}.xaxis.set_label_coords({val[0]:.4f}, {val[1]:.4f})')
        elif kind == 'ylabel':
            lines.append(f'{ax}.yaxis.set_label_coords({val[0]:.4f}, {val[1]:.4f})')
        elif kind == 'axes':
            i = int(idx) if idx else 0
            lines.append(
                f'_cax = [c for c in {ax}.child_axes '
                f'if getattr(c, "_colorbar", None) is not None][{i}]')
            lines.append('_cax.set_axes_locator(None)')
            lines.append(
                f'_cax.set_position([{val[0]:.4f}, {val[1]:.4f}, '
                f'{val[2]:.4f}, {val[3]:.4f}])')
        else:
            lines.append(f'# (unknown kind {kind!r}: {val})')
    return '\n'.join(lines)


def apply_overrides(fig, path, verbose=True):
    """Apply every override in *path* to *fig*.

    Returns ``(n_applied, warnings)``.  Unresolvable addresses are skipped with
    a warning; a changed label fingerprint warns but still applies.
    """
    overrides = read_overrides(path)
    applied, warnings = 0, []
    for address, entry in overrides.items():
        kind, target = resolve(fig, address)
        if target is None:
            warnings.append(f'{address}: no matching artist (skipped)')
            continue
        fp = entry.get('fingerprint')
        if kind in ('xlabel', 'ylabel') and fp is not None:
            axis = target.xaxis if kind == 'xlabel' else target.yaxis
            if axis.label.get_text() != fp:
                warnings.append(
                    f'{address}: label text changed '
                    f'({fp!r} -> {axis.label.get_text()!r}); applied anyway')
        try:
            apply_value(kind, target, entry['value'])
            applied += 1
        except Exception as e:  # pragma: no cover - defensive
            warnings.append(f'{address}: failed to apply ({e})')
    if verbose:
        if applied:
            print(f'[overrides] applied {applied} position override(s) from {path}')
        for w in warnings:
            print(f'[overrides] WARNING {w}')
    return applied, warnings
