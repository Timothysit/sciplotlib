# sciplotlib — LLM reference

sciplotlib is a Python library for composing publication-quality multi-panel figures with matplotlib. The core API is `FigureComposer` in `sciplotlib.compose`. Supporting utilities live in `sciplotlib.text`, `sciplotlib.style`, and `sciplotlib.polish`.

## Imports

```python
import sciplotlib.compose as splcompose   # FigureComposer
import sciplotlib.text   as spltext       # add_significance_bars, pval_to_stars
import sciplotlib.style  as splstyle      # get_style, get_palette, style_axes
import sciplotlib.polish as splpolish     # set_bounds, apply_gradient
```

## FigureComposer — full workflow

```python
composer = splcompose.FigureComposer(
    width_cm=18, height_cm=12,
    grid_rows=20, grid_cols=40,       # virtual grid; panels snap to cells
    dpi=600,
    stylesheet='mp-paper',            # mplstyle name (see Stylesheets)
    font_size=6.0,
    axis_label_font_size=6.0,
    title_font_size=6.0,
    label_font_size=14,               # panel letter size (a, b, c …)
    label_weight='bold',
    label_y=0.001,                    # vertical offset of panel letter
    spine_linewidth=0.7,
    tick_linewidth=0.7,
    tick_length=2.5,
    tick_pad=1.0,
    axis_label_pad=2.0,
    line_linewidth=1.0,               # normalises all Line2D widths
    wspace=0.4, hspace=1.5,
    margins={'left': 0.02, 'right': 0.98, 'bottom': 0.05, 'top': 0.96},
)
composer.apply_style()   # load stylesheet globally so panel cells inherit it
```

### add_panel

```python
composer.add_panel(
    label,          # str — single letter used as dict key and panel letter
    row, col,       # top-left grid cell (0-indexed)
    rowspan, colspan,
    file=None,      # optional path to .pkl / .png / .jpg / .svg
    no_axis=False,  # True → no axes created (image-only panel)
    axes_pad=None,  # dict with 'left','right','top','bottom' in fig fraction
    plot_func=None, # callable(ax) used by preview_image
)
```

Panels are positioned on a virtual grid. `rowspan`/`colspan` determine physical size proportionally.

### compose + plot

```python
fig, axes = composer.compose(wspace=None, hspace=None)
# axes is a dict keyed by label string

plot_panel_a(axes['a'])
plot_panel_b(axes['b'])
```

### Normalisation (call order matters)

```python
composer.normalize_fonts()       # sets all text to composer font sizes
composer.fit_axes_to_cells()     # shrinks axes so ticks/labels don't overlap adjacent panels
composer.normalize_spines()      # sets spine linewidths; NEVER clip spines — see Gotchas
composer.normalize_linewidths()  # sets Line2D widths to line_linewidth
```

`normalize_fonts()` also rescales **legend** entry text and legend titles to
`font_size`. This deliberately overrides any per-call `ax.legend(fontsize=...)`,
so don't pass an explicit legend `fontsize` expecting it to survive compose/save —
call `ax.legend(...)` without it and the legend will match the figure fonts.

`composer.to_image()` calls all four and returns a marimo-renderable PIL image.  
`composer.save(path)` calls the first three, then saves.

### Preview (per-panel, in marimo)

```python
# bottom of each panel cell:
_img = composer.preview_image('b', plot_func=plot_panel_b, normalize=True)
_img   # marimo displays it inline
```

`normalize=True` (default) applies all normalisation steps so the preview matches the final figure exactly.

### Save

```python
composer.save(
    'reports/figures/figure-4',
    formats=('pdf', 'svg', 'png'),   # any subset
    dpi=600,
    transparent=True,
)
```

### Stats report

```python
# in each panel cell — call after computing statistics:
composer.register_stats('e', [
    {
        'description': 'GLM-HMM 3-state vs LR log-likelihood',
        'test': 'Paired t-test (two-sided)',
        'statistic': -13.37, 'p_value': 9.67e-12, 'n': 22,
        'note': '3-state GLM-HMM outperforms LR',
    },
])
composer.register_stats('b', None)   # None = "no statistical tests apply"
# omitting register_stats → reported as "stats not yet registered"

# in the save cell:
composer.save_stats_report(
    'reports/figures/figure-4-stats',  # writes .md and .pdf
    title='Figure 4',
)
```

PDF backend priority: **weasyprint** (install `sciplotlib[stats-pdf]`) → **pandoc** → **matplotlib fallback** (always available).

Stat entry keys (all optional): `description`, `test`, `statistic`, `p_value`, `n`, `effect_size`, `ci`, `note`.

---

## Interactive position editing (drag editor + overrides)

`composer.launch_editor()` opens a native window where you drag artists to new
positions, and round-trips the moves back with **zero manual tagging**.

```python
# after compose() + plotting ALL panels:
composer.launch_editor(overrides_path='figure-3.overrides.json')  # drag, then CLOSE the window
```

**Draggable:** `ax.text()` texts, `place_image()` overlays (AnnotationBbox),
Rectangles/patches, **x/y axis labels**, and **inset/colorbar axes**. Panel
*main* axes are intentionally not draggable (the composer re-places them in
`fit_axes_to_cells`).

**On window close, two output modes:**
- **Panel roles** — axis labels and colorbars — are written to the
  `overrides_path` JSON, keyed by stable auto-addresses:
  `panel:<label>/xlabel`, `panel:<label>/ylabel`, `panel:<label>/colorbar`.
- **Everything else** prints `.set_position(...)` / `.set_label_coords(...)`
  snippets to the terminal to paste into your plotting code.

**Re-apply saved overrides on every render** (no code edits, survives re-runs):

```python
fig, axes = composer.compose()
# ... plot all panels ...
composer.apply_overrides('figure-3.overrides.json')   # no-op if the file is absent
composer.save('figures/figure-3')
```

Addresses are derived automatically from the panel label the composer stamps
(`ax._sciplotlib_panel`) plus matplotlib's own named roles (`cax._colorbar`,
`ax.xaxis.label`) — **no gids required**. Labels apply via `set_label_coords`;
colorbars via `set_position` (the inset locator is detached so the move sticks).

**Requirements & gotchas:**
- Needs an **interactive backend + a display**. Under marimo/uv the default is
  non-interactive Agg and uv's managed Python ships a broken Tk, so install Qt:
  **`uv add pyside6`**. The launcher then auto-selects `QtAgg` (and warns clearly
  if no GUI backend / display is found).
- It's a **native OS window on the machine running Python** — it will not appear
  for a remote/headless marimo (that's the inherent limit of `plt.show()`).
- In marimo the launching **cell blocks until you close the window**, and the
  editor's messages print to the **terminal running marimo**, not the cell.
- The editor renders at `screen_dpi` (default 100), not the figure's print dpi
  (e.g. 600), so the window fits the screen. Coordinates are all relative, so
  this never changes the saved override values. Pass `screen_dpi=` to resize.
- Two figure-lifecycle facts the editor handles for you: marimo detaches figures
  from pyplot after each cell (the editor re-attaches a manager so the window
  opens), and colorbars made with `ax.inset_axes` carry a locator that would
  re-pin them (detached on drag/apply).
- Dragging a colorbar **detaches it from its parent panel** (absolute position) —
  re-drag if you later change the grid, or delete its JSON entry to restore the
  inset default.
- Output now depends on code **+** the overrides JSON: commit the JSON next to
  the notebook, or **bake** it into code when the figure is final —
  `composer.print_overrides_as_code(path)` prints the equivalent explicit
  `set_label_coords` / `set_position` calls; paste them into the compose cell
  (where `axes` is in scope, replacing `apply_overrides`) and delete the JSON.

---

## Marimo integration

### Recommended cell structure

```
imports cell        →  mo, plt, np, splcompose, …
composer cell       →  FigureComposer() + add_panel calls + apply_style()
data loaders cell   →  @lru_cache functions — def _(): with NO inputs
panel a cell        →  define plot_panel_a, register_stats('a', …), preview_image
panel b cell        →  define plot_panel_b, register_stats('b', None), preview_image
…
compose cell        →  composer.compose() + all plot_panel_* calls + to_image()
save cell           →  composer.save(…) + composer.save_stats_report(…)
```

### Data loader pattern (prevents re-running on every upstream change)

```python
# data loaders cell — def _():   ← no inputs from other cells
import functools

@functools.lru_cache(maxsize=None)
def load_mouse_data():
    import matchingp.dataset as _mp   # all imports INSIDE the function
    return _mp.load_data(...)

return load_mouse_data   # export the function, not the data
```

`lru_cache` only speeds up the **interactive** marimo session (one long-lived
process). It does **nothing** for headless rendering: each `marimo export script`
+ `python run.py` is a **fresh process**, so every render pays the full
data-load/compute cost again. When you iterate on layout you re-render many times —
add a **persistent disk cache** so heavy loaders are computed once and reused
across renders (stdlib only, no new deps):

```python
# put this in its own early cell, export `disk_cache`
def disk_cache(version='v1'):          # bump version to invalidate after code change
    import functools, hashlib, pickle
    from pathlib import Path
    cache_dir = Path.home() / '.cache' / 'myproject_figure_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            src = f'{fn.__name__}|{version}|{args!r}|{sorted(kwargs.items())!r}'
            key = hashlib.md5(src.encode()).hexdigest()   # md5(repr), NOT hash() — hash() is per-process randomised
            fpath = cache_dir / f'{fn.__name__}.{key}.pkl'
            if fpath.exists():
                with open(fpath, 'rb') as f:
                    return pickle.load(f)
            result = fn(*args, **kwargs)
            with open(fpath, 'wb') as f:
                pickle.dump(result, f)
            return result
        return wrapper
    return deco
```

```python
# data loaders cell — def _(disk_cache):
@disk_cache(version='2026-07-08')      # returns a DataFrame / arrays that pickle cleanly
def load_mouse_data():
    import matchingp.dataset as _mp
    return _mp.load_data(...)
return load_mouse_data
```

Notes: key on `md5(repr(...))`, never Python's `hash()` (it is per-process
randomised by `PYTHONHASHSEED`, so the key would differ every render → always a miss).
Bump `version` whenever you change a loader's body. `joblib.Memory` does the same
thing with automatic source-change detection, but its `inspect.getsource` step is
unreliable for functions defined inside exec'd marimo cells — the explicit
`version` tag above is more robust for this workflow.

### Headless rendering workflow

There is no `marimo run --headless`. To render a figure notebook to files:

```bash
marimo export script figure-3.py -o run.py   # flatten to a plain script
python run.py                                 # runs all cells top-to-bottom, hits the save cell
```

Run `run.py` **from the notebook's own directory** (not a temp/scratch dir) when
any panel loads assets via `Path(__file__).parent / 'figure-parts' / ...` or when
the save cell writes to `Path(__file__).parent / 'figures'` — `__file__` resolves
to `run.py`'s location, so those relative paths only work if it sits beside the
notebook. A KeyError/exception in any cell aborts the whole render (cells run in
dependency order), so keep loaders robust.

### Inset axes inside a panel

Use `ax.inset_axes([x0, y0, w, h])` (all in parent-axes fraction) instead of `plt.subplots`. Inset axes are exempt from GridSpec layout, so they never interfere with `fit_axes_to_cells`. Never call `fig.subplots_adjust` inside a panel function — it corrupts the GridSpec.

### Axis-off panels (schematics)

```python
ax.axis('off')   # fit_axes_to_cells automatically skips this panel
```

To add inset axes with their own ticks inside an off panel, use `ax.inset_axes(...)` and only call `axis('off')` on the parent.

---

## add_significance_bars

```python
from sciplotlib.text import add_significance_bars

add_significance_bars(
    ax,
    pairs=[(3, 6), (3, 7)],   # x-positions in data coords
    pvalues=[p1, p2],          # converted to stars automatically
    # OR: labels=['***', 'n.s.'],  explicit text overrides pvalues
    pad=None,          # gap above data, default 6% of y-range
    tick_height=None,  # end-tick length, default 30% of pad
    linewidth=0.7,
    fontsize=5,
    show_ns=True,      # set False to skip non-significant pairs
    expand_ylim=True,  # auto-expand ylim to fit brackets
)
```

Brackets are stacked automatically (narrowest span placed lowest). Star thresholds: `*` p<0.05, `**` p<0.01, `***` p<0.001, `****` p<0.0001.

**Stack spacing is set by the label height, not by `pad` alone.** Overlapping brackets
are placed `2*pad` apart, so `pad` must be at least half the rendered label height or
the labels land on the bracket above. Watch for `fontsize=` at the call site being
overridden by `normalize_fonts()` — a call written for 5 pt labels gets 9 pt ones, and
the spacing that fit no longer does.

**Brackets may live above the axes on purpose** — in a panel's top margin, level with
its neighbours' titles — which keeps the data band full height. That needs
`expand_ylim=False` and `allow_above_ylim=True`; the latter only suppresses the warning
that otherwise fires when a bracket ends up outside the view. Two things make it work:
labels are drawn `annotation_clip=False` (matplotlib's `annotate` otherwise *silently
drops* a label whose `xy` is outside the axes — the bracket line draws and the star
vanishes), and the bracket lines are `clip_on=False`. In a composed figure, plot such a
panel **after** `compose()`, or `clip_all_axes` will clip the lines away.

Without `allow_above_ylim`, a bracket above the view limits raises a warning naming the
ylim you need — the failure is otherwise invisible to any collision check, because a
label that was never drawn cannot be found overlapping anything.

---

## Stylesheets

Available names for `FigureComposer(stylesheet=...)` and `splstyle.get_style(...)`:

| Name | Description |
|------|-------------|
| `nature-reviews` | Nature Publishing Group style |
| `nature` | Minimal Nature style |
| `mp-paper` | matching pennies paper style |
| `modern` | Clean sans-serif |
| `economist` | Economist magazine style |
| `dark` | Dark background |
| `default` | sciplotlib default |

```python
# Apply to a single axes post-hoc:
splstyle.style_axes(ax, style='nature-reviews', font_size=8)

# Apply via context manager:
with plt.style.context(splstyle.get_style('nature-reviews')):
    fig, ax = plt.subplots()
    ...
```

---

## Layout checks — overlaps, gaps, clipping

`sciplotlib.collide` finds artists that overlap, sit closer than a minimum gap,
run off the canvas, or get cut off by their own clip box. Hand-placed labels in
schematics drift out of position whenever fonts or panel sizes change, and the
failure is silent — a descender vanishing under an image is invisible until
someone looks at the printed page.

Labels are checked against each other **and against the drawing**: curves,
markers and axis spines. `min_gap_pt` is how you ask for breathing room between
an in-plot label and the line it names, or between a label and the spine it sits
beside.

```python
import sciplotlib.collide as splcollide

composer.check_layout()                     # 1 pt of clearance — what save() does
composer.check_layout(min_gap_pt=0.0)       # overlaps only
composer.check_layout(min_gap_pt=1.5)       # also flag anything within 1.5 pt
composer.check_layout(min_gap_pt=1.0, overlay_path='figures/f3-collisions.png')
```

Report lines are numbered to match the red boxes in the overlay image:

```
Layout check: 2 overlap(s), 2 pair(s) closer than 1.5 pt
   1. [c] overlap: text "Algorithm 0" overlaps image 500x457 (2.33 pt^2, hidden underneath)  ->  move up >= 0.7 pt
   2. [j] overlap: text "End" overlaps text "P(stochastic)" (0.78 pt^2)  ->  move up >= 0.7 pt
   3. [k] too-close: text "Late sessions" is 0.72 pt from left spine  ->  move right >= 1.8 pt
   4. [o] too-close: text "Mouse" is 1.80 pt from line (gray)  ->  move down >= 1.1 pt
```

`composer.save()` runs the check by default and prints findings without blocking
the save (`check_layout=False` to skip, `min_gap_pt=` to change the requirement).

### How it measures — and why not bounding boxes

Default `precision='ink'`: every candidate artist is drawn alone into an Agg
buffer and reduced to a mask of the pixels it actually paints. Boxes would be
useless here — a text's box spans the font's whole ascent-to-descent band, so
boxes touch long before glyphs do, and an SVG rendered to RGBA is mostly
transparent, so **text inside a hollow cartoon overlaps the image's box while
touching none of its strokes**. Working from ink means deliberate arrangements
(a label inside a drawn monitor, an icon in its empty middle) come out clean
without needing to be whitelisted. `precision='bbox'` is the fast, pessimistic
alternative (it adds a containment rule to suppress the worst false positives).

Only artists matplotlib *really draws* are considered: one pass runs with every
`draw` wrapped, and anything never called is dropped. Without that, out-of-view
tick labels and every tick/axis label under `ax.axis('off')` — all still
`get_visible() == True`, all happy to paint when asked directly — would collide
with real content.

### What is checked against what

`kinds` defaults to `('text', 'image', 'line', 'spine')` (`collide.DEFAULT_KINDS`).
Three exclusions are deliberate, and each would otherwise bury the real findings:

| Excluded | Why |
|---|---|
| **Pairs with no text** (`require_text`, auto-on when any drawing kind is checked) | Plot elements are *supposed* to touch: curves cross, markers pile up, a spine meets its ticks. A label touching any of them is the defect. |
| **`'patch'`, `'fill'`** (opt-in) | A '?' on a cartoon monitor, a value in a heatmap cell, a label across a pale error band — all ordinary practice. |
| **Gridlines, axes/figure background patches** (always) | Each spans a whole region, so every label inside the axes would "overlap" it. The background patch is found by identity from `ax.patch`, since its own `.axes` attribute is not reliably set. |

A tick label is allowed to sit close to **its own** axes' spine and tick marks —
that gap is what `tick_pad` sets — but an actual overlap there is still reported,
and any *other* text near that spine is reported normally. That distinction is
what makes `min_gap_pt` usable at all: without it every tick label in the figure
fires at any threshold above `tick_pad`.

### Reporting

Each `Collision` carries `kind` (`'overlap'`, `'too-close'`, `'outside-figure'`,
`'clipped'`), `gap_pt`, `overlap_pt2`, `panel`, `bbox_fig`, `occluded` (the other
artist is drawn on top, so this one is hidden), and `suggestion` — the smallest
axis-aligned move that clears the problem.

The suggestion is measured by **sliding the mask** until the gap is satisfied
(`scipy` distance transform; falls back to box arithmetic without it). Box
arithmetic is hopeless against anything non-convex: to clear a *curve's* box, a
label would be told to move below the curve's lowest point anywhere in the panel
— "move down >= 5.2 pt" where 1.8 pt of local clearance is the real answer.

### Silencing deliberate overlaps

```python
splcollide.exempt_from_collision_check(txt)   # ignore this artist entirely
splcollide.allow_overlap(txt, image)          # allow one pair, keep checking both otherwise
```

### Signatures

```python
splcollide.find_collisions(
    fig,
    min_gap_pt=1.0,              # required clear space in points; 0.0 = overlaps only
    kinds=DEFAULT_KINDS,         # text, image, line, spine (+ opt-in patch, fill, legend)
    precision='ink',             # or 'bbox'
    check_dpi=200,               # mask resolution: 1 pt = check_dpi/72 px
    alpha_threshold=16,          # alpha counting as ink (skips AA fringe)
    ignore_contained=None,       # default False for ink, True for bbox
    check_figure_bounds=True,
    require_text=None,           # auto: True once any drawing kind is checked
    include=None,                # include(artist) -> bool filter
) -> list[Collision]

splcollide.check_layout(fig, min_gap_pt=1.0, verbose=True, limit=None, **kw)
splcollide.format_collisions(collisions, min_gap_pt=0.0, limit=None)
splcollide.save_collision_overlay(fig, collisions, path, dpi=150, limit=None)
```

Use as a build-time assertion: `assert not splcollide.check_layout(fig)`.

**Cost:** two figure draws plus one isolated draw per candidate — on a 20-panel
figure that is well under a second, so leaving it on in `save()` is free.

**`min_gap_pt` defaults to 1 pt, not 0.** The worry was that figures pack cartoon
text tightly on purpose and a non-zero default would be noisy. Measured across four
full-page paper figures it produced **four findings, all real** — so the noise never
materialised, while overlap-only checking was blind to the commonest mistake in
hand-placed text: a label anchored *on* something. `ax.text(0, y, ...,
transform=ax.transAxes)` puts a label on the y-axis spine, clearing it only by the
first glyph's side bearing (~0.7 pt) — an overlap check sees nothing. Pass
`min_gap_pt=0.0` for overlaps only.

---

## Other utilities

```python
# Clip axes to data range (ticks define axis extent):
splpolish.set_bounds(fig, ax)

# Color palettes:
colors = splstyle.get_palette('nature-reviews', output_type='hex')  # list of hex strings

# Convert p-value to stars:
from sciplotlib.text import pval_to_stars
pval_to_stars(0.003)   # → '**'

# Save figure (legacy, prefer composer.save):
import sciplotlib.util as splutil
splutil.savefig(fig, 'path/to/figure', dpi=300, fig_exts=['.png', '.svg'])
```

---

## Gotchas

**Never clip Spine objects.** Spines sit exactly on the axes boundary, so clipping halves their visible stroke width. Remove any `spine.set_clip_on(True)` calls.

**`fit_axes_to_cells` grouping.** Panels are aligned by `(row_start, row_end)` pairs, not just `row_start`. Panels that share a start row but have different rowspans are in different alignment groups — this is intentional.

**`fit_axes_to_cells` overrides `set_position`.** Its last pass forces every panel in an alignment group to one `y0` and height, so growing or shrinking one panel's axes inside its plot function is silently undone. To give one panel of a row more room, change what it *draws* (e.g. put significance brackets in the top margin instead of inside the data range), or move it to its own alignment group.

**`subplots_adjust` corrupts GridSpec.** Never call `fig.subplots_adjust` or `plt.tight_layout` inside a panel function. These affect the top-level GridSpec and misalign all other panels.

**Inset axes bounds are in axes-fraction coordinates.** They don't move when you change `xlim`/`ylim`. Recompute bounds analytically if you change the data limits after placing insets: `y0_frac = (y_data - ylim_min) / (ylim_max - ylim_min)`.

**marimo `_`-prefix variables are cell-private.** Functions or variables named `_foo` cannot be returned from a marimo cell and are not visible to other cells. Don't prefix exported names with `_`.

**`axes_pad={}` sentinel.** Pass `axes_pad={}` (empty dict, not `None`) to `add_panel` to explicitly opt out of `fit_axes_to_cells` for one panel while keeping the default for others.
