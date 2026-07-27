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

**`subplots_adjust` corrupts GridSpec.** Never call `fig.subplots_adjust` or `plt.tight_layout` inside a panel function. These affect the top-level GridSpec and misalign all other panels.

**Inset axes bounds are in axes-fraction coordinates.** They don't move when you change `xlim`/`ylim`. Recompute bounds analytically if you change the data limits after placing insets: `y0_frac = (y_data - ylim_min) / (ylim_max - ylim_min)`.

**marimo `_`-prefix variables are cell-private.** Functions or variables named `_foo` cannot be returned from a marimo cell and are not visible to other cells. Don't prefix exported names with `_`.

**`axes_pad={}` sentinel.** Pass `axes_pad={}` (empty dict, not `None`) to `add_panel` to explicitly opt out of `fit_axes_to_cells` for one panel while keeping the default for others.
