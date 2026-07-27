# Plotting style notes

Working notes on the plotting styles in this folder, the current house
preference, and external themes worth borrowing from. Keep this updated when the
aesthetic evolves.

## Current house style: `tim-modern.mplstyle`

A clean, modern publication style. Lineage: `nature-reviews` → a "modern" pass
→ a frameless matplotx-inspired look.

- **Frameless**: no spines, no tick marks — soft horizontal gridlines
  (`grid.color c8c8c8`, `axes.grid.axis y`) carry the y-scale on a white
  background. Inspired by matplotx / "dufte" (see below).
- **Ink**: soft near-black `#222` (not pure black).
- **Type**: sans-serif (Helvetica/Arial), base 8 / title 9 / ticks 7; editable
  text on export (`pdf/ps.fonttype 42`, `svg.fonttype none`).
- **Markers**: flat, edgeless.
- **Sizes** (house convention): 3×3 square, 3×4 tall, 6×3 wide. Set
  `figure.figsize` per figure; default is 3×3.
- **DPI**: 150 on screen, 600 on export.
- **Categorical cycle**: validated colorblind-safe set
  `2a78d6, 1baf7a, eda100, 008300, 4a3aa7, e34948, e87ba4, eb6834`
  (checked with a CVD-separation validator; ordering is the safety mechanism —
  don't reorder casually).

## Reference themes / inspiration

### matplotx (Nico Schlömer) — https://github.com/nschloe/matplotx
The frameless direction of `tim-modern`. Its `dufte` style: **no spines, no
ticks, horizontal gridlines only**, white background, and **direct line-end
labels on the right** instead of a legend box (via `matplotx.line_labels()`).
Worth adopting the line-end labels for multi-series line plots.

### pilot (Oliver Hawkins) — https://github.com/olihawkins/pilot
"Attractive, minimal, general-purpose ggplot2 theme with an accessible discrete
palette." Details to imitate:
- **Titles aligned to the plot's outer edge**, not the plot area
  (`add_pilot_titles()`) — a distinctive, tidy look. In matplotlib approximate
  with left-aligned `fig.suptitle`/`Axes.set_title(loc='left')` anchored to the
  figure/axes edge.
- **Configurable gridlines**: `grid = "h" | "v" | "hv" | none`; **optional axes**
  chosen per side (`t/r/b/l`). Same spirit as our frameless + y-grid default.
- **Accessible categorical palette** (7 colors, CVD-tuned) — an alternate cycle
  to consider:
  navy `#204466`, blue `#249db5`, green `#30c788`, yellow `#ffc517`,
  orange `#f28100`, brown `#b84818`, purple `#9956db`.
- **Type**: Avenir Next on macOS, system sans elsewhere.

## Ideas to try next
- Add a `tim-modern`-based variant with matplotx-style **direct line-end labels**
  for line plots (drop the legend box).
- **Left-align titles to the plot edge** (pilot-style) instead of centered.
- Ship pilot's accessible palette as a selectable alternate `prop_cycle`.
- Optional `grid` variants (`h` / `v` / `hv`) as separate stylesheets or a helper.
