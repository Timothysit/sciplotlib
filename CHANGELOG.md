# Changelog

## 0.0.9 — 2026-10-09

> **Note:** this entry was AI-generated (Claude Code) from the diff, not
> written by the author. Hand-written release notes may replace it later.

### Added
- **Posters** — `sciplotlib.poster.PosterComposer`, a `FigureComposer` subclass
  for posters: standard paper sizes (A0–A4, 16:9 monitor), type and line
  weights scaled to the paper, titled sections (`add_section`), a title banner
  with logos (`add_title`, auto-reversed text on dark bands, logo tinting), and
  wrapped prose / bulleted text blocks (`add_text_block`).
- **Slides** — `sciplotlib.slides.SlideComposer` and `SlideDeck`: exact
  PowerPoint/Keynote slide sizes rendered to one multi-page PDF, progressive
  reveals (`stages=N`), speaker notes (`save_notes`), `first_number=` for decks
  that start partway through another, ticks on by default, and a title band /
  footer reserved outside the content grid.
- **Redraw hooks** — `set_redraw_hook` / `run_redraw_hook` rebuild
  size-dependent content (e.g. wrapped text) when a panel is resized; run by
  `fit_axes_to_cells` and the editor's reflow toggle.
- `exempt_from_linewidth_normalization`, and `add_panel(fit_exempt=True)` /
  `add_panel(no_label=True)`.
- **Layout checks** — `find_clipped_data` / `check_clipping` (data cut off at
  the axes limits) and `find_crowded_data` / `check_crowding` (data crowding the
  axes edges), also available as `composer.check_clipping()` /
  `composer.check_crowding()`. Annotation leaders (arrows) are now checked.
- `overrides.check_child_overrides`.

### Changed
- **Panel editor overhaul** — fits the figure to the window by default
  (`view_dpi='fit'`), zoom and pan, faster redraws, multi-select and
  drag-to-select, hide / delete / restore, in-place text editing, undo for
  these edits, and a reflow toggle.
- `launch_editor(screen_dpi=None)` now defaults to fit-to-window.
- Bulleted text blocks draw each line as its own mark + body pair, so lines
  containing mathtext no longer misalign the bullets.
- `CLAUDE.md` documents posters, slides and the new composer features.
