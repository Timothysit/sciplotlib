"""label_case re-cases the drawn panel letter but keeps the panel key."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sciplotlib import compose as splcompose


def _letters(fig):
    return sorted(t.get_text() for t in fig.texts)


def test_upper_case_draws_capitals_but_keeps_keys():
    c = splcompose.FigureComposer(width_cm=8, height_cm=4, grid_rows=1,
                                  grid_cols=2, label_case='upper')
    c.add_panel('a', row=0, col=0, rowspan=1, colspan=1)
    c.add_panel('b', row=0, col=1, rowspan=1, colspan=1)
    fig, axes = c.compose()
    assert set(axes) == {'a', 'b'}
    assert _letters(fig) == ['A', 'B']
    plt.close(fig)


def test_default_leaves_label_unchanged():
    c = splcompose.FigureComposer(width_cm=8, height_cm=4, grid_rows=1, grid_cols=1)
    c.add_panel('a', row=0, col=0, rowspan=1, colspan=1)
    fig, _ = c.compose()
    assert _letters(fig) == ['a']
    plt.close(fig)
