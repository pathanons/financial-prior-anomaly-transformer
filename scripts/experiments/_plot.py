"""Small shared template for research-figure styling and output."""

from pathlib import Path

import matplotlib.pyplot as plt


BLUE = "#2a78d6"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BACKGROUND = "#fcfcfb"


def style_axis(ax, grid_axis):
    ax.set_facecolor(BACKGROUND)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=SECONDARY_INK)


def save_figure(fig, path, dpi=180):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=BACKGROUND)
    plt.close(fig)
    print(f"Wrote {path}")
