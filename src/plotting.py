"""Publication-style matplotlib helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import EVENTS


def set_academic_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9, 4.8),
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def annotate_events(ax, selected: list[str] | None = None) -> None:
    names = selected or ["COVID-19 outbreak", "Vietnam border reopening", "Tourism recovery period"]
    ymin, ymax = ax.get_ylim()
    y = ymin + 0.92 * (ymax - ymin)
    for name in names:
        date = pd.Timestamp(EVENTS[name])
        ax.axvline(date, color="#8a8a8a", linewidth=0.8, alpha=0.55)
        ax.text(date, y, name, rotation=90, va="top", ha="right", fontsize=8, color="#555555")


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
