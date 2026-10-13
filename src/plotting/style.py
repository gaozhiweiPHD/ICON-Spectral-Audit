"""Shared plotting style matched to the figures in ``ICON.pdf``.

The manuscript figures use Matplotlib's default categorical colours, compact
panels, moderately heavy lines/markers, light grey grids, and pale blue
training-range bands.  Keep those choices here so that revision-only figures
look like part of the same paper rather than a separate plotting package.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


PREDICTION_COLOR = "#1f77b4"  # tab:blue
SPECTRAL_COLOR = "#ff7f0e"    # tab:orange
MATRIX_COLOR = "#2ca02c"      # tab:green
BLOCK_COLOR = "#9467bd"       # tab:purple
TRAINING_RANGE_COLOR = "#1f77b4"

PDE_TRAINING_RANGES = {
    "advection": (0.5, 1.0),
    "burgers": (0.005, 0.05),
    "allen_cahn": (0.02, 0.08),
}

PDE_XLABELS = {
    "advection": r"hidden speed $c$",
    "burgers": r"hidden viscosity $c$",
    "allen_cahn": r"hidden parameter $\varepsilon$",
}


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
            "font.size": 10.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 11.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
            "axes.linewidth": 1.0,
            "axes.axisbelow": True,
            "grid.color": "#b0b0b0",
            "grid.alpha": 0.32,
            "grid.linewidth": 0.8,
            "legend.framealpha": 0.88,
            "legend.edgecolor": "#c8c8c8",
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def mean_interval(values: np.ndarray, interval: str = "std") -> tuple[np.ndarray, np.ndarray]:
    """Return mean and symmetric standard-deviation or normal 95% CI half-width."""
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
    if interval == "std":
        return mean, std
    if interval == "ci95":
        return mean, 1.96 * std / np.sqrt(max(values.shape[0], 1))
    raise ValueError("interval must be 'std' or 'ci95'")


def errorbar_points(
    ax,
    x,
    mean,
    half_width,
    *,
    color,
    label=None,
    marker="o",
    log_y=False,
):
    """Discrete samples with manuscript-style uncertainty bars and guide line."""
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    half_width = np.asarray(half_width, dtype=float)
    if log_y:
        # Matplotlib cannot render error-bar endpoints at or below zero.
        half_width = np.minimum(half_width, 0.98 * mean)
    ax.errorbar(
        x,
        mean,
        yerr=half_width,
        fmt=f"{marker}-",
        color=color,
        ecolor=color,
        linewidth=2.0,
        elinewidth=1.35,
        capthick=1.2,
        capsize=3.0,
        markerfacecolor=color,
        markeredgecolor=color,
        label=label,
        zorder=3,
    )
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both")


def add_training_range(ax, pde: str, *, label: str | None = "training range") -> None:
    """Add the pale blue parameter-range band used in Figures 1, 5, and 11."""
    lower, upper = PDE_TRAINING_RANGES[pde]
    ax.axvspan(
        lower,
        upper,
        color=TRAINING_RANGE_COLOR,
        alpha=0.18,
        linewidth=0,
        label=label,
        zorder=0,
    )


def inset_label(ax, text: str, *, color: str = "black", dark_box: bool = False) -> None:
    """Place a compact panel label inside an axis, as in the manuscript maps."""
    facecolor = "black" if dark_box else "white"
    ax.text(
        0.03,
        0.94,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=color,
        bbox=dict(facecolor=facecolor, alpha=0.76, edgecolor="none", pad=2.2),
    )


def square_matrix_axis(ax) -> None:
    """Make a matrix panel visually square without altering its data."""
    ax.set_aspect("equal", adjustable="box")
