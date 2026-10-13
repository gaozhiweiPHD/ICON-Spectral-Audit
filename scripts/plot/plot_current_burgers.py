#!/usr/bin/env python3
"""Safely redraw manuscript Burgers panels from the preserved NPZ (no training)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from common.project import BURGERS_FIGURE_DIR
from plotting.style import apply_paper_style, errorbar_points, PREDICTION_COLOR, add_training_range
from plotting.regimes import style_parameter_sweep, add_regime_key, comparison_filename


def save(fig, path, overwrite):
    if (path.exists() or path.with_suffix(".pdf").exists()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; use --output-dir or --overwrite")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=BURGERS_FIGURE_DIR / "burgers_paper_style_diagnostics_results.npz")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures/revision/replotted_current_burgers")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.input, allow_pickle=True)
    apply_paper_style()

    fig, ax = plt.subplots(figsize=(5.2, 3.7), constrained_layout=True)
    errorbar_points(ax, data["prediction_c_values"], data["prediction_err_mean"],
                    data["prediction_err_std"], color=PREDICTION_COLOR)
    ax.set_xlabel("hidden viscosity"); ax.set_ylabel("relative L2 prediction error")
    style_parameter_sweep(ax, "burgers")
    add_training_range(ax, "burgers")
    ax.legend(frameon=True)
    add_regime_key(fig)
    save(fig, Path(comparison_filename(args.output_dir / "burgers_relative_prediction_error_over_hidden_viscosity.png")), args.overwrite)

    kinds = ("smooth", "steep", "near_shock")
    fig, axes = plt.subplots(3, 4, figsize=(14.2, 8.2), sharex=True, constrained_layout=True)
    for row, kind in enumerate(kinds):
        k = data[f"base_context_{kind}_k"]
        c_values = data[f"base_context_{kind}_c_values"]
        icon = data[f"base_context_{kind}_icon_mean"]
        true = data[f"base_context_{kind}_true"]
        cumulative = (data[f"base_context_{kind}_cum_rel_err_cos"],
                      data[f"base_context_{kind}_cum_rel_err_sin"])
        for col, component in enumerate(("cos", "sin")):
            ax = axes[row, col]
            for index, viscosity in enumerate(c_values):
                color = plt.cm.viridis(index / max(len(c_values) - 1, 1))
                ax.plot(k, true[index, :, col], color=color, linestyle="-", alpha=0.8)
                ax.plot(k, icon[index, :, col], color=color, linestyle="--", label=f"nu={viscosity:g}")
            ax.axvline(16, color="0.35", linestyle=":", linewidth=1)
            ax.grid(True); ax.set_aspect("auto")
            ax.set_ylabel(f"{kind.replace('_', ' ')}: {component} gain")
            err_ax = axes[row, col + 2]
            for index, viscosity in enumerate(c_values):
                color = plt.cm.viridis(index / max(len(c_values) - 1, 1))
                err_ax.plot(k, cumulative[col][index], color=color, label=f"nu={viscosity:g}")
            err_ax.axvline(16, color="0.35", linestyle=":", linewidth=1)
            err_ax.grid(True); err_ax.set_ylabel(f"cumulative {component} error")
    for ax in axes[-1]: ax.set_xlabel("Fourier mode k")
    axes[0, 1].legend(frameon=False, fontsize=8)
    save(fig, args.output_dir / "burgers_base_state_context_spectrum_with_cumulative_relative_error.png", args.overwrite)

    fig, axes = plt.subplots(3, 3, figsize=(9.4, 8.8), constrained_layout=True)
    im_main = im_error = None
    feature_vmax = max(
        float(np.abs(data[f"feature_{kind}_A_true"]).max())
        for kind in kinds
    )
    feature_vmax = max(feature_vmax, max(
        float(np.abs(data[f"feature_{kind}_A_icon"]).max())
        for kind in kinds
    ), 1e-12)
    error_vmax = max(
        max(float(np.abs(data[f"feature_{kind}_A_err"]).max()) for kind in kinds),
        1e-12,
    )
    for row, kind in enumerate(kinds):
        true = np.abs(data[f"feature_{kind}_A_true"])
        icon = np.abs(data[f"feature_{kind}_A_icon"])
        error = np.abs(data[f"feature_{kind}_A_err"])
        for col, matrix in enumerate((true, icon)):
            im_main = axes[row, col].imshow(matrix, origin="lower", aspect="equal", cmap="magma", vmin=0, vmax=feature_vmax)
        im_error = axes[row, 2].imshow(error, origin="lower", aspect="equal", cmap="cividis", vmin=0, vmax=error_vmax)
        axes[row, 0].set_ylabel(kind.replace("_", " "))
    for col, title in enumerate(("true", "ICON", "absolute error")): axes[0, col].set_title(title)
    for ax in axes.ravel(): ax.set_xlabel("input Fourier index"); ax.set_ylabel("output Fourier index")
    fig.colorbar(im_main, ax=axes[:, :2], shrink=0.75); fig.colorbar(im_error, ax=axes[:, 2], shrink=0.75)
    save(fig, args.output_dir / "burgers_fourier_feature_maps_true_icon_absolute_error.png", args.overwrite)


if __name__ == "__main__":
    main()
