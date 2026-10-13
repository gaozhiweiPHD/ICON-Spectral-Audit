#!/usr/bin/env python3
"""Redraw square feature maps from saved arrays (legacy figure filenames).

The final Burgers figure uses the verified 16-pair extension when available;
the original 14-pair archive remains unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from plotting.style import apply_paper_style, inset_label, square_matrix_axis


def matrix_grid(
    matrices,
    labels,
    kinds,
    output,
    *,
    map_mode="signed",
    map_cmap=None,
    error_cmap="magma",
):
    """Render square maps while preserving each manuscript figure's semantics."""
    matrices = [np.asarray(matrix) for matrix in matrices]
    if map_mode not in {"signed", "absolute"}:
        raise ValueError("map_mode must be 'signed' or 'absolute'")
    map_values = [m for m, kind in zip(matrices, kinds) if kind == "map"]
    error_values = [m for m, kind in zip(matrices, kinds) if kind == "error"]
    vmax = max([float(np.max(np.abs(m))) for m in map_values] + [1e-12])
    error_vmax = max([float(np.max(np.abs(m))) for m in error_values] + [1e-12])
    if map_cmap is None:
        map_cmap = "coolwarm" if map_mode == "signed" else "magma"
    n_panels = len(matrices)
    figsize = (12.6, 3.7) if n_panels == 3 else (18.0, 3.8)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes)
    im_map = im_error = None
    for ax, matrix, label, kind in zip(axes, matrices, labels, kinds):
        if kind == "map":
            display = matrix if map_mode == "signed" else np.abs(matrix)
            vmin = -vmax if map_mode == "signed" else 0.0
            im_map = ax.imshow(
                display, origin="lower", aspect="equal", interpolation="nearest",
                cmap=map_cmap, vmin=vmin, vmax=vmax,
            )
        else:
            im_error = ax.imshow(
                np.abs(matrix), origin="lower", aspect="equal", interpolation="nearest",
                cmap=error_cmap, vmin=0.0, vmax=error_vmax,
            )
        square_matrix_axis(ax)
        inset_label(ax, label)
        ax.set_xlabel("input Fourier index")
        ax.set_ylabel("output Fourier index")

    map_axes = [ax for ax, kind in zip(axes, kinds) if kind == "map"]
    error_axes = [ax for ax, kind in zip(axes, kinds) if kind == "error"]
    if n_panels == 5:
        if im_map is not None:
            fig.colorbar(im_map, ax=map_axes, shrink=0.78, location="bottom", pad=0.08)
        if im_error is not None:
            fig.colorbar(im_error, ax=error_axes, shrink=0.78, location="bottom", pad=0.08)
    else:
        if im_map is not None:
            fig.colorbar(im_map, ax=map_axes, shrink=0.82)
        if im_error is not None:
            fig.colorbar(im_error, ax=error_axes, shrink=0.82)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def relative_frobenius(error, reference):
    return float(np.linalg.norm(error) / (np.linalg.norm(reference) + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "figures/revision/paper_style/square_feature_maps",
    )
    parser.add_argument("--overwrite", action="store_true")
    extended_burgers = PROJECT_ROOT / "experiments/revision/feature_map/burgers_near_shock_16_pairs.npz"
    parser.add_argument(
        "--burgers-feature-data", type=Path,
        default=extended_burgers if extended_burgers.exists() else PROJECT_ROOT / "experiments/burgers/figures/current/burgers_paper_style_diagnostics_results.npz",
        help="Saved Burgers projection; defaults to the verified 16-pair extension when present.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / f"fig{number:02d}_square_feature_map.png" for number in (4, 7, 10, 13, 19)]
    if not args.overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing: raise FileExistsError(f"refusing to overwrite {existing}; pass --overwrite")
    apply_paper_style()

    adv = np.load(PROJECT_ROOT / "experiments/advection/figures/current/advection_aligned_paper_style_diagnostics_results.npz")
    adv_rel_f = relative_frobenius(adv["A_error"], adv["A_true"])
    matrix_grid(
        [adv["A_true"], adv["A_icon"], adv["A_error"]],
        ["true", "ICON", f"abs. error, Rel-F={adv_rel_f:.2e}"],
        ["map", "map", "error"], outputs[0],
        map_mode="signed", map_cmap="coolwarm", error_cmap="magma",
    )

    burgers = np.load(args.burgers_feature_data, allow_pickle=True)
    burgers_rel_f = relative_frobenius(
        burgers["feature_near_shock_A_err"], burgers["feature_near_shock_A_true"]
    )
    matrix_grid(
        [burgers["feature_near_shock_A_true"], burgers["feature_near_shock_A_icon"], burgers["feature_near_shock_A_err"]],
        ["near_shock, true", "near_shock, ICON", f"absolute error, Rel-F={burgers_rel_f:.2e}"],
        ["map", "map", "error"], outputs[1],
        map_mode="absolute", map_cmap="magma", error_cmap="cividis",
    )

    corrupted = np.load(PROJECT_ROOT / "experiments/burgers/corrupted_prompt/run/wrong_eval_figs/burgers_wrong_negative_control_results.npz")
    matrix_grid(
        [corrupted["feature_A_true"], corrupted["feature_A_icon_wrong"], corrupted["feature_rel_err_map_wrong"]],
        ["true", "corrupted ICON", "relative error"],
        ["map", "map", "error"], outputs[2],
        map_mode="absolute", map_cmap="magma", error_cmap="cividis",
    )

    ac = np.load(PROJECT_ROOT / "experiments/allen_cahn/figures/current/allen_cahn_paper_style_diagnostics_results.npz")
    ac_rel_f = relative_frobenius(ac["feature_interface_A_err"], ac["feature_interface_A_true"])
    matrix_grid(
        [ac["feature_interface_A_true"], ac["feature_interface_A_icon"], ac["feature_interface_A_err"]],
        ["interface-like, true", "interface-like, ICON", f"absolute error, Rel-F={ac_rel_f:.2e}"],
        ["map", "map", "error"], outputs[3],
        map_mode="absolute", map_cmap="magma", error_cmap="magma",
    )

    ablation = np.load(PROJECT_ROOT / "experiments/advection/bandwidth_ablation/figures/advection_mode_ablation_results.npz")
    matrix_grid(
        [ablation["feature_A_true"], ablation["feature_A_mode8"], ablation["feature_A_mode16"],
         ablation["feature_E_mode8"], ablation["feature_E_mode16"]],
        ["true", "ICON, train modes = 8", "ICON, train modes = 16",
         "error, train modes = 8", "error, train modes = 16"],
        ["map", "map", "map", "error", "error"], outputs[4],
        map_mode="signed", map_cmap="coolwarm", error_cmap="cividis",
    )
    print("saved square feature-map redraws to", args.output_dir)


if __name__ == "__main__":
    main()
