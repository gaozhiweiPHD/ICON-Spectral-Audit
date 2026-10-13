#!/usr/bin/env python3
"""Plot discrete repeated-prompt means and uncertainty from raw CSV rows."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.project import FIGURES_ROOT, RESULTS_ROOT
from plotting.regimes import (style_parameter_curves, style_parameter_sweep, add_regime_key, comparison_filename)
from plotting.style import (
    BLOCK_COLOR,
    MATRIX_COLOR,
    PDE_XLABELS,
    PREDICTION_COLOR,
    SPECTRAL_COLOR,
    add_training_range,
    apply_paper_style,
    errorbar_points,
    mean_interval,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", required=True, choices=["advection", "burgers", "allen_cahn"])
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_ROOT / "revision/paper_style")
    parser.add_argument("--interval", choices=["std", "ci95"], default="std")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_grouped(path: Path):
    grouped = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[float(row["parameter"])].append(row)
    if not grouped:
        raise ValueError(f"no rows in {path}")
    return grouped


def summaries(grouped, metric, interval):
    x, mean, half = [], [], []
    for parameter in sorted(grouped):
        values = np.asarray([float(r[metric]) for r in grouped[parameter]], dtype=float)
        values = values[np.isfinite(values)]
        m, h = mean_interval(values[:, None], interval)
        x.append(parameter)
        mean.append(m[0])
        half.append(h[0])
    return np.asarray(x), np.asarray(mean), np.asarray(half)


def save(fig, path, overwrite):
    path = Path(comparison_filename(path))
    if (path.exists() or path.with_suffix(".pdf").exists()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    result_dir = RESULTS_ROOT / "revision/prompt_variability"
    full_range_path = result_dir / f"{args.pde}_prompt_stats_full_range.csv"
    training_range_path = result_dir / f"{args.pde}_prompt_stats.csv"
    # Prefer the paper-matched grid once it has been evaluated, but retain a
    # fallback so the preserved training-range-only results remain plottable.
    input_path = args.input or (
        full_range_path if full_range_path.exists() else training_range_path
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_grouped(input_path)
    print(
        f"plotting {input_path} with parameter range "
        f"[{min(grouped):g}, {max(grouped):g}] ({len(grouped)} points)"
    )
    apply_paper_style()
    uncertainty_label = r"mean $\pm$ SD" if args.interval == "std" else "mean with 95% CI"
    xlabel = PDE_XLABELS[args.pde]

    x, mean, half = summaries(grouped, "prediction_error", args.interval)
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    add_training_range(ax, args.pde)
    errorbar_points(
        ax, x, mean, half, color=PREDICTION_COLOR,
        label=uncertainty_label, log_y=True,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"relative $L^2$ prediction error")
    style_parameter_sweep(ax, args.pde)
    ax.legend(frameon=True)
    add_regime_key(fig)
    save(fig, args.output_dir / f"{args.pde}_prediction_error_repeated_prompts.png", args.overwrite)

    metrics = [
        ("prediction_error", r"relative $L^2$ prediction error", PREDICTION_COLOR),
        ("spectral_error", "final accumulated spectral error", SPECTRAL_COLOR),
        ("block_error" if args.pde == "advection" else "matrix_error",
         "mean block-rotation error" if args.pde == "advection" else "relative Frobenius error",
         BLOCK_COLOR if args.pde == "advection" else MATRIX_COLOR),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.45), constrained_layout=True)
    for panel_index, (ax, (metric, ylabel, color)) in enumerate(zip(axes, metrics)):
        x, mean, half = summaries(grouped, metric, args.interval)
        add_training_range(ax, args.pde, label="training range" if panel_index == 0 else None)
        errorbar_points(
            ax, x, mean, half, color=color,
            label=uncertainty_label if panel_index == 0 else None,
            log_y=(metric == "prediction_error"),
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    for ax in axes:
        style_parameter_sweep(ax, args.pde)
    axes[0].legend(frameon=True, loc="best")
    add_regime_key(fig)
    save(fig, args.output_dir / f"{args.pde}_spectral_robustness_summary.png", args.overwrite)


if __name__ == "__main__":
    main()
