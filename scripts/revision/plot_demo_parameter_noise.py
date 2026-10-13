#!/usr/bin/env python3
"""Plot demo-speed mismatch raw repetitions with uncertainty bars."""

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
from plotting.style import BLOCK_COLOR, PREDICTION_COLOR, SPECTRAL_COLOR, apply_paper_style, errorbar_points, mean_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RESULTS_ROOT / "revision/demo_parameter_noise/advection_demo_parameter_noise.csv")
    parser.add_argument("--output", type=Path, default=FIGURES_ROOT / "revision/paper_style/advection_demo_parameter_noise.png")
    parser.add_argument("--interval", choices=["std", "ci95"], default="std")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")
    grouped = defaultdict(list)
    with args.input.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[(float(row["c_query"]), float(row["sigma"]))].append(row)
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.45), constrained_layout=True)
    panels = [("prediction_error", r"relative $L^2$ prediction error", PREDICTION_COLOR),
              ("spectral_error", "final accumulated spectral error", SPECTRAL_COLOR),
              ("block_error", "mean block-rotation error", BLOCK_COLOR)]
    speeds = sorted({key[0] for key in grouped})
    for ax, (metric, ylabel, color) in zip(axes, panels):
        for index, speed in enumerate(speeds):
            sigmas = sorted(key[1] for key in grouped if key[0] == speed)
            means, halfs = [], []
            for sigma in sigmas:
                values = np.asarray([float(row[metric]) for row in grouped[(speed, sigma)]])
                mean, half = mean_interval(values[:, None], args.interval)
                means.append(mean[0]); halfs.append(half[0])
            # Preserve the metric's hue while varying lightness across query speeds.
            query_color = plt.cm.Blues(0.45 + 0.45 * index / max(len(speeds) - 1, 1)) if metric == "prediction_error" else (
                plt.cm.Oranges(0.45 + 0.45 * index / max(len(speeds) - 1, 1)) if metric == "spectral_error"
                else plt.cm.Purples(0.45 + 0.45 * index / max(len(speeds) - 1, 1))
            )
            errorbar_points(
                ax, sigmas, means, halfs, color=query_color,
                label=rf"$c_{{\rm query}}={speed:g}$",
                log_y=(metric == "prediction_error"),
            )
        ax.set_xlabel(r"demo-speed noise $\sigma$")
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
