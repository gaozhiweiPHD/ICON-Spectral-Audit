#!/usr/bin/env python3
"""Plot a completed training-proximity CSV (no JAX dependency)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from common.project import EXPERIMENTS_ROOT, FIGURES_ROOT
from plotting.style import PREDICTION_COLOR, SPECTRAL_COLOR, apply_paper_style


def rankdata(values):
    order = np.argsort(values); ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", required=True, choices=["advection", "burgers", "allen_cahn"])
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    input_path = args.input or EXPERIMENTS_ROOT / "revision/training_proximity" / f"{args.pde}_training_proximity.csv"
    output = args.output or FIGURES_ROOT / "revision/paper_style" / f"{args.pde}_training_proximity.png"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    with input_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    d = np.asarray([float(r["d_NN"]) for r in rows])
    pred = np.asarray([float(r["prediction_error"]) for r in rows])
    spec = np.asarray([float(r["spectral_error"]) for r in rows])
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), constrained_layout=True)
    panels = [(pred, r"relative $L^2$ prediction error", PREDICTION_COLOR),
              (spec, "spectral error", SPECTRAL_COLOR)]
    for ax, (values, ylabel, color) in zip(axes, panels):
        pearson = np.corrcoef(d, values)[0, 1]
        spearman = np.corrcoef(rankdata(d), rankdata(values))[0, 1]
        ax.scatter(d, values, color=color, alpha=0.8)
        ax.set_xlabel(r"relative nearest-neighbor distance $d_{\rm NN}$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both")
        ax.text(
            0.04, 0.94,
            rf"Pearson = {pearson:.2f}" + "\n" + rf"Spearman = {spearman:.2f}",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(facecolor="white", alpha=0.78, edgecolor="none"),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight"); fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
