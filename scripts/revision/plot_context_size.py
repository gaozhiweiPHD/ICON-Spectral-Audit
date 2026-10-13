#!/usr/bin/env python3
"""Plot context-size ablation raw rows (no JAX dependency)."""

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
from common.project import EXPERIMENTS_ROOT, FIGURES_ROOT
from plotting.style import PDE_XLABELS, apply_paper_style


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", required=True, choices=["advection", "burgers", "allen_cahn"])
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    input_path = args.input or EXPERIMENTS_ROOT / "revision/context_size" / f"{args.pde}_context_size.csv"
    output = args.output or FIGURES_ROOT / "revision/paper_style" / f"{args.pde}_context_size.png"
    if output.exists() and not args.overwrite: raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    groups = defaultdict(list)
    with input_path.open(newline="") as handle:
        for row in csv.DictReader(handle): groups[(int(row["N_demo"]), float(row["parameter"]))].append(row)
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), constrained_layout=True)
    for ax, metric, ylabel in zip(axes, ("prediction_error", "spectral_error"),
                                  (r"relative $L^2$ prediction error", "spectral error")):
        for n_demo in sorted({key[0] for key in groups}):
            parameters = sorted(key[1] for key in groups if key[0] == n_demo)
            means = [np.mean([float(r[metric]) for r in groups[(n_demo, p)]]) for p in parameters]
            stds = [np.std([float(r[metric]) for r in groups[(n_demo, p)]], ddof=1) for p in parameters]
            ax.errorbar(parameters, means, yerr=stds, marker="o", capsize=3,
                        label=rf"$N_{{\rm demo}}={n_demo}$")
        ax.set_xlabel(PDE_XLABELS[args.pde]); ax.set_ylabel(ylabel); ax.grid(True, which="both")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight"); fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__": main()
