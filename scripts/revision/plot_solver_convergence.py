#!/usr/bin/env python3
"""Plot the small solver-convergence audit CSV."""

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
from plotting.style import apply_paper_style


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=EXPERIMENTS_ROOT / "revision/solver_convergence/solver_convergence.csv")
    parser.add_argument("--output", type=Path, default=FIGURES_ROOT / "revision/paper_style/solver_convergence.png")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite: raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")
    with args.input.open(newline="") as handle: rows = list(csv.DictReader(handle))
    pdes = sorted({row["PDE"] for row in rows})
    metrics = ("baseline_vs_time_refined_rel_l2", "baseline_vs_space_time_refined_rel_l2")
    labels = ("time refined", "space + time refined")
    means = np.asarray([[np.mean([float(r[m]) for r in rows if r["PDE"] == p]) for m in metrics] for p in pdes])
    stds = np.asarray([[np.std([float(r[m]) for r in rows if r["PDE"] == p], ddof=1) for m in metrics] for p in pdes])
    apply_paper_style(); fig, ax = plt.subplots(figsize=(5.8, 3.7), constrained_layout=True)
    x = np.arange(len(pdes)); width = 0.34
    for index, label in enumerate(labels):
        ax.bar(x + (index - 0.5) * width, means[:, index], width, yerr=stds[:, index], capsize=3, label=label)
    ax.set_xticks(x, [p.replace("_", "--") for p in pdes]); ax.set_yscale("log")
    ax.set_ylabel("relative final-state difference"); ax.legend(frameon=True); ax.grid(True, axis="y", which="both")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight"); fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__": main()
