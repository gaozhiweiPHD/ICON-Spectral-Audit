#!/usr/bin/env python3
"""Render regime comparisons from preserved arrays, with no model execution.

Only selected plot function definitions are loaded from the original sources.
This avoids importing their JAX training/evaluation code and preserves their
panel layouts, labels, colours and curves. CSV summaries use their own CLI.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from plotting.regimes import (
    add_regime_key, comparison_filename,
    style_parameter_curves, style_parameter_sweep,
)
from plotting.style import apply_paper_style, PDE_TRAINING_RANGES


def load_plots(source, names, output_dir, **extra):
    """Compile only named plotting definitions; never run source module bodies."""
    path = ROOT / source
    if path.suffix == ".ipynb":
        sources = ["".join(cell.get("source", []))
                   for cell in json.loads(path.read_text())["cells"]
                   if cell["cell_type"] == "code"]
    else:
        sources = [path.read_text()]
    namespace = dict(plt=plt, np=np, os=os, n_modes=16, fig_dir=str(output_dir),
                     style_parameter_curves=style_parameter_curves,
                     style_parameter_sweep=style_parameter_sweep,
                     add_regime_key=add_regime_key,
                     comparison_filename=comparison_filename)

    def save(*args):
        fig, filename = args if len(args) == 2 else (plt.gcf(), args[0])
        if not fig.get_constrained_layout():
            fig.tight_layout()
        # The plot functions already request a comparison name.
        path = Path(filename)
        if path.parent == Path("."):
            path = output_dir / path
        paths = [path.with_suffix(ext) for ext in (".png", ".pdf")]
        for output in paths:
            if output.exists():
                raise FileExistsError(f"refusing to overwrite {output}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for output in paths:
            fig.savefig(output, dpi=300, bbox_inches="tight")
        print(path.stem, flush=True)
        plt.close(fig)

    namespace.update(save_or_show=save, savefig=save, save_figure=save,
                     save_current_figure=save, **extra)
    found = set()
    for source_text in sources:
        for node in ast.parse(source_text).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
                found.add(node.name)
    if found != set(names):
        raise ValueError(f"missing plotting functions: {set(names) - found}")
    return namespace


def render(output_dir):
    apply_paper_style()
    d = np.load(ROOT / "experiments/advection/figures/current/advection_aligned_paper_style_diagnostics_results.npz")
    result = {"prediction_error": d["prediction_error"], "query_context": {
        "k": d["context_k"], "c_values": d["context_c_values"],
        "icon_mean": d["context_icon_mean"], "true": d["context_true"]}}
    names = ["plot_prediction_error", "plot_query_spectrum_with_cumulative_error"]
    ns = load_plots("experiments/advection/evaluation/evaluate_advection.py", names,
                    output_dir, c_range=PDE_TRAINING_RANGES["advection"],
                    cumulative_relative_errors=lambda *_: tuple(d[f"query_cum_rel_err_{key}"] for key in ("cos", "sin", "all")))
    for name in names:
        ns[name](result)

    d = np.load(ROOT / "experiments/allen_cahn/figures/current/allen_cahn_paper_style_diagnostics_results.npz")
    result = {"pred_error": d["pred_error"], "query_spectrum": {
        key: d[f"query_{key}"] for key in
        ("k", "eps_values", "icon_mean", "cum_rel_err_cos", "cum_rel_err_sin")}}
    names = ["plot_prediction_error_over_eps", "plot_query_spectrum_with_cumulative_error"]
    ns = load_plots("experiments/allen_cahn/evaluation/evaluate_allen_cahn.py", names,
                    output_dir, eps_range=PDE_TRAINING_RANGES["allen_cahn"])
    for name in names:
        ns[name](result)

    d = np.load(ROOT / "experiments/burgers/figures/current/burgers_paper_style_diagnostics_results.npz")
    result = {"prediction_error": {"c_values": d["prediction_c_values"],
               "err_mean": d["prediction_err_mean"], "err_std": d["prediction_err_std"]},
              "query_spectrum": {key: d[f"query_{key}"] for key in
               ("k", "icon_mean", "icon_std", "cum_rel_err_cos", "cum_rel_err_sin")}}
    result["query_spectrum"]["c_values"] = d["prediction_c_values"]
    names = ["plot_prediction_error_over_c_paper", "plot_query_spectrum_and_error_paper"]
    ns = load_plots("experiments/burgers/notebooks/icon_burgers.ipynb", names,
                    output_dir, c_range=PDE_TRAINING_RANGES["burgers"])
    for name in names:
        ns[name](result)

    d = np.load(ROOT / "experiments/burgers/corrupted_prompt/run/wrong_eval_figs/burgers_wrong_negative_control_results.npz")
    result = {"prediction_wrong": {"c_values": d["pred_c_values"], "err_mean": d["pred_wrong_mean"], "err_std": d["pred_wrong_std"]},
              "prediction_clean": {"c_values": d["pred_c_values"], "err_mean": d["pred_clean_mean"]} if "pred_clean_mean" in d else None,
              "query_wrong": {"k": d["query_k"], "c_values": d["query_c_values"],
                 "icon_mean": d["query_wrong_icon_mean"],
                 "cum_rel_err_cos": d["query_wrong_cum_rel_err_cos"],
                 "cum_rel_err_sin": d["query_wrong_cum_rel_err_sin"]}}
    names = ["plot_prediction_error_over_c", "plot_query_spectrum_wrong"]
    ns = load_plots("experiments/burgers/corrupted_prompt/evaluate_corrupted_prompt.py", names,
                    output_dir, c_range=PDE_TRAINING_RANGES["burgers"])
    for name in names:
        ns[name](result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "figures/revision/regime_style")
    args = parser.parse_args()
    render(args.output_dir)


if __name__ == "__main__":
    main()
