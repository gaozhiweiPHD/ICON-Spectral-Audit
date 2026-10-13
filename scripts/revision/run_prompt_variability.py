#!/usr/bin/env python3
"""Repeated-prompt statistics using existing trained checkpoints only."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.project import RESULTS_ROOT
from models.revision_runtime import (
    default_parameters,
    load_main_params,
    load_runtime,
    make_prompt,
    prompt_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", required=True, choices=["advection", "burgers", "allen_cahn"])
    parser.add_argument("--parameters", nargs="+", type=float)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-mode", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    parameters = tuple(args.parameters or default_parameters(args.pde))
    output = args.output or RESULTS_ROOT / "revision/prompt_variability" / f"{args.pde}_prompt_stats.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite explicitly")
    runtime = load_runtime(args.pde)
    params = load_main_params(args.pde, runtime)
    rows = []
    for p_index, parameter in enumerate(parameters):
        query_seed = args.seed + 100_000 + p_index
        for repeat in range(args.repeats):
            prompt_seed = args.seed + 10_000 * p_index + repeat
            _, data_one, label = make_prompt(
                runtime, args.pde, parameter, prompt_seed, query_seed, n_demo=5
            )
            metrics = prompt_metrics(
                runtime, args.pde, params, data_one, label, parameter, args.max_mode
            )
            rows.append(
                {
                    "PDE": args.pde,
                    "parameter": parameter,
                    "prompt_seed": prompt_seed,
                    "query_seed": query_seed,
                    **metrics,
                }
            )
            print(args.pde, parameter, repeat + 1, metrics, flush=True)
    fields = [
        "PDE", "parameter", "prompt_seed", "query_seed", "prediction_error",
        "spectral_error", "matrix_error", "block_error",
    ]
    with output.open("x" if not args.overwrite else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} raw rows to {output}")


if __name__ == "__main__":
    main()
