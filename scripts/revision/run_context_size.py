#!/usr/bin/env python3
"""Inference-only context-size ablation for sizes supported by the checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.project import EXPERIMENTS_ROOT
from models.revision_runtime import (
    default_parameters,
    load_main_params,
    load_runtime,
    make_prompt,
    prompt_metrics,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", required=True, choices=["advection", "burgers", "allen_cahn"])
    parser.add_argument("--context-sizes", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--parameters", nargs="+", type=float)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-mode", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    unsupported = [n for n in args.context_sizes if n not in (1, 2, 3, 4, 5)]
    if unsupported:
        raise ValueError(
            f"context sizes {unsupported} exceed the five-demo positional embedding; "
            "see experiments/revision/context_size/README.md"
        )
    output = args.output or EXPERIMENTS_ROOT / "revision/context_size" / f"{args.pde}_context_size.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    runtime = load_runtime(args.pde)
    params = load_main_params(args.pde, runtime)
    parameters = tuple(args.parameters or default_parameters(args.pde))
    rows = []
    for n_demo in args.context_sizes:
        for p_index, parameter in enumerate(parameters):
            query_seed = args.seed + 100_000 + p_index
            for repeat in range(args.repeats):
                prompt_seed = args.seed + n_demo * 1_000_000 + p_index * 10_000 + repeat
                _, data_one, label = make_prompt(
                    runtime, args.pde, parameter, prompt_seed, query_seed, n_demo=n_demo
                )
                metrics = prompt_metrics(
                    runtime, args.pde, params, data_one, label, parameter, args.max_mode
                )
                rows.append({
                    "PDE": args.pde, "parameter": parameter, "N_demo": n_demo,
                    "prompt_seed": prompt_seed, "query_seed": query_seed, **metrics,
                })
                print(args.pde, n_demo, parameter, repeat + 1, metrics, flush=True)
    fields = ["PDE", "parameter", "N_demo", "prompt_seed", "query_seed",
              "prediction_error", "spectral_error", "matrix_error", "block_error"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} raw rows to {output}")


if __name__ == "__main__":
    main()
