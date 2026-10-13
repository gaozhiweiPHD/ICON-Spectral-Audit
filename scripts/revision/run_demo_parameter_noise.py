#!/usr/bin/env python3
"""Advection demonstration-speed mismatch using the existing checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.project import RESULTS_ROOT
from models.revision_runtime import load_main_params, load_runtime, prompt_metrics


def one_pair(runtime, speed, key):
    return runtime.generate_shift_raw_data(
        key, 1, 1, runtime.n_grid, runtime.n_modes,
        (float(speed), float(speed) + 1e-8), runtime.dt,
        runtime.fourier_decay, runtime.initial_amp, runtime.bias_scale,
    )


def assemble_prompt(runtime, query_speed, sigma, prompt_seed, query_seed, clip_speeds=True):
    rng = np.random.default_rng(prompt_seed)
    demo_speeds = query_speed + rng.normal(0.0, sigma, size=5)
    if clip_speeds:
        demo_speeds = np.clip(demo_speeds, runtime.c_range[0], runtime.c_range[1])
    demo_parts = [
        one_pair(runtime, speed, runtime.jax.random.PRNGKey(prompt_seed + 100 + i))
        for i, speed in enumerate(demo_speeds)
    ]
    query_part = one_pair(runtime, query_speed, runtime.jax.random.PRNGKey(query_seed))
    parts = demo_parts + [query_part]
    cat = lambda name: runtime.jnp.concatenate([getattr(part, name) for part in parts], axis=1)
    raw = runtime.RawShift(
        cond_k=cat("cond_k"), cond_v=cat("cond_v"),
        qoi_k=cat("qoi_k"), qoi_v=cat("qoi_v"),
        c=query_part.c, shift=query_part.shift,
        coeff_sin=cat("coeff_sin"), coeff_cos=cat("coeff_cos"), bias=cat("bias"),
    )
    data, label = runtime.build_icon_data(raw, demo_num=5)
    data_one = runtime.tree.tree_map(lambda x: x[0], data)
    return demo_speeds, data_one, label[0, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-speeds", nargs="+", type=float, default=[0.6, 0.8, 1.0])
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.01, 0.025, 0.05])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-mode", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--no-clip-speeds", action="store_true")
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT / "revision/demo_parameter_noise/advection_demo_parameter_noise.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")
    runtime = load_runtime("advection")
    params = load_main_params("advection", runtime)
    rows = []
    for c_index, query_speed in enumerate(args.query_speeds):
        if not runtime.c_range[0] <= query_speed <= runtime.c_range[1]:
            raise ValueError(f"query speed {query_speed} is outside training range {runtime.c_range}")
        query_seed = args.seed + 100_000 + c_index
        for sigma_index, sigma in enumerate(args.sigmas):
            if sigma < 0:
                raise ValueError("sigma must be nonnegative")
            for repeat in range(args.repeats):
                prompt_seed = args.seed + c_index * 100_000 + sigma_index * 1_000 + repeat
                demo_speeds, data_one, label = assemble_prompt(
                    runtime, query_speed, sigma, prompt_seed, query_seed,
                    clip_speeds=not args.no_clip_speeds,
                )
                metrics = prompt_metrics(
                    runtime, "advection", params, data_one, label, query_speed, args.max_mode
                )
                rows.append({
                    "PDE": "advection", "c_query": query_speed, "sigma": sigma,
                    "prompt_seed": prompt_seed, "query_seed": query_seed,
                    "demo_speeds": json.dumps(demo_speeds.tolist()), **metrics,
                })
                print(query_speed, sigma, repeat + 1, metrics, flush=True)
    fields = ["PDE", "c_query", "sigma", "prompt_seed", "query_seed", "demo_speeds",
              "prediction_error", "spectral_error", "matrix_error", "block_error"]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} raw rows to {args.output}")


if __name__ == "__main__":
    main()
