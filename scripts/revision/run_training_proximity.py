#!/usr/bin/env python3
"""Nearest-neighbor diagnostic against the exactly reconstructed offline IC pool."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.project import EXPERIMENTS_ROOT
from models.revision_runtime import default_parameters, load_main_params, load_runtime, make_prompt


def fourier_values(coeff, n_grid=100):
    x = np.arange(n_grid, dtype=np.float32) / n_grid
    modes = np.arange(1, coeff.shape[-1] + 1, dtype=np.float32)
    phase = 2.0 * np.pi * modes[:, None] * x[None, :]
    sin_basis, cos_basis = np.sin(phase), np.cos(phase)
    return np.einsum("...k,kn->...n", coeff[..., 0, :], sin_basis) + np.einsum(
        "...k,kn->...n", coeff[..., 1, :], cos_basis
    )


def reconstruct_training_inputs(runtime, pde):
    """Reproduce PRNG draws for the 2,000 x 20 training initial conditions."""
    base_key = runtime.jax.random.PRNGKey(0)
    scale = runtime.initial_amp / (runtime.jnp.arange(1, runtime.n_modes + 1) ** runtime.fourier_decay)
    if pde in ("advection", "allen_cahn"):
        train_key = runtime.jax.random.split(base_key, 3)[0]
        coefficient_key = runtime.jax.random.split(train_key, 3)[1]
        coeff = runtime.jax.random.normal(
            coefficient_key, (2000, 20, 2, runtime.n_modes)
        ) * scale
        return fourier_values(np.asarray(coeff), runtime.n_grid).reshape(-1, runtime.n_grid)
    train_key = runtime.jax.random.split(base_key, 4)[1]
    chunks = []
    n_done = 0
    chunk_id = 0
    while n_done < 2000:
        n_chunk = min(64, 2000 - n_done)
        key_i = runtime.jax.random.fold_in(train_key, chunk_id)
        coefficient_key = runtime.jax.random.split(key_i, 3)[1]
        coeff = runtime.jax.random.normal(
            coefficient_key, (n_chunk, 20, 2, runtime.n_modes)
        ) * scale
        chunks.append(fourier_values(np.asarray(coeff), runtime.n_grid).reshape(-1, runtime.n_grid))
        n_done += n_chunk
        chunk_id += 1
    return np.concatenate(chunks, axis=0)


def nearest_relative_distance(query, pool, chunk=4096):
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    qnorm = np.linalg.norm(query) + 1e-12
    best = np.inf
    for start in range(0, len(pool), chunk):
        distances = np.linalg.norm(pool[start:start + chunk] - query[None, :], axis=1) / qnorm
        best = min(best, float(np.min(distances)))
    return best


def prediction_spectral(runtime, pde, params, data_one, label, parameter, max_mode):
    pred = runtime.predict_one(params, data_one).reshape(-1)
    truth = label.reshape(-1)
    prediction = float(runtime.jnp.linalg.norm(pred - truth) / (runtime.jnp.linalg.norm(truth) + 1e-12))
    u0 = data_one.quest_cond_v.reshape(-1)
    icon = np.asarray(runtime.icon_mode_gain_by_jvp_fast(params, data_one, max_mode=max_mode))[:, 1:3]
    true = np.asarray(runtime.true_mode_gain_by_jvp_u0_fast(u0, parameter, max_mode=max_mode))[:, 1:3]
    spectral = float(np.linalg.norm(icon - true) / (np.linalg.norm(true) + 1e-12))
    return prediction, spectral


def rankdata(values):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", required=True, choices=["advection", "burgers", "allen_cahn"])
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--max-mode", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS_ROOT / "revision/training_proximity")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    csv_path = args.output_dir / f"{args.pde}_training_proximity.csv"
    if csv_path.exists() and not args.overwrite:
        raise FileExistsError("refusing to overwrite proximity outputs; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(args.pde)
    params = load_main_params(args.pde, runtime)
    pool = reconstruct_training_inputs(runtime, args.pde)
    parameters = default_parameters(args.pde)
    rows = []
    for index in range(args.queries):
        parameter = parameters[index % len(parameters)]
        prompt_seed = args.seed + index
        query_seed = args.seed + 100_000 + index
        _, data_one, label = make_prompt(runtime, args.pde, parameter, prompt_seed, query_seed)
        query = np.asarray(data_one.quest_cond_v).reshape(-1)
        distance = nearest_relative_distance(query, pool)
        prediction, spectral = prediction_spectral(runtime, args.pde, params, data_one, label, parameter, args.max_mode)
        rows.append({"PDE": args.pde, "parameter": parameter, "prompt_seed": prompt_seed,
                     "query_seed": query_seed, "d_NN": distance,
                     "prediction_error": prediction, "spectral_error": spectral})
        print(index + 1, distance, prediction, spectral, flush=True)
    fields = list(rows[0])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    d = np.asarray([r["d_NN"] for r in rows])
    pred = np.asarray([r["prediction_error"] for r in rows])
    spec = np.asarray([r["spectral_error"] for r in rows])
    pearson_pred = float(np.corrcoef(d, pred)[0, 1]); pearson_spec = float(np.corrcoef(d, spec)[0, 1])
    spearman_pred = float(np.corrcoef(rankdata(d), rankdata(pred))[0, 1])
    spearman_spec = float(np.corrcoef(rankdata(d), rankdata(spec))[0, 1])
    print(f"prediction associations: Pearson={pearson_pred:.4f}, Spearman={spearman_pred:.4f}")
    print(f"spectral associations: Pearson={pearson_spec:.4f}, Spearman={spearman_spec:.4f}")
    print("Associations are descriptive only; they do not establish or refute memorization.")


if __name__ == "__main__":
    main()
