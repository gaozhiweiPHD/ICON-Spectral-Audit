#!/usr/bin/env python3
"""Small reference-solver convergence audit; no ICON training or checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from common.project import EXPERIMENTS_ROOT


def random_fourier(key, n, samples, modes=16, amplitude=0.7):
    coeff = jax.random.normal(key, (samples, 2, modes)) * (
        amplitude / jnp.arange(1, modes + 1, dtype=jnp.float32)
    )
    x = jnp.arange(n, dtype=jnp.float32) / n
    phase = 2.0 * jnp.pi * jnp.arange(1, modes + 1)[:, None] * x[None, :]
    return coeff[:, 0] @ jnp.sin(phase) + coeff[:, 1] @ jnp.cos(phase)


def rk4_scan(u0, parameter, nsteps, rhs):
    dt = 0.25 / nsteps
    def body(u, _):
        k1 = rhs(u, parameter)
        k2 = rhs(u + 0.5 * dt * k1, parameter)
        k3 = rhs(u + 0.5 * dt * k2, parameter)
        k4 = rhs(u + dt * k3, parameter)
        return u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, None
    return jax.lax.scan(body, u0, None, length=nsteps)[0]


def burgers_solve(u0, viscosity, nsteps):
    n = u0.shape[-1]
    frequency = jnp.fft.fftfreq(n, d=1.0 / n)
    wave = 2.0 * jnp.pi * frequency
    dealias = (jnp.abs(frequency) <= n / 3.0).astype(jnp.complex64)
    def rhs(u, nu):
        u_hat = jnp.fft.fft(u, axis=-1)
        u2_hat = jnp.fft.fft(u * u, axis=-1) * dealias
        return (-0.5 * jnp.fft.ifft(1j * wave * u2_hat, axis=-1).real
                + nu * jnp.fft.ifft(-(wave ** 2) * u_hat, axis=-1).real)
    return rk4_scan(u0, viscosity, nsteps, rhs)


def allen_cahn_solve(u0, epsilon, nsteps):
    n = u0.shape[-1]
    wave = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=1.0 / n)
    def rhs(u, eps):
        u_xx = jnp.fft.ifft(-(wave ** 2) * jnp.fft.fft(u, axis=-1), axis=-1).real
        return eps ** 2 * u_xx + u - u ** 3
    # The training generator clips each completed RK4 step. Implement explicitly.
    dt = 0.25 / nsteps
    def body(u, _):
        k1 = rhs(u, epsilon); k2 = rhs(u + 0.5 * dt * k1, epsilon)
        k3 = rhs(u + 0.5 * dt * k2, epsilon); k4 = rhs(u + dt * k3, epsilon)
        updated = u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return jnp.clip(updated, -3.0, 3.0), None
    return jax.lax.scan(body, u0, None, length=nsteps)[0]


def rel_rows(reference, comparison):
    reference = np.asarray(reference); comparison = np.asarray(comparison)
    return np.linalg.norm(comparison - reference, axis=1) / (np.linalg.norm(reference, axis=1) + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--output", type=Path, default=EXPERIMENTS_ROOT / "revision/solver_convergence/solver_convergence.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")
    rows = []
    settings = [("burgers", 0.02, burgers_solve), ("allen_cahn", 0.05, allen_cahn_solve)]
    for pde_index, (pde, parameter, solve) in enumerate(settings):
        key = jax.random.PRNGKey(args.seed + pde_index)
        # Same Fourier coefficients are reproduced at both resolutions by reusing key.
        u100 = random_fourier(key, 100, args.samples)
        u200 = random_fourier(key, 200, args.samples)
        baseline = solve(u100, parameter, 800)
        time_refined = solve(u100, parameter, 1600)
        # 3200 keeps the N=200 explicit Burgers diffusion step inside RK4 stability.
        space_time_refined = solve(u200, parameter, 3200)[:, ::2]
        time_error = rel_rows(time_refined, baseline)
        space_time_error = rel_rows(space_time_refined, baseline)
        for sample in range(args.samples):
            rows.append({
                "PDE": pde, "parameter": parameter, "seed": args.seed + pde_index,
                "sample": sample, "baseline_grid": 100, "baseline_steps": 800,
                "time_refined_steps": 1600, "space_refined_grid": 200,
                "space_time_refined_steps": 3200,
                "baseline_vs_time_refined_rel_l2": float(time_error[sample]),
                "baseline_vs_space_time_refined_rel_l2": float(space_time_error[sample]),
            })
        print(pde, "time-refined", time_error, "space+time-refined", space_time_error, flush=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"saved {len(rows)} raw rows to {args.output}")


if __name__ == "__main__":
    main()
