"""Thin adapters around the preserved, paper-producing evaluation code."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

from common.project import PROJECT_ROOT


RUNTIME_PATHS = {
    "advection": PROJECT_ROOT / "experiments/advection/evaluation/evaluate_advection.py",
    "burgers": PROJECT_ROOT / "experiments/burgers/corrupted_prompt/evaluate_corrupted_prompt.py",
    "allen_cahn": PROJECT_ROOT / "experiments/allen_cahn/evaluation/evaluate_allen_cahn.py",
}


def load_runtime(pde: str):
    """Load one preserved evaluator without importing any training driver."""
    if pde not in RUNTIME_PATHS:
        raise ValueError(f"unknown PDE {pde!r}")
    os.environ.setdefault("ICON_NUMERICS_ONLY", "1")
    module_name = f"icon_revision_{pde}_runtime"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = RUNTIME_PATHS[pde]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_main_params(pde: str, runtime):
    if pde == "advection":
        return runtime.load_checkpoint(runtime.find_latest_checkpoint(runtime.ckpt_dir))
    if pde == "allen_cahn":
        return runtime.load_checkpoint(runtime.find_latest_checkpoint())
    if pde == "burgers":
        if runtime.params_clean is None:
            raise FileNotFoundError("clean Burgers checkpoint was not loaded")
        return runtime.params_clean
    raise ValueError(pde)


def parameter_name(pde: str) -> str:
    return {"advection": "c", "burgers": "viscosity", "allen_cahn": "epsilon"}[pde]


def default_parameters(pde: str) -> tuple[float, ...]:
    return {
        "advection": (0.5, 0.625, 0.75, 0.875, 1.0),
        "burgers": (0.005, 0.0125, 0.025, 0.0375, 0.05),
        "allen_cahn": (0.02, 0.035, 0.05, 0.065, 0.08),
    }[pde]


def fixed_query(runtime, pde: str, parameter: float, query_seed: int):
    """Generate one query initial condition independently of the demonstrations."""
    key = runtime.jax.random.PRNGKey(query_seed)
    if pde == "advection":
        raw = runtime.generate_shift_raw_data(
            key, 1, 1, runtime.n_grid, runtime.n_modes,
            (parameter, parameter + 1e-8), runtime.dt,
            runtime.fourier_decay, runtime.initial_amp, runtime.bias_scale,
        )
    elif pde == "burgers":
        raw = runtime.generate_burgers_raw_data(
            key, 1, 1, runtime.n_grid, runtime.n_modes,
            (parameter, parameter + 1e-8), runtime.T_final,
            runtime.fourier_decay, runtime.bias_scale, runtime.initial_amp,
            runtime.n_solver_steps,
        )
    else:
        raw = runtime.generate_allen_cahn_raw_data(
            key, 1, 1, runtime.n_grid, runtime.n_modes,
            (parameter, parameter + 1e-8), runtime.T_final,
            runtime.fourier_decay, runtime.bias_scale, runtime.initial_amp,
            runtime.n_solver_steps,
        )
    return raw.cond_v[0, 0, :, 0]


def make_prompt(
    runtime,
    pde: str,
    parameter: float,
    prompt_seed: int,
    query_seed: int,
    n_demo: int = 5,
):
    """Construct one prompt, resampling demos while holding its query fixed."""
    if n_demo > 5:
        raise ValueError("existing checkpoints support at most five demonstrations")
    query_u0 = fixed_query(runtime, pde, parameter, query_seed)
    key = runtime.jax.random.PRNGKey(prompt_seed)
    pair_count = n_demo + 1
    if pde == "advection":
        raw = runtime.generate_shift_raw_data(
            key, 1, pair_count, runtime.n_grid, runtime.n_modes,
            (parameter, parameter + 1e-8), runtime.dt,
            runtime.fourier_decay, runtime.initial_amp, runtime.bias_scale,
        )
        x = raw.qoi_k[:, n_demo, :, 0]
        shifted = runtime.jnp.mod(x - raw.shift[:, None], 1.0)
        query_y = runtime.periodic_linear_interp_grid(query_u0[None, :], shifted)
    elif pde == "burgers":
        raw = runtime.generate_burgers_raw_data(
            key, 1, pair_count, runtime.n_grid, runtime.n_modes,
            (parameter, parameter + 1e-8), runtime.T_final,
            runtime.fourier_decay, runtime.bias_scale, runtime.initial_amp,
            runtime.n_solver_steps,
        )
        query_y = runtime.solve_burgers_batch_jit(
            query_u0[None, :], raw.c, runtime.T_final, nsteps=runtime.n_solver_steps
        )
    else:
        raw = runtime.generate_allen_cahn_raw_data(
            key, 1, pair_count, runtime.n_grid, runtime.n_modes,
            (parameter, parameter + 1e-8), runtime.T_final,
            runtime.fourier_decay, runtime.bias_scale, runtime.initial_amp,
            runtime.n_solver_steps,
        )
        query_y = runtime.solve_allen_cahn_batch_jit(
            query_u0[None, :], raw.eps, runtime.T_final, nsteps=runtime.n_solver_steps
        )
    raw = raw._replace(
        cond_v=raw.cond_v.at[:, n_demo, :, 0].set(query_u0[None, :]),
        qoi_v=raw.qoi_v.at[:, n_demo, :, 0].set(query_y),
    )
    data, label = runtime.build_icon_data(raw, demo_num=n_demo)
    data_one = runtime.tree.tree_map(lambda x: x[0], data)
    return raw, data_one, label[0, 0]


def prompt_metrics(runtime, pde: str, params, data_one, label, parameter: float, max_mode: int = 16):
    """Compute the raw metrics requested by the reviewer for one prompt."""
    pred = runtime.predict_one(params, data_one).reshape(-1)
    truth = label.reshape(-1)
    prediction_error = float(
        runtime.jnp.linalg.norm(pred - truth) / (runtime.jnp.linalg.norm(truth) + 1e-12)
    )
    u0 = data_one.quest_cond_v.reshape(-1)
    gi = np.asarray(runtime.icon_mode_gain_by_jvp_fast(params, data_one, max_mode=max_mode))[:, 1:3]
    gt = np.asarray(runtime.true_mode_gain_by_jvp_u0_fast(u0, parameter, max_mode=max_mode))[:, 1:3]
    spectral_error = float(np.linalg.norm(gi - gt) / (np.linalg.norm(gt) + 1e-12))

    # Compute the Petrov--Galerkin matrix directly from directional JVPs.
    # Forming the full 100x100 Jacobian with jacrev causes a multi-GB reverse-
    # mode attention allocation on some GPUs. Only J Phi is required here, so
    # evaluate the 2*max_mode Fourier directions in small batches instead.
    if pde == "advection":
        basis_fn = runtime.paired_fourier_basis
    elif pde == "burgers":
        basis_fn = runtime.fourier_feature_basis
    else:
        basis_fn = runtime.fourier_pair_bases

    phi = basis_fn(int(u0.shape[0]), max_mode)
    psi = basis_fn(int(pred.shape[0]), max_mode)

    if pde in ("advection", "burgers"):
        icon_map = runtime.icon_query_map(params, data_one)
    else:
        def icon_map(u_flat):
            data_new = data_one._replace(
                quest_cond_v=u_flat.reshape(data_one.quest_cond_v.shape)
            )
            return runtime.predict_one(params, data_new).reshape(-1)

    def projected_jvp_map(function, input_value, input_basis, output_basis, batch_size=4):
        response_chunks = []
        tangents = input_basis.T
        for start in range(0, tangents.shape[0], batch_size):
            tangent_batch = tangents[start:start + batch_size]
            responses = runtime.jax.vmap(
                lambda tangent: runtime.jax.jvp(
                    function, (input_value,), (tangent,)
                )[1]
            )(tangent_batch)
            response_chunks.append(responses)
        j_phi = runtime.jnp.concatenate(response_chunks, axis=0).T
        return output_basis.T @ j_phi

    ai = np.asarray(projected_jvp_map(icon_map, u0, phi, psi))
    if pde == "advection":
        at = np.asarray(runtime.exact_fourier_map(parameter, max_pair_mode=max_mode))
    elif pde == "burgers":
        true_map = lambda u: runtime.true_burgers_map_u0(u, parameter)
        at = np.asarray(projected_jvp_map(true_map, u0, phi, psi))
    else:
        true_map = lambda u: runtime.true_allen_cahn_map_u0(u, parameter)
        at = np.asarray(projected_jvp_map(true_map, u0, phi, psi))
    matrix_error = float(np.linalg.norm(ai - at, ord="fro") / (np.linalg.norm(at, ord="fro") + 1e-12))

    block_error = np.nan
    if pde == "advection":
        block_errors = []
        for mode in range(max_mode):
            sl = slice(2 * mode, 2 * mode + 2)
            block_errors.append(
                np.linalg.norm(ai[sl, sl] - at[sl, sl], ord="fro")
                / (np.linalg.norm(at[sl, sl], ord="fro") + 1e-12)
            )
        block_error = float(np.mean(block_errors))
    return {
        "prediction_error": prediction_error,
        "spectral_error": spectral_error,
        "matrix_error": matrix_error,
        "block_error": block_error,
    }
