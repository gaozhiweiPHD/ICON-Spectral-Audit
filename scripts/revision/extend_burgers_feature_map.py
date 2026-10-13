#!/usr/bin/env python3
"""Extend Figure 8 to 16 Fourier pairs with its original context and checkpoint.

The saved 14-pair block must be reproduced before the new arrays are accepted.
This is inference and differentiation only; no model weights are updated.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.revision_runtime import load_runtime, load_main_params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/revision/feature_map/burgers_near_shock_16_pairs.npz")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    runtime = load_runtime("burgers")
    params = load_main_params("burgers", runtime)
    base = runtime.make_burgers_base_state("near_shock", n_grid=100)
    # Match the notebook's make_fixed_c_batch (including its narrow interval).
    raw = runtime.generate_burgers_raw_data(
        runtime.jax.random.PRNGKey(44424), 1, runtime.num_pairs,
        runtime.n_grid, runtime.n_modes, (0.02, 0.02 + 1e-9),
        runtime.T_final, runtime.fourier_decay, runtime.bias_scale,
        runtime.initial_amp, runtime.n_solver_steps,
    )
    data, _ = runtime.build_icon_data(raw, demo_num=5)
    context = runtime.tree.tree_map(lambda x: x[0], data)
    context = context._replace(quest_cond_v=base)
    u0 = context.quest_cond_v.reshape(-1)
    basis = runtime.fourier_feature_basis(100, 16)

    def project(function):
        # The original figure used reverse-mode AD. Reproduce that operation;
        # float32 forward-mode PDE derivatives differ measurably here.
        # Preserve the original jacrev batching too: changing batch size alters
        # the float32 attention reductions enough to fail the reproduction test.
        jacobian = runtime.jax.jacrev(function)(u0)
        jacobian.block_until_ready()
        print("completed original full-Jacobian reverse derivative", flush=True)
        return np.asarray(basis.T @ jacobian @ basis)

    true_map = project(lambda u: runtime.true_burgers_map_u0(u, 0.02))
    icon_map = project(runtime.icon_query_map(params, context))
    old_path = ROOT / "experiments/burgers/figures/current/burgers_paper_style_diagnostics_results.npz"
    with np.load(old_path) as old:
        checks = {}
        for name, matrix in (("true", true_map), ("icon", icon_map)):
            reference = old[f"feature_near_shock_A_{name}"]
            delta = matrix[:28, :28] - reference
            checks[name] = float(np.linalg.norm(delta) / np.linalg.norm(reference))
            print(f"Original {name} block relative discrepancy: {checks[name]}", flush=True)
            if not np.isfinite(checks[name]) or checks[name] > 1e-5:
                raise RuntimeError(f"Original 14-pair block failed reproduction: {name} {checks[name]}")
    error = icon_map - true_map
    metadata = dict(parameter=0.02, context_seed=44424, fourier_pairs=16,
                    original_pairs=14, original_block_relative_discrepancies=checks,
                    clean_checkpoint=str(runtime.loaded_clean_ckpt),
                    relative_frobenius_error=float(np.linalg.norm(error) / np.linalg.norm(true_map)),
                    original_data=str(old_path.relative_to(ROOT)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, feature_near_shock_A_true=true_map,
             feature_near_shock_A_icon=icon_map, feature_near_shock_A_err=error,
             metadata=np.asarray(json.dumps(metadata)))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
