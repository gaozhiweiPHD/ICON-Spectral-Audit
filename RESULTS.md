# Executed reproduction audit

All final representative checks returned zero; this is not a full experiment/training reproduction.
The bounded verification ran on a Slurm compute node, without training. No job remains active.

| Check | Exit code | Seconds |
|---|---:|---:|
| numerical_environment | 0 | 8.68 |
| plot_environment | 0 | 14.88 |
| numerical_dependency_check | 0 | 0.37 |
| plot_dependency_check | 0 | 0.23 |
| solver_convergence | 0 | 3.76 |
| model_and_spectral_reproduction | 0 | 78.46 |
| synthetic_samples | 0 | 15.51 |
| archived_figure_redraws | 0 | 44.93 |
| figure_asset_export | 0 | 0.06 |

Three additional release-integrity tests passed. Local README link targets exist.

## Model metrics against stored rows

One training-range prompt/query seed pair per PDE, with 16 Fourier pairs, was predefined.
All prediction errors matched their archived values. Advection and Allen–Cahn spectral/matrix
metrics matched exactly in this run. Burgers spectral error differed by 7.45e-9 and its
matrix error differed by 3.23e-5 (0.019% relative); both passed the preset tolerances.
No historical controls or all-parameter sweeps were reevaluated.

## Solver comparison and negative outcomes

With JAX_PLATFORMS=cpu, all eight terminal-state refinement rows matched archived CPU
values exactly. The separate initial GPU rerun gave much larger grid-refinement differences
(approximately 7.7e-5–2.9e-4). Those rows are retained; the backend sensitivity cause
is not isolated. This does not certify convergence of reference tangent maps.

The initial missing-CSV packaging failures and full-environment pip check failure are preserved.
Project-specific numerical and separate plotting dependency checks passed.
Figure 16 and Figure 17 have original assets but incomplete array-only historical redraw inputs.
Archived Burgers viscosity 0.1 has no finite prediction/spectral/matrix metrics; it remains
a documented solver failure, not a learned-model error.

## Actual environments

Numerics: Python 3.10.20; jax 0.6.2, jaxlib 0.6.2, flax 0.10.7, optax 0.2.8, einshape 1.0, numpy 2.2.6, scipy 1.15.3.
Numerical GPU device: NVIDIA GeForce RTX 3090. Solver refinement used CPU separately.
Plotting: Python 3.13.9; NumPy 2.3.5, Matplotlib 3.10.6, Pillow 12.0.0.
PyTorch is installed in the recovered numerical environment but is not used by this project.
A fresh installation and the manual GitHub Actions recipe were not executed.

Full JSON reports and raw rerun solver rows are under results/verification/.
Security findings, scope and size inventory are in docs/SECURITY_AUDIT.json.

## GitHub verification

The actual private repository was cloned from GitHub after push. All 33 packaged dataset/checkpoint SHA-256 checks and the three integrity tests passed. The tested remote commit and scope are recorded in results/verification/github_clone_checks.json. Subsequent changes are verification-report and status/inventory documentation only. Anonymous public access has not been enabled or claimed.
