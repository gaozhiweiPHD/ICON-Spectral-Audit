# Reproducibility protocol

The reference manuscript audited here is the October 5, 2026 version of
`paper/main.tex` in the original ICON project: 22 numbered figures and 24
referenced assets. Mathematical appendices are in that source; no separate
supplementary dataset/PDF was found. The older source map used Figures 1–19
and is superseded by this release's figure table. Reviewer correspondence and
manuscript source/PDF/Overleaf ZIP are intentionally excluded from publication.

## Execution scope and versions

See `results/verification/verification_report.json` for executed commands,
exit codes, durations and bounded diagnostic messages; `model_checks.json`
records tolerance checks against the archived numerical rows.
Environment files record exact Python, JAX, jaxlib, Flax, Optax, einshape,
NumPy, SciPy, Matplotlib, Pillow, msgpack and orbax versions.
PyTorch may be installed in an environment but is not imported by the project.

Do not infer a complete result reproduction from source presence or a successful
plot. Full retraining, all parameter sweeps and the reverse-mode 16-pair Burgers
extension have not been rerun in this release audit. Hardware, AD batching and
float32 reductions can change last digits; bitwise identity is not asserted.

The recovered numerical environment lacks Pillow and cannot import Matplotlib
normally. Numerical adapters set `ICON_NUMERICS_ONLY=1`; plotting-only scripts
run in a separate complete Matplotlib environment. A clean combined environment
installation has not been tested.

## Data generation and training

All PDEs are periodic on [0,1), sampled on 100 equispaced points. Initial
conditions have independent Gaussian sine/cosine coefficients scaled by 0.7/m,
16 Fourier modes, and zero bias. The bandwidth control uses 8 modes.
The training pool has 2,000 hidden-parameter draws, each with 20 function pairs,
and is generated in memory, not stored as a separate dataset.
The source PRNG splitting/chunking is authoritative: using NumPy in place of
JAX or changing chunk sizes does not reproduce this pool.

Training seed is 0 for all five training scripts. The main architecture has
6 layers, 8 heads, 256 model dimensions, head dimension 32, widening factor 4,
five demonstrations and one query. AdamW uses learning rate 2e-4, weight decay
1e-4, 2,000 warmup steps, gradient clipping 1, batch size 4 and 200,000 updates.
See `configs/manuscript.json` and the preserved sources for validation pools,
optimizer schedule and every PRNG operation.

Advection: c in [0.5,1.0], T=0.5. Training outputs evaluate the Fourier shift
analytically; fixed gridded evaluation queries use periodic linear interpolation.
The two are distinct numerical reference maps.
Burgers: viscosity in [0.005,0.05], T=0.25; Fourier pseudo-spectral conservative
nonlinearity, 2/3 dealiasing, 800 explicit RK4 steps.
Allen–Cahn: epsilon in [0.02,0.08], T=0.25; pseudo-spectral Laplacian, 800 RK4
steps and clipping to [-3,3] after each completed step.
No adaptive solver tolerance is used.

The corrupted Burgers training control uses one coherent demonstration
viscosity and a different query viscosity. Test prompts are clean and share
one operator. This differs from the advection Gaussian demo-speed perturbation.

Run the five training commands in README. They intentionally keep historical
numerics; training wrappers only redirect outputs. Training completion and
fresh-checkpoint equivalence have not been tested. Do not run training on an
HPC login node.

Small new release prompts (not historical offline-pool extracts):
```bash
python scripts/generate_samples.py --output-dir outputs/samples --seed 20261013
```
The three PDE seeds are 20261013, 20261014, 20261015, respectively; parameters
are 0.75, 0.025, 0.05. Each NPZ contains six pairs of grid inputs/outputs.
Generation calls the preserved solvers and records provenance JSON. It currently
loads the Burgers runtime/checkpoints for compatibility but performs no training.

## Evaluation versus differentiation

Prediction evaluation fixes model parameters and constructs new prompts.
Fourier gain diagnostics differentiate with respect to query values, holding the
context fixed. `prompt_metrics` uses directional JVPs and Fourier projection,
not full 100x100 reverse-mode Jacobian allocation. Gains are directional norms,
not singular values. Projected matrix errors are relative Frobenius errors.
Main error bars are sample SD across 10 contexts for one fixed query per
parameter, not population-wide confidence intervals.

Representative all-three-PDE check:
```bash
python scripts/verify_models.py --output outputs/model_checks.json
```
It regenerates the first archived training-range row per PDE using stored
prompt/query seeds, all 16 Fourier pairs, rtol=0.005 and atol=5e-5.
These tolerances permit backend float32 variation; no universal numerical error
bound follows from passing this test.

Full training-range repeated-context diagnostics, one PDE at a time:
```bash
python scripts/revision/run_prompt_variability.py --pde advection --repeats 10 --max-mode 16 --seed 20260929 --output outputs/advection_id.csv
python scripts/revision/run_prompt_variability.py --pde burgers --repeats 10 --max-mode 16 --seed 20260929 --output outputs/burgers_id.csv
python scripts/revision/run_prompt_variability.py --pde allen_cahn --repeats 10 --max-mode 16 --seed 20260929 --output outputs/allen_cahn_id.csv
```
Full manuscript grids (100, 90 and 90 rows):
```bash
python scripts/revision/run_prompt_variability.py --pde advection --parameters 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 --repeats 10 --max-mode 16 --seed 20260929 --output outputs/advection_full.csv
python scripts/revision/run_prompt_variability.py --pde burgers --parameters 0.002 0.0035 0.005 0.01 0.02 0.035 0.05 0.07 0.1 --repeats 10 --max-mode 16 --seed 20260929 --output outputs/burgers_full.csv
python scripts/revision/run_prompt_variability.py --pde allen_cahn --parameters 0.01 0.015 0.02 0.035 0.05 0.065 0.08 0.10 0.12 --repeats 10 --max-mode 16 --seed 20260929 --output outputs/allen_cahn_full.csv
```
The Burgers c=0.1 reference can become nonfinite under the inherited explicit
solver. Preserve that negative result; do not classify it as network error.
Plotters filter nonfinite metric values. Record valid row counts when analyzing.

Context-size ablation (150 rows per PDE), supported sizes 1/3/5:
```bash
python scripts/revision/run_context_size.py --pde advection --context-sizes 1 3 5 --repeats 10 --seed 20261001 --output outputs/advection_context.csv
python scripts/revision/run_context_size.py --pde burgers --context-sizes 1 3 5 --repeats 10 --seed 20261001 --output outputs/burgers_context.csv
python scripts/revision/run_context_size.py --pde allen_cahn --context-sizes 1 3 5 --repeats 10 --seed 20261001 --output outputs/allen_cahn_context.csv
```
The checkpoint embedding does not support more than five demonstrations.

Noise (120 rows), clipped demo speeds; use `--no-clip-speeds` only for a new,
different experiment:
```bash
python scripts/revision/run_demo_parameter_noise.py --query-speeds 0.6 0.8 1.0 --sigmas 0 0.01 0.025 0.05 --repeats 10 --seed 20261002 --output outputs/noise.csv
```

Training-proximity experiments have 30 queries per PDE; only advection is in
the current manuscript figure. Run each separately:
```bash
python scripts/revision/run_training_proximity.py --pde advection --queries 30 --seed 20261003 --output-dir outputs/proximity-advection
python scripts/revision/run_training_proximity.py --pde burgers --queries 30 --seed 20261003 --output-dir outputs/proximity-burgers
python scripts/revision/run_training_proximity.py --pde allen_cahn --queries 30 --seed 20261003 --output-dir outputs/proximity-allen-cahn
```
The source uses a simple rank routine; results are descriptive correlations.
Reconstruction/distance diagnostics do not establish absence of memorization.

Reference-solver refinement, eight rows:
```bash
JAX_PLATFORMS=cpu python scripts/revision/run_solver_convergence.py --samples 4 --seed 20261004 --output outputs/solver.csv
```
This checks terminal states for four ICs at Burgers 0.02 and Allen–Cahn 0.05:
100/800 against 100/1600 and restricted 200/3200. It does not test tangent-map
convergence, a convergence rate or a uniform discretization bound.

Reverse-Jacobian near-shock 16-pair extension (large memory; not rerun here):
```bash
python scripts/revision/extend_burgers_feature_map.py --output outputs/burgers_16_pairs.npz
```
It uses context seed 44424, viscosity 0.02 and the clean step-200000 checkpoint;
the historical 14-pair block must match before accepting the new arrays.

Documented full checkpoint evaluations, not recertified here:
python scripts/evaluate.py --experiment advection --output-dir outputs/full-advection
python scripts/evaluate.py --experiment allen_cahn --output-dir outputs/full-allen-cahn
python scripts/evaluate.py --experiment advection_frequency8 --output-dir outputs/full-frequency8
python scripts/evaluate.py --experiment burgers_corrupted --output-dir outputs/full-corrupted
Clean Burgers evaluation uses the numerical repeated-prompt commands above;
its historical comprehensive notebook remains preserved without a certified CLI.
The original comprehensive evaluators remain at their recorded paths. Running
them directly can overwrite bundled arrays/figures, so use the provided
`scripts/evaluate.py` wrapper and a fresh output directory instead. Legacy
full evaluations are documented but not recertified in this audit.

## Figures and integrity

```bash
python scripts/reproduce_plots.py --output-dir outputs/redrawn
python scripts/export_figures.py --output-dir outputs/manuscript-assets
python scripts/audit_release.py
sha256sum -c data/MANIFEST.sha256
```
Every archived figure is in the JSON/Markdown manuscript map. Redrawing is
separate from numerical reevaluation. Missing statistics for Figure 17 and
missing archived deterministic prediction arrays for Figure 16 prevent complete
array-only redraw of those panels; their original assets remain available.
Source generators/evaluators for new runs are supplied, without a claim that
they reconstruct the historical image exactly.

Manifests include bundled NPZs, CSVs, samples and final checkpoints. Numeric
arrays are preserved; machine-dependent checkpoint provenance strings were
normalized in three NPZs. See `docs/PACKAGING_CHANGES.json`.
`SOURCE_INVENTORY.csv` includes excluded source files; `FILE_INVENTORY.csv`
lists publication-copy files and hashes, with documented self-hash exclusions.

## Lightweight integrity tests

Run python -m unittest discover -s tests -v to check current figure assets, projected-Jacobian error identities and finite six-pair sample shapes. The manually triggered GitHub Actions workflow has not been executed remotely or validated in a fresh environment.

## Environment limitations discovered in execution

The full recovered environment's pip check fails: its Matplotlib installation
lacks several dependencies, and unrelated installed packages also have missing
dependencies. That genuine report is retained under
results/verification/initial_attempt/. Project-specific numerical and separate
plotting dependency checks inspect only the applicable dependency closure.

The first solver rerun used GPU and produced space/time-refinement differences
of about 7.7e-5–2.9e-4 rather than the archived CPU values. It is retained as a
backend sensitivity observation, not relabeled as agreement. The targeted
comparison now explicitly sets JAX_PLATFORMS=cpu. The cause of GPU differences
has not been isolated; no uniform solver or tangent-convergence guarantee is
claimed. Original code/arrays are preserved, and CPU/GPU runs are distinguished.
