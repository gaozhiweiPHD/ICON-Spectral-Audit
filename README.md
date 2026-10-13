# Spectral Audit of In-Context Operator Networks

Reproducibility materials for the manuscript submitted to **npj Artificial
Intelligence**. This is an independently organized copy of the authors' ICON
working project. Original experiments and manuscript files were left intact.

**Publication status:** [GitHub repository](https://github.com/gaozhiweiPHD/ICON-Spectral-Audit) is private and has been pushed successfully. Public release approval and
original-work licensing confirmation are still required. See [STATUS.md](STATUS.md) for actual test results,
licensing decisions and missing materials. No full retraining or full parameter
sweep is claimed to have been reproduced.

The models use **JAX/Flax**, not PyTorch. All experimental data are generated
synthetically from periodic one-dimensional PDE solvers. Five final trained
parameter checkpoints are included; intermediate snapshots remain in the
original research project.

| Directory | Contents |
|---|---|
| `src/` | Prompt utilities, runtime adapters, plotting style |
| `experiments/` | Original training/evaluation code, final checkpoints, NPZ/CSV results |
| `configs/` | Recovered manuscript settings and environment records |
| `data/` | Small release samples, provenance, SHA-256 dataset manifests |
| `results/` | Repeated-context experiments and release verification reports |
| `figures/manuscript/` | 24 preserved assets referenced by the current 22 figures |
| `scripts/` | Portable reproduction, training, diagnostics, plotting and audit commands |
| `docs/` | Inventory, figure mapping, reproducibility and security/licensing reports |

Start with [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
[docs/MANUSCRIPT_FIGURES.md](docs/MANUSCRIPT_FIGURES.md) maps every current
figure to its data and commands, including missing redraw inputs.

## Installation

A clean installation is a **recipe**, not a tested environment recreation.
The executed numerical tests use the recovered Python 3.10.20/JAX environment;
plotting tests use the separately recorded Python 3.13.9 environment.
Their complete relevant version reports are in `results/verification/`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

The default requirements select CPU JAX. GPU installation is hardware dependent;
use a compatible JAX installation and record versions with
`python scripts/environment_report.py`. PyTorch is not needed for these scripts.

## Representative verification and figures

Run from the repository root, on a compute node when using HPC:

```bash
python scripts/verification_suite.py
python scripts/reproduce_plots.py --output-dir outputs/redrawn
python scripts/export_figures.py --output-dir outputs/manuscript-assets
python scripts/generate_samples.py --output-dir outputs/samples --seed 20261013
python scripts/audit_release.py
sha256sum -c data/MANIFEST.sha256
```

The verification suite also accepts `--numerical-python` and `--plot-python`
for separate environments. It writes new outputs rather than replacing archived
data. `export_figures.py` copies preserved assets; it **does not** recompute
numerical results. Array redraws do not promise byte-identical image/layout
reproduction.

## Evaluation and spectral diagnostics

```bash
python scripts/verify_models.py --output outputs/model_checks.json
python scripts/revision/run_prompt_variability.py --pde advection --repeats 10 --max-mode 16 --seed 20260929 --output outputs/advection_prompt_stats.csv
JAX_PLATFORMS=cpu python scripts/revision/run_solver_convergence.py --samples 4 --seed 20261004 --output outputs/solver_convergence.csv
```

The first command regenerates one archived row for each PDE with prediction,
directional Fourier gains and projected-Jacobian errors. It is a representative
check, not certification of all archived results. Full experiment commands and
parameter grids are in the reproducibility guide.

## Training

Training is computationally expensive and was **not rerun** for this release.
The preserved Burgers drivers execute training at module level; never import
them to evaluate a checkpoint. Use the explicit wrapper and an empty output root:

```bash
python scripts/train.py --experiment advection --run-root outputs/train-advection
python scripts/train.py --experiment burgers --run-root outputs/train-burgers
python scripts/train.py --experiment allen_cahn --run-root outputs/train-allen-cahn
python scripts/train.py --experiment advection_frequency8 --run-root outputs/train-frequency8
python scripts/train.py --experiment burgers_corrupted --run-root outputs/train-corrupted
```

These are documented full-run entry points, not tested full-training claims.
Seed 0, 200,000 updates and the original settings are preserved. Parameter-only
checkpoints do not contain optimizer state for exact training resumption.

## Data, rights and provenance

Read [data/README.md](data/README.md), [docs/SECURITY_AND_LICENSING.md](docs/SECURITY_AND_LICENSING.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE).
No external experimental dataset is bundled. Public visibility will only be
enabled after author approval; final journal availability statements depend on
verification of the resulting public URL.
