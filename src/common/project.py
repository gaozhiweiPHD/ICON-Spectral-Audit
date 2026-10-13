"""Canonical project paths.

All new code resolves paths from this file instead of depending on the current
working directory.  ``ICON_PROJECT_ROOT`` may be set to test a relocated copy.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("ICON_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()

RUN_ROOT = Path(os.environ.get("ICON_RUN_ROOT", PROJECT_ROOT)).resolve()
EXPERIMENTS_ROOT = RUN_ROOT / "experiments"
RESULTS_ROOT = RUN_ROOT / "results"
FIGURES_ROOT = RUN_ROOT / "figures"

ADVECTION_CHECKPOINT_DIR = EXPERIMENTS_ROOT / "advection" / "checkpoints" / "main"
BURGERS_CHECKPOINT_DIR = EXPERIMENTS_ROOT / "burgers" / "checkpoints" / "main"
ALLEN_CAHN_CHECKPOINT_DIR = EXPERIMENTS_ROOT / "allen_cahn" / "checkpoints" / "main"

ADVECTION_FIGURE_DIR = EXPERIMENTS_ROOT / "advection" / "figures" / "current"
BURGERS_FIGURE_DIR = EXPERIMENTS_ROOT / "burgers" / "figures" / "current"
ALLEN_CAHN_FIGURE_DIR = EXPERIMENTS_ROOT / "allen_cahn" / "figures" / "current"

ADVECTION_FREQUENCY8_DIR = (
    EXPERIMENTS_ROOT / "advection" / "bandwidth_ablation" / "frequency8_run"
)
ADVECTION_ABLATION_FIGURE_DIR = (
    EXPERIMENTS_ROOT / "advection" / "bandwidth_ablation" / "figures"
)
BURGERS_CORRUPTED_RUN_DIR = (
    EXPERIMENTS_ROOT / "burgers" / "corrupted_prompt" / "run"
)
