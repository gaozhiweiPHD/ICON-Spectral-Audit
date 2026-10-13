# Statistical scope of archived diagnostics

The 280 full-range repeated-context rows cover one fixed query per parameter,
with ten independent contexts. The 150 training-range rows are separately
retained; do not pool overlapping protocols as independent repetitions.
The three context-size CSVs have 450 rows (sizes 1/3/5, five parameters,
ten contexts per group). The noise experiment has 120 rows; the three
training-proximity files contain 90 queries in total, with only 30 advection
queries used by the current manuscript. Solver refinement has eight rows.

Main error bars are one sample standard deviation, not 95% confidence
intervals and not uncertainty across training seeds. Detailed gain spectra
use the original protocols (e.g. mean gains over five contexts, followed by
error calculation); this differs from averaging per-context errors.
The bandwidth control's original prediction experiment uses twelve prompts.

results/verification/archived_statistics.json records finite counts, means and
sample SD for every archived full-range parameter/metric group. NaN block
errors for nonlinear PDEs mean not applicable. Nonfinite Burgers reference
metrics at viscosity 0.1 are a solver limitation and are explicitly excluded
from numerical comparison, never treated as learned-model errors.
No claim of statistical significance, generalization across training seeds,
uniform solver accuracy or a causal interpretation of proximity correlations
is made by this release audit.
