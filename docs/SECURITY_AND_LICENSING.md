# Security, scope and licensing review

Only the ICON project was inventoried. Existing repositories and unrelated
projects were not modified. The private publication copy is a new directory.
No credential stores, SSH keys, cloud configuration or unrelated research data
were read. GitHub authentication is checked with CLI status, never by printing
a token or reading token files.

The candidate excludes logs, notebook outputs, bytecode, private reviewer
correspondence, manuscript PDFs/TeX/Overleaf ZIP, machine-specific Slurm scripts,
redundant figure outputs, and 95 intermediate checkpoint snapshots. The source
inventory records exclusions. Original files remain in place.

`SECURITY_AUDIT.json` reports credential-pattern checks on publication text,
Python syntax, safe NPZ loads, absolute path metadata checks, symlinks and file
sizes. It reports paths/categories only, never a secret value.
Binary weight/PDF/image content is not exhaustively analyzed. Passing a regex
scan is not a proof of absence of confidential information or a full dependency
vulnerability audit. User review is required before public visibility.

All source and candidate files are below GitHub's per-file 100 MiB limit.
The complete 100-snapshot training history is about 1.9 GB and is deliberately
not in Git; a later full-history release should use LFS or an archival service.
This task creates no public archive or invented persistent DOI.

The local project did not contain a LICENSE. MIT for original code and
CC BY 4.0 for original synthetic data, figures and trained weights have been
proposed to the authors; no original-work license is granted until approved.
The public upstream ICON project is MIT licensed. Its Copyright (c) 2023 Liu Yang
notice and complete license are retained; prompt-assembly/mask utilities share
upstream source. Two functions match exactly after AST normalization;
other functions have local modifications. Architecture and prompt-code
attribution is retained conservatively across original experiment scripts.
Exact historical line-by-line origin has not been certified.

Author confirmation of rights and the final license choice are required before
public release. Dependency distributions are not vendored; their licenses
remain separate. All original experimental datasets are synthetic, so no
unpublished third-party experimental data is included.

Private upload has been authorized by the user. Public visibility is a separate
approval and will not be changed by the preparation/push script.

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
