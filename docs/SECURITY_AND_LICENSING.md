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
vulnerability audit. The user reviewed the candidate and approved public visibility on 13 October 2026.

All source and candidate files are below GitHub's per-file 100 MiB limit.
The complete 100-snapshot training history is about 1.9 GB and is deliberately
not in Git; a later full-history release should use LFS or an archival service.
This task creates no public archive or invented persistent DOI.

The original local project did not contain a LICENSE. The user has now
confirmed redistribution rights and approved MIT for original code and
CC BY 4.0 for original synthetic data, figures and trained weights.
The complete texts are in LICENSE and LICENSE-DATA; scope is in docs/LICENSING.md.
The public upstream ICON project is MIT licensed. Its Copyright (c) 2023 Liu Yang
notice and complete license are retained; prompt-assembly/mask utilities share
upstream source. Two functions match exactly after AST normalization;
other functions have local modifications. Architecture and prompt-code
attribution is retained conservatively across original experiment scripts.
Exact historical line-by-line origin has not been certified.

The user explicitly confirmed rights, selected the licenses and approved
public release on 13 October 2026. Dependency distributions are not vendored; their licenses
remain separate. All original experimental datasets are synthetic, so no
unpublished third-party experimental data is included.

Private upload has been authorized by the user. Public visibility was separately approved by the user and is changed only
through the approved publication step, not the private-repository creation helper.

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

## Actual upload status

The cleaned candidate has been pushed to https://github.com/gaozhiweiPHD/ICON-Spectral-Audit with initial PRIVATE visibility verified, then changed to PUBLIC after user approval. The initial authentication blocker is resolved. No pre-existing remote repository was overwritten. The user subsequently approved public release and the original-work licenses on 13 October 2026. Anonymous public HTML/API/raw download and credential-free clone access passed; see results/verification/public_access_checks.json.
