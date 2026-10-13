# Release preparation status

Local publication copy is organized and representative checks have passed.
The original ICON working project was not modified.
A new PRIVATE GitHub repository has not been created. The guarded upload was
attempted and stopped before creation because GitHub CLI is not authenticated.
The user must configure authentication in their own terminal.
Public visibility has not been authorized.

## Executed checks

- Three model/checkpoint tests passed: one archived seed pair per PDE,
  predictions and all 16 Fourier-pair JVP metrics; rtol=0.005, atol=5e-5.
- CPU solver refinement regenerated all eight archived numerical rows exactly.
- Fifteen archived-array/CSV plotting commands returned zero.
- All 24 current manuscript assets exported successfully.
- Three six-pair synthetic example prompts generated successfully.
- Three scientific-array/figure/sample integrity tests passed.
- A real local Git clone passed all 33 dataset/checkpoint SHA-256 checks and
  the three integrity tests; CSV bytes survive Git transport unchanged.
- Numerical and separate plotting dependency closures passed.
- Local document-link and evaluator-symbol checks passed.
- Candidate security scan had no credential, machine-path, unsafe-NPZ,
  file-size or Python-syntax findings. See audit scope limitations.

## Preserved failures and limitations

The first packaging attempt omitted revision CSVs and failed corresponding
model/plot checks; the CSVs were restored and all those checks passed on rerun.
The full recovered environment's pip check fails (partially installed plotting
packages and unrelated package dependencies). It was not silently repaired.
GPU solver refinement produced discrepancies different from historical CPU
values; both are preserved under results/verification/initial_attempt/.
The rerun explicitly uses CPU for this comparison.

Missing or not certified:
- Figure 17 lacks archived prediction means/SDs required for exact redraw.
- Figure 16's deterministic Allen–Cahn prediction arrays were not saved in
  its main NPZ; its original image and evaluator are preserved.
- Training pools were generated in memory, not stored separately.
- Allen–Cahn training curves were not located.
- Checkpoints contain parameters, not exact-resume optimizer state.
- Historical image/checkpoint linkage is inferred from final-step defaults and
  records, without a complete original build log.
- Full retraining, full sweeps, control-model reevaluation, the reverse-mode
  Burgers extension, clean environment installation and remote CI were not run.
- Solver checks do not establish tangent-map convergence or a uniform bound.
- No separate supplementary file was found; appendices are in the manuscript.
- Original-work rights/license choice await author confirmation.
- Final journal availability statements await a verified public URL.

## Before publication

Authenticate GitHub CLI in the user's own terminal, confirm the target owner
and licenses, push the reviewed copy to a new private repository, and verify
the remote commit. Then provide the URL/inventory/security report for explicit
public-visibility approval. No helper in this repository can make it public.
