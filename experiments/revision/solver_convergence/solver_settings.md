# Reference-solver settings (audited from source)

## Advection

- Domain and boundary condition: `[0,1)`, periodic.
- Model/evaluation grid: 100 equispaced points.
- Horizon: `T = 0.5` (named `dt` in the legacy script).
- Reference solution: analytic periodic shift `u(x,T)=u(x-cT mod 1,0)` for
  generated Fourier functions; periodic linear interpolation is used only when
  a user-supplied gridded base state is shifted.
- No time-stepping tolerance applies.

## Viscous Burgers

- Equation: `u_t + u u_x = nu u_xx`.
- Domain and boundary condition: `[0,1)`, periodic.
- Spatial discretization: Fourier pseudo-spectral derivatives on 100
  equispaced points; conservative nonlinear form `-0.5 d_x(u^2)` with a 2/3
  dealiasing mask.
- Time integrator: explicit classical RK4, 800 fixed steps.
- Horizon: `T = 0.25`; `dt = 0.25/800 = 3.125e-4`.
- Adaptive tolerances: none.

## Allen--Cahn

- Equation: `u_t = epsilon^2 u_xx + u - u^3`.
- Domain and boundary condition: `[0,1)`, periodic.
- Spatial discretization: Fourier pseudo-spectral second derivative on 100
  equispaced points. The nonlinear reaction is evaluated pointwise. No
  dealiasing mask is applied in the recorded generator.
- Time integrator: explicit classical RK4, 800 fixed steps; every step applies
  the inherited safety clip `u in [-3,3]` (normally inactive at the amplitudes
  used).
- Horizon: `T = 0.25`; `dt = 0.25/800 = 3.125e-4`.
- Adaptive tolerances: none.

A small convergence script is provided at
`scripts/revision/run_solver_convergence.py`. It uses identical initial states
for baseline/refined solves and reports final-state relative differences. It is
an evaluation-only diagnostic and does not retrain ICON.

## Completed small audit

Four shared Fourier initial conditions were evaluated on CPU. Relative final
state differences were:

- Burgers, 100/800 versus 100/1600: `3.95e-7` to `5.79e-7`.
- Burgers, 100/800 versus downsampled 200/3200: `7.15e-7` to `4.09e-6`.
- Allen--Cahn, 100/800 versus 100/1600: `4.51e-7` to `6.45e-7`.
- Allen--Cahn, 100/800 versus downsampled 200/3200: `1.03e-6` to `2.03e-6`.

Raw rows are in `solver_convergence.csv`; the compact log-scale plot is
`figures/revision/solver_convergence.{png,pdf}`. These values are substantially
below the discrepancies visible in the existing ICON diagnostic plots.
