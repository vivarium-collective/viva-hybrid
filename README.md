# viva-hybrid

Trial: does a pure-Rust hybrid ODE/SSA simulator (model defined as SBML + partition metadata) beat equivalent Python (plain + numba) on an overhead-dominated model class?

Go/no-go bar vs the numba baseline: >=5x go, <2x dead, 2-5x marginal. See `docs/superpowers/specs/` and `docs/superpowers/plans/`.
