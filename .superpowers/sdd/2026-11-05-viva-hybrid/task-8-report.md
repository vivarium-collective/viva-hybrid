# Task 8 report - numba hybrid simulator

RED: test collection failed (ModuleNotFoundError hybrid_numba). GREEN: 5 passed (continuous analytic, determinism, no-negative, big-dt single event, statistical agreement vs simulate_plain over 1000 seeds on birth-death).

## Lowering
`lower(model)` (plain Python, once) yields: x0 (f64), S int64 [reaction, species], is_stoch mask, cont_idx/stoch_idx (reaction indices in model order per partition), and flat rate programs.
Op-code format: postfix, all reactions concatenated; reaction r owns [off[r], off[r+1]). Arrays code (int64: 0 const, 1 species-read, 2 add, 3 sub, 4 mul, 5 div, 6 pow), fval (const operand; params folded to const), ival (species index). `_eval` is an @njit stack machine; stack depth computed at lowering. div matches compile_rate (x/0 -> +-inf, 0/0 -> nan).
The whole loop (`_run`: RK4 on augmented state, crossing + re-integration, selection, guard, first-reach sampling) is @njit(cache=True) over these arrays, same algorithm and draw order as Task 6/7.

## RNG
`np.random.seed(seed)` + `np.random.random()` inside njit; redraw on 0.0 so uniform in (0,1). Different stream from Task 7/Rust; same-seed determinism tested; fidelity is statistical.

## Notes
No numba constraints blocked anything. First call compiles (~5 s; cached to __pycache__ via cache=True, gitignored). Pow uses math.pow (raises/nan differences vs Python's ValueError->nan path are not exercised).
