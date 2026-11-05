# Trial report: Rust hybrid vs Python baselines

## Workload

- Model: bursting gene expression (partitioned, Haseltine-Rawlings hybrid)
- dt = 0.1, t_end = 300.0, sample_times = [20.0, 50.0, 100.0, 200.0, 300.0]
- Replicates: 200 (seeds 0..199); best of 5 repeats of the whole workload
- Work (Rust counters): 631,988 RK4 steps, 61,144 stochastic events
- Timing scope: numba JIT warmed before timing; Rust time is the in-process `sim_seconds` of the simulate loops only (no process start, IR parse, or JSON output); plain/numba time only `simulate_*` calls (IR loading excluded).

## Timings (best-of-k)

| engine | total ms | us per RK4 step |
|---|---:|---:|
| Rust | 64.5 | 0.102 |
| numba (warm) | 132.2 | 0.209 |
| plain Python | 3021.8 | 4.781 |

## Speedup

- Rust vs plain Python: **46.8x**
- Rust vs numba: **2.05x**

## Phase breakdown (approximate)

Method: separate instrumented runs, not the headline runs. Rust: `Instant` timers inside the loop (config `phases: true`); the timer calls themselves cost tens of ns each and a whole RK4 step is only ~0.1 us uninstrumented, so the instrumented Rust run is several times slower than the headline run and its shares are dominated by timer overhead inside the RHS regions; the Rust split is indicative at best. Python plain: `cProfile` on a subset of replicates (function-level; profiler overhead inflates call-heavy regions such as the rate closures). numba cannot be split without altering its compiled code and is not broken down. Treat all shares as coarse.

| phase | Rust share | plain-Python share |
|---|---:|---:|
| ODE RHS: continuous reactions | 24% | (in RHS total) |
| propensity eval: stochastic sum inside RHS | 29% | (in RHS total) |
| ODE RHS total | 54% | 78% |
| RK4 combination / step overhead | 37% | 13% |
| event bookkeeping (select, apply, RNG, capture) | 1.2% | in remainder |
| remainder (loop, state copies, capture on accepted steps) | 8% | 9% |

## Verdict

MARGINAL: Rust is 2.05x vs numba (2-5x); decide from the phase breakdown.

## Machine

Apple M4 Max, macOS-15.2-arm64-arm-64bit, Python 3.12.12


## Interpretation

On this overhead-dominated class (a handful of tiny rate laws, ~0.1 us per RK4 step once compiled) Rust is about 47x faster than plain Python but only about 2.05x faster than numba, which sits right at the 2x floor of the marginal band. Nearly all of the interpreter-removal win is already available from numba; Rust adds a further ~2x from tighter inner loops (statically compiled rate laws behind trait objects, no AST-bytecode interpreter as in the numba twin) and no Python-side call setup. Per-step work is a few flops, so there is little left for a compiler to win: this verdict says "define abstractly, compile to Rust" is worth roughly 2x over a competent numba implementation on this class, not an order of magnitude, and the 2.05x is close enough to the 2x threshold that it should not be called a clear pass. Caveats: the numba twin evaluates rate laws through a bytecode interpreter loop rather than specialised code, so a codegen'd numba kernel might narrow the gap, and the Rust phase split is dominated by instrumentation overhead. Absolute milliseconds are specific to this machine (Apple M4 Max); the ratios are the signal, and the 2.05x is stable across two full runs (2.07x, 2.05x).
