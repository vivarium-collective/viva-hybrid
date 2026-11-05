# Trial report: Rust hybrid vs Python baselines

## Workload

- Model: bursting gene expression (partitioned, Haseltine-Rawlings hybrid)
- dt = 0.1, t_end = 300.0, sample_times = [20.0, 50.0, 100.0, 200.0, 300.0]
- Replicates: 200 (seeds 0..199); best of 5 repeats of the whole workload
- Work (Rust counters): 631,988 RK4 steps, 61,144 stochastic events
- Timing scope: numba JIT warmed before timing; Rust time is the in-process `sim_seconds` of the simulate loops only (no process start, IR parse, or JSON output); plain/numba time only the per-seed run (IR loading and numba lowering done once, outside the timed region). Hybrid families use first-reach sampling (lag up to dt) while the exact SSA captures pre-event: different conventions by design.

## Timings (best-of-k)

| engine | total ms | us per RK4 step |
|---|---:|---:|
| Rust | 65.9 | 0.104 |
| numba (warm) | 132.4 | 0.209 |
| plain Python | 3035.5 | 4.803 |

## Speedup

- Rust vs plain Python: **46.1x**
- Rust vs numba: **2.01x**

## Phase breakdown (indicative only; not a decision input)

Method: separate instrumented runs, not the headline runs. Rust: `Instant` timers inside the loop (config `phases: true`); the timer calls themselves cost tens of ns each and a whole RK4 step is only ~0.1 us uninstrumented, so the instrumented Rust run is several times slower than the headline run and its shares are dominated by timer overhead inside the RHS regions; the Rust split is indicative at best. Python plain: `cProfile` on a subset of replicates (function-level; profiler overhead inflates call-heavy regions such as the rate closures). numba cannot be split without altering its compiled code and is not broken down. Treat all shares as coarse.

| phase | Rust share | plain-Python share |
|---|---:|---:|
| ODE RHS: continuous reactions | 25% | (in RHS total) |
| propensity eval: stochastic sum inside RHS | 29% | (in RHS total) |
| ODE RHS total | 54% | 78% |
| RK4 combination / step overhead | 37% | 13% |
| event bookkeeping (select, apply, RNG, capture) | 1.2% | in remainder |
| remainder (loop, state copies, capture on accepted steps) | 8% | 9% |

## Verdict

MARGINAL: Rust is 2.01x vs numba (2-5x); decide qualitatively (the phase breakdown is indicative only).

## Machine

Apple M4 Max, macOS-15.2-arm64-arm-64bit, Python 3.12.12


## Interpretation

On this overhead-dominated class (a handful of tiny rate laws, ~0.1 us per RK4 step once compiled) Rust is about 46x faster than plain Python but only about 2.0x faster than numba (2.00x and 2.01x in two full runs), landing exactly on the boundary between the marginal band and "thesis dead". The earlier 2.05x was biased: the numba timing included per-seed Python lowering and a duplicated propensity evaluation per event; both were removed (lowering once outside the timed region; propensities computed once), which moved numba's time by under 1% (the biases were tiny) but the ratio is now an honest apples-to-apples figure. Nearly all of the interpreter-removal win is already available from numba; Rust adds ~2x from statically compiled rate laws and no Python-side call setup. Per-step work is a few flops, so the verdict is: "define abstractly, compile to Rust" buys roughly 2x over a competent numba implementation on this class, not an order of magnitude, and this should be read as marginal-to-negative rather than a pass. Caveats: the numba twin evaluates rate laws through a bytecode interpreter loop, so a code-generated numba kernel might narrow the gap further; the phase table is timer-dominated and indicative only, and does not drive the verdict. Absolute milliseconds are specific to this machine (Apple M4 Max); the ratios are the signal.
