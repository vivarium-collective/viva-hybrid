# viva-hybrid — design (trial)

**Date:** 2026-11-05
**Status:** design, pre-approval

## Intent and the why

The Oct 2026 v2ecoli performance investigation (`viva-core/docs/performance-investigation-2026-10.md`) concluded that a pure-Rust engine is *not* a lever for the whole-cell model, because ~93% of each tick is already-compiled numerics (GLPK, LSODA, numba) and only ~7% is engine/glue overhead. But that verdict is a property of v2ecoli's compute profile, not of the engine idea. The open question the investigation left: **for a model class whose per-tick cost is dominated by overhead rather than by a big compiled solver, does defining the model abstractly and running it in pure Rust win — and by how much?**

A **hybrid continuous-ODE / stochastic-SSA** simulation is the ideal test of that thesis:
- Stochastic (SSA) reactions fire many small, cheap events — each event is mostly propensity re-evaluation, RNG, and bookkeeping, i.e. exactly the overhead-dominated regime.
- The ODE↔SSA coupling exchanges state at every event — the "communication overhead" the pure-Rust rewrite is meant to delete.
- If pure Rust can't beat an equivalent Python hybrid *here*, the thesis is dead for every model class. If it wins big here, we've demonstrated the lever and the class of models it applies to.

This is a **trial for measurement**, not a product. Its deliverable is a credible speedup number backed by a fidelity proof, plus a reusable (if minimal) Rust hybrid simulator.

## Goals

1. A standalone `viva-hybrid` repo (Rust library + a bench/CLI binary; Python reference under `python/`).
2. Load a model from **SBML** (species, reactions, kinetic laws, parameters) **+ partition metadata** tagging each reaction `continuous` or `stochastic`.
3. Simulate with a **partitioned hybrid method** (Haseltine–Rawlings coupling): continuous reactions advance as a coupled ODE; stochastic reactions fire as discrete SSA events whose time-dependent propensities are integrated alongside the ODE, firing when the integrated propensity crosses a drawn exponential threshold.
4. **Pure-Rust runtime**, with rate-law evaluation behind a `trait` so a future codegen / embedded JIT (Cranelift/LLVM) can slot in without touching the solver — the "codegen-ready" decision from brainstorming.
5. **Two Python baselines** of the *same* model and *same* method, bracketing the honest range: (a) a **plain-Python** SSA loop ("what people actually write"), and (b) a **numba-jitted** SSA loop (the hardest honest competitor — the compiled regime where this thesis died in v2ecoli). Both honest-quality, no strawman.
6. A **statistical fidelity check**: run N stochastic replicates on both sides; confirm the distributions of key species at sampled times agree within a defined tolerance.
7. A **benchmark report**: wall-clock speedup Rust vs *both* Python baselines, plus where the time goes (ODE RHS vs propensity eval vs event bookkeeping).

## Success criteria & decision rule

This trial feeds a go/no-go on building a Rust engine for overhead-dominated model classes. The threshold is measured against the **numba-jitted baseline** (the honest hardest competitor); the plain-Python number is reported alongside as context. *(applying the user's bar to the compiled baseline is the builder's call.)*

- **≥ 5× over numba → go.** The lever is real for this model class; worth building the engine.
- **< 2× over numba → thesis dead for this class.** Same verdict as the whole cell.
- **2–5× → marginal**; decide with the time-breakdown (is the remaining cost removable, or is it compiled numerics again?).

## Non-goals

- Not general SBML coverage — only the subset the chosen benchmark needs.
- Not codegen/JIT in this trial — the trait seam is built, the second impl is not.
- Not wired into the pbg workspace / vivarium-workbench.
- Not multi-compartment; no SBML events, delays, or algebraic rules unless the benchmark requires them.
- Not distributed; single-threaded per replicate (replicate-level `rayon` parallelism is optional and must be reported separately so it doesn't inflate the per-sim number).

## Anti-goals (failure even if it "works")

- **A speedup number with no fidelity proof.** Agreement with the Python reference comes first; an unvalidated fast simulator proves nothing.
- **A rigged benchmark.** Too trivial (≈100% overhead) just re-runs the toy spike and proves nothing new; too dominated by a big stiff ODE re-runs the whole-cell story. The benchmark must be a *genuine* hybrid with a real SSA event load.
- **A strawman Python baseline.** The reference must use scipy's compiled integrator and vectorized numpy propensities so the measured speedup is honest, not an artifact of a deliberately slow Python.

## Constraints

- **Same computation, no approximation** beyond the hybrid partition itself — and the partition is part of the *model spec* (the user's metadata), not an engine-side accuracy shortcut.
- Rust **stable** toolchain. RNG seedable and reproducible (`rand` + `rand_chacha`) for both the fidelity test and deterministic replicate streams.
- The continuous part is expected **non-stiff** for the default benchmark; if a chosen model is stiff, the integrator choice must account for it (noted under "left to builder").

## The how, as decided in brainstorming

- **Language / layout:** Rust (stable), a Cargo workspace with one crate `viva-hybrid` (lib + bench bin). Python reference lives in `python/` (libsbml + scipy + numpy).
- **Method:** Haseltine–Rawlings partitioned hybrid (continuous ODE for fast reactions + integrated-propensity SSA for slow reactions, coupled on one time axis).
- **Rate-law evaluation:** parse each kinetic law's math (from SBML) once into a compact closure/bytecode form implementing `trait RateLaw { fn eval(&self, state: &State) -> f64 }`. Closure-based now; codegen/JIT is a later impl behind the same trait. (Not a naive per-eval tree-walk.)
- **Model input:** SBML for the math + a **partition sidecar** (`partition.yaml`: `reaction_id -> continuous | stochastic`) for the metadata. Sidecar chosen over SBML `<annotation>` for trial simplicity and robustness.
- **Validation:** N replicates each side; compare mean and variance (and distribution) of key species at sampled times within tolerance.

## Decisions settled during review (builder's call — flagged for you to veto)

- **SBML parsing via a shared IR, not native Rust.** Python/libsbml parses the SBML + partition sidecar once into a shared JSON IR (species, reactions, parsed rate-law ASTs, parameters, partition tags). **Both** the Rust runtime and the Python reference consume the *identical* IR. Parsing is one-time and outside the measured hot loop, so this costs zero fidelity, saves a large native-MathML-parser build, and guarantees both sides run the same math.
- **Fair comparison is lock-stepped, not idiomatic-best.** Because the user's intent is "the *same* computation in pure Rust," both sides use the **same numerical method, integration order, tolerances, and RNG draw order** — the only thing that differs by language is the evaluation machinery (the measured variable). This forces an integrator implementable identically on both sides (e.g. explicit Dormand–Prince RK45 with matched tolerances), *not* scipy-LSODA vs a different Rust scheme.
- **Fidelity = impl agreement + an exact-SSA sanity cross-check.** The primary proof is Rust-hybrid vs Python-hybrid agreeing (proves the Rust port is correct). Added: a full-SSA (exact Gillespie) run of the benchmark to confirm the Haseltine–Rawlings partition is itself valid in the chosen regime — otherwise two implementations could agree on a scientifically wrong answer. Cheap for a small benchmark; makes the number honest.

## What's left to the builder

- **Benchmark model + source.** Default: a bursting gene-expression network (stochastic promoter on/off + transcription bursts; continuous/faster protein dynamics) — a real SSA load *and* a continuous part. Builder picks exact species/reactions/rate constants from a cited source, or swaps in an equivalent genuine hybrid (e.g. a predator–prey with a continuous resource). Schlögl is a fallback stress model but is pure-SSA bistable, so less representative of the hybrid coupling.
- **Integrator:** explicit adaptive RK (Dormand–Prince / RK45) via a crate (`ode_solvers`) vs hand-rolled — but constrained to be implementable *identically* on both sides per the lock-step decision above, and decided against the measured stiffness of the chosen model.
- **SBML subset:** which MathML constructs to support — exactly what the benchmark's kinetic laws use (mass-action and Hill/MM at most, likely).
- **Fidelity test specifics:** which species, which timepoints, moment-matching vs a Kolmogorov–Smirnov test, and the numeric tolerance / replicate count.
- **Repo hygiene:** whether the trial gets CI, licensing header, etc. (minimal for a trial).
