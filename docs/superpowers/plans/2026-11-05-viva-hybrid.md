# viva-hybrid Trial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a pure-Rust hybrid ODE/SSA simulator, fed a model defined as SBML + partition metadata, beats equivalent Python (plain and numba) on an overhead-dominated model class — with fidelity proven first.

**Architecture:** Python/libsbml parses SBML + a partition sidecar once into a shared JSON **IR** (species, parameters, reactions with stoichiometry, parsed rate-law ASTs, partition tags). Both the Rust runtime and the Python reference consume that identical IR. The simulator is a Haseltine–Rawlings partitioned hybrid: an augmented ODE `[species; integrated-stochastic-propensity]` advanced by a fixed-step RK4, firing a discrete stochastic jump when the integrated propensity crosses a drawn exponential threshold. Rate-law evaluation sits behind a Rust `trait` so a later codegen/JIT impl can replace the closure-based one without touching the solver.

**Tech Stack:** Rust stable (crate `viva-hybrid`; `serde`/`serde_json`, `rand` + `rand_chacha`). Python 3 (`libsbml`, `numpy`, `scipy`, `numba`, `pytest`, `pyyaml`).

**Spec:** `docs/superpowers/specs/2026-11-05-viva-hybrid-design.md` (currently at `scratchpad/viva-hybrid-design.md`; Task 1 commits it into the repo).

## Global Constraints

- **Rust stable toolchain only.** No nightly features.
- **Same computation, no approximation beyond the partition.** The partition (which reactions are continuous vs stochastic) is part of the *model spec*, not an engine shortcut.
- **Lock-stepped numerics across languages:** identical method (fixed-step RK4), identical step `dt`, identical event-crossing rule, and identical RNG *draw order and count per event* on both sides. The engine machinery (closure/bytecode in Rust vs numpy/numba in Python) is the only intended difference — it is the measured variable.
- **RNG fidelity is statistical, not bit-identical across languages.** Within one language, a fixed seed reproduces a run exactly. Across Rust↔Python, PRNG algorithms differ, so cross-language agreement is proven by matching *distributions* over N replicates (mean, variance, K–S), never bit-for-bit.
- **Go/no-go bar (vs the numba baseline):** ≥5× → go; <2× → thesis dead for this class; 2–5× → marginal, decide from the time-breakdown. Plain-Python number reported as context.
- **No AI attribution** in commits or PR bodies.

## Review Focus

- **Unknown / missing partition tag** — a reaction tagged neither `continuous` nor `stochastic` (or absent from the sidecar) must fail loudly at IR build, not default silently. *(test in Task 3)*
- **Degenerate partitions** — zero stochastic reactions (pure ODE) and zero continuous reactions (pure SSA) must both run without divide-by-zero in reaction selection or an empty-ODE crash. *(tests in Task 6)*
- **Multiple / first-step threshold crossings** — when `dt` is large relative to the event rate, the integrated-propensity crossing must be detected and resolved to the crossing time, not skipped or double-counted within one step. *(test in Task 6)*
- **Negative species from a jump** — a stochastic jump that would drive a species count below zero (large-`dt` or fp slop firing a near-zero-propensity reaction) must be guarded/rejected, not emit nonphysical state. *(test in Task 6)*
- **Unsupported rate-law op** — a MathML/AST construct outside the supported op set must fail loudly in the IR builder, not emit a silently-wrong AST. *(test in Task 2 / Task 3)*

---

### Task 1: Repo scaffold + commit spec

**Files:**
- Create: `~/code/viva-hybrid/` (new git repo), `Cargo.toml` (workspace), `crates/viva-hybrid/Cargo.toml`, `crates/viva-hybrid/src/lib.rs` (empty), `python/pyproject.toml`, `python/viva_hybrid_ref/__init__.py`, `.gitignore`, `README.md`
- Move in: this plan → `docs/superpowers/plans/2026-11-05-viva-hybrid.md`; design → `docs/superpowers/specs/2026-11-05-viva-hybrid-design.md`

**Interfaces:**
- Produces: a buildable empty Rust crate `viva-hybrid`; an importable `viva_hybrid_ref` Python package.

- [ ] **Step 1:** `git init ~/code/viva-hybrid`; create the Cargo workspace (`members = ["crates/viva-hybrid"]`) with crate deps `serde`, `serde_json`, `rand`, `rand_chacha` (pin exact versions ≥2 weeks old); create the Python package with deps `libsbml`, `numpy`, `scipy`, `numba`, `pyyaml`, `pytest`; `.gitignore` (`/target`, `__pycache__`, `*.pyc`, `.venv`). README states the trial's purpose + the go/no-go bar.
- [ ] **Step 2:** Copy the design doc and this plan to the paths above.
- [ ] **Step 3:** Run `cargo build` — Expected: PASS (empty lib). Create `python/.venv`, install deps, `python -c "import viva_hybrid_ref"` — Expected: no error.
- [ ] **Step 4:** Commit. `git add -A && git commit -m "scaffold: viva-hybrid trial repo + approved design"`

---

### Task 2: Shared IR types (Rust + Python)

**Files:**
- Create: `crates/viva-hybrid/src/ir.rs`, `python/viva_hybrid_ref/ir.py`
- Test: `crates/viva-hybrid/tests/ir_roundtrip.rs`, `python/tests/test_ir.py`
- Fixture: `tests/fixtures/mini_ir.json` (hand-written: 2 species, 1 param, 2 reactions — one `continuous` mass-action `k*S`, one `stochastic`)

**Interfaces:**
- Produces (Rust): `struct Model { species: Vec<Species>, params: BTreeMap<String,f64>, reactions: Vec<Reaction> }`; `struct Species { id: String, initial: f64 }`; `struct Reaction { id: String, partition: Partition, stoichiometry: BTreeMap<String,i64>, rate: Ast }`; `enum Partition { Continuous, Stochastic }`; `enum Ast { Const(f64), Param(String), Species(String), Add(Box<Ast>,Box<Ast>), Sub(..), Mul(..), Div(..), Pow(..) }` — all `#[derive(Serialize, Deserialize)]`. `Model::from_json(&str) -> Result<Model>`.
- Produces (Python): dataclasses mirroring the above; `load_ir(path) -> Model`, `dump_ir(model) -> str` producing the identical JSON shape.

- [ ] **Step 1:** Write failing Rust test `ir_roundtrip`: deserialize `mini_ir.json` into `Model`, assert species initials, the continuous reaction's `rate` is `Mul(Param("k"), Species("S"))`, partitions parsed; re-serialize and assert it deserializes identically.
- [ ] **Step 2:** Run `cargo test --test ir_roundtrip` — Expected: FAIL (no `ir` module).
- [ ] **Step 3:** Implement `ir.rs` with the types + serde derives + `from_json`. `partition` serializes as `"continuous"`/`"stochastic"` (serde rename). An unknown partition string deserializes to an error (covers Review Focus: bad tag).
- [ ] **Step 4:** Run `cargo test --test ir_roundtrip` — Expected: PASS.
- [ ] **Step 5:** Write failing Python test `test_ir`: `load_ir("tests/fixtures/mini_ir.json")`, assert same fields; `dump_ir` round-trips to equal dict.
- [ ] **Step 6:** Run `pytest python/tests/test_ir.py` — Expected: FAIL.
- [ ] **Step 7:** Implement `ir.py` dataclasses + `load_ir`/`dump_ir` (json). Unknown op string / unknown partition raises `ValueError`.
- [ ] **Step 8:** Run `pytest python/tests/test_ir.py` — Expected: PASS.
- [ ] **Step 9:** Commit.

---

### Task 3: SBML + partition → IR (Python/libsbml)

**Files:**
- Create: `python/viva_hybrid_ref/sbml_to_ir.py`
- Test: `python/tests/test_sbml_to_ir.py`
- Fixture: `tests/fixtures/mini.xml` (SBML L3 with the 2 reactions of `mini_ir.json`), `tests/fixtures/mini.partition.yaml` (`R1: continuous`, `R2: stochastic`)

**Interfaces:**
- Consumes: Task 2 Python IR dataclasses + `dump_ir`.
- Produces: `sbml_to_ir(sbml_path, partition_path) -> Model`; CLI `python -m viva_hybrid_ref.sbml_to_ir <sbml> <partition> -o <ir.json>`.

- [ ] **Step 1:** Write failing test `test_sbml_to_ir`: convert `mini.xml` + `mini.partition.yaml`, assert the resulting `Model` equals the `mini_ir.json` fixture (species initials from SBML initial amounts, params from SBML parameters, stoichiometry signs from reactants/products, rate AST parsed from each reaction's kinetic-law MathML, partition from the sidecar).
- [ ] **Step 2:** Add tests for Review Focus: (a) a reaction absent from the partition sidecar → `ValueError`; (b) a kinetic law using an unsupported MathML operator → `ValueError` naming the op.
- [ ] **Step 3:** Run `pytest python/tests/test_sbml_to_ir.py` — Expected: FAIL.
- [ ] **Step 4:** Implement `sbml_to_ir.py`: walk libsbml model; translate MathML AST nodes to the IR `Ast` for the supported op set (`+ - * / ^`, numeric constants, parameter and species references); map SBML reactant/product stoichiometry to signed integers; attach partition from the sidecar; raise on missing tag or unsupported op.
- [ ] **Step 5:** Run `pytest python/tests/test_sbml_to_ir.py` — Expected: PASS.
- [ ] **Step 6:** Commit.

---

### Task 4: Rate-law compiler + trait (Rust)

**Files:**
- Create: `crates/viva-hybrid/src/ratelaw.rs`
- Test: `crates/viva-hybrid/tests/ratelaw.rs`

**Interfaces:**
- Consumes: `Ast`, `Model` (Task 2).
- Produces: `trait RateLaw { fn eval(&self, state: &[f64]) -> f64 }`; `fn compile(ast: &Ast, species_index: &BTreeMap<String,usize>, params: &BTreeMap<String,f64>) -> Box<dyn RateLaw>`. `state` is species amounts indexed by `species_index`. The closure-based impl folds params to constants at compile time. (The trait is the codegen seam.)

- [ ] **Step 1:** Write failing test: compile `Mul(Param("k"), Species("S"))` with `k=0.5`, `S` at index 0; `eval(&[10.0])` → `5.0`. Add a `Pow`/`Div` case to exercise the op set.
- [ ] **Step 2:** Run `cargo test --test ratelaw` — Expected: FAIL.
- [ ] **Step 3:** Implement `compile` recursively into a boxed closure (or a small bytecode `Vec<Op>` + `eval` loop — implementer's choice behind the trait). Params resolved to `Const` at compile time; species resolved to index reads.
- [ ] **Step 4:** Run `cargo test --test ratelaw` — Expected: PASS.
- [ ] **Step 5:** Commit.

---

### Task 5: Fixed-step RK4 integrator (Rust)

**Files:**
- Create: `crates/viva-hybrid/src/ode.rs`
- Test: `crates/viva-hybrid/tests/ode.rs`

**Interfaces:**
- Produces: `fn rk4_step(deriv: &dyn Fn(&[f64], &mut [f64]), y: &[f64], dt: f64, out: &mut [f64])` — classic 4th-order step; `deriv(y, dydt)` writes the derivative in place. Operates on the augmented state vector (species + 1 trailing integrated-propensity component).

- [ ] **Step 1:** Write failing test: integrate `dy/dt = -k y` (k=1, y0=1) for t=1 with dt=1e-3 via repeated `rk4_step`; assert `|y - e^{-1}| < 1e-6`.
- [ ] **Step 2:** Run `cargo test --test ode` — Expected: FAIL.
- [ ] **Step 3:** Implement classic RK4.
- [ ] **Step 4:** Run `cargo test --test ode` — Expected: PASS.
- [ ] **Step 5:** Commit.

---

### Task 6: Haseltine–Rawlings hybrid simulator (Rust)

**Files:**
- Create: `crates/viva-hybrid/src/hybrid.rs`, `crates/viva-hybrid/src/rng.rs`
- Test: `crates/viva-hybrid/tests/hybrid.rs`

**Interfaces:**
- Consumes: `Model`, `RateLaw`/`compile` (Task 4), `rk4_step` (Task 5).
- Produces: `struct SimConfig { dt: f64, t_end: f64, sample_times: Vec<f64>, seed: u64 }`; `fn simulate(model: &Model, cfg: &SimConfig) -> Trajectory` where `Trajectory { times: Vec<f64>, species: Vec<Vec<f64>> }` (species-major, sampled at `sample_times`). `rng.rs`: `ChaCha20`-backed seedable uniform `next_uniform() -> f64`.
- Algorithm (pins the method so both languages match): augmented state `[x; g]`. Deriv: for each `continuous` reaction add `nu * rate(x)` to the species components; the trailing `g` component's deriv is the sum of `stochastic` reaction propensities. Draw threshold `T = -ln(u)`. Integrate with `rk4_step(dt)`; after each step, if `g >= T`, estimate crossing fraction by linear interpolation on `g` over the step, re-integrate from the step start by one RK4 step of the fractional `dt` to land at the crossing, select the firing stochastic reaction with a second uniform proportional to its propensity at the crossing, apply its integer stoichiometry, reset `g=0`, redraw `T`. Guard: if a jump would make any species < 0, reject the jump and continue (Review Focus). Record species at each `sample_time` by interpolation-free nearest-step capture (dt divides sample spacing).

- [ ] **Step 1:** Write failing test `pure_continuous`: a model with only a `continuous` decay reaction reproduces the ODE analytic within 1e-5 (no stochastic component; `g` stays 0, no events).
- [ ] **Step 2:** Write failing test `pure_stochastic_birth_death`: only `stochastic` birth (rate `k`) + death (rate `g*X`); over 2000 seeds the mean of `X` at steady state is within 3 s.e. of `k/g`.
- [ ] **Step 3:** Write failing test `determinism`: same seed → identical `Trajectory` (bit-for-bit).
- [ ] **Step 4:** Write failing tests for Review Focus: `big_dt_single_event` (dt large vs rate fires exactly one event per crossing, `g` resets, no double count); `no_negative_species` (a death reaction at `X=0` never produces `X<0`).
- [ ] **Step 5:** Run `cargo test --test hybrid` — Expected: FAIL (no `hybrid`/`rng`).
- [ ] **Step 6:** Implement `rng.rs` then `hybrid.rs` per the algorithm above.
- [ ] **Step 7:** Run `cargo test --test hybrid` — Expected: PASS.
- [ ] **Step 8:** Commit.

---

### Task 7: Python reference — plain hybrid + exact SSA

**Files:**
- Create: `python/viva_hybrid_ref/hybrid_plain.py`, `python/viva_hybrid_ref/exact_ssa.py`
- Test: `python/tests/test_hybrid_plain.py`, `python/tests/test_exact_ssa.py`

**Interfaces:**
- Consumes: Task 2 Python `Model`.
- Produces: `simulate_plain(model, dt, t_end, sample_times, seed) -> Trajectory` — the *same* algorithm as Task 6 (augmented RK4 + integrated-propensity events), plain numpy/Python. `gillespie(model, t_end, sample_times, seed) -> Trajectory` — exact direct-method SSA treating *all* reactions as stochastic (ground truth).

- [ ] **Step 1:** Write failing tests: `test_hybrid_plain` determinism (same seed → identical trajectory) + the pure-continuous analytic match; `test_exact_ssa` birth-death mean within 3 s.e. of `k/g`.
- [ ] **Step 2:** Run the two test files — Expected: FAIL.
- [ ] **Step 3:** Implement `hybrid_plain.py` (mirror the Task 6 algorithm exactly, same draw order) and `exact_ssa.py` (direct method).
- [ ] **Step 4:** Run the two test files — Expected: PASS.
- [ ] **Step 5:** Commit.

---

### Task 8: Python reference — numba hybrid

**Files:**
- Create: `python/viva_hybrid_ref/hybrid_numba.py`
- Test: `python/tests/test_hybrid_numba.py`

**Interfaces:**
- Consumes: Task 2 `Model` (lowered to numba-friendly arrays: stoichiometry matrix, per-reaction partition mask, a compiled propensity kernel).
- Produces: `simulate_numba(model, dt, t_end, sample_times, seed) -> Trajectory` — same algorithm, `@njit` hot loop. Rate laws lowered to a numba-compatible evaluator (array of op-codes or a generated kernel) since arbitrary Python closures aren't `njit`-able.

- [ ] **Step 1:** Write failing test: on the birth-death model, `simulate_numba` and `simulate_plain` agree *statistically* over 1000 seeds (mean/variance within tolerance) — not bit-identical (different RNG path under njit).
- [ ] **Step 2:** Run — Expected: FAIL.
- [ ] **Step 3:** Implement `hybrid_numba.py`: lower the IR to arrays + an op-code rate evaluator, `@njit` the event loop, numba-compatible RNG seeded per replicate.
- [ ] **Step 4:** Run — Expected: PASS.
- [ ] **Step 5:** Commit.

---

### Task 9: Benchmark model (SBML + partition)

**Files:**
- Create: `models/burst_gene_expression.xml`, `models/burst_gene_expression.partition.yaml`, `models/SOURCE.md` (citation)
- Test: `python/tests/test_benchmark_model.py`

**Interfaces:**
- Produces: a genuine hybrid benchmark — stochastic promoter on/off + bursty transcription (slow, discrete) feeding continuous/faster protein translation + decay. Partition tags the promoter/transcription reactions `stochastic` and the protein dynamics `continuous`. Rate constants from a cited source (builder picks; `SOURCE.md` records it). The event load must be substantial enough to land the speedup in the decision-relevant range (per anti-goal: not ~100% overhead, not stiff-ODE-dominated).

- [ ] **Step 1:** Write failing test: `sbml_to_ir` converts the benchmark without error; the IR has ≥1 continuous and ≥1 stochastic reaction; initial state and parameters match `SOURCE.md`.
- [ ] **Step 2:** Run — Expected: FAIL (files absent).
- [ ] **Step 3:** Author the SBML + partition + `SOURCE.md`.
- [ ] **Step 4:** Run — Expected: PASS.
- [ ] **Step 5:** Commit.

---

### Task 10: Fidelity gate (Rust ↔ Python ↔ exact-SSA)

**Files:**
- Create: `python/viva_hybrid_ref/compare.py`, a thin Rust CLI `crates/viva-hybrid/src/bin/run_ir.rs` (loads IR + config JSON, writes replicate trajectories to JSON/CSV for the comparator)
- Test: `python/tests/test_fidelity.py`

**Interfaces:**
- Consumes: Task 6 (`run_ir` bin), Task 7 (`simulate_plain`, `gillespie`), Task 9 (benchmark).
- Produces: `compare_distributions(traj_sets, species, times, tol) -> Report` (per-species mean/variance deltas + two-sample K–S p-values); `run_all(model, cfg, n_replicates) -> {rust, plain, exact}` trajectory sets.

- [ ] **Step 1:** Write failing test `test_fidelity` on the benchmark, N replicates: (a) **Rust-hybrid vs plain-Python-hybrid** agree (mean/variance within tol, K–S not rejected at α); (b) **hybrid vs exact-SSA** agree for the key species (validates the partition itself). Tolerances and N recorded in the test.
- [ ] **Step 2:** Run — Expected: FAIL.
- [ ] **Step 3:** Implement `run_ir.rs` (invoke `simulate` over seeds, emit trajectories) and `compare.py`.
- [ ] **Step 4:** Run `pytest python/tests/test_fidelity.py` — Expected: PASS. **This is the gate: no timing number is meaningful until this passes.**
- [ ] **Step 5:** Commit.

---

### Task 11: Timing harness + trial report

**Files:**
- Create: `python/viva_hybrid_ref/bench.py`, `docs/results/trial-report.md` (generated)
- Test: `python/tests/test_bench_smoke.py`

**Interfaces:**
- Consumes: Task 6 (`run_ir` bin), Task 7 (`simulate_plain`), Task 8 (`simulate_numba`), Task 9 (benchmark).
- Produces: `bench(model, cfg, repeats) -> Timings` (wall-clock for Rust / plain / numba, best-of-k, warm numba); `report(timings, phase_breakdown) -> markdown` emitting the speedup bracket (Rust vs plain, **Rust vs numba**), the per-phase split (ODE RHS / propensity eval / event bookkeeping, instrumented identically or via sampling so instrumentation doesn't skew the overhead-dominated loop), and the go/no-go verdict against ≥5× / <2×.

- [ ] **Step 1:** Write failing smoke test: `bench` on a short config returns positive finite timings for all three engines and a report string containing both ratios and the verdict line.
- [ ] **Step 2:** Run — Expected: FAIL.
- [ ] **Step 3:** Implement `bench.py` (exclude numba JIT-warmup from the timed region; best-of-k; identical total work across engines) and the report generator.
- [ ] **Step 4:** Run `pytest python/tests/test_bench_smoke.py` — Expected: PASS. Then run the real benchmark and write `docs/results/trial-report.md` with actual numbers + verdict.
- [ ] **Step 5:** Commit.

---

## Self-review notes

- **Spec coverage:** SBML+partition load → T3; shared IR both sides → T2/T3; Haseltine–Rawlings hybrid → T6/T7; trait/codegen-ready → T4; two Python baselines → T7/T8; exact-SSA cross-check → T7/T10; statistical fidelity → T10; speedup bracket + ≥5×/<2× verdict → T11; benchmark not rigged → T9. Covered.
- **Lock-step vs cross-language RNG:** resolved in Global Constraints — same algorithm/dt/draw-order; fidelity is statistical across languages, deterministic within one. This refines design decision #2 (the design's "same RNG draw order" meant work-fairness, not bit-identical values across Rust/Python). Builder's clarification, flagged here.
- **Review Focus:** all five pinned to owning-task tests (T2, T3, T6).
