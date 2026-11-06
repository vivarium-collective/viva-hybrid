# viva-hybrid regime-map investigation — design

**Date:** 2026-11-05
**Status:** design (user opted out of further questions — "follow your recommendation, go")

## Intent and the why

The trial (`docs/results/trial-report.md`) measured ONE benchmark: the pure-Rust hybrid engine was 46.1× faster than plain Python but only **2.01× vs the honest numba baseline → MARGINAL**, sitting on the ≥5× go / <2× dead line. That is a single point. The strategic question it can't answer alone: **for what *kinds* of models does a pure-Rust hybrid engine actually clear the bar — and where does the hybrid partition stay accurate?** This investigation answers that as a **regime map** over model characteristics, built and rendered through the **vivarium-workbench** so it is a reusable, inspectable Investigation rather than a one-off script.

## Goals

1. **Promote viva-hybrid to a viva workspace** (`/viva-workspace` in-place) so the vivarium-workbench can host it.
2. **Wrap the engines as process-bigraph Steps** (`/viva-expert`), each consuming an IR model + sim config and emitting a trajectory (+ timing for the simulators, ground truth for exact): `RustHybrid`, `PythonHybridPlain`, `PythonHybridNumba`, `ExactSSA`. Reuse the trial's existing engines (do not fork them); the Rust Step bridges the real engine via the `run_ir` release binary (simplest, already exists) unless PyO3 is clearly better.
3. **A parametrized model generator** emitting an IR (and SBML for provenance) for a point in regime space: `(event_density, stochastic_fraction, system_size, timescale_separation, dt)`. Synthetic reaction networks so the axes vary **independently** (the single gene-expression benchmark can't span system size, etc.).
4. **A sweep harness**: over a grid on the chosen axes, run all engines × N replicates; record **performance** (Rust-vs-numba and Rust-vs-plain speedup, reusing the trial's fair-measurement rules — warm numba, pure `sim_seconds` for Rust, best-of-k) and **accuracy** (hybrid-vs-exact-SSA agreement; dt→0 convergence on the secondary slice).
5. **A viva Investigation** (`/viva-investigation`) with member **Studies** (`/viva-study`) — one per map/axis — and **Visualizations** (`/viva-viz`): the regime maps.
6. **Rendered in the workbench** and publishable read-only.

## The map (headline deliverable)

2-D heatmaps over pairs of the spine axes, shaded by **Rust-vs-numba speedup**, with the **≥5× and <2× contours** drawn, and **masked/hatched where the hybrid stops being faithful** to exact SSA. The one-sentence read: *"here is the region where Rust wins AND you can trust the answer."* Plus convergence slices (accuracy vs dt) and a findings write-up generalizing the trial's single point.

## Axes (locked per recommendation)

- **Spine (maps built on pairs):** (1) event density / stochastic load per sim-time, (2) stochastic-vs-continuous reaction fraction, (3) system size (species/reactions).
- **Secondary slices:** (4) timescale separation (mostly accuracy / partition validity), (5) dt (accuracy/convergence + cost).

## Non-goals

- Not codegen/JIT Rust — still the trait seam only.
- Not a cluster-scale sweep by default — local, modest grid with the release binary; the harness is scriptable to the mini/Ray if a denser map is wanted later.
- Not new SBML coverage beyond what the generator needs.
- Not re-deriving the single-benchmark trial — it stands; this generalizes it.
- Not changing the engines' numerics or the lock-step/statistical-fidelity contract.

## Anti-goals (failure even if it "runs")

- A map too coarse to show the contour (too few grid points / too few replicates for a stable ratio).
- **Speedup shown without the accuracy mask** — a fast-but-wrong region presented as a win.
- Engines run on non-identical workloads per regime point (unfair comparison).
- Fabricated/placeholder workbench content — Studies/Composites/runs and the maps must be real (reuses the no-fake-previews / models-are-real-composites conventions).

## Constraints

- Reuse the trial's engines (`hybrid.rs`, `hybrid_plain`, `hybrid_numba`, `gillespie`, `run_ir`, `compare.py`) and the fidelity harness.
- Fair-measurement rules from the trial carry over verbatim (warm numba; Rust pure `sim_seconds`; best-of-k; identical workload).
- vivarium-workbench + process-bigraph are installed here (confirmed: workbench CLI in `~/code/venv`, process-bigraph 1.10.1, viva_superpowers 0.24.0). The workbench must be running for the Study/Investigation steps (`/viva-workbench start`).
- viva_ package-prefix convention for the workspace package.

## The how, as decided

- **Integration-first vertical slice** (de-risk before the sweep): promote the workspace → wrap `RustHybrid` as a Step → run ONE model through it in the workbench and see a trajectory. Only once that holds do we build the generator + sweep + maps.
- Engines as Steps; a Composite = (engine × model at a regime point); a Study = one map/axis sweep; Visualizations = the heatmaps.
- Sweep runs locally with the release Rust binary; replicate counts tuned so the Rust-vs-numba ratio is stable (the trial showed run-to-run noise near 2.0 — size N accordingly).

## What's left to the builder

- Exact grid resolution + replicate counts (tune runtime vs contour clarity; the ratio must be stable enough to place the ≥5×/<2× contours).
- The synthetic generator's reaction-network topology (how size/density/fraction/timescale are parametrized into reactions + partition).
- Rust Step bridge mechanism: `run_ir` subprocess (default, exists) vs PyO3.
- Workbench Study/Composite/Investigation schema specifics (driven by `/viva-study`, `/viva-investigation`).
- Heatmap viz implementation (within `/viva-viz`): contour drawing + accuracy mask.
- Whether the sweep is later scaled to the mini/Ray (left as an option).
