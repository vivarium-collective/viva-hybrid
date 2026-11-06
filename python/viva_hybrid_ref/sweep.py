"""Regime-map sweep harness: speedup (Rust vs numba/plain) + fidelity (vs exact SSA).

Performance reuses `bench.bench` unchanged (numba warmed, lowering outside the timed
region, Rust in-process `sim_seconds`, best-of-k, identical seeds/workload).
Fidelity reuses `compare.run_all` / `compare_distributions` with the same SE-based
tolerances as the fidelity gate (|z|<=4 mean & variance, K-S alpha=0.001).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import tempfile
import warnings
from pathlib import Path

from .bench import bench
from .compare import REPO, RUN_IR, compare_distributions, run_all
from .model_generator import generate_model, write_ir

AXES = ("event_density", "stochastic_fraction", "system_size", "timescale_sep", "dt")
TOL = dict(z_mean=4.0, z_var=4.0, ks_alpha=0.001)   # same as test_fidelity.py


def _ensure_binary():
    if not RUN_IR.exists():
        subprocess.run(["cargo", "build", "--release", "--bin", "run_ir"], cwd=REPO, check=True)


def _fidelity(model, cfg, n_reps, seed):
    """Hybrid (Rust) vs exact SSA at all sample times: mean, variance AND K-S on ALL
    species, continuous-partition ones included. Dropping a continuous species'
    intrinsic variance IS the partition's approximation error, so it must count
    against fidelity. Tolerances are the gate's (|z|<=4, K-S alpha=0.001)."""
    ids = [s.id for s in model.species]
    r = run_all(model, cfg, n_reps, seed_start=seed * 100_000)
    with warnings.catch_warnings():   # scipy notes K-S falling back to the asymptotic p (ties)
        warnings.filterwarnings("ignore", message="ks_2samp: Exact calculation")
        rep = compare_distributions(r["rust"], r["exact"], ids, list(cfg["sample_times"]), TOL, ids)
    zs = [max(abs(x.z_mean), abs(x.z_var)) for x in rep.rows]
    ks = [x.ks_p for x in rep.rows if not math.isnan(x.ks_p)]
    fails = sorted({f"{x.species}@{x.time:g}:{x.why}" for x in rep.rows if not x.ok})
    return dict(fidelity_ok=bool(rep.passed), worst_abs_z=float(max(zs)),
                min_ks_p=float(min(ks)) if ks else None,
                n_fail_rows=sum(not x.ok for x in rep.rows), n_rows=len(rep.rows),
                fidelity_fail=fails)


def sweep_point(*, event_density, stochastic_fraction, system_size, timescale_sep, dt, t_end,
                sample_times, n_time_reps, n_time_repeats, n_fidelity_reps, seed=0) -> dict:
    axes = dict(event_density=event_density, stochastic_fraction=stochastic_fraction,
                system_size=system_size, timescale_sep=timescale_sep, dt=dt)
    out = dict(axes, t_end=t_end, seed=seed, n_time_reps=n_time_reps,
               n_time_repeats=n_time_repeats, n_fidelity_reps=n_fidelity_reps)
    _ensure_binary()
    model = generate_model(event_density=event_density, stochastic_fraction=stochastic_fraction,
                           system_size=int(system_size), timescale_sep=timescale_sep, seed=seed)
    with tempfile.TemporaryDirectory() as d:
        write_ir(model, Path(d) / "ir.json")   # the IR the engines consume
    cfg = dict(dt=dt, t_end=t_end, sample_times=list(sample_times))
    t = bench(model, cfg, repeats=n_time_repeats, replicates=n_time_reps)
    out.update(rust_vs_numba=t.vs_numba, rust_vs_plain=t.vs_plain,
               rust_ms=t.rust_s * 1e3, numba_ms=t.numba_s * 1e3, plain_ms=t.plain_s * 1e3,
               steps=t.steps, events=t.events)
    out.update(_fidelity(model, cfg, n_fidelity_reps, seed))
    return out


def run_sweep(axis_x, axis_y, *, fixed, grid, timing, fidelity) -> list[dict]:
    """grid: {axis: [values]} for axis_x/axis_y; fixed: values for the other axes
    (+ t_end, sample_times); timing: {n_time_reps, n_time_repeats}; fidelity: {n_fidelity_reps}.
    A failing point is recorded with an `error` marker, never dropped."""
    assert axis_x in AXES and axis_y in AXES and axis_x != axis_y
    results = []
    for x, y in itertools.product(grid[axis_x], grid[axis_y]):
        kw = {**fixed, axis_x: x, axis_y: y, **timing, **fidelity}
        try:
            results.append(sweep_point(**kw))
        except Exception as e:  # record, keep going, surface loudly
            print(f"ERROR at {axis_x}={x} {axis_y}={y}: {e!r}")
            results.append({**{a: kw[a] for a in AXES}, "error": repr(e)})
    return results


def save_results(results, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=1))
    return p


def summary(results, axis_x, axis_y) -> str:
    L = [f"{axis_x:>14} {axis_y:>14} {'rust/numba':>10} {'rust/plain':>10} {'worst|z|':>8}  fid"]
    for r in results:
        if "error" in r:
            L.append(f"{r[axis_x]:>14g} {r[axis_y]:>14g}  ERROR {r['error']}")
        else:
            L.append(f"{r[axis_x]:>14g} {r[axis_y]:>14g} {r['rust_vs_numba']:10.2f} "
                     f"{r['rust_vs_plain']:10.1f} {r['worst_abs_z']:8.2f}  {'ok' if r['fidelity_ok'] else 'FAIL'}")
    return "\n".join(L)


# Two default maps, both x stochastic_fraction (timescale_sep=10, dt=0.01, t_end=1):
#   density x fraction  (system_size=10)   and   system_size x fraction (density=100).
# Timing: 20 replicate seeds x best-of-3. Fidelity: N=400 hybrid + 400 exact replicates
# (variance is load-bearing, so SE of the variance ~7% of sd; tolerances unchanged).
_COMMON = dict(timescale_sep=10.0, dt=0.01, t_end=1.0, sample_times=[0.25, 0.5, 1.0])
_F = [0.1, 0.25, 0.5, 0.75, 1.0]
MAPS = [
    dict(axes=("event_density", "stochastic_fraction"),
         grid=dict(event_density=[1, 10, 100, 1000], stochastic_fraction=_F),
         fixed=dict(_COMMON, system_size=10)),
    dict(axes=("system_size", "stochastic_fraction"),
         grid=dict(system_size=[3, 10, 30, 100], stochastic_fraction=_F),
         fixed=dict(_COMMON, event_density=100.0)),
]
TIMING = dict(n_time_reps=20, n_time_repeats=3)
FIDELITY = dict(n_fidelity_reps=400)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "reports" / "regime"))
    a = ap.parse_args(argv)
    allres = []
    for mp in MAPS:
        ax, ay = mp["axes"]
        res = run_sweep(ax, ay, fixed=mp["fixed"], grid=mp["grid"], timing=TIMING, fidelity=FIDELITY)
        path = save_results(res, Path(a.out_dir) / f"sweep_{ax}_x_{ay}.json")
        print(summary(res, ax, ay))
        print(f"wrote {path}")
        allres.append(res)
    return allres


if __name__ == "__main__":
    main()
