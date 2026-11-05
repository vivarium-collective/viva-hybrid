"""Timing harness: Rust hybrid vs plain-Python vs numba-warm, plus trial report.

Fair-measurement rules: setup excluded (numba JIT warmed first; Rust time is the
`sim_seconds` the binary measures around only its simulate loops; plain times
only `simulate_plain` calls), identical workload, best-of-k.
"""
from __future__ import annotations

import cProfile
import json
import pstats
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .compare import REPO, RUN_IR
from .hybrid_numba import simulate_numba
from .hybrid_plain import simulate_plain
from .ir import dump_ir


@dataclass
class Timings:
    rust_s: float
    plain_s: float
    numba_s: float
    steps: int       # RK4 step attempts across all replicates (Rust counter)
    events: int      # stochastic events across all replicates (Rust counter)
    replicates: int
    repeats: int
    cfg: dict

    @property
    def vs_plain(self) -> float:
        return self.plain_s / self.rust_s

    @property
    def vs_numba(self) -> float:
        return self.numba_s / self.rust_s


def _rust_raw(model, cfg, seeds, phases=False, binary=RUN_IR) -> dict:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "ir.json").write_text(dump_ir(model))
        (d / "cfg.json").write_text(json.dumps(
            {"dt": cfg["dt"], "t_end": cfg["t_end"],
             "sample_times": list(cfg["sample_times"]), "seeds": list(seeds),
             "phases": phases}))
        subprocess.run([str(binary), str(d / "ir.json"), str(d / "cfg.json"),
                        str(d / "out.json")], check=True)
        return json.loads((d / "out.json").read_text())


def bench(model, cfg, repeats=3, replicates=None, binary=RUN_IR) -> Timings:
    """Best-of-`repeats` wall time of the whole workload per engine.

    cfg: dt, t_end, sample_times, and optionally `replicates` (default 10).
    """
    n = replicates if replicates is not None else cfg.get("replicates", 10)
    seeds = list(range(n))
    st = list(cfg["sample_times"])
    dt, te = cfg["dt"], cfg["t_end"]

    rust, info = [], None
    for _ in range(repeats):
        out = _rust_raw(model, cfg, seeds, binary=binary)
        rust.append(out["sim_seconds"])
        info = out

    simulate_numba(model, dt, te, st, 0)      # JIT warmup, untimed

    def timed(fn):
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            for s in seeds:
                fn(model, dt, te, st, s)
            best = min(best, time.perf_counter() - t0)
        return best

    return Timings(min(rust), timed(simulate_plain), timed(simulate_numba),
                   int(info["steps"]), int(info["events"]), n, repeats, dict(cfg))


def phase_breakdown(model, cfg, replicates=10, binary=RUN_IR) -> dict:
    """Best-effort, approximate phase split. Separate (instrumented) runs only;
    never used for headline timings."""
    seeds = list(range(replicates))
    ph = _rust_raw(model, cfg, seeds, phases=True, binary=binary)["phase_ns"]
    # Python plain via cProfile (inflates absolute time; only shares are used).
    st = list(cfg["sample_times"])
    prof = cProfile.Profile()
    prof.enable()
    for s in seeds[:max(1, replicates // 5)]:
        simulate_plain(model, cfg["dt"], cfg["t_end"], st, s)
    prof.disable()
    stats = pstats.Stats(prof).stats
    cum = {k[2]: v[3] for k, v in stats.items()}
    tot = {k[2]: v[2] for k, v in stats.items()}
    total = cum["simulate_plain"]
    rhs = cum["deriv"]
    rk4_combine = tot["rk4"]
    return {"rust_ns": ph, "python": {
        "total": total, "rhs": rhs, "rk4_combine": rk4_combine,
        "other": total - rhs - rk4_combine}}


def _verdict(r: float) -> str:
    if r >= 5.0:
        return f"GO: Rust is {r:.1f}x faster than numba (>= 5x)."
    if r < 2.0:
        return (f"NO-GO: Rust is only {r:.2f}x vs numba (< 2x); the thesis is dead "
                "for this model class.")
    return (f"MARGINAL: Rust is {r:.2f}x vs numba (2-5x); decide from the phase "
            "breakdown.")


def report(t: Timings, pb: dict | None) -> str:
    c = t.cfg
    L = ["# Trial report: Rust hybrid vs Python baselines", "",
         "## Workload", "",
         f"- Model: bursting gene expression (partitioned, Haseltine-Rawlings hybrid)",
         f"- dt = {c['dt']}, t_end = {c['t_end']}, sample_times = {list(c['sample_times'])}",
         f"- Replicates: {t.replicates} (seeds 0..{t.replicates - 1}); best of {t.repeats} repeats of the whole workload",
         f"- Work (Rust counters): {t.steps:,} RK4 steps, {t.events:,} stochastic events",
         "- Timing scope: numba JIT warmed before timing; Rust time is the in-process "
         "`sim_seconds` of the simulate loops only (no process start, IR parse, or JSON output); "
         "plain/numba time only `simulate_*` calls (IR loading excluded).", "",
         "## Timings (best-of-k)", "",
         "| engine | total ms | us per RK4 step |", "|---|---:|---:|"]
    for name, s in (("Rust", t.rust_s), ("numba (warm)", t.numba_s), ("plain Python", t.plain_s)):
        L.append(f"| {name} | {s * 1e3:.1f} | {s / t.steps * 1e6:.3f} |")
    L += ["", "## Speedup", "",
          f"- Rust vs plain Python: **{t.vs_plain:.1f}x**",
          f"- Rust vs numba: **{t.vs_numba:.2f}x**", ""]
    if pb:
        r = pb["rust_ns"]
        tot = r["total"] or 1
        rhs_c, rhs_s = r["rhs_continuous"], r["rhs_stochastic"]
        rk_other = r["rk4_total"] - rhs_c - rhs_s
        ev = r["event_bookkeeping"]
        rest = tot - r["rk4_total"] - ev
        p = pb["python"]
        L += ["## Phase breakdown (approximate)", "",
              "Method: separate instrumented runs, not the headline runs. Rust: `Instant` "
              "timers inside the loop (config `phases: true`); the timer calls themselves "
              "cost tens of ns each and a whole RK4 step is only ~0.1 us uninstrumented, so "
              "the instrumented Rust run is several times slower than the headline run and "
              "its shares are dominated by timer overhead inside the RHS regions; the Rust "
              "split is indicative at best. Python plain: `cProfile` on a subset of "
              "replicates (function-level; profiler overhead inflates call-heavy regions "
              "such as the rate closures). numba cannot be split without altering its "
              "compiled code and is not broken down. Treat all shares as coarse.", "",
              "| phase | Rust share | plain-Python share |", "|---|---:|---:|",
              f"| ODE RHS: continuous reactions | {rhs_c / tot:.0%} | (in RHS total) |",
              f"| propensity eval: stochastic sum inside RHS | {rhs_s / tot:.0%} | (in RHS total) |",
              f"| ODE RHS total | {(rhs_c + rhs_s) / tot:.0%} | {p['rhs'] / p['total']:.0%} |",
              f"| RK4 combination / step overhead | {rk_other / tot:.0%} | {p['rk4_combine'] / p['total']:.0%} |",
              f"| event bookkeeping (select, apply, RNG, capture) | {ev / tot:.1%} | in remainder |",
              f"| remainder (loop, state copies, capture on accepted steps) | {rest / tot:.0%} | {p['other'] / p['total']:.0%} |",
              ""]
    L += ["## Verdict", "", _verdict(t.vs_numba), ""]
    return "\n".join(L)


def main():
    import platform
    from .sbml_to_ir import sbml_to_ir
    M = REPO / "models"
    model = sbml_to_ir(M / "burst_gene_expression.xml", M / "burst_gene_expression.partition.yaml")
    cfg = dict(dt=0.1, t_end=300.0, sample_times=[20.0, 50.0, 100.0, 200.0, 300.0])
    n = 200
    subprocess.run(["cargo", "build", "--release", "--bin", "run_ir"], cwd=REPO, check=True)
    t = bench(model, cfg, repeats=5, replicates=n)
    pb = phase_breakdown(model, cfg, replicates=n)
    mach = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    out = report(t, pb) + f"\n## Machine\n\n{mach}, {platform.platform()}, Python {platform.python_version()}\n"
    print(out)
    return out


if __name__ == "__main__":
    main()
