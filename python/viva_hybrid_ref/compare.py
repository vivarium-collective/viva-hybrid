"""Fidelity comparison: Rust hybrid vs plain-Python hybrid vs exact Gillespie SSA."""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

from ._common import Trajectory
from .exact_ssa import gillespie
from .hybrid_plain import simulate_plain
from .ir import dump_ir

REPO = Path(__file__).resolve().parents[2]
RUN_IR = REPO / "target" / "release" / "run_ir"


def run_rust(model, cfg, seeds, binary=RUN_IR) -> list:
    """Run the Rust `run_ir` binary; return a list of Trajectory (one per seed).

    Output schema of the binary: {"times", "species_ids", "seeds",
    "replicates": [replicate][species][time]}.
    """
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "ir.json").write_text(dump_ir(model))
        (d / "cfg.json").write_text(json.dumps(
            {"dt": cfg["dt"], "t_end": cfg["t_end"],
             "sample_times": list(cfg["sample_times"]), "seeds": list(seeds)}))
        subprocess.run([str(binary), str(d / "ir.json"), str(d / "cfg.json"),
                        str(d / "out.json")], check=True)
        out = json.loads((d / "out.json").read_text())
    assert out["species_ids"] == [s.id for s in model.species]
    return [Trajectory(times=list(out["times"]), species=rep)
            for rep in out["replicates"]]


def run_all(model, cfg, n_replicates, seed_start=0, binary=RUN_IR) -> dict:
    """Replicate sets for the three simulators. Each uses seeds
    seed_start..seed_start+n (hybrids) with the exact SSA on a disjoint block."""
    st = list(cfg["sample_times"])
    seeds = list(range(seed_start, seed_start + n_replicates))
    return {
        "rust": run_rust(model, cfg, seeds, binary),
        "plain": [simulate_plain(model, cfg["dt"], cfg["t_end"], st, s) for s in seeds],
        "exact": [gillespie(model, cfg["t_end"], st, s + 10_000_000) for s in seeds],
    }


@dataclass
class Row:
    species: str
    time: float
    mean_a: float
    mean_b: float
    var_a: float
    var_b: float
    z_mean: float      # (mean_a-mean_b)/SE
    z_var: float       # (var_a-var_b)/SE (SE from sample 4th moments)
    ks_p: float        # nan when K-S was not requested for this species
    ok: bool
    why: str = ""


@dataclass
class Report:
    rows: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.ok for r in self.rows)

    def format(self) -> str:
        h = (f"{'species':8} {'t':>6} {'mean_a':>10} {'mean_b':>10} {'z_mean':>7} "
             f"{'var_a':>11} {'var_b':>11} {'z_var':>7} {'ks_p':>7}  ok")
        lines = [h]
        for r in self.rows:
            lines.append(f"{r.species:8} {r.time:6.0f} {r.mean_a:10.3f} {r.mean_b:10.3f} "
                         f"{r.z_mean:7.2f} {r.var_a:11.2f} {r.var_b:11.2f} {r.z_var:7.2f} "
                         f"{r.ks_p:7.3f}  {'ok' if r.ok else 'FAIL ' + r.why}")
        return "\n".join(lines)


def _col(trajs, sp_idx, k):
    return np.array([t.species[sp_idx][k] for t in trajs], dtype=float)


def _var_se(x):
    """SE of the sample variance, using the empirical 4th central moment
    (valid for non-Gaussian, heavy-tailed bursty counts): sqrt((m4 - s^4)/n)."""
    n = len(x)
    c = x - x.mean()
    m4, s2 = np.mean(c ** 4), np.mean(c ** 2)
    return np.sqrt(max(m4 - s2 ** 2, 0.0) / n)


def compare_distributions(set_a, set_b, species, times, tol, species_ids,
                          check_var=None, check_ks=None) -> Report:
    """Compare replicate sets per species and time.

    species: names to compare; times: sample times (subset of the trajectories'
    times); species_ids: model species order (to index trajectories).
    tol: {"z_mean": max |z| for mean diff, "z_var": max |z| for variance diff,
          "ks_alpha": reject when K-S p < alpha}. z = diff / standard error.
    check_var / check_ks: optional collections of species names for which the
    variance / K-S check is applied (default: all). Others are still reported.
    """
    check_var = set(species if check_var is None else check_var)
    check_ks = set(species if check_ks is None else check_ks)
    grid = list(set_a[0].times)
    rep = Report()
    for name in species:
        i = species_ids.index(name)
        for t in times:
            k = min(range(len(grid)), key=lambda j: abs(grid[j] - t))
            a, b = _col(set_a, i, k), _col(set_b, i, k)
            va, vb = a.var(ddof=1), b.var(ddof=1)
            se_m = np.sqrt(va / len(a) + vb / len(b))
            se_v = np.hypot(_var_se(a), _var_se(b))
            zm = (a.mean() - b.mean()) / se_m if se_m > 0 else 0.0
            zv = (va - vb) / se_v if se_v > 0 else 0.0
            p = float(stats.ks_2samp(a, b).pvalue) if name in check_ks else float("nan")
            why = []
            if abs(zm) > tol["z_mean"]:
                why.append("mean")
            if name in check_var and abs(zv) > tol["z_var"]:
                why.append("var")
            if name in check_ks and p < tol["ks_alpha"]:
                why.append("ks")
            rep.rows.append(Row(name, t, a.mean(), b.mean(), va, vb, zm, zv, p,
                                not why, ",".join(why)))
    return rep
