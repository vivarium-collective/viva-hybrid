"""Fidelity gate: Rust hybrid == plain-Python hybrid (== exact SSA for the partition).

Run config: dt=0.1 min, t_end=300 min, samples at 20/50/100/200/300 min
(mRNA relaxes in ~10 min, protein tau ~50 min => t>=100 is quasi-stationary),
N=1000 replicates per simulator.

Tolerances are standard-error based, not tuned:
  * mean:     |z| <= 4 where z = (mean_a-mean_b)/sqrt(var_a/N+var_b/N)
  * variance: |z| <= 4 with SE from the empirical 4th moment (bursty, heavy-tailed)
  * K-S:      reject only when p < 0.001
  With ~10 checks per statistic per comparison, a correct implementation has a
  family-wise false-fail rate of order 10 * 6e-5 (|z|>4) ~ 1e-3 for mean/var and
  ~1e-2 for K-S; seeds are fixed, so the result is deterministic.
Exact-SSA specifics: mRNA is discrete in both -> mean, variance and K-S.
Protein is real-valued in the hybrid but integer in the SSA, so K-S on protein
is skipped (continuous vs discrete); protein mean and variance are compared.
(The hybrid drops translation shot noise, ~mean=500 of variance ~7e4, i.e. <1%,
far below the ~4% variance SE at N=1000, so it is not detectable here.)
"""
from pathlib import Path

import pytest

from viva_hybrid_ref.compare import RUN_IR, compare_distributions, run_all
from viva_hybrid_ref.sbml_to_ir import sbml_to_ir

M = Path(__file__).resolve().parents[2] / "models"
N = 1000
CFG = dict(dt=0.1, t_end=300.0, sample_times=[20.0, 50.0, 100.0, 200.0, 300.0])
TIMES = CFG["sample_times"]
TOL = dict(z_mean=4.0, z_var=4.0, ks_alpha=0.001)
KEY = ["mRNA", "protein"]

pytestmark = pytest.mark.skipif(
    not RUN_IR.exists(), reason="build first: cargo build --release --bin run_ir")


@pytest.fixture(scope="module")
def runs():
    model = sbml_to_ir(M / "burst_gene_expression.xml",
                       M / "burst_gene_expression.partition.yaml")
    ids = [s.id for s in model.species]
    return ids, run_all(model, CFG, N)


def test_rust_matches_plain_hybrid(runs, capsys):
    ids, r = runs
    rep = compare_distributions(r["rust"], r["plain"], KEY, TIMES, TOL, ids)
    with capsys.disabled():
        print("\nRust vs plain-Python hybrid\n" + rep.format())
    assert rep.passed, rep.format()


def test_hybrid_matches_exact_ssa(runs, capsys):
    ids, r = runs
    rep = compare_distributions(r["rust"], r["exact"], KEY, TIMES, TOL, ids,
                                check_ks=["mRNA"])
    with capsys.disabled():
        print("\nRust hybrid vs exact SSA\n" + rep.format())
    assert rep.passed, rep.format()
