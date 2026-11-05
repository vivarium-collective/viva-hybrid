from pathlib import Path

from viva_hybrid_ref.exact_ssa import gillespie
from viva_hybrid_ref.sbml_to_ir import sbml_to_ir

M = Path(__file__).resolve().parents[2] / "models"
PARAMS = dict(k_on=0.05, k_off=0.15, k_tx=2.0, k_dm=0.1, k_tl=2.0, k_dp=0.02)


def _ir():
    return sbml_to_ir(M / "burst_gene_expression.xml",
                      M / "burst_gene_expression.partition.yaml")


def test_converts_and_is_hybrid():
    parts = {r.id: r.partition for r in _ir().reactions}
    assert parts == {"R_act": "stochastic", "R_deact": "stochastic",
                     "R_tx": "stochastic", "R_dm": "stochastic",
                     "R_tl": "continuous", "R_dp": "continuous"}


def test_state_and_params_match_source_md():
    m = _ir()
    assert {s.id: s.initial for s in m.species} == {
        "G_off": 1.0, "G_on": 0.0, "mRNA": 0.0, "protein": 0.0}
    assert m.params == PARAMS
    src = (M / "SOURCE.md").read_text()
    for k, v in PARAMS.items():
        assert f"| {k}" in src and f"| {v} " in src.split(f"| {k}")[1].split("\n")[0]


def test_model_produces_protein():
    tr = gillespie(_ir(), 300.0, [300.0], 1)
    assert tr.species[3][-1] > 0
