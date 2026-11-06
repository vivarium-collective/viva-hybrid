import math

import pytest

from viva_hybrid_ref import exact_ssa
from viva_hybrid_ref.exact_ssa import gillespie
from viva_hybrid_ref.ir import Model, dump_ir, load_ir, model_to_dict
from viva_hybrid_ref.model_generator import generate_model, write_ir

KW = dict(event_density=50.0, stochastic_fraction=0.5, system_size=8)


def _stoch_propensity(m):
    from viva_hybrid_ref._common import compile_rate
    idx = {s.id: i for i, s in enumerate(m.species)}
    x = [s.initial for s in m.species]
    return sum(compile_rate(r.rate, m.params, idx)(x)
               for r in m.reactions if r.partition == "stochastic")


def test_valid_ir_roundtrip(tmp_path):
    m = generate_model(**KW)
    assert isinstance(m, Model)
    assert model_to_dict(load_ir(_w(tmp_path, m))) == model_to_dict(m)
    assert load_ir(_w(tmp_path, m)).reactions[0].rate == m.reactions[0].rate
    assert dump_ir(m) == (tmp_path / "m.json").read_text()


def _w(tmp_path, m):
    p = tmp_path / "m.json"
    write_ir(m, p)
    return p


@pytest.mark.parametrize("s", [1, 2, 5, 30])
def test_size_and_connectivity(s):
    m = generate_model(event_density=10.0, stochastic_fraction=0.5, system_size=s)
    assert len(m.species) == s
    touched = {k for r in m.reactions for k in r.stoichiometry}
    assert touched == {sp.id for sp in m.species}
    if s > 1:  # each species shares a reaction with another species
        nbr = {sp.id: set() for sp in m.species}
        for r in m.reactions:
            ks = list(r.stoichiometry)
            for a in ks:
                nbr[a].update(k for k in ks if k != a)
        assert all(nbr.values())
        seen, todo = set(), [m.species[0].id]
        while todo:
            a = todo.pop()
            if a not in seen:
                seen.add(a)
                todo.extend(nbr[a])
        assert len(seen) == s


@pytest.mark.parametrize("f", [0.0, 0.1, 0.33, 0.5, 0.9, 1.0])
def test_stochastic_fraction(f):
    m = generate_model(event_density=10.0, stochastic_fraction=f, system_size=10)
    n = len(m.reactions)
    k = sum(r.partition == "stochastic" for r in m.reactions)
    assert abs(k / n - f) <= 1.0 / n
    assert {r.partition for r in m.reactions} <= {"continuous", "stochastic"}


def test_event_density_sets_initial_propensity():
    for ed in (5.0, 500.0):
        m = generate_model(event_density=ed, stochastic_fraction=0.4, system_size=6)
        assert _stoch_propensity(m) == pytest.approx(ed, rel=1e-9)


def test_timescale_sep():
    m = generate_model(event_density=10.0, stochastic_fraction=0.5, system_size=20,
                       timescale_sep=100.0)
    def mean(part):
        v = [m.params[f"k_{r.id}"]
             for r in m.reactions if r.partition == part and r.id.startswith("deg")]
        return sum(v) / len(v)
    assert 30 < mean("continuous") / mean("stochastic") < 300


def test_dynamics_nontrivial():
    m = generate_model(**KW)
    ts = [0.0, 0.5, 1.0, 2.0]
    tr = gillespie(m, 2.0, ts, 1)
    flat = [v for col in tr.species for v in col]
    assert all(math.isfinite(v) and v >= 0 for v in flat)
    assert any(col[-1] != col[0] for col in tr.species)


def test_determinism_and_seed_sensitivity():
    assert generate_model(**KW, seed=3) == generate_model(**KW, seed=3)
    assert generate_model(**KW, seed=3) != generate_model(**KW, seed=4)


def test_event_rate_increases_with_density(monkeypatch):
    calls = []
    real = exact_ssa.next_uniform
    monkeypatch.setattr(exact_ssa, "next_uniform", lambda rng: (calls.append(1), real(rng))[1])

    def events(ed):
        m = generate_model(event_density=ed, stochastic_fraction=0.5, system_size=6)
        m.reactions = [r for r in m.reactions if r.partition == "stochastic"]
        calls.clear()
        gillespie(m, 1.0, [1.0], 0)
        return len(calls) // 2

    lo, hi = events(20.0), events(400.0)
    assert hi > 3 * lo > 0


@pytest.mark.parametrize("kw", [
    dict(system_size=0), dict(system_size=-1), dict(system_size=2.5),
    dict(stochastic_fraction=-0.1), dict(stochastic_fraction=1.1),
    dict(event_density=0.0), dict(event_density=-1.0),
    dict(timescale_sep=0.0)])
def test_invalid_params_raise(kw):
    args = dict(event_density=10.0, stochastic_fraction=0.5, system_size=4)
    args.update(kw)
    with pytest.raises(ValueError):
        generate_model(**args)
