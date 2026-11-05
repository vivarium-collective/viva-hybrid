import math
from viva_hybrid_ref.ir import Model, Species, Reaction, Ast
from viva_hybrid_ref.hybrid_plain import simulate_plain


def _decay():
    rate = Ast("mul", (Ast("param", "k"), Ast("species", "X")))
    return Model([Species("X", 100.0)], {"k": 0.5},
                 [Reaction("d", "continuous", {"X": -1}, rate)])


def _bd():
    return Model([Species("X", 0.0)], {"k": 10.0, "d": 1.0}, [
        Reaction("b", "stochastic", {"X": 1}, Ast("param", "k")),
        Reaction("dth", "stochastic", {"X": -1},
                 Ast("mul", (Ast("param", "d"), Ast("species", "X"))))])


def test_pure_continuous_analytic():
    ts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    tr = simulate_plain(_decay(), 0.01, 5.0, ts, 1)
    assert tr.times == ts
    for t, x in zip(ts, tr.species[0]):
        assert abs(x - 100 * math.exp(-0.5 * t)) < 1e-5


def test_determinism():
    ts = [0.0, 5.0, 10.0]
    a = simulate_plain(_bd(), 0.1, 10.0, ts, 12345)
    b = simulate_plain(_bd(), 0.1, 10.0, ts, 12345)
    assert a.species == b.species and a.times == b.times
    assert any(simulate_plain(_bd(), 0.1, 10.0, ts, s).species != a.species for s in range(5))


def test_no_negative():
    m = Model([Species("X", 0.0)], {"d": 5.0},
              [Reaction("d", "stochastic", {"X": -1}, Ast("param", "d"))])
    tr = simulate_plain(m, 0.1, 5.0, [0.0, 2.5, 5.0], 3)
    assert tr.species[0] == [0.0, 0.0, 0.0]


def test_big_dt_single_event():
    m = Model([Species("X", 1.0)], {"d": 1000.0},
              [Reaction("d", "stochastic", {"X": -1},
                        Ast("mul", (Ast("param", "d"), Ast("species", "X"))))])
    tr = simulate_plain(m, 1000.0, 1000.0, [0.0, 500.0, 1000.0], 7)
    assert tr.species[0] == [1.0, 0.0, 0.0]
