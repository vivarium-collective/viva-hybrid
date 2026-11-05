import math
from viva_hybrid_ref.ir import Model, Species, Reaction, Ast
from viva_hybrid_ref.exact_ssa import gillespie


def _bd():
    return Model([Species("X", 0.0)], {"k": 10.0, "d": 1.0}, [
        Reaction("b", "continuous", {"X": 1}, Ast("param", "k")),
        Reaction("dth", "continuous", {"X": -1},
                 Ast("mul", (Ast("param", "d"), Ast("species", "X"))))])


def test_birth_death_mean():
    n = 2000
    xs = [gillespie(_bd(), 50.0, [50.0], s).species[0][0] for s in range(n)]
    mean = sum(xs) / n
    se = math.sqrt(10.0 / n)
    assert abs(mean - 10.0) < 3 * se


def test_determinism_and_shape():
    ts = [0.0, 1.0, 2.0]
    a = gillespie(_bd(), 2.0, ts, 5)
    assert a.times == ts and len(a.species) == 1 and len(a.species[0]) == 3
    assert a.species[0][0] == 0.0
    assert a == gillespie(_bd(), 2.0, ts, 5)
