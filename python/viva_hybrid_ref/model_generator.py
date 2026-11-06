"""Parametrized synthetic hybrid-model generator for the regime-map sweep.

Topology (S = system_size species X0..X{S-1}, all mass-action):
  prod_i : 0 -> Xi      rate k            (zeroth order)
  deg_i  : Xi -> 0      rate k * Xi
  conv_i : Xi -> Xi+1   rate k * Xi       (i < S-1; nearest-neighbour chain)
=> 3S-1 reactions, one connected component for any S.

Knob -> constants mapping (every reaction r gets a "scale" sc(r) and a jitter
j(r) ~ U(0.7,1.3) drawn from the seed):
  * stochastic_fraction f: n_stoch = round(f * R) reactions, chosen by a
    seed-shuffled permutation of reaction indices, are tagged "stochastic".
  * Rate constants are k = sc * j * shape, with shape = 1 (deg), 0.5 (conv),
    N0_i (prod), so each species sits near its initial copy number N0_i
    (integer in [10, 30]) -> non-stiff, no blow-up, no extinction.
  * event_density: the stochastic scale s is solved so the summed propensity
    of the stochastic reactions at the initial state equals event_density
    exactly (propensity is linear in s). If f == 0 there are no stochastic
    reactions, so the same normalisation is applied to all reactions
    (a reference scale only).
  * timescale_sep: continuous reactions use scale c = timescale_sep * s, i.e.
    they run timescale_sep times faster than the stochastic ones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .ir import Ast, Model, Reaction, Species, dump_ir

_CONV_SHAPE = 0.5


def _check(event_density, stochastic_fraction, system_size, timescale_sep):
    if isinstance(system_size, bool) or not isinstance(system_size, (int, np.integer)) or system_size < 1:
        raise ValueError(f"system_size must be an integer >= 1, got {system_size!r}")
    if not 0.0 <= stochastic_fraction <= 1.0:
        raise ValueError(f"stochastic_fraction must be in [0,1], got {stochastic_fraction!r}")
    if not event_density > 0.0:
        raise ValueError(f"event_density must be > 0, got {event_density!r}")
    if not timescale_sep > 0.0:
        raise ValueError(f"timescale_sep must be > 0, got {timescale_sep!r}")


def generate_model(*, event_density: float, stochastic_fraction: float, system_size: int,
                   timescale_sep: float = 10.0, seed: int = 0) -> Model:
    _check(event_density, stochastic_fraction, system_size, timescale_sep)
    S = int(system_size)
    rng = np.random.default_rng(seed)
    n0 = [int(v) for v in rng.integers(10, 31, size=S)]

    # (id, stoichiometry, rate-shape, species the propensity is proportional to or None)
    spec = []
    for i in range(S):
        spec.append((f"prod_{i}", {f"X{i}": 1}, float(n0[i]), None))
        spec.append((f"deg_{i}", {f"X{i}": -1}, 1.0, f"X{i}"))
        if i < S - 1:
            spec.append((f"conv_{i}", {f"X{i}": -1, f"X{i+1}": 1}, _CONV_SHAPE, f"X{i}"))
    R = len(spec)
    jitter = rng.uniform(0.7, 1.3, size=R)
    n_stoch = int(round(stochastic_fraction * R))
    stoch = set(int(j) for j in rng.permutation(R)[:n_stoch])

    # Unit-scale propensity at the initial state (prod: shape; others: shape * N0).
    def unit_prop(j):
        _, _, shape, sp = spec[j]
        return jitter[j] * shape * (1.0 if sp is None else n0[int(sp[1:])])

    ref = stoch if stoch else set(range(R))
    s = event_density / sum(unit_prop(j) for j in ref)
    c = timescale_sep * s

    params, reactions = {}, []
    for j, (rid, stoich, shape, sp) in enumerate(spec):
        is_s = j in stoch
        k = (s if is_s else c) * jitter[j] * shape
        params[f"k_{rid}"] = float(k)
        rate = Ast("param", f"k_{rid}")
        if sp is not None:
            rate = Ast("mul", (rate, Ast("species", sp)))
        reactions.append(Reaction(rid, "stochastic" if is_s else "continuous", stoich, rate))
    return Model(species=[Species(f"X{i}", float(n0[i])) for i in range(S)],
                 params=params, reactions=reactions)


def write_ir(model: Model, path) -> Path:
    p = Path(path)
    p.write_text(dump_ir(model))
    return p
