"""Parametrized synthetic hybrid-model generator for the regime-map sweep.

Topology (S = system_size species X0..X{S-1}, all mass-action):
  prod_i : 0 -> Xi      rate  P_i                  (zeroth order)
  deg_i  : Xi -> 0      rate  k_i * Xi
  conv_i : Xi -> Xi+1   rate  q_i * Xi             (i < S-1; nearest-neighbour chain)
=> 3S-1 reactions, one connected component for any S.

Partition is per SPECIES: stochastic_fraction f gives n_stoch = round(f*S)
stochastic species (at least 1 if f > 0), chosen by a seed-shuffled
permutation. A species' prod_i and deg_i always share its partition.
conv_i couples two species, so it is tagged stochastic if either endpoint is.

Rate scales (relative): stochastic species scale s, continuous species scale
c = timescale_sep * s. k_i = scale_i * jitter_i (jitter ~ U(0.7,1.3)).
conv_i uses q_i = 0.2 * min(k_i, k_{i+1}) (the slower side), and
  P_i = k_i*N0_i + q_i*N0_i - q_{i-1}*N0_{i-1}
so the net flux of every species is exactly zero at X = N0 (integers in
[10,30]). N0 is therefore a deterministic fixed point whatever the
partition draw or timescale_sep: sep only changes how FAST species relax,
never the copy-number level. (q_{i-1}N0_{i-1} <= 6 k_i < k_i N0_i, so P_i > 0.)

event_density: s is solved so the summed propensity of the stochastic
reactions at X = N0 equals event_density exactly (propensity is linear in s).
If there are no stochastic reactions (f == 0) the same normalisation is
applied to all reactions as a reference scale only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .ir import Ast, Model, Reaction, Species, dump_ir

_CONV = 0.2


def _check(event_density, stochastic_fraction, system_size, timescale_sep):
    if isinstance(system_size, bool) or not isinstance(system_size, (int, np.integer)) or system_size < 1:
        raise ValueError(f"system_size must be an integer >= 1, got {system_size!r}")
    if not 0.0 <= stochastic_fraction <= 1.0:
        raise ValueError(f"stochastic_fraction must be in [0,1], got {stochastic_fraction!r}")
    if not (np.isfinite(event_density) and event_density > 0.0):
        raise ValueError(f"event_density must be finite and > 0, got {event_density!r}")
    if not (np.isfinite(timescale_sep) and timescale_sep > 0.0):
        raise ValueError(f"timescale_sep must be finite and > 0, got {timescale_sep!r}")


def generate_model(*, event_density: float, stochastic_fraction: float, system_size: int,
                   timescale_sep: float = 10.0, seed: int = 0) -> Model:
    _check(event_density, stochastic_fraction, system_size, timescale_sep)
    S = int(system_size)
    rng = np.random.default_rng(seed)
    n0 = [int(v) for v in rng.integers(10, 31, size=S)]
    jitter = rng.uniform(0.7, 1.3, size=S)
    n_stoch = int(round(stochastic_fraction * S))
    if stochastic_fraction > 0.0:
        n_stoch = max(n_stoch, 1)
    stoch = set(int(j) for j in rng.permutation(S)[:n_stoch])

    def build(s):
        """Rate constants with stochastic scale s (continuous scale sep*s)."""
        k = [(s if i in stoch else timescale_sep * s) * jitter[i] for i in range(S)]
        q = [_CONV * min(k[i], k[i + 1]) for i in range(S - 1)]
        P = [k[i] * n0[i] + (q[i] * n0[i] if i < S - 1 else 0.0)
             - (q[i - 1] * n0[i - 1] if i > 0 else 0.0) for i in range(S)]
        return k, q, P

    def stoch_conv(i):
        return i in stoch or (i + 1) in stoch

    # Propensities at X=N0 for unit s, summed over the reference reaction set.
    k, q, P = build(1.0)
    total_stoch = sum(P[i] + k[i] * n0[i] for i in stoch) + \
        sum(q[i] * n0[i] for i in range(S - 1) if stoch_conv(i))
    total_all = sum(P) + sum(k[i] * n0[i] for i in range(S)) + sum(q[i] * n0[i] for i in range(S - 1))
    s = event_density / (total_stoch if stoch else total_all)
    k, q, P = build(s)

    params, reactions = {}, []

    def add(rid, part, stoich, const, sp=None):
        params[f"k_{rid}"] = float(const)
        rate = Ast("param", f"k_{rid}")
        if sp is not None:
            rate = Ast("mul", (rate, Ast("species", sp)))
        reactions.append(Reaction(rid, part, stoich, rate))

    for i in range(S):
        part = "stochastic" if i in stoch else "continuous"
        add(f"prod_{i}", part, {f"X{i}": 1}, P[i])
        add(f"deg_{i}", part, {f"X{i}": -1}, k[i], f"X{i}")
        if i < S - 1:
            add(f"conv_{i}", "stochastic" if stoch_conv(i) else "continuous",
                {f"X{i}": -1, f"X{i+1}": 1}, q[i], f"X{i}")
    return Model(species=[Species(f"X{i}", float(n0[i])) for i in range(S)],
                 params=params, reactions=reactions)


def write_ir(model: Model, path) -> Path:
    p = Path(path)
    p.write_text(dump_ir(model))
    return p
