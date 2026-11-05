"""Exact Gillespie direct-method SSA treating all reactions as stochastic."""
from __future__ import annotations

import math

import numpy as np

from ._common import Sampler, Trajectory, compile_rate, next_uniform


def gillespie(model, t_end, sample_times, seed) -> Trajectory:
    m = len(model.species)
    index = {s.id: i for i, s in enumerate(model.species)}
    rxns = [(compile_rate(r.rate, model.params, index),
             [(index[k], v) for k, v in r.stoichiometry.items()])
            for r in model.reactions]
    rng = np.random.default_rng(seed)
    x = [s.initial for s in model.species]
    samp = Sampler(m, sample_times)
    t = 0.0
    samp.capture(t, x)
    while t < t_end:
        props = [rate(x) for rate, _ in rxns]
        total = sum(props)
        if not total > 0.0:
            break
        t += -math.log(next_uniform(rng)) / total
        if t > t_end:
            break
        samp.capture_before(t, x)  # samples before the event see the pre-event state
        target = next_uniform(rng) * total
        j, acc = len(rxns) - 1, 0.0
        for k, a in enumerate(props):
            acc += a
            if acc >= target:
                j = k
                break
        for s, nu in rxns[j][1]:
            x[s] += nu
        samp.capture(t, x)
    samp.capture(t_end, x)
    return samp.trajectory()
