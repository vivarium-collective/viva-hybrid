"""Plain-Python twin of the Rust Haseltine-Rawlings hybrid simulator (Task 6)."""
from __future__ import annotations

import math

import numpy as np

from ._common import Sampler, Trajectory, compile_rate, next_uniform

STOP_EPS = 1e-12


def simulate_plain(model, dt, t_end, sample_times, seed) -> Trajectory:
    m = len(model.species)
    index = {s.id: i for i, s in enumerate(model.species)}
    cont, stoch = [], []
    for r in model.reactions:
        rate = compile_rate(r.rate, model.params, index)
        st = [(index[k], v) for k, v in r.stoichiometry.items()]
        (cont if r.partition == "continuous" else stoch).append((rate, st))

    def deriv(z):
        dz = [0.0] * (m + 1)
        x = z[:m]
        for rate, st in cont:
            a = rate(x)
            for s, nu in st:
                dz[s] += float(nu) * a
        dz[m] = sum(rate(x) for rate, _ in stoch)
        return dz

    def rk4(z, h):
        k1 = deriv(z)
        k2 = deriv([zi + 0.5 * h * k for zi, k in zip(z, k1)])
        k3 = deriv([zi + 0.5 * h * k for zi, k in zip(z, k2)])
        k4 = deriv([zi + h * k for zi, k in zip(z, k3)])
        return [zi + h / 6.0 * (a + 2 * b + 2 * c + d)
                for zi, a, b, c, d in zip(z, k1, k2, k3, k4)]

    rng = np.random.default_rng(seed)
    z = [s.initial for s in model.species] + [0.0]
    thr = -math.log(next_uniform(rng))
    samp = Sampler(m, sample_times)
    t = 0.0
    samp.capture(t, z)
    while t + STOP_EPS < t_end:
        h = min(dt, t_end - t)
        saved = z
        nxt = rk4(saved, h)
        if nxt[m] < thr:
            z = nxt
            t += h
            samp.capture(t, z)
            continue
        g0, g1 = saved[m], nxt[m]
        f = min(1.0, max(0.0, (thr - g0) / (g1 - g0)))
        z = rk4(saved, f * h)
        t += f * h
        u_sel = next_uniform(rng)
        props = [rate(z[:m]) for rate, _ in stoch]
        target = u_sel * sum(props)
        j, acc = len(stoch) - 1, 0.0
        for k, a in enumerate(props):
            acc += a
            if acc >= target:
                j = k
                break
        prop = list(z[:m])
        for s, nu in stoch[j][1]:
            prop[s] += nu
        if all(v >= 0 for v in prop):
            z[:m] = prop
        z[m] = 0.0
        thr = -math.log(next_uniform(rng))
        samp.capture(t, z)
    return samp.trajectory()
