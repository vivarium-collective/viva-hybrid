"""Numba-jitted twin of the hybrid simulator (Task 8).

The model is lowered once (plain Python) to flat typed arrays; the whole event
loop then runs inside @njit.

Op-code program format (postfix, one program per reaction, all programs
concatenated; reaction r owns slots [off[r], off[r+1])):
  code[i]  int64 op: 0 const, 1 species-read, 2 add, 3 sub, 4 mul, 5 div, 6 pow
  fval[i]  float64 operand for const (params are folded to const)
  ival[i]  int64 species index for species-read
Evaluated by a stack machine; stack depth is computed at lowering time.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from ._common import SAMPLE_EPS, Trajectory

STOP_EPS = 1e-12
_OPS = {"add": 2, "sub": 3, "mul": 4, "div": 5, "pow": 6}


def _emit(ast, params, index, code, fval, ival):
    """Append postfix program for ast; return max stack depth needed."""
    op, arg = ast.op, ast.arg
    if op in ("const", "param"):
        code.append(0)
        fval.append(float(arg) if op == "const" else float(params[arg]))
        ival.append(0)
        return 1
    if op == "species":
        code.append(1)
        fval.append(0.0)
        ival.append(index[arg])
        return 1
    if op not in _OPS:
        raise ValueError(f"unknown ast op: {op!r}")
    da = _emit(arg[0], params, index, code, fval, ival)
    db = _emit(arg[1], params, index, code, fval, ival)
    code.append(_OPS[op])
    fval.append(0.0)
    ival.append(0)
    return max(da, db + 1)


def lower(model):
    m = len(model.species)
    index = {s.id: i for i, s in enumerate(model.species)}
    nr = len(model.reactions)
    S = np.zeros((nr, m), dtype=np.int64)
    is_stoch = np.zeros(nr, dtype=np.bool_)
    code, fval, ival, off = [], [], [], [0]
    depth = 1
    for r, rx in enumerate(model.reactions):
        for k, v in rx.stoichiometry.items():
            S[r, index[k]] = v
        is_stoch[r] = rx.partition != "continuous"
        depth = max(depth, _emit(rx.rate, model.params, index, code, fval, ival))
        off.append(len(code))
    return dict(
        x0=np.array([s.initial for s in model.species], dtype=np.float64),
        S=S, is_stoch=is_stoch,
        cont_idx=np.array([r for r in range(nr) if not is_stoch[r]], dtype=np.int64),
        stoch_idx=np.array([r for r in range(nr) if is_stoch[r]], dtype=np.int64),
        code=np.array(code, dtype=np.int64), fval=np.array(fval, dtype=np.float64),
        ival=np.array(ival, dtype=np.int64), off=np.array(off, dtype=np.int64),
        depth=depth)


@njit(cache=True)
def _eval(r, x, code, fval, ival, off, stack):
    sp = 0
    for i in range(off[r], off[r + 1]):
        c = code[i]
        if c == 0:
            stack[sp] = fval[i]
            sp += 1
        elif c == 1:
            stack[sp] = x[ival[i]]
            sp += 1
        else:
            b = stack[sp - 1]
            a = stack[sp - 2]
            sp -= 1
            if c == 2:
                v = a + b
            elif c == 3:
                v = a - b
            elif c == 4:
                v = a * b
            elif c == 5:
                if b != 0.0:
                    v = a / b
                elif a != 0.0:
                    v = math.copysign(math.inf, a)
                else:
                    v = math.nan
            else:
                v = math.pow(a, b)
            stack[sp - 1] = v
    return stack[0]


@njit(cache=True)
def _deriv(z, dz, m, S, cont_idx, stoch_idx, code, fval, ival, off, stack):
    for i in range(m + 1):
        dz[i] = 0.0
    x = z[:m]
    for q in range(cont_idx.shape[0]):
        r = cont_idx[q]
        a = _eval(r, x, code, fval, ival, off, stack)
        for s in range(m):
            nu = S[r, s]
            if nu != 0:
                dz[s] += float(nu) * a
    tot = 0.0
    for q in range(stoch_idx.shape[0]):
        tot += _eval(stoch_idx[q], x, code, fval, ival, off, stack)
    dz[m] = tot


@njit(cache=True)
def _rk4(z, h, out, ws, m, S, cont_idx, stoch_idx, code, fval, ival, off, stack):
    n = m + 1
    k1, k2, k3, k4, tmp = ws[0], ws[1], ws[2], ws[3], ws[4]
    _deriv(z, k1, m, S, cont_idx, stoch_idx, code, fval, ival, off, stack)
    for i in range(n):
        tmp[i] = z[i] + 0.5 * h * k1[i]
    _deriv(tmp, k2, m, S, cont_idx, stoch_idx, code, fval, ival, off, stack)
    for i in range(n):
        tmp[i] = z[i] + 0.5 * h * k2[i]
    _deriv(tmp, k3, m, S, cont_idx, stoch_idx, code, fval, ival, off, stack)
    for i in range(n):
        tmp[i] = z[i] + h * k3[i]
    _deriv(tmp, k4, m, S, cont_idx, stoch_idx, code, fval, ival, off, stack)
    for i in range(n):
        out[i] = z[i] + h / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@njit(cache=True)
def _next_uniform():
    u = np.random.random()
    while u == 0.0:
        u = np.random.random()
    return u


@njit(cache=True)
def _capture(t, z, m, sample_times, out, nxt):
    while nxt < sample_times.shape[0] and sample_times[nxt] <= t + SAMPLE_EPS:
        for i in range(m):
            out[i, nxt] = z[i]
        nxt += 1
    return nxt


@njit(cache=True)
def _run(x0, S, cont_idx, stoch_idx, code, fval, ival, off, depth,
         dt, t_end, sample_times, seed):
    np.random.seed(seed)
    m = x0.shape[0]
    n = m + 1
    out = np.zeros((m, sample_times.shape[0]))
    stack = np.zeros(depth + 1)
    ws = np.zeros((5, n))
    z = np.zeros(n)
    z[:m] = x0
    saved = np.zeros(n)
    nz = np.zeros(n)
    thr = -math.log(_next_uniform())
    nxt = _capture(0.0, z, m, sample_times, out, 0)
    t = 0.0
    ns = stoch_idx.shape[0]
    while t + STOP_EPS < t_end:
        h = min(dt, t_end - t)
        saved[:] = z
        _rk4(saved, h, nz, ws, m, S, cont_idx, stoch_idx, code, fval, ival, off, stack)
        if nz[m] < thr:
            z[:] = nz
            t += h
            nxt = _capture(t, z, m, sample_times, out, nxt)
            continue
        g0 = saved[m]
        g1 = nz[m]
        f = min(1.0, max(0.0, (thr - g0) / (g1 - g0)))
        _rk4(saved, f * h, z, ws, m, S, cont_idx, stoch_idx, code, fval, ival, off, stack)
        t += f * h
        u_sel = _next_uniform()
        tot = 0.0
        for q in range(ns):
            tot += _eval(stoch_idx[q], z[:m], code, fval, ival, off, stack)
        target = u_sel * tot
        j = ns - 1
        acc = 0.0
        for q in range(ns):
            acc += _eval(stoch_idx[q], z[:m], code, fval, ival, off, stack)
            if acc >= target:
                j = q
                break
        if ns > 0:
            r = stoch_idx[j]
            ok = True
            for s in range(m):
                if z[s] + S[r, s] < 0:
                    ok = False
            if ok:
                for s in range(m):
                    z[s] += S[r, s]
        z[m] = 0.0
        thr = -math.log(_next_uniform())
        nxt = _capture(t, z, m, sample_times, out, nxt)
    return out


def simulate_numba(model, dt, t_end, sample_times, seed) -> Trajectory:
    L = lower(model)
    ts = np.asarray(sample_times, dtype=np.float64)
    out = _run(L["x0"], L["S"], L["cont_idx"], L["stoch_idx"], L["code"], L["fval"],
               L["ival"], L["off"], L["depth"], float(dt), float(t_end), ts, int(seed))
    return Trajectory(times=list(sample_times), species=[row.tolist() for row in out])
