"""Shared helpers for the Python reference simulators."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trajectory:
    """times copies sample_times; species[i][k] = species i at times[k] (model order)."""
    times: list = field(default_factory=list)
    species: list = field(default_factory=list)


def compile_rate(ast, params, index):
    """Compile an Ast to a closure f(x) with params folded and species by index."""
    op, arg = ast.op, ast.arg
    if op == "const":
        c = float(arg)
        return lambda x: c
    if op == "param":
        c = float(params[arg])
        return lambda x: c
    if op == "species":
        i = index[arg]
        return lambda x: x[i]
    a, b = compile_rate(arg[0], params, index), compile_rate(arg[1], params, index)
    if op == "add":
        return lambda x: a(x) + b(x)
    if op == "sub":
        return lambda x: a(x) - b(x)
    if op == "mul":
        return lambda x: a(x) * b(x)
    if op == "div":
        def div(x):
            d = b(x)
            return a(x) / d if d != 0.0 else math.copysign(math.inf, a(x)) if a(x) != 0.0 else math.nan
        return div
    if op == "pow":
        def pw(x):
            try:
                return math.pow(a(x), b(x))
            except (ValueError, OverflowError):
                return math.nan
        return pw
    raise ValueError(f"unknown ast op: {op!r}")


def next_uniform(rng):
    """Uniform in the open interval (0,1): 53-bit value, exact 0.0 rejected."""
    while True:
        u = int(rng.integers(0, 1 << 53)) / 9007199254740992.0
        if u != 0.0:
            return u


SAMPLE_EPS = 1e-9


class Sampler:
    """Interpolation-free first-reach capture (same rule as the Rust simulator)."""

    def __init__(self, n_species, sample_times):
        self.times = list(sample_times)
        self.cols = [[0.0] * len(self.times) for _ in range(n_species)]
        self.next = 0

    def capture(self, t, x):
        while self.next < len(self.times) and self.times[self.next] <= t + SAMPLE_EPS:
            for i in range(len(self.cols)):
                self.cols[i][self.next] = x[i]
            self.next += 1

    def trajectory(self):
        return Trajectory(times=list(self.times), species=self.cols)

    def capture_before(self, t, x):
        """Record samples strictly earlier than an event at t (pre-event state)."""
        while self.next < len(self.times) and self.times[self.next] < t:
            for i in range(len(self.cols)):
                self.cols[i][self.next] = x[i]
            self.next += 1
