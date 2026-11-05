"""Vertical-slice test: the Rust hybrid engine runs as a pbg Step via build_core().

Bridges the real compiled `run_ir` binary (built on demand). Skips only if
`cargo` is unavailable AND the binary isn't present (can't build the engine).
"""
import shutil
from pathlib import Path

import pytest

from process_bigraph import Composite, gather_emitter_results
from viva_hybrid.core import build_core
from viva_hybrid.composites.regime import rust_hybrid_benchmark

_REPO = Path(__file__).resolve().parents[1]
_BIN = _REPO / "target" / "release" / "run_ir"


pytestmark = pytest.mark.skipif(
    not _BIN.exists() and shutil.which("cargo") is None,
    reason="run_ir binary absent and cargo unavailable to build it",
)


def test_generator_registered():
    from process_bigraph.composite_generator import _REGISTRY
    assert any(eid.endswith(".rust_hybrid_benchmark") for eid in _REGISTRY), \
        f"rust_hybrid_benchmark not in registry: {list(_REGISTRY)[:8]}"


def test_rust_engine_runs_in_workbench_composite():
    core = build_core()
    doc = rust_hybrid_benchmark(core, n_replicates=8, dt=0.1, t_end=100.0)
    sim = Composite({"state": doc}, core=core)
    sim.run(1.0)
    rows = gather_emitter_results(sim)[("emitter",)]
    assert rows, "no emitter rows"
    last = rows[-1]
    # The real Rust engine produced a timing and did real work.
    assert last["sim_seconds"] > 0.0
    assert last["steps"] > 0
    assert last["events"] > 0
    # The bursting gene-expression model makes protein (the observable species).
    assert last["observable"] > 0.0
