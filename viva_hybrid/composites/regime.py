"""Showcase composite: the Rust hybrid engine running the bursting
gene-expression benchmark inside the vivarium-workbench.

This is the vertical-slice proof that the Rust core runs under the workbench.
The regime-map investigation's bulk sweep drives the engines through a harness;
this generator makes a single engine run browsable in the Composites tab.
"""
from pathlib import Path

from process_bigraph.composite_generator import composite_generator

_REPO = Path(__file__).resolve().parents[2]
_BENCHMARK_IR = _REPO / "datasets" / "burst_gene_expression.ir.json"


@composite_generator(
    name="rust_hybrid_benchmark",
    description=(
        "Rust hybrid ODE/SSA engine on the bursting gene-expression benchmark; "
        "emits per-species final means plus the engine's own simulate wall-time "
        "and RK4-step / stochastic-event counts."
    ),
    parameters={
        "n_replicates": {"type": "integer", "default": 50,
                         "description": "Number of replicate seeds"},
        "dt": {"type": "float", "default": 0.1,
               "description": "Fixed RK4 step size (minutes)"},
        "t_end": {"type": "float", "default": 300.0,
                  "description": "Simulation horizon (minutes)"},
    },
)
def rust_hybrid_benchmark(core=None, *, n_replicates=50, dt=0.1, t_end=300.0):
    return {
        "rust_hybrid": {
            "_type": "step",
            "address": "local:RustHybridStep",
            "config": {
                "ir_path": str(_BENCHMARK_IR),
                "dt": dt,
                "t_end": t_end,
                # derive from t_end so the last sample (the observable) is always reached
                "sample_times": [t_end * f for f in (0.25, 0.5, 0.75, 1.0)],
                "n_replicates": n_replicates,
                "seed_start": 0,
            },
            "inputs": {},
            "outputs": {
                "observable": ["stores", "observable"],
                "sim_seconds": ["stores", "sim_seconds"],
                "steps": ["stores", "steps"],
                "events": ["stores", "events"],
            },
        },
        "stores": {
            "observable": 0.0,   # mean protein at final sample time (model observable)
            "sim_seconds": 0.0,
            "steps": 0,
            "events": 0,
        },
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "observable": "float",
                    "sim_seconds": "float",
                    "steps": "integer",
                    "events": "integer",
                }
            },
            "inputs": {
                "observable": ["stores", "observable"],
                "sim_seconds": ["stores", "sim_seconds"],
                "steps": ["stores", "steps"],
                "events": ["stores", "events"],
            },
        },
    }
