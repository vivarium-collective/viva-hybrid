"""Real bridge: the Rust hybrid ODE/SSA engine as a process-bigraph Step.

This drives the GENUINE compiled Rust engine (`crates/viva-hybrid`, binary
`target/release/run_ir`) — it is not a reimplementation. `update()` shells out
to the real binary with an IR model + run-config and parses the trajectories it
produces. The release binary is built on first use if absent.

It is a `Step` (not a time-stepped `Process`): one invocation runs a whole
hybrid simulation (N replicate seeds) and emits summary observables — the
engine's own simulate wall-time, its step/event counts, and the per-species
mean at the final sample time. The regime-map sweep harness drives the engine
directly for bulk runs; this Step is what makes the Rust core runnable and
inspectable inside the vivarium-workbench.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from process_bigraph import Step

# viva_hybrid/processes/rust_hybrid.py -> repo root is two parents up.
_REPO = Path(__file__).resolve().parents[2]
_BIN = _REPO / "target" / "release" / "run_ir"
_CARGO_MANIFEST = _REPO / "Cargo.toml"


def ensure_binary() -> Path:
    """Return the path to the release `run_ir` binary, building it if absent."""
    if _BIN.exists():
        return _BIN
    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError(
            f"run_ir binary missing at {_BIN} and `cargo` is not on PATH; "
            "build it with `cargo build --release --bin run_ir`."
        )
    subprocess.run(
        [cargo, "build", "--release", "--bin", "run_ir",
         "--manifest-path", str(_CARGO_MANIFEST)],
        check=True,
    )
    if not _BIN.exists():
        raise RuntimeError(f"cargo build did not produce {_BIN}")
    return _BIN


class RustHybridStep(Step):
    """Run a hybrid ODE/SSA model (given as an IR JSON file) through the Rust engine.

    The model + simulation config are construction-time config (a Step runs a
    whole simulation per invocation, so there is no time-varying input a sibling
    would write — hence no input ports). Outputs are summary observables the
    workbench can render.
    """

    config_schema = {
        "ir_path": {"_type": "string", "_default": ""},        # path to an IR model JSON
        "dt": {"_type": "float", "_default": 0.1},
        "t_end": {"_type": "float", "_default": 300.0},
        "sample_times": {"_type": "list[float]",
                         "_default": [50.0, 100.0, 200.0, 300.0]},
        "n_replicates": {"_type": "integer", "_default": 50},
        "seed_start": {"_type": "integer", "_default": 0},
        # which species' final-time mean to expose as the scalar `observable`
        "observable_species": {"_type": "string", "_default": "protein"},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {
            # mean of config['observable_species'] at the last sample time, over replicates
            "observable": "float",
            "sim_seconds": "float",             # Rust engine simulate wall-time (pure loop)
            "steps": "integer",                 # total RK4 steps across replicates
            "events": "integer",                # total stochastic events across replicates
        }

    def update(self, state):
        ir_path = str(self.config["ir_path"])
        if not ir_path:
            raise RuntimeError(
                "RustHybridStep requires config['ir_path'] (path to an IR model JSON)."
            )
        sample_times = [float(t) for t in self.config["sample_times"]]
        t_end = float(self.config["t_end"])
        if sample_times and max(sample_times) > t_end + 1e-9:
            raise RuntimeError(
                f"sample_times max ({max(sample_times)}) exceeds t_end ({t_end}); "
                "samples past the horizon are never reached and read as 0."
            )
        binary = ensure_binary()
        n = int(self.config["n_replicates"])
        seed_start = int(self.config["seed_start"])
        seeds = list(range(seed_start, seed_start + n))

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = {
                "dt": float(self.config["dt"]),
                "t_end": t_end,
                "sample_times": sample_times,
                "seeds": seeds,
            }
            (tmp / "cfg.json").write_text(json.dumps(cfg))
            out_path = tmp / "out.json"
            subprocess.run(
                [str(binary), ir_path, str(tmp / "cfg.json"), str(out_path)],
                check=True,
            )
            out = json.loads(out_path.read_text())

        reps = out["replicates"]          # reps[r][i][k]
        ids = out["species_ids"]
        n_reps = len(reps)
        last_k = len(out["times"]) - 1

        target = str(self.config["observable_species"])
        if target not in ids:
            raise RuntimeError(
                f"observable_species '{target}' not in model species {ids}"
            )
        i = ids.index(target)
        observable = (
            sum(reps[r][i][last_k] for r in range(n_reps)) / n_reps
            if n_reps else 0.0
        )

        return {
            "observable": observable,
            "sim_seconds": float(out["sim_seconds"]),
            "steps": int(out["steps"]),
            "events": int(out["events"]),
        }
