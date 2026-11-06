import json
import math

from viva_hybrid_ref.sweep import run_sweep, save_results


def test_tiny_sweep(tmp_path):
    res = run_sweep(
        "event_density", "stochastic_fraction",
        fixed=dict(system_size=3, timescale_sep=10.0, dt=0.02, t_end=0.5, sample_times=[0.25, 0.5]),
        grid=dict(event_density=[10.0, 100.0], stochastic_fraction=[0.34, 1.0]),
        timing=dict(n_time_reps=3, n_time_repeats=2), fidelity=dict(n_fidelity_reps=30))
    assert len(res) == 4
    for r in res:
        assert "error" not in r, r
        assert r["event_density"] in (10.0, 100.0) and r["stochastic_fraction"] in (0.34, 1.0)
        for k in ("rust_vs_numba", "rust_vs_plain"):
            assert math.isfinite(r[k]) and r[k] > 0
        assert isinstance(r["fidelity_ok"], bool)
        assert r["steps"] > 0 and r["events"] > 0
    p = save_results(res, tmp_path / "s.json")
    assert len(json.loads(p.read_text())) == 4
