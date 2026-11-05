import math
import subprocess
from pathlib import Path

from viva_hybrid_ref.bench import bench, phase_breakdown, report
from viva_hybrid_ref.compare import REPO
from viva_hybrid_ref.sbml_to_ir import sbml_to_ir

M = Path(__file__).resolve().parents[2] / "models"
CFG = dict(dt=0.1, t_end=20.0, sample_times=[10.0, 20.0])


def test_bench_and_report():
    subprocess.run(["cargo", "build", "--release", "--bin", "run_ir"], cwd=REPO, check=True)
    model = sbml_to_ir(M / "burst_gene_expression.xml", M / "burst_gene_expression.partition.yaml")
    t = bench(model, CFG, repeats=2, replicates=3)
    for v in (t.rust_s, t.plain_s, t.numba_s):
        assert math.isfinite(v) and v > 0
    assert t.steps > 0
    text = report(t, phase_breakdown(model, CFG, replicates=3))
    assert "Rust vs plain Python" in text and "Rust vs numba" in text
    assert any(w in text for w in ("GO:", "NO-GO:", "MARGINAL:"))
