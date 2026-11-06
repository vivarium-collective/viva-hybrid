import json
from pathlib import Path

import pytest

from viva_hybrid_ref.regime_viz import render_regime_map

REG = Path(__file__).resolve().parents[2] / "reports" / "regime"
CASES = [
    ("event_density_x_stochastic_fraction", ["event density", "stochastic fraction"]),
    ("system_size_x_stochastic_fraction", ["system size", "stochastic fraction"]),
]


@pytest.mark.parametrize("axes,names", CASES)
def test_render(tmp_path, axes, names):
    src = REG / f"sweep_{axes}.json"
    cells = json.loads(src.read_text())
    assert any(c["fidelity_ok"] for c in cells) and any(not c["fidelity_ok"] for c in cells)
    out = tmp_path / "m.html"
    render_regime_map(src, out, title="T")
    html = out.read_text()
    assert len(html) > 5000 and "plotly" in html and "Plotly.newPlot" in html
    for n in names:
        assert n in html
    for c in cells:
        assert f'{c["rust_vs_numba"]:.1f}' in html or str(c["rust_vs_numba"])[:6] in html
    assert "fidelity FAIL" in html and "fidelity OK" in html
    assert "numba" in html and "NOT swept" in html
