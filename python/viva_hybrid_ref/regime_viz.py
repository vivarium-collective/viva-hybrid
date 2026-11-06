"""Regime-map figures: where does a Rust hybrid engine pay off AND stay accurate?

Colour = Rust-vs-numba speedup in three bands keyed to the 2x / 5x thresholds
(grey <2x dead, light blue 2-5x, dark blue >=5x go). Cells whose hybrid-vs-exact-SSA
fidelity check failed are washed out and marked with an X.
"""
from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

PLOTLY_CDN = "https://cdn.plot.ly/plotly-3.7.0.min.js"
DEAD, GO = 2.0, 5.0
ZMAX = 10.0
# single-hue (blue) sequential for 2x+, neutral grey for the dead band
COLORSCALE = [
    [0.0, "#d9d9d9"], [DEAD / ZMAX, "#d9d9d9"],
    [DEAD / ZMAX, "#9ecae1"], [GO / ZMAX, "#9ecae1"],
    [GO / ZMAX, "#08519c"], [1.0, "#08519c"],
]
INK, MUTED = "#1f2933", "#52606d"

AXIS_LABELS = {
    "event_density": "event density (reaction events per unit time per species, swept 1-1000)",
    "system_size": "system size (number of species, swept 3-100)",
    "stochastic_fraction": "stochastic fraction (share of reactions treated stochastically)",
}
AXES = ("event_density", "system_size", "stochastic_fraction")

CAPTION = (
    "Speedup is vs the <b>numba</b> baseline, the fair competitor (vs plain Python it is "
    "&ge;~28&times; everywhere). Swept range: copy numbers ~10&ndash;30, density 1&ndash;1000, "
    "system_size 3&ndash;100, timescale_sep=10, dt=0.01. "
    "<b>Copy number is NOT swept</b> and is the axis most likely to open a faithful-and-fast region. "
    "At stochastic_fraction = 1.0 the hybrid <i>is</i> the exact stochastic path, so fidelity "
    "there tests engine correctness, not an approximation. No claim is made beyond this regime."
)


def _axes(cells):
    keys = [k for k in AXES if k in cells[0] and len({c[k] for c in cells}) > 1]
    if len(keys) != 2:
        raise ValueError(f"expected exactly 2 swept axes, found {keys}")
    # stochastic_fraction on y, the other on x
    y = "stochastic_fraction" if "stochastic_fraction" in keys else keys[1]
    x = next(k for k in keys if k != y)
    return x, y


def _fmt(v):
    return f"{v:g}"


def render_regime_map(json_path, out_html, *, title) -> None:
    cells = json.loads(Path(json_path).read_text())
    x_key, y_key = _axes(cells)
    xs = sorted({c[x_key] for c in cells})
    ys = sorted({c[y_key] for c in cells})
    xl, yl = [_fmt(v) for v in xs], [_fmt(v) for v in ys]
    by = {(c[x_key], c[y_key]): c for c in cells}

    z = [[by[(x, y)]["rust_vs_numba"] for x in xs] for y in ys]
    custom = [[(by[(x, y)]["rust_vs_plain"], by[(x, y)]["worst_abs_z"],
                "OK" if by[(x, y)]["fidelity_ok"] else "FAIL") for x in xs] for y in ys]

    fig = go.Figure(go.Heatmap(
        x=xl, y=yl, z=z, customdata=custom, zmin=0, zmax=ZMAX, colorscale=COLORSCALE,
        xgap=2, ygap=2, texttemplate="%{z:.1f}×", textfont=dict(size=14),
        colorbar=dict(title="Rust vs numba", tickvals=[0, DEAD, GO, ZMAX],
                      ticktext=["0", "2× dead below", "5× go at/above", "10×+"],
                      len=0.7, y=0.55),
        hovertemplate=(f"{x_key}=%{{x}}<br>{y_key}=%{{y}}<br>Rust vs numba: %{{z:.2f}}×"
                       "<br>Rust vs plain: %{customdata[0]:.1f}×"
                       "<br>worst |z|: %{customdata[1]:.1f}<br>fidelity: %{customdata[2]}<extra></extra>"),
        name="speedup",
    ))

    fail_x, fail_y, ok_x, ok_y = [], [], [], []
    shapes = []
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if by[(x, y)]["fidelity_ok"]:
                ok_x.append(xl[j]); ok_y.append(yl[i])
            else:
                fail_x.append(xl[j]); fail_y.append(yl[i])
                # desaturating wash over untrustworthy cells (category axis: index coords)
                shapes.append(dict(type="rect", xref="x", yref="y", layer="above",
                                   x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                                   fillcolor="#ffffff", opacity=0.55, line=dict(width=0)))
    fig.add_trace(go.Scatter(
        x=fail_x, y=fail_y, mode="markers", name="✗ fidelity FAIL – not trustworthy",
        marker=dict(symbol="x-thin", size=34, line=dict(color=INK, width=2.5)),
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=ok_x, y=ok_y, mode="markers", name="□ fidelity OK (outlined)",
        marker=dict(symbol="square-open", size=62, line=dict(color="#0b6e4f", width=3)),
        hoverinfo="skip"))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.01, font=dict(size=18, color=INK)),
        shapes=shapes, template="plotly_white", font=dict(color=INK, size=13),
        xaxis=dict(title=AXIS_LABELS.get(x_key, x_key), type="category", showgrid=False),
        yaxis=dict(title=AXIS_LABELS.get(y_key, y_key), type="category", showgrid=False),
        legend=dict(orientation="h", y=-0.28, x=0, font=dict(color=INK)),
        margin=dict(l=80, r=40, t=70, b=230), height=760,
        annotations=[dict(text=CAPTION, xref="paper", yref="paper", x=0, y=-0.42,
                          xanchor="left", yanchor="top", showarrow=False, align="left",
                          font=dict(size=12, color=MUTED), width=900)],
    )
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs=PLOTLY_CDN, full_html=True)


def main() -> None:
    d = Path(__file__).resolve().parents[2] / "reports" / "regime"
    jobs = [
        ("event_density_x_stochastic_fraction",
         "Regime map: event density × stochastic fraction"),
        ("system_size_x_stochastic_fraction",
         "Regime map: system size × stochastic fraction"),
    ]
    for axes, title in jobs:
        render_regime_map(d / f"sweep_{axes}.json", d / f"regime_map_{axes}.html", title=title)
        print("wrote", d / f"regime_map_{axes}.html")


if __name__ == "__main__":
    main()
