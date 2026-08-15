"""Visualization tests.

The load-bearing assertions are on the exact integers in ``_data``: the
waterfall's segments must satisfy the spec §7.4 identity in half-micro
units. Renderer tests are smoke-level (Agg backend); the SVG renderer is
additionally checked for determinism and escaping, with no matplotlib
requirement.
"""

import pytest

from compileml.runtime import decide
from compileml.viz._data import band_table, driver_table, waterfall_segments
from compileml.viz.svg import waterfall_svg

X_ROW = [1.0, 5.0, 7.0]  # rich row from the hand-built conftest artifact


@pytest.fixture
def decision(artifact):
    return decide(artifact, X_ROW, include_contributions=True)


# ----------------------------------------------------------------- _data
def test_waterfall_segments_identity_exact(decision):
    data = waterfall_segments(decision)
    lhs = 2 * (data["raw_micro"] - data["baseline_micro"])
    assert lhs == sum(seg["half_micro"] for seg in data["segments"])


def test_waterfall_segments_remainder_balances(decision):
    # Truncating to one feature must still balance exactly via the remainder.
    data = waterfall_segments(decision, max_features=1)
    kinds = [seg["kind"] for seg in data["segments"]]
    assert kinds == ["impact", "remainder"]
    lhs = 2 * (data["raw_micro"] - data["baseline_micro"])
    assert lhs == sum(seg["half_micro"] for seg in data["segments"])


def test_waterfall_requires_contributions(artifact):
    plain = decide(artifact, X_ROW)  # no include_contributions
    with pytest.raises(ValueError, match="include_contributions"):
        waterfall_segments(plain)


def test_driver_table_exact(artifact, decision):
    names, rows = driver_table([decision, decision])
    assert names == artifact["features"]["names"]
    micro = decision["micro_scale"]
    expected = [
        c["impact_half_micro"] / (2 * micro)
        for c in sorted(decision["contributions"], key=lambda c: c["index"])
    ]
    assert rows[0] == expected == rows[1]


def test_band_table(artifact):
    decisions = [
        decide(artifact, X_ROW, explain=False),
        decide(artifact, [0.0, 20.0, 0.0], explain=False),
    ]
    table = band_table(decisions, y=[1, 0])
    assert sum(t["n"] for t in table) == 2
    assert all(t["bad_rate"] is not None for t in table)


# ------------------------------------------------------------------- SVG
def test_svg_deterministic_and_escaped(decision):
    svg_a = waterfall_svg(decision)
    svg_b = waterfall_svg(decision)
    assert svg_a == svg_b  # same payload, same bytes
    assert svg_a.startswith("<svg")
    assert f"Band {decision['band']}" in svg_a

    hostile = dict(decision)
    hostile["contributions"] = [dict(c) for c in decision["contributions"]]
    hostile["contributions"][0]["feature"] = "<script>alert('x')</script>"
    rendered = waterfall_svg(hostile)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_svg_needs_no_matplotlib():
    import compileml.viz._data as data_module
    import compileml.viz.svg as svg_module

    for module in (svg_module, data_module):
        assert "matplotlib" not in {name.split(".")[0] for name in vars(module)}


# ------------------------------------------------------------ matplotlib
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")


@pytest.fixture
def population(artifact):
    import numpy as np

    rng = np.random.default_rng(3)
    rows = np.column_stack([rng.uniform(-1, 2, 80), rng.uniform(0, 30, 80), rng.uniform(0, 10, 80)])
    decisions = [decide(artifact, [float(v) for v in r], include_contributions=True) for r in rows]
    y = rng.integers(0, 2, 80)
    return decisions, y


def test_waterfall_renders(decision):
    from compileml.viz import waterfall

    fig, ax = waterfall(decision, labels={"f0": "Feature Zero"})
    assert ax.get_title(loc="left").startswith("Band")
    assert len(ax.patches) >= 2  # segment bars + score bar
    mpl.pyplot.close(fig)


def test_decision_drivers_renders(population):
    from compileml.viz import decision_drivers

    decisions, y = population
    fig, ax = decision_drivers(decisions, y=y, top=3)
    assert len(ax.collections) > 0
    mpl.pyplot.close(fig)


def test_band_drivers_renders(population):
    from compileml.viz import band_drivers

    decisions, y = population
    fig, axes = band_drivers(decisions, y=y, top_k=3, cols=2)
    assert axes.size >= len({d["band"] for d in decisions})
    mpl.pyplot.close(fig)


def test_band_ladder_renders(population):
    from compileml.viz import band_ladder

    decisions, y = population
    fig, ax = band_ladder(decisions, y)
    assert len(ax.patches) == len({d["band"] for d in decisions})
    mpl.pyplot.close(fig)
