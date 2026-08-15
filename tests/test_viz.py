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
    import compileml.viz._arrow as arrow_module
    import compileml.viz._data as data_module
    import compileml.viz.svg as svg_module

    for module in (svg_module, data_module, arrow_module):
        assert "matplotlib" not in {name.split(".")[0] for name in vars(module)}


# --------------------------------------------------------- arrow geometry
# The original waterfall's debug checklist, encoded as tests.
def test_tri_frac_adaptive():
    from compileml.viz._arrow import tri_frac

    assert tri_frac(0.0, 100.0) == 0.50  # zero-width bar: max head
    assert tri_frac(100.0, 100.0) == pytest.approx(0.05)  # full-range bar: min head
    assert tri_frac(50.0, 100.0) == pytest.approx(0.275)  # linear in between
    assert tri_frac(10_000.0, 100.0) == 0.05  # clamps below
    # Debug checklist: "heads all 50% -> scale is 0 (degenerate viewRange)"
    assert tri_frac(10.0, 0.0) == 0.50


def test_arrow_points_seven_direction_aware():
    from compileml.viz._arrow import arrow_points

    right = arrow_points(0.0, 100.0, 50.0, half_h=9.0, overhang=2.0, tri_len=20.0)
    left = arrow_points(100.0, 0.0, 50.0, half_h=9.0, overhang=2.0, tri_len=20.0)
    assert len(right) == len(left) == 7

    # Junction between start and tip, flipped per direction — the checklist's
    # "heads inverted on negative bars -> re computed with + rectLen in both".
    assert right[1][0] == pytest.approx(80.0)  # xStart + rectLen
    assert left[1][0] == pytest.approx(20.0)  # xStart - rectLen
    assert right[3] == (100.0, 50.0)  # tip at xEnd, yMid
    assert left[3] == (0.0, 50.0)

    # Barb: head visibly wider than the shaft — the checklist's "heads flush
    # with shaft -> missing ±2 overhang points".
    top, bot = 50.0 - 9.0, 50.0 + 9.0
    assert right[2][1] == top - 2.0
    assert right[4][1] == bot + 2.0

    # Same point ordering both directions: winding stays consistent.
    assert [p[1] for p in right] == [p[1] for p in left]


def test_arrow_dead_zone():
    from compileml.viz._arrow import arrow_points

    assert arrow_points(10.0, 10.2, 50.0, tri_len=0.05) is None  # < 0.3 px
    assert arrow_points(10.0, 10.4, 50.0, tri_len=0.1) is not None


def test_svg_bars_are_single_polygons(decision):
    import re

    svg = waterfall_svg(decision)
    polygons = re.findall(r'<polygon points="([^"]+)" fill="#[0-9a-f]{6}" opacity="0.92"/>', svg)
    n_segments = len(waterfall_segments(decision)["segments"])
    assert len(polygons) == n_segments  # one polygon per bar: body + head, one path
    for points in polygons:
        assert len(points.split()) == 7  # 7-point arrow
    assert "<rect" in svg  # background + score bar remain rects
    assert 'stroke="#' not in svg.split("<polygon")[1].split(">")[0]  # no stroke on arrows


def test_svg_dead_zone_renders_tick(decision):
    # The dead-zone is relative: one dominant bar sets the scale, and a
    # 1-half-micro bar becomes sub-pixel next to it -> 6 px tick, no smear.
    tiny = dict(decision)
    tiny["contributions"] = [dict(c) for c in decision["contributions"]]
    tiny["contributions"][1]["impact_half_micro"] = 1
    svg = waterfall_svg(tiny, max_features=3)
    assert svg.count("<polygon") >= 1  # the dominant bars stay arrows
    assert 'stroke-width="1.5"' in svg  # the sub-pixel bar became a tick


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


def test_decision_drivers_original_geometry(population):
    """Pin the authored beeswarm design: sizes 6/7 with bad-on-top, dashed
    zero line, exact title/xlabel/legend text, reason codes as y labels."""
    from compileml.viz import decision_drivers

    decisions, y = population
    fig, ax = decision_drivers(decisions, y=y)

    sizes = {int(c.get_sizes()[0]) for c in ax.collections if len(c.get_sizes())}
    assert {6, 7} <= sizes  # good s=6, bad s=point_size+1
    assert 140 in sizes  # mean-impact tick, s=140

    zero_lines = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert zero_lines and zero_lines[0].get_alpha() == 0.35

    assert ax.get_title(loc="left") == "Decision Drivers (Sorted by Mean Abs Impact)"
    assert ax.get_xlabel() == "Delta Latent (Whitebox)"
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_texts == ["Repayment (y=0)", "Default (y=1)", "Mean Impact"]

    # y labels are reason CODES from the artifact dictionary
    labels = {t.get_text() for t in ax.get_yticklabels()}
    assert "F0" in labels  # conftest artifact: f0 -> code "F0"
    mpl.pyplot.close(fig)


def test_decision_drivers_impact_mode_without_y(population):
    from compileml.viz import decision_drivers

    decisions, _ = population
    fig, ax = decision_drivers(decisions)  # color_by="auto" -> impact mode
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_texts == ["Risk-decreasing impact", "Risk-increasing impact", "Mean Impact"]
    mpl.pyplot.close(fig)


def test_pile_jitter_only_breaks_piles():
    import numpy as np

    from compileml.viz.plots import _pile_jitter

    codes = np.array(["A"] * 5 + ["B"], dtype=object)
    impacts = np.array([0.1] * 5 + [0.3])
    out = _pile_jitter(codes, impacts, np.random.default_rng(0))
    assert out[5] == 0.3  # singleton pile: untouched
    width = min(0.18, 0.06 * np.sqrt(5))
    assert np.all(np.abs(out[:5] - 0.1) <= width)  # pile spread, bounded
    assert np.any(out[:5] != 0.1)


def test_band_conditioned_drivers_original_design(population):
    from compileml.viz import band_conditioned_decision_drivers, band_drivers

    assert band_drivers is band_conditioned_decision_drivers  # alias kept

    decisions, y = population
    fig, axes = band_conditioned_decision_drivers(decisions, y=y, cols=2)
    assert axes.size >= len({d["band"] for d in decisions})

    panel_titles = {ax.get_title(loc="left") for ax in axes if ax.get_title(loc="left")}
    assert any(t.startswith("Band G") for t in panel_titles)
    assert fig.get_suptitle() == "Decision Drivers - Band Conditioned"
    fig_texts = [t.get_text() for t in fig.texts]
    assert any("deterministic pairwise-interaction attribution" in t for t in fig_texts)
    legend_texts = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend_texts == [
        "Observed repayment (y=0)",
        "Observed default (y=1)",
        "Mean Impact",
    ]
    mpl.pyplot.close(fig)


def test_band_conditioned_target_band_validates(population):
    from compileml.viz import band_conditioned_decision_drivers

    decisions, y = population
    with pytest.raises(ValueError, match="not found"):
        band_conditioned_decision_drivers(decisions, y=y, target_band="G99")


def test_band_ladder_renders(population):
    from compileml.viz import band_ladder

    decisions, y = population
    fig, ax = band_ladder(decisions, y)
    assert len(ax.patches) == len({d["band"] for d in decisions})
    mpl.pyplot.close(fig)
