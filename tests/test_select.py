"""compile_selected: the protocol, and the properties #75 promises.

The fixture data is small and the grids are short so the suite stays fast;
the properties do not depend on scale.
"""

import copy
import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier

import compileml
from compileml.artifact import build_artifact
from compileml.bands import monotone_quantile_bands
from compileml.batch import score_batch
from compileml.compile import extract_trees, train_whitebox
from compileml.runtime import canonical_hash, verify_artifact
from compileml.select import (
    choose,
    compile_selected,
    duplicate_rows,
    make_partitions,
    row_content_hashes,
)
from compileml.tune import sweep_whitebox
from compileml.validate import validate_artifact

N, P = 4000, 6
FEATURES = [f"f{i}" for i in range(P)]
SMALL_GRID = {"alpha": (0.0, 1.0), "trees": (20, 40), "depth": (1, 2)}


class CountingCeiling(ClassifierMixin, BaseEstimator):
    """A ceiling that counts its fits, so the protocol's fit budget is testable."""

    fits = 0

    def __init__(self, max_iter=60):
        self.max_iter = max_iter

    def fit(self, X, y):
        CountingCeiling.fits += 1
        self.model_ = HistGradientBoostingClassifier(
            max_iter=self.max_iter, max_depth=3, learning_rate=0.1, random_state=0
        ).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((N, P))
    logit = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.7 * X[:, 2] * X[:, 3] - 1.0
    y = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


@pytest.fixture(scope="module")
def run(data):
    """One flat run with a counting factory, shared by the tests that read it."""
    X, y = data
    calls = {"factory": 0}
    CountingCeiling.fits = 0

    def factory(Xf, yf):
        calls["factory"] += 1
        return CountingCeiling().fit(Xf, yf)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compile_selected(
            X,
            y,
            ceiling=factory,
            reference="woe",
            grid=SMALL_GRID,
            n_boot=30,
            k_folds=3,
            min_select_events=50,
            seed=3,
        )
    return result, calls


# ---------------------------------------------------------------- partitions
def test_partitions_are_disjoint_and_cover_every_row(data):
    _, y = data
    parts = make_partitions(y, seed=0)
    all_idx = np.sort(np.concatenate([parts.fit, parts.select, parts.report]))
    assert np.array_equal(all_idx, np.arange(N))
    assert abs(len(parts.report) / N - 0.2) < 0.01 and abs(len(parts.fit) / N - 0.6) < 0.01
    # stratified: event rates agree
    rates = [y[getattr(parts, k)].mean() for k in ("fit", "select", "report")]
    assert max(rates) - min(rates) < 0.02


def test_out_of_time_report_is_the_latest_slice(data):
    _, y = data
    date = np.arange(N)  # row i happened on day i
    parts = make_partitions(y, date=date, seed=0)
    assert parts.report.min() > max(parts.fit.max(), parts.select.max())
    assert parts.method == "out_of_time"


def test_group_split_keeps_each_group_in_one_partition(data):
    _, y = data
    group = np.arange(N) // 10  # ten rows per applicant
    parts = make_partitions(y, group=group, seed=0)
    seen = {}
    for name in ("fit", "select", "report"):
        for g in np.unique(group[getattr(parts, name)]):
            assert seen.setdefault(g, name) == name, f"group {g} straddles partitions"
    assert parts.method == "group"


def test_explicit_partitions_must_not_overlap(data):
    _, y = data
    with pytest.raises(ValueError, match="overlap"):
        make_partitions(y, explicit={"fit": [0, 1, 2], "select": [2, 3], "report": [4]})
    with pytest.raises(ValueError, match="sum to 1"):
        make_partitions(y, fractions=(0.5, 0.3, 0.3))


def test_content_duplicates_are_counted_and_thresholded(data):
    X, y = data
    hashes = row_content_hashes(X, y)
    parts = make_partitions(y, seed=0)
    assert duplicate_rows(hashes, parts) == 0
    X_dup = X.copy()
    X_dup[parts.report[:100]] = X[parts.fit[:100]]  # report rows copied from fit
    y_dup = y.copy()
    y_dup[parts.report[:100]] = y[parts.fit[:100]]
    assert duplicate_rows(row_content_hashes(X_dup, y_dup), parts) == 100
    with pytest.raises(ValueError, match="identical"):
        compile_selected(
            X_dup,
            y_dup,
            partitions={"fit": parts.fit, "select": parts.select, "report": parts.report},
            reference=0.1,
            grid={"alpha": (1.0,), "trees": (20,), "depth": (1,)},
            n_boot=5,
            min_select_events=10,
            duplicate_threshold=0.01,
        )


def test_row_id_overlap_is_a_hard_failure(data):
    X, y = data
    parts = make_partitions(y, seed=0)
    row_id = np.arange(N)
    row_id[parts.report[0]] = row_id[parts.fit[0]]  # one id in two partitions
    with pytest.raises(ValueError, match="row id"):
        compile_selected(
            X,
            y,
            row_id=row_id,
            partitions={"fit": parts.fit, "select": parts.select, "report": parts.report},
            reference=0.1,
            grid={"alpha": (1.0,), "trees": (20,), "depth": (1,)},
            n_boot=5,
        )


# ------------------------------------------------------------ the prefix trick
def test_prefix_of_a_fit_is_the_smaller_fit(data):
    """The first 20 trees of an 80-tree fit compile to the same artifact as a 20-tree fit."""
    X, y = data
    target = y.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        big, _ = train_whitebox(X, target, n_estimators=80, backend="hist", random_state=0)
        small, _ = train_whitebox(X, target, n_estimators=20, backend="hist", random_state=0)
        extracted = extract_trees(big)
        prefix = copy.copy(extracted)
        prefix.trees = extracted.trees[:20]
        latent = np.clip(small.predict(X), 0, 1)
        bands = monotone_quantile_bands(latent, y, n_bands=6)
        art_prefix = build_artifact(prefix, FEATURES, np.median(X, axis=0), bands)
        art_small = build_artifact(small, FEATURES, np.median(X, axis=0), bands)
    assert np.array_equal(
        score_batch(art_prefix, X)["raw_micro"], score_batch(art_small, X)["raw_micro"]
    )


# ------------------------------------------------------------------ ceiling
def test_ceiling_is_tuned_once_and_refit_frozen(run):
    """One search, K fits for out-of-fold targets, one refit for Report scoring."""
    result, calls = run
    k = 3
    assert calls["factory"] == 1
    prov = result.artifact["metadata"]["provenance"]["ceiling"]
    assert prov["factory_calls"] == 1
    expected = 1 + k + 1  # tune, out-of-fold on Fit, refit on Fit ∪ Select for Report
    if result.selected["alpha"] < 1.0:
        expected += k  # out-of-fold regenerated on Fit ∪ Select for the final refit
    assert prov["fits"] == expected == CountingCeiling.fits


# ---------------------------------------------------------------- selection
def _rows(y, preds):
    return {
        key: {
            "alpha": key[0],
            "n_estimators": key[1],
            "max_depth": key[2],
            "monotone": False,
            "eps": 50.0,
            "_raw": raw,
            "_pd": np.clip(raw / raw.max(), 0, 1),
        }
        for key, raw in preds.items()
    }


def test_tie_rule_prefers_alpha_one_then_fewer_trees_then_lower_depth():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(int)
    same = (y * 1000 + rng.integers(0, 200, 500)).astype(np.int64)
    rows = _rows(
        y,
        {
            (0.0, 20, 1, 0): same,
            (1.0, 80, 2, 0): same,
            (1.0, 20, 2, 0): same,
            (1.0, 20, 1, 0): same,
        },
    )
    chosen = choose(rows, y, metric="gini", tie_se=1.0, n_boot=20, seed=0)
    assert chosen["key"] == (1.0, 20, 1, 0)
    assert chosen["n_tied"] == 4


def test_a_clearly_better_configuration_wins_regardless_of_the_prior():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(int)
    sharp = (y * 1000 + rng.integers(0, 100, 500)).astype(np.int64)
    noise = rng.integers(0, 1000, 500).astype(np.int64)
    rows = _rows(y, {(0.0, 80, 2, 0): sharp, (1.0, 20, 1, 0): noise})
    chosen = choose(rows, y, metric="gini", tie_se=1.0, n_boot=20, seed=0)
    assert chosen["key"] == (0.0, 80, 2, 0)
    assert chosen["n_tied"] == 1


def test_selection_curve_and_selected_agree(run):
    result, _ = run
    curve = result.selection_curve
    assert len(curve) == 2 * 2 * 2  # alpha x trees x depth
    assert sum(r["selected"] for r in curve) == 1
    top = next(r for r in curve if r["selected"])
    assert (top["alpha"], top["n_estimators"], top["max_depth"]) == (
        result.selected["alpha"],
        result.selected["n_estimators"],
        result.selected["max_depth"],
    )
    assert all("_raw" not in r for r in curve)
    assert all(r["eps"] > 0 for r in curve)


# ------------------------------------------------------ floor and no ceiling
def test_floor_gate_stops_without_an_artifact(data):
    X, y = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stopped = compile_selected(
            X[:1500],
            y[:1500],
            reference=0.99,
            grid={"alpha": (1.0,), "trees": (20,), "depth": (2,)},
            n_boot=5,
            min_select_events=10,
            seed=3,
        )
        forced = compile_selected(
            X[:1500],
            y[:1500],
            reference=0.99,
            grid={"alpha": (1.0,), "trees": (20,), "depth": (2,)},
            n_boot=5,
            min_select_events=10,
            seed=3,
            allow_below_floor=True,
        )
    assert stopped.artifact is None and stopped.floor_gate["passed"] is False
    with pytest.raises(RuntimeError, match="floor"):
        stopped.report()
    assert forced.artifact is not None


def test_no_ceiling_forces_labels_and_reports_no_retention(data):
    X, y = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compile_selected(
            X[:1500],
            y[:1500],
            reference=0.2,
            grid={"trees": (20,), "depth": (2,)},
            n_boot=5,
            min_select_events=10,
            seed=3,
        )
    assert result.selected["alpha"] == 1.0
    assert any("forced" in n for n in result.notes)
    report = result.report()
    assert report["retention_pct"] is None and report["floor_ratio_pct"] > 0


# --------------------------------------------------------- report and hash
def test_report_is_read_once_and_written_into_the_hash(run):
    result, _ = run
    before = result.artifact["artifact_hash"]
    assert verify_artifact(result.artifact)
    report = result.report()
    prov = result.artifact["metadata"]["provenance"]
    assert prov["report"]["evaluations"] == 1
    assert report["artifact_hash"] == result.artifact["artifact_hash"] != before
    assert verify_artifact(result.artifact)
    assert report["retention_ci"][0] <= report["retention_pct"] <= report["retention_ci"][1]
    assert report["floor_ratio_ci"][0] <= report["floor_ratio_pct"] <= report["floor_ratio_ci"][1]
    with pytest.raises(RuntimeError, match="already been evaluated"):
        result.report()


def test_selection_never_computes_attribution(data, monkeypatch):
    import compileml.runtime.explain as explain

    def boom(*args, **kwargs):
        raise AssertionError("attribution was computed during selection")

    monkeypatch.setattr(explain, "contributions_half_micro", boom)
    X, y = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compile_selected(
            X[:1500],
            y[:1500],
            reference=0.1,
            grid={"alpha": (1.0,), "trees": (20,), "depth": (1,)},
            n_boot=5,
            min_select_events=10,
            seed=3,
        )
    assert result.artifact is not None


# ------------------------------------------------------------------ nested
def test_small_select_falls_back_to_nested_cv(data):
    X, y = data
    with pytest.warns(UserWarning, match="nested"):
        result = compile_selected(
            X[:1200],
            y[:1200],
            reference=0.1,
            grid={"alpha": (1.0,), "trees": (20,), "depth": (1,)},
            n_boot=5,
            k_folds=3,
            min_select_events=10_000,
            seed=3,
        )
    assert result.selected["nested"] is True
    assert result.artifact["metadata"]["provenance"]["partitions"]["method"].endswith("_nested")
    assert result.report()["rows"] == len(result.partitions.report)


# ---------------------------------------------------------------- check 11
def test_check_11_passes_on_a_reported_artifact_and_catches_tampering(run, data):
    result, _ = run
    X, y = data
    if result.report_evaluations == 0:
        result.report()
    idx = result.partitions.report
    v = validate_artifact(result.artifact, X[idx], y[idx], require_selection_hygiene=True)
    c11 = v["checks"]["11_selection_hygiene"]
    assert c11["pass"] and not c11["skipped"] and c11["issues"] == []

    tampered = copy.deepcopy(result.artifact)
    prov = tampered["metadata"]["provenance"]
    prov["report"]["evaluations"] = 2
    prov["target"]["alpha"] = 0.5
    prov["soft_targets"]["cross_fitted"] = False
    tampered["artifact_hash"] = canonical_hash(tampered)
    advisory = validate_artifact(tampered, X[idx], y[idx])["checks"]["11_selection_hygiene"]
    assert advisory["pass"] and len(advisory["issues"]) == 2
    required = validate_artifact(tampered, X[idx], y[idx], require_selection_hygiene=True)
    assert required["checks"]["11_selection_hygiene"]["pass"] is False
    assert required["all_pass"] is False

    plain = copy.deepcopy(result.artifact)
    del plain["metadata"]["provenance"]
    plain["artifact_hash"] = canonical_hash(plain)
    assert validate_artifact(plain)["checks"]["11_selection_hygiene"]["skipped"]


# --------------------------------------------------------- warnings and API
def test_sweep_whitebox_warns_about_its_old_defaults(data):
    X, y = data
    teacher = 1 / (1 + np.exp(-(X[:, 0] - 1)))
    with pytest.warns(FutureWarning, match="alpha_grid"):
        sweep_whitebox(
            X[:800], teacher[:800], y[:800], trees_grid=(10,), depth_grid=(1,), reference=0.1
        )
    with pytest.warns(UserWarning, match="choose"):
        sweep_whitebox(
            X[:800],
            teacher[:800],
            y[:800],
            trees_grid=(10,),
            depth_grid=(1,),
            alpha_grid=(0.0,),
            X_val=X[800:1000],
            y_val=y[800:1000],
            teacher_latent_val=teacher[800:1000],
            reference=0.1,
        )


def test_eps_warning_when_endpoints_only_and_events_are_scarce(data):
    X, y = data
    with pytest.warns(UserWarning, match="events per split"):
        compile_selected(
            X[:600],
            y[:600],
            ceiling=lambda Xf, yf: CountingCeiling(max_iter=20).fit(Xf, yf),
            reference=0.1,
            grid={"alpha": (0.0, 1.0), "trees": (160,), "depth": (2,)},
            n_boot=5,
            k_folds=2,
            min_select_events=1,
            seed=3,
        )


def test_compile_selected_is_importable_from_the_root_lazily():
    assert compileml.compile_selected is compile_selected
    missing = "no_such_thing"
    with pytest.raises(AttributeError):
        getattr(compileml, missing)
