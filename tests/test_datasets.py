"""Dataset loaders. No test here touches the network.

``load_credit_default`` is exercised against a small file written into a
temporary cache with a patched checksum, which covers the parsing, the
column selection and the failure modes without depending on GitHub being
reachable from a CI runner.
"""

import gzip
import hashlib

import numpy as np
import pytest

from compileml.datasets import credit as credit_mod
from compileml.datasets import load_credit_default, make_credit_data

HEADER = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *[f"PAY_{i}" for i in range(1, 7)],
    *[f"BILL_AMT{i}" for i in range(1, 7)],
    *[f"PAY_AMT{i}" for i in range(1, 7)],
    "DEFAULT",
]


def _write_cache(tmp_path, monkeypatch, rows=3):
    """Write a tiny credit_default.csv.gz and point the checksum at it."""
    lines = [",".join(HEADER)]
    for r in range(rows):
        # Distinct per-column values so column selection is observable.
        lines.append(",".join(str(r * 100 + c) for c in range(len(HEADER))))
    blob = gzip.compress(("\n".join(lines) + "\n").encode("utf-8"), 9, mtime=0)

    path = tmp_path / "credit_default.csv.gz"
    path.write_bytes(blob)
    monkeypatch.setattr(credit_mod, "CREDIT_DEFAULT_SHA256", hashlib.sha256(blob).hexdigest())
    return path


def test_excludes_demographics_by_default(tmp_path, monkeypatch):
    _write_cache(tmp_path, monkeypatch)
    X, y, names = load_credit_default(cache_dir=tmp_path, download=False)

    assert names == list(credit_mod.BEHAVIOURAL_FEATURES)
    assert X.shape == (3, 19)
    for banned in credit_mod.DEMOGRAPHIC_FEATURES:
        assert banned not in names


def test_include_demographics_puts_them_first(tmp_path, monkeypatch):
    _write_cache(tmp_path, monkeypatch)
    X, y, names = load_credit_default(cache_dir=tmp_path, download=False, include_demographics=True)

    assert names[:4] == list(credit_mod.DEMOGRAPHIC_FEATURES)
    assert X.shape == (3, 23)


def test_columns_are_selected_by_name_not_position(tmp_path, monkeypatch):
    _write_cache(tmp_path, monkeypatch)
    X, _, names = load_credit_default(cache_dir=tmp_path, download=False)

    # Row r, column c was written as r*100 + c. LIMIT_BAL is source column 0.
    assert X[0, names.index("LIMIT_BAL")] == 0
    assert X[1, names.index("LIMIT_BAL")] == 100
    # PAY_1 is source column 5, so row 0 holds 5 rather than 1.
    assert X[0, names.index("PAY_1")] == 5


def test_target_is_the_last_column(tmp_path, monkeypatch):
    _write_cache(tmp_path, monkeypatch)
    _, y, _ = load_credit_default(cache_dir=tmp_path, download=False)
    assert y.tolist() == [23, 123, 223]
    assert y.dtype == np.int64


def test_cold_cache_without_download_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="download=False"):
        load_credit_default(cache_dir=tmp_path, download=False)


def test_corrupt_cache_is_deleted_rather_than_trusted(tmp_path, monkeypatch):
    path = _write_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(credit_mod, "CREDIT_DEFAULT_SHA256", "0" * 64)

    with pytest.raises(FileNotFoundError):
        load_credit_default(cache_dir=tmp_path, download=False)
    assert not path.exists(), "a file failing its checksum must not survive"


def test_environment_variable_sets_the_cache(tmp_path, monkeypatch):
    _write_cache(tmp_path, monkeypatch)
    monkeypatch.setenv("COMPILEML_DATA_HOME", str(tmp_path))
    X, _, _ = load_credit_default(download=False)
    assert X.shape == (3, 19)


def test_synthetic_is_deterministic():
    a = make_credit_data(n_rows=500, seed=7)
    b = make_credit_data(n_rows=500, seed=7)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_synthetic_seed_changes_the_draw():
    a = make_credit_data(n_rows=500, seed=7)
    b = make_credit_data(n_rows=500, seed=8)
    assert not np.array_equal(a[0], b[0])


def test_synthetic_shape_and_names():
    X, y, names = make_credit_data(n_rows=250, n_features=8, seed=1)
    assert X.shape == (250, 8)
    assert y.shape == (250,)
    assert len(names) == 8
    assert names[0] == "utilization"
    assert set(np.unique(y)) <= {0, 1}


def test_synthetic_requires_the_signal_columns():
    with pytest.raises(ValueError, match="at least 5"):
        make_credit_data(n_rows=10, n_features=4)
