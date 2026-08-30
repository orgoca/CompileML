"""The UCI *default of credit card clients* panel, as a compilable dataset.

Every worked example in the documentation starts from the same real data,
for the same reason the artifact carries its own hash: a number nobody can
reproduce is a claim, not a result. Synthetic data can demonstrate an API,
but it cannot show that a band ladder survives a genuinely lumpy score
distribution, or that reason codes read sensibly when the features have
meanings.

The panel is 30,000 Taiwanese credit-card accounts observed over six
monthly cycles in 2005, with a next-month default flag. It is small enough
to compile in seconds, entirely integer-valued, and free of missing values
— which makes it a poor test of the missing-value policy and an unusually
clean test of everything else.

**Demographics are excluded by default.** The source carries ``SEX``,
``EDUCATION``, ``MARRIAGE`` and ``AGE``. Fitting a credit model on sex or
marital status is prohibited in most consumer-lending jurisdictions, and a
library aimed at regulated risk teams should not ship examples that do it
casually. ``include_demographics=True`` returns them for anyone studying
disparity in the data itself; it is a deliberate opt-in, and it does not
make this library a fair-lending toolkit.

The file is fetched once and cached rather than shipped in the wheel. The
runtime is meant to be small enough to vendor into a constrained codebase,
and a megabyte of teaching data has no business travelling with it.

Source: Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining
techniques for the predictive accuracy of probability of default of credit
card clients.* Expert Systems with Applications, 36(2), 2473-2480.
UCI Machine Learning Repository, CC BY 4.0.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
import urllib.request
from pathlib import Path

import numpy as np

#: Release asset rather than a UCI URL. The upstream archive is a legacy
#: ``.xls`` behind a path that has already been reorganized once; this is a
#: byte-stable copy under the project's own control, converted to gzipped
#: CSV so that reading it needs nothing beyond the standard library.
CREDIT_DEFAULT_URL = (
    "https://github.com/orgoca/CompileML/releases/download/data-v1/credit_default.csv.gz"
)

#: Verified on every read. A cached file that no longer matches is deleted
#: and refetched rather than trusted, on the same reasoning as
#: ``verify_artifact``: silent corruption is worse than a loud failure.
CREDIT_DEFAULT_SHA256 = "5c78cf55667297d99fa2aa4f4583e472f7826f3e0a653f75b1cdaf3529d017ce"

#: Order is fixed. Feature indices appear in artifacts, scorecards and
#: reason codes, so a reordering here would silently invalidate every
#: committed artifact built from this dataset.
BEHAVIOURAL_FEATURES = (
    "LIMIT_BAL",
    "PAY_1",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
)

DEMOGRAPHIC_FEATURES = ("SEX", "EDUCATION", "MARRIAGE", "AGE")

TARGET = "DEFAULT"


def _cache_dir(cache_dir: str | os.PathLike | None) -> Path:
    """Resolve the cache location.

    Explicit argument wins, then ``COMPILEML_DATA_HOME``, then
    ``~/.compileml/data``. The environment variable exists so that CI and
    air-gapped installs can seed the cache without a network call.
    """
    if cache_dir is not None:
        return Path(cache_dir)
    env = os.environ.get("COMPILEML_DATA_HOME")
    if env:
        return Path(env)
    return Path.home() / ".compileml" / "data"


def _fetch(path: Path, *, download: bool) -> bytes:
    """Return the verified gzip bytes, downloading only if necessary."""
    if path.is_file():
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() == CREDIT_DEFAULT_SHA256:
            return blob
        path.unlink()  # corrupt or stale; refetch rather than trust

    if not download:
        raise FileNotFoundError(
            f"{path} is missing and download=False. Seed the cache manually, "
            f"or set COMPILEML_DATA_HOME to a directory containing "
            f"{path.name}."
        )

    with urllib.request.urlopen(CREDIT_DEFAULT_URL) as response:  # noqa: S310
        blob = response.read()

    digest = hashlib.sha256(blob).hexdigest()
    if digest != CREDIT_DEFAULT_SHA256:
        raise OSError(
            "Downloaded credit_default.csv.gz does not match its recorded "
            f"SHA-256.\n  expected {CREDIT_DEFAULT_SHA256}\n  received {digest}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return blob


def load_credit_default(
    *,
    include_demographics: bool = False,
    cache_dir: str | os.PathLike | None = None,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the UCI credit-card default panel.

    Parameters
    ----------
    include_demographics:
        Include ``SEX``, ``EDUCATION``, ``MARRIAGE`` and ``AGE``. Excluded
        by default; see the module docstring for why.
    cache_dir:
        Where to keep the downloaded file. Defaults to
        ``$COMPILEML_DATA_HOME`` or ``~/.compileml/data``.
    download:
        When False, raise instead of reaching the network if the cache is
        cold.

    Returns
    -------
    ``(X, y, feature_names)`` — ``X`` float64 of shape ``(30000, 19)``, or
    ``(30000, 23)`` with demographics; ``y`` int of shape ``(30000,)`` with
    a 22.12% base rate; ``feature_names`` in column order.

    Notes
    -----
    Every value in the source is an integer, including the currency amounts
    (New Taiwan dollars, unrounded). ``X`` is returned as float64 because
    that is what the compile side expects, but no precision is lost in the
    conversion, and the split comparisons the artifact performs are exact.
    """
    path = _cache_dir(cache_dir) / "credit_default.csv.gz"
    blob = _fetch(path, download=download)

    with gzip.open(io.BytesIO(blob), "rt", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)

    wanted = list(BEHAVIOURAL_FEATURES)
    if include_demographics:
        wanted = list(DEMOGRAPHIC_FEATURES) + wanted

    index = {name: i for i, name in enumerate(header)}
    missing = [name for name in (*wanted, TARGET) if name not in index]
    if missing:
        raise OSError(f"credit_default.csv.gz is missing columns: {missing}")

    columns = [index[name] for name in wanted]
    target = index[TARGET]

    X = np.empty((len(rows), len(columns)), dtype=np.float64)
    y = np.empty(len(rows), dtype=np.int64)
    for r, row in enumerate(rows):
        for c, column in enumerate(columns):
            X[r, c] = float(row[column])
        y[r] = int(row[target])

    return X, y, wanted
