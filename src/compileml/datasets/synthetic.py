"""A synthetic credit-shaped generator, for when the network is not there.

``load_credit_default`` needs one download. This does not, which makes it
the right choice for a doctest, an air-gapped evaluation, or a test that
should not depend on anything outside the process.

The data-generating process is deliberately simple and deliberately
non-additive: two main effects, one genuine pairwise interaction, and a
third main effect, pushed through a logistic link. The interaction is the
point. A depth-1 whitebox cannot represent it, a depth-2 whitebox can, and
that gap is what makes the depth axis in ``sweep_whitebox`` worth sweeping
on data whose answer is known in advance.
"""

from __future__ import annotations

import numpy as np


def make_credit_data(
    n_rows: int = 40_000,
    n_features: int = 23,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate a synthetic credit-style dataset with a known structure.

    Parameters
    ----------
    n_rows:
        Number of rows.
    n_features:
        Number of features. Only the first five carry signal; the rest are
        standard normal noise, which is realistic and keeps the attribution
        cost formula honest at a defensible feature count.
    seed:
        Seeds a ``numpy.random.Generator``. The same seed gives the same
        bytes on any platform NumPy supports.

    Returns
    -------
    ``(X, y, feature_names)`` — ``X`` float64 ``(n_rows, n_features)``,
    ``y`` int, and generic ``f00…`` names for the noise columns with
    meaningful names for the five that matter.

    Notes
    -----
    The committed benchmark in ``benchmarks/run_benchmarks.py`` keeps its
    own copy of this process on purpose: it draws from a generator that is
    shared with later timing code, so routing it through this function
    would shift the random stream and move every published benchmark
    number. Do not consolidate them without regenerating the benchmark.
    """
    if n_features < 5:
        raise ValueError(f"n_features must be at least 5, got {n_features}")

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features))

    logit = (
        1.3 * X[:, 0]  # utilization
        - 1.0 * X[:, 1]  # payment ratio
        + 0.8 * X[:, 2] * X[:, 3]  # interaction: balance x limit
        + 0.5 * X[:, 4]
        - 1.2
    )
    p_default = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_rows) < p_default).astype(int)

    names = ["utilization", "payment_ratio", "balance", "credit_limit", "inquiries"]
    names += [f"f{i:02d}" for i in range(5, n_features)]

    return X, y, names
