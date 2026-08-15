"""Band builder tests: quantile, monotone-quantile, and search-and-certify."""

import numpy as np
import pytest

from compileml.bands import (
    BandSpec,
    governance_bands,
    monotone_quantile_bands,
    quantile_bands,
    semantic_bands,
)

RNG = np.random.default_rng(11)
N = 12_000
LATENT = np.clip(RNG.beta(2.0, 5.0, N), 0, 1)
Y = (RNG.random(N) < LATENT).astype(int)  # bad rate rises with latent


def test_quantile_bands_basic():
    spec = quantile_bands(LATENT, n_bands=10)
    assert isinstance(spec, BandSpec)
    assert spec.n_bands == 10
    assert all(b > a for a, b in zip(spec.edges, spec.edges[1:]))
    assert sum(spec.metadata["counts"]) == N
    # quantile bands should be roughly equal-volume
    assert min(spec.metadata["counts"]) > N / 10 * 0.5


def test_monotone_quantile_bands():
    spec = monotone_quantile_bands(LATENT, Y, n_bands=10)
    assert spec.n_bands == 10
    smoothed = spec.metadata["smoothed_bad_rate"]
    assert all(b >= a for a, b in zip(smoothed, smoothed[1:]))
    # On this construction the empirical rates should already be monotone-ish
    emp = spec.metadata["empirical_bad_rate"]
    assert emp[-1] > emp[0]


def test_monotone_quantile_merge_path():
    # Noisy outcomes on a narrow latent range force empirical inversions.
    rng = np.random.default_rng(3)
    flat_latent = np.clip(0.5 + 0.01 * rng.standard_normal(4000), 0, 1)
    noisy_y = rng.integers(0, 2, 4000)
    spec = monotone_quantile_bands(
        flat_latent, noisy_y, n_bands=10, allow_merge=True, merge_eps=0.001
    )
    assert spec.n_bands < 10  # merges happened
    assert spec.metadata["merges"]
    violations = np.diff(spec.metadata["empirical_bad_rate"])
    assert np.min(violations) >= -0.001 - 1e-12


# Plateau data: five true risk classes. Within a plateau the latent has no
# residual rank power (jitter is independent of y), so the statistically
# supported band count is exactly five.
_rng_p = np.random.default_rng(5)
_PLATEAU_PD = [0.08, 0.20, 0.35, 0.55, 0.78]
_plateau_idx = _rng_p.integers(0, 5, 15_000)
PLATEAU_LATENT = np.array([0.10, 0.28, 0.46, 0.64, 0.85])[_plateau_idx] + _rng_p.uniform(
    -0.01, 0.01, 15_000
)
PLATEAU_Y = (_rng_p.random(15_000) < np.array(_PLATEAU_PD)[_plateau_idx]).astype(int)


def test_semantic_bands_discovers_true_band_count():
    # eps_auc is the strictness knob: proto bands straddling the sharp plateau
    # jumps retain ~0.53 within-band AUC after dilution, so discovery needs
    # eps_auc >= ~0.05 here. At the default 0.02 the honest answer is 1 band.
    spec = semantic_bands(
        PLATEAU_LATENT,
        PLATEAU_Y,
        max_bands=10,
        min_band_size=400,
        eps_auc_search=0.05,
        eps_auc_cert=0.05,
        n_boot_cert=60,
    )
    assert spec.n_bands == 5
    meta = spec.metadata
    pds = meta["band_pd"]
    assert all(b > a for a, b in zip(pds, pds[1:]))  # separated implies increasing
    for est, true in zip(pds, _PLATEAU_PD):
        assert est == pytest.approx(true, abs=0.03)
    assert len(meta["band_pd_ci"]) == 5
    assert len(meta["band_auc_ci"]) == 5
    # Jeffreys CI separation held at the chosen delta_sep
    assert all(m >= meta["params"]["delta_sep"] for m in meta["sep_margins"])
    assert "residual_rank_present" in meta["flags"]


def test_semantic_bands_strict_eps_collapses_honestly():
    # Same data, default strictness: straddling protos exceed the residual
    # rank cap, so the algorithm refuses to certify discrete classes.
    spec = semantic_bands(
        PLATEAU_LATENT, PLATEAU_Y, max_bands=10, min_band_size=400, n_boot_cert=40
    )
    assert spec.n_bands == 1
    assert spec.metadata["flags"]["no_discrete_classes"]


def test_semantic_bands_refuse_to_invent_structure():
    # Pure-noise outcomes: the honest answer is one band, flagged as such.
    rng = np.random.default_rng(17)
    noise_y = rng.integers(0, 2, len(PLATEAU_LATENT))
    spec = semantic_bands(
        PLATEAU_LATENT,
        noise_y,
        max_bands=10,
        min_band_size=400,
        eps_auc_search=0.05,
        n_boot_cert=40,
    )
    assert spec.n_bands == 1
    assert spec.metadata["flags"]["no_discrete_classes"]


def test_governance_bands_ttest_separation():
    spec = governance_bands(
        PLATEAU_LATENT,
        PLATEAU_Y,
        max_bands=10,
        min_band_size=800,
        proto_min_size=300,
        alpha_sep=0.001,
        min_delta_pd=0.05,
    )
    assert 5 <= spec.n_bands <= 7  # deterministic on this data; greedy merges may
    meta = spec.metadata  # keep a legitimate mixture band or two
    pds = meta["band_pd"]
    assert all(b >= a for a, b in zip(pds, pds[1:]))  # monotone enforced
    assert all(p <= meta["params"]["alpha_sep"] for p in meta["adjacent_p_ttest"])
    assert all(d >= meta["params"]["min_delta_pd"] for d in meta["adjacent_delta_pd"])
    assert all(a <= meta["params"]["auc_max_within"] + 1e-9 for a in meta["band_auc"])


def test_governance_bands_refuse_noise():
    rng = np.random.default_rng(29)
    spec = governance_bands(
        PLATEAU_LATENT, rng.integers(0, 2, len(PLATEAU_LATENT)), max_bands=10, min_band_size=800
    )
    assert spec.n_bands == 1
    assert spec.metadata["flags"]["no_discrete_classes"]


def test_semantic_bands_rejects_tiny_samples():
    with pytest.raises(ValueError, match="not enough samples"):
        semantic_bands(LATENT[:100], Y[:100], min_band_size=300)


def test_builders_are_deterministic():
    a = semantic_bands(
        PLATEAU_LATENT,
        PLATEAU_Y,
        max_bands=6,
        min_band_size=500,
        eps_auc_search=0.05,
        n_boot_cert=40,
    )
    b = semantic_bands(
        PLATEAU_LATENT,
        PLATEAU_Y,
        max_bands=6,
        min_band_size=500,
        eps_auc_search=0.05,
        n_boot_cert=40,
    )
    assert a.edges == b.edges
    assert a.metadata == b.metadata  # no timestamps: identical inputs, identical spec
