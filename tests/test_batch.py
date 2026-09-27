"""The batch scorer is the runtime's score path, vectorised: same integers, every row."""

import copy

import numpy as np
import pytest

from compileml.batch import score_batch
from compileml.runtime import ArtifactError, decide, load_artifact

FIELDS = ("raw_micro", "latent_micro", "latent_int", "band_idx", "pd_ppm")


@pytest.fixture(scope="module")
def artifact():
    return load_artifact("tests/data/reference_artifact.json")


@pytest.fixture(scope="module")
def rows(artifact):
    """Ordinary rows, missing values, and values far enough out to hit both clamps."""
    p = len(artifact["features"]["names"])
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, p)) * 2
    X[::7, 0] = np.nan
    X[::11, 3] = 1e6
    X[::13, 4] = -1e6
    return X


def _agrees(art, X):
    batch = score_batch(art, X)
    for i, row in enumerate(X):
        d = decide(art, [None if v != v else float(v) for v in row], explain=False)
        assert tuple(d[f] for f in FIELDS) == tuple(int(batch[f][i]) for f in FIELDS), i


def test_bit_identical_to_decide(artifact, rows):
    _agrees(artifact, rows)


@pytest.mark.parametrize("variant", ["step", "none", "float32"])
def test_bit_identical_across_calibration_modes_and_precision(artifact, rows, variant):
    art = copy.deepcopy(artifact)
    if variant == "step":
        art["calibration"]["mode"] = "step"
    elif variant == "none":
        art["calibration"] = None
    else:
        art["model"]["input_precision"] = "float32"
    _agrees(art, rows)


def test_reject_policy_refuses_missing_values(artifact, rows):
    art = copy.deepcopy(artifact)
    art["features"]["missing_policy"] = "reject"
    with pytest.raises(ArtifactError, match="reject"):
        score_batch(art, rows)
    score_batch(art, np.nan_to_num(rows))  # no missing values: fine


def test_wrong_width_is_refused(artifact, rows):
    with pytest.raises(ArtifactError, match="feature columns"):
        score_batch(artifact, rows[:, :-1])
