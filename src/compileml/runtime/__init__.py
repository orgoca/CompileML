"""CompileML decision runtime — standard library only.

This subpackage is the deployable half of CompileML: given a compiled
decision artifact (a JSON document conforming to docs/ARTIFACT_SPEC.md),
it scores, bands, calibrates, and explains using nothing outside the
Python standard library. That property is enforced by a unit test.
"""

from compileml.runtime.bands import band_index, band_label
from compileml.runtime.calibrate import PD_SCALE, calibrate_ppm
from compileml.runtime.decide import decide
from compileml.runtime.explain import (
    contributions_half_micro,
    display_impacts,
    format_reasons,
)
from compileml.runtime.io import (
    ARTIFACT_TYPE,
    SCHEMA_VERSION,
    ArtifactError,
    canonical_hash,
    load_artifact,
    validate_structure,
    verify_artifact,
)
from compileml.runtime.score import LEAF, latent_from_raw, score_micro

__all__ = [
    "ARTIFACT_TYPE",
    "SCHEMA_VERSION",
    "PD_SCALE",
    "LEAF",
    "ArtifactError",
    "band_index",
    "band_label",
    "calibrate_ppm",
    "canonical_hash",
    "contributions_half_micro",
    "decide",
    "display_impacts",
    "format_reasons",
    "latent_from_raw",
    "load_artifact",
    "score_micro",
    "validate_structure",
    "verify_artifact",
]
