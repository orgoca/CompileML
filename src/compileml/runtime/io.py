"""Artifact loading, canonical hashing, and verification (ARTIFACT_SPEC.md §9).

The hash covers the canonical JSON serialization of the whole document
minus the ``artifact_hash`` field itself. Loading verifies by default:
an artifact that does not match its hash is refused, not repaired.
"""

from __future__ import annotations

import hashlib
import json
from os import PathLike

ARTIFACT_TYPE = "compileml.decision_artifact"
SCHEMA_VERSION = 2


class ArtifactError(ValueError):
    """Raised for malformed, unsupported, or tampered artifacts."""


def canonical_hash(artifact: dict) -> str:
    """SHA-256 hex digest of the canonical serialization (spec §9)."""
    body = {k: v for k, v in artifact.items() if k != "artifact_hash"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_artifact(artifact: dict) -> bool:
    """True iff the stored hash matches the recomputed canonical hash."""
    stored = artifact.get("artifact_hash")
    return bool(stored) and stored == canonical_hash(artifact)


def validate_structure(artifact: dict) -> None:
    """Cheap structural checks; raises ArtifactError on violation."""
    if artifact.get("artifact_type") != ARTIFACT_TYPE:
        raise ArtifactError(
            f"artifact_type must be {ARTIFACT_TYPE!r}, got {artifact.get('artifact_type')!r}"
        )
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(
            f"schema_version must be {SCHEMA_VERSION}, got {artifact.get('schema_version')!r}"
        )
    scale = artifact["scale"]
    micro = artifact["model"]["micro_scale"]
    if micro % scale != 0:
        raise ArtifactError("micro_scale must be an integer multiple of scale")
    edges = artifact["bands"]["edges_int"]
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ArtifactError("bands.edges_int must be strictly increasing")
    if len(artifact["bands"]["labels"]) != len(edges) - 1:
        raise ArtifactError("bands.labels length must equal len(edges_int) - 1")
    names = artifact["features"]["names"]
    baseline = artifact["features"]["baseline"]
    if len(names) != len(baseline):
        raise ArtifactError("features.names and features.baseline must have equal length")
    cal = artifact.get("calibration")
    if cal:
        f, pd = cal["f_micro"], cal["pd_ppm"]
        if len(f) != len(pd) or not f:
            raise ArtifactError("calibration.f_micro and pd_ppm must be equal-length, non-empty")
        if any(b <= a for a, b in zip(f, f[1:])):
            raise ArtifactError("calibration.f_micro must be strictly increasing")
        if any(b < a for a, b in zip(pd, pd[1:])):
            raise ArtifactError("calibration.pd_ppm must be non-decreasing")


def load_artifact(path: str | PathLike, *, verify: bool = True) -> dict:
    """Load an artifact from JSON, verifying hash and structure by default."""
    with open(path, encoding="utf-8") as f:
        artifact = json.load(f)
    validate_structure(artifact)
    if verify and not verify_artifact(artifact):
        raise ArtifactError(
            "artifact_hash mismatch: the document does not match its stored hash "
            "(tampered, truncated, or re-serialized non-canonically)"
        )
    return artifact
