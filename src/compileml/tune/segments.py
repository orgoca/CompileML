"""Retention by segment, and decision agreement across a cutoff range.

A portfolio retention figure is an average, and distillation loss is almost
never uniform. It is also dominated by the easy separations — the obviously
good against the obviously bad — while a lending decision is made in a narrow
band of risk where the cutoff sits, and that band differs by segment. This
module reports what compilation cost where each segment's decisions are
actually made.

It never chooses a cutoff. A cutpoint needs its own study before deployment;
a *range* of PD is enough to say whether the compiled artifact is sound
anywhere that study could land. Ranges are policy, so they are inputs to the
report and never part of the artifact.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from compileml.runtime import decide
from compileml.runtime.calibrate import PD_SCALE, calibrate_ppm


def _gini(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")  # one class: no ranking to measure
    return 2 * float(roc_auc_score(y, score)) - 1


def segment_retention(y, teacher, latent) -> dict:
    """Gini of teacher and whitebox, retention, and rank agreement, on one set of rows."""
    y = np.asarray(y, dtype=int)
    teacher_gini = _gini(y, teacher)
    gini = _gini(y, latent)
    spearman = float(spearmanr(teacher, latent)[0]) if len(y) > 2 else float("nan")
    return {
        "n": int(len(y)),
        "bad_rate": float(y.mean()) if len(y) else float("nan"),
        "teacher_gini": teacher_gini,
        "gini": gini,
        "gini_retention_pct": (
            100 * gini / teacher_gini if teacher_gini == teacher_gini and teacher_gini > 0 else None
        ),
        "spearman_vs_teacher": spearman,
    }


def _resolve_ranges(cutoff_ranges, labels: list[str]) -> dict[str, tuple[float, float]]:
    if cutoff_ranges is None:
        return {}
    if isinstance(cutoff_ranges, dict):
        unknown = sorted(set(map(str, cutoff_ranges)) - set(labels))
        if unknown:
            raise ValueError(f"cutoff_ranges names segments that are not present: {unknown}")
        pairs = {str(k): v for k, v in cutoff_ranges.items()}
    else:
        pairs = {label: cutoff_ranges for label in labels}
    resolved = {}
    for label, pair in pairs.items():
        lo, hi = (float(v) for v in pair)
        if not 0.0 < lo < hi < 1.0:
            raise ValueError(
                f"cutoff range {pair!r} for segment {label!r} must be PD fractions with "
                "0 < low < high < 1 — for example (0.02, 0.06), not (2, 6)"
            )
        resolved[label] = (lo, hi)
    return resolved


def _agreement(pd_ppm, teacher, y, lo: float, hi: float, n_cutoffs: int) -> dict:
    """Artifact against teacher at equal approval volume, at every cutoff in the range.

    The artifact approves applicants whose emitted PD is at or below the
    cutoff. The teacher approves the same number of its own lowest-risk
    applicants, so the comparison measures ranking, not two different
    calibrations.
    """
    n = len(y)
    order = np.argsort(teacher, kind="stable")
    points = []
    for cutoff in np.linspace(lo, hi, n_cutoffs):
        by_artifact = pd_ppm <= int(round(cutoff * PD_SCALE))
        k = int(by_artifact.sum())
        by_teacher = np.zeros(n, dtype=bool)
        by_teacher[order[:k]] = True
        swapped = int((by_artifact & ~by_teacher).sum())  # equal volume: in == out
        bad_artifact = float(y[by_artifact].mean()) if k else float("nan")
        bad_teacher = float(y[by_teacher].mean()) if k else float("nan")
        points.append(
            {
                "cutoff_pd": float(cutoff),
                "approval_rate": k / n,
                "disagreement_rate": 2 * swapped / n,
                "bad_rate_approved": bad_artifact,
                "bad_rate_approved_teacher": bad_teacher,
                "bad_rate_gap": bad_artifact - bad_teacher,
            }
        )
    rates = [p["disagreement_rate"] for p in points]
    gaps = [p["bad_rate_gap"] for p in points if p["bad_rate_gap"] == p["bad_rate_gap"]]
    return {
        "max_disagreement_rate": max(rates),
        "mean_disagreement_rate": float(np.mean(rates)),
        "max_bad_rate_gap": max(gaps) if gaps else float("nan"),
        "approval_rate_range": [points[0]["approval_rate"], points[-1]["approval_rate"]],
        "points": points,
    }


def _band_edges_in_range(artifact: dict, lo: float, hi: float) -> dict:
    """Where the ladder's interior edges fall in PD, and whether any land in the range.

    An approval cutoff on a band ladder can only sit on a band edge. If no
    edge's calibrated PD falls inside the range, no cutoff inside it can be
    expressed on this ladder, however well the model ranks.
    """
    micro = int(artifact["model"]["micro_scale"])
    ratio = micro // int(artifact["scale"])
    edges = [int(e) for e in artifact["bands"]["edges_int"]][1:-1]
    edge_pd = [
        calibrate_ppm(e * ratio, artifact.get("calibration"), micro) / PD_SCALE for e in edges
    ]
    inside = [p for p in edge_pd if lo <= p <= hi]
    return {
        "edge_pds_in_range": inside,
        "edges_in_range": len(inside),
        "cutoff_expressible": bool(inside),
    }


def retention_by_segment(
    artifact: dict,
    teacher_latent,
    X,
    y,
    *,
    segments=None,
    cutoff_ranges=None,
    n_cutoffs: int = 21,
) -> dict:
    """What compilation cost, per segment and across each segment's cutoff range.

    Score ``X`` through the artifact and report, per segment:

    - **retention** — teacher and whitebox Gini, ``gini_retention_pct`` and
      Spearman agreement, on the segment's own rows;
    - **decision agreement** (when the segment has a range) — at every cutoff
      across the PD range, the share of applicants the artifact decides
      differently from the teacher at the same approval volume, and the bad
      rate each approves;
    - **band resolution** (when the segment has a range) — how many of the
      ladder's edges fall inside the range, since a cutoff can only sit on
      one.

    ``segments`` is a label per row; omit it to treat the rows as one
    segment. ``cutoff_ranges`` is PD as fractions: a mapping of segment label
    to ``(low, high)``, or a single ``(low, high)`` for every segment. A
    segment without a range gets retention only. Use holdout rows — in-sample
    retention flatters every artifact. Cutoff ranges need a calibrated
    artifact: without a calibration table the emitted "PD" is the raw score
    rescaled, and a PD range would be measured on the wrong scale.

    The report names ``worst_retention_segment`` and
    ``worst_disagreement_segment``, so an average cannot hide one segment.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    teacher = np.asarray(teacher_latent, dtype=float).reshape(-1)
    n = len(y_arr)
    if X_arr.shape[0] != n or teacher.shape[0] != n:
        raise ValueError("X, y and teacher_latent must have the same number of rows")
    groups = (
        np.asarray([str(s) for s in np.asarray(segments).reshape(-1)])
        if segments is not None
        else np.full(n, "all")
    )
    if groups.shape[0] != n:
        raise ValueError("segments must have one label per row")
    labels = sorted(set(groups.tolist()))
    ranges = _resolve_ranges(cutoff_ranges, labels)
    if ranges and not artifact.get("calibration"):
        raise ValueError(
            "cutoff ranges are PD, and this artifact has no calibration table, so the "
            "PD it emits is its raw score rescaled; build it with calibration_latent= "
            "and calibration_y= before studying cutoff ranges"
        )

    decisions = [decide(artifact, [float(v) for v in row], explain=False) for row in X_arr]
    latent = np.array([d["raw_micro"] for d in decisions], dtype=float)
    pd_ppm = np.array([d["pd_ppm"] for d in decisions], dtype=np.int64)

    report: dict = {}
    for label in labels:
        m = groups == label
        entry = segment_retention(y_arr[m], teacher[m], latent[m])
        if label in ranges:
            lo, hi = ranges[label]
            entry["cutoff_range"] = [lo, hi]
            entry["decisions"] = _agreement(pd_ppm[m], teacher[m], y_arr[m], lo, hi, n_cutoffs)
            entry["bands"] = _band_edges_in_range(artifact, lo, hi)
        else:
            entry["cutoff_range"] = entry["decisions"] = entry["bands"] = None
        report[label] = entry

    retained = {
        k: v["gini_retention_pct"] for k, v in report.items() if v["gini_retention_pct"] is not None
    }
    disagreed = {
        k: v["decisions"]["max_disagreement_rate"] for k, v in report.items() if v["decisions"]
    }
    return {
        "segments": report,
        "worst_retention_segment": min(retained, key=lambda k: retained[k]) if retained else None,
        "worst_disagreement_segment": (
            max(disagreed, key=lambda k: disagreed[k]) if disagreed else None
        ),
        "n_cutoffs": int(n_cutoffs),
    }
