"""What the COBOL and SQL exports share when they emit reasons (spec §7).

Both exporters must answer the same questions the same way — whether the
artifact's attribution is exact, which features a tree splits on, what code a
feature cites, where the baseline sits — so the answers live in one place
rather than in two copies that could drift.
"""

from __future__ import annotations

from compileml.runtime.decide import _apply_precision

LEAF = -2

# A depth <= 2 tree splits on at most three features; that bounds the subset
# walks per tree at eight and is what makes exact attribution exact.
MAX_TREE_FEATURES = 3


class ExportError(ValueError):
    """An artifact an exporter will not render, with a stable ``code``.

    Codes (documented in ``docs/howto/deploy.md``):

    - ``EXPLAIN_NOT_EXACT`` — ``explain=True`` on an artifact whose attribution
      is not exact (whitebox depth > 2). Reason codes would not reconcile to
      the score, so none are emitted. Both exports.
    - ``REASON_CODE_NOT_ASCII`` — a reason code that is not printable ASCII.
      Mainframe character sets would not carry it unchanged; give the feature
      an ASCII ``code`` in the artifact's reason dictionary. COBOL export only.
    """

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def split_features(tree: dict) -> list[int]:
    """The distinct features a tree splits on, in index order."""
    return sorted({f for f in tree["feature"] if f != LEAF})


def require_exact(artifact: dict) -> None:
    """Refuse to emit reasons that would not reconcile to the score."""
    exact = artifact.get("runtime", {}).get("exact_attribution") is True
    trees = artifact["model"]["trees"]
    if not exact or any(len(split_features(t)) > MAX_TREE_FEATURES for t in trees):
        raise ExportError(
            "EXPLAIN_NOT_EXACT",
            "this artifact's attribution is not exact (whitebox depth > 2), so "
            "reason codes would not reconcile to the score; compile at depth <= 2 "
            "or export without explain",
        )


def resolve_top_k(artifact: dict, top_k: int | None) -> int:
    k = int(top_k if top_k is not None else artifact.get("runtime", {}).get("top_k", 5))
    if k < 1:
        raise ValueError("top_k must be at least 1 when explain=True")
    return k


def explain_baseline(artifact: dict) -> list[float]:
    """The baseline as the runtime compares it, after input precision is applied."""
    return _apply_precision(
        [float(b) for b in artifact["features"]["baseline"]],
        artifact["model"].get("input_precision", "float64"),
    )


def reason_codes(artifact: dict) -> tuple[list[str], list[str], list[bool]]:
    """Adverse and favorable code per feature, and whether it may be cited (spec §7.6).

    A suppressed feature is never cited; its codes are empty strings.
    """
    dictionary = artifact.get("reasons") or {}
    negative: list[str] = []
    positive: list[str] = []
    eligible: list[bool] = []
    for name in (str(n) for n in artifact["features"]["names"]):
        entry = dictionary.get(name) or {}
        if entry.get("suppress"):
            negative.append("")
            positive.append("")
            eligible.append(False)
            continue
        negative.append(str(entry.get("code", f"NEGATIVE_{name}")))
        positive.append(str(entry.get("code", f"POSITIVE_{name}")))
        eligible.append(True)
    return negative, positive, eligible
