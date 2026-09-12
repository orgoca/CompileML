"""The audit object: compute once, read section by section.

A fairness report is read in pieces — a committee works through outcome,
then performance, then behaviour — so the sections are addressable
individually rather than delivered as one block. ``compute_all`` populates
everything available; ``section(n)`` returns one; ``summary`` prints the
digest a reviewer reads first.

What this is not: a verdict. It produces evidence a validator can inspect,
and it will not tell you whether a model complies with anything. That
boundary is deliberate and appears in the docs as well as here.
"""

from __future__ import annotations

import numpy as np

from compileml.fairness import metrics

SECTIONS = {
    1: ("representation", "Representation and base rates"),
    2: ("score_distribution", "Score distribution"),
    3: ("approval_rates", "Approval rates and adverse impact"),
    4: ("calibration", "Calibration by group"),
    5: ("error_rates", "Error rates"),
    6: ("attribution_disparity", "Attribution disparity"),
    7: ("feature_swing", "Feature swing"),
    8: ("attribution_concentration", "Attribution concentration"),
    9: ("boundary_fragility", "Boundary fragility"),
    10: ("reason_parity", "Adverse-action reason parity"),
    11: ("counterfactual", "Counterfactual"),
}

LAYERS = {
    "Outcome fairness": (1, 2, 3),
    "Performance fairness": (4, 5),
    "Decision geometry": (6, 7, 8, 9, 10, 11),
}


class FairnessAudit:
    """Three-layer fairness audit over compiled decisions.

    Args:
        decisions: ``decide()`` payloads, one per row. Pass
            ``include_contributions=True`` when calling ``decide`` or §6 and
            §8 cannot run.
        y: Binary outcomes aligned with ``decisions``.
        protected: Protected attribute, aligned. Any hashable values.
        labels: Optional ``{value: display_name}``.
        threshold_int: Cutoff in display-scale latent units; approved below
            it. Defaults to the median score, which is a reporting
            convenience and not a policy — set it to the cutoff you actually
            use.
        artifact: Needed for §7, §8 and §11, which re-read the compiled model.
        X: Feature rows, needed for the same three sections.
        protected_feature: Name of the protected attribute *if it is a model
            input*. Leave ``None`` when it is not — §11 then reports itself
            inapplicable, which is the desired result.
    """

    def __init__(
        self,
        decisions,
        y,
        protected,
        *,
        labels=None,
        threshold_int: int | None = None,
        artifact: dict | None = None,
        X=None,
        protected_feature: str | None = None,
    ):
        self.decisions = list(decisions)
        self.y = np.asarray(y, dtype=int).reshape(-1)
        self.protected = np.asarray(protected).reshape(-1)
        self.labels = labels
        self.artifact = artifact
        self.X = None if X is None else np.asarray(X, dtype=float)
        self.protected_feature = protected_feature

        n = len(self.decisions)
        if not (n == self.y.size == self.protected.size):
            raise ValueError(
                f"decisions ({n}), y ({self.y.size}) and protected "
                f"({self.protected.size}) must be the same length"
            )
        if n == 0:
            raise ValueError("no decisions to audit")

        latent = np.array([d["latent_int"] for d in self.decisions], dtype=float)
        self.threshold_int = (
            int(threshold_int) if threshold_int is not None else int(np.median(latent))
        )
        self.feature_names = list(artifact["features"]["names"]) if artifact else None
        self.results: dict[str, dict] = {}
        self.skipped: dict[str, str] = {}

    # ------------------------------------------------------------------
    def compute_all(self, *, sections=None) -> dict:
        """Run every section whose inputs are present; record why others skip."""
        wanted = set(sections or SECTIONS)
        kw = {"labels": self.labels}

        def run(num, fn):
            if num not in wanted:
                return
            key = SECTIONS[num][0]
            try:
                self.results[key] = fn()
            except Exception as exc:  # a skipped section must say why
                self.skipped[key] = f"{type(exc).__name__}: {exc}"

        run(
            1,
            lambda: metrics.representation(
                self.y, self.protected, X=self.X, feature_names=self.feature_names, **kw
            ),
        )
        run(2, lambda: metrics.score_distribution(self.decisions, self.protected, **kw))
        run(
            3,
            lambda: metrics.approval_rates(
                self.decisions, self.protected, self.threshold_int, **kw
            ),
        )
        run(4, lambda: metrics.calibration_by_group(self.decisions, self.y, self.protected, **kw))
        run(
            5,
            lambda: metrics.error_rates(
                self.decisions, self.y, self.protected, self.threshold_int, **kw
            ),
        )
        run(
            6,
            lambda: metrics.attribution_disparity(
                self.decisions, self.protected, self._require_names(), **kw
            ),
        )
        run(
            7,
            lambda: metrics.feature_swing(
                self._require_artifact(), self._require_X(), self.protected, **kw
            ),
        )
        run(
            8,
            lambda: metrics.attribution_concentration(
                self.decisions, self.protected, self._require_names(), **kw
            ),
        )
        run(
            9,
            lambda: metrics.boundary_fragility(
                self.decisions, self.protected, self.threshold_int, **kw
            ),
        )
        run(10, lambda: metrics.reason_parity(self.decisions, self.protected, **kw))
        run(
            11,
            lambda: metrics.counterfactual(
                self._require_artifact(),
                self._require_X(),
                self.protected,
                self.protected_feature,
                **kw,
            ),
        )
        return self.results

    def section(self, number: int) -> dict:
        """One section by number, computing it on demand."""
        if number not in SECTIONS:
            raise KeyError(f"section {number} does not exist; sections are 1-11")
        key = SECTIONS[number][0]
        if key not in self.results and key not in self.skipped:
            self.compute_all(sections=[number])
        if key in self.skipped:
            raise RuntimeError(f"section {number} unavailable — {self.skipped[key]}")
        return self.results[key]

    # ------------------------------------------------------------------
    def _require_artifact(self):
        if self.artifact is None:
            raise ValueError("needs artifact=; this section re-reads the compiled model")
        return self.artifact

    def _require_X(self):
        if self.X is None:
            raise ValueError("needs X=; this section evaluates the model on feature rows")
        return self.X

    def _require_names(self):
        if self.feature_names is None:
            raise ValueError("needs artifact= to name features")
        return self.feature_names

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """The digest a reviewer reads first. Numbers, not verdicts."""
        if not self.results and not self.skipped:
            self.compute_all()
        out = ["Fairness audit", "=" * 62]

        rep = self.results.get("representation")
        appr = self.results.get("approval_rates")
        err = self.results.get("error_rates")
        frag = self.results.get("boundary_fragility")
        if rep:
            for name, g in rep["groups"].items():
                out.append(f"\n  {name}  (n={g['n']:,}  {g['share']:.1%})")
                out.append(f"    observed bad rate : {g['base_rate']:.2%}")
                if appr and name in appr["groups"]:
                    a = appr["groups"][name]
                    lo, hi = a["rate_ci"]
                    out.append(f"    approval rate     : {a['rate']:.2%}  [{lo:.2%}, {hi:.2%}]")
                if err and name in err["groups"]:
                    e = err["groups"][name]
                    out.append(f"    FPR / FNR         : {e['fpr']:.2%} / {e['fnr']:.2%}")
                if frag and name in frag["groups"]:
                    f = frag["groups"][name]
                    out.append(
                        f"    near the cutoff   : {f['share_within_band']:.1%} "
                        f"within {frag['near_band']} points"
                    )

        if appr and "adverse_impact_ratio" in appr:
            r = appr["adverse_impact_ratio"]
            window = "inside" if appr["within_four_fifths"] else "OUTSIDE"
            out.append(
                f"\n  adverse impact ratio ({appr['ratio_of']}): {r:.3f}  "
                f"— {window} the 0.80-1.25 window"
            )

        att = self.results.get("attribution_disparity")
        if att:
            out.append(
                f"\n  mean score gap decomposed exactly "
                f"(residual {att['residual']:.0f}); largest drivers:"
            )
            for row in att["by_feature"][:3]:
                out.append(f"    {row['feature']:<22} {row['share_pct']:>7.1f}% of the gap")

        rp = self.results.get("reason_parity")
        if rp and "total_variation" in rp:
            out.append(
                f"\n  adverse-action reason divergence (total variation): "
                f"{rp['total_variation']:.3f}"
            )

        if self.skipped:
            out.append("\n  not computed:")
            for key, why in self.skipped.items():
                out.append(f"    {key}: {why[:96]}")

        out.append(
            "\n  This is evidence, not a verdict. It does not certify compliance\n"
            "  with ECOA, Regulation B, or anything else."
        )
        out.append("=" * 62)
        return "\n".join(out)

    def print_summary(self) -> None:
        print(self.summary())
