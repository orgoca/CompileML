"""The other side of the retention comparison.

``gini_retention_pct`` measures distance to a teacher ceiling and can never
reveal that a plain logistic regression would have scored higher. This
subpackage supplies the floor, so both numbers can be reported together.
"""

from compileml.reference.woe import ReferenceModel, fit_reference, reference_gini

__all__ = ["ReferenceModel", "fit_reference", "reference_gini"]
