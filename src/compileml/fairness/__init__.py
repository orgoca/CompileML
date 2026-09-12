"""Fair-lending audit over compiled decisions.

Three layers, eleven sections: what decisions were made, how well the model
predicts per group, and how it behaves locally. The first two layers are
commodity — `fairlearn` and `aif360` do them well. The third is where an
exact attribution changes the answer rather than decorating it.

Nothing here uses SHAP, and two sections are better for it: the attribution
decomposition sums to the group score gap exactly rather than approximately,
and reason parity audits the adverse-action codes an applicant would
actually be sent rather than a ranking standing in for them.

It produces evidence a validator can inspect. It does not certify compliance
with anything.
"""

from compileml.fairness.audit import LAYERS, SECTIONS, FairnessAudit
from compileml.fairness.metrics import (
    approval_rates,
    attribution_concentration,
    attribution_disparity,
    boundary_fragility,
    calibration_by_group,
    counterfactual,
    error_rates,
    feature_swing,
    model_interaction_structure,
    reason_parity,
    representation,
    score_distribution,
    wilson_interval,
)

__all__ = [
    "LAYERS",
    "SECTIONS",
    "FairnessAudit",
    "approval_rates",
    "attribution_disparity",
    "boundary_fragility",
    "calibration_by_group",
    "counterfactual",
    "model_interaction_structure",
    "error_rates",
    "feature_swing",
    "attribution_concentration",
    "reason_parity",
    "representation",
    "score_distribution",
    "wilson_interval",
]
