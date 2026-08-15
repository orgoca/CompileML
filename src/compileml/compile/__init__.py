"""Compile side of CompileML: distillation, extraction, quantization.

These modules may use numpy / scikit-learn / xgboost / lightgbm. Nothing
here is needed at decision time — that is ``compileml.runtime``'s job.
"""

from compileml.compile.distill import train_whitebox
from compileml.compile.extract import (
    ExtractedModel,
    extract_trees,
    score_float,
    validate_extraction,
)
from compileml.compile.quantize import max_depth, quantization_error_bound, quantize_model, rha

__all__ = [
    "ExtractedModel",
    "extract_trees",
    "max_depth",
    "quantization_error_bound",
    "quantize_model",
    "rha",
    "score_float",
    "train_whitebox",
    "validate_extraction",
]
