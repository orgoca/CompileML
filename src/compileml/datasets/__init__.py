"""Datasets the documentation, examples and tests compile against.

Two functions, for two situations. ``load_credit_default`` fetches a real
credit panel once and caches it — the right default for anything a reader
is meant to reproduce. ``make_credit_data`` generates a synthetic set with
a known structure and needs no network at all.
"""

from compileml.datasets.credit import (
    CREDIT_DEFAULT_SHA256,
    CREDIT_DEFAULT_URL,
    load_credit_default,
)
from compileml.datasets.synthetic import make_credit_data

__all__ = [
    "CREDIT_DEFAULT_SHA256",
    "CREDIT_DEFAULT_URL",
    "load_credit_default",
    "make_credit_data",
]
