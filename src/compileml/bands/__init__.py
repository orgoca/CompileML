"""Risk band construction: plain quantile, monotone-quantile, and
search-and-certify builders. All return a :class:`BandSpec` consumed by
``compileml.artifact.build_artifact``.
"""

from compileml.bands.builders import BandSpec, monotone_quantile_bands, quantile_bands
from compileml.bands.certified import governance_bands, semantic_bands
from compileml.bands.efficiency import band_efficiency

__all__ = [
    "BandSpec",
    "band_efficiency",
    "governance_bands",
    "monotone_quantile_bands",
    "quantile_bands",
    "semantic_bands",
]
