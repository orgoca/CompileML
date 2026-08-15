"""Visualization of decision payloads (optional extra: ``compileml[viz]``).

Design rule: plots draw ``decide()`` payloads — the deployed integers —
and never recompute them. The matplotlib renderers load lazily so the
base install stays lean; ``waterfall_svg`` needs no plotting stack at all.
"""

from compileml.viz.svg import waterfall_svg

_MPL_FUNCTIONS = {"waterfall", "decision_drivers", "band_drivers", "band_ladder"}

__all__ = ["band_drivers", "band_ladder", "decision_drivers", "waterfall", "waterfall_svg"]


def __getattr__(name):
    if name in _MPL_FUNCTIONS:
        try:
            from compileml.viz import plots
        except ImportError as exc:  # matplotlib missing
            raise ImportError(
                f"compileml.viz.{name} requires matplotlib — install the viz extra: "
                "pip install compileml[viz]"
            ) from exc
        return getattr(plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
