"""SQL export: the full decision pipeline as one generated query.

Emits a single SELECT that reproduces the runtime bit-for-bit on any
engine with IEEE-754 doubles and 64-bit integer division truncating
toward zero (SQLite, PostgreSQL, DuckDB, BigQuery, ...):

- tree traversal as nested CASE WHEN on float64 columns vs shortest
  round-trip threshold literals;
- integer accumulation of the artifact's ``value_micro`` values;
- clamp, ``div_rha`` display conversion, strict-``<`` band ladder;
- calibrated PD in ppm via the integer piecewise-linear table.

Unlike the COBOL export (score + band), the SQL export carries the whole
governed output: raw/clamped latent, display score, band label, and PD.
"""

from __future__ import annotations

LEAF = -2


def _lit(threshold: float) -> str:
    return repr(float(threshold))


def _q(name: str) -> str:
    """Quote an identifier (double quotes, doubling embedded quotes)."""
    return '"' + str(name).replace('"', '""') + '"'


def _tree_case(tree: dict, node: int, cols: list[str], depth: int) -> str:
    if tree["feature"][node] == LEAF:
        return str(int(tree["value_micro"][node]))
    pad = "\n" + "  " * (depth + 3)
    col = cols[tree["feature"][node]]
    left = _tree_case(tree, tree["left"][node], cols, depth + 1)
    right = _tree_case(tree, tree["right"][node], cols, depth + 1)
    return (
        f"CASE WHEN {col} <= {_lit(tree['threshold'][node])}"
        f"{pad}THEN {left}"
        f"{pad}ELSE {right} END"
    )


def _div_rha_sql(num: str, den: int) -> str:
    """Non-negative div_rha (spec §2.2) using truncating integer division."""
    return f"(2 * {num} + {den}) / (2 * {den})"


def _band_case(edges_int: list[int], labels: list[str]) -> str:
    parts = ["CASE"]
    for cutoff, label in zip(edges_int[1:-1], labels[:-1]):
        parts.append(f"    WHEN latent_int < {cutoff} THEN '{label}'")
    parts.append(f"    ELSE '{labels[-1]}'")
    parts.append("  END")
    return "\n  ".join(parts)


def _pd_case(calibration: dict | None, micro_scale: int) -> str:
    if not calibration:
        return _div_rha_sql(f"latent_micro * {1_000_000}", micro_scale)
    f = [int(v) for v in calibration["f_micro"]]
    pd = [int(v) for v in calibration["pd_ppm"]]
    mode = calibration.get("mode", "linear_int")
    parts = ["CASE"]
    parts.append(f"    WHEN latent_micro <= {f[0]} THEN {pd[0]}")
    for i in range(1, len(f)):
        if mode == "step":
            parts.append(f"    WHEN latent_micro < {f[i]} THEN {pd[i - 1]}")
        else:
            num = f"({pd[i]} - {pd[i - 1]}) * (latent_micro - {f[i - 1]})"
            den = f[i] - f[i - 1]
            parts.append(
                f"    WHEN latent_micro < {f[i]} THEN {pd[i - 1]} + {_div_rha_sql(num, den)}"
            )
    parts.append(f"    ELSE {pd[-1]}")
    parts.append("  END")
    return "\n  ".join(parts)


def export_sql(artifact: dict, *, table: str = "features", dialect: str = "ansi") -> str:
    """Render the artifact as one SQL query over ``table``.

    ``table`` must expose one float64 column per feature, named exactly as
    in ``features.names`` (imputation per the artifact's missing policy is
    the loader's job — SQL NULL comparisons would otherwise silently skip
    branches). Returns all source columns plus raw_micro, latent_micro,
    latent_int, band, and pd_ppm.

    Dialects: "ansi" (GREATEST/LEAST) or "sqlite" (two-arg MAX/MIN).
    """
    if dialect not in ("ansi", "sqlite"):
        raise ValueError("dialect must be 'ansi' or 'sqlite'")
    model = artifact["model"]
    micro_scale = int(model["micro_scale"])
    scale = int(artifact["scale"])
    ratio = micro_scale // scale
    cols = [_q(n) for n in artifact["features"]["names"]]

    greatest, least = ("GREATEST", "LEAST") if dialect == "ansi" else ("MAX", "MIN")

    tree_terms = ",\n    ".join(
        f"({_tree_case(tree, 0, cols, 0)}) AS tree_{i}" for i, tree in enumerate(model["trees"])
    )
    tree_sum = " + ".join(f"tree_{i}" for i in range(len(model["trees"])))

    return f"""-- CompileML decision artifact export
-- artifact_hash: {artifact.get('artifact_hash', 'unknown')}
-- Integer-exact pipeline: score -> clamp -> display -> band -> PD (ppm).
WITH tree_scores AS (
  SELECT
    src.*,
    {tree_terms}
  FROM {table} AS src
),
scored AS (
  SELECT *, {int(model['base_micro'])} + {tree_sum} AS raw_micro
  FROM tree_scores
),
clamped AS (
  SELECT *, {greatest}(0, {least}(raw_micro, {micro_scale})) AS latent_micro
  FROM scored
),
displayed AS (
  SELECT *, {_div_rha_sql('latent_micro', ratio)} AS latent_int
  FROM clamped
)
SELECT
  *,
  {_band_case([int(e) for e in artifact['bands']['edges_int']],
              [str(x) for x in artifact['bands']['labels']])} AS band,
  {_pd_case(artifact.get('calibration'), micro_scale)} AS pd_ppm
FROM displayed
"""
