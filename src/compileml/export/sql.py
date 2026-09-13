"""SQL export: the full decision pipeline as one generated query.

Emits a single SELECT that reproduces the runtime bit-for-bit on any
engine with IEEE-754 doubles and 64-bit integer division truncating
toward zero (SQLite, PostgreSQL, DuckDB, BigQuery, ...):

- tree traversal as nested CASE WHEN on float64 columns vs shortest
  round-trip threshold literals;
- integer accumulation of the artifact's ``value_micro`` values;
- clamp, ``div_rha`` display conversion, strict-``<`` band ladder;
- calibrated PD in ppm via the integer piecewise-linear table;
- with ``explain=True``, exact attribution, display impacts and reason codes
  (spec §7), one row out per row in.

Ranking features within a row — for the largest-remainder rounding and for
choosing reasons — is done without window functions or a row key: a
feature's rank is the number of features ahead of it, which is plain CASE
arithmetic. The query text grows with the square of the feature count, which
is comfortable at credit-model sizes.
"""

from __future__ import annotations

from compileml.export._explain import (
    LEAF,
    explain_baseline,
    reason_codes,
    require_exact,
    resolve_top_k,
    split_features,
)


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
    if len(labels) == 1:
        return f"'{labels[0]}'"
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


def _walk_case(tree: dict, node: int, cols: list[str], baselined: dict[int, float]) -> str:
    """One subset walk: features in ``baselined`` are resolved now, the rest compared live."""
    feature = tree["feature"][node]
    if feature == LEAF:
        return str(int(tree["value_micro"][node]))
    if feature in baselined:
        go_left = baselined[feature] <= float(tree["threshold"][node])
        return _walk_case(tree, tree["left" if go_left else "right"][node], cols, baselined)
    left = _walk_case(tree, tree["left"][node], cols, baselined)
    right = _walk_case(tree, tree["right"][node], cols, baselined)
    return (
        f"CASE WHEN {cols[feature]} <= {_lit(tree['threshold'][node])} THEN {left} ELSE {right} END"
    )


def _balanced_sum(terms: list[str]) -> str:
    """Sum terms as a balanced tree, so expression depth grows as log n, not n."""
    if not terms:
        return "0"
    while len(terms) > 1:
        terms = [
            f"({terms[i]} + {terms[i + 1]})" if i + 1 < len(terms) else terms[i]
            for i in range(0, len(terms), 2)
        ]
    return terms[0]


def _count(conditions: list[str]) -> str:
    return _balanced_sum([f"CASE WHEN {c} THEN 1 ELSE 0 END" for c in conditions])


def _sql_string(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _explain_ctes(artifact: dict, cols: list[str], ratio: int, top_k: int) -> tuple[str, str]:
    """CTEs computing contributions, display impacts and reason slots (spec §7).

    Working columns are prefixed ``cml_``; the reason slots are
    ``reason_neg_<s>_code`` / ``_impact`` and ``reason_pos_<s>_code`` / ``_impact``.

    Every working CTE is ``MATERIALIZED``. Later stages refer to earlier
    columns many times, and an engine that inlines CTEs re-derives each
    reference from scratch: at 23 features and 30 trees, SQLite took 492 s
    for 50 rows inlined and 0.18 s materialized. That sets the engine floor
    for ``explain=True`` — SQLite 3.35+, PostgreSQL 12+, DuckDB.
    """
    n = len(cols)
    ratio2 = 2 * ratio
    baseline = explain_baseline(artifact)
    negative, positive, eligible = reason_codes(artifact)
    cited = [j for j in range(n) if eligible[j]]

    # c2_j = 2 * d_j - isum_j, summed tree by tree over the tree's own features.
    c2_terms: list[list[str]] = [[] for _ in range(n)]
    for tree in artifact["model"]["trees"]:
        feats = split_features(tree)
        walks = [
            _walk_case(
                tree,
                0,
                cols,
                {j: baseline[j] for b, j in enumerate(feats) if (mask >> b) & 1},
            )
            for mask in range(1 << len(feats))
        ]
        for b, j in enumerate(feats):
            c2_terms[j].append(f"2 * ({walks[0]} - {walks[1 << b]})")
            for a, i in enumerate(feats):
                if i != j:
                    ma, mb = 1 << a, 1 << b
                    c2_terms[j].append(
                        f"-({walks[0]} - {walks[ma]} - {walks[mb]} + {walks[ma | mb]})"
                    )

    def c2(j: int) -> str:
        return f"cml_c2_{j}"

    def r(j: int) -> str:
        return f"cml_r_{j}"

    contrib = ",\n    ".join(f"{_balanced_sum(c2_terms[j])} AS {c2(j)}" for j in range(n))
    # Floor remainder in [0, ratio2), using only non-negative integer division.
    remainders = ",\n    ".join(
        f"CASE WHEN {c2(j)} >= 0 THEN {c2(j)} - ({c2(j)} / {ratio2}) * {ratio2}"
        f" WHEN (-{c2(j)}) - ((-{c2(j)}) / {ratio2}) * {ratio2} = 0 THEN 0"
        f" ELSE {ratio2} - ((-{c2(j)}) - ((-{c2(j)}) / {ratio2}) * {ratio2}) END AS {r(j)}"
        for j in range(n)
    )
    sum_c2 = _balanced_sum([c2(j) for j in range(n)])
    sum_r = _balanced_sum([r(j) for j in range(n)])
    target = (
        f"CASE WHEN cml_sum_c2 >= 0 THEN (2 * cml_sum_c2 + {ratio2}) / (2 * {ratio2})"
        f" ELSE -((-2 * cml_sum_c2 + {ratio2}) / (2 * {ratio2})) END"
    )

    impact_cols, rank_cols = [], []
    for j in cited:
        # Units go to the largest remainders, lower index first on ties.
        ahead = [f"{r(i)} >= {r(j)}" for i in range(j)] + [
            f"{r(i)} > {r(j)}" for i in range(j + 1, n)
        ]
        impact_cols.append(
            f"({c2(j)} - {r(j)}) / {ratio2}"
            f" + CASE WHEN {_count(ahead)} < cml_deficit THEN 1 ELSE 0 END AS cml_impact_{j}"
        )
        # Among cited features only: larger adverse contribution ranks first,
        # more negative favorable contribution ranks first, lower index on ties.
        adverse_ahead = [f"{c2(i)} >= {c2(j)}" for i in cited if i < j] + [
            f"{c2(i)} > {c2(j)}" for i in cited if i > j
        ]
        favorable_ahead = [f"{c2(i)} <= {c2(j)}" for i in cited if i < j] + [
            f"{c2(i)} < {c2(j)}" for i in cited if i > j
        ]
        rank_cols.append(f"{_count(adverse_ahead)} AS cml_adverse_rank_{j}")
        rank_cols.append(f"{_count(favorable_ahead)} AS cml_favorable_rank_{j}")

    slots = []
    for s in range(top_k):
        for direction, sign, codes in (("neg", ">", negative), ("pos", "<", positive)):
            rank = "adverse" if direction == "neg" else "favorable"
            whens = [(f"{c2(j)} {sign} 0 AND cml_{rank}_rank_{j} = {s}", j) for j in cited]
            if whens:
                code = " ".join(f"WHEN {w} THEN {_sql_string(codes[j])}" for w, j in whens)
                impact = " ".join(f"WHEN {w} THEN cml_impact_{j}" for w, j in whens)
                slots.append(f"CASE {code} ELSE NULL END AS reason_{direction}_{s + 1}_code")
                slots.append(f"CASE {impact} ELSE NULL END AS reason_{direction}_{s + 1}_impact")
            else:
                slots.append(f"NULL AS reason_{direction}_{s + 1}_code")
                slots.append(f"NULL AS reason_{direction}_{s + 1}_impact")

    working = ",\n    ".join(impact_cols + rank_cols) if impact_cols else "0 AS cml_nothing_cited"
    ctes = f"""cml_contrib AS MATERIALIZED (
  SELECT *,
    {contrib}
  FROM decided
),
cml_remainder AS MATERIALIZED (
  SELECT *,
    {remainders},
    {sum_c2} AS cml_sum_c2
  FROM cml_contrib
),
cml_deficit AS MATERIALIZED (
  SELECT *,
    {target} - (cml_sum_c2 - ({sum_r})) / {ratio2} AS cml_deficit
  FROM cml_remainder
),
cml_ranked AS MATERIALIZED (
  SELECT *,
    {working}
  FROM cml_deficit
)"""
    final = "SELECT\n  *,\n  " + ",\n  ".join(slots) + "\nFROM cml_ranked\n"
    return ctes, final


def export_sql(
    artifact: dict,
    *,
    table: str = "features",
    dialect: str = "ansi",
    explain: bool = False,
    top_k: int | None = None,
) -> str:
    """Render the artifact as one SQL query over ``table``.

    ``table`` must expose one float64 column per feature, named exactly as
    in ``features.names`` (imputation per the artifact's missing policy is
    the loader's job — SQL NULL comparisons would otherwise silently skip
    branches). Returns all source columns plus raw_micro, latent_micro,
    latent_int, band, and pd_ppm.

    Dialects: "ansi" (GREATEST/LEAST) or "sqlite" (two-arg MAX/MIN).

    With ``explain=True`` each row also carries the top ``top_k`` adverse and
    favorable reasons as ``reason_neg_<s>_code`` / ``reason_neg_<s>_impact``
    and ``reason_pos_<s>_code`` / ``reason_pos_<s>_impact`` (``NULL`` where a
    slot is empty) — the codes and display-scale integer impacts
    ``decide(..., explain=True)`` returns, in order. Working columns used to
    compute them are prefixed ``cml_``. ``top_k`` defaults to the artifact's
    own.

    Raises:
        ExportError: ``EXPLAIN_NOT_EXACT`` when attribution is not exact.
    """
    if dialect not in ("ansi", "sqlite"):
        raise ValueError("dialect must be 'ansi' or 'sqlite'")
    model = artifact["model"]
    micro_scale = int(model["micro_scale"])
    scale = int(artifact["scale"])
    ratio = micro_scale // scale
    cols = [_q(n) for n in artifact["features"]["names"]]

    greatest, least = ("GREATEST", "LEAST") if dialect == "ansi" else ("MAX", "MIN")
    if explain:
        require_exact(artifact)
        k = resolve_top_k(artifact, top_k)

    tree_terms = ",\n    ".join(
        f"({_tree_case(tree, 0, cols, 0)}) AS tree_{i}" for i, tree in enumerate(model["trees"])
    )
    tree_sum = " + ".join(f"tree_{i}" for i in range(len(model["trees"])))

    head = f"""-- CompileML decision artifact export
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
)"""
    decision = f"""
  *,
  {_band_case([int(e) for e in artifact['bands']['edges_int']],
              [str(x) for x in artifact['bands']['labels']])} AS band,
  {_pd_case(artifact.get('calibration'), micro_scale)} AS pd_ppm"""
    if not explain:
        return f"{head}\nSELECT{decision}\nFROM displayed\n"
    ctes, final = _explain_ctes(artifact, cols, ratio, k)
    return f"{head},\ndecided AS (\n  SELECT{decision}\n  FROM displayed\n),\n{ctes}\n{final}"
