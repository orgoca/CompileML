"""COBOL export: the decision artifact as a native mainframe scorer.

Emits a self-contained COBOL program (``>>SOURCE FORMAT FREE``; compiles
under GnuCOBOL and Enterprise COBOL 6+) that reproduces the runtime's
integer pipeline bit-for-bit:

- leaf accumulation uses the artifact's ``value_micro`` integers verbatim
  (``ADD 21077 TO F-ACCUM-MICRO``) — nothing is re-rounded at export time;
- the display conversion is the spec §2.2 ``div_rha`` formula in integer
  ``COMPUTE`` (truncating division equals floor for non-negative values);
- the band ladder uses strict ``<`` against the interior integer edges,
  which is exactly ``bisect_right`` (spec §5);
- the calibrated PD walks the artifact's integer table with the same
  first-match rule and ``div_rha`` interpolation as the runtime (spec §6).
  The table's numerators are never negative — ``f_micro`` is strictly
  increasing and ``pd_ppm`` non-decreasing, both enforced on load — so the
  non-negative ``div_rha`` form is exact;
- with ``explain=True``, exact attribution, display impacts and reason codes
  (spec §7). Each tree's features and the artifact's baseline are known when
  the program is generated, so every comparison against a baseline value is
  resolved then: each of a tree's subset walks becomes a short, fixed IF tree
  over the real inputs, and the rest is integer arithmetic and small loops.

Feature inputs are declared ``COMP-2`` (IEEE binary64 under GnuCOBOL and
Enterprise COBOL with IEEE arithmetic). Threshold literals are emitted in
shortest round-trip form, so the compiled comparison sees the identical
float64 the Python runtime sees. For decimal-arithmetic targets, build the
artifact with quantized thresholds (``build_artifact(threshold_decimals=…)``)
so every runtime, Python included, compares identical values (spec §11).
"""

from __future__ import annotations

import re

from compileml.export._explain import (
    LEAF,
    ExportError,
    explain_baseline,
    reason_codes,
    require_exact,
    resolve_top_k,
    split_features,
)

# Working-storage names the program declares itself. A feature whose
# sanitized name would collide with one of these gets a suffix instead.
RESERVED = {"F-ACCUM-MICRO", "F-LATENT-MICRO", "F-LATENT-INT", "F-PD-STEP", "F-PD-PPM"}


def _cobol_name(name: str, used: set[str]) -> str:
    """Sanitize a feature name into a unique COBOL identifier (max 30 chars)."""
    base = re.sub(r"[^A-Z0-9]+", "-", name.upper()).strip("-") or "FEATURE"
    candidate = f"F-{base}"[:30].rstrip("-")
    n = 2
    while candidate in used:
        suffix = f"-{n}"
        candidate = (f"F-{base}"[: 30 - len(suffix)]).rstrip("-") + suffix
        n += 1
    used.add(candidate)
    return candidate


def _literal(threshold: float) -> str:
    """Shortest round-trip float literal, COBOL-style exponent."""
    text = repr(float(threshold)).replace("e", "E")
    if "E" in text:
        # COBOL floating literals require a decimal point in the mantissa
        # ('1E-05' is invalid; '1.0E-05' is not).
        mantissa, exponent = text.split("E")
        if "." not in mantissa:
            mantissa += ".0"
        text = f"{mantissa}E{exponent}"
    return text


def _emit_tree(tree: dict, names: list[str], indent: int) -> list[str]:
    lines: list[str] = []

    def recurse(node: int, depth: int) -> None:
        pad = " " * (indent + 4 * depth)
        if tree["feature"][node] == LEAF:
            value = int(tree["value_micro"][node])
            if value >= 0:
                lines.append(f"{pad}ADD {value} TO F-ACCUM-MICRO")
            else:
                lines.append(f"{pad}SUBTRACT {-value} FROM F-ACCUM-MICRO")
            return
        feat = names[tree["feature"][node]]
        lines.append(f"{pad}IF {feat} <= {_literal(tree['threshold'][node])}")
        recurse(tree["left"][node], depth + 1)
        lines.append(f"{pad}ELSE")
        recurse(tree["right"][node], depth + 1)
        lines.append(f"{pad}END-IF")

    recurse(0, 0)
    return lines


def _emit_walk(
    tree: dict, names: list[str], baselined: dict[int, float], target: str, indent: int
) -> list[str]:
    """One subset walk: features in ``baselined`` are resolved now, the rest compared live."""
    lines: list[str] = []

    def recurse(node: int, depth: int) -> None:
        pad = " " * (indent + 4 * depth)
        feature = tree["feature"][node]
        if feature == LEAF:
            lines.append(f"{pad}MOVE {int(tree['value_micro'][node])} TO {target}")
        elif feature in baselined:
            go_left = baselined[feature] <= float(tree["threshold"][node])
            recurse(tree["left"][node] if go_left else tree["right"][node], depth)
        else:
            lines.append(f"{pad}IF {names[feature]} <= {_literal(tree['threshold'][node])}")
            recurse(tree["left"][node], depth + 1)
            lines.append(f"{pad}ELSE")
            recurse(tree["right"][node], depth + 1)
            lines.append(f"{pad}END-IF")

    recurse(0, 0)
    return lines


def _emit_tree_attribution(
    index: int, tree: dict, names: list[str], baseline: list[float]
) -> list[str]:
    """Spec §7 for one tree: its subset walks, main effects and pair interactions."""
    feats = split_features(tree)
    k = len(feats)
    lines = [f"ATTR-TREE-{index:04d}."]
    for mask in range(1 << k):
        baselined = {j: baseline[j] for b, j in enumerate(feats) if (mask >> b) & 1}
        lines += _emit_walk(tree, names, baselined, f"X-WALK({mask + 1})", indent=4)
    for b, j in enumerate(feats):
        lines.append(
            f"    COMPUTE X-D({j + 1}) = X-D({j + 1}) + X-WALK(1) - X-WALK({(1 << b) + 1})"
        )
    for a in range(k):
        for b in range(a + 1, k):
            ma, mb = 1 << a, 1 << b
            lines += [
                "    COMPUTE X-INTER = X-WALK(1)"
                f" - X-WALK({ma + 1}) - X-WALK({mb + 1}) + X-WALK({(ma | mb) + 1})",
                f"    ADD X-INTER TO X-ISUM({feats[a] + 1})",
                f"    ADD X-INTER TO X-ISUM({feats[b] + 1})",
            ]
    lines.append("    CONTINUE.")
    return lines


def _cobol_string(text: str) -> str:
    if not text:
        return "SPACES"  # a zero-length literal is not valid COBOL
    return "'" + text.replace("'", "''") + "'"


def _reason_codes(artifact: dict) -> tuple[list[str], list[str], list[int]]:
    """The shared reason-code table, with COBOL's printable-ASCII rule applied."""
    negative, positive, eligible = reason_codes(artifact)
    names = [str(n) for n in artifact["features"]["names"]]
    for name, neg, pos, ok in zip(names, negative, positive, eligible):
        for code in (neg, pos) if ok else ():
            if not (code.isascii() and code.isprintable()):
                raise ExportError(
                    "REASON_CODE_NOT_ASCII",
                    f"reason code {code!r} for feature {name!r} is not printable ASCII; "
                    "give the feature an ASCII 'code' in the reason dictionary",
                )
    return negative, positive, [int(e) for e in eligible]


def _emit_value_table(name: str, pic: str, values: list[str]) -> list[str]:
    lines = [f"01 {name}-VALUES."]
    lines += [f"   05 FILLER PIC {pic} VALUE {v}." for v in values]
    lines += [
        f"01 {name}-TABLE REDEFINES {name}-VALUES.",
        f"   05 {name} PIC {pic} OCCURS {len(values)}.",
    ]
    return lines


def _emit_explain(artifact: dict, n: int, ratio: int, top_k: int) -> list[str]:
    """Per-feature contributions, display impacts (§7.5) and reason selection (§7.6)."""
    ratio2 = 2 * ratio
    lines = ["EXPLAIN-ONE.", "    INITIALIZE X-FEATURE-STATE", "    INITIALIZE REASONS"]
    for i, tree in enumerate(artifact["model"]["trees"]):
        if split_features(tree):
            lines.append(f"    PERFORM ATTR-TREE-{i + 1:04d}")
    lines += [
        "    MOVE 0 TO X-SUMC2",
        "    MOVE 0 TO X-SUMQ",
        f"    PERFORM VARYING X-J FROM 1 BY 1 UNTIL X-J > {n}",
        "        COMPUTE X-C2(X-J) = 2 * X-D(X-J) - X-ISUM(X-J)",
        "        ADD X-C2(X-J) TO X-SUMC2",
        "        *> floor division: COBOL truncates toward zero, so step down once",
        f"        COMPUTE X-Q(X-J) = X-C2(X-J) / {ratio2}",
        f"        COMPUTE X-R(X-J) = X-C2(X-J) - X-Q(X-J) * {ratio2}",
        "        IF X-R(X-J) < 0",
        "            SUBTRACT 1 FROM X-Q(X-J)",
        f"            ADD {ratio2} TO X-R(X-J)",
        "        END-IF",
        "        ADD X-Q(X-J) TO X-SUMQ",
        "        MOVE X-Q(X-J) TO X-IMPACT(X-J)",
        "    END-PERFORM",
        "    *> display target: div_rha(sum of c2, 2 * ratio), signed (spec 7.5)",
        "    IF X-SUMC2 >= 0",
        f"        COMPUTE X-TARGET = (2 * X-SUMC2 + {ratio2}) / (2 * {ratio2})",
        "    ELSE",
        f"        COMPUTE X-TARGET = (0 - 2 * X-SUMC2 + {ratio2}) / (2 * {ratio2})",
        "        COMPUTE X-TARGET = 0 - X-TARGET",
        "    END-IF",
        "    COMPUTE X-DEFICIT = X-TARGET - X-SUMQ",
        "    IF X-DEFICIT > 0",
        "        PERFORM GIVE-ONE-UNIT X-DEFICIT TIMES",
        "    END-IF",
        f"    PERFORM PICK-ADVERSE {top_k} TIMES",
        f"    PERFORM PICK-FAVORABLE {top_k} TIMES",
        "    CONTINUE.",
        "",
        "GIVE-ONE-UNIT.",
        "    *> largest remainder first, lower feature index on ties",
        "    MOVE 0 TO X-BEST",
        f"    PERFORM VARYING X-J FROM 1 BY 1 UNTIL X-J > {n}",
        "        IF X-GIVEN(X-J) = 0",
        "            IF X-BEST = 0",
        "                MOVE X-J TO X-BEST",
        "            ELSE",
        "                IF X-R(X-J) > X-R(X-BEST)",
        "                    MOVE X-J TO X-BEST",
        "                END-IF",
        "            END-IF",
        "        END-IF",
        "    END-PERFORM",
        "    IF X-BEST > 0",
        "        MOVE 1 TO X-GIVEN(X-BEST)",
        "        ADD 1 TO X-IMPACT(X-BEST)",
        "    END-IF",
        "    CONTINUE.",
        "",
    ]
    for para, sign, cmp, table, slot in (
        ("PICK-ADVERSE", ">", ">", "X-NEG-CODE", "NEG"),
        ("PICK-FAVORABLE", "<", "<", "X-POS-CODE", "POS"),
    ):
        lines += [
            f"{para}.",
            "    MOVE 0 TO X-BEST",
            f"    PERFORM VARYING X-J FROM 1 BY 1 UNTIL X-J > {n}",
            f"        IF X-ELIGIBLE(X-J) = 1 AND X-TAKEN(X-J) = 0 AND X-C2(X-J) {sign} 0",
            "            IF X-BEST = 0",
            "                MOVE X-J TO X-BEST",
            "            ELSE",
            f"                IF X-C2(X-J) {cmp} X-C2(X-BEST)",
            "                    MOVE X-J TO X-BEST",
            "                END-IF",
            "            END-IF",
            "        END-IF",
            "    END-PERFORM",
            "    IF X-BEST > 0",
            "        MOVE 1 TO X-TAKEN(X-BEST)",
            f"        ADD 1 TO REASON-{slot}-COUNT",
            f"        MOVE {table}(X-BEST) TO REASON-{slot}-CODE(REASON-{slot}-COUNT)",
            f"        MOVE X-IMPACT(X-BEST) TO REASON-{slot}-IMPACT(REASON-{slot}-COUNT)",
            "    END-IF",
            "    CONTINUE.",
            "",
        ]
    return lines


def _emit_calibration(calibration: dict | None, micro_scale: int) -> list[str]:
    """The spec §6 calibration as one EVALUATE, mirroring the SQL export's CASE."""
    lines = ["CALIBRATE-PD."]
    if not calibration:
        lines += [
            "    *> no calibration table: div_rha(latent_micro * 1000000, micro_scale)",
            "    COMPUTE F-PD-PPM =",
            f"        (2 * F-LATENT-MICRO * 1000000 + {micro_scale}) / (2 * {micro_scale})",
            "    CONTINUE.",
        ]
        return lines

    f = [int(v) for v in calibration["f_micro"]]
    pd = [int(v) for v in calibration["pd_ppm"]]
    mode = calibration.get("mode", "linear_int")
    lines += [
        f"    *> spec 6: {len(f)}-point table, mode {mode}",
        "    EVALUATE TRUE",
        f"        WHEN F-LATENT-MICRO <= {f[0]}",
        f"            MOVE {pd[0]} TO F-PD-PPM",
    ]
    for i in range(1, len(f)):
        lines.append(f"        WHEN F-LATENT-MICRO < {f[i]}")
        if mode == "step":
            lines.append(f"            MOVE {pd[i - 1]} TO F-PD-PPM")
            continue
        delta, den = pd[i] - pd[i - 1], f[i] - f[i - 1]
        # The quotient is stored before it is added, so no compiler's rules
        # for fractional intermediates can move the rounding.
        lines += [
            "            COMPUTE F-PD-STEP =",
            f"                (2 * {delta} * (F-LATENT-MICRO - {f[i - 1]}) + {den})"
            f" / (2 * {den})",
            f"            COMPUTE F-PD-PPM = {pd[i - 1]} + F-PD-STEP",
        ]
    lines += [
        "        WHEN OTHER",
        f"            MOVE {pd[-1]} TO F-PD-PPM",
        "    END-EVALUATE",
        "    CONTINUE.",
    ]
    return lines


def export_cobol(
    artifact: dict,
    *,
    program_id: str = "CMLSCORE",
    driver_rows: list | None = None,
    explain: bool = False,
    top_k: int | None = None,
) -> str:
    """Render the artifact's score, band and calibrated PD as a COBOL program.

    The generated program reads features from WORKING-STORAGE (integration
    point: MOVE caller values in, or adapt to a LINKAGE SECTION), then leaves
    ``F-LATENT-INT`` (display-scale score), ``FINAL-BAND`` and ``F-PD-PPM``
    (calibrated PD, parts per million) populated.

    With ``driver_rows`` (a list of feature rows), the program becomes a
    parity harness instead: it scores every row and DISPLAYs
    ``latent_int band pd_ppm`` one row per line — compile it, run it, and diff
    the output against the Python runtime. CI does exactly that under GnuCOBOL.

    With ``explain=True`` the program also computes exact attribution and
    leaves the top ``top_k`` adverse and favorable reasons in
    ``REASON-NEG-CODE(i)`` / ``REASON-NEG-IMPACT(i)`` and
    ``REASON-POS-CODE(i)`` / ``REASON-POS-IMPACT(i)``, with
    ``REASON-NEG-COUNT`` and ``REASON-POS-COUNT`` saying how many are filled.
    Codes and display-scale integer impacts match
    ``decide(..., explain=True)`` exactly; message text stays with the
    institution's letter templates. ``top_k`` defaults to the artifact's own.
    The driver harness then also prints each reason.

    Raises:
        ExportError: ``EXPLAIN_NOT_EXACT`` or ``REASON_CODE_NOT_ASCII`` —
            see :class:`ExportError`.
    """
    model = artifact["model"]
    micro_scale = int(model["micro_scale"])
    scale = int(artifact["scale"])
    ratio = micro_scale // scale
    edges_int = [int(e) for e in artifact["bands"]["edges_int"]]
    labels = [str(x) for x in artifact["bands"]["labels"]]
    feature_names = artifact["features"]["names"]

    used: set[str] = set(RESERVED)
    cobol_names = [_cobol_name(n, used) for n in feature_names]
    label_width = max(len(x) for x in labels)
    n_features = len(feature_names)

    if explain:
        require_exact(artifact)
        k = resolve_top_k(artifact, top_k)
        neg_codes, pos_codes, eligible = _reason_codes(artifact)
        code_width = max([len(c) for c in neg_codes + pos_codes] + [1])
        baseline = explain_baseline(artifact)

    out: list[str] = []
    push = out.append
    # Seven leading spaces: compilers parse fixed-format until the directive
    # takes effect, and a directive starting in column 1 puts text in the
    # indicator column (column 7). Area-B placement is accepted by both
    # GnuCOBOL and Enterprise COBOL.
    push("       >>SOURCE FORMAT FREE")
    push("IDENTIFICATION DIVISION.")
    push(f"PROGRAM-ID. {program_id}.")
    push("*> ------------------------------------------------------------")
    push("*> CompileML decision artifact export (score + band + PD).")
    push(f"*> artifact_hash: {artifact.get('artifact_hash', 'unknown')}")
    push(f"*> micro_scale: {micro_scale}   display scale: {scale}")
    push("*> Integer-exact: leaf values below are the artifact's integers.")
    push("*> ------------------------------------------------------------")
    push("DATA DIVISION.")
    push("WORKING-STORAGE SECTION.")
    push("01 F-ACCUM-MICRO      PIC S9(15) COMP-5 VALUE 0.")
    push("01 F-LATENT-MICRO     PIC S9(15) COMP-5 VALUE 0.")
    push("01 F-LATENT-INT       PIC S9(9)  COMP-5 VALUE 0.")
    push(f"01 FINAL-BAND         PIC X({label_width})   VALUE SPACES.")
    push("01 F-PD-STEP          PIC S9(9)  COMP-5 VALUE 0.")
    push("01 F-PD-PPM           PIC S9(9)  COMP-5 VALUE 0.")
    if explain:
        push("*> ---- explanation (spec 7) ----")
        push("01 X-J                PIC S9(9)  COMP-5 VALUE 0.")
        push("01 X-BEST             PIC S9(9)  COMP-5 VALUE 0.")
        push("01 X-INTER            PIC S9(18) COMP-5 VALUE 0.")
        push("01 X-SUMC2            PIC S9(18) COMP-5 VALUE 0.")
        push("01 X-SUMQ             PIC S9(18) COMP-5 VALUE 0.")
        push("01 X-TARGET           PIC S9(18) COMP-5 VALUE 0.")
        push("01 X-DEFICIT          PIC S9(18) COMP-5 VALUE 0.")
        push("01 X-WALKS.")
        push("   05 X-WALK          PIC S9(15) COMP-5 OCCURS 8.")
        push("01 X-FEATURE-STATE.")
        push(f"   05 X-FEATURE OCCURS {n_features}.")
        for field, pic in (
            ("X-D", "S9(18) COMP-5"),
            ("X-ISUM", "S9(18) COMP-5"),
            ("X-C2", "S9(18) COMP-5"),
            ("X-Q", "S9(18) COMP-5"),
            ("X-R", "S9(18) COMP-5"),
            ("X-IMPACT", "S9(9) COMP-5"),
            ("X-GIVEN", "9"),
            ("X-TAKEN", "9"),
        ):
            push(f"      10 {field:<14} PIC {pic}.")
        out.extend(
            _emit_value_table(
                "X-NEG-CODE", f"X({code_width})", [_cobol_string(c) for c in neg_codes]
            )
        )
        out.extend(
            _emit_value_table(
                "X-POS-CODE", f"X({code_width})", [_cobol_string(c) for c in pos_codes]
            )
        )
        out.extend(_emit_value_table("X-ELIGIBLE", "9", [str(e) for e in eligible]))
        push("01 REASONS.")
        for slot in ("NEG", "POS"):
            push(f"   05 REASON-{slot}-COUNT  PIC S9(4) COMP-5.")
            push(f"   05 REASON-{slot} OCCURS {k}.")
            push(f"      10 REASON-{slot}-CODE   PIC X({code_width}).")
            push(f"      10 REASON-{slot}-IMPACT PIC S9(9) COMP-5.")
    push("01 FEATURE-INPUTS.")
    for name, original in zip(cobol_names, feature_names):
        push(f"   05 {name:<28} COMP-2 VALUE 0.  *> {original}")
    push("PROCEDURE DIVISION.")
    push("MAIN-PARA.")
    if driver_rows is None:
        push("    PERFORM SCORE-ONE")
        push("    GOBACK.")
    else:
        for row in driver_rows:
            if len(row) != len(feature_names):
                raise ValueError(f"driver row has {len(row)} values, expected {len(feature_names)}")
            for name, value in zip(cobol_names, row):
                push(f"    MOVE {_literal(float(value))} TO {name}")
            push("    PERFORM SCORE-ONE")
            push("    DISPLAY F-LATENT-INT ' ' FINAL-BAND ' ' F-PD-PPM")
            if explain:
                push("    DISPLAY 'R ' REASON-NEG-COUNT ' ' REASON-POS-COUNT")
                for slot, tag in (("NEG", "N"), ("POS", "P")):
                    push(f"    PERFORM VARYING X-J FROM 1 BY 1 UNTIL X-J > REASON-{slot}-COUNT")
                    push(
                        f"        DISPLAY '{tag} ' REASON-{slot}-CODE(X-J)"
                        f" ' ' REASON-{slot}-IMPACT(X-J)"
                    )
                    push("    END-PERFORM")
        push("    GOBACK.")
    push("")
    push("SCORE-ONE.")
    push(f"    MOVE {int(model['base_micro'])} TO F-ACCUM-MICRO")
    for i in range(len(model["trees"])):
        push(f"    PERFORM SCORE-TREE-{i + 1:04d}")
    push("    PERFORM CLAMP-AND-SCALE")
    push("    PERFORM ASSIGN-BAND")
    push("    PERFORM CALIBRATE-PD")
    if explain:
        push("    PERFORM EXPLAIN-ONE")
    push("    CONTINUE.")
    push("")

    for i, tree in enumerate(model["trees"]):
        push(f"SCORE-TREE-{i + 1:04d}.")
        out.extend(_emit_tree(tree, cobol_names, indent=4))
        push("    CONTINUE.")
        push("")

    push("CLAMP-AND-SCALE.")
    push("    IF F-ACCUM-MICRO < 0")
    push("        MOVE 0 TO F-LATENT-MICRO")
    push("    ELSE")
    push(f"        IF F-ACCUM-MICRO > {micro_scale}")
    push(f"            MOVE {micro_scale} TO F-LATENT-MICRO")
    push("        ELSE")
    push("            MOVE F-ACCUM-MICRO TO F-LATENT-MICRO")
    push("        END-IF")
    push("    END-IF")
    push("    *> div_rha (spec 2.2): truncation == floor for non-negatives")
    push("    COMPUTE F-LATENT-INT =")
    push(f"        (2 * F-LATENT-MICRO + {ratio}) / (2 * {ratio})")
    push("    CONTINUE.")
    push("")

    push("ASSIGN-BAND.")
    push("    EVALUATE TRUE")
    for cutoff, label in zip(edges_int[1:-1], labels[:-1]):
        push(f"        WHEN F-LATENT-INT < {cutoff}")
        push(f"            MOVE '{label}' TO FINAL-BAND")
    push("        WHEN OTHER")
    push(f"            MOVE '{labels[-1]}' TO FINAL-BAND")
    push("    END-EVALUATE")
    push("    CONTINUE.")
    push("")

    out.extend(_emit_calibration(artifact.get("calibration"), micro_scale))

    if explain:
        push("")
        out.extend(_emit_explain(artifact, n_features, ratio, k))
        for i, tree in enumerate(model["trees"]):
            if split_features(tree):
                out.extend(_emit_tree_attribution(i + 1, tree, cobol_names, baseline))
                push("")
    return "\n".join(out) + "\n"
