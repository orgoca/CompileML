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
  non-negative ``div_rha`` form is exact.

Feature inputs are declared ``COMP-2`` (IEEE binary64 under GnuCOBOL and
Enterprise COBOL with IEEE arithmetic). Threshold literals are emitted in
shortest round-trip form, so the compiled comparison sees the identical
float64 the Python runtime sees. For decimal-arithmetic targets, build the
artifact with quantized thresholds (``build_artifact(threshold_decimals=…)``)
so every runtime, Python included, compares identical values (spec §11).
"""

from __future__ import annotations

import re

LEAF = -2

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

    Scope note: reason codes are not emitted yet; they come only from the
    Python runtime (#14).
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
        push("    GOBACK.")
    push("")
    push("SCORE-ONE.")
    push(f"    MOVE {int(model['base_micro'])} TO F-ACCUM-MICRO")
    for i in range(len(model["trees"])):
        push(f"    PERFORM SCORE-TREE-{i + 1:04d}")
    push("    PERFORM CLAMP-AND-SCALE")
    push("    PERFORM ASSIGN-BAND")
    push("    PERFORM CALIBRATE-PD")
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
    return "\n".join(out) + "\n"
