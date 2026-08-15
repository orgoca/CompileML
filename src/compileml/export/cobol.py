"""COBOL export: the decision artifact as a native mainframe scorer.

Emits a self-contained COBOL program (``>>SOURCE FORMAT FREE``; compiles
under GnuCOBOL and Enterprise COBOL 6+) that reproduces the runtime's
integer pipeline bit-for-bit:

- leaf accumulation uses the artifact's ``value_micro`` integers verbatim
  (``ADD 21077 TO F-ACCUM-MICRO``) — nothing is re-rounded at export time;
- the display conversion is the spec §2.2 ``div_rha`` formula in integer
  ``COMPUTE`` (truncating division equals floor for non-negative values);
- the band ladder uses strict ``<`` against the interior integer edges,
  which is exactly ``bisect_right`` (spec §5).

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
    return repr(float(threshold)).replace("e", "E")


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


def export_cobol(
    artifact: dict,
    *,
    program_id: str = "CMLSCORE",
    driver_rows: list | None = None,
) -> str:
    """Render the artifact's score + band pipeline as a COBOL program.

    The generated program reads features from WORKING-STORAGE (integration
    point: MOVE caller values in, or adapt to a LINKAGE SECTION), then leaves
    ``F-LATENT-INT`` (display-scale score) and ``FINAL-BAND`` populated.

    With ``driver_rows`` (a list of feature rows), the program becomes a
    parity harness instead: it scores every row and DISPLAYs
    ``latent_int band`` one row per line — compile it, run it, and diff the
    output against the Python runtime. CI does exactly that under GnuCOBOL.

    Scope note: this emits score, clamp, display conversion, and band
    assignment — the mainframe decision path. Calibrated PDs and reason
    codes remain runtime/SQL concerns.
    """
    model = artifact["model"]
    micro_scale = int(model["micro_scale"])
    scale = int(artifact["scale"])
    ratio = micro_scale // scale
    edges_int = [int(e) for e in artifact["bands"]["edges_int"]]
    labels = [str(x) for x in artifact["bands"]["labels"]]
    feature_names = artifact["features"]["names"]

    used: set[str] = set()
    cobol_names = [_cobol_name(n, used) for n in feature_names]
    label_width = max(len(x) for x in labels)

    out: list[str] = []
    push = out.append
    push(">>SOURCE FORMAT FREE")
    push("IDENTIFICATION DIVISION.")
    push(f"PROGRAM-ID. {program_id}.")
    push("*> ------------------------------------------------------------")
    push("*> CompileML decision artifact export (score + band).")
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
            push("    DISPLAY F-LATENT-INT ' ' FINAL-BAND")
        push("    GOBACK.")
    push("")
    push("SCORE-ONE.")
    push(f"    MOVE {int(model['base_micro'])} TO F-ACCUM-MICRO")
    for i in range(len(model["trees"])):
        push(f"    PERFORM SCORE-TREE-{i + 1:04d}")
    push("    PERFORM CLAMP-AND-SCALE")
    push("    PERFORM ASSIGN-BAND")
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
    return "\n".join(out) + "\n"
