"""Export parity tests.

The SQL test executes the generated query in a real SQL engine (SQLite) and
asserts row-for-row integer equality with the Python runtime — latent, band,
and PD. The COBOL tests verify that the emitted program carries the
artifact's integers verbatim and that every threshold literal round-trips to
the exact float64 the runtime compares; the run-parity test then compiles the
program under GnuCOBOL and asserts the same latent, band and PD equality for
every calibration mode. It skips where cobc is absent; CI installs it.
"""

import re
import sqlite3
import warnings

import numpy as np
import pytest

from compileml.artifact import build_artifact
from compileml.bands import monotone_quantile_bands
from compileml.compile import train_whitebox
from compileml.export import export_cobol, export_sql
from compileml.runtime import decide

RNG = np.random.default_rng(31)
N, P = 2500, 5
FEATURES = [f"f{i}" for i in range(P)]


@pytest.fixture(scope="module")
def fitted():
    X = RNG.standard_normal((N, P))
    teacher = 1.0 / (1.0 + np.exp(-(1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3])))
    y = (RNG.random(N) < teacher).astype(int)
    model, _ = train_whitebox(X, teacher, n_estimators=40, random_state=2)
    latent = np.clip(model.predict(X), 0, 1)
    spec = monotone_quantile_bands(latent, y, n_bands=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifact = build_artifact(
            model,
            FEATURES,
            np.median(X, axis=0),
            spec,
            calibration_latent=latent,
            calibration_y=y,
        )
    return X, artifact


# ------------------------------------------------------------------- SQL
def test_sql_executes_with_exact_parity(fitted):
    """Generated SQL vs Python runtime: integer equality on every row."""
    X, artifact = fitted
    sql = export_sql(artifact, table="features", dialect="sqlite")

    con = sqlite3.connect(":memory:")
    con.execute(f"CREATE TABLE features ({', '.join(f'{n} REAL' for n in FEATURES)})")
    rows = [[float(v) for v in row] for row in X[:400]]
    con.executemany(f"INSERT INTO features VALUES ({', '.join('?' * P)})", rows)

    got = con.execute(sql).fetchall()
    cols = [d[0] for d in con.execute(sql).description]
    i_latent = cols.index("latent_int")
    i_band = cols.index("band")
    i_pd = cols.index("pd_ppm")
    i_raw = cols.index("raw_micro")

    for row, sql_row in zip(rows, got):
        ref = decide(artifact, row, explain=False)
        assert int(sql_row[i_raw]) == ref["raw_micro"]
        assert int(sql_row[i_latent]) == ref["latent_int"]
        assert str(sql_row[i_band]) == ref["band"]
        assert int(sql_row[i_pd]) == ref["pd_ppm"]
    con.close()


def test_sql_step_calibration_parity(fitted):
    X, artifact = fitted
    import copy

    art = copy.deepcopy(artifact)
    art["calibration"]["mode"] = "step"
    sql = export_sql(art, table="features", dialect="sqlite")

    con = sqlite3.connect(":memory:")
    con.execute(f"CREATE TABLE features ({', '.join(f'{n} REAL' for n in FEATURES)})")
    rows = [[float(v) for v in row] for row in X[:150]]
    con.executemany(f"INSERT INTO features VALUES ({', '.join('?' * P)})", rows)
    got = con.execute(sql).fetchall()
    cols = [d[0] for d in con.execute(sql).description]
    i_pd = cols.index("pd_ppm")
    for row, sql_row in zip(rows, got):
        assert int(sql_row[i_pd]) == decide(art, row, explain=False)["pd_ppm"]
    con.close()


def test_sql_ansi_dialect_renders(fitted):
    _, artifact = fitted
    sql = export_sql(artifact, dialect="ansi")
    assert "GREATEST(0, LEAST(raw_micro" in sql
    assert artifact["artifact_hash"] in sql


# ------------------------------------------------------------------ COBOL
def test_cobol_carries_artifact_integers_verbatim(fitted):
    _, artifact = fitted
    text = export_cobol(artifact)

    # Every leaf integer appears as an ADD/SUBTRACT with no re-rounding.
    emitted_adds = set()
    for m in re.finditer(r"ADD (\d+) TO F-ACCUM-MICRO", text):
        emitted_adds.add(int(m.group(1)))
    for m in re.finditer(r"SUBTRACT (\d+) FROM F-ACCUM-MICRO", text):
        emitted_adds.add(-int(m.group(1)))
    artifact_leaves = {
        int(v)
        for tree in artifact["model"]["trees"]
        for f, v in zip(tree["feature"], tree["value_micro"])
        if f == -2
    }
    assert artifact_leaves <= emitted_adds | {0}

    # Base score moved verbatim.
    assert f"MOVE {artifact['model']['base_micro']} TO F-ACCUM-MICRO" in text

    # Every threshold literal round-trips to the exact float64.
    thresholds = {
        float(tree["threshold"][i])
        for tree in artifact["model"]["trees"]
        for i, f in enumerate(tree["feature"])
        if f != -2
    }
    emitted = {
        float(m.group(1).replace("E", "e"))
        for m in re.finditer(r"IF F-F\d+ <= ([0-9.eE+-]+)", text)
    }
    assert thresholds == emitted

    # Band ladder: strict < on every interior integer edge (bisect_right).
    for cutoff in artifact["bands"]["edges_int"][1:-1]:
        assert f"WHEN F-LATENT-INT < {cutoff}" in text

    # div_rha display conversion with the right ratio.
    ratio = artifact["model"]["micro_scale"] // artifact["scale"]
    assert f"(2 * F-LATENT-MICRO + {ratio}) / (2 * {ratio})" in text
    assert artifact["artifact_hash"] in text

    # Calibration table (spec §6): both ends verbatim, every knot a cutoff.
    f = artifact["calibration"]["f_micro"]
    pd = artifact["calibration"]["pd_ppm"]
    assert "PERFORM CALIBRATE-PD" in text
    assert f"WHEN F-LATENT-MICRO <= {f[0]}\n            MOVE {pd[0]} TO F-PD-PPM" in text
    for knot in f[1:]:
        assert f"WHEN F-LATENT-MICRO < {knot}\n" in text
    assert f"WHEN OTHER\n            MOVE {pd[-1]} TO F-PD-PPM" in text


def test_cobol_sanitizes_awkward_feature_names(fitted):
    _, artifact = fitted
    import copy

    art = copy.deepcopy(artifact)
    art["features"]["names"] = ["bills_paid_late!", "bills paid late", "übers2", "x" * 60, "pd_ppm"]
    text = export_cobol(art)
    assert "05 F-BILLS-PAID-LATE " in text
    assert "F-BILLS-PAID-LATE-2" in text  # collision gets a suffix
    # A feature must never shadow the program's own working storage.
    assert "05 F-PD-PPM-2 " in text
    for line in text.splitlines():
        for name in re.findall(r"05 (\S+)", line):
            assert len(name) <= 30


@pytest.mark.parametrize("calibration", ["linear_int", "step", "none"])
def test_cobol_run_parity_under_gnucobol(fitted, tmp_path, calibration):
    """Compile the driver-harness export with GnuCOBOL, run it, and diff the
    printed latent, band and PD against the Python runtime, for every
    calibration mode. Skips where cobc is absent; CI installs GnuCOBOL."""
    import copy
    import shutil
    import subprocess

    if shutil.which("cobc") is None:
        pytest.skip("GnuCOBOL not installed (CI runs this)")
    X, fitted_artifact = fitted
    artifact = copy.deepcopy(fitted_artifact)
    if calibration == "none":
        artifact["calibration"] = None
    else:
        artifact["calibration"]["mode"] = calibration
    # Extreme rows push the raw score past both clamps, so the table's two
    # ends are exercised as well as its interior segments.
    extremes = [[8.0] * P, [-8.0] * P, [8.0, -8.0, 8.0, 8.0, 0.0], [-8.0, 8.0, -8.0, -8.0, 0.0]]
    rows = [[float(v) for v in row] for row in X[:200]] + extremes

    src = tmp_path / "harness.cob"
    src.write_text(export_cobol(artifact, driver_rows=rows), encoding="utf-8")
    exe = tmp_path / "harness"
    compiled = subprocess.run(
        ["cobc", "-x", "-o", str(exe), str(src)], capture_output=True, text=True
    )
    assert compiled.returncode == 0, compiled.stderr

    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    lines = [ln for ln in run.stdout.strip().splitlines() if ln.strip()]
    assert len(lines) == len(rows)
    for row, line in zip(rows, lines):
        latent_txt, band_txt, pd_txt = line.split()
        ref = decide(artifact, row, explain=False)
        assert int(latent_txt.replace("+", "")) == ref["latent_int"]
        assert band_txt.strip() == ref["band"]
        assert int(pd_txt.replace("+", "")) == ref["pd_ppm"]


def test_sql_1_band_export(fitted):
    import copy

    _, artifact = fitted
    art = copy.deepcopy(artifact)
    art["bands"]["edges_int"] = [0, 1_000_000]
    art["bands"]["labels"] = ["G01"]

    sql = export_sql(art, table="features", dialect="sqlite")

    con = sqlite3.connect(":memory:")
    con.execute(f"CREATE TABLE features ({', '.join(f'{n} REAL' for n in FEATURES)})")

    con.execute(sql)
    con.close()
