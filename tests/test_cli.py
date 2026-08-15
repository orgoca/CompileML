"""CLI tests: every subcommand through main(argv) with real files."""

import csv
import json
import pickle
import warnings

import numpy as np
import pytest

from compileml.artifact import build_artifact, save_artifact
from compileml.bands import monotone_quantile_bands
from compileml.cli import main
from compileml.compile import train_whitebox

RNG = np.random.default_rng(41)
N, P = 1500, 4
FEATURES = [f"c{i}" for i in range(P)]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cli")
    X = RNG.standard_normal((N, P))
    teacher = 1.0 / (1.0 + np.exp(-(X[:, 0] - 0.6 * X[:, 1])))
    y = (RNG.random(N) < teacher).astype(int)
    model, _ = train_whitebox(X, teacher, n_estimators=25, random_state=3)
    latent = np.clip(model.predict(X), 0, 1)
    spec = monotone_quantile_bands(latent, y, n_bands=6)
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
    artifact_path = tmp / "artifact.json"
    save_artifact(artifact, artifact_path)

    csv_path = tmp / "data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURES + ["DEFAULT"])
        for row, label in zip(X[:1200], y[:1200]):
            writer.writerow([f"{v:.6f}" for v in row] + [int(label)])

    model_path = tmp / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return tmp, artifact, str(artifact_path), str(csv_path), str(model_path)


def test_verify_ok(env, capsys):
    _, _, artifact_path, *_ = env
    assert main(["verify", artifact_path]) == 0
    assert "OK: hash verified" in capsys.readouterr().out


def test_verify_fails_on_tamper(env, tmp_path, capsys):
    _, artifact, *_ = env
    import copy

    bad = copy.deepcopy(artifact)
    bad["scale"] = 100  # break hash
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    assert main(["verify", str(bad_path)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_inspect(env, capsys):
    _, artifact, artifact_path, *_ = env
    assert main(["inspect", artifact_path]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["artifact_hash"] == artifact["artifact_hash"]
    assert summary["n_features"] == P
    assert summary["exact_attribution"] is True


def test_score_single_row(env, capsys):
    _, _, artifact_path, *_ = env
    assert main(["score", artifact_path, "--features", "0.5,-1.0,0.2,0.0", "--explain"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["band"].startswith("G")
    assert "reasons_negative" in payload


def test_score_handles_missing_token(env, capsys):
    _, _, artifact_path, *_ = env
    assert main(["score", artifact_path, "--features", "0.5,nan,0.2,"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert 0 <= payload["latent_int"] <= 1000


def test_score_csv(env, tmp_path):
    _, _, artifact_path, csv_path, _ = env
    out = tmp_path / "scores.csv"
    assert main(["score", artifact_path, "--csv", csv_path, "--out", str(out)]) == 0
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1200
    assert all(r["band"].startswith("G") for r in rows)


def test_export_sql_and_cobol(env, tmp_path):
    _, _, artifact_path, *_ = env
    sql_out = tmp_path / "scorer.sql"
    cob_out = tmp_path / "scorer.cob"
    assert (
        main(
            [
                "export",
                artifact_path,
                "--target",
                "sql",
                "--out",
                str(sql_out),
                "--dialect",
                "sqlite",
            ]
        )
        == 0
    )
    assert main(["export", artifact_path, "--target", "cobol", "--out", str(cob_out)]) == 0
    assert "WITH tree_scores AS" in sql_out.read_text(encoding="utf-8")
    assert "PROGRAM-ID. CMLSCORE." in cob_out.read_text(encoding="utf-8")


def test_validate_passes_and_gates(env, capsys):
    _, _, artifact_path, csv_path, _ = env
    assert main(["validate", artifact_path, "--csv", csv_path, "--y-col", "DEFAULT"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["all_pass"]
    # Reason gate: fixture artifact has no reason dictionary -> strict mode fails.
    assert (
        main(
            [
                "validate",
                artifact_path,
                "--csv",
                csv_path,
                "--y-col",
                "DEFAULT",
                "--require-reasons",
            ]
        )
        == 1
    )


def test_compile_end_to_end(env, tmp_path, capsys):
    _, _, _, csv_path, model_path = env
    out = tmp_path / "compiled.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert (
            main(
                [
                    "compile",
                    "--model",
                    model_path,
                    "--csv",
                    csv_path,
                    "--y-col",
                    "DEFAULT",
                    "--n-bands",
                    "6",
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
    assert main(["verify", str(out)]) == 0
    assert main(["score", str(out), "--features", "0.1,0.2,0.3,0.4"]) == 0
