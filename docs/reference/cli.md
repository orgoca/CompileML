# CLI

The `compileml` command ships with the package. Artifact-side commands
(`inspect`, `verify`, `score`, `export`) run on the standard library alone.

## compileml compile

```bash
compileml compile --model model.pkl --csv train.csv --y-col DEFAULT \
                  --n-bands 10 --reasons reasons.json --out decision.json
```

Loads a pickled fitted model, reads features from the CSV (all columns except
`--y-col`), imputes medians, builds monotone-quantile bands and isotonic
calibration, and writes a hashed artifact. `--reasons` takes the reason
dictionary as JSON ([format](../howto/reason-codes.md)).

## compileml verify

```bash
compileml verify decision.json     # exit 0 = hash + structure OK, 1 otherwise
```

## compileml inspect

```bash
compileml inspect decision.json
```

Prints a JSON summary: hash, feature count, tree count, whitebox depth,
`exact_attribution`, band ladder, calibration mode, missing policy, reason
coverage.

## compileml score

```bash
# one row, with reasons
compileml score decision.json --features "0.52,1.3,0.0,…" --explain

# a CSV, sub-millisecond path, results to file
compileml score decision.json --csv applicants.csv --out scores.csv
```

Empty values, `nan`, and `null` in `--features` are treated as missing and
handled per the artifact's missing policy.

## compileml validate

```bash
compileml validate decision.json --csv holdout.csv --y-col DEFAULT --require-reasons
```

Runs the [eight-check framework](../howto/validate.md); prints the full
evidence report as JSON; exits non-zero if any check fails — suitable as a CI
deployment gate.

## compileml export

```bash
compileml export decision.json --target sql   --dialect ansi --out scorer.sql
compileml export decision.json --target cobol --program-id CMLSCORE --out scorer.cob
```
