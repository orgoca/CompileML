---
name: Bug report
about: Something behaves differently from what the docs or the spec say
labels: bug
---

**What happened, and what you expected instead.**

**How to reproduce it.** A few lines that fail are worth more than a
description. Synthetic data is fine — `compileml.datasets.make_credit_data`
needs no download.

```python
```

**Version and platform.** `compileml.__version__`, Python, OS.

---

For a parity or determinism bug — a runtime, SQL, or COBOL disagreement —
these three usually pin it down on their own, so include them if you can:

- the `artifact_hash`
- the output of `compileml inspect your_artifact.json`
- one input row that shows the disagreement

Do not attach a real portfolio. If the bug only reproduces on data you
cannot share, say so and describe its shape — base rate, row count, feature
count, anything unusual about the distribution. Several real bugs here have
turned out to be about a low base rate rather than the data itself.
