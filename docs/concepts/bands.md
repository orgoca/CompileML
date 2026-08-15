# Risk bands

Bands are the governed unit of a credit decision: policy attaches to bands,
not raw scores. CompileML freezes band edges as fixed-point integers inside
the artifact, so band assignment is an integer comparison with one boundary
convention everywhere ([spec §5](../ARTIFACT_SPEC.md)): left-closed,
right-open, cutoff belongs to the upper band.

## Builders

All builders return a `BandSpec` (float edges + labels + evidence metadata)
that `build_artifact` converts to the integer ladder — refusing edges that
collide after fixed-point conversion.

### `quantile_bands(latent, n_bands)`
Equal-volume bands. No outcome data needed.

### `monotone_quantile_bands(latent, y, n_bands, allow_merge=…)`
Quantile edges plus empirical bad rates and isotonic-smoothed semantics in the
metadata. With `allow_merge=True`, adjacent bands whose empirical rates invert
by more than `merge_eps` are merged — trading band count for guaranteed-monotone
empirical semantics.

### `semantic_bands(latent, y, …)` — search and certify
Discovers the **maximum number of statistically separable bands**: every band
needs `min_band_size` observations, adjacent Jeffreys PD intervals must be
separated by `delta_sep`, and no band may retain internal rank power above
`0.5 + eps_auc` (the score must be "used up" within each band). The search runs
on cheap point estimates; the final banding is certified once with bootstrap
AUC intervals, and all evidence ships in the metadata.

### `governance_bands(latent, y, …)` — the committee variant
Welch t-test separation of adjacent bad rates, a within-band residual-AUC cap,
and optionally strictly monotone PDs. Conservative defaults, evidence in
metadata (`adjacent_p_ttest`, `adjacent_delta_pd`).

## The property worth demoing

The certified builders **refuse to invent structure**. Feed them outcomes that
are pure noise and they return one band flagged `no_discrete_classes` — the
statistically honest answer. Feed them five real risk plateaus and they find
five. There is a test pinning each behavior.

!!! note "Strictness is a knob"
    `eps_auc` controls how much residual within-band ranking you tolerate.
    Distilled whitebox latents are naturally plateaued (a depth-2, 120-tree
    model takes finitely many values), which is the intended input. A smooth,
    steadily-sloped latent may legitimately support only one band under a
    strict `eps_auc` — that is the method telling you band boundaries would be
    arbitrary, not a failure.

## Sizing note

Empirical bad rates need volume: at 50 observations per band, rate estimates
wobble by several points and monotonicity checks will flag noise. Validate
with samples that give each band a few hundred observations.
