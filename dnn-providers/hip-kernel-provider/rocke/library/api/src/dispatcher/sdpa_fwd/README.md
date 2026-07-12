# dispatcher/sdpa_fwd — FMHA (SDPA-fwd) featurizer + kernel-selection predictors

The op-specific half of the dispatcher for SDPA-forward, organized as the target
`dispatcher/<OP>/<ARCH>` layout:

```
sdpa_fwd/
  FmhaFeatures.hpp          # generated: named-field feature struct + to_array()
  FmhaFeaturizer.hpp        # generated: (problem, config, hw) -> FmhaFeatures
  <arch>/TiledSpecDefaults.hpp   # generated: per-arch knob defaults
  <arch>/model_tflops.c/.h       # committed predictor for (sdpa_fwd, arch, dtype)
```

## Generated pieces (do not hand-edit)

- **`FmhaFeatures.hpp` / `FmhaFeaturizer.hpp`** — emitted by
  `platform/.../heuristics/gen_fmha_featurizer.py` from the Python
  `FmhaFeatureEngine`. The featurizer mirrors `extract()` term-for-term in the
  same operation order, so the runtime vector is **bit-identical** to what the
  model trained on. Guaranteed by `test_fmha_featurizer_roundtrip.py` (Python↔C++
  exact match on edge-case fixtures).
- **`<arch>/TiledSpecDefaults.hpp`** — emitted by `gen_arch_tiled_defaults.py`,
  introspecting each arch's `UnifiedAttention2DTiledSpec`. Supplies defaults for
  arch-specific knobs not carried on the base `AotInstance`, until arch-specific
  instance subclasses exist.

## Committed predictors (Route A, static-linked)

One predictor per `(op, arch, dtype)`, tflops head only (the target the
dispatcher ranks on). Route A (`lgbm_to_c.py`) is the committed form because it
emits **unique per-(op,arch,dtype) symbols** — many predictors static-link into
one dispatcher binary with no collision (treelite emits a fixed `predict` symbol
and cannot). Route C (treelite) stays in the model dir as an independent
correctness cross-check only.

```
<arch>/model_tflops.c    // double rocke_score_<op>_<dtype>_<arch>_tflops(const double* f)
<arch>/model_tflops.h
<arch>/model_tflops.meta.json   // registry input: symbol/op/arch/dtype/num_features
```

## Contract

- `f[]` is the `FmhaFeatures::to_array()` vector in `feature_spec.json` order,
  length `num_features`. Build it with `fmha_featurize(...)`.
- The predictor returns the **raw booster sum** (log-space; trains on
  `log1p(tflops)`). `expm1` is monotone, so the argmax tie-break is exact on the
  raw score.
- **Drift guard:** the registry carries `num_features` per model; the tie-break
  asserts it equals `FmhaFeatures::kNumFeatures` and falls back to Phase-1
  first-match on mismatch (a feature-contract change can never feed a stale
  predictor a wrong-width vector).

## Adding a model to the build

`api/src/CMakeLists.txt` lists sources **explicitly** (no globbing). When you
commit a new predictor, add its `.c`, e.g.:

```cmake
    dispatcher/sdpa_fwd/gfx950/model_tflops.c
```

Then regenerate the registry (`gen_model_registry.py`) so
`rocke_lookup_model(op, arch, dtype)` sees it.
