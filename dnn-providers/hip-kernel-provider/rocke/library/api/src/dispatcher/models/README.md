# dispatcher/models — generated FMHA kernel-selection predictors

Standalone C predictors for the FMHA tie-break, generated from the rocKE
heuristics pipeline. They let the dispatcher score AOT kernel candidates
**without linking `liblightgbm.so` / `libgomp`** — the LightGBM booster is
lowered to nested `if/else` trees (see
`platform/python/rocke/heuristics/lgbm_to_c.py`).

## Contents

One predictor per `(op, dtype, arch)`, tflops head only (the target the
dispatcher ranks on):

```
fmha_<dtype>_<arch>_tflops.c   // double rocke_fmha_score_<dtype>_<arch>_tflops(const double* f)
fmha_<dtype>_<arch>_tflops.h   // extern decl + *_NUM_FEATURES
```

## Contract

- **Do not hand-edit.** Regenerate by retraining through the pipeline; the
  pipeline emits `model_tflops.c` / `.h` into the model dir, and promoting a
  model copies them here (header comment stamps the source model id).
- `f[]` is the feature vector in **`feature_spec.json` order** (indices 0..N-1).
  The C++ featurizer MUST fill it in that order; `*_NUM_FEATURES` is provided so
  the featurizer can static-assert agreement.
- The function returns the **raw booster sum** (log-space; the model trains on
  `log1p(tflops)`). `expm1` is monotonic, so the argmax tie-break is exact on the
  raw score — apply `expm1` only if a physical TFLOPS value is needed.

## Adding a model to the build

`api/src/CMakeLists.txt` lists sources **explicitly** (no globbing). When you
commit a new predictor, add its `.c` to that list, e.g.:

```cmake
    dispatcher/models/fmha_fp16_gfx950_tflops.c
```

Keeping it explicit matches the surrounding convention and makes the shipped
predictor set reviewable in one place.
