# Reference Data Scripts

## `verify_golden_bundles.py`

Standalone verifier for hipDNN golden bundle directories.

### Usage

From the `rocm-libraries/` repo root:

```bash
python dnn-providers/integration-tests/reference_data_scripts/verify_golden_bundles.py \
  [--default-tier quick|standard|comprehensive|full] \
  ROOT [ROOT ...]
```

`ROOT` may point at:
- `dnn-providers/integration-tests/integration_test_bundles/`
- a tier directory such as `.../integration_test_bundles/quick/`
- a canonical subtree or individual bundle directory containing graph bundles

If the input path has no `quick|standard|comprehensive|full` segment, the script
uses `--default-tier` for advisory naming and emits a warning.

### What it checks

Hard errors:
- graph JSON must parse and include non-empty `nodes` and `tensors`
- tensor entries must declare valid `uid`, `dims`, `strides`, and `data_type`
- duplicate tensor UIDs are rejected
- when a bundle carries a companion `<Name>.tensors.dvc`, every declared tensor must have a matching `<Name>.tensor<uid>.bin`
- present tensor files must match `element_space * element_size`
- present output tensors with floating dtypes must not contain NaN or Inf
- bundle total size must not exceed 2 MiB; larger bundles fail because they would quickly explode test artifact sizes
- discovered metadata sidecars (`meta.json` or `*.meta.json`) must parse and contain non-empty string fields `generator` and `reference_source`
- advisory naming must be derivable as `{Tier}/{Operation}/{Layout}/{DataType}/{Name}/{Name}.json`

Warnings only:
- stray non-graph `.json` files are ignored
- unexpected top-level directories under an `integration_test_bundles/`-style root are reported
- bundle total size above 1 MiB emits a warning
- missing tier segment falls back to `--default-tier`

### Output

- advisories go to stdout
- diagnostics and summary go to stderr
- exit code `0`: no hard errors
- exit code `1`: one or more hard errors

For a valid bundle such as `quick/BatchnormFwdInference/nchw/fp32/Small/Small.json`,
the advisory includes:

```text
canonical_path: quick/BatchnormFwdInference/nchw/fp32/Small/
full_test_name: quick_BatchnormFwdInference_nchw_fp32_Small.Small
```

### Data requirements

Real `.bin` tensor payloads are optional unless the bundle carries a companion
`<Name>.tensors.dvc`. When that pointer exists, the payloads must also be
present locally for byte-size and NaN/Inf checks; run `dvc pull` if only the
pointer file is present.

### Tests

Direct Python test run:

```bash
python dnn-providers/integration-tests/reference_data_scripts/tests/test_verify_golden_bundles.py
```

CTest registration after configuring `dnn-providers/integration-tests`:

```bash
ctest -R hipdnn_bundle_verifier_python_tests --output-on-failure
```

The unittest file covers:
- valid advisory output
- output NaN rejection
- size mismatch detection
- missing tensor file detection
- metadata field validation
- stray JSON warning
- unexpected top-level directory warning
- FP16 and BF16 non-finite detection
- bundle size warning/error thresholds
- optional tensors without a DVC companion pointer
