# Reference Data Scripts

## `verify_golden_bundles.py`

Standalone verifier for hipDNN golden bundle directories.

### Usage

From the `rocm-libraries/` repo root:

```bash
python dnn-providers/integration-tests/reference-data-scripts/verify_golden_bundles.py \
  [--default-tier quick|standard|comprehensive|full] \
  [--require-data] \
  ROOT [ROOT ...]
```

`ROOT` may point at:
- `dnn-providers/integration-tests/integration-test-bundles/`
- a tier directory such as `.../integration-test-bundles/quick/`
- a canonical subtree or individual bundle directory containing single-graph or
  template-sweep bundles

If the input path has no `quick|standard|comprehensive|full` segment, the script
uses `--default-tier` for advisory naming and emits a warning.

### What it checks

Hard errors, single-graph bundles (`{Name}.json`):
- graph JSON must parse and include non-empty `nodes` and `tensors`
- tensor entries must declare valid `uid`, `dims`, `strides`, and `data_type`
- duplicate tensor UIDs are rejected
- when a bundle carries a companion `<Name>.tensors.dvc`, every declared tensor must have a matching `<Name>.tensor<uid>.bin` when `--require-data` is passed
- present tensor files must match `element_space * element_size`
- present tensors (input or output) with floating dtypes must not contain NaN or Inf
- bundle total size must not exceed 2 MiB; larger bundles fail because they would quickly explode test artifact sizes
- discovered metadata sidecars (`meta.json` or `*.meta.json`) must parse and contain non-empty string fields `generator` and `reference_source`
- the optional metadata fields `no_cpu_reference`, `no_gpu_reference` and `expected_rejection` must be JSON booleans when present
- a `<Name>.meta.json` that sets both `no_cpu_reference` and `no_gpu_reference` to `true` must have golden data declared next to `<Name>.json`: a `<Name>.tensors.dvc` or a `<Name>.tensor<uid>.bin`
- a `<Name>.meta.json` that sets `expected_rejection` to `true` (an intentionally invalid graph the engine under test must reject; it is never executed) must not declare golden data: no `<Name>.tensors.dvc` and no `<Name>.tensor<uid>.bin` next to `<Name>.json`
- advisory naming must be derivable as `{Tier}/{Operation}/{Layout}/{DataType}/{Name}/{Name}.json`

Hard errors, template-sweep bundles (`graph.template.json` + `sweep.json`):
- a directory containing either file must contain both
- `graph.template.json` and `sweep.json` must parse as objects, with non-empty `nodes`/`tensors` and `cases`
- `cases[].id` is required, unique within the sweep, and must be lowercase_snake_case
- every `${case.<field>}` placeholder in the template (scalar fields, `attributes.<field>`, and per-tensor `dims`/`strides`/`data_type`) must have a matching value in the case's `values`
- a case's tensor set (`values.tensors[].uid`) must exactly match the template's tensor UIDs — omitted, duplicate, or unknown UIDs are errors
- resolved per-case tensor `dims`/`strides`/`data_type` are validated the same way as single-graph bundles
- each case's `metadata` object must contain non-empty string fields `generator` and `reference_source`
- a non-null `golden` requires a `path` pointing at a `tensors.dvc` file
- a case's `metadata` may set `no_cpu_reference` / `no_gpu_reference` / `expected_rejection`; each must be a JSON boolean when present
- a case whose `metadata` sets both `no_cpu_reference` and `no_gpu_reference` to `true` must declare golden data (a `golden` object with a `path`)
- a case whose `metadata` sets `expected_rejection` to `true` must not declare golden data (no `golden` object with a `path`)
- when a case has golden data, tensor files are expected at `golden/{CaseId}/tensor<uid>.bin` (not the template's directory) and validated the same as single-graph tensors (byte-size, NaN/Inf for both input and output tensors, per-case 2 MiB budget)
- advisory naming must be derivable as `{Tier}/{Operation}/{TopologyName}/sweep.json`

Warnings only:
- stray non-graph `.json` files are ignored
- unexpected top-level directories under an `integration-test-bundles/`-style root are reported
- bundle total size above 1 MiB emits a warning
- a `.tensors.dvc`-backed tensor file that is missing locally and `--require-data` was not passed (default): warns that the payload hasn't been pulled yet instead of failing
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

For a template-sweep case `quick/BatchnormFwdInference/Inference/sweep.json`
with `cases[].id: small_fp32_nchw`, the advisory includes:

```text
canonical_path: quick/BatchnormFwdInference/Inference/sweep.json
full_test_name: quick_BatchnormFwdInference_Inference.small_fp32_nchw
```

### Data requirements

Real `.bin` tensor payloads are optional unless the bundle carries a companion
`<Name>.tensors.dvc` (single-graph) or a case's `golden.path` points at a
`tensors.dvc` (template sweep). When that pointer exists but the payload isn't
present locally, the default run only warns (run `dvc pull` to fetch it); pass
`--require-data` to make the missing payload a hard error, which is how CI
enforces full validation after pulling. Whenever a payload *is* present,
byte-size and NaN/Inf checks always run regardless of `--require-data`.

### Tests

Direct Python test run:

```bash
python dnn-providers/integration-tests/reference-data-scripts/tests/test_verify_golden_bundles.py
```

CTest registration after configuring `dnn-providers/integration-tests`:

```bash
ctest -R hipdnn_bundle_verifier_python_tests --output-on-failure
```

The unittest file covers, for single-graph bundles:
- valid advisory output
- output and input NaN rejection
- size mismatch detection
- missing tensor file detection (warning by default, error with `--require-data`)
- metadata field validation
- stray JSON warning
- unexpected top-level directory warning
- FP16 and BF16 non-finite detection
- bundle size warning/error thresholds
- optional tensors without a DVC companion pointer

and for template-sweep bundles:
- valid advisory output per expanded case
- duplicate and non-snake_case `cases[].id` rejection
- missing/unknown tensor UID rejection against the template's tensor set
- missing placeholder value detection
- unused `values` entry warnings
- `golden.path` validation and per-case tensor file/NaN/size checks (input and output tensors)
- per-case metadata field validation
- a directory with only one of `graph.template.json`/`sweep.json`

## Sliding-window bounds in the generators

A left window bound has two common conventions, and each generator flag is named after the
one it uses:

| Convention | Name | `L = 3` keeps, for the query on the diagonal |
|---|---|---|
| hipDNN graph, cuDNN | `left_bound` (`--left-bound`) | the 3 keys ending at the diagonal, diagonal included |
| FlashAttention, AITER | `window_size_left` (`--window-left`) | the 3 keys before the diagonal, plus the diagonal (4 keys) |

`generate_sdpa_fwd_golden.py --left-bound L` writes `left_bound = L` to the graph.
`--window-left W` writes `left_bound = W + 1`; it exists for `--validate`, which passes the
window to AITER. The two flags are mutually exclusive. `left_bound` must be positive or -1
(unbounded).
