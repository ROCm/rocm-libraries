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
- `direction`, when present, must be `forward` or `backward`
- advisory naming must be derivable as `{Tier}/{Operation}/{Layout}/{DataType}/{Name}/{Name}.json`

Hard errors, backward bundles (metadata whose `direction` is `backward`), from
RFC 0011 4.1 and 4.2. These apply to both single-graph metadata sidecars and
template-sweep `cases[].metadata`:
- `reference_strategy` is required. RFC 0011 lists it as mandatory for every
  bundle that has metadata at all, but it is enforced only here: forward bundles
  migrated from the old test system are explicitly allowed to carry none, and
  rejecting them would be a regression rather than a check.
- `forward_source` is required, and must be an object with non-empty string
  `method`, `version` and `generated_at`. A backward bundle's `O` and `LSE` come
  from a forward pass; if that forward pass is regenerated on its own the
  backward bundle stays internally consistent with its stale inputs while
  silently describing a different reference implementation, and no test catches
  it. `forward_source` is what makes it detectable.
- `forward_source.bundle`, when present, must name the forward bundle's graph
  JSON relative to the backward bundle (for a sweep case, relative to that
  case's golden directory). When that bundle's metadata is in scope, its
  `direction`, `reference_backend` and `generated_at` are cross-checked against
  `forward_source` — RFC 0011 4.2's stale-pairing check. A forward bundle that
  is not on disk is skipped rather than failed, so verifying one subtree without
  its partner pulled still passes.

Hard errors, template-sweep bundles (`graph.template.json` + `sweep.json`):
- a directory containing either file must contain both
- `graph.template.json` and `sweep.json` must parse as objects, with non-empty `nodes`/`tensors` and `cases`
- `cases[].id` is required, unique within the sweep, and must be lowercase_snake_case
- every `${case.<field>}` placeholder in the template (scalar fields, `attributes.<field>`, and per-tensor `dims`/`strides`/`data_type`) must have a matching value in the case's `values`
- a case's tensor set (`values.tensors[].uid`) must exactly match the template's tensor UIDs — omitted, duplicate, or unknown UIDs are errors
- resolved per-case tensor `dims`/`strides`/`data_type` are validated the same way as single-graph bundles
- each case's `metadata` object must contain non-empty string fields `generator` and `reference_source`
- a non-null `golden` requires a `path` pointing at a `tensors.dvc` file
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

## `generate_sdpa_bwd_golden.py`

Generates a paired SDPA **forward** bundle and **backward** (`SdpaBwdNode`)
bundle from one seeded pipeline run, with PyTorch as the independent reference
source (RFC 0011). The CPU and GPU backward references are otherwise checked
mainly against each other, so a shared misunderstanding passes those gate
tests; golden data from a third source catches it.

One run performs the whole pipeline, as RFC 0011 requires for backward
operations:

1. draw `Q`, `K`, `V` and `dO` from a single seeded RNG stream;
2. run the forward pass to get `O` and `LSE`, and save the forward bundle;
3. run the backward pass on **that** `O` and `LSE`, and save the backward
   bundle, recording `forward_source` in its metadata.

### Usage

```bash
python dnn-providers/integration-tests/reference-data-scripts/generate_sdpa_bwd_golden.py \
  --base-filename <dir>/quick/SdpaBackward/bhsd/bf16/Default/Default \
  --forward-base-filename <dir>/quick/SdpaFwd/bhsd/bf16/Default/Default \
  --q-dims 1 2 128 64 --v-dims 1 2 128 64 --attn-scale 0.125
```

`--forward-base-filename` is required: the forward pass whose `O` and `LSE` the
backward bundle consumes must itself be saved as a bundle. Add `--right-bound 0`
for a causal mask, `--left-bound N` for a sliding window, and
`--diagonal-alignment bottom_right` to anchor the diagonal to the bottom-right
corner. `--validate` cross-checks the gradients against PyTorch autograd.

Add `--additive-mask` for an explicit attention-mask tensor of shape
`[1, 1, S_q, S_kv]` in the IO dtype, carried by both halves of the pair. It is
added to the scores *before* the diagonal band, which is the order the CPU
reference uses, so a finite bias can never resurrect a position the band drops.
`dBias` is not produced.

### Template-sweep output

`--sweep-bundle-dir <bundle dir> --case-id <lowercase_snake_case id>` attaches
the backward tensors to a checked-in template sweep instead of writing a
standalone single-graph bundle. The tensors go to
`<bundle dir>/golden/<case id>/tensor<uid>.bin` — the layout the harness
resolves from a case's `golden.path` — and the case is added to, or replaced in,
`<bundle dir>/sweep.json` with its per-tensor dims, strides and data types read
back out of the graph this run produced. Other cases in the sweep are left
untouched, so the graph-only cases captured from the C++ suite survive
regeneration.

The graph topology itself is **not** written: it already exists as the
topology's `graph.template.json`. That is the point — the golden data runs
through exactly the graph the sweep's existing cases run through. The paired
forward bundle is still written, as an ordinary single-graph bundle at
`--forward-base-filename`.

### Precision and reproducibility

`reference_strategy` is `matched_precision`, RFC 0011's strategy for SDPA
backward: inputs and gradients are stored in the bundle's IO dtype and every
intermediate and accumulation is FP32. Softmax probabilities are recovered from
the saved forward `LSE` rather than re-softmaxed, so the reference is anchored
to the forward statistics the way a flash-attention kernel is. Backward is
computed from the IO-dtype-rounded `O` that is written to the bundle, not from a
higher-precision pre-conversion value. Rounded intermediates (ALMIOPEN-2704) are
not modelled yet, which `intermediate_rounding: "none"` records.

Regenerating with the same seed and PyTorch version reproduces byte-identical
tensors. The RNG draw order (`Q`, `K`, `V`, `dO`) is part of that contract.

### Supported feature surface

Deliberately narrow; each unsupported feature fails loudly rather than emitting
a wrong golden. Supported: BHSD layout, FP16/BF16, equal query and key/value
head counts, independent `D_qk`/`D_v`, and the causal/sliding-window diagonal
band with either alignment, and an additive attention mask. Not supported yet:
GQA, FP8, BSHD and ragged layouts, GROUP mode, dropout, dBias, and ALiBi. Configurations that leave a
query row fully masked are rejected, because the resulting `-inf` LSE makes
every dependent gradient NaN.

Mask semantics, stride computation, the forward pass and the LSE come from
`utilities/sdpa_common.py`, which `generate_sdpa_fwd_golden.py` also uses, so
the two generators cannot drift apart.

### Tests

```bash
python dnn-providers/integration-tests/reference-data-scripts/tests/test_generate_sdpa_bwd_golden.py
```

```bash
ctest -R hipdnn_golden_sdpa_bwd_generator_python_tests --output-on-failure
```

The tests skip themselves when PyTorch is not installed. They cover that one run
writes both bundles, that the backward bundle's `forward_source` points at the
forward bundle it was produced from, the `matched_precision` strategy, the graph
node's gradient outputs and rank-4 FP32 stats tensor, tensor payload sizes,
seed reproducibility, acceptance by `verify_golden_bundles.py --require-data`,
rejection of unsupported configurations, and agreement with PyTorch autograd.
They also assert that the paired forward bundle is byte-identical to what
`generate_sdpa_fwd_golden.py` produces for the same configuration.
