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
- a ragged tensor's file must hold exactly its `elementSpace` elements (RFC 0014): `1 + sum over the non-batch axes of (extent - 1) * stride`, with `maxRow + 1`, where `maxRow` is the last row any sequence addresses, as the sequence axis extent. The rows come from all offsets and, when the SDPA node names the role's `seq_len`, that tensor's lengths; layouts whose sequences overlap or whose axes alias are rejected
- present tensors (input or output) with floating dtypes must not contain NaN or Inf
- bundle total size must not exceed 2 MiB; larger bundles fail because they would quickly explode test artifact sizes
- discovered metadata sidecars (`meta.json` or `*.meta.json`) must parse and contain non-empty string fields `generator` and `reference_source`
- the optional metadata fields `no_cpu_reference`, `no_gpu_reference` and `expected_rejection` must be JSON booleans when present
- a `<Name>.meta.json` that sets both `no_cpu_reference` and `no_gpu_reference` to `true` must have golden data declared next to `<Name>.json`: a `<Name>.tensors.dvc` or a `<Name>.tensor<uid>.bin`
- a `<Name>.meta.json` that sets `expected_rejection` to `true` (an intentionally invalid graph the engine under test must reject; it is never executed) must not declare golden data: no `<Name>.tensors.dvc` and no `<Name>.tensor<uid>.bin` next to `<Name>.json`
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
`--bias-dims` gives the mask any broadcast shape of `[B, H_q, S_q, S_kv]`, and
`--dbias` adds the `dBias` output (the unscaled `dS` reduced to that shape).

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

### Ragged operands

`--ragged-sides q|kv|qkv` (with `--seq-lens-q` and `--seq-lens-kv`) makes the
Q side (Q, O, dO, dQ, Stats), the K/V side (K, V, dK, dV) or every operand
ragged. Each ragged tensor gets an INT32 offset tensor and a
`ragged_offset_multiplier`. Batch `b` starts at its offset (its origin) and
holds `seq_len[b]` token rows; `offset[B]` is read with the table but neither used nor validated (origins come from
entries 0..B-1, lengths from `seq_len`). The `.bin` holds exactly
the `elementSpace` elements the sequences address (RFC 0014): up to the last row
of the furthest sequence and nothing after it. Elements no sequence addresses
are NaN in outputs and finite poison in inputs. The layout options take
`ROLE=VALUE` (roles `q k v o do dq dk dv stats`) and are repeatable:

| Option | Effect |
|---|---|
| `--ragged-blocks ROLE=N,N,..` | Rows each batch owns (at least `seq_len`, may exceed `S_max`); places the default origins one after another and sets `offset[B]`. |
| `--ragged-origins ROLE=N,N,..` | Explicit origins, in any order, whole tokens, in `--ragged-units`. |
| `--ragged-units elements\|tokens` | Offsets count elements (multiplier 1) or tokens (multiplier = the tensor's sequence stride). |
| `--ragged-token-stride ROLE=N` | Token stride in elements, at least `H*D` (fused or padded tokens). |
| `--ragged-head-major ROLE,ROLE` | Head-major `[H, T]`: token stride `D`, head stride = head pitch rows `* D`. |
| `--ragged-head-pitch ROLE=N` | Head pitch in token rows, at least the last row + 1. Default: the last row + 1 with `--ragged-origins`, else the sum of the blocks. |
| `--stats-layout th\|ht` | Stats token-major `[T, H]` or head-major `[H, T]` (any `H_q`). |

For example, Q-side tensors with batch 1 stored before batch 0 and head-major
Stats whose pitch is wider than the tokens:

```bash
python dnn-providers/integration-tests/reference-data-scripts/generate_sdpa_bwd_golden.py \
  --base-filename <dir>/bwd --forward-base-filename <dir>/fwd \
  --q-dims 2 2 4 2 --v-dims 2 2 4 2 --attn-scale 0.25 \
  --seq-lens-q 2 3 --seq-lens-kv 2 3 --ragged-sides q \
  --ragged-origins q=20,0 --stats-layout ht --ragged-head-pitch stats=10
```

Layouts whose sequences overlap or whose axes alias are rejected. The generator
reads every ragged tensor back by its origins and strides and checks that the
file holds exactly its `elementSpace` elements, that the Stats equal the LSE, and
that the gradients agree with FP64 autograd. `--stats-layout ht` needs Stats to
be ragged (`--ragged-sides q` or `qkv`) and is rejected otherwise.

The forward generator (`generate_sdpa_fwd_golden.py`) authors only bshd ragged
bundles without ragged Stats; head-major, shifted-origin and padded-THD forward
ragged bundles cannot be authored yet.

### Precision and reproducibility

`reference_strategy` is `matched_precision`, RFC 0011's strategy for SDPA
backward: inputs and gradients are stored in the bundle's IO dtype and every
intermediate and accumulation is FP32. Softmax probabilities are recovered from
the saved forward `LSE` rather than re-softmaxed, so the reference is anchored
to the forward statistics the way a flash-attention kernel is. Backward is
computed from the IO-dtype-rounded `O` that is written to the bundle, not from a
higher-precision pre-conversion value. Kernels that round intermediates such as
`P` and `dS` to the IO dtype before their matrix products are not modelled yet,
which `intermediate_rounding: "none"` records.

Regenerating with the same seed and PyTorch version reproduces byte-identical
tensors. The RNG draw order (`Q`, `K`, `V`, `dO`) is part of that contract.

### Supported feature surface

Each unsupported feature fails loudly rather than emitting a wrong golden.
Supported: BHSD layout, FP16/BF16, independent `D_qk`/`D_v`, the
causal/sliding-window diagonal band with either alignment, GQA/MQA
(`H_q` a multiple of `H_kv`), padding (`--seq-lens-q`/`--seq-lens-kv`), an
additive attention mask with `dBias`, explicit-mask dropout (`--dropout-p`), and
ragged operands (`--ragged-sides`, see above).
Padding, GQA, `dBias` and dropout run as single-graph bundles or as sweep
cases of a template that expresses them, and are always cross-checked against
FP64 autograd. Not supported yet:
FP8, BSHD layouts without ragged, Philox dropout, sinks and ALiBi.

Fully masked query rows (padded rows, and rows a band leaves empty in a padded
bundle) give `O = 0`, `LSE = -inf` and zero gradients, as in cuDNN. Without
padding, a band that leaves a row fully masked is rejected.

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

### Regenerating the bundles

`integration-test-bundles/quick/SdpaBackward/generate_golden_data.sh` (and the
`SdpaFwd` one) regenerate the checked-in bundles per tier. The discriminating
bundles (`Sq != Skv` and per-batch lengths that differ, plus GQA, `dBias` and
dropout for backward) are regenerated only with `GENERATE_DISCRIMINATING=1`;
pass the tier `none` to generate them alone.

## `generate_sdpa_fwd_golden.py`

Generates SDPA forward bundles from PyTorch (the MATH backend or an explicit FP32
reference); its module docstring lists the supported options. `--graph-only`
writes the graph and a `.meta.json` whose `inputs` tell the harness how to fill
each input, with no golden tensors, so the bundle is verified against the
reference executor instead (dense BATCH mode only).

## Sliding-window bounds in the generators

A left window bound has two common conventions, and each generator flag is named after the
one it uses:

| Convention | Name | `L = 3` keeps, for the query on the diagonal |
|---|---|---|
| hipDNN graph, cuDNN | `left_bound` (`--left-bound`) | the 3 keys ending at the diagonal, diagonal included |
| FlashAttention, AITER | `window_size_left` (`--window-left`) | the 3 keys before the diagonal, plus the diagonal (4 keys) |

`generate_sdpa_fwd_golden.py` and `generate_sdpa_fwd_paged_golden.py` take either flag:
`--left-bound L` writes `left_bound = L` to the graph, and `--window-left W` writes
`left_bound = W + 1` (it exists for windows quoted in FlashAttention terms, and for
`--validate`, which passes `window_size_left = L - 1` to AITER). The two flags are mutually
exclusive, and the bundle's `.meta.json` records the graph value as `config.left_bound`.
`generate_sdpa_bwd_golden.py` takes `--left-bound` only. `left_bound` must be positive or -1
(unbounded); hipDNN rejects 0 (`SdpaSemanticsRegister.md` SEM-005). The generators' masks
follow cuDNN's rule (SEM-001), so `generate_golden_data.sh` passes every window as
`--left-bound`.

## `generate_sdpa_bwd_bundles.py`

Writes the SdpaBackward sweeps from the SDPA backward test plan
(`docs/test-plans/sdpa-bwd.md`). Nobody edits those bundles by hand.

| File | Holds |
|---|---|
| `sdpa_bwd_plan.py` | levels, axes, legality layers, shape catalog (S, L, M, H, X), ceilings, cost model, tolerance classes and their rule, value domains, purpose rule, window semantics switch |
| `generate_sdpa_bwd_bundles.py` | level rules, graph templates, sweep writer, TOML rows writer, report |
| `sdpa_bwd_calibration.json` | the NumPy magnitude estimates behind the tolerance classes |

From the `rocm-libraries/` repo root:

```bash
python dnn-providers/integration-tests/reference-data-scripts/generate_sdpa_bwd_bundles.py \
  --bundle-dir dnn-providers/integration-tests/integration-test-bundles \
  --toml dnn-providers/hip-kernel-provider/config/ASM_SDPA_ENGINE.toml \
  --toml dnn-providers/hip-kernel-provider/config/hipkernel_AttentionGeneralBwd.toml \
  --report
```

- It writes `{quick,standard,comprehensive,full}/SdpaBackward/<Topology>/` with
  `graph.template.json`, `sweep.json` and `support.json`. It keeps harvested
  claims and drops only the claims of case ids that no longer exist.
- Independent PyTorch golden sweeps live under
  `quick/SdpaBackward/Golden/{Default,AdditiveMask}/`. Regenerate those with
  `quick/SdpaBackward/generate_golden_data.sh`; the plan generator leaves them
  and their paired forward bundles intact.
- A case whose oracle is not ready stays declared in the generator (level
  `notrun`) and is not written. The report lists these cases by gate.
- `--toml` rewrites the generated `[[tolerance_overrides]]` block of an engine
  TOML. Each case id ends with its tolerance class token.
- A run fails if a case's magnitude estimate is missing or stale. Rerun with
  `--calibrate` (needs NumPy) after a plan change that moves a case's shape,
  band, scale or value domain.
- `--dry-run --report` prints the counts, budgets, all-pairs proof, legality
  and gates without writing.

`tests/test_generate_sdpa_bwd_bundles.py` (CTest
`hipdnn_sdpa_bwd_bundle_generator_python_tests`, pytest) fails when the
checked-in sweeps or the TOML block differ from the generator output. It also
checks determinism, the legality of every written case, the all-pairs
coverage, the TOML rows and the traceability of `metadata.plan_items`.
