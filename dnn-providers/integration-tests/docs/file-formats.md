# Integration Test File Formats

This document describes every file that a developer reads or writes in the
cross-provider integration suite. For each file, it shows what the file is, where
it lives, who writes it, and which rules the harness enforces when it reads it.

- To **run** the suite and read its output, see [Running the Tests](running-tests.md).
- To **add** a bundle or **update** claims, see [Adding Tests and Updating Claims](adding-tests.md).

## Map

```
dnn-providers/integration-tests/
  integration-test-bundles/                 # the bundle tree (the test data)
    {tier}/                                 # quick | standard | comprehensive | full
      {Op}/{Layout}/{DataType}/{Name}/      # a single-graph bundle
        {Name}.json                         #   graph
        {Name}.meta.json                    #   metadata sidecar         (required)
        {Name}.support.json                 #   support-claim sidecar    (required)
        {Name}.tensors.dvc                  #   golden-data pointer      (optional)
        {Name}.tensor<uid>.bin              #   golden tensors, via DVC  (not in git)
      {Op}/{Topology}/                      # a template-sweep bundle
        graph.template.json                 #   topology with ${case.*} placeholders
        sweep.json                          #   case matrix (+ required per-case metadata)
        support.json                        #   support-claim sidecar    (required)
        golden/{CaseId}/tensors.dvc         #   golden-data pointer      (optional, per case)
        golden/{CaseId}/tensor<uid>.bin     #   golden tensors, via DVC  (not in git)
  test_categories.yaml                      # CTest tiers for this project's own binaries
  test_categories_external.yaml             # CTest tiers for pre-registered, non-GTest tests

dnn-providers/<provider>/
  config/<ENGINE_NAME>.toml                 # per-engine tolerances, validator overrides, skips
  test_categories_integration.yaml          # CTest tiers for the lane (hip-kernel-provider: one
                                            #   <ENGINE_NAME>_test_categories_integration.yaml per engine)
```

Terms (full list in the [index](README.md#terms)):

- A **bundle** is one graph plus everything that describes it. It is either a
  **single-graph bundle** (one `{Name}.json`) or a **sweep** (one
  `graph.template.json` expanded once per case in `sweep.json`). "Bundle" covers
  both.
- A **case** is one registered test. A single-graph bundle is one case; a sweep is
  one case per `cases[]` entry.

Every bundle carries its graph, its **metadata**, and a **support-claim
sidecar**. Golden data is the only optional part.

## Discovery and test names

The harness walks the bundle root (`--golden-data-dir`, default
`<exe>/../lib/integration-test-bundles/`) and registers:

- **Single-graph bundles:** every `.json` file, except `graph.template.json`,
  `sweep.json`, companion files whose last dotted segment is `meta` or `support`
  (`Small.meta.json`, `Small.support.json`), and anything under a sweep root.
  Other dotted names still count as graphs (`resnet50.v2.json`).
- **Sweeps:** every directory that contains *both* `graph.template.json` and
  `sweep.json`. Each `cases[].id` becomes one test.

To run against part of the tree, point `--golden-data-dir` (`--gd`) at a directory
that holds only the pieces you want. The pieces can be copies or directory
symlinks into an existing tree, for example `quick/SdpaFwd -> <install>/lib/integration-test-bundles/quick/SdpaFwd`.
Discovery follows directory links at any depth. Tests take their names from the
path through the link, so a linked subtree gets the same test names as a copy.
Discovery skips, with a warning, a link that leads the walk back into a directory
that it is already inside. This includes one of the link's own parents and the
parents of `--gd` itself. Discovery also skips, with a warning, a directory that
the run is not allowed to list.

GTest names come from the path, not from anything inside the files. Each path
segment is sanitized (any character other than `[A-Za-z0-9_]` becomes `_`):

| Bundle kind | Suite | Test | Example full name |
|---|---|---|---|
| Single graph | relative directory, segments joined by `_` | file stem | `quick_BatchnormFwdInference_nchw_fp32_Small.Small` |
| Sweep | sweep directory, segments joined by `_` | `cases[].id` | `quick_RMSNorm_Default.2_1_1_1_bfp16_nchw_fa8ec9` |

The tier directory is therefore always the suite prefix (`quick_*`, `standard_*`,
…), which the tier YAML files match on. If two bundles sanitize to the same full
name, registration aborts with `Bundle name collision` and names both paths.
Keep every bundle under a tier directory. If a graph is directly in the data
root, discovery aborts: the binary exits 1 with `Bundle content must live in a
sub-folder of the data root, not at the root itself`. A graph under any other
top-level folder registers under the name of that folder and matches no tier.

Discovery requires no other folder schema. The
`{tier}/{Op}/{Layout}/{DataType}/{Name}` convention below is a convention that
the tools and the verifier maintain. Registration does not enforce it.

## Single-graph bundle — `{Name}.json`

A single-graph bundle holds one concrete graph: the hipDNN frontend graph
serialized to JSON. The top level has graph-wide data types and two arrays:

- `nodes` — each with a `type` (for example `RMSNormAttributes`,
  `SdpaAttributes`), `inputs` / `outputs` mapping each port to a tensor `uid`
  (`null` for an unused optional port), and op-specific `attributes`.
- `tensors` — each with `uid`, `name`, `dims`, `strides`, `data_type`,
  `virtual`, and optionally a baked scalar `value` / `value_type` or
  `is_runtime_pass_by_value`.

Excerpt from `integration-test-bundles/quick/SdpaFwd/bhsd/fp16/hd128_causal_mha/Small/Small.json`:

```json
{
  "nodes": [
    {
      "type": "SdpaAttributes",
      "inputs":  { "q_tensor_uid": 0, "k_tensor_uid": 1, "v_tensor_uid": 2, "attn_mask_tensor_uid": null },
      "outputs": { "o_tensor_uid": 3, "stats_tensor_uid": null },
      "attributes": { "causal_mask": true, "attn_scale_value": 0.08838834764831843 }
    }
  ],
  "tensors": [
    { "uid": 0, "name": "Q", "dims": [2, 32, 2048, 128], "strides": [8388608, 262144, 128, 1], "data_type": "half", "virtual": false }
  ]
}
```

**Written by:** `--capture-bundles` (from a C++ graph test), a per-op golden-data
generator under `reference-data-scripts/`, or `import_graph.py` when a graph does
not fit an existing sweep. Developers rarely write it by hand.

**Tensor names matter.** `validator_overrides` in the engine TOML match the
tensor *name* (`RMSNorm_0::Y`). Names survive capture, but uids do not always.

**Path convention:** `{tier}/{Op}/{Layout}/{DataType}/{Name}/{Name}.json`, where
`Layout` is `nchw`/`nhwc`/`ncdhw`/`ndhwc` (SDPA uses `bhsd`) and `DataType` is
`fp16`/`fp32`/`bfp16`/`fp8`/`int8` (SDPA writes bfloat16 as `bf16`).

## Template-sweep bundle — `graph.template.json` + `sweep.json`

Use a sweep to test one topology across many shapes, dtypes, or layouts. The
template holds the parts that never vary (node types, wiring, uids, `virtual`
flags). Each sweep case holds the parts that vary.

### `graph.template.json`

The JSON is the same as in a single-graph bundle, but any JSON **value** can be a
placeholder string of exactly the form `"${case.<path>}"`. The harness replaces
the whole value. It does not substitute inside a longer string.

```json
{
  "io_data_type": "${case.io_data_type}",
  "tensors": [
    { "uid": 1, "name": "RMSNorm_0::Y", "data_type": "${case.data_type}",
      "dims": "${case.dims}", "strides": "${case.strides}", "virtual": false }
  ]
}
```

Resolution rules (`IntegrationTestBundle.hpp`):

- In any object that has an integer `uid` (a tensor), the harness looks up a
  placeholder first in the entry for that uid in the `values.tensors[]` of the
  case, and then in the case-wide `values`.
- Placeholders named `${case.dims}`, `${case.strides}` or `${case.data_type}`
  inside such an object **must** resolve from the per-tensor entry. A case-wide
  fallback for these three names is an error. Other names
  (`${case.io_data_type}`) can fall back.
- `<path>` can be a dotted path into a nested object in `values`.
- If a placeholder has no value, that case fails to load. The harness drops it
  from registration with an `ERROR` log line. A `values` key that no placeholder
  uses causes a warning (`Unused sweep value '…'`).

### `sweep.json`

```json
{
  "version": 1,
  "cases": [
    {
      "id": "small_fp32_nchw",
      "values": {
        "io_data_type": "float",
        "tensors": [
          { "uid": 0, "data_type": "float", "dims": [2, 3, 4, 5], "strides": [60, 20, 5, 1] }
        ]
      },
      "golden": { "path": "golden/small_fp32_nchw/tensors.dvc" },
      "metadata": {
        "format_version": 1,
        "generator": "generate_batchnorm_fwd_golden.py",
        "reference_source": "PyTorch 2.12.0+rocm7.2",
        "seed": 42
      }
    }
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `cases[]` | yes | One test per entry. |
| `cases[].id` | yes | Non-empty string, unique within the sweep. Becomes the GTest test name. |
| `cases[].values` | no | Placeholder values; `values.tensors[]` holds per-tensor values keyed by `uid` (each `uid` must exist in the template, once). |
| `cases[].metadata` | **yes** | Same schema as a [metadata sidecar](#metadata-sidecar--namemetajson). A case without it fails to load. |
| `cases[].golden` | no | `{"path": "golden/<CaseId>/tensors.dvc"}`. If absent or `null`, the case is graph-only. If present without a string `path`, it is an error. |
| `cases[].tensor_patches` | no | Structural per-tensor edits applied after placeholder expansion; see below. |
| `cases[].add_tensors` | no | Literal tensor objects appended to the expanded graph's `tensors[]`; see [`add_tensors`](#add_tensors-and-node_patches). |
| `cases[].node_patches` | no | Edits to one section of a node (flags, scalars, tensor uids), applied after `add_tensors`; see [`add_tensors` and `node_patches`](#add_tensors-and-node_patches). |

Case ids are handles, not descriptions. `import_graph.py` generates them as
`{shape}_{dtype}_{layout}_{≤3 salient attrs}[_{hash6}]`. It adds the `hash6`
suffix only when the readable part is not already unique.

**Do not write cases by hand.** Generate them with `import_graph.py` (see
[Adding Tests](adding-tests.md#add-a-bundle)). You can edit a field of an existing
case, for example to add its `golden` pointer. Do not rename ids without a
reason, because support-claim sidecars and the `ffm-quick` tier of
miopen-provider name cases by id.

#### `tensor_patches`

This field is for a case that differs from the template by more than a
substitution. For example, a tensor must become runtime pass-by-value, so it must
not carry a baked `value`:

```json
"tensor_patches": [
  { "uid": 4, "set": { "is_runtime_pass_by_value": true }, "remove": ["value", "value_type"] }
]
```

`set` upserts fields. Then `remove` erases fields. The `uid` must exist in the
expanded graph. If a tensor keeps `is_runtime_pass_by_value: true` and a `value`,
it fails validation at load.

A patch can also set a tensor's declared `alignment` (bytes, an integer of at least 1),
which a template otherwise hard-codes (the quick Matmul template declares 16):

```json
"tensor_patches": [
  { "uid": 1, "set": { "alignment": 4 } }
]
```

A gauntlet [space](running-tests.md#gauntlet-fuzz-runs) writes this patch from its
`alignment` axis, and `import_graph.py` writes it back when it freezes a graph whose
tensor alignment differs from the template's, so a frozen finding keeps the alignment
it failed at. Pair it with [`placement`](#placement-and-placement_uid) to put the
device pointers there.

#### `add_tensors` and `node_patches`

These two fields let a case change the graph's *topology* where a template cannot: switch an
optional input or output on (an extra tensor and the node key that names it) or change a node
flag. They are meant for generated cases (a gauntlet space switching an SDPA feature per
sample) but any case can use them. A case that has neither field loads exactly as before.

The harness applies a case's fields in this order, after placeholder expansion and before the
flatbuffer is built:

1. `add_tensors`
2. `node_patches`
3. [`tensor_patches`](#tensor_patches)

So a node patch can wire in a tensor that `add_tensors` added, and a tensor patch can edit
it. A patch on a uid that nothing has added is an error.

**`add_tensors`** is an array of tensor objects, in the same JSON as a template tensor,
with no placeholders (the values are final). Each is appended to `tensors[]`. A tensor
without an integer `uid`, an entry that is not an object, or a `uid` the graph already has (from the
template or from an earlier entry) is an error.

**`node_patches`** is an array of entries:

| Field | Required | Meaning |
|---|---|---|
| `node` | yes | The node to edit: its `name` (a string; it must be unique in `nodes[]`) or its index in `nodes[]` (an integer). An unknown name, an out-of-range index and a name that two nodes share are errors; use the index for the last. |
| `section` | yes | The object of the node that holds the keys: `attributes`, `parameters`, `inputs` or `outputs`. The node must already have it; a missing section is an error and is never created. Look at the template: SDPA forward keeps its flags under `attributes` and SDPA backward under `parameters`; both keep tensor uids under `inputs` and `outputs`. |
| `set` | one of `set`, `remove` | Object of key to new value. **Every key must already exist in the section**; setting a key the section lacks is an error, not an insert. A template lists each key of a node, nulls included, so every legal key is there. The harness refuses an unknown key because the JSON-to-flatbuffer converters ignore keys they do not know: a misspelled key would build the unchanged graph, and the case would pass without the feature it asked for. |
| `remove` | one of `set`, `remove` | Array of key names to erase, after `set` has run. Removing a key the section lacks is an error. |

Any other key in an entry is an error. Every error is an exception that names the sweep
case and the cause, for example
`sweep case 'stats_on': node_patches[1]: set key 'generate_stat' is not in the 'attributes'
section of node 'SdpaFwd_0' (set only overwrites existing keys)`. When a case is loaded to
run, the harness reports any such failure only as `INVALID_SWEEP_CASE`, as for any other bad
case.

Example: an SDPA forward case that switches stats on. It adds the log-sum-exp tensor, names
it in the node's outputs, and sets the flag. The template's `outputs.stats_tensor_uid` and
`attributes.generate_stats` are `null` and exist, so both `set`s are overwrites:

```json
"add_tensors": [
  { "uid": 4, "name": "SdpaFwd_0::Stats", "data_type": "float",
    "dims": [1, 2, 16, 1], "strides": [32, 16, 1, 1],
    "alignment": 16, "is_runtime_pass_by_value": false, "virtual": false }
],
"node_patches": [
  { "node": "SdpaFwd_0", "section": "outputs",    "set": { "stats_tensor_uid": 4 } },
  { "node": "SdpaFwd_0", "section": "attributes", "set": { "generate_stats": true } }
]
```

## Metadata sidecar — `{Name}.meta.json`

This file holds provenance and run guards for a single-graph bundle. **Every
single-graph bundle has one.** A sweep case carries the same object inline as
`cases[].metadata`, which every case requires. `src/harness/BundleMetadata.hpp`
parses it.

| Field | Type | Effect |
|---|---|---|
| `format_version` | int | **Required, must be `1`.** Otherwise the whole metadata object is rejected. |
| `enforcement_level` | string | `applicability`, `buildable`, or `full` (the default when absent). How far a run must get for a support claim to count as confirmed; see [Support claims](#support-claim-sidecars--namesupportjson-and-supportjson). Any other value rejects the metadata. |
| `purpose` | string | `claim`, `probe` or `regression`; absent when not set. Why the case exists: a graph an engine claims, a legal graph whose expected outcome is "decline with a reason, or accept and be correct", or a frozen fuzz finding. A tag only: it changes no verdict, and the [outcome report](running-tests.md#outcome-report) copies it into each record. Any other value rejects the metadata. |
| `minimum_vram_mb` | int | Skip the case when the device has less VRAM. |
| `gpu_architecture` | string | Golden data is locked to an arch. Skip the case unless the base arch token of the device matches (`gfx942` matches `gfx942:sramecc+:xnack-`, not `gfx940`). If absent, the data is portable. |
| `seed` | int | Seed for generated inputs. |
| `inputs` | object | Per-input fill recipes keyed by tensor uid as a string (`"3"`); see [`inputs`](#inputs--fill-recipes). A non-numeric key, or a recipe that does not parse, rejects the metadata. The O and stats inputs of an SDPA backward node are the exception: after the fill they are recomputed as the forward of its Q, K and V, under its scale, diagonal band and additive mask, so a recipe for them has no effect. A node whose forward the fill cannot derive (ALiBi, padding, sequence lengths, dropout, ragged tensors, mixed or unsupported data types, a scale tensor that is not a pass-by-value scalar, a Q, K, V or mask computed by the graph) makes the bundle unverifiable, with the reason; `deriveSdpaBackwardForwardOutputs()` is the authority. Shapes the CPU reference rejects, such as stats not shaped `[B, H, Sq, 1]`, keep the recipe, and the CPU reference then fails the bundle. |
| `validator` | string | `default` (when absent) or `match_nonfinite`: how float outputs are graded beyond the tolerance; see [`validator`](#validator). Any other value rejects the metadata. |
| `placement` | string | `default` (when absent), `exact` or `under`: where the engine run's device pointers sit relative to each tensor's declared `alignment`; see [`placement` and `placement_uid`](#placement-and-placement_uid). Any other value rejects the metadata. |
| `placement_uid` | int | The tensor `placement: "under"` misaligns. **Required** when `placement` is `under`; ignored by every other placement, so a space can always render it. A non-integer rejects the metadata. |
| `generator`, `generator_version`, `generated_at`, `reference_source`, `reference_source_hash`, `reference_strategy`, `rocm_version`, `operation`, `generation_command`, `notes` | string | Provenance only; no effect on the run. |

A bundle without its metadata is an error. These fields are required:

- `format_version` (`1`) — the harness rejects the metadata without it.
- `generator` and `reference_source`, non-empty — `reference-data-scripts/verify_golden_bundles.py`
  requires them. They show a reviewer where the graph and its expected values
  came from.

Generators can write extra keys (`config`, `input_range`, …). The harness ignores
keys that it does not know.

### `inputs` — fill recipes

`inputs` maps a tensor uid to the recipe that fills that input tensor. A recipe
replaces the harness's default fill for that tensor; every input without one keeps its
default.

```json
"inputs": {
  "0": { "kind": "nonzero", "lo": -1, "hi": 1 },
  "2": { "kind": "zeros" },
  "3": { "kind": "special", "special": "nan", "fraction": 0.05 },
  "4": { "seed": 7 }
}
```

| `kind` | Fields it reads | Every element is |
|---|---|---|
| `free` | `lo`, `hi` (default -1, 1) | uniform over `[lo, hi]` |
| `fixed` | `value` (default 0) | `value`. This is the constant fill. |
| `nonzero` | `lo`, `hi` (default -1, 1; `lo <= hi`) | a `free` draw that is not zero in the tensor's data type (redrawn while it rounds to zero, then the endpoint of larger magnitude). Unsupported when that endpoint is itself zero in the data type. For integer types the draw is uniform over the integers of `[lo, hi]` that the type can hold, excluding zero; it is unsupported when none remain (a range wholly outside the type, an empty one, or one holding only zero). |
| `zeros` | none | zero. Unsupported for `fp8_e8m0`, which has no zero. |
| `large` | `scale` (**required**, `> 0`) | a magnitude uniform in `[top/2, top]` with a random sign, where `top = min(scale, the data type's largest finite value)`. Never infinite. Integer types need `scale >= 1`. |
| `subnormal` | none | a subnormal value of the tensor's data type, random sign where signed. Unsupported for integers and for types without subnormals (`fp8_e8m0`). |
| `special` | `special` (**required**: `"nan"`, `"inf"` or `"-inf"`), `fraction` (**required**, `[0, 1]`), `lo`, `hi` (default -1, 1; `lo <= hi`) | `round(fraction * elements)` elements, and at least one when `fraction > 0`, are the non-finite `special`; the rest are `free` over `[lo, hi]`. Unsupported for integers, and for a value the type lacks (NaN for fp4 and fp6; infinity for fp4, fp6, `fp8_e4m3`, `fp8_e4m3_fnuz`, `fp8_e5m2_fnuz` and `fp8_e8m0`). |

Any entry may add `seed`, an unsigned 32-bit integer that seeds that tensor's draws. An
entry with a `seed` and no `kind` changes only the seed.

**The parse is strict.** An unknown `kind` (there is no `constant`; use `fixed`), a
`kind` that is not a string, an entry that is not an object, an entry with neither `kind` nor `seed`, a field the kind reads
that is not a finite number in the range of a 32-bit float, a bad `seed`, `lo > hi`
(`nonzero`, `special`), a `large` without `scale > 0`, or a `special` without a valid
`special` and a `fraction` in `[0, 1]` is a metadata error: the case registers as a
failing test that names the uid and the field, and no entry of that case is applied.
**Fields a kind does not read are ignored**, whatever they hold, so a space can render
every field for every kind and select the kind with one expression.

Whether a data type can hold a kind is known only when the tensor is filled. A kind the
type cannot hold (the "Unsupported" entries above) makes the case `UNEXERCISED`, never a pass.

### `validator`

| Value | Grading |
|---|---|
| `default` | The tolerance resolved for each tensor, unchanged: the plain allclose and RMS validators fail on any NaN or Inf. |
| `match_nonfinite` | Non-finite elements are matched by position; finite elements keep the resolved tolerance. |

`match_nonfinite` classifies each position by the reference and engine values:

- NaN against NaN matches, whatever the payloads and signs. Inf against Inf of the
  **same sign** matches. A matched position is excluded from the tolerance math.
- NaN against a finite value, a finite value against Inf, `+Inf` against `-Inf` and NaN
  against Inf all fail.
- Finite against finite is graded as usual: `atol + rtol * |reference|` for allclose, the
  relative RMS error for an RMS tensor. The resolved kind (including a per-tensor
  [TOML](#per-engine-test-config--configengine_nametoml) override) stays in force.
- A reference with **no finite element** (every element NaN or Inf) **fails**: with the
  finite part empty there is nothing to grade, and a vacuous pass would hide garbage on
  both sides.
- **Unwritten outputs.** With the mask on, NaN matches NaN, so an element the engine
  never wrote would pass. The engine run's output buffers are therefore prefilled with a
  NaN carrying a distinctive payload, and an engine element still holding it fails. The
  reference's buffers keep the plain NaN sentinel.
- It applies to `float`, `half`, `bfloat16` and `double` outputs only; every other
  output (integer, fp8, sub-byte) is graded as under `default`.
- It exists only on the host. Under `--validator auto` the case's comparison is
  **forced to the host**, even for a GPU reference's output, as `--validator cpu` does.
  An explicit `--validator gpu` contradicts it: the case fails before the engine runs, as
  `UNEXERCISED` at stage `setup`, with a message naming `--validator auto` and
  `--validator cpu` as the options.

Use it with the `special` fill above, so the case can expect non-finite output.

### `placement` and `placement_uid`

`placement` moves the engine run's device buffers relative to each tensor's declared
`alignment` N (see [`tensor_patches`](#tensor_patches)). It changes only the **engine
run**: a reference run is never placed, and nothing is placed when the run has no device
buffers (a host input policy).

| Value | Engine run's device pointers |
|---|---|
| `default` | Where the allocator put them. |
| `exact` | Every device pointer is aligned to its tensor's N and not to 2N. |
| `under` | The pointer of `placement_uid` is aligned to N/2 and not to N, so a backend that enforces N must reject it; every other pointer stays where it is. Needs N >= 2 for that tensor. |

`placement_uid` is **required** for `under` (the metadata is rejected without it) and
ignored for `default` and `exact`. For `under` the case expects the backend to refuse
that one pointer with `BAD_PARAM`; see [the finished record](#finished-record-fields).

A tensor stays where it is, with an `INFO` log line, when it is a runtime pass-by-value
scalar (a host pointer), an input of the sub-byte packed set, or has no declared
alignment. A `placement_uid` that names such a tensor, or no tensor of the run, or an
`under` tensor with N < 2, cannot be set up: the case is `UNEXERCISED` at stage `setup`.

### Finished-record fields

The harness echoes these into the `finished` record of the
[outcome report](running-tests.md#outcome-report), so a passing run shows what it was
asked to do:

| Field | Content |
|---|---|
| `inputs` | The recipes the harness parsed from `metadata.inputs`, keyed by uid; absent when the case sets none. |
| `validator` | `"default"` or `"match_nonfinite"`, as the metadata spells it. |
| `placement` | `"default"`, `"exact"` or `"under"`. |
| `placement_uid` | Present when the metadata sets one. |
| `expected_status` | `"BAD_PARAM"` on a pass that verified a rejection; absent otherwise. |
| `kernels` | The distinct kernel names the engine reported while the case ran, sorted. From the engine's `kernel: <engine> <kernel>` INFO line, written once per built plan; see [`kernels` in the outcome report](running-tests.md#outcome-report). Absent when the case reported none (it never built a plan, or the engine logs no kernel). Not a metadata field: the harness fills it. |

`validator`, `placement` and `placement_uid` are absent from the record of a bundle
that failed to load, which has no parsed metadata.

**The expected `BAD_PARAM` rule.** A case with `placement: "under"` passes only if the
engine accepted the graph and the backend then rejected the misaligned pointer of
`placement_uid` with `BAD_PARAM` for that uid. That is `PASS` at depth `verified` with
`expected_status: "BAD_PARAM"`, and nothing was compared. Every other result is a
failure: the backend executed the graph with the pointer below its alignment, the engine
declined after being handed the graph, or `BAD_PARAM` named another uid or another error.

## Golden data — `.tensors.dvc` and `.bin`

Golden tensors are optional. A bundle without them is **graph-only** and is
verified against the GPU or CPU reference executor instead. See
[Verification modes](running-tests.md#verification-modes).

- **Single graph:** `{Name}.tensor<uid>.bin`, one per tensor, all listed in one
  `{Name}.tensors.dvc` pointer.
- **Sweep:** `golden/{CaseId}/tensor<uid>.bin`, listed in
  `golden/{CaseId}/tensors.dvc`, referenced from the case's `golden.path`.

The `.bin` files are raw tensor bytes in the tensor's declared `data_type`.
Developers never commit them (`*.bin` is git-ignored). Git tracks only the `.dvc`
pointer. The pointer is a YAML list of outputs with their md5 and size:

```yaml
outs:
- path: Small.tensor0.bin
  md5: 8510f215d557db558ee2f131ed05d2e3
  size: 33554432
  hash: md5
  remote: golden-data
```

The repository's `.dvc/config` configures two DVC remotes:

| Remote | URL | Used by |
|---|---|---|
| `storage` (default) | `s3://therock-dvc/rocm-libraries` | legacy ops (pointers with no `remote:` key) |
| `golden-data` | `s3://therock-dvc/rocm-libraries/hipdnn/golden-data` | new hipDNN ops (pointers carry `remote: golden-data`) |

Each output names its own remote, so a bare `dvc pull` fetches everything. Both
remotes allow anonymous reads. To push, you need AWS credentials. The DVC
commands are in [Adding Tests](adding-tests.md#golden-data-with-dvc).

## Support-claim sidecars — `{Name}.support.json` and `support.json`

A support claim is a checked-in promise that a named **engine accepts** a graph
on a given **arch** and **platform**. It asserts acceptance, not correctness. The
output comparison checks correctness. A claim changes "the engine quietly stopped
supporting this graph" from a skip that nobody reads into a test failure.

| Bundle kind | Sidecar | Shape |
|---|---|---|
| Single graph `dir/Small.json` | `dir/Small.support.json` | `claims: { ENGINE: { arch: [platforms] } }` |
| Sweep `dir/sweep.json` | `dir/support.json` (one file for the whole sweep) | `claims: { ENGINE: [ { cases: [ids], support: { arch: [platforms] } } ] }` |

Single graph (`quick/BatchnormFwdInference/nchw/bfp16/Small/Small.support.json`,
one of its two engines shown), in the exact on-disk form:

```json
{
  "claims": {
    "HIP_MLOPS_ENGINE": {
      "gfx1151": [
        "windows"
      ],
      "gfx90a": [
        "linux"
      ],
      "gfx942": [
        "linux"
      ]
    }
  },
  "version": 1
}
```

Sweep (`quick/Layernorm/Variant2/support.json`, case list shortened):

```json
{
  "claims": {
    "HIP_MLOPS_ENGINE": [
      {
        "cases": [
          "2_2_3_2_2_bfp16_ncdhw_normalized_dim_count1",
          "2_2_3_2_2_bfp16_ncdhw_normalized_dim_count2"
        ],
        "support": {
          "gfx1151": [
            "windows"
          ],
          "gfx942": [
            "linux"
          ]
        }
      }
    ]
  },
  "version": 1
}
```

A bundle with no claims yet (see
[Adding Tests](adding-tests.md#when-no-engine-accepts-the-graph-yet)):

```json
{
  "claims": {},
  "version": 1
}
```

Rules:

- `version` must be `1`.
- **Engine** is the engine name the run passes as `--test-engine`
  (`MIOPEN_ENGINE`, `HIPBLASLT_ENGINE`, `HIP_MLOPS_ENGINE`, `ASM_SDPA_ENGINE`, …).
- **Arch** is the *base* token of the device — the part of `gcnArchName` before
  the first `:`. It matches exactly. `gfx942` covers `gfx942:sramecc+:xnack-`. It
  does not cover `gfx940`. There are no family wildcards.
- **Platform** is `linux` or `windows`.
- In a sweep sidecar, a case id must not appear in more than one group for each
  engine. A case that is in no group is unclaimed.
- **Every new bundle gets a sidecar.** A sidecar claims nothing about engines,
  archs, or platforms that it does not list. The absence of a claim is not a
  claim of non-support. A bundle that no engine accepts yet carries the empty
  sidecar above. A bundle with **no** sidecar is invisible to the claim
  machinery: it gets no verdict, not even `unclaimed_support`. Nothing enforces
  this rule. The run treats a missing sidecar as "no claims", and
  `verify_support_claims.py` checks the sidecars that exist, not the missing
  ones. Older bundles break the rule: most SDPA single-graph bundles and a few
  sweeps (`quick/SdpaBackward/Default`, `quick/Reduction/Default`,
  `quick/Pointwise/Binary`, `{quick,full}/BlockScaleDequantizeMatmul/Default`)
  have none.
- Sidecars are **machine-written** by `--write-support-claims`. The form is
  stricter than valid JSON: sorted keys, 2-space indent, one array element per
  line, trailing newline, LF line endings, sorted platforms and case ids, and
  one group per identical `support` map, with groups ordered by their first case
  id. The verifier rejects anything that is not `json.dumps(…, indent=2,
  sort_keys=True)` output. The writer rewrites anything that is not in its own
  form on its next run. Therefore, hand edits are limited to *retracting* a claim
  or writing an empty sidecar, in exactly that form.

`scripts/verify_support_claims.py` checks these items:

- the schema
- the canonical form
- that every claimed sweep case id exists in the sibling `sweep.json`
- that no id repeats for each engine
- that graphs with non-empty claims do not declare an unknown `enforcement_level`
- that no sidecar is orphaned (a `X.support.json` with no `X.json`, or a
  `support.json` outside a sweep root)

The script is registered as the `verify-support-claims` pre-commit hook. But the
pre-commit configuration of the repository currently excludes
`integration-test-bundles/` from every hook. After you change sidecars or
sweeps, **run it by hand**:

```bash
python dnn-providers/integration-tests/scripts/verify_support_claims.py
```

[Running the Tests](running-tests.md#support-claim-summary) describes how the
harness checks claims during a run, and every verdict that a run can report.
[Support Claim Enforcement](support-claim-enforcement.md) describes the harness
lifecycle behind it.

## Per-engine test config — `config/<ENGINE_NAME>.toml`

Each provider owns one TOML file for each engine. The file is passed to the
binary as `--test-config` (CTest lanes pass it for you). It changes how the
harness tests *that engine*. You do not recompile or change the bundles.

| Provider | Files |
|---|---|
| miopen-provider | `config/MIOPEN_ENGINE.toml` |
| hipblaslt-provider | `config/HIPBLASLT_ENGINE.toml` |
| hip-kernel-provider | `config/HIP_MLOPS_ENGINE.toml`, `config/ASM_SDPA_ENGINE.toml` |

```toml
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*_ConvolutionWrw_Default.8_16_16_16_16_bfp16_ndhwc_dil1x1x1_postpad0x0x0_prepad0x0x0"]
atol = 1.19
rtol = 0.2

[[validator_overrides]]
filters       = ["*LayernormBackward*"]
tensors       = ["*::DSCALE", "*::DBIAS"]
validator     = "rms"
rms_threshold = 1e-4

[[test_skips]]
archs     = ["gfx90a"]   # optional; omit to match every arch
platforms = ["windows"]                             # optional; omit to match both
filters   = ["*_SdpaFwd_Default.*_128_bfp16_*"]
reason    = "no applicable kernel for this shape on this arch"
```

| Table | Rules |
|---|---|
| `[meta]` | `version = 1` is required. A missing or unsupported version is a load error. The harness never ignores it silently. |
| `[[tolerance_overrides]]` | `filters`, `atol` and `rtol` are all required. **Later entries win** when several match. |
| `[[validator_overrides]]` | Applies when a `filters` glob matches the test **and** a `tensors` glob matches the output tensor's label (its name, or `uid=N` when unnamed). `validator` is `allclose`, `allclose_matching_infinities` or `rms`. `rms_threshold` is required and positive for `rms`, and forbidden for the other two. **Later entries win.** Allclose is the default everywhere. Only this table changes it. Use it for outputs where per-element comparison is the wrong question, not to get more tolerance. See the validators below. |
| `[[test_skips]]` | `filters` and `reason` are required. **The first matching entry wins** — this is the opposite order from the two tables above. `archs` is a substring match against the raw `gcnArchName` (so `gfx11` covers `gfx1100` and `gfx1151`). `platforms` is `linux` / `windows`. The skip message is `[arch <current gcnArchName>] <reason>`. |

`filters` are globs that match the full GTest name. This is the same string that
`--gtest_filter` matches, but the syntax is not `--gtest_filter` syntax: `:` does
not separate alternatives and a leading `-` does not negate. Use one array
element for each pattern. Use only `*` and `?` as wildcards. Linux matches with
`fnmatch` (case-sensitive). Windows matches with `PathMatchSpecA`
(case-insensitive).

The two non-default validators:

- **`rms`** — compares the aggregate relative-RMS error against
  `rms_threshold`. Use it for long reductions (layernorm/RMSNorm backward
  `dscale`/`dbias`). Their elements can cancel toward zero, so the per-element
  relative error is unbounded.
- **`allclose_matching_infinities`** — grades exactly as `allclose` at the same
  atol/rtol, with one exception: an element that is infinite with the *same sign*
  in both the reference and the engine output compares equal. NaN,
  opposite-signed infinities, and finite-versus-infinite still fail. Use it for
  outputs whose correct value is infinite on both sides, such as the log-sum-exp
  of a fully masked SDPA row. It is **host-only**. A tensor that it selects fails
  with `Validator override NOT APPLICABLE ON DEVICE` whenever its comparison runs
  on the device. Under `auto`, this is every run where the GPU reference
  produced the expected values. Run such a config with `--validator cpu`.
  Alternatively, narrow the `tensors` glob.

Neither validator is defined for integer outputs. If a glob selects an integer
output, that tensor fails with a message that names the glob. Narrow the glob to
float outputs.

The TOML applies to bundle and C++ graph tests alike. It never applies to
`hipdnn_golden_data_tests`. The config of an engine describes how far that engine
is allowed to drift. It cannot skip or loosen a check on our own golden data. The
full schema is in `src/harness/TestSettings.hpp`.

A `test_skips` entry records a *known* engine limitation. It skips in `SetUp()`,
before the support-claim query. Therefore, the harness neither enforces nor
refreshes a claim on a skipped case on that arch.

## Tier YAML files — `test_categories*.yaml`

These files map GTest name patterns to CTest labels and timeouts.
`shared/ctest/parse_test_categories.py` parses them at configure time.
`shared/ctest/README.md` is the reference for the format. There are three scopes.
Do not confuse them:

| File | Governs |
|---|---|
| `dnn-providers/integration-tests/test_categories.yaml` | This project's own binaries (`hipdnn_integration_tests_unit_tests`, `hipdnn_gpu_ref_tests`, `hipdnn_golden_data_tests`) |
| `dnn-providers/integration-tests/test_categories_external.yaml` | Pre-registered non-GTest CTest tests (the Python verifiers, test-name validation) |
| `dnn-providers/<provider>/test_categories_integration.yaml` (hip-kernel-provider: `HIP_MLOPS_ENGINE_test_categories_integration.yaml`, `ASM_SDPA_ENGINE_test_categories_integration.yaml`, `hipkernel_Gfx950AttentionDense_test_categories_integration.yaml`) | `hipdnn_integration_tests` **run against that provider's engine** — the lane's bundle suites |

A provider's own `test_categories.yaml` governs its native `*_plugin_tests`
binaries, not the shared suite.

```yaml
test_categories:
  quick:
    description: "quick-tier bundle sweeps plus always-built C++ tests"
    test_patterns: ["quick_*", "Smoke/*", "Quick/*"]
    exclude: ["*DISABLED*"]
    labels: ["quick", "pre-commit", "smoke"]

exclude_gpu:
  exclude_gpu_gfx110X_windows:
    test_patterns: ["*"]
    labels: ["quick", "standard", "ex_gpu_gfx110X"]

execution_settings:
  default_timeout: 300
  category_timeouts:
    quick: 1200
```

- `test_patterns` / `exclude` are GTest globs. `/` and `_` are literal. Therefore,
  bundle suites (`quick_*`) and C++ suites (`Smoke/*`) need separate patterns.
- Each category becomes one CTest test named `<prefix>_<category>_suite` with its
  `labels`.
- Tier inclusion is explicit, not inherited. In the provider files, the
  `standard` category lists the quick patterns *and* the standard patterns.
  Labels do not cascade on their own, so check the file that you edit.
  [Running the Tests](running-tests.md#test-tiers) has a table of what each
  provider tier selects.
- Keep `Smoke/*` and `Quick/*` in the quick tier of a provider, even on a default
  build. The always-built C++ tests register under those prefixes. The C++ graph
  tests also register under them when a branch builds them.
- `exclude_gpu_<arch>[_windows|_linux]` adds a per-arch variant suite
  (`<prefix>_<category>_<arch>_suite`, label `ex_gpu_<arch>`) that has the
  negative patterns of that arch. A gtest filter cannot express a
  match-everything pattern (`"*"`). Therefore, the variant is registered
  `DISABLED` and is never launched. Variants are generated for each `ex_gpu_*`
  label declared in the file, not for the arch that you build. To select one, use
  `ctest -L ex_gpu_<arch>`.
- miopen-provider's `ffm-quick` lists exact test names (sweep case ids).
  Regenerating a sweep can rename those ids and silently shrink that tier.

## Where each format is enforced

| Format | Parsed / checked by | When it fails |
|---|---|---|
| Graph `.json` | `IntegrationTestBundle.hpp` | Case dropped from registration with an `ERROR` log line; the run stays green with one test fewer |
| `graph.template.json` + `sweep.json` | `BundleDiscovery.hpp`, `IntegrationTestBundle.hpp` | A malformed `sweep.json` or a bad/duplicate id aborts registration; a bad case (missing placeholder value, missing metadata, bad `golden.path`) is dropped like a bad graph |
| `.meta.json` / `cases[].metadata` | `BundleMetadata.hpp`; `reference-data-scripts/verify_golden_bundles.py` | Required for new bundles and every sweep case, with `generator` and `reference_source`. **Malformed** metadata (invalid JSON, a wrong field type, an unsupported `format_version`, a non-integer `inputs` key) throws `BundleMetadataError`: that bundle registers as a **failing** test whose message names the file or case and what is wrong. This holds for every bundle type, graph-only bundles included. **Absent** metadata behaves as follows. At run time a sweep case without a `metadata` block is dropped; a single-graph bundle without `.meta.json` loads with empty metadata, unless it has golden `.bin` files, in which case it fails. Some SDPA single-graph bundles have no `.meta.json`, and many older sweeps lack `generator` or `reference_source`, so the verifier fails on the whole tree; run it on the directories you touched ([details](adding-tests.md#check-the-data-itself)) |
| `.support.json` / `support.json` | `SupportClaims.cpp` at run time; `scripts/verify_support_claims.py` by hand | Required for new bundles, but not enforced: a missing sidecar means "no claims" at run time and is not reported by the verifier. A sidecar that does not parse, and a broken claim, fail that bundle's test |
| Golden `.bin` / `.dvc` | `IntegrationTestBundle.hpp`; `reference-data-scripts/verify_golden_bundles.py` | Falls back to a reference in `auto` mode; fails in `golden` mode. A blob that is unreadable or has the wrong size fails the test that needs it, when that test runs |
| Engine `.toml` | `TestSettings.hpp` | Binary exits 1 at startup |
| `test_categories*.yaml` | `shared/ctest/parse_test_categories.py` | CMake **warning** at configure; that target's tier suites are not generated, so `ctest -L <tier>` runs fewer tests, with no other notice. Check the configure log or `ctest -N -L <tier>` |

The load problems that register a *failing* test are these: a tensor marked
`is_runtime_pass_by_value` that still carries a baked `value`, and malformed
metadata (above).
