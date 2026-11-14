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
        {Name}.support.json                 #   support-claim sidecar    (required; none for an expected-rejection bundle)
        {Name}.tensors.dvc                  #   golden-data pointer      (optional)
        {Name}.tensor<uid>.bin              #   golden tensors, via DVC  (not in git)
      {Op}/{Topology}/                      # a template-sweep bundle
        graph.template.json                 #   topology with ${case.*} placeholders
        sweep.json                          #   case matrix (+ required per-case metadata)
        support.json                        #   support-claim sidecar    (required)
        golden/{CaseId}/tensors.dvc         #   golden-data pointer      (optional, per case)
        golden/{CaseId}/tensor<uid>.bin     #   golden tensors, via DVC  (not in git)
  test_categories.yaml                      # CTest tiers for the harness unit tests and golden-data binaries
  test_categories_gpu_ref.yaml              # CTest tiers for hipdnn_gpu_ref_tests
  test_categories_external.yaml             # CTest tiers for pre-registered, non-GTest tests

dnn-providers/<provider>/
  config/<ENGINE_NAME>.toml                 # per-engine tolerances, validator overrides, skips
  test_categories_integration.yaml          # CTest tiers for the lane (hip-kernel-provider: one
                                            #   <ENGINE_NAME>_test_categories_integration.yaml per engine,
                                            #   except hipkernel_Gfx950Attention_..., shared by the three
                                            #   gfx950 attention engines, and one hipkernel_flydsl_<op>_...
                                            #   per FlyDSL op family)
```

Terms (full list in the [index](README.md#terms)):

- A **bundle** is one graph plus everything that describes it. It is either a
  **single-graph bundle** (one `{Name}.json`) or a **sweep** (one
  `graph.template.json` expanded once per case in `sweep.json`). "Bundle" covers
  both.
- A **case** is one registered test. A single-graph bundle is one case; a sweep is
  one case per `cases[]` entry.

Every bundle carries its graph, its **metadata**, and a **support-claim
sidecar**. Golden data is the only optional part. An
[expected-rejection bundle](#expected-rejection-bundles) is the exception: it
carries no support claim and no golden data.

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
| `version` | no | `1` or `2`; `1` when absent. A case with [`absent_tensors`](#optional-tensors--absent_tensors) needs `2`. Any other value fails every case of the sweep. |
| `cases[]` | yes | One test per entry. |
| `cases[].id` | yes | Non-empty string, unique within the sweep. Becomes the GTest test name. |
| `cases[].values` | no | Placeholder values; `values.tensors[]` holds per-tensor values keyed by `uid` (each `uid` must exist in the template, once). |
| `cases[].metadata` | **yes** | Same schema as a [metadata sidecar](#metadata-sidecar--namemetajson), including `no_cpu_reference`, `no_gpu_reference` and `expected_rejection`. A case without it fails to load. For rule R4, the `golden` object with a `path` of the same case counts as declared golden data. A case with `expected_rejection: true` must not have a `golden` object: that is a load error ([details](#expected-rejection-bundles)). |
| `cases[].golden` | no | `{"path": "golden/<CaseId>/tensors.dvc"}`. If absent or `null`, the case is graph-only. If present without a string `path`, it is an error. |
| `cases[].tensor_patches` | no | Structural per-tensor edits applied after placeholder expansion; see below. |
| `cases[].absent_tensors` | no | Uids of template tensors that this case does not have; see [Optional tensors](#optional-tensors--absent_tensors). Needs sweep `version` 2 and case metadata with `format_version` 2. |

Case ids are handles, not descriptions. `import_graph.py` generates them as
`{shape}_{dtype}_{layout}_{≤3 salient attrs}[_{hash6}]`. It adds the `hash6`
suffix only when the readable part is not already unique.

**Do not write cases by hand.** Generate them with `import_graph.py` (see
[Adding Tests](adding-tests.md#add-a-bundle)). You can edit a field of an existing
case, for example to add its `golden` pointer. Do not rename ids without a
reason, because support-claim sidecars name cases by id.

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

#### Optional tensors — `absent_tensors`

One template can hold the union of the tensors of a set of cases, for example
with and without a sink, a bias or Stats. Each case then names the template
tensors that its graph does not have:

```json
"absent_tensors": [7, 9]
```

For that case, the harness removes each named tensor from a copy of the
template. It sets each node port that holds the uid to `null`. It does this
before it expands the placeholders, so an absent tensor needs no case values.
Nothing else in the template changes, and no uid changes.

A **node port** is a key whose name ends in `_tensor_uid`, directly under the
`inputs` or `outputs` object of a node. `null` in a node port is the usual form
of an optional port that the graph does not use. A port that holds the uid as a
number with no fraction (`7.0`) also holds the uid.

An absent tensor costs nothing at run time. It gets no allocation, no fill, no
variant-pack entry and no comparison. The fill draws the seeds of the other
inputs in uid order without it. Give each input its own `seed` in the metadata
`inputs`, so that the contents of an input do not change when another tensor is
absent.

These cases fail to load. The case registers as a failing test, and the message
names the case, the uid (where there is one) and the rule:

- The case has no `metadata` block, or its metadata has `format_version` 1.
- The sweep `version` is not 2. A sweep without `version` has version 1.
- `absent_tensors` is not a non-empty array of integers, or names a uid twice.
- A uid is not a tensor of the template.
- The case names an absent tensor in other fields: a `values.tensors[]` entry,
  a key directly under another case value (for example `values.alignment.<uid>`), a
  `tensor_patches` entry, a metadata `inputs` entry (a seed-only entry too), a
  metadata `aliases` entry as the view or as the owner, a metadata
  `start_offsets` entry, or a metadata `live_rows` entry as the output, the
  `lengths` tensor or the `like` tensor.
- The case declares golden data (a `golden` value that is not `null`). Absent
  tensors and golden data do not go together.
- The template tensor is virtual, or its `virtual` flag is not a literal
  boolean in the template (for example a `${case.*}` placeholder).
- A node port of the template holds a value that is not a uid or `null`, for
  example a `${case.*}` placeholder. The harness cannot tell whether that port
  holds an absent tensor.
- Node ports of more than one node hold the uid, for example the output of one
  node that is the input of another node.
- A node holds the uid in a field that is not a node port, for example a flat
  uid list such as `output_tensor_uids`. The harness looks for the uid under
  each key whose name holds `uid`.
- Another template field refers to the uid, for example the
  `ragged_offset_tensor_uid` of another tensor.
- The graph does not build: an absent tensor fills a port that the op requires.
- No output remains to compare.

A single-graph bundle has no `absent_tensors`: its graph has exactly its
tensors.

Always set `format_version` to `2` in the metadata of a case that uses
`absent_tensors`. A harness that does not know `format_version` 2 rejects the
metadata. A harness that knows `format_version` 2 but not `absent_tensors`
ignores the key and the sweep `version`. It expands the full template, so an
absent tensor with placeholder dims makes the case fail expansion, and that
harness skips the case.

## Metadata sidecar — `{Name}.meta.json`

This file holds provenance and run guards for a single-graph bundle. **Every
single-graph bundle has one.** A sweep case carries the same object inline as
`cases[].metadata`, which every case requires. `src/harness/BundleMetadata.hpp`
parses it.

| Field | Type | Effect |
|---|---|---|
| `format_version` | int | **Required, `1` or `2`.** Metadata that uses a version-2 feature must use `2`; metadata that uses a version-2 feature with `1` is rejected. A harness rejects a version that it does not know. The version-2 features are the `values` [fill kind](#input-fill-specs), [`aliases`](#tensors-that-share-memory--aliases), [`start_offsets`](#tensors-that-start-past-an-aligned-address--start_offsets) and [`live_rows`](#outputs-with-rows-past-a-live-length--live_rows), and [`absent_tensors`](#optional-tensors--absent_tensors) in a sweep case. |
| `enforcement_level` | string | `applicability`, `buildable`, or `full` (the default when absent). How far a run must get for a support claim to count as confirmed; see [Support claims](#support-claim-sidecars--namesupportjson-and-supportjson). Any other value rejects the metadata. Not applicable to an expected-rejection bundle: `expected_rejection` overrides it. |
| `minimum_vram_mb` | int | Skip the case when the device has less VRAM. |
| `gpu_architecture` | string | Golden data is locked to an arch. Skip the case unless the base arch token of the device matches (`gfx942` matches `gfx942:sramecc+:xnack-`, not `gfx940`). If absent, the data is portable. |
| `seed` | int | Seed for generated inputs. |
| `inputs` | object | Per-input fill specs keyed by tensor uid as a string (`"3"`). See [Input fill specs](#input-fill-specs). A key that is not a tensor uid rejects the metadata, and so does an entry that is not a valid fill spec. The O and stats inputs of an SDPA backward node are the exception: after the fill they are recomputed as the forward of its Q, K and V, under its scale, diagonal band and additive mask, so an override for them has no effect. A node whose forward the fill cannot derive (ALiBi, padding, sequence lengths, dropout, ragged tensors, mixed or unsupported data types, a scale tensor that is not a pass-by-value scalar, a Q, K, V or mask computed by the graph) makes the bundle unverifiable, with the reason; `deriveSdpaBackwardForwardOutputs()` is the authority. Shapes the CPU reference rejects, such as stats not shaped `[B, H, Sq, 1]`, keep the override, and the CPU reference then fails the bundle. Likewise, the index input of a MAXPOOL ResampleBwd node is recomputed by `deriveResampleBackwardIndex()` when it is a graph input, so its override has no effect; an index produced by another graph node is not overridden. Non-INT32 indices make the bundle unverifiable. If the CPU forward rejects the ranks, the index keeps its fill and the backward reference then fails the bundle. |
| `aliases` | array of objects | Inputs that read the memory of another input and have no allocation of their own. See [Tensors that share memory](#tensors-that-share-memory--aliases). Needs `format_version` 2. |
| `start_offsets` | array of objects | Tensors whose data starts a number of bytes past an aligned address when the engine under test runs. See [Tensors that start past an aligned address](#tensors-that-start-past-an-aligned-address--start_offsets). Needs `format_version` 2. |
| `live_rows` | array of objects | Dense outputs whose positions past a live length per batch hold no defined value and are not compared. See [Outputs with rows past a live length](#outputs-with-rows-past-a-live-length--live_rows). Needs `format_version` 2. |
| `no_cpu_reference` | bool | Default `false`. `true`: the CPU reference executor is not an oracle for this bundle. A non-boolean value (`"yes"`, `1`, `null`) rejects the metadata. See [Disabling a reference oracle](#disabling-a-reference-oracle--no_cpu_reference-and-no_gpu_reference). |
| `no_gpu_reference` | bool | Default `false`. `true`: the GPU reference executor is not an oracle for this bundle. Same type rule. If both flags are `true`, the bundle must declare golden data (rule R4). |
| `expected_rejection` | bool | Default `false`. `true`: the graph is invalid on purpose and the engine under test must reject it. The harness never executes or compares it. A non-boolean value rejects the metadata. Overrides `enforcement_level`; `no_cpu_reference`, `no_gpu_reference`, `seed` and `inputs` are ignored. See [Expected-rejection bundles](#expected-rejection-bundles). |
| `generator`, `generator_version`, `generated_at`, `reference_source`, `reference_source_hash`, `reference_strategy`, `rocm_version`, `operation`, `generation_command`, `notes` | string | Provenance only; no effect on the run. |

A bundle without its metadata is an error. These fields are required:

- `format_version` (`1` or `2`) — the harness rejects the metadata without it.
- `generator` and `reference_source`, non-empty — `reference-data-scripts/verify_golden_bundles.py`
  requires them. They show a reviewer where the graph and its expected values
  came from.

Generators can write extra keys (`config`, `input_range`, …). The harness ignores
keys that it does not know.

### Input fill specs

Each entry of `inputs` is a JSON object that tells the harness how to fill one
input tensor. The harness fills an input tensor that has no entry with the
default fill for its operation and data type.

```json
"inputs": {
    "3": { "kind": "free", "lo": -2.0, "hi": 2.0, "seed": 17 },
    "5": { "kind": "fixed", "value": 0.125 },
    "7": { "kind": "values", "values": [64, 32], "repeat": true },
    "9": { "seed": 4 }
}
```

| `kind` | Keys | Fill |
|---|---|---|
| `free` | `lo`, `hi` (numbers, default `-1` and `1`) | Uniform random values in the range. |
| `fixed` | `value` (number, default `0`) | One value in every element. |
| `values` | `values` (non-empty array of numbers), `repeat` (boolean, default `true`) | The exact numbers given. Needs `format_version` 2. |
| absent | — | The default fill. Use this form to set only a `seed`. |

All kinds accept an optional `seed`, an unsigned integer. It replaces the seed
that the harness derives for this tensor from the metadata `seed`. It has no
effect on a `values` fill.

A `kind` that is not one of the names above rejects the metadata. A `kind` that is
not a string rejects the metadata too. The error message names the input uid and
the problem. The bundle then registers as a failing test.

Every form also rejects any key outside the ones listed for it above: the absent form
takes only `seed`. `free` and `fixed` are checked together against their combined key set
(`lo`, `hi`, `value`, `seed`), so either accepts all four keys regardless of which `kind` is
given; only a key outside that combined set is rejected.

#### The `values` kind

- Element *i* of the tensor gets `values[i % n]`, where *n* is the list length.
  *i* counts the elements in logical index order, the last dimension fastest. The
  strides of the tensor do not change which element gets which value.
- With `repeat: false`, *n* must be equal to the element count of the tensor.
  With `repeat: true`, *n* must not be more than the element count.
- If every number in the list is a JSON integer, the harness keeps the list as
  exact 64-bit integers. If one number has a fraction or an exponent (`2.5`,
  `1.0`, `1e3`), the harness keeps the whole list as double-precision numbers.
  The JSON reader reads an integer of 2^64 or more as a floating-point number, so
  such an integer also makes the list double-precision.
- Integer tensors (`INT8`, `UINT8`, `INT32`, `INT64`, `BOOLEAN`) take only an
  integer list. Each value must be in the range of the data type. `BOOLEAN` takes
  only `0` and `1`.
- Floating-point tensors (`FLOAT`, `HALF`, `BFLOAT16`, `DOUBLE` and the 8-bit
  floating-point types) take any list. The harness rounds each value to the data
  type. Integers stay exact up to 2^11 in `HALF`, 2^8 in `BFLOAT16`, 2^24 in
  `FLOAT` and 2^53 in `DOUBLE`. Larger integers round.
- The harness cannot fill the sub-byte types (`FP4_E2M1`, `FP6_E2M3`,
  `FP6_E3M2`, `INT4`) with `values`.

These entries reject the metadata: an empty or absent `values` array, a value
that is not a number (a string, a boolean, `null`, an array or an object), an
integer from 2^63 to 2^64 - 1, a `repeat` that is not a boolean, and any key other than
`kind`, `values`, `repeat` and `seed`.

These errors fail the test with a harness failure: a value out of range for the
data type, a list stored as double-precision numbers on an integer tensor, a list
longer than the tensor, a `repeat: false` list with the wrong length, and a
sub-byte tensor. The test does not skip, and the
engine does not run.

Always set `format_version` to `2` in metadata that uses `values`. The harness
rejects `values` in version-1 metadata. A harness older than the `values` kind
reads an unknown kind as `free`. In version-1 metadata it fills the tensor with
random values and reports no error. Version 2 makes such a harness reject the
metadata.

#### Ragged tensors in a bundle without tensor blobs

A bundle without tensor blobs can hold ragged tensors. Give each of these tensors
a `values` fill spec with its exact values:

- the `ragged_offset_tensor_uid` tensor of each ragged tensor;
- the `seq_len_q` and `seq_len_kv` tensors of each SDPA node that has a ragged
  tensor.

The harness fills each ragged input as a dense tensor of its `dims` and
`strides`, with the seed, as it fills any other input. The offset and seq_len
values select the rows that the engine and the references read. Positions
outside these rows hold seed values, and no comparison reads them.

When one of these tensors has no `values` fill spec, the test skips as
unverifiable: `ragged inputs need exact offset values or golden blobs`. The
engine does not run.

After the fill, and before the engine runs, the harness checks the layout of
each ragged tensor. The test fails with a harness failure, and does not skip,
when:

- the offsets and lengths do not give a valid ragged layout, for example a
  negative origin or two batches whose rows overlap;
- the last element that a ragged input reaches is past its dense buffer.

The offsets need not increase from batch to batch when the node has seq_len
tensors.

The harness compares a ragged output on the host only, as it compares a
[`live_rows`](#outputs-with-rows-past-a-live-length--live_rows) output. Under
`--validator auto` (the default) and `--validator cpu`, it copies output that a
GPU reference left on the device to the host. Under `--validator gpu` the output
fails with `Ragged output NOT COMPARABLE ON DEVICE`.

### Tensors that share memory — `aliases`

An alias makes one input tensor, the **view**, read the memory of another input
tensor, the **owner**. The view gets no allocation of its own.

```json
"aliases": [{"uid": 12, "of": 11, "byte_offset": 0}]
```

| Key | Value |
|---|---|
| `uid` | The uid of the view. |
| `of` | The uid of the owner. |
| `byte_offset` | Where the view starts, in bytes from the start of the owner's allocation. |

The view reads the owner's memory through its own dims, strides and data type
from the graph. The harness does not change the strides of the view.

The fill:

- The harness fills the owner once. First it sets every byte of the owner's
  allocation to zero, so the gaps between the owner's elements hold zeros. Then
  it fills the owner's elements on the host.
- The harness never fills a view. The view still takes its place in the seed
  sequence, so an alias does not change the contents of any other tensor.
- An `inputs` entry for a view can set a `seed` only. That entry has no effect.

These entries reject the metadata when it loads. The error names the rule and
the entry: the view and the owner, or, for an entry that cannot be read, its
index and the uids that can be read. The bundle then registers as a failing test.

- `aliases` is in version-1 metadata. The message is `aliases needs
  format_version 2`, whatever the value of `aliases`. Other metadata errors can
  be reported before it.
- `aliases` is not an array, or it is empty.
- An entry is not an object, does not have `uid`, `of` and `byte_offset`, has a
  value that is not a 64-bit signed integer, or has another key.
- `byte_offset` is negative.
- `uid` and `of` are the same tensor.
- The same `uid` is in two entries.
- A view is also the owner of another view.
- The `inputs` entry of a view has a `kind`.
- The view or the owner is not a tensor of the graph.
- The view or the owner is virtual or an output. An alias cannot describe an
  output that the graph writes over an input.
- The view or the owner is passed by value at run time, has a constant value
  (`value_type`), is ragged, or has a sub-byte data type (`FP4_E2M1`,
  `FP6_E2M3`, `FP6_E3M2`, `INT4`).
- The view or the owner has a negative dim, a different number of dims and
  strides, or a data type with no element size (`UNSET`).
- The view or the owner has a negative stride.
- The view and the owner have different data types.
- `byte_offset` is not a multiple of the element size of the view.
- `byte_offset` is not a multiple of the `alignment` of the view (default 16).
  Set a smaller `alignment` on the view to use a smaller offset.
- The `alignment` of the owner is not a multiple of the `alignment` of the view.
  The view's pointer is aligned only as far as the owner's allocation is.
- The owner has a dimension of zero.
- The view reaches past the end of the owner's allocation. The view reaches
  `byte_offset + (1 + Σ (dim − 1) × stride) × element size` bytes. The owner's
  allocation is `(1 + Σ (dim − 1) × stride) × element size` bytes, from the
  owner's dims and strides. The message gives the owner's size in bytes. A view
  with a dimension of zero reaches no bytes, but its `byte_offset` must not be
  more than the owner's size.
- The input fill of the graph derives some inputs from other inputs after the
  fill (an SDPA backward node, see `inputs` above). That fill needs each input as
  its own tensor.
- The bundle has golden data or any tensor blob (`{Name}.tensor<uid>.bin`), even
  when some tensors, such as the view, have no blob. A blob stores each tensor
  apart.

Always set `format_version` to `2` in metadata that uses `aliases`. A harness
that does not know `format_version` 2 rejects the metadata. A harness that
knows `format_version` 2 but not `aliases` ignores the key: each tensor gets its
own memory, and the harness reports no error. A harness that does not know
`aliases` also ignores the key in version-1 metadata.

### Tensors that start past an aligned address — `start_offsets`

A start offset makes the engine under test read or write a tensor at an address
that is a number of bytes past an aligned address. Use it to test an engine with
a misaligned input or output.

```json
"start_offsets": [{"uid": 3, "start_offset": 2}, {"uid": 7, "start_offset": 1}]
```

| Key | Value |
|---|---|
| `uid` | The uid of the tensor. An input or an output. |
| `start_offset` | The number of bytes from an address aligned to 256 bytes to the start of the tensor's data. From 1 to 255. |

The run:

- The harness fills the inputs and loads the golden data as usual.
- Before the engine runs, the harness copies each tensor in `start_offsets` to
  a buffer that starts `start_offset` bytes past an address aligned to 256 bytes.
  The engine gets the address of that copy. An output copy holds the sentinel.
- After the engine runs, the harness copies each output back into its tensor.
  An element that the engine does not write still holds the sentinel.
- The benchmark under `--benchmark` gets new copies that start at the same
  offsets.
- Each tensor in `start_offsets` gets a second device allocation while the
  engine runs. `minimum_vram_mb` does not include it.
- The CPU and GPU references use the tensors themselves, at aligned addresses.
  The comparison reads the tensors as usual.
- An alias view of a tensor in `start_offsets` moves with its owner. The view's
  address is the owner's new address plus the alias's `byte_offset`.

`start_offset` does not have to be a multiple of the element size. A tensor
with a 2-byte data type can start 1 byte past an aligned address.

These entries reject the metadata when it loads. The error names the rule and
the tensor, or, for an entry that cannot be read, its index. The bundle then
registers as a failing test.

- `start_offsets` is in version-1 metadata. The message is `start_offsets needs
  format_version 2`, whatever the value of `start_offsets`. Other metadata
  errors can be reported before it.
- `start_offsets` is not an array, or it is empty.
- An entry is not an object, does not have `uid` and `start_offset`, has a
  value that is not a 64-bit signed integer, or has another key.
- `start_offset` is less than 1 or more than 255.
- The same `uid` is in two entries.
- The tensor is not a tensor of the graph.
- The tensor is virtual, is passed by value at run time, or has a constant
  value (`value_type`).
- The tensor is the view of an alias. A view moves with its owner.
- The tensor has a sub-byte data type (`FP4_E2M1`, `FP6_E2M3`, `FP6_E3M2`,
  `INT4`).
- The tensor declares an `alignment` greater than 0 (default 16), and
  `start_offset` is not a multiple of it. An engine can trust the declared
  alignment. Set a smaller `alignment` on the tensor, or `0`, to use a smaller
  offset.

A bundle with golden data or tensor blobs can use `start_offsets`. A ragged
tensor and an output that the graph writes over an input can use it too.

Always set `format_version` to `2` in metadata that uses `start_offsets`. A
harness that does not know `format_version` 2 rejects the metadata. A harness
that knows `format_version` 2 but not `start_offsets` ignores the key: the
engine gets aligned addresses, and the harness reports no error.

### Outputs with rows past a live length — `live_rows`

A live-row entry tells the comparison which positions of a dense output hold no
defined value. Use it for an output beside ragged or padded sequences, where the
references do not write the rows past each sequence's live length. These rows
hold the sentinel NaN or another value that is not defined.

```json
"live_rows": [{"uid": 9, "axis": 2, "lengths": 14}, {"uid": 10, "like": 1}]
```

| Key | Value |
|---|---|
| `uid` | The uid of the output. |
| `axis` | The axis that the live length bounds. From 1 to the rank of the output minus 1. Axis 0 is the batch. Use with `lengths`. |
| `lengths` | The uid of an input tensor of type INT32 or INT64 with one element for each batch of the output (`dims[0]`). Element `b` is the live length of batch `b`. |
| `like` | The uid of a ragged tensor. The output uses the live lengths and the sequence axis of that tensor. Use without `axis`. |

An entry has `lengths` and `axis`, or `like` alone.

The comparison:

- A position of the output whose index on the axis is at least the live length
  of its batch is not compared. The comparison skips it in the expected tensor
  and in the actual tensor.
- The other positions are compared with the output's validator and tolerance.
- The engine harness applies `live_rows` when it compares with the golden data
  and with each reference. The golden data validation applies it too.

The comparison treats the output as it treats a ragged output:

- An expected value that is NaN at a live position is not compared.
- When no position is compared, the comparison fails. An output whose live
  lengths are all 0 fails.
- The comparison of a live-row output runs on the host only. Under
  `--validator auto` (the default) and `--validator cpu`, the harness copies
  output that a GPU reference left on the device to the host, and compares it
  there. A bundle without golden data can therefore use `live_rows`. Under
  `--validator gpu` the output fails with `Output with live rows NOT COMPARABLE
  ON DEVICE`. Rerun with `--validator auto` or `--validator cpu`.

When the test runs, the harness reads the lengths from the `lengths` tensor, or
from the `like` tensor. The test fails, and does not skip, when:

- the `lengths` or `like` tensor has no data in the run;
- the `like` tensor is not ragged, its sequence axis is not an axis of the
  output, or its extent on that axis is not the output's extent;
- a length is less than 0 or more than the output's extent on the axis.

These entries reject the metadata when it loads. The error names the rule and
the output, or, for an entry that cannot be read, its index. The bundle then
registers as a failing test.

- `live_rows` is in version-1 metadata. The message is `live_rows needs
  format_version 2`, whatever the value of `live_rows`. Other metadata errors
  can be reported before it.
- `live_rows` is not an array, or it is empty.
- An entry is not an object, does not have `uid`, has a value that is not a
  64-bit signed integer, or has another key.
- An entry does not have exactly one of `lengths` and `like`, has `lengths`
  without `axis`, or has `like` with `axis`.
- The same `uid` is in two entries.
- The output is not a tensor of the graph, is not an output that the
  comparison reads, or is ragged. Or its data type is not `FLOAT`, `HALF`,
  `BFLOAT16`, `DOUBLE`, `FP8_E4M3`, `FP8_E5M2`, `FP8_E4M3_FNUZ` or
  `FP8_E5M2_FNUZ`.
  A ragged output already skips the rows past its live lengths.
- `axis` is not in the range 1 to the rank of the output minus 1.
- The `lengths` tensor is not a tensor of the graph, is virtual, is an output,
  is passed by value at run time, or is the view of an alias.
- The `lengths` tensor does not have the data type INT32 or INT64, or its
  element count is not the output's `dims[0]`.
- The `like` tensor is not a tensor of the graph, is virtual, is not ragged,
  or does not have the output's `dims[0]`.

The graph does not record the sequence axis of a ragged tensor. Thus the load
does not check that the `like` tensor and the output have the same extent on
that axis. The run checks it, and the test fails when the extents differ.

A bundle with or without golden data or tensor blobs can use `live_rows`.

Always set `format_version` to `2` in metadata that uses `live_rows`. A harness
that does not know `format_version` 2 rejects the metadata. A harness that knows
`format_version` 2 but not `live_rows` ignores the key and compares every
position of the output.

### Disabling a reference oracle — `no_cpu_reference` and `no_gpu_reference`

Set a flag to `true` when the matching reference executor must not grade this
bundle. The flags are valid in `{Name}.meta.json` and in the `metadata` of a
sweep case. An absent flag is `false`. A value that is not a JSON boolean is a
metadata error: the bundle registers as a **failing** test whose message names
the file or case.

| Run | Effect |
|---|---|
| `--verification-mode=auto` | The oracle chain stays golden data → GPU reference → CPU reference. The harness never creates or probes a disabled reference. If no oracle verifies the bundle, the failure message lists the disabled one as `<CPU\|GPU> reference (disabled by bundle metadata)`. |
| `--verification-mode=cpu` with `no_cpu_reference`, or `gpu` with `no_gpu_reference` | The test **skips** before the engine runs. The message names the flag. |
| Both flags `true`, golden data declared but not pulled (no `.bin` on disk) | The test **skips** with `golden data not pulled; bundle disables both references`, before it fills inputs or runs the engine. `--verification-mode=golden` still **fails** on missing golden data. |

**Rule R4: if both flags are `true`, the bundle must declare golden data.**
Otherwise nothing could ever verify it. "Declared" means:

- **Single graph:** `{Name}.tensors.dvc` exists next to `{Name}.json`, or a golden
  blob `{Name}.tensor<uid>.bin` exists for every non-virtual output.
- **Sweep case:** the case has a `golden` object with a `path`.

A violation throws `BundleMetadataError` at load, so the bundle's test is red.
`reference-data-scripts/verify_golden_bundles.py` reports it as an error. The
flags affect `hipdnn_integration_tests` and the lanes of
`hipdnn_golden_data_tests`; see [Running the Tests](running-tests.md#verification-modes)
and [Disable a reference for a bundle](adding-tests.md#disable-a-reference-for-a-bundle).

### Expected-rejection bundles

Set `expected_rejection` to `true` when the graph is invalid on purpose and the
engine under test must **reject** it. The field is valid in `{Name}.meta.json`
and in the `metadata` of a sweep case. An absent field is `false`. A value that
is not a JSON boolean is a metadata error: the bundle registers as a **failing**
test whose message names the file or case.

The harness never fills inputs, executes the graph, compares outputs, or needs
golden data. The outcome depends on the run:

| Run | Outcome |
|---|---|
| Lane pinned with `--test-engine X` (every CI lane pins one) | **Pass** when the ranked-engine query resolved (`OK` or `GRAPH_NOT_SUPPORTED`) and X is not in the ranked list, or when X is ranked but plan build (`create_execution_plans`, `check_support`, `build_plans`) declines or fails. **Fail** with `Engine X accepted expected-rejection bundle <path>: it is ranked and its plans build` when the plans build. **Fail** when the ranked query did not resolve (acceptance is unknown, so the bundle cannot pass). **Fail** when graph deserialization (`from_binary`) failed: `expected-rejection bundle never reached an engine`. |
| No `--test-engine` | **Skip** as unverifiable: `expected_rejection requires --test-engine`. |

Every lane pins its own engine, so the per-lane runs together cover every
engine. A lane whose engine has no support for the op rejects **vacuously**: the
graph is absent from its ranked list. The stage that a pass reports is described
in [Running the Tests](running-tests.md#expected-rejection-bundles).

Rules:

- **Ignored fields.** `enforcement_level`, `no_cpu_reference`, `no_gpu_reference`,
  `seed` and `inputs` have no effect on such a bundle.
- **`--skip-graph-validation` does not cap it.** The plans are compiled, never
  executed, so the flag changes nothing.
- **No golden data.** Golden data next to an expected-rejection bundle is a load
  error (`BundleMetadataError`), so its test is red.
  `reference-data-scripts/verify_golden_bundles.py` reports it as an error.
  "Next to" means:
  - **Single graph:** `{Name}.tensors.dvc` exists next to `{Name}.json`, or a golden
    blob `{Name}.tensor<uid>.bin` exists for every non-virtual output.
  - **Sweep case:** the case has a `golden` object.
- **The invalidity must be engine-visible, not schema-level.** The engine can
  reject only a bundle that loads. A bundle whose JSON or graph cannot load is a
  **red** `Failed to load expected_rejection bundle …` test (the message starts
  with that text, then gives the reason), not a silent skip. A graph that loads
  but fails `from_binary` also fails. Pick a property that the engine checks
  (see [Add an expected-rejection bundle](adding-tests.md#add-an-expected-rejection-bundle)).
- **No support claim.** An expected-rejection bundle carries none. A non-empty
  claim that names it is an authoring error: any engine claim in
  `{Name}.support.json` of a single graph, or a claim group in a sweep's
  `support.json` whose `cases` lists the case id. The C++ test fails, and
  `scripts/verify_support_claims.py` reports an error. A sidecar that is merely
  present is not an error for a sweep, because its cases share `support.json`.
  An unclaimed expected-rejection case inside a claimed sweep is fine. The bundle
  or case is outside the claim model; see
  [Support Claim Enforcement](support-claim-enforcement.md#expected-rejection-bundles-are-outside-the-claim-model).

Checked-in example: `integration-test-bundles/quick/Matmul/ncl/fp32/ExpectedRejectionMatmulBatchMismatch/ExpectedRejectionMatmulBatchMismatch.json`
(with its `.meta.json`; no sidecar, no golden data). It is a rank-3 fp32 Matmul
copied from a hipBLASLt-claimed `quick/Matmul/Default` case, with the batch
dimension of B changed from 1 to 2 (A and C have 3), which is invalid per the
Matmul contract. hipBLASLt refuses it (`Unsupported input matrix batch
dimensions`), so it drops the graph from its ranked list. The claimed valid twin
(sweep case `3_16_128_fp32_ncl_6d8970`) passing in the same lane is the positive control. The other
lanes (MIOpen, hip-kernel) have no Matmul support, so they reject vacuously.

## Golden data — `.tensors.dvc` and `.bin`

Golden tensors are optional. A bundle without them is **graph-only** and is
verified against the GPU or CPU reference executor instead. See
[Verification modes](running-tests.md#verification-modes). A bundle that disables
both references must declare golden data
([rule R4](#disabling-a-reference-oracle--no_cpu_reference-and-no_gpu_reference)).

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
  sweeps (`BlockScaleDequantizeMatmul/Default`, `Reduction/Default`,
  `Pointwise/Binary` and `Pointwise/Activation`, in every tier that has them) have none.
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
| hip-kernel-provider | `config/HIP_MLOPS_ENGINE.toml`, `config/ASM_SDPA_ENGINE.toml`, `config/hipkernel_Gfx950Attention.toml`, `config/hipkernel_Gfx942AttentionUnified3d.toml`, `config/hipkernel_Gfx942AttentionDense.toml`, `config/hipkernel_flydsl_sdpa_bwd.toml` |

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
  of a fully masked SDPA row. In a bundle test, the `allclose` comparison of a
  floating-point output also matches same-signed infinities on the host. On the
  device, a float, half, bfloat16 or double output follows the same rule: a
  tensor that fails the device check is graded again on the host -- including a
  double output that fails the device check only on double-precision drift,
  which the host's float arithmetic then re-grades. In a C++
  graph test, this validator is **host-only**. A tensor that it selects fails
  with `Validator override NOT APPLICABLE ON DEVICE` whenever its comparison
  runs on the device. Under `auto`, this is every run where the GPU reference produced
  the expected values. Run such a config with `--validator cpu`.
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
`shared/ctest/README.md` is the reference for the format. There are four scopes.
Do not confuse them:

| File | Governs |
|---|---|
| `dnn-providers/integration-tests/test_categories.yaml` | This project's own binaries `hipdnn_integration_tests_unit_tests` and `hipdnn_golden_data_tests` |
| `dnn-providers/integration-tests/test_categories_gpu_ref.yaml` | `hipdnn_gpu_ref_tests`. Quick is an fp32 smoke run; reduced-precision and mixed-type cases are in standard |
| `dnn-providers/integration-tests/test_categories_external.yaml` | Pre-registered non-GTest CTest tests (the Python verifiers, test-name validation) |
| `dnn-providers/<provider>/test_categories_integration.yaml` (hip-kernel-provider: `HIP_MLOPS_ENGINE_test_categories_integration.yaml`, `ASM_SDPA_ENGINE_test_categories_integration.yaml`, `hipkernel_Gfx950Attention_test_categories_integration.yaml`, `hipkernel_Gfx942AttentionUnified3d_test_categories_integration.yaml`, `hipkernel_Gfx942AttentionDense_test_categories_integration.yaml`, `hipkernel_flydsl_sdpa_test_categories_integration.yaml`, `hipkernel_flydsl_rmsnorm_test_categories_integration.yaml`, `hipkernel_flydsl_rmsnorm_bwd_test_categories_integration.yaml`, `hipkernel_flydsl_sdpa_bwd_split_test_categories_integration.yaml`, `hipkernel_flydsl_sdpa_bwd_merged_test_categories_integration.yaml`) | `hipdnn_integration_tests` **run against that provider's engine** — the lane's bundle suites |

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

## Where each format is enforced

| Format | Parsed / checked by | When it fails |
|---|---|---|
| Graph `.json` | `IntegrationTestBundle.hpp` | Case dropped from registration with an `ERROR` log line; the run stays green with one test fewer |
| `graph.template.json` + `sweep.json` | `BundleDiscovery.hpp`, `IntegrationTestBundle.hpp` | A malformed `sweep.json` or a bad/duplicate id aborts registration; a bad case (missing placeholder value, missing metadata, bad `golden.path`) is dropped like a bad graph. A sweep `version` other than 1 or 2 registers a **failing** test for each case of the sweep. A case that breaks an [`absent_tensors`](#optional-tensors--absent_tensors) rule registers a **failing** test; an expansion error in such a case is dropped like any other |
| `.meta.json` / `cases[].metadata` | `BundleMetadata.hpp`; `reference-data-scripts/verify_golden_bundles.py` | Required for new bundles and every sweep case, with `generator` and `reference_source`. **Malformed** metadata (invalid JSON, a wrong field type, an unsupported `format_version`, a non-integer `inputs` key, an unknown or malformed [fill spec](#input-fill-specs), an [`aliases`](#tensors-that-share-memory--aliases), [`start_offsets`](#tensors-that-start-past-an-aligned-address--start_offsets) or [`live_rows`](#outputs-with-rows-past-a-live-length--live_rows) entry that the metadata or the graph rejects, a version-2 feature in version-1 metadata) throws `BundleMetadataError`: that bundle registers as a **failing** test whose message names the file or case and what is wrong. This holds for every bundle type, graph-only bundles included. A non-boolean `no_cpu_reference` or `no_gpu_reference` is a wrong field type. Setting both without declared golden data throws it too ([rule R4](#disabling-a-reference-oracle--no_cpu_reference-and-no_gpu_reference)); the verifier reports R4 as an error. A non-boolean `expected_rejection` is a wrong field type too, and golden data next to an expected-rejection bundle throws it ([details](#expected-rejection-bundles)). **Absent** metadata behaves as follows. At run time a sweep case without a `metadata` block is dropped; a single-graph bundle without `.meta.json` loads with empty metadata, unless it has golden `.bin` files, in which case it fails. Some SDPA single-graph bundles have no `.meta.json`, and many older sweeps lack `generator` or `reference_source`, so the verifier fails on the whole tree; run it on the directories you touched ([details](adding-tests.md#check-the-data-itself)) |
| `expected_rejection` bundle or case | `BundleMetadata.hpp` at load; the harness at run time; `reference-data-scripts/verify_golden_bundles.py`; `scripts/verify_support_claims.py` | `enforcement_level` and the oracle fields do not apply (it overrides them). Golden data next to it is a load error (`BundleMetadataError`) and a verifier error. A support claim that names it fails the C++ test and is a verifier error. A bundle that does not load is a red `FailedLoad` test, not a dropped case. [Details](#expected-rejection-bundles) |
| `.support.json` / `support.json` | `SupportClaims.cpp` at run time; `scripts/verify_support_claims.py` by hand | Required for new bundles (not for an expected-rejection bundle), but not enforced: a missing sidecar means "no claims" at run time and is not reported by the verifier. A sidecar that does not parse, and a broken claim, fail that bundle's test |
| Golden `.bin` / `.dvc` | `IntegrationTestBundle.hpp`; `reference-data-scripts/verify_golden_bundles.py` | Falls back to a reference in `auto` mode, except one that the bundle metadata disables; fails in `golden` mode. A blob that is unreadable or has the wrong size fails the test that needs it, when that test runs |
| Engine `.toml` | `TestSettings.hpp` | Binary exits 1 at startup |
| `test_categories*.yaml` | `shared/ctest/parse_test_categories.py` | CMake **warning** at configure; that target's tier suites are not generated, so `ctest -L <tier>` runs fewer tests, with no other notice. Check the configure log or `ctest -N -L <tier>` |

The load problems that register a *failing* test are these: a tensor marked
`is_runtime_pass_by_value` that still carries a baked `value`, malformed
metadata, a sweep `version` other than 1 or 2, and a case that breaks an
`absent_tensors` rule (above).
