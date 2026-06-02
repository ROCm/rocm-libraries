# Support Claims TOML Schema

Per-engine integration-test configs come in two paired files. The main
file (`<EngineName>.toml`) is hand-edited and holds tolerance overrides
and test skips. The sidecar (`<EngineName>.supported.toml`) is
machine-managed by `--write-support-claims` and holds the engine's
positive support claims per asic.

This page is the reference for the sidecar's schema. For an end-to-end
walkthrough see [bring-up](support-claims-bringup.md); for what each
verifier failure means and how to fix it see
[failure modes](support-claims-failures.md).

## File layout

Both files live next to each other in `dnn-providers/<provider>/config/`:

```
dnn-providers/miopen-provider/config/
├── MIOPEN_ENGINE.toml            # hand-edited
└── MIOPEN_ENGINE.supported.toml  # machine-managed
```

The sidecar is discovered automatically — `TestSettings` looks for
`<stem>.supported.toml` next to the main config supplied via
`--test-config`. No separate flag.

## `[meta]` (required in both files when sidecar is in use)

```toml
[meta]
version = 6
engine  = "MIOPEN_ENGINE"
```

- `version` — sidecar schema version. Currently `6`. History:
  - **v1** — initial schema with bare-node-name op_chains.
  - **v2** — op_chain extended with per-node `:variant[flags]` tags so
    Pointwise modes / params, Reduction modes, etc. partition graphs
    MIOpen dispatches differently.
  - **v3** — added `io_dtype_pairs = ["in->out", ...]` alongside an
    `io_dtypes` shorthand for the symmetric case.
  - **v4** — dropped the `io_dtypes` shorthand. `io_dtype_pairs` strings
    were the single form.
  - **v5** — replaced `io_dtype_pairs` strings with `dtype_combos`
    inline-table arrays carrying named fields: `{io, output?, compute,
    intermediate?}`. The schema now mirrors the support-matrix display
    and captures compute/intermediate dtypes that affect MIOpen's solver
    dispatch (previously recorded but not matched against).
  - **v6** — extended `describeNodeVariant()` with shape-flag tags on
    Conv (`1x1`, `grouped`, `multi_batch`, `non_square`, `padding`,
    `stride`, `dilation`) and Batchnorm-family (`multi_batch`). op_chain
    strings now read e.g. `"ConvFprop[1x1,grouped]"`. Engines that
    partition support along these shape axes (hipblaslt only handling
    1x1, hip-kernel skipping grouped/dilated) can record distinct
    matcher rectangles instead of over-claiming via the bare node type.
    Tag-only variants are appended directly with no leading `:` —
    `ConvFprop[1x1]` not `ConvFprop:[1x1]` — to keep the mode-bearing
    `Pointwise:RELU_FWD[upper_clip]` form visually distinct.

  Older-version sidecars are refused at load — regenerate via
  `--write-support-claims`. The main TOML's `[meta].version` is a
  separate, unrelated version stream.
- `engine` — required in the sidecar. Optional in the main file but
  must match the sidecar's value if present. Cross-checked at load
  against `--test-engine` so the same TOML can't be misapplied to a
  different plugin.

## `[[supported]]` blocks

Each block scopes a set of matchers to one `(arch, platform)`. One
block per asic — the multi-arch shorthand `archs = [...]` was rejected
in §11.5 because it couples updates that should be independent.

```toml
[[supported]]
arch     = "gfx942"           # required, exact match against archTokenOf(gcnArchName)
platform = "linux"            # optional, exact match against "windows" or "linux"
```

- `arch` is matched against the prefix of the raw `gcnArchName`
  before the first `:` — so `"gfx942"` matches `gfx942:sramecc+:xnack-`
  but `"gfx10"` does *not* match `gfx1030` (RFC 0012 §5.3).
- `arch` rejects `*`. Wildcards belong nowhere in the schema.
- `platform` defaults to "any" when omitted. When set, only `"windows"`
  or `"linux"` are accepted.

## `[[supported.matchers]]`

Each matcher claims that the cross-product of `op_chains × dtype_combos
× layouts` is fully supported by the engine on the owning block's
`(arch, platform)`. Dtype combos are TOML inline tables with named
fields that mirror the support-matrix markdown display.

```toml
[[supported.matchers]]
op_chains = [
    "ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD",
    "ConvFprop + Pointwise:ADD + Pointwise:SIGMOID",
]
dtype_combos = [
    {io="fp16", compute="fp32"},
    {io="bf16", compute="fp32", intermediate="bf16"},
    {io="fp32", compute="fp32", intermediate="fp32"},
    {io="fp16", output="fp32", compute="fp32"},   # asymmetric
]
layouts = ["NCHW", "NHWC"]
```

### Fields on `dtype_combos`

| Field           | Required | Meaning |
|-----------------|----------|---------|
| `io`            | yes      | Input dtype. Also the output dtype when `output` is omitted (symmetric case — the common one). |
| `output`        | no       | Output dtype. Set only when it differs from `io` (mixed-precision graphs). |
| `compute`       | yes      | Compute / accumulation dtype. |
| `intermediate`  | no       | Intermediate dtype. Set only when the graph specifies it (`graph.set_intermediate_data_type(...)`). |

### Loader rules

- `op_chains`, `dtype_combos`, and `layouts` are required and non-empty.
- All string values are matched exact-string. No `*`, no `?`, no fnmatch.
- `dtype_combos` entries must be inline tables (`{io=..., compute=...}`).
- Unknown keys inside a combo are rejected (catches typos and silent
  schema drift).
- Duplicate combos (compared by all four fields) are rejected.
- A test **matches** a matcher iff its observed
  `(op_chain, io, output, compute, intermediate, layout)` 6-tuple has
  a `dtype_combo` whose fields all match (output normalized to io for
  symmetric records on both sides) AND its op_chain is in `op_chains`
  AND its layout is in `layouts`.

### Why named fields

The named-field shape is deliberate and replaces earlier flat / arrow
string forms:

- **Mirrors the support-matrix display.** The markdown already shows
  `[io=bf16, compute=fp32, intermediate=fp32]`. The schema now uses the
  same shape, so the support matrix can in fact be rendered from the
  sidecar.
- **Captures what the engine actually dispatches on.** Previous schemas
  recorded `compute` / `intermediate` on the record side but ignored
  them in matching. Combos make them first-class matcher key fields —
  if MIOpen dispatches differently per compute dtype, the condenser
  will surface that as an S∩U conflict instead of silently mixing.
- **Extensible by adding keys, not by inventing syntax.** A future
  dispatch dimension (e.g. weight dtype, accumulator dtype) is just a
  new optional key. No new parsing convention, no schema migration of
  existing entries.

### Value spaces

| Field            | Source                                                  | Example values |
|------------------|---------------------------------------------------------|----------------|
| `op_chains`      | `describeGraphStructured(graph).opChain` — visit order + `to_string(NodeType)` joined by ` + `, with a `:VARIANT` suffix per node when the node's attributes affect MIOpen solver dispatch | `"ConvFprop"`, `"ConvFprop + Pointwise:RELU_FWD"`, `"BatchnormInference + Pointwise:RELU_FWD[upper_clip]"` |
| `dtype_combos`  | Inline tables with `io` (input dtype, from `graph.graph_attributes.get_io_data_type()`), optional `output` (when distinct), `compute` (from `graph.graph_attributes.get_compute_data_type()`), optional `intermediate` (from `graph.graph_attributes.get_intermediate_data_type()`). | `{io="fp16", compute="fp32"}`, `{io="bf16", compute="fp32", intermediate="bf16"}`, `{io="fp16", output="fp32", compute="fp32"}` (mixed-precision) |
| `layouts`        | Fixtures call `setTestCaseLayout("NCHW"|"NHWC"|...)`    | `"NCHW"`, `"NHWC"`, `"NCDHW"`, `"NDHWC"` |

### Per-node variant tags (v2)

`describeNodeVariant()` returns a stable string per node type when the
bare node type isn't enough to capture MIOpen's dispatch behavior.
Today:

| Node | Variant values | Why |
|------|----------------|-----|
| `Pointwise` | `MODE` plus `[flags]` listing which optional params are set — `lower_clip`, `upper_clip`, `lower_slope`, `swish_beta`, `elu_alpha`, `softplus_beta` (alphabetical, deterministic). Example: `RELU_FWD[lower_clip,upper_clip]`. | Different MIOpen solvers per (mode, params) combination — plain ReLU, ReLU6, clamp, and leaky-ReLU all use mode `RELU_FWD` but dispatch differently. |
| `Reduction` | `ADD`, `MAX`, ... when mode is set | Different solvers per reduction op. |
| `ConvFprop` / `ConvDgrad` / `ConvWgrad` | `[flags]` only, alphabetical: `1x1` (all spatial filter dims == 1), `grouped` (input channels / filter channels > 1), `multi_batch` (N > 1), `non_square` (spatial input dims differ), `padding` (any non-zero pre/post padding), `stride` (any stride ≠ 1), `dilation` (any dilation ≠ 1). Example: `ConvFprop[1x1,multi_batch]`. | Conv engines partition support along shape axes — MIOpen has a dedicated 1x1 solver path, hipblaslt only handles 1x1 (GEMM-backed), hip-kernel may skip grouped or dilated convs. Without these tags an engine that only supports 1x1 conv would over-claim by saying it supports `ConvFprop`. |
| `Batchnorm` / `BatchnormBackward` / `BatchnormInference` / `BatchnormInferenceVarianceExt` | `[multi_batch]` when input N > 1. | N=1 vs N>1 hits different MIOpen solver paths and is the partition axis for engines targeting single-batch inference. |

**Rule of thumb**: variants are extended per-node-type only when a real
S∩U conflict has demonstrated the bare node type is too coarse. Adding
speculative variants creates matcher-set noise (extra entries that the
engine treats identically). The condenser fails loudly on `S∩U≠∅` to
surface the next missing variant — when that fires, add a variant tag
for the offending node type, bump the schema version, regenerate.

The op_chain string format is a stability contract once this RFC ships
(RFC 0012 §12 risks). Any change to graph visit order, `to_string`
mapping, or pointwise mode formatting bumps `[meta].version` and
triggers coordinated regeneration of every sidecar.

## Worked example

```toml
# MIOPEN_ENGINE.supported.toml

[meta]
version = 6
engine  = "MIOPEN_ENGINE"

[[supported]]
arch = "gfx942"

# Plain Conv across all observed dtypes and layouts
[[supported.matchers]]
op_chains = ["ConvFprop", "ConvDgrad", "ConvWgrad"]
dtype_combos = [
    {io="fp16", compute="fp32", intermediate="fp16"},
    {io="fp32", compute="fp32", intermediate="fp32"},
    {io="bf16", compute="fp32", intermediate="bf16"},
]
layouts = ["NCHW", "NHWC"]

# Conv + Bias + Activation, including a mixed-precision fp16 → fp32
# variant some solvers expose for high-precision accumulation
[[supported.matchers]]
op_chains = [
    "ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD",
    "ConvFprop + Pointwise:ADD + Pointwise:SIGMOID",
]
dtype_combos = [
    {io="fp16", compute="fp32"},
    {io="fp32", compute="fp32"},
    {io="bf16", compute="fp32"},
    {io="fp16", output="fp32", compute="fp32"},   # asymmetric
]
layouts = ["NCHW", "NHWC"]

# gfx10 has no CK fusion kernels — list only plain Conv here
[[supported]]
arch = "gfx10"

[[supported.matchers]]
op_chains = ["ConvFprop", "ConvDgrad", "ConvWgrad"]
dtype_combos = [
    {io="fp16", compute="fp32", intermediate="fp16"},
    {io="fp32", compute="fp32", intermediate="fp32"},
    {io="bf16", compute="fp32", intermediate="bf16"},
]
layouts = ["NCHW", "NHWC"]
```

## What's deliberately *not* in the schema

- **`[[unsupported]]`** — absence from `[[supported]]` is the implicit
  unsupported state (RFC 0012 §11.2). Adding a positive-and-negative
  schema doubles the maintenance load and forces TOML updates whenever
  the engine grows capability.
- **`archs = [...]` multi-arch shorthand** (RFC 0012 §11.5).
- **Globs / wildcards** (RFC 0012 §11.4).

## Loader semantics

- Unknown top-level keys → logged-and-ignored (forward compatibility).
- Unknown keys inside `[meta]`, `[[supported]]`, or
  `[[supported.matchers]]` → also logged-and-ignored.
- Missing required fields → loud `std::runtime_error` with file and
  block index.
- `[meta].engine` mismatch between main, sidecar, or the
  `--test-engine` flag → loud `std::runtime_error`.

A sidecar with `[meta]` but zero `[[supported]]` blocks is legal — that
is the new-asic bring-up state. The verifier treats absence-of-block
for the current arch as "not enforced" rather than failing.
