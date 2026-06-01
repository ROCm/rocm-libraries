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
version = 3
engine  = "MIOPEN_ENGINE"
```

- `version` — sidecar schema version. Currently `3`. History:
  - **v1** — initial schema with bare-node-name op_chains.
  - **v2** — op_chain extended with per-node `:variant[flags]` tags so
    Pointwise modes / params, Reduction modes, etc. partition graphs
    MIOpen dispatches differently.
  - **v3** — matcher schema gains `io_dtype_pairs = ["in->out", ...]`
    alongside the existing `io_dtypes = ["..."]` shorthand. Captures
    mixed-precision graphs (e.g. fp16 input → fp32 output) that
    previously collapsed into the symmetric io_dtype field.

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

Each matcher claims that the cross-product of `op_chains × (dtype
dimension) × layouts` is fully supported by the engine on the owning
block's `(arch, platform)`. The dtype dimension can be expressed
two ways and the matcher can use either or both:

```toml
[[supported.matchers]]
op_chains = [
    "ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD",
    "ConvFprop + Pointwise:ADD + Pointwise:SIGMOID",
]
# Symmetric-shorthand: "fp16" covers the pair fp16->fp16.
io_dtypes = ["fp16", "fp32", "bf16"]
# Asymmetric pairs in "in->out" form. Optional.
io_dtype_pairs = ["fp16->fp32", "bf16->fp32"]
layouts   = ["NCHW", "NHWC"]
```

- `op_chains` and `layouts` are required and non-empty.
- At least one of `io_dtypes` and `io_dtype_pairs` must be present;
  both may appear on the same matcher and their effects union.
- All values are matched exact-string. No `*`, no `?`, no fnmatch.
- `io_dtype_pairs` entries must contain exactly one `->` separator
  with non-empty input and output sides.
- Duplicates in any array are rejected by the loader.
- A test **matches** a matcher iff its observed `(op_chain,
  input_dtype, output_dtype, layout)` tuple lies in the matcher's
  cross-product. Symmetric observations (`input == output`) match
  against `io_dtypes` first, then `io_dtype_pairs`; asymmetric
  observations match only against `io_dtype_pairs`.

### Value spaces

| Field      | Source                                                  | Example values |
|------------|---------------------------------------------------------|----------------|
| `op_chains`| `describeGraphStructured(graph).opChain` — visit order + `to_string(NodeType)` joined by ` + `, with a `:VARIANT` suffix per node when the node's attributes affect MIOpen solver dispatch | `"ConvFprop"`, `"ConvFprop + Pointwise:RELU_FWD"`, `"BatchnormInference + Pointwise:RELU_FWD[upper_clip]"` |
| `io_dtypes`| `to_string(graph.graph_attributes.get_io_data_type())`  | `"fp16"`, `"fp32"`, `"bf16"`, `"fp64"` |
| `layouts`  | Fixtures call `setTestCaseLayout("NCHW"|"NHWC"|...)`    | `"NCHW"`, `"NHWC"`, `"NCDHW"`, `"NDHWC"` |

### Per-node variant tags (v2)

`describeNodeVariant()` returns a stable string per node type when the
bare node type isn't enough to capture MIOpen's dispatch behavior.
Today:

| Node | Variant values | Why |
|------|----------------|-----|
| `Pointwise` | `MODE` plus `[flags]` listing which optional params are set — `lower_clip`, `upper_clip`, `lower_slope`, `swish_beta`, `elu_alpha`, `softplus_beta` (alphabetical, deterministic). Example: `RELU_FWD[lower_clip,upper_clip]`. | Different MIOpen solvers per (mode, params) combination — plain ReLU, ReLU6, clamp, and leaky-ReLU all use mode `RELU_FWD` but dispatch differently. |
| `Reduction` | `ADD`, `MAX`, ... when mode is set | Different solvers per reduction op. |

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
version = 1
engine  = "MIOPEN_ENGINE"

[[supported]]
arch = "gfx942"

# Plain Conv across all observed dtypes and layouts
[[supported.matchers]]
op_chains = ["ConvFprop", "ConvDgrad", "ConvWgrad"]
io_dtypes = ["fp16", "fp32", "bf16"]
layouts   = ["NCHW", "NHWC"]

# Conv + Bias + Activation
[[supported.matchers]]
op_chains = [
    "ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD",
    "ConvFprop + Pointwise:ADD + Pointwise:SIGMOID",
]
io_dtypes = ["fp16", "fp32", "bf16"]
layouts   = ["NCHW", "NHWC"]

# gfx10 has no CK fusion kernels — list only plain Conv here
[[supported]]
arch = "gfx10"

[[supported.matchers]]
op_chains = ["ConvFprop", "ConvDgrad", "ConvWgrad"]
io_dtypes = ["fp16", "fp32", "bf16"]
layouts   = ["NCHW", "NHWC"]
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
