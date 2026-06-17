# Portable CK-DSL IR Schema

This document proposes a versioned IR artifact that lets kernels authored with
the Python `ck_dsl` frontend ship as structured CK-DSL IR and be lowered online
by the pure-C `ck_dsl_c` backend.

The goal is not to parse Python source at runtime. The goal is to serialize the
already-built Python `KernelDef` graph into a stable format, load that artifact
from the hipDNN provider, import it into `ckc_kernel_def_t`, and then use the C
lowering/comgr path.

## Executive Goal

Support this hybrid deployment path:

```text
Offline:
  Python authoring -> KernelDef -> portable CK-DSL IR artifact + manifest

Online:
  hipDNN provider -> C IR import -> ckc_lower_kernel_to_llvm -> comgr -> HSACO
```

This sits between the two existing extremes:

```text
Fastest:
  offline Python -> .hsaco -> online load

Current simple JIT:
  offline Python -> .ll -> online comgr

Retargetable C-backend JIT:
  offline Python -> portable CK-DSL IR -> online C import + C lower + comgr
```

Use this path when an artifact should remain higher level than LLVM IR: for
example, when the same authored kernel must lower against different gfx targets,
ROCm LLVM flavors, or late backend policy choices without embedding CPython in
the provider.

## Non-Goals

- Do not parse or transpile Python source.
- Do not make hipDNN import `ck_dsl` Python modules.
- Do not require every kernel family to switch at once.
- Do not replace `.hsaco` artifacts for fixed production kernels where load-only
  is sufficient.
- Do not encode host-side Python selectors, launchers, dataclasses, or dynamic
  helper logic in the IR artifact. Only serialize the built `KernelDef`.

## Artifact Layout

Use a directory bundle first. A packed archive can be added later without
changing the schema.

```text
<kernel>.ckir/
  manifest.json          # existing launch/runtime metadata
  kernel.ir.json         # canonical debug/interchange format
  kernel.ir.cbor         # optional compact binary form, same data model
  kernel.ll              # optional cache/debug product
  kernel.hsaco           # optional prebuilt fast path
```

The `manifest.json` remains the launch contract: kernel name, ABI signature,
grid/block metadata, default shape, op kind, and tuning metadata. The IR file is
only the compiler input for the C backend.

The artifact store should prefer products in this order:

```text
1. .hsaco       -> load directly
2. .ll          -> comgr JIT directly
3. .ir.cbor     -> C import + C lower + comgr
4. .ir.json     -> C import + C lower + comgr
```

This order preserves the current fastest paths and treats portable IR as a
fallback or retargeting input.

## Top-Level Schema

Schema string:

```text
"ck.dsl.ir/v1"
```

Top-level shape:

```json
{
  "schema": "ck.dsl.ir/v1",
  "producer": {
    "name": "ck_dsl_python",
    "version": "0.1",
    "git_sha": "optional"
  },
  "requires": {
    "min_ckc_ir": 1,
    "opcodes": ["gpu.thread_id_x", "arith.fadd", "func.ret"],
    "features": ["fp16"]
  },
  "target": {
    "arch_hint": "gfx950",
    "llvm_flavor_hint": "llvm22"
  },
  "kernel": {
    "name": "example_kernel",
    "attrs": {},
    "params": [],
    "body": {
      "args": [],
      "ops": []
    }
  }
}
```

`target` is advisory. The online lowerer decides the actual arch and LLVM flavor
from the provider/device configuration. If the artifact requires a feature that
the target cannot lower, the importer or lowerer must reject it with a clear
diagnostic.

## Kernel Model

The schema mirrors the frozen C IR contract in `ckc/ir.h`:

| Python IR | Portable IR | C import target |
| --- | --- | --- |
| `KernelDef` | `kernel` object | `ckc_kernel_def_t` |
| `Region` | `{args, ops}` | `ckc_region_t` |
| `Op` | op object | `ckc_op_t` |
| `Value` | SSA ID | `ckc_value_t*` |
| `Type` | canonical type object/string | `ckc_type_t` |
| `attrs` | typed attr map | `ckc_attr_map_t` |

Every SSA value must have a stable ID unique within the kernel:

```json
{ "id": "v17", "type": "f32" }
```

The importer owns an ID table:

```text
SSA ID string -> ckc_value_t*
```

Operands refer to previously defined values by ID. Region arguments are defined
before the region body ops. Forward references are rejected.

## Types

For scalar types, use the canonical scalar spelling directly:

```json
"f32"
```

Composite types use structured objects:

```json
{ "kind": "vector", "elem": "f16", "count": 4 }
{ "kind": "ptr", "pointee": "f16", "space": "global" }
{ "kind": "smem", "elem": "f16", "shape": [64, 32] }
```

Allowed scalar spellings in v1:

```text
i1, i8, i16, i32, i64, f16, bf16, f32, fp8e4m3, bf8e5m2
```

The importer should canonicalize these through the same constructors used by
`ckc_ir_builder_t` so type identity and printed names match existing lowering.

## Attributes

Attributes must be typed. Do not infer type from JSON syntax.

```json
{
  "align": { "type": "i64", "value": 16 },
  "readonly": { "type": "bool", "value": true },
  "op_id": { "type": "string", "value": "mfma_f32_32x32x16_f16" },
  "scale": { "type": "f64", "value": 1.4426950408889634 }
}
```

Supported v1 attr kinds:

```text
i64, f64, string, bool, list
```

`list` is a list of nested attr maps. This matches the C IR's small variant map
and covers loop iter-arg metadata.

```json
{
  "iter_args": {
    "type": "list",
    "value": [
      { "name": { "type": "string", "value": "acc" } }
    ]
  }
}
```

## Ops

Each op carries the canonical opcode string, operands, results, attrs, and zero
or more nested regions:

```json
{
  "opcode": "mem.global_load",
  "operands": ["v0", "v4"],
  "results": [
    { "id": "v5", "type": "f16" }
  ],
  "attrs": {
    "align": { "type": "i64", "value": 2 }
  },
  "regions": []
}
```

Opcode strings should match the names already used by Python printing and C
`ckc_opcode_name()` wherever possible. The importer maps strings to
`ckc_opcode_t`. Unknown opcodes are hard errors.

Minimal scalar example:

```json
{
  "schema": "ck.dsl.ir/v1",
  "producer": { "name": "ck_dsl_python", "version": "0.1" },
  "requires": {
    "min_ckc_ir": 1,
    "opcodes": ["arith.constant", "arith.add", "func.ret"],
    "features": []
  },
  "kernel": {
    "name": "parity_kernel",
    "attrs": {},
    "params": [],
    "body": {
      "args": [],
      "ops": [
        {
          "opcode": "arith.constant",
          "operands": [],
          "results": [{ "id": "v0", "type": "i32" }],
          "attrs": { "value": { "type": "i64", "value": 1 } }
        },
        {
          "opcode": "arith.add",
          "operands": ["v0", "v0"],
          "results": [{ "id": "v1", "type": "i32" }],
          "attrs": {}
        },
        {
          "opcode": "func.ret",
          "operands": [],
          "results": [],
          "attrs": {}
        }
      ]
    }
  }
}
```

## Params And ABI

Kernel parameters appear in `kernel.params`, not as ordinary body ops. This
makes the launch ABI easy to validate against `manifest.json`.

```json
"params": [
  {
    "id": "v0",
    "name": "A",
    "type": { "kind": "ptr", "pointee": "f16", "space": "global" },
    "attrs": {
      "readonly": { "type": "bool", "value": true },
      "align": { "type": "i64", "value": 16 }
    }
  },
  {
    "id": "v1",
    "name": "M",
    "type": "i32",
    "attrs": {}
  }
]
```

Importer validation must check:

- param IDs are unique;
- param names match `manifest.args_signature` order and type width;
- pointer address spaces are known;
- ABI-affecting attrs are legal for the target backend.

## Regions

Control-flow ops carry nested regions. Region args define local SSA values that
are visible only in that region.

```json
{
  "opcode": "scf.for",
  "operands": ["v_lo", "v_hi", "v_step", "v_acc0"],
  "results": [{ "id": "v_acc_out", "type": "f32" }],
  "attrs": {
    "iv_name": { "type": "string", "value": "k0" },
    "unroll": { "type": "bool", "value": false },
    "elide_trailing_barrier": { "type": "bool", "value": true }
  },
  "regions": [
    {
      "args": [
        { "id": "v_k0", "name": "k0", "type": "i32" },
        { "id": "v_acc_iter", "name": "acc", "type": "f32" }
      ],
      "ops": [
        {
          "opcode": "arith.fadd",
          "operands": ["v_acc_iter", "v_one"],
          "results": [{ "id": "v_acc_next", "type": "f32" }],
          "attrs": {}
        },
        {
          "opcode": "scf.yield",
          "operands": ["v_acc_next"],
          "results": [],
          "attrs": {}
        }
      ]
    }
  ]
}
```

The importer should keep SSA scopes explicit:

- parent values are visible inside child regions;
- child region values are not visible after the region;
- op results are visible after the op in the enclosing region;
- duplicate IDs in overlapping scopes are rejected.

## Validation Policy

The importer should be strict by default:

```text
schema major mismatch        -> reject
unknown opcode               -> reject
unknown type kind            -> reject
unknown attr kind            -> reject
missing required field       -> reject
operand type mismatch        -> reject
forward operand reference    -> reject
duplicate SSA ID             -> reject
manifest ABI mismatch        -> reject
unknown optional metadata    -> ignore
```

Strict import is important because a malformed IR artifact otherwise fails much
later in LLVM or comgr with poor diagnostics.

## C API Additions

Add a small importer API under `ck_dsl_c/include/ckc/ir_import.h`:

```c
typedef struct ckc_import_options
{
    const char* expected_kernel_name;  /* optional */
    const char* target_arch;           /* optional validation hint */
    bool strict;                       /* default true */
} ckc_import_options_t;

ckc_status_t ckc_import_kernel_from_json(const char* text,
                                         const ckc_import_options_t* opts,
                                         ckc_ir_builder_t* out_builder,
                                         ckc_kernel_def_t** out_kernel,
                                         char* err,
                                         size_t err_cap);
```

Ownership:

- `out_builder` is initialized by the importer;
- `*out_kernel` is owned by `out_builder->arena`;
- caller frees with `ckc_ir_builder_free(out_builder)`;
- on failure, importer frees any partial builder state or leaves a documented
  initialized-empty builder.

The first importer can use the checked `ckc_b_*` builder APIs for common ops and
add an internal generic-op construction helper for opcodes that do not have a
public builder convenience.

## Python Export API

Add an exporter near the Python IR utilities:

```python
from ck_dsl.core.ir_export import export_kernel_ir

payload = export_kernel_ir(kernel, schema="ck.dsl.ir/v1")
```

Suggested API:

```python
def export_kernel_ir(
    kernel: KernelDef,
    *,
    include_debug_names: bool = True,
    target_hint: str | None = None,
    llvm_flavor_hint: str | None = None,
) -> dict:
    ...
```

Add artifact writer integration:

```python
write_artifact(
    artifact,
    output_dir,
    include_portable_ir=True,
)
```

For deterministic artifacts, the JSON writer must:

- sort object keys where order is not semantic;
- preserve op order and param order;
- emit stable SSA IDs;
- emit typed attrs, not Python reprs;
- avoid host-specific paths except optional debug metadata.

## Provider Integration

Extend `ck_dsl_runtime::ArtifactStore::Entry`:

```c++
std::string ir_json_path;
std::string ir_cbor_path;
```

Extend `ArtifactStore::make_kernel` or add a C-JIT-aware overload:

```text
if hsaco exists:
    Kernel::from_hsaco(...)
else if ll exists:
    Kernel::from_llvm_ir(...)
else if portable IR exists and CK_DSL_PROVIDER_C_JIT is enabled:
    import IR through ckc_import_kernel_from_json
    lower through ckc_lower_kernel_to_llvm_ex
    Kernel::from_llvm_ir(...)
else:
    error
```

The provider must not depend on Python for this path. It links only:

- `ck_dsl_runtime`;
- `libckc_core.a`;
- `libamd_comgr`;
- optional `lib_lightgbm` for selection.

## Implementation Milestones

### M0: Schema And Golden Export

- Add `core/ir_export.py`.
- Export scalar, memory, vector, and `scf.for` parity kernels to
  `kernel.ir.json`.
- Add golden JSON tests for stable IDs, attrs, params, and regions.
- Do not modify provider runtime yet.

### M1: C Import Smoke

- Add `ckc/ir_import.h` and `src/ir_import_json.c`.
- Import the four parity kernels and lower with `ckc_lower_kernel_to_llvm_ex`.
- Byte-compare Python-lowered `.ll` vs C-import-lowered `.ll` for the parity set.
- Reject malformed artifacts with useful diagnostics.

### M2: Artifact Store Fallback

- Teach `ArtifactStore` to discover `*.ir.json`.
- When no `.hsaco` or `.ll` exists, import portable IR and lower through C.
- Add a runtime test that ships only `manifest.json + kernel.ir.json`.

### M3: Selected Instance Coverage

- Export one GEMM, one conv, and one attention scalar/tiled smoke kernel.
- Verify manifest ABI matches imported params.
- Run provider end-to-end on `.ir.json`-only bundles.

### M4: Compact Binary Form

- Add `kernel.ir.cbor` once the JSON schema is stable.
- Keep JSON as the canonical debug/test format.
- Require byte-for-byte equivalence between JSON and CBOR importer results.

## Open Questions

- Should portable IR preserve source locations or only stable debug names?
- Should importer validation use manifest kind-specific ABI rules, or stay generic
  and leave kind rules to the provider?
- Should C lowering be allowed to run optimization passes on imported IR, and how
  should pass choices be encoded?
- Should arch-specific operations be represented as generic opcodes plus
  `op_id`, or as arch-specific opcodes? v1 should prefer generic opcode plus
  `op_id` when the current IR already does that.
- Should `.ll` and portable IR both ship in debug bundles, or should `.ll` be
  treated as a generated cache product?

## Recommended First Cut

Implement JSON export/import for the existing parity kernels first. This proves
the graph boundary without entangling the provider, ML heuristic, or large
instance builders.

After byte-identical parity works, wire the artifact store fallback. At that
point Python-authored kernels can ship as portable CK-DSL IR and be JIT-lowered
online by the C backend, while hipDNN remains CPython-free.
