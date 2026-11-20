# Declarative dispatch contract

[RUNBOOK.md](RUNBOOK.md) owns ordering. This page owns the declarative arm of a UDD
(Universal Dispatch Descriptor): one kernel launch written as formulas instead of a
native `IKernelDispatchHandler`. The specification is RFC 0017 §6
(`projects/hipdnn/docs/rfcs/0017_UniversalKernelDescriptor.md`); the loader is
`plugin_sdk/include/hipdnn_plugin_sdk/ingestor/CompiledLaunch.hpp`.

## When it fits

Use the declarative arm when every launch quantity is a formula over what the engine's
match binds, the kernel's KMD (Kernel Metadata Descriptor) values and device properties:

- one launch per kernel (the kernel carries `kernel_source`, not `sources`);
- a precompiled kernel: authored `hip`, `rocke` or `hsaco`, packed to `kpack`;
- arguments that are tensor pointers, the plan workspace, or `i32`/`u32`/`i64`/`f32`
  scalars computed when the plan is built.

Keep a native `dispatch_symbol` ([native-pack.md](native-pack.md)) for anything else:
several launches, `embedded_source` (the `direct_load` dialect), compile options chosen at
plan build, a scalar read from the variant pack at execute time, or host logic between
launches. A UDD carries exactly one of the two arms.

## Authoring steps

1. Give the engine a `graph_match.nodes` pattern that binds every tensor the kernel takes.
   A `pointer`, `dim` or `stride` argument must name a pattern variable; under a `native`
   match the loader cannot check the name, and an engine with no `graph_match` refuses
   such an argument.
2. Declare each `$kernel.*` field a formula reads in the engine's KMD.
3. Write the UDD: `grid` and `block` (all three axes), `args_signature` in the kernel's
   parameter order, and `shared_mem_bytes` / `workspace_bytes` when they are not 0.
4. Author the kernels as `hip`, `rocke` or `hsaco`, so `hkp_pack` packs them to `kpack`.
   The packed `signature` must agree with `args_signature` in count, kind and size.
5. Run `hkp_pack`, which applies the loader's shape checks (an expression object is
   compiled, and its references resolved, only by the loader), then the structural
   validator and the host checks (RUNBOOK stage 4). The provider registers the one handler
   every declarative UDD uses; a pack registers nothing for it.

## Worked example: `conv_fwd`

The declarative twin of the `conv_fwd` test pack lives in
`dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/test_descriptors/shared/conv_fwd/conv_fwd_declarative.*`.
The kernel is `ConvFwd(x, w, y, n, c, h, width, k, r, s)`, one thread per output element.

The engine binds the three tensors:

```json
"graph_match": {"nodes": [
  {"kind": "op", "id": "conv", "op": "convolution",
   "operands": {"x": "$x", "w": "$w"}, "results": {"y": "$y"}}
]}
```

The UDD replaces the native `hipkernel.conv_fwd.dispatch`:

```json
{
  "version": "1.0",
  "id": "9168d9cc-b56b-433e-b24c-b04052ba469f",
  "name": "conv_fwd declarative dispatch",
  "grid": {
    "x": {"ceil_div": [
      {"*": ["$x.dims[0]", "$w.dims[0]",
             {"-": [{"+": ["$x.dims[2]", 1]}, "$w.dims[2]"]},
             {"-": [{"+": ["$x.dims[3]", 1]}, "$w.dims[3]"]}]},
      "$kernel.block_size"]},
    "y": 1, "z": 1
  },
  "block": {"x": "$kernel.block_size", "y": 1, "z": 1},
  "workspace_bytes": 0,
  "args_signature": [
    {"name": "x",     "kind": "pointer", "source": {"from": "tensor", "ref": "$x"}},
    {"name": "w",     "kind": "pointer", "source": {"from": "tensor", "ref": "$w"}},
    {"name": "y",     "kind": "pointer", "source": {"from": "tensor", "ref": "$y"}},
    {"name": "n",     "kind": "scalar", "type": "i32", "source": {"from": "dim", "ref": "$x", "axis": 0}},
    {"name": "c",     "kind": "scalar", "type": "i32", "source": {"from": "dim", "ref": "$x", "axis": 1}},
    {"name": "h",     "kind": "scalar", "type": "i32", "source": {"from": "dim", "ref": "$x", "axis": 2}},
    {"name": "width", "kind": "scalar", "type": "i32", "source": {"from": "dim", "ref": "$x", "axis": 3}},
    {"name": "k",     "kind": "scalar", "type": "i32", "source": {"from": "dim", "ref": "$w", "axis": 0}},
    {"name": "r",     "kind": "scalar", "type": "i32", "source": {"from": "dim", "ref": "$w", "axis": 2}},
    {"name": "s",     "kind": "scalar", "type": "i32", "source": {"from": "dim", "ref": "$w", "axis": 3}}
  ]
}
```

- The grid is `ceil(n * k * p * q / block_size)`, with `p = h - r + 1` and
  `q = width - s + 1`, the geometry the native handler computes.
- `$kernel.block_size` is a KMD field; each kernel supplies its own value.
- The packed signature is three 8-byte `global_buffer` entries and seven 4-byte
  `by_value` entries, which is what these ten arguments declare.

## Formula rules

A formula is an expression object, a `$` reference or a number, in the JSON Expression
Language (`projects/hipdnn/docs/JsonExpression.md`). It reads the symbols criteria read:
the binding, `$graph.*`, `$kernel.*` (and `name[i]` of an `int_list` field) and
`$device.*`. Arithmetic is double, so an integer is a whole number of magnitude at most
2^53.

| Field | Accepted value |
|---|---|
| `grid` / `block` axis | integer in [1, 2^32 - 1] |
| `shared_mem_bytes` (default 0) | integer in [0, 2^32 - 1] |
| `workspace_bytes` (default 0) | integer in [0, 2^53] |
| `i32` / `u32` / `i64` scalar | integer in the type's range (`i64`: magnitude at most 2^53) |
| `f32` scalar | finite number within the float range |

| Argument `kind` | `type` | `source` |
|---|---|---|
| `pointer` | none | `{"from": "tensor", "ref": "$v"}` |
| `scalar` | `i32`, `u32`, `i64` or `f32` | `{"from": "dim" \| "stride", "ref": "$v", "axis": i}` or `{"from": "expr", "expr": E}` |
| `workspace` | none | none |

`ref` is `$` plus an identifier, `axis` a non-negative integer, and argument names are
non-empty and unique. Keys starting `x-` or `_` are ignored at every level.

## What is refused

| Refused | Where | Effect |
|---|---|---|
| Both `dispatch_symbol` and a declarative key, or neither | load (`hkp_pack` first) | UDD file skipped |
| `dispatch_symbol` equal to or under `hipdnn.declarative_launch` | load | UDD file skipped |
| Unknown key, missing `grid`/`block`/`args_signature` or axis, bad argument entry | load | UDD file skipped |
| `launches`, `intermediates`, `offset`, source `scalar` or `attribute` ("not supported yet") | load | UDD file skipped |
| A number literal outside its rule; a string, bool, null or array literal | load | UDD file skipped |
| A reference the engine does not publish, or a tensor argument that names no pattern variable | pairing UDD with engine | pack dropped |
| An `embedded_source` kernel, or a packed signature that disagrees | catalog build | kernel dropped |
| A multi-launch kernel (`sources`) | catalog build | kernel dropped |
| A formula that yields null or breaks its rule for one graph | workspace query / plan build | kernel declines that graph |

## Triage

Every load-time refusal is an ERROR the default log level hides, and a dropped pack or
engine looks like a plain decline. Run with the log level at INFO, or run the structural
validator, and search for these messages:

- `<UDD path> is not a valid descriptor:`: a static refusal. The message names the key,
  the argument and the rule. Every pack naming the UDD then drops with `names dispatch
  descriptor <id>, which no descriptor defines`.
- `pairs dispatch '<name>' ... with engine '<e>', but its launch reads $<ref>, ...` or
  `... binds argument '<a>' to $<v>, ...`, then `dropping the pack`: the pairing check.
  Bind the tensor in the pattern, declare the KMD field, or fix the reference.
- `is EMBEDDED_SOURCE, but pack ... launches it through a declarative UDD` or
  `takes the arguments ..., but the declarative UDD of pack ... passes ...`: author the
  kernel as a packed kind, or fix `args_signature` order, kind or type.
- `names unregistered dispatch symbol 'hipdnn.declarative_launch'`: the provider does not
  register the declarative handler; the set is dropped.
- `<udd name>: <field> evaluated to <value>, but it must be ...`: a formula that fails for
  this shape. A plan build logs it in a WARN (`could not build a plan for ... trying the
  next candidate`); a failing `workspace_bytes` logs it at INFO (`kernel ... declined the
  graph:`) and removes the kernel from the workspace query and the workspace-limit filter.

A plan built from a declarative UDD saves its resolved launch under
`hipdnn.declarative_launch.v1`; a restored plan does not re-evaluate the formulas.
