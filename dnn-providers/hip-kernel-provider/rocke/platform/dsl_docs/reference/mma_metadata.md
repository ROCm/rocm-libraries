# MMA Metadata and Queries

This reference describes the `MmaOp` metadata and `MmaCatalog` queries used to
select matrix operations. For the overall kernel-building workflow, see the
[authoring model](../architecture/authoring_model.md).

## Selection, IR Emission, and Dispatch

These APIs operate at different levels:

| API | Responsibility | Result |
|---|---|---|
| `target.mma.op_for_shape(...)` | Match one instruction's shape, A/B/C dtypes, family and scale contract in the target catalog | An `MmaOp`; no match returns `None`, ambiguity raises `ValueError`; no IR is emitted |
| `builder.mma(atom, a, b, c, *extra)` | Record the selected operation and its operands | One `tile.mma` IR operation; lowering later emits the LLVM intrinsic selected by `op_id` |
| `MfmaAtom.emit_scaled(...)` | Call the existing named 16x16x128 scaled-MFMA wrapper | Delegates to `builder.mma` with a fixed operation ID, regardless of the helper atom's shape or dtype. It passes the optional `a_format` and `b_format` (default `fp8e4m3`) to the builder. A and B must use the same format in this version |
| Operator dispatcher | Select a kernel implementation and specialization for a workload | A kernel builder and spec; that builder selects and emits instructions |

`mma` also accepts a raw operation-ID string, as the existing named wrappers
do. The selected `MmaOp` supplies accumulator size and dtype; the operation ID
identifies the concrete lowering. Matrix loads, packing and scale-tensor loads
belong to the surrounding kernel builder. A dispatcher is above this authoring
layer: request -> kernel/spec selection -> IR construction -> compilation/launch.
Direct examples may start from a spec and builder without dispatch.

### gfx950 Scaled-MFMA Paths

Both instruction shapes have rocKE lowering support, through different
authoring entrypoints:

| Path | Authoring entrypoint | Per-lane FP32 accumulator | Current consumer |
|---|---|---|---|
| Existing 16x16x128 scaled wrapper | `IRBuilder.mfma_scale_f32_16x16x128_f8f6f4(...)`, also called by `MfmaAtom.emit_scaled(...)` | 4 values | Direct operation authoring; `a_format` and `b_format` set the `cbsz` and `blgp` format selectors (default `fp8e4m3`). A and B must use the same format in this version. Every operand has 256 bits per lane |
| Unscaled 16x16x128 op for every format | `IRBuilder.mfma_f32_16x16x128_fp4fp6fp8(...)` | 4 values | Direct operation authoring; both scale operands are literal 0, and `a_format` and `b_format` select the data type. A and B must use the same format in this version |
| Existing 16x16x128 FP8 atom | `MfmaAtom.fp8_16x16x128().emit(...)` | 4 values | gfx950 FP8 fused-MoE dispatcher; the same scaled intrinsic is used with constant in-instruction scales |
| Native 32x32x64 scaled family | `op_for_shape(family="mfma_scaled", ...)`, then `IRBuilder.mma(atom, a, b, c, a_scale, b_scale)` | 16 values | Direct packed GEMM builder and untuned examples, with independent A/B formats and K32 E8M0 scales |

The [fused-MoE dispatcher](../../python/rocke/dispatch/families/moe.py) selects
the 16x16x128 FP8 atom on gfx950 and the 16x16x32 FP8 atom on gfx942. Its
[FP8 builder](../../python/rocke/instances/common/moe_fused_mega_fp8.py) uses
`atom.emit()` and applies FP32 block-scale factors outside the MFMA. It does
not call `emit_scaled()` with variable E8M0 operands. The native 32x32x64
GEMM instead loads raw E8M0 bytes and passes separate A/B scales to the
instruction. It currently has no GEMM dispatcher registration.

The new 32x32x64 lowering lives beside the existing 16x16x128 lowering in
[`core/lower_llvm.py`](../../python/rocke/core/lower_llvm.py) and
[`cpp/core/lower_llvm/mma.cpp`](../../cpp/core/lower_llvm/mma.cpp). Its catalog
and lane maps are in the existing architecture layer. The Python/C++
`instances/gfx950/block_scaled_gemm` builders and `examples/gfx950/gemm`
entrypoints mirror the gfx1250 scaled-GEMM structure. See the
[gfx950 family contract](../instances/gfx950_block_scaled_gemm.md) and
[example guide](../../python/rocke/examples/gfx950/gemm/README.md).

## Operand Metadata

MMA metadata uses `a_dtype`, `b_dtype`, and `c_dtype`, with C describing both
the accumulator input and result. Optional `a_scale_dtype`, `b_scale_dtype`,
and `scale_block_k` fields describe independent scale types and a shared K-group
size. `MmaScaleBlockK.K16` and `MmaScaleBlockK.K32` are the only scaled sizes;
all three fields are `None` for unscaled atoms. Integer 16/32 values from JSON
are normalized to the enum. Scale types are independent of matrix dtypes and
packed register types. Matrix fragment and layout accessors retain their A/B/C roles.

Scale dtypes normalize to the string-backed `MmaScaleDType` enum: `E8M0`,
`E4M3`, or `E5M3`. Existing string inputs remain accepted, and string formatting
and JSON retain the canonical `e8m0`, `e4m3`, and `e5m3` spellings.
`fp8e4m3` is an alias for `E4M3`. E5M3 is an unsigned scale format, distinct from
the signed E5M2 matrix format: `e5m2`, `fp8e5m2`, `bf8e5m2`, and `bf8` are
rejected as scale dtypes. LLVM defines separate matrix and scale format names
in [WMMAMods](https://github.com/ROCm/llvm-project/blob/6bd443a039454b7d2d9f34740ce0a6e3c10ac8d4/llvm/lib/Target/AMDGPU/Utils/AMDGPUAsmUtils.h#L146).
Enum membership describes a format; supported operations still come from the
target catalog and backend.

## Scale Layouts

`a_scale_frag_len` and `b_scale_frag_len` count logical scale elements per lane.
`a_scale_layout()` maps `(lane, slot)` to `(row, K-group)`;
`b_scale_layout()` maps it to `(K-group, column)`. These instruction-local maps
are separate from register byte packing and global tensor strides. Unscaled
atoms default to zero scale fragments and absent maps. An unavailable map raises
`NotImplementedError`, including for a scaled atom whose layout is not verified.
The native record appends corresponding counts and map pointers, so native
consumers must rebuild; existing role enum values remain unchanged.

### gfx1250 Scale Mapping and Packing

The four gfx1250 FP8/BF8 scaled atoms provide these scale maps: lane `l` and
slot `j` map to `(l % 16, j)` for A and `(j, l % 16)` for B. Both half-waves
duplicate the scales. K32 uses four E8M0 elements per lane packed into i32;
K16 uses eight packed into i64, with slot `j` at bit offset `8*j`. Matrix A/B
maps for these atoms remain unavailable. The gfx1250 loader applies tile and
instruction-step offsets to the scale coordinates before applying tensor strides.

## Catalog Queries

For scaled operations, query the complete contract and pass the selected atom
to `IRBuilder.mma`:

```python
from rocke.core.arch import ArchTarget, MmaScaleBlockK, MmaScaleDType

atom = ArchTarget.from_gfx("gfx1250").mma.op_for_shape(
    family="wmma_scaled",
    a_dtype="fp8", b_dtype="fp8", c_dtype="fp32",
    scales=(MmaScaleDType.E8M0, MmaScaleDType.E8M0, MmaScaleBlockK.K32),
    m=16, n=16, k=128,
)
assert atom is not None
# result = builder.mma(atom, a, b, c, scale_a, scale_b)
```

For gfx950, select the 32x32x64 scaled-MFMA family in the same way. This
example selects FP8 A and FP4 B, with independent E8M0 scales over K32:

```python
atom = ArchTarget.from_gfx("gfx950").mma.op_for_shape(
    family="mfma_scaled",
    a_dtype="fp8", b_dtype="fp4", c_dtype="fp32",
    scales=(MmaScaleDType.E8M0, MmaScaleDType.E8M0, MmaScaleBlockK.K32),
    m=32, n=32, k=64,
)
assert atom is not None
# result = builder.mma(atom, a, b, c, scale_a, scale_b)
```

This operation takes two `vec<i32x8>` packed payloads, a `vec<f32x16>`
accumulator and two i32 scale operands. The catalog describes all 25 ordered
FP8/BF8/FP6/BF6/FP4 pairs. The scale-group K=32 is separate from instruction
K=64. The existing 16x16x128 scaled wrapper retains its fixed operation ID;
`MfmaAtom.emit_scaled()` does not perform this catalog lookup.

Omitting `scales` leaves scales unconstrained. Passing `(a_type, b_type, block_k)`
matches the scale contract exactly; `(None, None, None)` selects unscaled atoms.
Partially specified scale contracts are invalid. Enumeration
and existence queries may match several records, while `op_for_shape` and
`select_largest_k` reject ambiguous exact matches or largest-K ties. The C queries
use the same rules: a NULL `rocke_mma_scale_filter_t` pointer is unconstrained,
and `{NULL, NULL, ROCKE_MMA_SCALE_NONE}` requests an unscaled atom. The C block
enum has `ROCKE_MMA_SCALE_K16` and `ROCKE_MMA_SCALE_K32`, plus the unscaled
sentinel. Scale value formats are `e8m0`, `e4m3`, and `e5m3`, with
`fp8e4m3` accepted as an alias for `e4m3`; actual target support comes from the
catalog.

### Native C/C++ Atom Lookup

When holding an `ArchTarget`, use `rocke_archtarget_op_for_shape` from
`rocke/helper_rocke.core.arch.h`. It forwards to
`rocke_mma_catalog_op_for_shape(&target->mma, ...)`, which is also available
directly from `rocke/arch_target.h`. Both select the same catalog record.
Kernel builders use that record's operation ID, fragment sizes, and layouts;
support checks use the lookup to establish whether the requested atom exists.
The `(m, n, k)` arguments describe one instruction atom, not the full GEMM.

This C++ example selects the gfx1250 FP8 atom with E8M0 scales shared over K32:

```cpp
#include "rocke/error.hpp"
#include "rocke/helper_rocke.core.arch.h"

int main()
{
    try
    {
        const auto* target = rocke_archtarget_from_gfx("gfx1250");
        const rocke_mma_scale_filter_t scales = {"e8m0", "e8m0", ROCKE_MMA_SCALE_K32};
        const auto* atom = rocke_archtarget_op_for_shape(
            target, "wmma_scaled", "fp8", "fp8", "fp32", 16, 16, 128, &scales);
        return atom ? 0 : 1;
    }
    catch(const ckc::Error&)
    {
        return 2;
    }
}
```

The helper requires the trailing filter argument in both C and C++.
Pass `NULL` (`nullptr` in C++) for unconstrained scales, or a pointer to
`{NULL, NULL, ROCKE_MMA_SCALE_NONE}` to select only unscaled atoms.
The scale-contract rules above apply. No match, or a null target, returns
`NULL`; invalid filters and ambiguous matches raise `ckc::Error`. Handle those
errors at a C++ boundary before returning to C. In particular, the scaled
FP8/BF8 `16x16x128` shapes have both K16 and K32 records, so an unconstrained
lookup is ambiguous.

## Operation IDs and Migration

Scaled-WMMA IDs have the form
`wmma_<gfx>_<acc>_<MxNxK>_<a>_<b>_scale_<a_scale>_<b_scale>_k<block>`.
For example, `wmma_gfx1250_f32_16x16x128_fp8_fp8_scale_e8m0_e8m0_k32`
has atom K=128 and one scale per 32 K elements for both inputs. Both scale
types are written even when equal. Lowering reads the catalog fields, never
parses the ID, and selects LLVM intrinsic names and packed carriers separately.
Existing `wmma_scale*_f32_*` and dotted `wmma.scaled.*` IDs are retired, as are
the dedicated scaled builder wrappers; serialized IR using those IDs must be regenerated.
Use `tile.mma` with a resolved catalog atom. Other MMA operation IDs retain
their existing spelling.

## Logical TF32 operands

On gfx942, `tf32` (`xf32` alias) selects `mfma_f32_16x16x8_xf32` or
`mfma_f32_32x32x4_xf32`. Both take two logical TF32 A/B elements per lane;
C/D are FP32 with four or sixteen elements. FP32 queries retain the ordinary
FP32 atoms. The logical type uses one I32 storage word per element and is
bitcast to the intrinsic's float-vector ABI during lowering.

On gfx950, `tf32` selects the emulated `mfma_f32_16x16x32_xf32` or
`mfma_f32_32x32x16_xf32`: eight logical TF32 A/B elements per lane, C/D FP32 with
four or sixteen elements, so the same `(tf32, tf32, fp32)` query returns an atom with
the same operand ABI on either target. Such a row carries an `emulation` descriptor
(`MmaOp.emulation`, `is_native` is `False`) naming the physical BF16 atom it expands
into; its fragment lengths and layout maps are inherited from that atom, and
the catalog refuses a row whose physical atom is not a native atom of the same arch.
The lowering reinterprets the TF32 payload as FP32 and splits it, so no payload bit is
dropped; see the [emulation description](../architecture/xf32_bf16x3_emulation.md).

See the [TF32 numerical example](../../python/rocke/examples/gfx942/tf32_numerics/README.md)
for the distinction between raw FP32 payloads and explicit RNE preparation,
and for a reproducible comparison through both engines.

## FP64 operands

On gfx942, an `fp64`/`fp64`/`fp64` query selects `mfma_f64_16x16x4_f64`
(`m=n=16`) or `mfma_f64_4x4x4_f64` (`m=n=4`, four independent blocks). A/B take
one f64 element per lane, as `f64` or a one-element `vec<f64x1>` (reinterpreted
with one bitcast); another number format is rejected. C/D take four and one, so
`IRBuilder.mma` returns `<4 x f64>` or `<1 x f64>`. The accumulator element
follows the atom's `c_dtype`. Both C/D layouts differ from the f32 atoms (see
the [MFMA atom catalog](mfma_atom_catalog.md)). The probe instance
`instances/gfx942/f64_mma_probe.py` drives both atoms through both engines.
