# MMA Metadata and Queries

This reference describes the `MmaOp` metadata and `MmaCatalog` queries used to
select matrix operations. For the overall kernel-building workflow, see the
[authoring model](../architecture/authoring_model.md).

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
