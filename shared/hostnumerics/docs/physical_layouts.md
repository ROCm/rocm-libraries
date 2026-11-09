# Physical layouts

A tensor's natural order identifies which value belongs to each coordinate.
A kernel may load the same values from a different physical order. HostNumerics
describes that rearrangement in Core and supplies AMD GPU kernel presets in
`roc::hostnumerics-amd-gpu-layout`.

## Padding, reshape, and permutation

[`ReshapePermutation`](../include/roc/hostnumerics/reshape_permutation.hpp)
describes three steps: pad a natural shape without changing its coordinates,
reshape the padded storage into factors, and order those factors from slowest
to fastest. All shapes use last-dimension-fastest order. The explicit
`Permutation` must contain every reshaped axis once.

`physicalIndex(i)` maps a valid flat natural index to an element offset in the
result. It does not copy values or select a padding value. Construction checks
the natural and padded ranks, containment of the natural extents, preservation
of the padded element count by the reshape, and the size and stride limits of
`Shape` and `Layout`. A scalar has one element; an empty tensor has no valid
natural index. The [tutorial test](../tests/reshape_permutation_test.cpp) shows
the complete mapping and materialization of a small padded tensor.

## MX scale presets

An MX scale is shared by a block of operand values along the reduction dimension.
For one scale per row and block of K values, the natural scale shape is
`[rows, ceil(K / blockSize)]`, with one byte per scale. A matrix instruction
specifies the association between scales and operand values in registers.
The kernel chooses their global-memory layout and the loads or rearrangements
that deliver them to those registers. A GPU architecture therefore does not
uniquely determine the layout of a scale buffer.

[`MxScaleStoragePlan`](../include/roc/hostnumerics/amd_gpu_layout/mx.hpp) provides
these named kernel conventions. Each plan exposes its full `ReshapePermutation`
through `transform()`, along with natural and physical byte counts.

| Preset | Padding and reshape | Axis order, slowest to fastest |
| --- | --- | --- |
| `Natural` | Keep the input shape. | Identity. |
| `Tile32x8` | Pad `[rows, blocks]` to `[R, S]`, multiples of `[32, 8]`; reshape to `[R/32, 2, 16, S/8, 2, 4]`. | `{0, 3, 5, 2, 4, 1}` |
| `FastDimensionTiles` | Set `q = 128 / blockSize`, pad the fast extent to `F`, a multiple of `q`; reshape to `[slow, F/q, q]`. | `{1, 0, 2}` |

`Tile32x8` describes the scale order used by current gfx950 kernels, including
the shuffle in AITER. `FastDimensionTiles` describes the current gfx1250
convention and supports block sizes 16 and 32. Callers first orient scales as
the selected kernel expects, then choose the preset. Architecture-name lookup
belongs in the product adapter that knows which kernel will consume the buffer.

`copyMxScaleStorageToPhysicalLayout` materializes the plan with zero bytes in
padding. It preserves each scale byte without interpreting its numerical value.
The [MX tutorial and tests](../tests/amd_gpu_mx_layout_test.cpp) show usage and
check exact output orders and padding.

## Operand copies

[`copyOperandToPhysicalLayout`](../include/roc/hostnumerics/amd_gpu_layout/operand.hpp)
accepts K-contiguous input with explicit K and MN extents. Optional swizzle and
tile configurations select the kernel's lane/register order and contiguous
workgroup tiles. Their factor permutations use the same slowest-to-fastest
convention as Core. The byte copy is shared with MX scales and preserves any
trivially copyable element type; these routines do not encode or decode scalars.
The [operand tutorial](../tests/amd_gpu_operand_layout_test.cpp) demonstrates
both transformations.
