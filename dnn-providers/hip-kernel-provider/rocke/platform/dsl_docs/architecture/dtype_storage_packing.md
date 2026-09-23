# Logical dtypes, storage, and operand packing

Matrix inputs and scales use three separate contracts:

1. A logical dtype identifies the numerical encoding and its bit width.
2. Tensor storage describes how elements occupy an addressable buffer.
3. An instruction fragment describes lane mapping, carrier capacity, and padding.

The selected atom determines supported operand combinations and instruction
selectors. An integer carrier does not identify the numerical format.

## Contents

- [Logical dtypes](#logical-dtypes)
- [Common packing](#common-packing)
- [Tensor storage](#tensor-storage)
- [Matrix and scale layouts](#matrix-and-scale-layouts)
- [Mirroring and follow-up work](#mirroring-and-follow-up-work)

## Logical dtypes

[`dtype_info`](../../python/rocke/core/dtypes.py) and
[`rocke_dtype_info`](../../cpp/include/rocke/dtypes.h) return target-independent
encoding descriptors. `dtype_to_ir_type` and `rocke_dtype_to_ir_type` resolve
logical types, including distinct FP4 E2M1, FP6 E2M3, FP6 E3M2, and scale types.
`quant_ir_type` delegates type resolution to this common resolver.

Recognition and serialization do not enable scalar conversion, arithmetic, or
an instruction on a target. In this implementation, FP4/FP6 and the nominal scale
types are used in authoring metadata; storage helpers expand their transport into
existing integer operations before serialization. Raw scalar lowering of these
types remains unsupported. Existing FP16/BF16 and FP8/BF8 lowering is unchanged.

Scale-format validation remains separate from matrix alias normalization:
E5M3 is not BF8 E5M2. Registering a scale dtype does not enable a backend selector.

## Common packing

`BitPacking` / `rocke_bit_packing_t` describes unsigned bit patterns with an
encoded width and slot width. Dense packing uses equal widths. A wider slot has
unused high bits, which packers zero. The current encoding is least-significant
bit first in a little-endian byte stream.

| Format | Dense group occupying whole 32-bit words |
|---|---|
| FP16/BF16 | 2 elements in 1 word |
| FP8/BF8 | 4 elements in 1 word |
| FP6/BF6 | 16 elements across 3 words |
| FP4 | 8 elements in 1 word |

FP6 elements can cross byte and word boundaries. Packing cannot be expressed
generally as `32 // element_bits` elements per word.

`FragmentPacking` / `rocke_fragment_packing_t` combines bit packing with a logical
count and a carrier width/count. Capacity is checked independently of payload
size; unused carriers are zero. Carriers may hold integer bits or typed values,
depending on the instruction ABI.

Host pack/unpack routines process bit patterns only. They do not round, clamp,
decode floats, or apply scales. Their tests use literal expected bits and
independent integer arithmetic; numerical tests remain separate.

## Tensor storage

`TensorStorage` / `rocke_tensor_storage_t` initially supports a two-dimensional
logical shape with a contiguous packed inner axis, an explicit byte row stride,
bit offset, and base alignment. Nonnegative sizes and offsets are checked against
the native uint64 range. A/B descriptors are independent.

For example, dense FP6 uses four elements in three bytes; `slot_bits=8` describes
one six-bit pattern in each byte instead. Both have the same logical dtype.
Address calculation includes the bit offset, and allocation bounds include the
partial tail byte. Empty views require no accesses.

[`storage_ir_type`](../../python/rocke/helpers/mma_io.py) chooses an addressable unit. FP4/FP6 use I8
units while retaining their logical identities in the storage descriptor.
This function is deliberately separate from `dtype_to_ir_type`.

The fragment loader accepts complete, byte-aligned chunks. It derives load
alignment from base alignment, row stride, chunk spacing, and origin. A 24-byte
FP6 chunk becomes 16-byte and 8-byte loads without reading past the chunk.
Partial fragments and nonzero bit origins are rejected by this initial loader.
The host bit packer writes a fresh, exclusively owned output buffer; concurrent
packed stores are not provided.

## Matrix and scale layouts

`MatrixFragmentLayout` describes contiguous K chunks interleaved between lane
groups. Coordinate mapping belongs to the atom-specific layout, not to the
generic bit packer. `ScaledWmmaOp.matrix_layout(operand)` selects independent
A/B layouts. Existing `a_frag_len`/`b_frag_len` retain their ABI-vector meaning.

The current gfx1250 scaled matrix layouts have 64 elements per lane and sixteen
i32 carriers. FP8 occupies all sixteen words. FP4 occupies eight and pads eight.
The FP6 transport descriptor occupies twelve and pads four, without enabling
FP6 catalog entries or numerical conversions.

`ScaleAssociation` records `block_k`: how many source K elements share one
scale. This is separate from the bit layout. Existing `ScalePacking(count,
block_k)` remains a compatibility adapter, exposing both the association and a
shared `FragmentPacking`.

Current scaled-WMMA scale fragments hold four eight-bit patterns in i32 for K32,
or eight in i64 for K16. The first K group occupies the low byte. The A/B scale
coordinate maps determine which scales each lane loads. The same
`pack_fragment_bits` helper can pack six-bit fields that cross word boundaries.

## Mirroring and follow-up work

The Python and C++ helpers expand descriptors before serialization. Their emitted
IR and HIP source are tested byte for byte, including FP16/BF16 typed carriers,
FP4/FP6/FP8 matrix payloads, and scale word packing. Existing GEMM signatures
remain compatible. HIP emits declarations for any encountered vector widths
absent from its fixed compatibility prologue.

The shared descriptor/helper changes can be consumed by both FP4 and FP6
branches. FP4/FP8 builder migration can use them directly. Future FP6 numerical
integration should consume the six-bit storage and fragment descriptors and
retain separate target and conversion validation.

First-class tensor-view/fragment IR nodes, arbitrary packed axes, masked partial
tiles, concurrent packed stores, and scalar low-bit conversions are separate
extensions. Any descriptor retained in future IR nodes must serialize, validate,
participate in cache keys, and have a native implementation.
