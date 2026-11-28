---
name: i4
aliases: [i4, int4, u4, uint4]
family: integer
bits: [4]
variants:
  - {arch: all, encoding: "two's complement nibble (i4); unsigned nibble (u4); packed two per byte"}
status: on develop; dequantize only
related: [platform/python/rocke/helpers/i4_dequant.py]
---

# i4 and u4 (packed)

4-bit integers that exist only packed: the IR refuses a scalar or vector `i4`
or `u4` value, and a kernel keeps the words in `i32` storage and unpacks them
([dtype_storage_packing.md](../architecture/dtype_storage_packing.md)). Ranges
are in the [Quick values](./README.md#integer-formats) table.

## Unpacking

`unpack_i4_byte_to_pair_i32` reads one byte as two signed nibbles, **low
nibble first** (bits 0-3 are the first element), and sign-extends each to
-8 .. 7 with a mask and a conditional subtract
([helpers/i4_dequant.py](../../python/rocke/helpers/i4_dequant.py)).
This matches CK Tile's i4 weight packer.

## Conversions

| helper | result | exactness |
|---|---|---|
| `unpack_i4_byte_to_pair_f32`, `_f16` | the integer as f32 / f16 | exact: every i4 value is representable |
| `dequant_i4_byte_to_f16_pair(..., scale)` | `f16(i4 * scale)` with the product in f32 | one rounding, to f16 (IEEE, see [f16](./datatype_f16.md)) |

There is no f32 -> i4 quantizer. The integer WMMA reads `i4` / `u4` operands
with the signedness taken from the operand's dtype.

## Policy

Dequantization is exact up to the final rounding of the destination type.
There is no quantizer, so no rounding or saturation policy for producing
i4 yet.

## Tests

No dedicated test pins the nibble order or sign extension yet; the i4 GEMM
families exercise it end to end.

## How to decide

- **Dequantizing i4 weights**: use the helpers above; do not shift bytes by
  hand, since the nibble order is a contract with the packer.
- **Producing i4**: pack on the host for now.
