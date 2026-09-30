/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.i4_dequant.h -- C99 port of four symbols from
 * rocke/helpers/i4_dequant.py:
 *
 *   Python                              C99 (this header)
 *   ---------------------------------   ---------------------------------------
 *   unpack_i4_byte_to_pair_i32(b, byte) rocke_unpack_i4_byte_to_pair_i32(...)
 *   unpack_i4_byte_to_pair_f32(b, byte) rocke_unpack_i4_byte_to_pair_f32(...)
 *   unpack_i4_byte_to_pair_f16(b, byte) rocke_unpack_i4_byte_to_pair_f16(...)
 *   dequant_i4_byte_to_f16_pair(b, ...) rocke_dequant_i4_byte_to_f16_pair(...)
 *
 * SCOPE: only these four symbols. The fp8 / bf8 dequant siblings
 * (dequant_i4_byte_to_fp8_pair / _bf8_pair, which route through
 * rocke.helpers.quant.quantize_scalar_f32), the i8 alias
 * (unpack_i4_byte_to_pair_i8, a thin wrapper over the i32 form), and the OCP MX
 * fp4 / fp6 unpackers are NOT in scope for this phase.
 *
 * Unlike the atoms / distribution helpers, every symbol here EMITS IR via the
 * builder (rocke_b_* op calls). The op sequence is reproduced byte-faithfully from
 * the Python so the lowered IR is identical:
 *
 *   unpack_i4_byte_to_pair_i32 emits, in order:
 *       sext(byte -> i32); const_i32(0x0F); const_i32(8); const_i32(16);
 *       const_i32(4); lshr; land(low); land(high);
 *       cmp_ge(low,8); sub(low,16); select(low);
 *       cmp_ge(high,8); sub(high,16); select(high)
 *     NOTE: the Python evaluates the arguments of land(high) before the land of
 *     low: `low_unsigned = land(byte, mask)` then
 *     `high_unsigned = land(lshr(byte, const_i32(4)), mask)`. The const_i32(4)
 *     and lshr are emitted as part of computing high_unsigned, AFTER low_unsigned
 *     and AFTER the const_i32(8)/const_i32(16) literals (mask_lo, c8, c16 are all
 *     bound before either land). The C port preserves this exact evaluation
 *     order.
 *
 *   unpack_i4_byte_to_pair_f32 emits the i32 unpack, then sitofp_f32(low),
 *       sitofp_f32(high).
 *
 *   unpack_i4_byte_to_pair_f16 emits the f32 unpack, then trunc_f32_to_f16(low),
 *       trunc_f32_to_f16(high).
 *
 *   dequant_i4_byte_to_f16_pair emits the f32 unpack, then
 *       fmul(low, scale); trunc_f32_to_f16(low);
 *       fmul(high, scale); trunc_f32_to_f16(high)
 *     i.e. each lane's fmul is emitted immediately before its trunc (Python
 *     evaluates `trunc_f32_to_f16(fmul(low, scale))` fully before the high lane).
 *
 * Each Python helper returns a `Tuple[Value, Value]`. The C port writes the two
 * results through `out_low` / `out_high` out-params (either may be NULL to skip)
 * and returns a rocke_status_t. The Python type guard
 *
 *     if packed_byte.type.name != "i8": raise ValueError(...)
 *
 * (present only on unpack_i4_byte_to_pair_i32) maps onto a ROCKE_ERR_VALUE sticky
 * error with a Python-matching message; all four entry points propagate it
 * because they all funnel through the i32 unpack.
 *
 * Sticky-error model (shared with the rest of the C port): if the builder is
 * already in an error state every call is a no-op that returns the existing
 * status and leaves the out-params untouched.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_I4_DEQUANT_H
#define ROCKE_HELPER_ROCKE_HELPERS_I4_DEQUANT_H

#include "rocke/ir.h" /* rocke_status_t, rocke_ir_builder_t, rocke_value_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------ unpack_i4_byte_to_pair_i32 *
 *
 * Python:
 *
 *     def unpack_i4_byte_to_pair_i32(b, packed_byte) -> (Value, Value):
 *         if packed_byte.type.name != "i8": raise ValueError(...)
 *         byte_i32 = b.sext(packed_byte, I32)
 *         mask_lo = b.const_i32(0x0F)
 *         c8 = b.const_i32(8)
 *         c16 = b.const_i32(16)
 *         low_unsigned  = b.land(byte_i32, mask_lo)
 *         high_unsigned = b.land(b.lshr(byte_i32, b.const_i32(4)), mask_lo)
 *         low_signed  = b.select(b.cmp_ge(low_unsigned,  c8),
 *                                b.sub(low_unsigned,  c16), low_unsigned)
 *         high_signed = b.select(b.cmp_ge(high_unsigned, c8),
 *                                b.sub(high_unsigned, c16), high_unsigned)
 *         return low_signed, high_signed
 *
 * One byte of packed i4 weights -> two sign-extended i32 values in [-8, 7].
 * `packed_byte` must be of type "i8" (the Python type guard); otherwise the
 * builder records ROCKE_ERR_VALUE and the function returns it. Out-params receive
 * the low and high nibble results. */
rocke_status_t rocke_unpack_i4_byte_to_pair_i32(rocke_ir_builder_t* b,
                                                rocke_value_t* packed_byte,
                                                rocke_value_t** out_low,
                                                rocke_value_t** out_high);

/* ------------------------------------------------ unpack_i4_byte_to_pair_f32 *
 *
 * Python:
 *     low_i32, high_i32 = unpack_i4_byte_to_pair_i32(b, packed_byte)
 *     return b.sitofp_f32(low_i32), b.sitofp_f32(high_i32)
 *
 * One byte of packed i4 -> two f32 values (signed, in [-8, 7]). */
rocke_status_t rocke_unpack_i4_byte_to_pair_f32(rocke_ir_builder_t* b,
                                                rocke_value_t* packed_byte,
                                                rocke_value_t** out_low,
                                                rocke_value_t** out_high);

/* ------------------------------------------------ unpack_i4_byte_to_pair_f16 *
 *
 * Python:
 *     low_f32, high_f32 = unpack_i4_byte_to_pair_f32(b, packed_byte)
 *     return b.trunc_f32_to_f16(low_f32), b.trunc_f32_to_f16(high_f32)
 *
 * One byte of packed i4 -> two f16 values (signed, in [-8, 7]). Lossless
 * sitofp-to-f32 followed by a trunc to f16. */
rocke_status_t rocke_unpack_i4_byte_to_pair_f16(rocke_ir_builder_t* b,
                                                rocke_value_t* packed_byte,
                                                rocke_value_t** out_low,
                                                rocke_value_t** out_high);

/* ------------------------------------------------ dequant_i4_byte_to_f16_pair *
 *
 * Python:
 *     def dequant_i4_byte_to_f16_pair(b, packed_byte, *, scale) -> (Value, Value):
 *         low_f32, high_f32 = unpack_i4_byte_to_pair_f32(b, packed_byte)
 *         low_f16  = b.trunc_f32_to_f16(b.fmul(low_f32,  scale))
 *         high_f16 = b.trunc_f32_to_f16(b.fmul(high_f32, scale))
 *         return low_f16, high_f16
 *
 * Full i4 -> f16 dequant for one packed byte (WMMA prep): unpack to f32,
 * multiply by the forward per-group `scale` in f32, and truncate once to f16.
 * `scale` is the keyword-only argument of the Python signature; here it is a
 * required positional. */
rocke_status_t rocke_dequant_i4_byte_to_f16_pair(rocke_ir_builder_t* b,
                                                 rocke_value_t* packed_byte,
                                                 rocke_value_t* scale,
                                                 rocke_value_t** out_low,
                                                 rocke_value_t** out_high);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_I4_DEQUANT_H */
