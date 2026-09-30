/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.mx_scale.h -- C99 port of rocke.helpers.mx_scale.
 *
 * Scope of THIS file: only `decode_mx_scale_e8m0` is ported here (the other MX
 * helpers in helpers/mx_scale.py -- apply_mx_scale, load_and_decode_mx_scale_byte,
 * load_and_decode_mx_scales_wave -- are out of scope for this phase).
 *
 * MX (microscaling) shared-exponent helper. Decodes an 8-bit unbiased E8M0
 * exponent (one byte per 32-element mantissa block) into an f32 multiplier
 * ``2^(e - 127)``, byte-faithfully to the Python:
 *
 *     def decode_mx_scale_e8m0(b: IRBuilder, e8m0: Value) -> Value:
 *         if e8m0.type.name != "i8":
 *             raise ValueError(...)
 *         e_i32 = b.sext(e8m0, I32)
 *         is_zero = b.cmp_eq(e_i32, b.const_i32(0))
 *         is_nan  = b.cmp_eq(e_i32, b.const_i32(255))
 *         e_minus_127_f32 = b.fsub(b.sitofp_f32(e_i32), b.const_f32(127.0))
 *         raw_scale = b.exp2(e_minus_127_f32)
 *         zero = b.const_f32(0.0)
 *         return b.select(b.lor(is_zero, is_nan), zero, raw_scale)
 *
 * Sentinels: ``e == 0`` (subnormal) and ``e == 255`` (NaN) both decode to
 * ``0.0`` so downstream accumulators are not poisoned, matching the AMDGPU MX
 * MFMA hardware path. ``1 <= e <= 254`` decodes to ``2^(e - 127)``.
 *
 * The Python raises ValueError on a non-i8 input. C99 has no exceptions, so the
 * builder-aware spelling follows the rest of the C port's sticky-error model:
 * on a non-i8 input it records ROCKE_ERR_VALUE + a Python-matching message on the
 * builder and returns NULL. If the builder is already in an error state it is a
 * no-op returning NULL, like every other rocke_b_* call.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_MX_SCALE_H
#define ROCKE_HELPER_ROCKE_HELPERS_MX_SCALE_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Decode an 8-bit E8M0 exponent value into an f32 multiplier ``2^(e - 127)``,
 * with the ``e == 0`` / ``e == 255`` sentinels mapped to ``0.0``.
 *
 * `e8m0` must be an i8 Value. On a non-i8 input the builder's sticky error
 * (ROCKE_ERR_VALUE) is set with the Python-matching message and NULL is returned.
 * Returns the f32 multiplier Value (ready for an MX-scale fmul). */
rocke_value_t* rocke_decode_mx_scale_e8m0(rocke_ir_builder_t* b, rocke_value_t* e8m0);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_MX_SCALE_H */
