/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.mx_scale.h -- C99 port of rocke.helpers.mx_scale.
 *
 * Scope of THIS file: only `decode_mx_scale_e8m0` is ported here (the other MX
 * helpers in helpers/mx_scale.py -- apply_mx_scale, load_and_decode_mx_scale_byte,
 * load_and_decode_mx_scales_wave -- are out of scope for this phase).
 *
 * MX (microscaling) shared-exponent helper. Decodes an 8-bit E8M0 scale code
 * (one byte per 32-element mantissa block) into its f32 multiplier, per the OCP
 * MX spec, byte-faithfully to the Python:
 *
 *     def decode_mx_scale_e8m0(b: IRBuilder, e8m0: Value) -> Value:
 *         if e8m0.type.name != "i8":
 *             raise ValueError(...)
 *         code = b.zext(e8m0, I32)
 *         is_nan_code = b.cmp_eq(code, b.const_i32(255))
 *         exponent = b.fsub(b.sitofp_f32(code), b.const_f32(127.0))
 *         scale = b.exp2(exponent)
 *         return b.select(is_nan_code, b.const_f32(math.nan), scale)
 *
 * The byte is an unsigned biased exponent: codes 0..254 decode to exactly
 * ``2^(code - 127)`` (code 0 is ``2^-127``, an f32 subnormal) and code 255
 * decodes to NaN.
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

/* Decode an 8-bit E8M0 scale code into an f32 multiplier: ``2^(code - 127)``
 * for codes 0..254, NaN for code 255.
 *
 * `e8m0` must be an i8 Value. On a non-i8 input the builder's sticky error
 * (ROCKE_ERR_VALUE) is set with the Python-matching message and NULL is returned.
 * Returns the f32 multiplier Value (ready for an MX-scale fmul). */
rocke_value_t* rocke_decode_mx_scale_e8m0(rocke_ir_builder_t* b, rocke_value_t* e8m0);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_MX_SCALE_H */
