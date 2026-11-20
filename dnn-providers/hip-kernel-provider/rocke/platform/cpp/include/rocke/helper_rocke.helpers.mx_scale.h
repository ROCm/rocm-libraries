/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.mx_scale.h -- C99 port of rocke.helpers.mx_scale.
 *
 * Scope of THIS file: `decode_mx_scale_e8m0`, `add_mx_scaled_dot`, the
 * element formats (`MxElementFormat` and the four fp8 / bf8 encodings) and
 * `largest_safe_mx_scale_code` are ported here (the other MX helpers in
 * helpers/mx_scale.py -- apply_mx_scale, load_and_decode_mx_scale_byte,
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

#include <stddef.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Python MxElementFormat: an element format's largest finite magnitude, which
 * bounds the products of an MX group; name names the format in messages. */
typedef struct rocke_mx_element_format
{
    const char* name;
    double largest_value;
} rocke_mx_element_format_t;

/* The fp8 and bf8 encodings matrix cores read (Python FP8_E4M3_OCP,
 * BF8_E5M2_OCP, FP8_E4M3_FNUZ, BF8_E5M2_FNUZ). */
extern const rocke_mx_element_format_t ROCKE_FP8_E4M3_OCP;
extern const rocke_mx_element_format_t ROCKE_BF8_E5M2_OCP;
extern const rocke_mx_element_format_t ROCKE_FP8_E4M3_FNUZ;
extern const rocke_mx_element_format_t ROCKE_BF8_E5M2_FNUZ;

/* Python largest_safe_mx_scale_code(a_format, b_format, products): the largest
 * E8M0 code an A and a B scale may both have so that a sum of `products`
 * scaled A x B products stays a binade below the f32 overflow threshold
 * (|sum| <= products * a_max * b_max * 2^(2c - 254) <= 2^127). See the Python
 * docstring.
 *
 * Returns ROCKE_OK and writes the code to *out, or ROCKE_ERR_VALUE with the
 * Python-matching message in err (capacity err_cap, may be NULL) when
 * `products` is not positive. */
rocke_status_t rocke_largest_safe_mx_scale_code(const rocke_mx_element_format_t* a_format,
                                                const rocke_mx_element_format_t* b_format,
                                                int products,
                                                int* out,
                                                char* err,
                                                size_t err_cap);

/* Decode an 8-bit E8M0 scale code into an f32 multiplier: ``2^(code - 127)``
 * for codes 0..254, NaN for code 255.
 *
 * `e8m0` must be an i8 Value. On a non-i8 input the builder's sticky error
 * (ROCKE_ERR_VALUE) is set with the Python-matching message and NULL is returned.
 * Returns the f32 multiplier Value (ready for an MX-scale fmul). */
rocke_value_t* rocke_decode_mx_scale_e8m0(rocke_ir_builder_t* b, rocke_value_t* e8m0);

/* Python add_mx_scaled_dot(b, acc, dot, a_codes, b_codes):
 * ``acc + dot * 2^(ea + eb)`` element by element, NaN for an element whose A or
 * B code is 255. `acc` and `dot` are matching f32 vectors of n elements;
 * `a_codes` and `b_codes` hold the n A and B E8M0 bytes (i8) of the elements
 * (the same Value may repeat when elements share a scale; each distinct Value
 * is read once). Each term is ldexp(dot, ea + eb): one rounding, correct for
 * every pair of codes, as the product of two decoded scales is never formed.
 *
 * On a non-f32-vector dot, a mismatched accumulator, a code count that is not
 * n or a non-i8 code the builder's sticky error (ROCKE_ERR_VALUE) is set with
 * the Python-matching message and NULL is returned. Returns the result vector
 * Value. */
rocke_value_t* rocke_add_mx_scaled_dot(rocke_ir_builder_t* b,
                                       rocke_value_t* acc,
                                       rocke_value_t* dot,
                                       rocke_value_t* const* a_codes,
                                       size_t n_a_codes,
                                       rocke_value_t* const* b_codes,
                                       size_t n_b_codes);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_MX_SCALE_H */
