/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.mx_scale.h -- C99 port of rocke.helpers.mx_scale.
 *
 * Scope of THIS file: `decode_mx_scale_e8m0`, `fma_with_mx_scale_pair`, the
 * element formats (`MxElementFormat` and the four fp8 / bf8 encodings) and
 * `exact_mx_scale_split` are ported here (the other MX helpers in
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

/* Python MxElementFormat: the values an element format can hold, as far as an
 * MX group's dot product depends on them. The smallest positive value is
 * 2^smallest_exponent (the smallest subnormal) and every value is a multiple
 * of it; largest_value is the largest finite magnitude; name appears in error
 * messages. */
typedef struct rocke_mx_element_format
{
    const char* name;
    int smallest_exponent;
    double largest_value;
} rocke_mx_element_format_t;

/* The fp8 and bf8 encodings matrix cores read (Python FP8_E4M3_OCP,
 * BF8_E5M2_OCP, FP8_E4M3_FNUZ, BF8_E5M2_FNUZ). */
extern const rocke_mx_element_format_t ROCKE_FP8_E4M3_OCP;
extern const rocke_mx_element_format_t ROCKE_BF8_E5M2_OCP;
extern const rocke_mx_element_format_t ROCKE_FP8_E4M3_FNUZ;
extern const rocke_mx_element_format_t ROCKE_BF8_E5M2_FNUZ;

/* Python MxScaleSplit: how rocke_fma_with_mx_scale_pair splits 2^(ea + eb)
 * into two f32 factors. ea + eb is split as pre + fused with fused clamped to
 * fused_min..fused_max and pre the rest, then pre is saturated at
 * rocke_mx_scale_split_pre_max. As exponent_cap is at least fused_max, that
 * is the split of ea + eb saturated at exponent_cap. */
typedef struct rocke_mx_scale_split
{
    int exponent_cap;
    int fused_min;
    int fused_max;
} rocke_mx_scale_split_t;

/* Python MxScaleSplit.pre_max: the largest pre, exponent_cap - fused_max.
 * ea + eb above exponent_cap has fused = fused_max and pre beyond this. */
int rocke_mx_scale_split_pre_max(const rocke_mx_scale_split_t* split);

/* Python exact_mx_scale_split(a_format, b_format, group_size): the split that
 * makes rocke_fma_with_mx_scale_pair exact (one rounding, IEEE special values
 * and signed zeros) for every dot product of group_size A and B elements of
 * these formats. See the Python docstring for the derivation.
 *
 * Returns ROCKE_OK and fills *out, or ROCKE_ERR_VALUE with the
 * Python-matching message in err (capacity err_cap, may be NULL) when the
 * formats' bounds admit no exact split. */
rocke_status_t rocke_exact_mx_scale_split(const rocke_mx_element_format_t* a_format,
                                          const rocke_mx_element_format_t* b_format,
                                          int group_size,
                                          rocke_mx_scale_split_t* out,
                                          char* err,
                                          size_t err_cap);

/* Decode an 8-bit E8M0 scale code into an f32 multiplier: ``2^(code - 127)``
 * for codes 0..254, NaN for code 255.
 *
 * `e8m0` must be an i8 Value. On a non-i8 input the builder's sticky error
 * (ROCKE_ERR_VALUE) is set with the Python-matching message and NULL is returned.
 * Returns the f32 multiplier Value (ready for an MX-scale fmul). */
rocke_value_t* rocke_decode_mx_scale_e8m0(rocke_ir_builder_t* b, rocke_value_t* e8m0);

/* ``addend + value * 2^(ea + eb)`` for two E8M0 scale bytes, rounded once;
 * NaN if either code is 255. ``ea``, ``eb`` are the unbiased exponents of the
 * i8 codes `a_e8m0`, `b_e8m0`; `value` and `addend` are matching f32 vectors.
 *
 * The two exponents are added as integers and never formed into one f32
 * scale, which would overflow to Inf or flush to zero for large or small
 * codes: ``ea + eb`` is split as ``pre + fused`` with ``fused`` clamped to
 * split->fused_min..fused_max and ``pre`` the rest, saturated at
 * rocke_mx_scale_split_pre_max(split), and the result is
 * ``fma(value * 2^pre, 2^fused, addend)``. With `split` from
 * rocke_exact_mx_scale_split for the formats of the group's elements, that
 * is the exact result rounded once for every value the group can produce.
 * See the Python docstring of rocke.helpers.mx_scale.fma_with_mx_scale_pair.
 *
 * On a non-f32-vector value, a mismatched addend or a non-i8 code the
 * builder's sticky error (ROCKE_ERR_VALUE) is set with the Python-matching
 * message and NULL is returned. Returns the result vector Value. */
rocke_value_t* rocke_fma_with_mx_scale_pair(rocke_ir_builder_t* b,
                                            rocke_value_t* value,
                                            rocke_value_t* a_e8m0,
                                            rocke_value_t* b_e8m0,
                                            rocke_value_t* addend,
                                            const rocke_mx_scale_split_t* split);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_MX_SCALE_H */
