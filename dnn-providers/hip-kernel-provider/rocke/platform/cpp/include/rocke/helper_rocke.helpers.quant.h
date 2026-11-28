/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.quant.h -- C99 port of rocke.helpers.quant.
 *
 *   Python (helpers/quant.py)                         C99 (this file)
 *   _I8_MAX_ABS                                   ->  ROCKE_QUANT_I8_MAX_ABS
 *   _canon(qdtype)                                ->  rocke_quant_canon()
 *   quant_ir_type(qdtype)                         ->  rocke_quant_ir_type() /
 *                                                     rocke_b_quant_ir_type()
 *   quant_max_abs(qdtype, arch=arch)              ->  rocke_quant_max_abs()
 *   validate_quant_out_dtype(out_dtype, arch)     ->  rocke_validate_quant_out_dtype()
 *   quantize_scalar_f32(b, x, inv_scale=,
 *                       qdtype=, arch=)           ->  rocke_b_quantize_scalar_f32()
 *
 * Python's quantize_prescaled_f32 (the same clamp and conversion for a value
 * the caller has already scaled) has no C port yet; port it when a C++
 * builder needs it.
 *
 * rocke_quant_ir_type delegates to rocke_dtype_to_ir_type for nominal types,
 * preserving the existing quant aliases and adding FP4/FP6 recognition.
 * Returns a borrowed static singleton or NULL for an unsupported name.
 * The builder-aware variant also records the corresponding error.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_QUANT_H
#define ROCKE_HELPER_ROCKE_HELPERS_QUANT_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Python _I8_MAX_ABS: the i8 clamp magnitude, CK Tile's symmetric
 * [-127, 127] rather than the type's -128. */
#define ROCKE_QUANT_I8_MAX_ABS 127.0

/* Python _canon(qdtype): the canonical name ("i8", "fp8e4m3" or "bf8e5m2") of
 * a scalar-quantization alias, or NULL for any other name. Borrowed static
 * string. */
const char* rocke_quant_canon(const char* qdtype);

/* Pure quant-dtype-alias-string -> canonical scalar Type.
 *
 * Accepts (matching Python `_QDTYPE_ALIAS`):
 *   "i8", "int8"                  -> rocke_i8()
 *   "fp8e4m3", "fp8", "fp8_e4m3", "e4m3" -> rocke_fp8e4m3()
 *   "bf8e5m2", "bf8", "fp8_e5m2"  -> rocke_bf8e5m2()
 *   "fp4", "fp4e2m1"              -> rocke_fp4e2m1()
 *   "fp6", "fp6e2m3"              -> rocke_fp6e2m3()
 *   "bf6", "fp6e3m2"              -> rocke_fp6e3m2()
 * Returns NULL for any other value (the Python ValueError path). */
const rocke_type_t* rocke_quant_ir_type(const char* qdtype);

/* Builder-aware variant. Same mapping as rocke_quant_ir_type, but on an
 * unsupported dtype it sets the builder's sticky error (ROCKE_ERR_VALUE) with the
 * Python-matching message and returns NULL. If the builder is already in an
 * error state it is a no-op returning NULL, like every other rocke_b_* call. */
const rocke_type_t* rocke_b_quant_ir_type(rocke_ir_builder_t* b, const char* qdtype);

/* Python quant_max_abs(qdtype, arch=arch): the clamp magnitude of
 * rocke_b_quantize_scalar_f32. 127 for i8 (CK Tile's symmetric range); for
 * fp8e4m3 / bf8e5m2 the largest finite value of the encoding `arch` converts
 * to (rocke_fp8_element_format): 240 / 57344 for gfx942's FNUZ, 448 / 57344
 * for OCP. `qdtype` may be any i8 / fp8 / bf8 alias Python's _canon accepts.
 *
 * Returns ROCKE_OK and writes the bound to *out, or ROCKE_ERR_VALUE with the
 * Python-matching message in err (capacity err_cap, may be NULL) for an
 * unsupported dtype or an fp8 / bf8 dtype on a target with no known encoding. */
rocke_status_t rocke_quant_max_abs(
    const char* qdtype, const char* arch, double* out, char* err, size_t err_cap);

/* Python validate_quant_out_dtype(out_dtype, arch): whether a quantizing
 * kernel can write `out_dtype` on `arch`. `out_dtype` must be "i8", "fp8e4m3"
 * or "bf8e5m2" (no aliases); fp8 / bf8 need a CDNA target (the
 * v_cvt_pk_{fp8,bf8}_f32 conversion) with a known encoding
 * (rocke_quant_max_abs). Returns true, or false with the Python-matching reason
 * in `reason` (capacity reason_cap, may be NULL). `arch` must be a known
 * target: Python raises KeyError for any other, and this returns false with
 * "unknown arch <arch>". */
bool rocke_validate_quant_out_dtype(const char* out_dtype,
                                    const char* arch,
                                    char* reason,
                                    size_t reason_cap);

/* Python quantize_scalar_f32(b, x_f32, inv_scale=inv_scale, qdtype=qdtype,
 * arch=arch): x_f32 * inv_scale, clamped to rocke_quant_max_abs and converted
 * to qdtype in `arch`'s encoding. For fp8 / bf8 a NaN skips the clamp and
 * stays NaN; finite values and +-inf saturate. For i8 a NaN gives -127.
 *
 * Sticky-error model: on a non-f32 x_f32 / inv_scale or a refused qdtype /
 * arch it records ROCKE_ERR_VALUE with the Python-matching message and
 * returns NULL; on a builder already in an error state it returns NULL. */
rocke_value_t* rocke_b_quantize_scalar_f32(rocke_ir_builder_t* b,
                                           rocke_value_t* x_f32,
                                           rocke_value_t* inv_scale,
                                           const char* qdtype,
                                           const char* arch);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_QUANT_H */
