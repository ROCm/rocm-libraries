/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_ck_dsl.helpers.quant.h -- C99 port of ck_dsl.helpers.quant.
 *
 * Scope of THIS file: only `quant_ir_type` is ported here (the other quant
 * helpers in helpers/quant.py -- quantize_scalar_f32, dequantize_scalar_to_f32,
 * pack_quant_chunk_local_f32, store_packed_chunk_local, quant_max_abs,
 * ir_to_qdtype -- are out of scope for this phase). The function maps a
 * quant-dtype alias spelling string to the canonical IR scalar type,
 * byte-identically to the Python:
 *
 *     def quant_ir_type(qdtype: str) -> Type:
 *         canon = _canon(qdtype)
 *         if canon == "i8":      return I8
 *         if canon == "fp8e4m3": return FP8E4M3
 *         if canon == "bf8e5m2": return BF8E5M2
 *         raise ValueError(...)
 *
 * with `_canon` normalising the accepted aliases:
 *
 *     "i8" / "int8"                       -> "i8"
 *     "fp8e4m3" / "fp8" / "fp8_e4m3"      -> "fp8e4m3"
 *     "bf8e5m2" / "bf8" / "fp8_e5m2"      -> "bf8e5m2"
 *
 * and raising ValueError for any other spelling.
 *
 * The Python helper takes no IRBuilder and raises ValueError on an unsupported
 * dtype. C99 has no exceptions, so we expose two faithful spellings, mirroring
 * the sibling helper_ck_dsl.helpers.io port:
 *
 *   ckc_quant_ir_type(qdtype)
 *       Pure map. Returns the interned scalar singleton
 *       (ckc_i8()/ckc_fp8e4m3()/ckc_bf8e5m2()) for any accepted alias; returns
 *       NULL for anything else (the analog of "raise ValueError"). No error
 *       state is set because there is no builder.
 *
 *   ckc_b_quant_ir_type(b, qdtype)
 *       Builder-aware spelling matching the rest of the C port's sticky-error
 *       model: on an unsupported dtype it records CKC_ERR_VALUE + a message on
 *       the builder (mirroring the Python ValueError text) and returns NULL.
 */
#ifndef CKC_HELPER_CK_DSL_HELPERS_QUANT_H
#define CKC_HELPER_CK_DSL_HELPERS_QUANT_H

#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Pure quant-dtype-alias-string -> canonical scalar Type.
 *
 * Accepts (matching Python `_QDTYPE_ALIAS`):
 *   "i8", "int8"                  -> ckc_i8()
 *   "fp8e4m3", "fp8", "fp8_e4m3"  -> ckc_fp8e4m3()
 *   "bf8e5m2", "bf8", "fp8_e5m2"  -> ckc_bf8e5m2()
 * Returns NULL for any other value (the Python ValueError path). */
const ckc_type_t* ckc_quant_ir_type(const char* qdtype);

/* Builder-aware variant. Same mapping as ckc_quant_ir_type, but on an
 * unsupported dtype it sets the builder's sticky error (CKC_ERR_VALUE) with the
 * Python-matching message and returns NULL. If the builder is already in an
 * error state it is a no-op returning NULL, like every other ckc_b_* call. */
const ckc_type_t* ckc_b_quant_ir_type(ckc_ir_builder_t* b, const char* qdtype);

#ifdef __cplusplus
}
#endif

#endif /* CKC_HELPER_CK_DSL_HELPERS_QUANT_H */
