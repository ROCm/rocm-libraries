/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.quant.h -- C99 port of rocke.helpers.quant.
 *
 * Type resolution only; scalar quantization and conversion remain separate.
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

/* Pure quant-dtype-alias-string -> canonical scalar Type.
 *
 * Accepts (matching Python `_QDTYPE_ALIAS`):
 *   "i8", "int8"                  -> rocke_i8()
 *   "fp8e4m3", "fp8", "fp8_e4m3"  -> rocke_fp8e4m3()
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

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_QUANT_H */
