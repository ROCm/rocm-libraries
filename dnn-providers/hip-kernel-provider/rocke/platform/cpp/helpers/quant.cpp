// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Mirrors helpers/quant.py::quant_ir_type. Preserve the existing quant aliases
 * and delegate logical type resolution to rocke_dtype_to_ir_type. FP4/FP6
 * recognition is independent of scalar quantization/conversion support.
 */

#include "rocke/helper_rocke.helpers.quant.h"

#include <string.h>

#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live */

const rocke_type_t* rocke_quant_ir_type(const char* qdtype)
{
    if(qdtype == NULL)
    {
        return NULL;
    }
    /* `_canon` -> "i8" : keys "i8", "int8". */
    if(strcmp(qdtype, "i8") == 0 || strcmp(qdtype, "int8") == 0)
    {
        return rocke_dtype_to_ir_type("i8");
    }
    /* `_canon` -> "fp8e4m3" : keys "fp8e4m3", "fp8", "fp8_e4m3". */
    if(strcmp(qdtype, "fp8e4m3") == 0 || strcmp(qdtype, "fp8") == 0
       || strcmp(qdtype, "fp8_e4m3") == 0)
    {
        return rocke_dtype_to_ir_type("fp8e4m3");
    }
    /* `_canon` -> "bf8e5m2" : keys "bf8e5m2", "bf8", "fp8_e5m2". */
    if(strcmp(qdtype, "bf8e5m2") == 0 || strcmp(qdtype, "bf8") == 0
       || strcmp(qdtype, "fp8_e5m2") == 0)
    {
        return rocke_dtype_to_ir_type("bf8e5m2");
    }
    if(strcmp(qdtype, "fp4") == 0 || strcmp(qdtype, "fp4e2m1") == 0 || strcmp(qdtype, "fp6") == 0
       || strcmp(qdtype, "fp6e2m3") == 0 || strcmp(qdtype, "bf6") == 0
       || strcmp(qdtype, "fp6e3m2") == 0)
        return rocke_dtype_to_ir_type(qdtype);
    /* Python: _canon raises ValueError. No builder here, so signal via NULL. */
    return NULL;
}

const rocke_type_t* rocke_b_quant_ir_type(rocke_ir_builder_t* b, const char* qdtype)
{
    const rocke_type_t* ty;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!rocke_i_live(b))
    {
        return NULL;
    }

    ty = rocke_quant_ir_type(qdtype);
    if(ty == NULL)
    {
        /* Mirror the Python `_canon` ValueError, including the {qdtype!r}
         * single-quote repr for the (non-NULL) string case and the sorted
         * key list `sorted(_QDTYPE_ALIAS)`. NULL is reported as "None" to
         * match Python's repr(None). */
        return (const rocke_type_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "unsupported quant dtype %s%s%s; expected one of "
            "['bf8', 'bf8e5m2', 'fp8', 'fp8_e4m3', 'fp8_e5m2', 'fp8e4m3', "
            "'i8', 'int8']",
            qdtype ? "'" : "",
            qdtype ? qdtype : "None",
            qdtype ? "'" : "");
    }
    return ty;
}
