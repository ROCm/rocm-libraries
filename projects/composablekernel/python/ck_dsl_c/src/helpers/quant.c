/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_ck_dsl.helpers.quant.c -- C99 port of
 * ck_dsl.helpers.quant.quant_ir_type.
 *
 * Faithful translation of:
 *
 *     _QDTYPE_ALIAS = {
 *         "i8": "i8", "int8": "i8",
 *         "fp8e4m3": "fp8e4m3", "fp8": "fp8e4m3", "fp8_e4m3": "fp8e4m3",
 *         "bf8e5m2": "bf8e5m2", "bf8": "bf8e5m2", "fp8_e5m2": "bf8e5m2",
 *     }
 *
 *     def _canon(qdtype: str) -> QDType:
 *         if qdtype not in _QDTYPE_ALIAS:
 *             raise ValueError(
 *                 f"unsupported quant dtype {qdtype!r}; expected one of "
 *                 f"{sorted(_QDTYPE_ALIAS)}")
 *         return _QDTYPE_ALIAS[qdtype]
 *
 *     def quant_ir_type(qdtype: str) -> Type:
 *         canon = _canon(qdtype)
 *         if canon == "i8":      return I8
 *         if canon == "fp8e4m3": return FP8E4M3
 *         if canon == "bf8e5m2": return BF8E5M2
 *         raise ValueError(f"unreachable: canon={canon!r}")
 *
 * Mapping invariants (must stay byte-identical to the Python so downstream IR is
 * identical):
 *   "i8" / "int8"                  -> ckc_i8()        (Python I8)
 *   "fp8e4m3" / "fp8" / "fp8_e4m3" -> ckc_fp8e4m3()   (Python FP8E4M3)
 *   "bf8e5m2" / "bf8" / "fp8_e5m2" -> ckc_bf8e5m2()   (Python BF8E5M2)
 *   else                           -> NULL            (Python ValueError)
 */

#include "ckc/helper_ck_dsl.helpers.quant.h"

#include <string.h>

#include "ckc/ir_internal.h" /* ckc_i_set_err, ckc_i_live */

const ckc_type_t* ckc_quant_ir_type(const char* qdtype)
{
    if (qdtype == NULL)
    {
        return NULL;
    }
    /* `_canon` -> "i8" : keys "i8", "int8". */
    if (strcmp(qdtype, "i8") == 0 || strcmp(qdtype, "int8") == 0)
    {
        return ckc_i8();
    }
    /* `_canon` -> "fp8e4m3" : keys "fp8e4m3", "fp8", "fp8_e4m3". */
    if (strcmp(qdtype, "fp8e4m3") == 0 || strcmp(qdtype, "fp8") == 0 ||
        strcmp(qdtype, "fp8_e4m3") == 0)
    {
        return ckc_fp8e4m3();
    }
    /* `_canon` -> "bf8e5m2" : keys "bf8e5m2", "bf8", "fp8_e5m2". */
    if (strcmp(qdtype, "bf8e5m2") == 0 || strcmp(qdtype, "bf8") == 0 ||
        strcmp(qdtype, "fp8_e5m2") == 0)
    {
        return ckc_bf8e5m2();
    }
    /* Python: _canon raises ValueError. No builder here, so signal via NULL. */
    return NULL;
}

const ckc_type_t* ckc_b_quant_ir_type(ckc_ir_builder_t* b, const char* qdtype)
{
    const ckc_type_t* ty;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if (!ckc_i_live(b))
    {
        return NULL;
    }

    ty = ckc_quant_ir_type(qdtype);
    if (ty == NULL)
    {
        /* Mirror the Python `_canon` ValueError, including the {qdtype!r}
         * single-quote repr for the (non-NULL) string case and the sorted
         * key list `sorted(_QDTYPE_ALIAS)`. NULL is reported as "None" to
         * match Python's repr(None). */
        return (const ckc_type_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE,
            "unsupported quant dtype %s%s%s; expected one of "
            "['bf8', 'bf8e5m2', 'fp8', 'fp8_e4m3', 'fp8_e5m2', 'fp8e4m3', "
            "'i8', 'int8']",
            qdtype ? "'" : "", qdtype ? qdtype : "None", qdtype ? "'" : "");
    }
    return ty;
}
