// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Mirrors helpers/quant.py: _canon, quant_ir_type, quant_max_abs,
 * validate_quant_out_dtype and quantize_scalar_f32. One alias table answers
 * both of Python's alias maps (_QDTYPE_ALIAS for scalar quantization,
 * _QUANT_TYPE_ALIAS for type recognition, which adds the FP4/FP6 names) and
 * spells both refusal messages. FP4/FP6 recognition is independent of scalar
 * quantization/conversion support.
 */

#include "rocke/helper_rocke.helpers.quant.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h" /* rocke_arch_target_from_gfx */
#include "rocke/helper_rocke.helpers.fp8_encoding.h" /* rocke_fp8_element_format */
#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live */

/* Python _QUANT_TYPE_ALIAS, sorted by alias as Python's sorted() orders the
 * names in the refusal messages. scalar marks the _QDTYPE_ALIAS entries, the
 * dtypes scalar quantization accepts. */
typedef struct rocke_quant_alias
{
    const char* alias;
    const char* canon;
    bool scalar;
} rocke_quant_alias_t;

static const rocke_quant_alias_t ROCKE_QUANT_ALIASES[] = {
    {"bf6", "fp6e3m2", false},
    {"bf8", "bf8e5m2", true},
    {"bf8e5m2", "bf8e5m2", true},
    {"e4m3", "fp8e4m3", true},
    {"fp4", "fp4e2m1", false},
    {"fp4e2m1", "fp4e2m1", false},
    {"fp6", "fp6e2m3", false},
    {"fp6e2m3", "fp6e2m3", false},
    {"fp6e3m2", "fp6e3m2", false},
    {"fp8", "fp8e4m3", true},
    {"fp8_e4m3", "fp8e4m3", true},
    {"fp8_e5m2", "bf8e5m2", true},
    {"fp8e4m3", "fp8e4m3", true},
    {"i8", "i8", true},
    {"int8", "i8", true},
};

#define ROCKE_QUANT_ALIAS_COUNT (sizeof(ROCKE_QUANT_ALIASES) / sizeof(ROCKE_QUANT_ALIASES[0]))

/* The table row of `qdtype`, or NULL. */
static const rocke_quant_alias_t* quant_alias(const char* qdtype)
{
    size_t i;

    for(i = 0; qdtype != NULL && i < ROCKE_QUANT_ALIAS_COUNT; ++i)
    {
        if(strcmp(qdtype, ROCKE_QUANT_ALIASES[i].alias) == 0)
        {
            return &ROCKE_QUANT_ALIASES[i];
        }
    }
    return NULL;
}

/* Python: f"unsupported quant dtype {qdtype!r}; expected one of {sorted(names)}"
 * where names are the scalar-quantization aliases (scalar_only) or all of them. */
static void format_unsupported(char* err, size_t err_cap, const char* qdtype, bool scalar_only)
{
    size_t i;
    size_t used;
    bool first = true;

    if(err == NULL || err_cap == 0)
    {
        return;
    }
    used = (size_t)snprintf(err,
                            err_cap,
                            "unsupported quant dtype %s%s%s; expected one of [",
                            qdtype != NULL ? "'" : "",
                            qdtype != NULL ? qdtype : "None",
                            qdtype != NULL ? "'" : "");
    for(i = 0; i < ROCKE_QUANT_ALIAS_COUNT && used < err_cap; ++i)
    {
        if(scalar_only && !ROCKE_QUANT_ALIASES[i].scalar)
        {
            continue;
        }
        used += (size_t)snprintf(
            err + used, err_cap - used, "%s'%s'", first ? "" : ", ", ROCKE_QUANT_ALIASES[i].alias);
        first = false;
    }
    if(used < err_cap)
    {
        snprintf(err + used, err_cap - used, "]");
    }
}

const char* rocke_quant_canon(const char* qdtype)
{
    const rocke_quant_alias_t* row = quant_alias(qdtype);
    return (row != NULL && row->scalar) ? row->canon : NULL;
}

const rocke_type_t* rocke_quant_ir_type(const char* qdtype)
{
    const rocke_quant_alias_t* row = quant_alias(qdtype);
    /* Python: _canon raises ValueError. No builder here, so signal via NULL. */
    return row != NULL ? rocke_dtype_to_ir_type(row->canon) : NULL;
}

const rocke_type_t* rocke_b_quant_ir_type(rocke_ir_builder_t* b, const char* qdtype)
{
    char err[ROCKE_ERR_MSG_CAP];
    const rocke_type_t* ty;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!rocke_i_live(b))
    {
        return NULL;
    }

    ty = rocke_quant_ir_type(qdtype);
    if(ty == NULL)
    {
        /* Mirror the Python `quant_ir_type` ValueError, listing
         * sorted(_QUANT_TYPE_ALIAS). */
        format_unsupported(err, sizeof(err), qdtype, false);
        return (const rocke_type_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", err);
    }
    return ty;
}

rocke_status_t rocke_quant_max_abs(
    const char* qdtype, const char* arch, double* out, char* err, size_t err_cap)
{
    const char* canon = rocke_quant_canon(qdtype);
    const rocke_mx_element_format_t* format;

    if(out == NULL)
    {
        if(err != NULL && err_cap > 0)
        {
            snprintf(err, err_cap, "rocke_quant_max_abs: NULL out");
        }
        return ROCKE_ERR_VALUE;
    }
    if(canon == NULL)
    {
        /* Python _canon: ValueError listing sorted(_QDTYPE_ALIAS). */
        format_unsupported(err, err_cap, qdtype, true);
        return ROCKE_ERR_VALUE;
    }
    if(strcmp(canon, "i8") == 0)
    {
        *out = ROCKE_QUANT_I8_MAX_ABS;
        return ROCKE_OK;
    }
    format = rocke_fp8_element_format(canon, arch, err, err_cap);
    if(format == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    *out = format->largest_value;
    return ROCKE_OK;
}

bool rocke_validate_quant_out_dtype(const char* out_dtype,
                                    const char* arch,
                                    char* reason,
                                    size_t reason_cap)
{
    const rocke_arch_target_t* target;
    const char* canon;
    double bound;

    if(reason != NULL && reason_cap > 0)
    {
        reason[0] = '\0';
    }
    /* Python: out_dtype not in get_args(QDType). A canonical name is the
     * canonical form of itself; an alias is not. */
    canon = rocke_quant_canon(out_dtype);
    if(canon == NULL || strcmp(canon, out_dtype) != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "unsupported out_dtype %s%s%s",
                     out_dtype != NULL ? "'" : "",
                     out_dtype != NULL ? out_dtype : "None",
                     out_dtype != NULL ? "'" : "");
        }
        return false;
    }
    if(strcmp(out_dtype, "i8") == 0)
    {
        return true;
    }
    target = rocke_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "unknown arch %s", arch != NULL ? arch : "(null)");
        }
        return false;
    }
    /* fp8 / bf8 need the CDNA-only v_cvt_pk_{fp8,bf8}_f32 conversion. */
    if(target->family == NULL || strcmp(target->family, "cdna") != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "out_dtype '%s' needs the CDNA-only v_cvt_pk_{fp8,bf8}_f32 "
                     "conversion; %s (family %s%s%s) has no fp8/bf8 pack op -- use "
                     "out_dtype='i8'",
                     out_dtype,
                     arch,
                     target->family != NULL ? "'" : "",
                     target->family != NULL ? target->family : "None",
                     target->family != NULL ? "'" : "");
        }
        return false;
    }
    /* Python: quant_max_abs(out_dtype, arch=arch) refuses a target with no
     * known encoding; its message is the reason. */
    return rocke_quant_max_abs(out_dtype, arch, &bound, reason, reason_cap) == ROCKE_OK;
}

rocke_value_t* rocke_b_quantize_scalar_f32(rocke_ir_builder_t* b,
                                           rocke_value_t* x_f32,
                                           rocke_value_t* inv_scale,
                                           const char* qdtype,
                                           const char* arch)
{
    char err[ROCKE_ERR_MSG_CAP];
    const char* canon;
    double qmax;
    rocke_value_t* c_pos;
    rocke_value_t* c_neg;
    rocke_value_t* scaled;
    rocke_value_t* clamped;
    rocke_value_t* is_nan;
    rocke_value_t* saturated;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(x_f32 == NULL || x_f32->type == NULL || strcmp(x_f32->type->name, "f32") != 0)
    {
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "quantize_scalar_f32 expects f32 input, got %s",
            (x_f32 != NULL && x_f32->type != NULL) ? x_f32->type->name : "None");
    }
    if(inv_scale == NULL || inv_scale->type == NULL || strcmp(inv_scale->type->name, "f32") != 0)
    {
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "quantize_scalar_f32 expects f32 inv_scale, got %s",
            (inv_scale != NULL && inv_scale->type != NULL) ? inv_scale->type->name : "None");
    }
    /* canon = _canon(qdtype); qmax = quant_max_abs(canon, arch=arch) */
    if(rocke_quant_max_abs(qdtype, arch, &qmax, err, sizeof(err)) != ROCKE_OK)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", err);
    }
    canon = rocke_quant_canon(qdtype);
    /* c_pos = b.const_f32(qmax); c_neg = b.const_f32(-qmax)
     * scaled = b.fmul(x_f32, inv_scale)
     * clamped = b.clamp_f32(scaled, c_neg, c_pos) */
    c_pos = rocke_b_const_f32(b, qmax);
    c_neg = rocke_b_const_f32(b, -qmax);
    scaled = rocke_b_fmul(b, x_f32, inv_scale);
    clamped = rocke_b_clamp_f32(b, scaled, c_neg, c_pos);
    if(strcmp(canon, "i8") == 0)
    {
        return rocke_b_cvt_f32_to_i8_sat(b, clamped);
    }
    /* saturated = b.select(b.fcmp("uno", scaled, scaled), scaled, clamped):
     * a NaN keeps its encoding instead of becoming the clamp's lower bound. */
    is_nan = rocke_b_fcmp(b, "uno", scaled, scaled);
    saturated = rocke_b_select(b, is_nan, scaled, clamped);
    if(strcmp(canon, "fp8e4m3") == 0)
    {
        return rocke_b_cvt_f32_to_fp8(b, saturated);
    }
    return rocke_b_cvt_f32_to_bf8(b, saturated);
}
