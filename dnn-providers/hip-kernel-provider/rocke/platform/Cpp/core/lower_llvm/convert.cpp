// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_llvm_lower_llvm_convert.c -- BUCKET 2 of the C99 port of
 * rocke.core.lower_llvm: the width/format conversion op handlers.
 *
 * Covers all the cast/convert handlers: zext/sext/trunc/bitcast,
 * f32<->f16 (fpext/fptrunc), sitofp, rint, and the fp8/bf8/i8
 * quant/dequant family (cvt_*_fp8/bf8/i8, scalef32 fused scale+dequant,
 * packed x4 variants). These are a faithful translation of the Python
 * ``_op_arith_*`` methods listed for this bucket.
 *
 * Shared helpers (rocke_ll_emit, rocke_ll_emitf, rocke_ll_operand,
 * rocke_ll_llvm_type, rocke_ll_need, rocke_ll_fail, ...) live in bucket 0 and are
 * reached through rocke/lower_llvm_internal.h.
 */

#include "rocke/lower_llvm_internal.h"

#include <stddef.h>

namespace ckc
{

/* ---------------------------------------------------------------------------
 * Small local helpers
 * ------------------------------------------------------------------------- */

/* The single result name for a one-result op (Python op.result.name). Every
 * handler in this bucket produces exactly one result, mirroring the Python
 * ``(...) = op.operands`` / ``op.result`` pattern. */
static const char* rocke_ll_result_name(const rocke_op_t* op)
{
    if(op->num_results < 1 || op->results[0] == NULL)
    {
        return "%__bad_result";
    }
    return op->results[0]->name;
}

/* "<result_name><suffix>" arena-formatted derived SSA name (Python
 * f"{op.result.name}{suffix}"). */
static const char* rocke_ll_derived(rocke_lower_t* L, const rocke_op_t* op, const char* suffix)
{
    const char* base = rocke_ll_result_name(op);
    char* s = rocke_arena_printf(&L->arena, "%s%s", base, suffix);
    return s ? s : base;
}

/* ---------------------------------------------------------------------------
 * zext / sext / trunc / bitcast
 * ------------------------------------------------------------------------- */

static void _op_arith_zext(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    rocke_ll_emitf(L,
                   "  %s = zext %s %s to %s",
                   rocke_ll_result_name(op),
                   rocke_ll_llvm_type(L, v->type),
                   rocke_ll_operand(L, v),
                   rocke_ll_llvm_type(L, op->results[0]->type));
}

static void _op_arith_sext(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    rocke_ll_emitf(L,
                   "  %s = sext %s %s to %s",
                   rocke_ll_result_name(op),
                   rocke_ll_llvm_type(L, v->type),
                   rocke_ll_operand(L, v),
                   rocke_ll_llvm_type(L, op->results[0]->type));
}

static void _op_arith_trunc(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    rocke_ll_emitf(L,
                   "  %s = trunc %s %s to %s",
                   rocke_ll_result_name(op),
                   rocke_ll_llvm_type(L, v->type),
                   rocke_ll_operand(L, v),
                   rocke_ll_llvm_type(L, op->results[0]->type));
}

static void _op_arith_bitcast(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    rocke_ll_emitf(L,
                   "  %s = bitcast %s %s to %s",
                   rocke_ll_result_name(op),
                   rocke_ll_llvm_type(L, v->type),
                   rocke_ll_operand(L, v),
                   rocke_ll_llvm_type(L, op->results[0]->type));
}

/* ---------------------------------------------------------------------------
 * f32 <-> f16 / sitofp / rint
 * ------------------------------------------------------------------------- */

static void _op_arith_trunc_f32_to_f16(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    rocke_ll_emitf(
        L, "  %s = fptrunc float %s to half", rocke_ll_result_name(op), rocke_ll_operand(L, v));
}

static void _op_arith_cast_to_f32(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    rocke_ll_emitf(L,
                   "  %s = fpext %s %s to float",
                   rocke_ll_result_name(op),
                   rocke_ll_llvm_type(L, v->type),
                   rocke_ll_operand(L, v));
}

static void _op_arith_cast_f32_to(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    rocke_ll_emitf(L,
                   "  %s = fptrunc float %s to %s",
                   rocke_ll_result_name(op),
                   rocke_ll_operand(L, v),
                   rocke_ll_llvm_type(L, op->results[0]->type));
}

static void _op_arith_sitofp_f32(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    if(v->type == NULL || v->type->name == NULL || v->type->kind != ROCKE_TYPE_SCALAR
       || v->type->scalar != ROCKE_SCALAR_I32)
    {
        rocke_ll_fail(L,
                      ROCKE_ERR_NOTIMPL,
                      "arith.sitofp_f32 supports i32 input only, got %s",
                      (v->type && v->type->name) ? v->type->name : "<null>");
    }
    rocke_ll_emitf(
        L, "  %s = sitofp i32 %s to float", rocke_ll_result_name(op), rocke_ll_operand(L, v));
}

static void _op_arith_rint_f32(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    rocke_ll_need(L, "rint.f32");
    rocke_ll_emitf(L,
                   "  %s = call float @llvm.rint.f32(float %s)",
                   rocke_ll_result_name(op),
                   rocke_ll_operand(L, v));
}

/* ---------------------------------------------------------------------------
 * fp8 / bf8 -> f32 (single element, byte-lane 0)
 * ------------------------------------------------------------------------- */

static void _op_arith_cvt_fp8_to_f32(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const char* res = rocke_ll_result_name(op);
    const char* tmp = rocke_ll_derived(L, op, "x");
    rocke_ll_need(L, "amdgcn.cvt.f32.fp8");
    rocke_ll_emitf(L, "  %s = zext i8 %s to i32", tmp, rocke_ll_operand(L, v));
    rocke_ll_emitf(L, "  %s = call float @llvm.amdgcn.cvt.f32.fp8(i32 %s, i32 0)", res, tmp);
}

static void _op_arith_cvt_bf8_to_f32(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const char* res = rocke_ll_result_name(op);
    const char* tmp = rocke_ll_derived(L, op, "x");
    rocke_ll_need(L, "amdgcn.cvt.f32.bf8");
    rocke_ll_emitf(L, "  %s = zext i8 %s to i32", tmp, rocke_ll_operand(L, v));
    rocke_ll_emitf(L, "  %s = call float @llvm.amdgcn.cvt.f32.bf8(i32 %s, i32 0)", res, tmp);
}

/* ---------------------------------------------------------------------------
 * packed <4 x fp8/bf8> -> <4 x f32>  (2x v_cvt_pk_f32_*)
 * ------------------------------------------------------------------------- */

static void _op_arith_cvt_pk_f32_fp8x4(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const char* res = rocke_ll_result_name(op);
    const char* packed_i32 = rocke_ll_derived(L, op, "p");
    const char* lo = rocke_ll_derived(L, op, "lo");
    const char* hi = rocke_ll_derived(L, op, "hi");
    rocke_ll_need(L, "amdgcn.cvt.pk.f32.fp8");
    rocke_ll_emitf(L, "  %s = bitcast <4 x i8> %s to i32", packed_i32, rocke_ll_operand(L, v));
    rocke_ll_emitf(
        L, "  %s = call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %s, i1 false)", lo, packed_i32);
    rocke_ll_emitf(
        L, "  %s = call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %s, i1 true)", hi, packed_i32);
    rocke_ll_emitf(L,
                   "  %s = shufflevector <2 x float> %s, <2 x float> %s, "
                   "<4 x i32> <i32 0, i32 1, i32 2, i32 3>",
                   res,
                   lo,
                   hi);
}

static void _op_arith_cvt_pk_f32_bf8x4(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const char* res = rocke_ll_result_name(op);
    const char* packed_i32 = rocke_ll_derived(L, op, "p");
    const char* lo = rocke_ll_derived(L, op, "lo");
    const char* hi = rocke_ll_derived(L, op, "hi");
    rocke_ll_need(L, "amdgcn.cvt.pk.f32.bf8");
    rocke_ll_emitf(L, "  %s = bitcast <4 x i8> %s to i32", packed_i32, rocke_ll_operand(L, v));
    rocke_ll_emitf(
        L, "  %s = call <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32 %s, i1 false)", lo, packed_i32);
    rocke_ll_emitf(
        L, "  %s = call <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32 %s, i1 true)", hi, packed_i32);
    rocke_ll_emitf(L,
                   "  %s = shufflevector <2 x float> %s, <2 x float> %s, "
                   "<4 x i32> <i32 0, i32 1, i32 2, i32 3>",
                   res,
                   lo,
                   hi);
}

/* ---------------------------------------------------------------------------
 * fused scale + dequant: <4 x fp8/bf8> + f32 scale -> <4 x f32>
 * ------------------------------------------------------------------------- */

static void _op_arith_cvt_scalef32_pk_f32_fp8(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* scale = op->operands[1];
    const char* res = rocke_ll_result_name(op);
    const char* packed_i32 = rocke_ll_derived(L, op, "p");
    const char* lo = rocke_ll_derived(L, op, "lo");
    const char* hi = rocke_ll_derived(L, op, "hi");
    const char* scale_op;
    rocke_ll_need(L, "amdgcn.cvt.scalef32.pk.f32.fp8");
    rocke_ll_emitf(L, "  %s = bitcast <4 x i8> %s to i32", packed_i32, rocke_ll_operand(L, v));
    scale_op = rocke_ll_operand(L, scale);
    rocke_ll_emitf(L,
                   "  %s = call <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.fp8("
                   "i32 %s, float %s, i1 false)",
                   lo,
                   packed_i32,
                   scale_op);
    scale_op = rocke_ll_operand(L, scale);
    rocke_ll_emitf(L,
                   "  %s = call <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.fp8("
                   "i32 %s, float %s, i1 true)",
                   hi,
                   packed_i32,
                   scale_op);
    rocke_ll_emitf(L,
                   "  %s = shufflevector <2 x float> %s, <2 x float> %s, "
                   "<4 x i32> <i32 0, i32 1, i32 2, i32 3>",
                   res,
                   lo,
                   hi);
}

static void _op_arith_cvt_scalef32_pk_f32_bf8(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* scale = op->operands[1];
    const char* res = rocke_ll_result_name(op);
    const char* packed_i32 = rocke_ll_derived(L, op, "p");
    const char* lo = rocke_ll_derived(L, op, "lo");
    const char* hi = rocke_ll_derived(L, op, "hi");
    const char* scale_op;
    rocke_ll_need(L, "amdgcn.cvt.scalef32.pk.f32.bf8");
    rocke_ll_emitf(L, "  %s = bitcast <4 x i8> %s to i32", packed_i32, rocke_ll_operand(L, v));
    scale_op = rocke_ll_operand(L, scale);
    rocke_ll_emitf(L,
                   "  %s = call <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.bf8("
                   "i32 %s, float %s, i1 false)",
                   lo,
                   packed_i32,
                   scale_op);
    scale_op = rocke_ll_operand(L, scale);
    rocke_ll_emitf(L,
                   "  %s = call <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.bf8("
                   "i32 %s, float %s, i1 true)",
                   hi,
                   packed_i32,
                   scale_op);
    rocke_ll_emitf(L,
                   "  %s = shufflevector <2 x float> %s, <2 x float> %s, "
                   "<4 x i32> <i32 0, i32 1, i32 2, i32 3>",
                   res,
                   lo,
                   hi);
}

/* ---------------------------------------------------------------------------
 * f32 -> fp8 / bf8 (single element)
 * ------------------------------------------------------------------------- */

static void _op_arith_cvt_f32_to_fp8(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const char* res = rocke_ll_result_name(op);
    const char* packed = rocke_ll_derived(L, op, "p");
    rocke_ll_need(L, "amdgcn.cvt.pk.fp8.f32");
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.amdgcn.cvt.pk.fp8.f32("
                   "float %s, float 0.000000e+00, i32 0, i1 false)",
                   packed,
                   rocke_ll_operand(L, v));
    rocke_ll_emitf(L, "  %s = trunc i32 %s to i8", res, packed);
}

static void _op_arith_cvt_f32_to_bf8(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const char* res = rocke_ll_result_name(op);
    const char* packed = rocke_ll_derived(L, op, "p");
    rocke_ll_need(L, "amdgcn.cvt.pk.bf8.f32");
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.amdgcn.cvt.pk.bf8.f32("
                   "float %s, float 0.000000e+00, i32 0, i1 false)",
                   packed,
                   rocke_ll_operand(L, v));
    rocke_ll_emitf(L, "  %s = trunc i32 %s to i8", res, packed);
}

/* ---------------------------------------------------------------------------
 * packed <4 x f32> -> <4 x fp8/bf8>  (2x v_cvt_pk_*_f32)
 * ------------------------------------------------------------------------- */

static void _op_arith_cvt_pk_fp8_f32x4(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const char* res = rocke_ll_result_name(op);
    const char* e[4];
    const char* lo = rocke_ll_derived(L, op, "lo");
    const char* packed = rocke_ll_derived(L, op, "p");
    int i;
    rocke_ll_need(L, "amdgcn.cvt.pk.fp8.f32");
    for(i = 0; i < 4; i++)
    {
        char suffix[8];
        suffix[0] = 'e';
        suffix[1] = (char)('0' + i);
        suffix[2] = '\0';
        e[i] = rocke_ll_derived(L, op, suffix);
        rocke_ll_emitf(
            L, "  %s = extractelement <4 x float> %s, i32 %d", e[i], rocke_ll_operand(L, v), i);
    }
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.amdgcn.cvt.pk.fp8.f32("
                   "float %s, float %s, i32 0, i1 false)",
                   lo,
                   e[0],
                   e[1]);
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.amdgcn.cvt.pk.fp8.f32("
                   "float %s, float %s, i32 %s, i1 true)",
                   packed,
                   e[2],
                   e[3],
                   lo);
    rocke_ll_emitf(L, "  %s = bitcast i32 %s to <4 x i8>", res, packed);
}

static void _op_arith_cvt_pk_bf8_f32x4(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const char* res = rocke_ll_result_name(op);
    const char* e[4];
    const char* lo = rocke_ll_derived(L, op, "lo");
    const char* packed = rocke_ll_derived(L, op, "p");
    int i;
    rocke_ll_need(L, "amdgcn.cvt.pk.bf8.f32");
    for(i = 0; i < 4; i++)
    {
        char suffix[8];
        suffix[0] = 'e';
        suffix[1] = (char)('0' + i);
        suffix[2] = '\0';
        e[i] = rocke_ll_derived(L, op, suffix);
        rocke_ll_emitf(
            L, "  %s = extractelement <4 x float> %s, i32 %d", e[i], rocke_ll_operand(L, v), i);
    }
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.amdgcn.cvt.pk.bf8.f32("
                   "float %s, float %s, i32 0, i1 false)",
                   lo,
                   e[0],
                   e[1]);
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.amdgcn.cvt.pk.bf8.f32("
                   "float %s, float %s, i32 %s, i1 true)",
                   packed,
                   e[2],
                   e[3],
                   lo);
    rocke_ll_emitf(L, "  %s = bitcast i32 %s to <4 x i8>", res, packed);
}

/* ---------------------------------------------------------------------------
 * saturating f32 -> i8 (single element + packed x4)
 * ------------------------------------------------------------------------- */

static void _op_arith_cvt_f32_to_i8_sat(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const char* res = rocke_ll_result_name(op);
    const char* rounded = rocke_ll_derived(L, op, "r");
    const char* as_i32 = rocke_ll_derived(L, op, "i");
    const char* smax_v = rocke_ll_derived(L, op, "smax");
    const char* smin_v = rocke_ll_derived(L, op, "smin");
    rocke_ll_need(L, "rint.f32");
    rocke_ll_emitf(
        L, "  %s = call float @llvm.rint.f32(float %s)", rounded, rocke_ll_operand(L, v));
    rocke_ll_emitf(L, "  %s = fptosi float %s to i32", as_i32, rounded);
    rocke_ll_need(L, "smax.i32");
    rocke_ll_need(L, "smin.i32");
    rocke_ll_emitf(L, "  %s = call i32 @llvm.smax.i32(i32 -128, i32 %s)", smax_v, as_i32);
    rocke_ll_emitf(L, "  %s = call i32 @llvm.smin.i32(i32 127, i32 %s)", smin_v, smax_v);
    rocke_ll_emitf(L, "  %s = trunc i32 %s to i8", res, smin_v);
}

static void _op_arith_cvt_pk_i8_f32x4(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const char* res = rocke_ll_result_name(op);
    const char* clamped[4];
    const char* lo_hilo = rocke_ll_derived(L, op, "lh");
    const char* hi_hilo = rocke_ll_derived(L, op, "hh");
    const char* packed = rocke_ll_derived(L, op, "p");
    int i;
    rocke_ll_need(L, "rint.f32");
    rocke_ll_need(L, "smax.i32");
    rocke_ll_need(L, "smin.i32");
    rocke_ll_need(L, "amdgcn.perm");
    for(i = 0; i < 4; i++)
    {
        char sfx[8];
        const char *e, *r, *ai, *mx, *mn;
        sfx[0] = 'e';
        sfx[1] = (char)('0' + i);
        sfx[2] = '\0';
        e = rocke_ll_derived(L, op, sfx);
        sfx[0] = 'r';
        r = rocke_ll_derived(L, op, sfx);
        sfx[0] = 'i';
        ai = rocke_ll_derived(L, op, sfx);
        sfx[0] = 'm';
        sfx[1] = 'x';
        sfx[2] = (char)('0' + i);
        sfx[3] = '\0';
        mx = rocke_ll_derived(L, op, sfx);
        sfx[0] = 'm';
        sfx[1] = 'n';
        sfx[2] = (char)('0' + i);
        sfx[3] = '\0';
        mn = rocke_ll_derived(L, op, sfx);
        rocke_ll_emitf(
            L, "  %s = extractelement <4 x float> %s, i32 %d", e, rocke_ll_operand(L, v), i);
        rocke_ll_emitf(L, "  %s = call float @llvm.rint.f32(float %s)", r, e);
        rocke_ll_emitf(L, "  %s = fptosi float %s to i32", ai, r);
        rocke_ll_emitf(L, "  %s = call i32 @llvm.smax.i32(i32 -128, i32 %s)", mx, ai);
        rocke_ll_emitf(L, "  %s = call i32 @llvm.smin.i32(i32 127, i32 %s)", mn, mx);
        clamped[i] = mn;
    }
    /* Pack four clamped i32 into a single i32 via v_perm_b32 byte-selects.
     * Selectors match the Python verbatim: 1284 (0x00000504) twice then
     * 84148480 (0x05040100). */
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.amdgcn.perm(i32 %s, i32 %s, i32 1284)",
                   lo_hilo,
                   clamped[1],
                   clamped[0]);
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.amdgcn.perm(i32 %s, i32 %s, i32 1284)",
                   hi_hilo,
                   clamped[3],
                   clamped[2]);
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.amdgcn.perm(i32 %s, i32 %s, i32 84148480)",
                   packed,
                   hi_hilo,
                   lo_hilo);
    rocke_ll_emitf(L, "  %s = bitcast i32 %s to <4 x i8>", res, packed);
}

/* ---------------------------------------------------------------------------
 * Registration hook (called once by bucket 0's init)
 * ------------------------------------------------------------------------- */

void rocke_ll_register_convert(void)
{
    rocke_ll_set_handler(ROCKE_OP_ARITH_ZEXT, _op_arith_zext);
    rocke_ll_set_handler(ROCKE_OP_ARITH_SEXT, _op_arith_sext);
    rocke_ll_set_handler(ROCKE_OP_ARITH_TRUNC, _op_arith_trunc);
    rocke_ll_set_handler(ROCKE_OP_ARITH_BITCAST, _op_arith_bitcast);

    rocke_ll_set_handler(ROCKE_OP_ARITH_TRUNC_F32_TO_F16, _op_arith_trunc_f32_to_f16);
    rocke_ll_set_handler(ROCKE_OP_ARITH_CAST_TO_F32, _op_arith_cast_to_f32);
    rocke_ll_set_handler(ROCKE_OP_ARITH_CAST_F32_TO, _op_arith_cast_f32_to);
    rocke_ll_set_handler(ROCKE_OP_ARITH_SITOFP_F32, _op_arith_sitofp_f32);
    rocke_ll_set_handler(ROCKE_OP_ARITH_RINT_F32, _op_arith_rint_f32);

    rocke_ll_set_handler(ROCKE_OP_ARITH_CVT_FP8_TO_F32, _op_arith_cvt_fp8_to_f32);
    rocke_ll_set_handler(ROCKE_OP_ARITH_CVT_BF8_TO_F32, _op_arith_cvt_bf8_to_f32);
    rocke_ll_set_handler(ROCKE_OP_ARITH_CVT_PK_F32_FP8X4, _op_arith_cvt_pk_f32_fp8x4);
    rocke_ll_set_handler(ROCKE_OP_ARITH_CVT_PK_F32_BF8X4, _op_arith_cvt_pk_f32_bf8x4);

    rocke_ll_set_handler(ROCKE_OP_ARITH_CVT_SCALEF32_PK_F32_FP8, _op_arith_cvt_scalef32_pk_f32_fp8);
    rocke_ll_set_handler(ROCKE_OP_ARITH_CVT_SCALEF32_PK_F32_BF8, _op_arith_cvt_scalef32_pk_f32_bf8);

    rocke_ll_set_handler(ROCKE_OP_ARITH_CVT_F32_TO_FP8, _op_arith_cvt_f32_to_fp8);
    rocke_ll_set_handler(ROCKE_OP_ARITH_CVT_F32_TO_BF8, _op_arith_cvt_f32_to_bf8);
    rocke_ll_set_handler(ROCKE_OP_ARITH_CVT_PK_FP8_F32X4, _op_arith_cvt_pk_fp8_f32x4);
    rocke_ll_set_handler(ROCKE_OP_ARITH_CVT_PK_BF8_F32X4, _op_arith_cvt_pk_bf8_f32x4);
    rocke_ll_set_handler(ROCKE_OP_ARITH_CVT_PK_I8_F32X4, _op_arith_cvt_pk_i8_f32x4);
    rocke_ll_set_handler(ROCKE_OP_ARITH_CVT_F32_TO_I8_SAT, _op_arith_cvt_f32_to_i8_sat);
}

} /* namespace ckc */
