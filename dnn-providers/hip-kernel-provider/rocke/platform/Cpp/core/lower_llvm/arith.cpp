// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_llvm_lower_llvm_arith.c -- BUCKET 1 of the C99 port of
 * rocke.core.lower_llvm.
 *
 * Faithful translation of the scalar arith / math.* / gpu.* per-op handlers:
 *   arith.constant, arith.constant_vec,
 *   arith.add/sub/mul/div/mod (integer),
 *   arith.fadd/fsub/fmul/fdiv/fneg/fabs/fma/fmax3/fmin3 (float),
 *   arith.cmp/fcmp, arith.fmax/fmin,
 *   arith.select/and/or/not/smax/smin/xor/shl/lshr/umul_hi_i32,
 *   math.exp2/log2/rcp/rcp_fast/sqrt/rsqrt/tanh,
 *   gpu.thread_id / gpu.block_id.
 *
 * Every shared helper (rocke_ll_emit, rocke_ll_operand, rocke_ll_llvm_type,
 * rocke_ll_need, rocke_ll_fresh, rocke_ll_fp32_hex, rocke_ll_binop, rocke_ll_fail, ...)
 * lives in BUCKET 0; this file only calls them through the internal header.
 */
#include "rocke/lower_llvm_internal.h"

#include <stdio.h>
#include <string.h>

namespace ckc
{

/* ------------------------------------------------------------------ helpers */

/* Python: op.result -- exactly one result; the handlers below always have one
 * (the producing builder guaranteed it). Returns op->results[0]. */
static const rocke_value_t* ll_result(const rocke_op_t* op)
{
    return (op && op->num_results > 0) ? op->results[0] : NULL;
}

/* Map an FP scalar type name to its LLVM textual type, mirroring the Python
 * `{"f32": "float", "f16": "half", "bf16": "bfloat"}.get(ty_name)` dicts.
 * Returns NULL for an unsupported type. */
static const char* ll_fp_llvm_ty(const char* ty_name)
{
    if(!ty_name)
    {
        return NULL;
    }
    if(strcmp(ty_name, "f32") == 0)
    {
        return "float";
    }
    if(strcmp(ty_name, "f16") == 0)
    {
        return "half";
    }
    if(strcmp(ty_name, "bf16") == 0)
    {
        return "bfloat";
    }
    return NULL;
}

/* ------------------------------------------------------------------ arith */

/* Python _op_arith_constant: constants are emitted lazily at point of use. */
static void _op_arith_constant(rocke_lower_t* L, const rocke_op_t* op)
{
    (void)L;
    (void)op;
    /* No-op: arith.constant literals are inlined by rocke_ll_operand. */
}

/* Python _op_arith_constant_vec. */
static void _op_arith_constant_vec(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    if(res->type && res->type->kind == ROCKE_TYPE_VECTOR)
    {
        double fill = 0.0;
        rocke_attr_get_float(&op->attrs, "fill", &fill);
        if(fill == 0.0)
        {
            /* zeroinitializer is the canonical form (works for any vector). */
            const char* ty = rocke_ll_llvm_type(L, res->type);
            rocke_ll_emitf(L,
                           "  %s = select i1 true, %s zeroinitializer, "
                           "%s zeroinitializer",
                           res->name,
                           ty,
                           ty);
            return;
        }
    }
    rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "arith.constant_vec");
}

/* The same-type binary handlers all defer to the shared rocke_ll_binop helper
 * (Python self._binop(op, llvm_op)). */
static void _op_arith_add(rocke_lower_t* L, const rocke_op_t* op)
{
    rocke_ll_binop(L, op, "add nsw");
}
static void _op_arith_sub(rocke_lower_t* L, const rocke_op_t* op)
{
    rocke_ll_binop(L, op, "sub nsw");
}
static void _op_arith_mul(rocke_lower_t* L, const rocke_op_t* op)
{
    rocke_ll_binop(L, op, "mul nsw");
}
static void _op_arith_div(rocke_lower_t* L, const rocke_op_t* op)
{
    rocke_ll_binop(L, op, "sdiv");
}
static void _op_arith_mod(rocke_lower_t* L, const rocke_op_t* op)
{
    rocke_ll_binop(L, op, "srem");
}
static void _op_arith_fadd(rocke_lower_t* L, const rocke_op_t* op)
{
    rocke_ll_binop(L, op, "fadd");
}
static void _op_arith_fsub(rocke_lower_t* L, const rocke_op_t* op)
{
    rocke_ll_binop(L, op, "fsub");
}
static void _op_arith_fmul(rocke_lower_t* L, const rocke_op_t* op)
{
    rocke_ll_binop(L, op, "fmul");
}
static void _op_arith_fdiv(rocke_lower_t* L, const rocke_op_t* op)
{
    rocke_ll_binop(L, op, "fdiv");
}

/* Python _op_arith_fneg. */
static void _op_arith_fneg(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* v = op->operands[0];
    rocke_ll_emitf(
        L, "  %s = fneg %s %s", res->name, rocke_ll_llvm_type(L, v->type), rocke_ll_operand(L, v));
}

/* Python _op_arith_fabs -> llvm.fabs.<ty>. */
static void _op_arith_fabs(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* v = op->operands[0];
    const char* ty_name = v->type ? v->type->name : NULL;
    const char* llvm_ty = ll_fp_llvm_ty(ty_name);
    if(llvm_ty == NULL)
    {
        rocke_ll_fail(
            L, ROCKE_ERR_NOTIMPL, "fabs: unsupported FP type %s", ty_name ? ty_name : "(null)");
    }
    char key[32];
    snprintf(key, sizeof key, "fabs.%s", ty_name);
    rocke_ll_need(L, key);
    rocke_ll_emitf(L,
                   "  %s = call %s @llvm.fabs.%s(%s %s)",
                   res->name,
                   llvm_ty,
                   ty_name,
                   llvm_ty,
                   rocke_ll_operand(L, v));
}

/* Python _op_arith_fma -> llvm.fmuladd.<ty>. */
static void _op_arith_fma(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* c = op->operands[2];
    const char* ty_name = a->type ? a->type->name : NULL;
    const char* llvm_ty = ll_fp_llvm_ty(ty_name);
    if(llvm_ty == NULL)
    {
        rocke_ll_fail(
            L, ROCKE_ERR_NOTIMPL, "fma: unsupported FP type %s", ty_name ? ty_name : "(null)");
    }
    char key[32];
    snprintf(key, sizeof key, "fmuladd.%s", ty_name);
    rocke_ll_need(L, key);
    rocke_ll_emitf(L,
                   "  %s = call %s @llvm.fmuladd.%s(%s %s, %s %s, %s %s)",
                   res->name,
                   llvm_ty,
                   ty_name,
                   llvm_ty,
                   rocke_ll_operand(L, a),
                   llvm_ty,
                   rocke_ll_operand(L, b),
                   llvm_ty,
                   rocke_ll_operand(L, c));
}

/* Shared body for fmax3 / fmin3: two back-to-back maxnum/minnum calls.
 * `op_kind` is "max" or "min" (selects maxnum/minnum + the bc fresh prefix). */
static void ll_fminmax3(rocke_lower_t* L,
                        const rocke_op_t* op,
                        const char* intrin,
                        const char* fresh_prefix)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* c = op->operands[2];
    const char* ty_name = a->type ? a->type->name : NULL;
    const char* llvm_ty = ll_fp_llvm_ty(ty_name);
    if(llvm_ty == NULL)
    {
        rocke_ll_fail(L,
                      ROCKE_ERR_NOTIMPL,
                      "%s3: unsupported FP type %s",
                      intrin,
                      ty_name ? ty_name : "(null)");
    }
    char key[32];
    snprintf(key, sizeof key, "%s.%s", intrin, ty_name);
    rocke_ll_need(L, key);
    const char* inner = rocke_ll_fresh(L, fresh_prefix);
    rocke_ll_emitf(L,
                   "  %s = call %s @llvm.%s.%s(%s %s, %s %s)",
                   inner,
                   llvm_ty,
                   intrin,
                   ty_name,
                   llvm_ty,
                   rocke_ll_operand(L, b),
                   llvm_ty,
                   rocke_ll_operand(L, c));
    rocke_ll_emitf(L,
                   "  %s = call %s @llvm.%s.%s(%s %s, %s %s)",
                   res->name,
                   llvm_ty,
                   intrin,
                   ty_name,
                   llvm_ty,
                   rocke_ll_operand(L, a),
                   llvm_ty,
                   inner);
}

/* Python _op_arith_fmax3 -> maxnum(a, maxnum(b, c)). */
static void _op_arith_fmax3(rocke_lower_t* L, const rocke_op_t* op)
{
    ll_fminmax3(L, op, "maxnum", "fmax3.bc");
}
/* Python _op_arith_fmin3 -> minnum(a, minnum(b, c)). */
static void _op_arith_fmin3(rocke_lower_t* L, const rocke_op_t* op)
{
    ll_fminmax3(L, op, "minnum", "fmin3.bc");
}

/* Python _op_arith_cmp. */
static void _op_arith_cmp(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const char* pred = rocke_attr_get_str(&op->attrs, "pred");
    if(pred == NULL)
    {
        pred = "lt";
    }
    const char* llvm_pred;
    if(strcmp(pred, "lt") == 0)
    {
        llvm_pred = "slt";
    }
    else if(strcmp(pred, "le") == 0)
    {
        llvm_pred = "sle";
    }
    else if(strcmp(pred, "gt") == 0)
    {
        llvm_pred = "sgt";
    }
    else if(strcmp(pred, "ge") == 0)
    {
        llvm_pred = "sge";
    }
    else if(strcmp(pred, "eq") == 0)
    {
        llvm_pred = "eq";
    }
    else if(strcmp(pred, "ne") == 0)
    {
        llvm_pred = "ne";
    }
    else
    {
        rocke_ll_fail(L, ROCKE_ERR_KEY, "arith.cmp: unknown pred %s", pred);
    }
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    rocke_ll_emitf(L,
                   "  %s = icmp %s %s %s, %s",
                   res->name,
                   llvm_pred,
                   rocke_ll_llvm_type(L, a->type),
                   rocke_ll_operand(L, a),
                   rocke_ll_operand(L, b));
}

/* Python _op_arith_fcmp (pred passed through verbatim). */
static void _op_arith_fcmp(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const char* pred = rocke_attr_get_str(&op->attrs, "pred");
    if(pred == NULL)
    {
        pred = "olt";
    }
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    rocke_ll_emitf(L,
                   "  %s = fcmp %s %s %s, %s",
                   res->name,
                   pred,
                   rocke_ll_llvm_type(L, a->type),
                   rocke_ll_operand(L, a),
                   rocke_ll_operand(L, b));
}

/* Shared body for fmax / fmin (single maxnum/minnum call). */
static void ll_fminmax(rocke_lower_t* L, const rocke_op_t* op, const char* intrin)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const char* ty_name = a->type ? a->type->name : NULL;
    const char* llvm_ty = ll_fp_llvm_ty(ty_name);
    if(llvm_ty == NULL)
    {
        rocke_ll_fail(L,
                      ROCKE_ERR_NOTIMPL,
                      "%s: unsupported FP type %s",
                      intrin,
                      ty_name ? ty_name : "(null)");
    }
    char key[32];
    snprintf(key, sizeof key, "%s.%s", intrin, ty_name);
    rocke_ll_need(L, key);
    rocke_ll_emitf(L,
                   "  %s = call %s @llvm.%s.%s(%s %s, %s %s)",
                   res->name,
                   llvm_ty,
                   intrin,
                   ty_name,
                   llvm_ty,
                   rocke_ll_operand(L, a),
                   llvm_ty,
                   rocke_ll_operand(L, b));
}

/* Python _op_arith_fmax. */
static void _op_arith_fmax(rocke_lower_t* L, const rocke_op_t* op)
{
    ll_fminmax(L, op, "maxnum");
}
/* Python _op_arith_fmin. */
static void _op_arith_fmin(rocke_lower_t* L, const rocke_op_t* op)
{
    ll_fminmax(L, op, "minnum");
}

/* Python _op_arith_select. */
static void _op_arith_select(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* cond = op->operands[0];
    const rocke_value_t* lhs = op->operands[1];
    const rocke_value_t* rhs = op->operands[2];
    rocke_ll_emitf(L,
                   "  %s = select i1 %s, %s %s, %s %s",
                   res->name,
                   rocke_ll_operand(L, cond),
                   rocke_ll_llvm_type(L, lhs->type),
                   rocke_ll_operand(L, lhs),
                   rocke_ll_llvm_type(L, rhs->type),
                   rocke_ll_operand(L, rhs));
}

/* Python _op_arith_and. */
static void _op_arith_and(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    rocke_ll_emitf(L,
                   "  %s = and %s %s, %s",
                   res->name,
                   rocke_ll_llvm_type(L, a->type),
                   rocke_ll_operand(L, a),
                   rocke_ll_operand(L, b));
}

/* Python _op_arith_or. */
static void _op_arith_or(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    rocke_ll_emitf(L,
                   "  %s = or %s %s, %s",
                   res->name,
                   rocke_ll_llvm_type(L, a->type),
                   rocke_ll_operand(L, a),
                   rocke_ll_operand(L, b));
}

/* Python _op_arith_not (xor against all-ones / true). */
static void _op_arith_not(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* a = op->operands[0];
    const char* ty = rocke_ll_llvm_type(L, a->type);
    const char* mask
        = (a->type && a->type->name && strcmp(a->type->name, "i1") == 0) ? "true" : "-1";
    rocke_ll_emitf(L, "  %s = xor %s %s, %s", res->name, ty, rocke_ll_operand(L, a), mask);
}

/* Python _op_arith_smax -> llvm.smax.i32. */
static void _op_arith_smax(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    rocke_ll_need(L, "smax.i32");
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.smax.i32(i32 %s, i32 %s)",
                   res->name,
                   rocke_ll_operand(L, a),
                   rocke_ll_operand(L, b));
}

/* Python _op_arith_smin -> llvm.smin.i32. */
static void _op_arith_smin(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    rocke_ll_need(L, "smin.i32");
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.smin.i32(i32 %s, i32 %s)",
                   res->name,
                   rocke_ll_operand(L, a),
                   rocke_ll_operand(L, b));
}

/* Python _op_arith_xor (result-typed). */
static void _op_arith_xor(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    rocke_ll_emitf(L,
                   "  %s = xor %s %s, %s",
                   res->name,
                   rocke_ll_llvm_type(L, res->type),
                   rocke_ll_operand(L, a),
                   rocke_ll_operand(L, b));
}

/* Python _op_arith_shl (result-typed). */
static void _op_arith_shl(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    rocke_ll_emitf(L,
                   "  %s = shl %s %s, %s",
                   res->name,
                   rocke_ll_llvm_type(L, res->type),
                   rocke_ll_operand(L, a),
                   rocke_ll_operand(L, b));
}

/* Python _op_arith_lshr (result-typed). */
static void _op_arith_lshr(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    rocke_ll_emitf(L,
                   "  %s = lshr %s %s, %s",
                   res->name,
                   rocke_ll_llvm_type(L, res->type),
                   rocke_ll_operand(L, a),
                   rocke_ll_operand(L, b));
}

/* Python _op_arith_umul_hi_i32: zext / mul / lshr / trunc -> v_mul_hi_u32. */
static void _op_arith_umul_hi_i32(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const char* a64 = rocke_ll_fresh(L, "za64");
    const char* b64 = rocke_ll_fresh(L, "zb64");
    const char* prod = rocke_ll_fresh(L, "prod64");
    const char* hi64 = rocke_ll_fresh(L, "hi64");
    rocke_ll_emitf(L, "  %s = zext i32 %s to i64", a64, rocke_ll_operand(L, a));
    rocke_ll_emitf(L, "  %s = zext i32 %s to i64", b64, rocke_ll_operand(L, b));
    rocke_ll_emitf(L, "  %s = mul i64 %s, %s", prod, a64, b64);
    rocke_ll_emitf(L, "  %s = lshr i64 %s, 32", hi64, prod);
    rocke_ll_emitf(L, "  %s = trunc i64 %s to i32", res->name, hi64);
}

/* ------------------------------------------------------------------ math.* */

/* Shared body for the f32-only unary intrinsic math ops. `intrin` is the LLVM
 * intrinsic suffix after "@llvm." (e.g. "exp2.f32", "amdgcn.rcp.f32"); `key`
 * is the _need() key. */
static void ll_math_f32_unary(rocke_lower_t* L,
                              const rocke_op_t* op,
                              const char* op_label,
                              const char* key,
                              const char* intrin)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* v = op->operands[0];
    if(!(v->type && v->type->name && strcmp(v->type->name, "f32") == 0))
    {
        rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "math.%s currently supports f32", op_label);
    }
    rocke_ll_need(L, key);
    rocke_ll_emitf(
        L, "  %s = call float @llvm.%s(float %s)", res->name, intrin, rocke_ll_operand(L, v));
}

/* Python _op_math_exp2. */
static void _op_math_exp2(rocke_lower_t* L, const rocke_op_t* op)
{
    ll_math_f32_unary(L, op, "exp2", "exp2.f32", "exp2.f32");
}
/* Python _op_math_log2. */
static void _op_math_log2(rocke_lower_t* L, const rocke_op_t* op)
{
    ll_math_f32_unary(L, op, "log2", "log2.f32", "log2.f32");
}

/* Python _op_math_rcp: fdiv 1.0, x (NOT an intrinsic; type-generic). */
static void _op_math_rcp(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const rocke_value_t* v = op->operands[0];
    const char* one;
    if(v->type && v->type->name && strcmp(v->type->name, "f32") == 0)
    {
        one = rocke_ll_fp32_hex(L, 1.0);
    }
    else
    {
        one = "1.000000e+00";
    }
    rocke_ll_emitf(L,
                   "  %s = fdiv %s %s, %s",
                   res->name,
                   rocke_ll_llvm_type(L, v->type),
                   one,
                   rocke_ll_operand(L, v));
}

/* Python _op_math_rcp_fast -> llvm.amdgcn.rcp.f32. */
static void _op_math_rcp_fast(rocke_lower_t* L, const rocke_op_t* op)
{
    ll_math_f32_unary(L, op, "rcp_fast", "rcp.f32", "amdgcn.rcp.f32");
}
/* Python _op_math_sqrt -> llvm.sqrt.f32. */
static void _op_math_sqrt(rocke_lower_t* L, const rocke_op_t* op)
{
    ll_math_f32_unary(L, op, "sqrt", "sqrt.f32", "sqrt.f32");
}
/* Python _op_math_rsqrt -> llvm.amdgcn.rsq.f32. */
static void _op_math_rsqrt(rocke_lower_t* L, const rocke_op_t* op)
{
    ll_math_f32_unary(L, op, "rsqrt", "rsqrt.f32", "amdgcn.rsq.f32");
}
/* Python _op_math_tanh -> llvm.tanh.f32. */
static void _op_math_tanh(rocke_lower_t* L, const rocke_op_t* op)
{
    ll_math_f32_unary(L, op, "tanh", "tanh.f32", "tanh.f32");
}

/* ------------------------------------------------------------------ gpu.* */

/* Python _op_gpu_thread_id -> llvm.amdgcn.workitem.id.<axis>. */
static void _op_gpu_thread_id(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const char* axis = rocke_attr_get_str(&op->attrs, "axis");
    if(axis == NULL)
    {
        axis = "x";
    }
    char key[32];
    snprintf(key, sizeof key, "workitem.%s", axis);
    rocke_ll_need(L, key);
    rocke_ll_emitf(L, "  %s = call i32 @llvm.amdgcn.workitem.id.%s()", res->name, axis);
}

/* Python _op_gpu_block_id -> llvm.amdgcn.workgroup.id.<axis>. */
static void _op_gpu_block_id(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* res = ll_result(op);
    if(!rocke_ll_live(L) || !res)
    {
        return;
    }
    const char* axis = rocke_attr_get_str(&op->attrs, "axis");
    if(axis == NULL)
    {
        axis = "x";
    }
    char key[32];
    snprintf(key, sizeof key, "workgroup.%s", axis);
    rocke_ll_need(L, key);
    rocke_ll_emitf(L, "  %s = call i32 @llvm.amdgcn.workgroup.id.%s()", res->name, axis);
}

/* ------------------------------------------------------------- registration */

void rocke_ll_register_arith(void)
{
    rocke_ll_set_handler(ROCKE_OP_ARITH_CONSTANT, _op_arith_constant);
    rocke_ll_set_handler(ROCKE_OP_ARITH_CONSTANT_VEC, _op_arith_constant_vec);
    rocke_ll_set_handler(ROCKE_OP_ARITH_ADD, _op_arith_add);
    rocke_ll_set_handler(ROCKE_OP_ARITH_SUB, _op_arith_sub);
    rocke_ll_set_handler(ROCKE_OP_ARITH_MUL, _op_arith_mul);
    rocke_ll_set_handler(ROCKE_OP_ARITH_DIV, _op_arith_div);
    rocke_ll_set_handler(ROCKE_OP_ARITH_MOD, _op_arith_mod);
    rocke_ll_set_handler(ROCKE_OP_ARITH_FADD, _op_arith_fadd);
    rocke_ll_set_handler(ROCKE_OP_ARITH_FSUB, _op_arith_fsub);
    rocke_ll_set_handler(ROCKE_OP_ARITH_FMUL, _op_arith_fmul);
    rocke_ll_set_handler(ROCKE_OP_ARITH_FDIV, _op_arith_fdiv);
    rocke_ll_set_handler(ROCKE_OP_ARITH_FNEG, _op_arith_fneg);
    rocke_ll_set_handler(ROCKE_OP_ARITH_FABS, _op_arith_fabs);
    rocke_ll_set_handler(ROCKE_OP_ARITH_FMA, _op_arith_fma);
    rocke_ll_set_handler(ROCKE_OP_ARITH_FMAX3, _op_arith_fmax3);
    rocke_ll_set_handler(ROCKE_OP_ARITH_FMIN3, _op_arith_fmin3);
    rocke_ll_set_handler(ROCKE_OP_ARITH_CMP, _op_arith_cmp);
    rocke_ll_set_handler(ROCKE_OP_ARITH_FCMP, _op_arith_fcmp);
    rocke_ll_set_handler(ROCKE_OP_ARITH_FMAX, _op_arith_fmax);
    rocke_ll_set_handler(ROCKE_OP_ARITH_FMIN, _op_arith_fmin);
    rocke_ll_set_handler(ROCKE_OP_ARITH_SELECT, _op_arith_select);
    rocke_ll_set_handler(ROCKE_OP_ARITH_AND, _op_arith_and);
    rocke_ll_set_handler(ROCKE_OP_ARITH_OR, _op_arith_or);
    rocke_ll_set_handler(ROCKE_OP_ARITH_NOT, _op_arith_not);
    rocke_ll_set_handler(ROCKE_OP_ARITH_SMAX, _op_arith_smax);
    rocke_ll_set_handler(ROCKE_OP_ARITH_SMIN, _op_arith_smin);
    rocke_ll_set_handler(ROCKE_OP_ARITH_XOR, _op_arith_xor);
    rocke_ll_set_handler(ROCKE_OP_ARITH_SHL, _op_arith_shl);
    rocke_ll_set_handler(ROCKE_OP_ARITH_LSHR, _op_arith_lshr);
    rocke_ll_set_handler(ROCKE_OP_ARITH_UMUL_HI_I32, _op_arith_umul_hi_i32);

    rocke_ll_set_handler(ROCKE_OP_MATH_EXP2, _op_math_exp2);
    rocke_ll_set_handler(ROCKE_OP_MATH_LOG2, _op_math_log2);
    rocke_ll_set_handler(ROCKE_OP_MATH_RCP, _op_math_rcp);
    rocke_ll_set_handler(ROCKE_OP_MATH_RCP_FAST, _op_math_rcp_fast);
    rocke_ll_set_handler(ROCKE_OP_MATH_SQRT, _op_math_sqrt);
    rocke_ll_set_handler(ROCKE_OP_MATH_RSQRT, _op_math_rsqrt);
    rocke_ll_set_handler(ROCKE_OP_MATH_TANH, _op_math_tanh);

    rocke_ll_set_handler(ROCKE_OP_GPU_THREAD_ID, _op_gpu_thread_id);
    rocke_ll_set_handler(ROCKE_OP_GPU_BLOCK_ID, _op_gpu_block_id);
}

} /* namespace ckc */
