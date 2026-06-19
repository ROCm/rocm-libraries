/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * lower_llvm_lower_llvm_arith.c -- BUCKET 1 of the C99 port of
 * ck_dsl.core.lower_llvm.
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
 * Every shared helper (ckc_ll_emit, ckc_ll_operand, ckc_ll_llvm_type,
 * ckc_ll_need, ckc_ll_fresh, ckc_ll_fp32_hex, ckc_ll_binop, ckc_ll_fail, ...)
 * lives in BUCKET 0; this file only calls them through the internal header.
 */
#include "ckc/lower_llvm_internal.h"

#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ helpers */

/* Python: op.result -- exactly one result; the handlers below always have one
 * (the producing builder guaranteed it). Returns op->results[0]. */
static const ckc_value_t *ll_result(const ckc_op_t *op) {
    return (op && op->num_results > 0) ? op->results[0] : NULL;
}

/* Map an FP scalar type name to its LLVM textual type, mirroring the Python
 * `{"f32": "float", "f16": "half", "bf16": "bfloat"}.get(ty_name)` dicts.
 * Returns NULL for an unsupported type. */
static const char *ll_fp_llvm_ty(const char *ty_name) {
    if (!ty_name) {
        return NULL;
    }
    if (strcmp(ty_name, "f32") == 0) {
        return "float";
    }
    if (strcmp(ty_name, "f16") == 0) {
        return "half";
    }
    if (strcmp(ty_name, "bf16") == 0) {
        return "bfloat";
    }
    return NULL;
}

/* ------------------------------------------------------------------ arith */

/* Python _op_arith_constant: constants are emitted lazily at point of use. */
static void _op_arith_constant(ckc_lower_t *L, const ckc_op_t *op) {
    (void)L;
    (void)op;
    /* No-op: arith.constant literals are inlined by ckc_ll_operand. */
}

/* Python _op_arith_constant_vec. */
static void _op_arith_constant_vec(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    if (res->type && res->type->kind == CKC_TYPE_VECTOR) {
        double fill = 0.0;
        ckc_attr_get_float(&op->attrs, "fill", &fill);
        if (fill == 0.0) {
            /* zeroinitializer is the canonical form (works for any vector). */
            const char *ty = ckc_ll_llvm_type(L, res->type);
            ckc_ll_emitf(L,
                         "  %s = select i1 true, %s zeroinitializer, "
                         "%s zeroinitializer",
                         res->name, ty, ty);
            return;
        }
    }
    ckc_ll_fail(L, CKC_ERR_NOTIMPL, "arith.constant_vec");
}

/* The same-type binary handlers all defer to the shared ckc_ll_binop helper
 * (Python self._binop(op, llvm_op)). */
static void _op_arith_add(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_binop(L, op, "add nsw");
}
static void _op_arith_sub(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_binop(L, op, "sub nsw");
}
static void _op_arith_mul(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_binop(L, op, "mul nsw");
}
static void _op_arith_div(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_binop(L, op, "sdiv");
}
static void _op_arith_mod(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_binop(L, op, "srem");
}
static void _op_arith_fadd(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_binop(L, op, "fadd");
}
static void _op_arith_fsub(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_binop(L, op, "fsub");
}
static void _op_arith_fmul(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_binop(L, op, "fmul");
}
static void _op_arith_fdiv(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_binop(L, op, "fdiv");
}

/* Python _op_arith_fneg. */
static void _op_arith_fneg(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *v = op->operands[0];
    ckc_ll_emitf(L, "  %s = fneg %s %s",
                 res->name, ckc_ll_llvm_type(L, v->type), ckc_ll_operand(L, v));
}

/* Python _op_arith_fabs -> llvm.fabs.<ty>. */
static void _op_arith_fabs(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *v = op->operands[0];
    const char *ty_name = v->type ? v->type->name : NULL;
    const char *llvm_ty = ll_fp_llvm_ty(ty_name);
    if (llvm_ty == NULL) {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "fabs: unsupported FP type %s",
                    ty_name ? ty_name : "(null)");
    }
    char key[32];
    snprintf(key, sizeof key, "fabs.%s", ty_name);
    ckc_ll_need(L, key);
    ckc_ll_emitf(L,
                 "  %s = call %s @llvm.fabs.%s(%s %s)",
                 res->name, llvm_ty, ty_name, llvm_ty, ckc_ll_operand(L, v));
}

/* Python _op_arith_fma -> llvm.fmuladd.<ty>. */
static void _op_arith_fma(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    const ckc_value_t *c = op->operands[2];
    const char *ty_name = a->type ? a->type->name : NULL;
    const char *llvm_ty = ll_fp_llvm_ty(ty_name);
    if (llvm_ty == NULL) {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "fma: unsupported FP type %s",
                    ty_name ? ty_name : "(null)");
    }
    char key[32];
    snprintf(key, sizeof key, "fmuladd.%s", ty_name);
    ckc_ll_need(L, key);
    ckc_ll_emitf(L,
                 "  %s = call %s @llvm.fmuladd.%s(%s %s, %s %s, %s %s)",
                 res->name, llvm_ty, ty_name,
                 llvm_ty, ckc_ll_operand(L, a),
                 llvm_ty, ckc_ll_operand(L, b),
                 llvm_ty, ckc_ll_operand(L, c));
}

/* Shared body for fmax3 / fmin3: two back-to-back maxnum/minnum calls.
 * `op_kind` is "max" or "min" (selects maxnum/minnum + the bc fresh prefix). */
static void ll_fminmax3(ckc_lower_t *L, const ckc_op_t *op,
                        const char *intrin, const char *fresh_prefix) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    const ckc_value_t *c = op->operands[2];
    const char *ty_name = a->type ? a->type->name : NULL;
    const char *llvm_ty = ll_fp_llvm_ty(ty_name);
    if (llvm_ty == NULL) {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "%s3: unsupported FP type %s",
                    intrin, ty_name ? ty_name : "(null)");
    }
    char key[32];
    snprintf(key, sizeof key, "%s.%s", intrin, ty_name);
    ckc_ll_need(L, key);
    const char *inner = ckc_ll_fresh(L, fresh_prefix);
    ckc_ll_emitf(L,
                 "  %s = call %s @llvm.%s.%s(%s %s, %s %s)",
                 inner, llvm_ty, intrin, ty_name,
                 llvm_ty, ckc_ll_operand(L, b),
                 llvm_ty, ckc_ll_operand(L, c));
    ckc_ll_emitf(L,
                 "  %s = call %s @llvm.%s.%s(%s %s, %s %s)",
                 res->name, llvm_ty, intrin, ty_name,
                 llvm_ty, ckc_ll_operand(L, a),
                 llvm_ty, inner);
}

/* Python _op_arith_fmax3 -> maxnum(a, maxnum(b, c)). */
static void _op_arith_fmax3(ckc_lower_t *L, const ckc_op_t *op) {
    ll_fminmax3(L, op, "maxnum", "fmax3.bc");
}
/* Python _op_arith_fmin3 -> minnum(a, minnum(b, c)). */
static void _op_arith_fmin3(ckc_lower_t *L, const ckc_op_t *op) {
    ll_fminmax3(L, op, "minnum", "fmin3.bc");
}

/* Python _op_arith_cmp. */
static void _op_arith_cmp(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const char *pred = ckc_attr_get_str(&op->attrs, "pred");
    if (pred == NULL) {
        pred = "lt";
    }
    const char *llvm_pred;
    if (strcmp(pred, "lt") == 0) {
        llvm_pred = "slt";
    } else if (strcmp(pred, "le") == 0) {
        llvm_pred = "sle";
    } else if (strcmp(pred, "gt") == 0) {
        llvm_pred = "sgt";
    } else if (strcmp(pred, "ge") == 0) {
        llvm_pred = "sge";
    } else if (strcmp(pred, "eq") == 0) {
        llvm_pred = "eq";
    } else if (strcmp(pred, "ne") == 0) {
        llvm_pred = "ne";
    } else {
        ckc_ll_fail(L, CKC_ERR_KEY, "arith.cmp: unknown pred %s", pred);
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    ckc_ll_emitf(L,
                 "  %s = icmp %s %s %s, %s",
                 res->name, llvm_pred, ckc_ll_llvm_type(L, a->type),
                 ckc_ll_operand(L, a), ckc_ll_operand(L, b));
}

/* Python _op_arith_fcmp (pred passed through verbatim). */
static void _op_arith_fcmp(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const char *pred = ckc_attr_get_str(&op->attrs, "pred");
    if (pred == NULL) {
        pred = "olt";
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    ckc_ll_emitf(L,
                 "  %s = fcmp %s %s %s, %s",
                 res->name, pred, ckc_ll_llvm_type(L, a->type),
                 ckc_ll_operand(L, a), ckc_ll_operand(L, b));
}

/* Shared body for fmax / fmin (single maxnum/minnum call). */
static void ll_fminmax(ckc_lower_t *L, const ckc_op_t *op, const char *intrin) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    const char *ty_name = a->type ? a->type->name : NULL;
    const char *llvm_ty = ll_fp_llvm_ty(ty_name);
    if (llvm_ty == NULL) {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "%s: unsupported FP type %s",
                    intrin, ty_name ? ty_name : "(null)");
    }
    char key[32];
    snprintf(key, sizeof key, "%s.%s", intrin, ty_name);
    ckc_ll_need(L, key);
    ckc_ll_emitf(L,
                 "  %s = call %s @llvm.%s.%s(%s %s, %s %s)",
                 res->name, llvm_ty, intrin, ty_name,
                 llvm_ty, ckc_ll_operand(L, a),
                 llvm_ty, ckc_ll_operand(L, b));
}

/* Python _op_arith_fmax. */
static void _op_arith_fmax(ckc_lower_t *L, const ckc_op_t *op) {
    ll_fminmax(L, op, "maxnum");
}
/* Python _op_arith_fmin. */
static void _op_arith_fmin(ckc_lower_t *L, const ckc_op_t *op) {
    ll_fminmax(L, op, "minnum");
}

/* Python _op_arith_select. */
static void _op_arith_select(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *cond = op->operands[0];
    const ckc_value_t *lhs = op->operands[1];
    const ckc_value_t *rhs = op->operands[2];
    ckc_ll_emitf(L,
                 "  %s = select i1 %s, %s %s, %s %s",
                 res->name, ckc_ll_operand(L, cond),
                 ckc_ll_llvm_type(L, lhs->type), ckc_ll_operand(L, lhs),
                 ckc_ll_llvm_type(L, rhs->type), ckc_ll_operand(L, rhs));
}

/* Python _op_arith_and. */
static void _op_arith_and(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    ckc_ll_emitf(L,
                 "  %s = and %s %s, %s",
                 res->name, ckc_ll_llvm_type(L, a->type),
                 ckc_ll_operand(L, a), ckc_ll_operand(L, b));
}

/* Python _op_arith_or. */
static void _op_arith_or(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    ckc_ll_emitf(L,
                 "  %s = or %s %s, %s",
                 res->name, ckc_ll_llvm_type(L, a->type),
                 ckc_ll_operand(L, a), ckc_ll_operand(L, b));
}

/* Python _op_arith_not (xor against all-ones / true). */
static void _op_arith_not(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *a = op->operands[0];
    const char *ty = ckc_ll_llvm_type(L, a->type);
    const char *mask =
        (a->type && a->type->name && strcmp(a->type->name, "i1") == 0)
            ? "true"
            : "-1";
    ckc_ll_emitf(L, "  %s = xor %s %s, %s",
                 res->name, ty, ckc_ll_operand(L, a), mask);
}

/* Python _op_arith_smax -> llvm.smax.i32. */
static void _op_arith_smax(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    ckc_ll_need(L, "smax.i32");
    ckc_ll_emitf(L,
                 "  %s = call i32 @llvm.smax.i32(i32 %s, i32 %s)",
                 res->name, ckc_ll_operand(L, a), ckc_ll_operand(L, b));
}

/* Python _op_arith_smin -> llvm.smin.i32. */
static void _op_arith_smin(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    ckc_ll_need(L, "smin.i32");
    ckc_ll_emitf(L,
                 "  %s = call i32 @llvm.smin.i32(i32 %s, i32 %s)",
                 res->name, ckc_ll_operand(L, a), ckc_ll_operand(L, b));
}

/* Python _op_arith_xor (result-typed). */
static void _op_arith_xor(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    ckc_ll_emitf(L,
                 "  %s = xor %s %s, %s",
                 res->name, ckc_ll_llvm_type(L, res->type),
                 ckc_ll_operand(L, a), ckc_ll_operand(L, b));
}

/* Python _op_arith_shl (result-typed). */
static void _op_arith_shl(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    ckc_ll_emitf(L,
                 "  %s = shl %s %s, %s",
                 res->name, ckc_ll_llvm_type(L, res->type),
                 ckc_ll_operand(L, a), ckc_ll_operand(L, b));
}

/* Python _op_arith_lshr (result-typed). */
static void _op_arith_lshr(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    ckc_ll_emitf(L,
                 "  %s = lshr %s %s, %s",
                 res->name, ckc_ll_llvm_type(L, res->type),
                 ckc_ll_operand(L, a), ckc_ll_operand(L, b));
}

/* Python _op_arith_umul_hi_i32: zext / mul / lshr / trunc -> v_mul_hi_u32. */
static void _op_arith_umul_hi_i32(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    const char *a64 = ckc_ll_fresh(L, "za64");
    const char *b64 = ckc_ll_fresh(L, "zb64");
    const char *prod = ckc_ll_fresh(L, "prod64");
    const char *hi64 = ckc_ll_fresh(L, "hi64");
    ckc_ll_emitf(L, "  %s = zext i32 %s to i64", a64, ckc_ll_operand(L, a));
    ckc_ll_emitf(L, "  %s = zext i32 %s to i64", b64, ckc_ll_operand(L, b));
    ckc_ll_emitf(L, "  %s = mul i64 %s, %s", prod, a64, b64);
    ckc_ll_emitf(L, "  %s = lshr i64 %s, 32", hi64, prod);
    ckc_ll_emitf(L, "  %s = trunc i64 %s to i32", res->name, hi64);
}

/* ------------------------------------------------------------------ math.* */

/* Shared body for the f32-only unary intrinsic math ops. `intrin` is the LLVM
 * intrinsic suffix after "@llvm." (e.g. "exp2.f32", "amdgcn.rcp.f32"); `key`
 * is the _need() key. */
static void ll_math_f32_unary(ckc_lower_t *L, const ckc_op_t *op,
                              const char *op_label, const char *key,
                              const char *intrin) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *v = op->operands[0];
    if (!(v->type && v->type->name && strcmp(v->type->name, "f32") == 0)) {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "math.%s currently supports f32",
                    op_label);
    }
    ckc_ll_need(L, key);
    ckc_ll_emitf(L,
                 "  %s = call float @llvm.%s(float %s)",
                 res->name, intrin, ckc_ll_operand(L, v));
}

/* Python _op_math_exp2. */
static void _op_math_exp2(ckc_lower_t *L, const ckc_op_t *op) {
    ll_math_f32_unary(L, op, "exp2", "exp2.f32", "exp2.f32");
}
/* Python _op_math_log2. */
static void _op_math_log2(ckc_lower_t *L, const ckc_op_t *op) {
    ll_math_f32_unary(L, op, "log2", "log2.f32", "log2.f32");
}

/* Python _op_math_rcp: fdiv 1.0, x (NOT an intrinsic; type-generic). */
static void _op_math_rcp(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const ckc_value_t *v = op->operands[0];
    const char *one;
    if (v->type && v->type->name && strcmp(v->type->name, "f32") == 0) {
        one = ckc_ll_fp32_hex(L, 1.0);
    } else {
        one = "1.000000e+00";
    }
    ckc_ll_emitf(L,
                 "  %s = fdiv %s %s, %s",
                 res->name, ckc_ll_llvm_type(L, v->type), one,
                 ckc_ll_operand(L, v));
}

/* Python _op_math_rcp_fast -> llvm.amdgcn.rcp.f32. */
static void _op_math_rcp_fast(ckc_lower_t *L, const ckc_op_t *op) {
    ll_math_f32_unary(L, op, "rcp_fast", "rcp.f32", "amdgcn.rcp.f32");
}
/* Python _op_math_sqrt -> llvm.sqrt.f32. */
static void _op_math_sqrt(ckc_lower_t *L, const ckc_op_t *op) {
    ll_math_f32_unary(L, op, "sqrt", "sqrt.f32", "sqrt.f32");
}
/* Python _op_math_rsqrt -> llvm.amdgcn.rsq.f32. */
static void _op_math_rsqrt(ckc_lower_t *L, const ckc_op_t *op) {
    ll_math_f32_unary(L, op, "rsqrt", "rsqrt.f32", "amdgcn.rsq.f32");
}
/* Python _op_math_tanh -> llvm.tanh.f32. */
static void _op_math_tanh(ckc_lower_t *L, const ckc_op_t *op) {
    ll_math_f32_unary(L, op, "tanh", "tanh.f32", "tanh.f32");
}

/* ------------------------------------------------------------------ gpu.* */

/* Python _op_gpu_thread_id -> llvm.amdgcn.workitem.id.<axis>. */
static void _op_gpu_thread_id(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const char *axis = ckc_attr_get_str(&op->attrs, "axis");
    if (axis == NULL) {
        axis = "x";
    }
    char key[32];
    snprintf(key, sizeof key, "workitem.%s", axis);
    ckc_ll_need(L, key);
    ckc_ll_emitf(L,
                 "  %s = call i32 @llvm.amdgcn.workitem.id.%s()",
                 res->name, axis);
}

/* Python _op_gpu_block_id -> llvm.amdgcn.workgroup.id.<axis>. */
static void _op_gpu_block_id(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    const char *axis = ckc_attr_get_str(&op->attrs, "axis");
    if (axis == NULL) {
        axis = "x";
    }
    char key[32];
    snprintf(key, sizeof key, "workgroup.%s", axis);
    ckc_ll_need(L, key);
    ckc_ll_emitf(L,
                 "  %s = call i32 @llvm.amdgcn.workgroup.id.%s()",
                 res->name, axis);
}

/* ------------------------------------------------------------- registration */

void ckc_ll_register_arith(void) {
    ckc_ll_set_handler(CKC_OP_ARITH_CONSTANT,     _op_arith_constant);
    ckc_ll_set_handler(CKC_OP_ARITH_CONSTANT_VEC, _op_arith_constant_vec);
    ckc_ll_set_handler(CKC_OP_ARITH_ADD,          _op_arith_add);
    ckc_ll_set_handler(CKC_OP_ARITH_SUB,          _op_arith_sub);
    ckc_ll_set_handler(CKC_OP_ARITH_MUL,          _op_arith_mul);
    ckc_ll_set_handler(CKC_OP_ARITH_DIV,          _op_arith_div);
    ckc_ll_set_handler(CKC_OP_ARITH_MOD,          _op_arith_mod);
    ckc_ll_set_handler(CKC_OP_ARITH_FADD,         _op_arith_fadd);
    ckc_ll_set_handler(CKC_OP_ARITH_FSUB,         _op_arith_fsub);
    ckc_ll_set_handler(CKC_OP_ARITH_FMUL,         _op_arith_fmul);
    ckc_ll_set_handler(CKC_OP_ARITH_FDIV,         _op_arith_fdiv);
    ckc_ll_set_handler(CKC_OP_ARITH_FNEG,         _op_arith_fneg);
    ckc_ll_set_handler(CKC_OP_ARITH_FABS,         _op_arith_fabs);
    ckc_ll_set_handler(CKC_OP_ARITH_FMA,          _op_arith_fma);
    ckc_ll_set_handler(CKC_OP_ARITH_FMAX3,        _op_arith_fmax3);
    ckc_ll_set_handler(CKC_OP_ARITH_FMIN3,        _op_arith_fmin3);
    ckc_ll_set_handler(CKC_OP_ARITH_CMP,          _op_arith_cmp);
    ckc_ll_set_handler(CKC_OP_ARITH_FCMP,         _op_arith_fcmp);
    ckc_ll_set_handler(CKC_OP_ARITH_FMAX,         _op_arith_fmax);
    ckc_ll_set_handler(CKC_OP_ARITH_FMIN,         _op_arith_fmin);
    ckc_ll_set_handler(CKC_OP_ARITH_SELECT,       _op_arith_select);
    ckc_ll_set_handler(CKC_OP_ARITH_AND,          _op_arith_and);
    ckc_ll_set_handler(CKC_OP_ARITH_OR,           _op_arith_or);
    ckc_ll_set_handler(CKC_OP_ARITH_NOT,          _op_arith_not);
    ckc_ll_set_handler(CKC_OP_ARITH_SMAX,         _op_arith_smax);
    ckc_ll_set_handler(CKC_OP_ARITH_SMIN,         _op_arith_smin);
    ckc_ll_set_handler(CKC_OP_ARITH_XOR,          _op_arith_xor);
    ckc_ll_set_handler(CKC_OP_ARITH_SHL,          _op_arith_shl);
    ckc_ll_set_handler(CKC_OP_ARITH_LSHR,         _op_arith_lshr);
    ckc_ll_set_handler(CKC_OP_ARITH_UMUL_HI_I32,  _op_arith_umul_hi_i32);

    ckc_ll_set_handler(CKC_OP_MATH_EXP2,          _op_math_exp2);
    ckc_ll_set_handler(CKC_OP_MATH_LOG2,          _op_math_log2);
    ckc_ll_set_handler(CKC_OP_MATH_RCP,           _op_math_rcp);
    ckc_ll_set_handler(CKC_OP_MATH_RCP_FAST,      _op_math_rcp_fast);
    ckc_ll_set_handler(CKC_OP_MATH_SQRT,          _op_math_sqrt);
    ckc_ll_set_handler(CKC_OP_MATH_RSQRT,         _op_math_rsqrt);
    ckc_ll_set_handler(CKC_OP_MATH_TANH,          _op_math_tanh);

    ckc_ll_set_handler(CKC_OP_GPU_THREAD_ID,      _op_gpu_thread_id);
    ckc_ll_set_handler(CKC_OP_GPU_BLOCK_ID,       _op_gpu_block_id);
}
