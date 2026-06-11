/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ir_ir_arith.c -- bucket "ir_arith" of the C99 port of ck_dsl.core.ir.
 *
 * Implements arith.* constants / integer / logic / float ops, comparisons,
 * math.* transcendentals, clamp/select, and every scalar cast/conversion
 * (zext..cvt_*). Faithful translation of ir.py lines ~364-942 + the
 * fp16_zero / zero_vec_f32 / zero_vec / bitcast helpers.
 *
 * All shared plumbing (ckc_i_op, ckc_i_op1, ckc_i_binop, ckc_i_unop,
 * ckc_i_attrs, ckc_i_set_err, ckc_i_live, ...) lives in bucket 0 (ir_core.c);
 * this file only calls it via ir_internal.h.
 */
#include "ckc/ir_internal.h"

/* ------------------------------------------------------------ arith constants */

ckc_value_t *ckc_b_const_i32(ckc_ir_builder_t *b, int64_t value)
{
    ckc_attr_map_t a;
    if (!ckc_i_live(b)) return NULL;
    a = ckc_i_attrs(b);
    ckc_attr_set_int(b, &a, "value", value);
    ckc_attr_set_str(b, &a, "ity", "i32");
    return ckc_i_op1(b, CKC_OP_ARITH_CONSTANT, NULL, 0, ckc_i32(), &a, "c");
}

ckc_value_t *ckc_b_const_i64(ckc_ir_builder_t *b, int64_t value)
{
    ckc_attr_map_t a;
    if (!ckc_i_live(b)) return NULL;
    a = ckc_i_attrs(b);
    ckc_attr_set_int(b, &a, "value", value);
    ckc_attr_set_str(b, &a, "ity", "i64");
    return ckc_i_op1(b, CKC_OP_ARITH_CONSTANT, NULL, 0, ckc_i64(), &a, "c");
}

ckc_value_t *ckc_b_const_f32(ckc_ir_builder_t *b, double value)
{
    ckc_attr_map_t a;
    if (!ckc_i_live(b)) return NULL;
    a = ckc_i_attrs(b);
    ckc_attr_set_float(b, &a, "value", value);
    ckc_attr_set_str(b, &a, "ity", "f32");
    return ckc_i_op1(b, CKC_OP_ARITH_CONSTANT, NULL, 0, ckc_f32(), &a, "c");
}

ckc_value_t *ckc_b_fp16_zero(ckc_ir_builder_t *b)
{
    ckc_attr_map_t a;
    if (!ckc_i_live(b)) return NULL;
    a = ckc_i_attrs(b);
    ckc_attr_set_float(b, &a, "value", 0.0);
    ckc_attr_set_str(b, &a, "ity", "f16");
    return ckc_i_op1(b, CKC_OP_ARITH_CONSTANT, NULL, 0, ckc_f16(), &a, "c");
}

/* Shared body for the constant_vec emitters. `hint` is the literal Python
 * "cz{n}" result_name_hint formed by the caller. */
static ckc_value_t *zero_vec_impl(ckc_ir_builder_t *b, const ckc_type_t *elem,
                                  const char *elem_name, int n, const char *hint)
{
    ckc_attr_map_t a;
    const ckc_type_t *vt;
    vt = ckc_vector_type(b, elem, n);
    if (!vt) return NULL;
    a = ckc_i_attrs(b);
    ckc_attr_set_float(b, &a, "fill", 0.0);
    ckc_attr_set_str(b, &a, "elem", elem_name);
    ckc_attr_set_int(b, &a, "vec", n);
    return ckc_i_op1(b, CKC_OP_ARITH_CONSTANT_VEC, NULL, 0, vt, &a, hint);
}

/* Build the "cz{n}" hint into an arena-owned string, matching Python's
 * result_name_hint=f"cz{n}". */
static const char *cz_hint(ckc_ir_builder_t *b, int n)
{
    char *out = ckc_arena_printf(&b->arena, "cz%d", n);
    if (!out) return (const char *)ckc_i_set_err(b, CKC_ERR_OOM, "OOM cz hint");
    return out;
}

ckc_value_t *ckc_b_zero_vec_f32(ckc_ir_builder_t *b, int n)
{
    const char *hint;
    if (!ckc_i_live(b)) return NULL;
    if (n <= 0)
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "zero_vec_f32 needs positive n, got %d", n);
    hint = cz_hint(b, n);
    if (!hint) return NULL;
    return zero_vec_impl(b, ckc_f32(), "f32", n, hint);
}

ckc_value_t *ckc_b_zero_vec(ckc_ir_builder_t *b, const ckc_type_t *elem, int n)
{
    const char *hint;
    if (!ckc_i_live(b)) return NULL;
    if (!elem)
        return ckc_i_set_err(b, CKC_ERR_VALUE, "zero_vec elem is NULL");
    if (ckc_type_eq(elem, ckc_f32()))
        return ckc_b_zero_vec_f32(b, n);
    if (ckc_type_eq(elem, ckc_f16()) || ckc_type_eq(elem, ckc_bf16()) ||
        ckc_type_eq(elem, ckc_fp8e4m3()) || ckc_type_eq(elem, ckc_bf8e5m2()) ||
        ckc_type_eq(elem, ckc_i8()) || ckc_type_eq(elem, ckc_i32())) {
        hint = cz_hint(b, n);
        if (!hint) return NULL;
        return zero_vec_impl(b, elem, elem->name, n, hint);
    }
    return ckc_i_set_err(b, CKC_ERR_VALUE, "zero_vec unsupported elem %s",
                         elem->name);
}

/* ------------------------------------------------------ arith integer / logic */

ckc_value_t *ckc_b_add(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_ADD, a, c, "add");
}

ckc_value_t *ckc_b_sub(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_SUB, a, c, "sub");
}

ckc_value_t *ckc_b_mul(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_MUL, a, c, "mul");
}

ckc_value_t *ckc_b_div(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_DIV, a, c, "div");
}

ckc_value_t *ckc_b_mod(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_MOD, a, c, "mod");
}

ckc_value_t *ckc_b_land(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_AND, a, c, "and");
}

ckc_value_t *ckc_b_lor(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_OR, a, c, "or");
}

ckc_value_t *ckc_b_lnot(ckc_ir_builder_t *b, ckc_value_t *a)
{
    return ckc_i_unop(b, CKC_OP_ARITH_NOT, a, "not");
}

ckc_value_t *ckc_b_smax(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_SMAX, a, c, "smax");
}

ckc_value_t *ckc_b_smin(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_SMIN, a, c, "smin");
}

ckc_value_t *ckc_b_xor(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_XOR, a, c, "xor");
}

ckc_value_t *ckc_b_shl(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_SHL, a, c, "shl");
}

ckc_value_t *ckc_b_lshr(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_LSHR, a, c, "lshr");
}

ckc_value_t *ckc_b_umul_hi_i32(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    ckc_value_t *operands[2];
    if (!ckc_i_live(b)) return NULL;
    if (!a || !c)
        return ckc_i_set_err(b, CKC_ERR_VALUE, "umul_hi_i32 NULL operand");
    if (!ckc_i_type_is(a->type, "i32") || !ckc_i_type_is(c->type, "i32"))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "umul_hi_i32 expects i32 operands, got %s / %s",
                             a->type->name, c->type->name);
    operands[0] = a;
    operands[1] = c;
    return ckc_i_op1(b, CKC_OP_ARITH_UMUL_HI_I32, operands, 2, ckc_i32(),
                     NULL, "umh");
}

/* ------------------------------------------------------------- arith float */

ckc_value_t *ckc_b_fadd(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_FADD, a, c, "fadd");
}

ckc_value_t *ckc_b_fsub(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_FSUB, a, c, "fsub");
}

ckc_value_t *ckc_b_fmul(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_FMUL, a, c, "fmul");
}

ckc_value_t *ckc_b_fdiv(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_FDIV, a, c, "fdiv");
}

ckc_value_t *ckc_b_fneg(ckc_ir_builder_t *b, ckc_value_t *a)
{
    return ckc_i_unop(b, CKC_OP_ARITH_FNEG, a, "fneg");
}

ckc_value_t *ckc_b_fabs(ckc_ir_builder_t *b, ckc_value_t *a)
{
    return ckc_i_unop(b, CKC_OP_ARITH_FABS, a, "fabs");
}

ckc_value_t *ckc_b_fma(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c,
                       ckc_value_t *d)
{
    ckc_value_t *operands[3];
    if (!ckc_i_live(b)) return NULL;
    if (!a || !c || !d)
        return ckc_i_set_err(b, CKC_ERR_VALUE, "fma NULL operand");
    if (!ckc_type_eq(a->type, c->type) || !ckc_type_eq(c->type, d->type))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "fma expects matching types; got %s, %s, %s",
                             a->type->name, c->type->name, d->type->name);
    operands[0] = a;
    operands[1] = c;
    operands[2] = d;
    return ckc_i_op1(b, CKC_OP_ARITH_FMA, operands, 3, a->type, NULL, "fma");
}

ckc_value_t *ckc_b_fmax(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_FMAX, a, c, "fmax");
}

ckc_value_t *ckc_b_fmin(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return ckc_i_binop(b, CKC_OP_ARITH_FMIN, a, c, "fmin");
}

ckc_value_t *ckc_b_fmax3(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c,
                         ckc_value_t *d)
{
    ckc_value_t *operands[3];
    if (!ckc_i_live(b)) return NULL;
    if (!a || !c || !d)
        return ckc_i_set_err(b, CKC_ERR_VALUE, "fmax3 NULL operand");
    if (!ckc_type_eq(a->type, c->type) || !ckc_type_eq(c->type, d->type))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "fmax3 expects matching types; got %s, %s, %s",
                             a->type->name, c->type->name, d->type->name);
    operands[0] = a;
    operands[1] = c;
    operands[2] = d;
    return ckc_i_op1(b, CKC_OP_ARITH_FMAX3, operands, 3, a->type, NULL, "fmax3");
}

ckc_value_t *ckc_b_fmin3(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c,
                         ckc_value_t *d)
{
    ckc_value_t *operands[3];
    if (!ckc_i_live(b)) return NULL;
    if (!a || !c || !d)
        return ckc_i_set_err(b, CKC_ERR_VALUE, "fmin3 NULL operand");
    if (!ckc_type_eq(a->type, c->type) || !ckc_type_eq(c->type, d->type))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "fmin3 expects matching types; got %s, %s, %s",
                             a->type->name, c->type->name, d->type->name);
    operands[0] = a;
    operands[1] = c;
    operands[2] = d;
    return ckc_i_op1(b, CKC_OP_ARITH_FMIN3, operands, 3, a->type, NULL, "fmin3");
}

ckc_value_t *ckc_b_clamp_f32(ckc_ir_builder_t *b, ckc_value_t *v,
                             ckc_value_t *lo, ckc_value_t *hi)
{
    /* Python: min(hi, max(lo, v)) == self.fmin(hi, self.fmax(lo, v)). */
    ckc_value_t *inner;
    if (!ckc_i_live(b)) return NULL;
    if (!v)
        return ckc_i_set_err(b, CKC_ERR_VALUE, "clamp_f32 NULL value");
    if (!ckc_i_type_is(v->type, "f32"))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "clamp_f32 expects f32 input, got %s", v->type->name);
    inner = ckc_b_fmax(b, lo, v);
    return ckc_b_fmin(b, hi, inner);
}

/* ----------------------------------------------------- comparisons (-> i1) */

static ckc_value_t *cmp_impl(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c,
                             const char *pred, const char *hint)
{
    ckc_value_t *operands[2];
    ckc_attr_map_t at;
    if (!ckc_i_live(b)) return NULL;
    if (!a || !c)
        return ckc_i_set_err(b, CKC_ERR_VALUE, "cmp NULL operand");
    operands[0] = a;
    operands[1] = c;
    at = ckc_i_attrs(b);
    ckc_attr_set_str(b, &at, "pred", pred);
    return ckc_i_op1(b, CKC_OP_ARITH_CMP, operands, 2, ckc_i1(), &at, hint);
}

ckc_value_t *ckc_b_cmp_lt(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return cmp_impl(b, a, c, "lt", "lt");
}

ckc_value_t *ckc_b_cmp_le(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return cmp_impl(b, a, c, "le", "le");
}

ckc_value_t *ckc_b_cmp_gt(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return cmp_impl(b, a, c, "gt", "gt");
}

ckc_value_t *ckc_b_cmp_ge(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return cmp_impl(b, a, c, "ge", "ge");
}

ckc_value_t *ckc_b_cmp_eq(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return cmp_impl(b, a, c, "eq", "eq");
}

ckc_value_t *ckc_b_cmp_ne(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *c)
{
    return cmp_impl(b, a, c, "ne", "ne");
}

ckc_value_t *ckc_b_fcmp(ckc_ir_builder_t *b, const char *pred, ckc_value_t *a,
                        ckc_value_t *c)
{
    ckc_value_t *operands[2];
    ckc_attr_map_t at;
    static const char *const valid[] = {
        "olt", "ole", "ogt", "oge", "oeq", "one", "ord", "uno"
    };
    int i;
    bool ok = false;
    if (!ckc_i_live(b)) return NULL;
    if (!pred)
        return ckc_i_set_err(b, CKC_ERR_VALUE, "fcmp predicate is NULL");
    for (i = 0; i < (int)(sizeof(valid) / sizeof(valid[0])); ++i) {
        const char *p = pred, *q = valid[i];
        while (*p && *q && *p == *q) { ++p; ++q; }
        if (*p == '\0' && *q == '\0') { ok = true; break; }
    }
    if (!ok)
        return ckc_i_set_err(b, CKC_ERR_VALUE, "unsupported fcmp predicate '%s'",
                             pred);
    if (!a || !c)
        return ckc_i_set_err(b, CKC_ERR_VALUE, "fcmp NULL operand");
    operands[0] = a;
    operands[1] = c;
    at = ckc_i_attrs(b);
    ckc_attr_set_str(b, &at, "pred", pred);
    return ckc_i_op1(b, CKC_OP_ARITH_FCMP, operands, 2, ckc_i1(), &at, "fcmp");
}

/* ------------------------------------------------------------------- math.* */

ckc_value_t *ckc_b_exp2(ckc_ir_builder_t *b, ckc_value_t *a)
{
    return ckc_i_unop(b, CKC_OP_MATH_EXP2, a, "exp2");
}

ckc_value_t *ckc_b_log2(ckc_ir_builder_t *b, ckc_value_t *a)
{
    return ckc_i_unop(b, CKC_OP_MATH_LOG2, a, "log2");
}

ckc_value_t *ckc_b_rcp(ckc_ir_builder_t *b, ckc_value_t *a)
{
    return ckc_i_unop(b, CKC_OP_MATH_RCP, a, "rcp");
}

ckc_value_t *ckc_b_rcp_fast(ckc_ir_builder_t *b, ckc_value_t *a)
{
    return ckc_i_unop(b, CKC_OP_MATH_RCP_FAST, a, "rcpf");
}

ckc_value_t *ckc_b_sqrt(ckc_ir_builder_t *b, ckc_value_t *a)
{
    return ckc_i_unop(b, CKC_OP_MATH_SQRT, a, "sqrt");
}

ckc_value_t *ckc_b_rsqrt(ckc_ir_builder_t *b, ckc_value_t *a)
{
    return ckc_i_unop(b, CKC_OP_MATH_RSQRT, a, "rsq");
}

ckc_value_t *ckc_b_tanh(ckc_ir_builder_t *b, ckc_value_t *a)
{
    return ckc_i_unop(b, CKC_OP_MATH_TANH, a, "tanh");
}

/* -------------------------------------------------------- casts / conversions */

ckc_value_t *ckc_b_zext(ckc_ir_builder_t *b, ckc_value_t *v,
                        const ckc_type_t *target)
{
    ckc_value_t *operands[1];
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "zext NULL value");
    if (!target) return ckc_i_set_err(b, CKC_ERR_VALUE, "zext NULL target");
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_ZEXT, operands, 1, target, NULL, "zx");
}

ckc_value_t *ckc_b_sext(ckc_ir_builder_t *b, ckc_value_t *v,
                        const ckc_type_t *target)
{
    ckc_value_t *operands[1];
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "sext NULL value");
    if (!target) return ckc_i_set_err(b, CKC_ERR_VALUE, "sext NULL target");
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_SEXT, operands, 1, target, NULL, "sx");
}

ckc_value_t *ckc_b_trunc(ckc_ir_builder_t *b, ckc_value_t *v,
                         const ckc_type_t *target)
{
    ckc_value_t *operands[1];
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "trunc NULL value");
    if (!target) return ckc_i_set_err(b, CKC_ERR_VALUE, "trunc NULL target");
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_TRUNC, operands, 1, target, NULL, "tr");
}

ckc_value_t *ckc_b_bitcast(ckc_ir_builder_t *b, ckc_value_t *v,
                           const ckc_type_t *target)
{
    ckc_value_t *operands[1];
    ckc_attr_map_t at;
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "bitcast NULL value");
    if (!target) return ckc_i_set_err(b, CKC_ERR_VALUE, "bitcast NULL target");
    operands[0] = v;
    at = ckc_i_attrs(b);
    ckc_attr_set_str(b, &at, "target", target->name);
    return ckc_i_op1(b, CKC_OP_ARITH_BITCAST, operands, 1, target, &at, "bc");
}

ckc_value_t *ckc_b_select(ckc_ir_builder_t *b, ckc_value_t *cond,
                          ckc_value_t *lhs, ckc_value_t *rhs)
{
    ckc_value_t *operands[3];
    if (!ckc_i_live(b)) return NULL;
    if (!cond || !lhs || !rhs)
        return ckc_i_set_err(b, CKC_ERR_VALUE, "select NULL operand");
    operands[0] = cond;
    operands[1] = lhs;
    operands[2] = rhs;
    return ckc_i_op1(b, CKC_OP_ARITH_SELECT, operands, 3, lhs->type, NULL, "sel");
}

ckc_value_t *ckc_b_masked_select(ckc_ir_builder_t *b, ckc_value_t *cond,
                                 ckc_value_t *lhs, ckc_value_t *rhs)
{
    /* Python masked_select simply delegates to select. */
    return ckc_b_select(b, cond, lhs, rhs);
}

ckc_value_t *ckc_b_trunc_f32_to_f16(ckc_ir_builder_t *b, ckc_value_t *v)
{
    ckc_value_t *operands[1];
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "trunc_f32_to_f16 NULL value");
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_TRUNC_F32_TO_F16, operands, 1, ckc_f16(),
                     NULL, "t");
}

ckc_value_t *ckc_b_rint_f32(ckc_ir_builder_t *b, ckc_value_t *v)
{
    ckc_value_t *operands[1];
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "rint_f32 NULL value");
    if (!ckc_i_type_is(v->type, "f32"))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "rint_f32 expects f32 input, got %s", v->type->name);
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_RINT_F32, operands, 1, ckc_f32(),
                     NULL, "rint");
}

ckc_value_t *ckc_b_cast_to_f32(ckc_ir_builder_t *b, ckc_value_t *v)
{
    ckc_value_t *operands[1];
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "cast_to_f32 NULL value");
    if (ckc_i_type_is(v->type, "f32"))
        return v;
    if (!ckc_i_type_is(v->type, "f16") && !ckc_i_type_is(v->type, "bf16"))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cast_to_f32 unsupported from %s", v->type->name);
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_CAST_TO_F32, operands, 1, ckc_f32(),
                     NULL, "f32");
}

ckc_value_t *ckc_b_cast_f32_to(ckc_ir_builder_t *b, ckc_value_t *v,
                               const ckc_type_t *target)
{
    ckc_value_t *operands[1];
    ckc_attr_map_t at;
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "cast_f32_to NULL value");
    if (!target) return ckc_i_set_err(b, CKC_ERR_VALUE, "cast_f32_to NULL target");
    if (!ckc_i_type_is(v->type, "f32"))
        return ckc_i_set_err(b, CKC_ERR_VALUE, "cast_f32_to expects f32 input");
    if (ckc_i_type_is(target, "f32"))
        return v;
    if (!ckc_i_type_is(target, "f16") && !ckc_i_type_is(target, "bf16"))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cast_f32_to unsupported to %s", target->name);
    operands[0] = v;
    at = ckc_i_attrs(b);
    ckc_attr_set_str(b, &at, "target", target->name);
    return ckc_i_op1(b, CKC_OP_ARITH_CAST_F32_TO, operands, 1, target,
                     &at, "cast");
}

ckc_value_t *ckc_b_sitofp_f32(ckc_ir_builder_t *b, ckc_value_t *v)
{
    ckc_value_t *operands[1];
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "sitofp_f32 NULL value");
    if (!ckc_i_type_is(v->type, "i32"))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "sitofp_f32 expects i32 input, got %s", v->type->name);
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_SITOFP_F32, operands, 1, ckc_f32(),
                     NULL, "sitof");
}

ckc_value_t *ckc_b_cvt_fp8_to_f32(ckc_ir_builder_t *b, ckc_value_t *v)
{
    ckc_value_t *operands[1];
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "cvt_fp8_to_f32 NULL value");
    if (!ckc_i_type_is(v->type, "fp8e4m3"))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_fp8_to_f32 expects fp8e4m3 input, got %s",
                             v->type->name);
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_CVT_FP8_TO_F32, operands, 1, ckc_f32(),
                     NULL, "dq8");
}

ckc_value_t *ckc_b_cvt_bf8_to_f32(ckc_ir_builder_t *b, ckc_value_t *v)
{
    ckc_value_t *operands[1];
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "cvt_bf8_to_f32 NULL value");
    if (!ckc_i_type_is(v->type, "bf8e5m2"))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_bf8_to_f32 expects bf8e5m2 input, got %s",
                             v->type->name);
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_CVT_BF8_TO_F32, operands, 1, ckc_f32(),
                     NULL, "dqb8");
}

ckc_value_t *ckc_b_cvt_pk_f32_fp8x4(ckc_ir_builder_t *b, ckc_value_t *v)
{
    ckc_value_t *operands[1];
    const ckc_type_t *vt;
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "cvt_pk_f32_fp8x4 NULL value");
    if (!ckc_i_is_vector(v->type, "fp8e4m3", 4))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_pk_f32_fp8x4 expects vec<fp8e4m3x4> input, got %s",
                             v->type->name);
    vt = ckc_vector_type(b, ckc_f32(), 4);
    if (!vt) return NULL;
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_CVT_PK_F32_FP8X4, operands, 1, vt,
                     NULL, "dq8x4");
}

ckc_value_t *ckc_b_cvt_pk_f32_bf8x4(ckc_ir_builder_t *b, ckc_value_t *v)
{
    ckc_value_t *operands[1];
    const ckc_type_t *vt;
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "cvt_pk_f32_bf8x4 NULL value");
    if (!ckc_i_is_vector(v->type, "bf8e5m2", 4))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_pk_f32_bf8x4 expects vec<bf8e5m2x4> input, got %s",
                             v->type->name);
    vt = ckc_vector_type(b, ckc_f32(), 4);
    if (!vt) return NULL;
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_CVT_PK_F32_BF8X4, operands, 1, vt,
                     NULL, "dqb8x4");
}

ckc_value_t *ckc_b_cvt_scalef32_pk_f32_fp8x4(ckc_ir_builder_t *b, ckc_value_t *v,
                                             ckc_value_t *scale)
{
    ckc_value_t *operands[2];
    const ckc_type_t *vt;
    if (!ckc_i_live(b)) return NULL;
    if (!v || !scale)
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_scalef32_pk_f32_fp8x4 NULL operand");
    if (!ckc_i_is_vector(v->type, "fp8e4m3", 4))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_scalef32_pk_f32_fp8x4 expects vec<fp8e4m3x4>, got %s",
                             v->type->name);
    if (!ckc_i_type_is(scale->type, "f32"))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_scalef32_pk_f32_fp8x4 scale must be f32, got %s",
                             scale->type->name);
    vt = ckc_vector_type(b, ckc_f32(), 4);
    if (!vt) return NULL;
    operands[0] = v;
    operands[1] = scale;
    return ckc_i_op1(b, CKC_OP_ARITH_CVT_SCALEF32_PK_F32_FP8, operands, 2, vt,
                     NULL, "sdq8x4");
}

ckc_value_t *ckc_b_cvt_scalef32_pk_f32_bf8x4(ckc_ir_builder_t *b, ckc_value_t *v,
                                             ckc_value_t *scale)
{
    ckc_value_t *operands[2];
    const ckc_type_t *vt;
    if (!ckc_i_live(b)) return NULL;
    if (!v || !scale)
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_scalef32_pk_f32_bf8x4 NULL operand");
    if (!ckc_i_is_vector(v->type, "bf8e5m2", 4))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_scalef32_pk_f32_bf8x4 expects vec<bf8e5m2x4>, got %s",
                             v->type->name);
    if (!ckc_i_type_is(scale->type, "f32"))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_scalef32_pk_f32_bf8x4 scale must be f32");
    vt = ckc_vector_type(b, ckc_f32(), 4);
    if (!vt) return NULL;
    operands[0] = v;
    operands[1] = scale;
    return ckc_i_op1(b, CKC_OP_ARITH_CVT_SCALEF32_PK_F32_BF8, operands, 2, vt,
                     NULL, "sdqb8x4");
}

ckc_value_t *ckc_b_cvt_f32_to_fp8(ckc_ir_builder_t *b, ckc_value_t *v)
{
    ckc_value_t *operands[1];
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "cvt_f32_to_fp8 NULL value");
    if (!ckc_i_type_is(v->type, "f32"))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_f32_to_fp8 expects f32 input, got %s",
                             v->type->name);
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_CVT_F32_TO_FP8, operands, 1, ckc_fp8e4m3(),
                     NULL, "q8");
}

ckc_value_t *ckc_b_cvt_f32_to_bf8(ckc_ir_builder_t *b, ckc_value_t *v)
{
    ckc_value_t *operands[1];
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "cvt_f32_to_bf8 NULL value");
    if (!ckc_i_type_is(v->type, "f32"))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_f32_to_bf8 expects f32 input, got %s",
                             v->type->name);
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_CVT_F32_TO_BF8, operands, 1, ckc_bf8e5m2(),
                     NULL, "qb8");
}

ckc_value_t *ckc_b_cvt_f32_to_i8_sat(ckc_ir_builder_t *b, ckc_value_t *v)
{
    ckc_value_t *operands[1];
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "cvt_f32_to_i8_sat NULL value");
    if (!ckc_i_type_is(v->type, "f32"))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_f32_to_i8_sat expects f32 input, got %s",
                             v->type->name);
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_CVT_F32_TO_I8_SAT, operands, 1, ckc_i8(),
                     NULL, "qi8");
}

ckc_value_t *ckc_b_cvt_pk_fp8_f32x4(ckc_ir_builder_t *b, ckc_value_t *v)
{
    ckc_value_t *operands[1];
    const ckc_type_t *vt;
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "cvt_pk_fp8_f32x4 NULL value");
    if (!ckc_i_is_vector(v->type, "f32", 4))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_pk_fp8_f32x4 expects vec<f32x4> input, got %s",
                             v->type->name);
    vt = ckc_vector_type(b, ckc_fp8e4m3(), 4);
    if (!vt) return NULL;
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_CVT_PK_FP8_F32X4, operands, 1, vt,
                     NULL, "q8x4");
}

ckc_value_t *ckc_b_cvt_pk_bf8_f32x4(ckc_ir_builder_t *b, ckc_value_t *v)
{
    ckc_value_t *operands[1];
    const ckc_type_t *vt;
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "cvt_pk_bf8_f32x4 NULL value");
    if (!ckc_i_is_vector(v->type, "f32", 4))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_pk_bf8_f32x4 expects vec<f32x4> input, got %s",
                             v->type->name);
    vt = ckc_vector_type(b, ckc_bf8e5m2(), 4);
    if (!vt) return NULL;
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_CVT_PK_BF8_F32X4, operands, 1, vt,
                     NULL, "qb8x4");
}

ckc_value_t *ckc_b_cvt_pk_i8_f32x4(ckc_ir_builder_t *b, ckc_value_t *v)
{
    ckc_value_t *operands[1];
    const ckc_type_t *vt;
    if (!ckc_i_live(b)) return NULL;
    if (!v) return ckc_i_set_err(b, CKC_ERR_VALUE, "cvt_pk_i8_f32x4 NULL value");
    if (!ckc_i_is_vector(v->type, "f32", 4))
        return ckc_i_set_err(b, CKC_ERR_VALUE,
                             "cvt_pk_i8_f32x4 expects vec<f32x4> input, got %s",
                             v->type->name);
    vt = ckc_vector_type(b, ckc_i8(), 4);
    if (!vt) return NULL;
    operands[0] = v;
    return ckc_i_op1(b, CKC_OP_ARITH_CVT_PK_I8_F32X4, operands, 1, vt,
                     NULL, "qi8x4");
}
