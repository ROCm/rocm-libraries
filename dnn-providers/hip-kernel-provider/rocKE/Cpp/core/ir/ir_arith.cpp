// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * ir_ir_arith.c -- bucket "ir_arith" of the C99 port of rocke.core.ir.
 *
 * Implements arith.* constants / integer / logic / float ops, comparisons,
 * math.* transcendentals, clamp/select, and every scalar cast/conversion
 * (zext..cvt_*). Faithful translation of ir.py lines ~364-942 + the
 * fp16_zero / zero_vec_f32 / zero_vec / bitcast helpers.
 *
 * All shared plumbing (rocke_i_op, rocke_i_op1, rocke_i_binop, rocke_i_unop,
 * rocke_i_attrs, rocke_i_set_err, rocke_i_live, ...) lives in bucket 0 (ir_core.c);
 * this file only calls it via ir_internal.h.
 */
#include "rocke/ir_internal.h"

/* ------------------------------------------------------------ arith constants */

rocke_value_t* rocke_b_const_i32(rocke_ir_builder_t* b, int64_t value)
{
    rocke_attr_map_t a;
    if(!rocke_i_live(b))
        return NULL;
    a = rocke_i_attrs(b);
    rocke_attr_set_int(b, &a, "value", value);
    rocke_attr_set_str(b, &a, "ity", "i32");
    return rocke_i_op1(b, ROCKE_OP_ARITH_CONSTANT, NULL, 0, rocke_i32(), &a, "c");
}

rocke_value_t* rocke_b_const_i64(rocke_ir_builder_t* b, int64_t value)
{
    rocke_attr_map_t a;
    if(!rocke_i_live(b))
        return NULL;
    a = rocke_i_attrs(b);
    rocke_attr_set_int(b, &a, "value", value);
    rocke_attr_set_str(b, &a, "ity", "i64");
    return rocke_i_op1(b, ROCKE_OP_ARITH_CONSTANT, NULL, 0, rocke_i64(), &a, "c");
}

rocke_value_t* rocke_b_const_f32(rocke_ir_builder_t* b, double value)
{
    rocke_attr_map_t a;
    if(!rocke_i_live(b))
        return NULL;
    a = rocke_i_attrs(b);
    rocke_attr_set_float(b, &a, "value", value);
    rocke_attr_set_str(b, &a, "ity", "f32");
    return rocke_i_op1(b, ROCKE_OP_ARITH_CONSTANT, NULL, 0, rocke_f32(), &a, "c");
}

rocke_value_t* rocke_b_fp16_zero(rocke_ir_builder_t* b)
{
    rocke_attr_map_t a;
    if(!rocke_i_live(b))
        return NULL;
    a = rocke_i_attrs(b);
    rocke_attr_set_float(b, &a, "value", 0.0);
    rocke_attr_set_str(b, &a, "ity", "f16");
    return rocke_i_op1(b, ROCKE_OP_ARITH_CONSTANT, NULL, 0, rocke_f16(), &a, "c");
}

/* Shared body for the constant_vec emitters. `hint` is the literal Python
 * "cz{n}" result_name_hint formed by the caller. */
static rocke_value_t* zero_vec_impl(
    rocke_ir_builder_t* b, const rocke_type_t* elem, const char* elem_name, int n, const char* hint)
{
    rocke_attr_map_t a;
    const rocke_type_t* vt;
    vt = rocke_vector_type(b, elem, n);
    if(!vt)
        return NULL;
    a = rocke_i_attrs(b);
    rocke_attr_set_float(b, &a, "fill", 0.0);
    rocke_attr_set_str(b, &a, "elem", elem_name);
    rocke_attr_set_int(b, &a, "vec", n);
    return rocke_i_op1(b, ROCKE_OP_ARITH_CONSTANT_VEC, NULL, 0, vt, &a, hint);
}

/* Build the "cz{n}" hint into an arena-owned string, matching Python's
 * result_name_hint=f"cz{n}". */
static const char* cz_hint(rocke_ir_builder_t* b, int n)
{
    char* out = rocke_arena_printf(&b->arena, "cz%d", n);
    if(!out)
        return (const char*)rocke_i_set_err(b, ROCKE_ERR_OOM, "OOM cz hint");
    return out;
}

rocke_value_t* rocke_b_zero_vec_f32(rocke_ir_builder_t* b, int n)
{
    const char* hint;
    if(!rocke_i_live(b))
        return NULL;
    if(n <= 0)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "zero_vec_f32 needs positive n, got %d", n);
    hint = cz_hint(b, n);
    if(!hint)
        return NULL;
    return zero_vec_impl(b, rocke_f32(), "f32", n, hint);
}

rocke_value_t* rocke_b_zero_vec(rocke_ir_builder_t* b, const rocke_type_t* elem, int n)
{
    const char* hint;
    if(!rocke_i_live(b))
        return NULL;
    if(!elem)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "zero_vec elem is NULL");
    if(rocke_type_eq(elem, rocke_f32()))
        return rocke_b_zero_vec_f32(b, n);
    if(rocke_type_eq(elem, rocke_f16()) || rocke_type_eq(elem, rocke_bf16())
       || rocke_type_eq(elem, rocke_fp8e4m3()) || rocke_type_eq(elem, rocke_bf8e5m2())
       || rocke_type_eq(elem, rocke_i8()) || rocke_type_eq(elem, rocke_i32()))
    {
        hint = cz_hint(b, n);
        if(!hint)
            return NULL;
        return zero_vec_impl(b, elem, elem->name, n, hint);
    }
    return (rocke_value_t*)rocke_i_set_err(
        b, ROCKE_ERR_VALUE, "zero_vec unsupported elem %s", elem->name);
}

/* ------------------------------------------------------ arith integer / logic */

rocke_value_t* rocke_b_add(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_ADD, a, c, "add");
}

rocke_value_t* rocke_b_sub(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_SUB, a, c, "sub");
}

rocke_value_t* rocke_b_mul(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_MUL, a, c, "mul");
}

rocke_value_t* rocke_b_div(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_DIV, a, c, "div");
}

rocke_value_t* rocke_b_mod(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_MOD, a, c, "mod");
}

rocke_value_t* rocke_b_land(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_AND, a, c, "and");
}

rocke_value_t* rocke_b_lor(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_OR, a, c, "or");
}

rocke_value_t* rocke_b_lnot(rocke_ir_builder_t* b, rocke_value_t* a)
{
    return rocke_i_unop(b, ROCKE_OP_ARITH_NOT, a, "not");
}

rocke_value_t* rocke_b_smax(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_SMAX, a, c, "smax");
}

rocke_value_t* rocke_b_smin(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_SMIN, a, c, "smin");
}

rocke_value_t* rocke_b_xor(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_XOR, a, c, "xor");
}

rocke_value_t* rocke_b_shl(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_SHL, a, c, "shl");
}

rocke_value_t* rocke_b_lshr(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_LSHR, a, c, "lshr");
}

rocke_value_t* rocke_b_umul_hi_i32(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    rocke_value_t* operands[2];
    if(!rocke_i_live(b))
        return NULL;
    if(!a || !c)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "umul_hi_i32 NULL operand");
    if(!rocke_i_type_is(a->type, "i32") || !rocke_i_type_is(c->type, "i32"))
        return (rocke_value_t*)rocke_i_set_err(b,
                                               ROCKE_ERR_VALUE,
                                               "umul_hi_i32 expects i32 operands, got %s / %s",
                                               a->type->name,
                                               c->type->name);
    operands[0] = a;
    operands[1] = c;
    return rocke_i_op1(b, ROCKE_OP_ARITH_UMUL_HI_I32, operands, 2, rocke_i32(), NULL, "umh");
}

/* ------------------------------------------------------------- arith float */

rocke_value_t* rocke_b_fadd(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_FADD, a, c, "fadd");
}

rocke_value_t* rocke_b_fsub(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_FSUB, a, c, "fsub");
}

rocke_value_t* rocke_b_fmul(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_FMUL, a, c, "fmul");
}

rocke_value_t* rocke_b_fdiv(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_FDIV, a, c, "fdiv");
}

rocke_value_t* rocke_b_fneg(rocke_ir_builder_t* b, rocke_value_t* a)
{
    return rocke_i_unop(b, ROCKE_OP_ARITH_FNEG, a, "fneg");
}

rocke_value_t* rocke_b_fabs(rocke_ir_builder_t* b, rocke_value_t* a)
{
    return rocke_i_unop(b, ROCKE_OP_ARITH_FABS, a, "fabs");
}

rocke_value_t*
    rocke_b_fma(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, rocke_value_t* d)
{
    rocke_value_t* operands[3];
    if(!rocke_i_live(b))
        return NULL;
    if(!a || !c || !d)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "fma NULL operand");
    if(!rocke_type_eq(a->type, c->type) || !rocke_type_eq(c->type, d->type))
        return (rocke_value_t*)rocke_i_set_err(b,
                                               ROCKE_ERR_VALUE,
                                               "fma expects matching types; got %s, %s, %s",
                                               a->type->name,
                                               c->type->name,
                                               d->type->name);
    operands[0] = a;
    operands[1] = c;
    operands[2] = d;
    return rocke_i_op1(b, ROCKE_OP_ARITH_FMA, operands, 3, a->type, NULL, "fma");
}

rocke_value_t* rocke_b_fmax(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_FMAX, a, c, "fmax");
}

rocke_value_t* rocke_b_fmin(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_binop(b, ROCKE_OP_ARITH_FMIN, a, c, "fmin");
}

rocke_value_t*
    rocke_b_fmax3(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, rocke_value_t* d)
{
    rocke_value_t* operands[3];
    if(!rocke_i_live(b))
        return NULL;
    if(!a || !c || !d)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "fmax3 NULL operand");
    if(!rocke_type_eq(a->type, c->type) || !rocke_type_eq(c->type, d->type))
        return (rocke_value_t*)rocke_i_set_err(b,
                                               ROCKE_ERR_VALUE,
                                               "fmax3 expects matching types; got %s, %s, %s",
                                               a->type->name,
                                               c->type->name,
                                               d->type->name);
    operands[0] = a;
    operands[1] = c;
    operands[2] = d;
    return rocke_i_op1(b, ROCKE_OP_ARITH_FMAX3, operands, 3, a->type, NULL, "fmax3");
}

rocke_value_t*
    rocke_b_fmin3(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, rocke_value_t* d)
{
    rocke_value_t* operands[3];
    if(!rocke_i_live(b))
        return NULL;
    if(!a || !c || !d)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "fmin3 NULL operand");
    if(!rocke_type_eq(a->type, c->type) || !rocke_type_eq(c->type, d->type))
        return (rocke_value_t*)rocke_i_set_err(b,
                                               ROCKE_ERR_VALUE,
                                               "fmin3 expects matching types; got %s, %s, %s",
                                               a->type->name,
                                               c->type->name,
                                               d->type->name);
    operands[0] = a;
    operands[1] = c;
    operands[2] = d;
    return rocke_i_op1(b, ROCKE_OP_ARITH_FMIN3, operands, 3, a->type, NULL, "fmin3");
}

rocke_value_t*
    rocke_b_clamp_f32(rocke_ir_builder_t* b, rocke_value_t* v, rocke_value_t* lo, rocke_value_t* hi)
{
    /* Python: min(hi, max(lo, v)) == self.fmin(hi, self.fmax(lo, v)). */
    rocke_value_t* inner;
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "clamp_f32 NULL value");
    if(!rocke_i_type_is(v->type, "f32"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "clamp_f32 expects f32 input, got %s", v->type->name);
    inner = rocke_b_fmax(b, lo, v);
    return rocke_b_fmin(b, hi, inner);
}

/* ----------------------------------------------------- comparisons (-> i1) */

static rocke_value_t* cmp_impl(
    rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, const char* pred, const char* hint)
{
    rocke_value_t* operands[2];
    rocke_attr_map_t at;
    if(!rocke_i_live(b))
        return NULL;
    if(!a || !c)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cmp NULL operand");
    operands[0] = a;
    operands[1] = c;
    at = rocke_i_attrs(b);
    rocke_attr_set_str(b, &at, "pred", pred);
    return rocke_i_op1(b, ROCKE_OP_ARITH_CMP, operands, 2, rocke_i1(), &at, hint);
}

rocke_value_t* rocke_b_cmp_lt(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return cmp_impl(b, a, c, "lt", "lt");
}

rocke_value_t* rocke_b_cmp_le(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return cmp_impl(b, a, c, "le", "le");
}

rocke_value_t* rocke_b_cmp_gt(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return cmp_impl(b, a, c, "gt", "gt");
}

rocke_value_t* rocke_b_cmp_ge(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return cmp_impl(b, a, c, "ge", "ge");
}

rocke_value_t* rocke_b_cmp_eq(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return cmp_impl(b, a, c, "eq", "eq");
}

rocke_value_t* rocke_b_cmp_ne(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return cmp_impl(b, a, c, "ne", "ne");
}

rocke_value_t*
    rocke_b_fcmp(rocke_ir_builder_t* b, const char* pred, rocke_value_t* a, rocke_value_t* c)
{
    rocke_value_t* operands[2];
    rocke_attr_map_t at;
    static const char* const valid[] = {"olt", "ole", "ogt", "oge", "oeq", "one", "ord", "uno"};
    int i;
    bool ok = false;
    if(!rocke_i_live(b))
        return NULL;
    if(!pred)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "fcmp predicate is NULL");
    for(i = 0; i < (int)(sizeof(valid) / sizeof(valid[0])); ++i)
    {
        const char *p = pred, *q = valid[i];
        while(*p && *q && *p == *q)
        {
            ++p;
            ++q;
        }
        if(*p == '\0' && *q == '\0')
        {
            ok = true;
            break;
        }
    }
    if(!ok)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "unsupported fcmp predicate '%s'", pred);
    if(!a || !c)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "fcmp NULL operand");
    operands[0] = a;
    operands[1] = c;
    at = rocke_i_attrs(b);
    rocke_attr_set_str(b, &at, "pred", pred);
    return rocke_i_op1(b, ROCKE_OP_ARITH_FCMP, operands, 2, rocke_i1(), &at, "fcmp");
}

/* ------------------------------------------------------------------- math.* */

rocke_value_t* rocke_b_exp2(rocke_ir_builder_t* b, rocke_value_t* a)
{
    return rocke_i_unop(b, ROCKE_OP_MATH_EXP2, a, "exp2");
}

rocke_value_t* rocke_b_log2(rocke_ir_builder_t* b, rocke_value_t* a)
{
    return rocke_i_unop(b, ROCKE_OP_MATH_LOG2, a, "log2");
}

rocke_value_t* rocke_b_rcp(rocke_ir_builder_t* b, rocke_value_t* a)
{
    return rocke_i_unop(b, ROCKE_OP_MATH_RCP, a, "rcp");
}

rocke_value_t* rocke_b_rcp_fast(rocke_ir_builder_t* b, rocke_value_t* a)
{
    return rocke_i_unop(b, ROCKE_OP_MATH_RCP_FAST, a, "rcpf");
}

rocke_value_t* rocke_b_sqrt(rocke_ir_builder_t* b, rocke_value_t* a)
{
    return rocke_i_unop(b, ROCKE_OP_MATH_SQRT, a, "sqrt");
}

rocke_value_t* rocke_b_rsqrt(rocke_ir_builder_t* b, rocke_value_t* a)
{
    return rocke_i_unop(b, ROCKE_OP_MATH_RSQRT, a, "rsq");
}

rocke_value_t* rocke_b_tanh(rocke_ir_builder_t* b, rocke_value_t* a)
{
    return rocke_i_unop(b, ROCKE_OP_MATH_TANH, a, "tanh");
}

/* -------------------------------------------------------- casts / conversions */

rocke_value_t* rocke_b_zext(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target)
{
    rocke_value_t* operands[1];
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "zext NULL value");
    if(!target)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "zext NULL target");
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_ZEXT, operands, 1, target, NULL, "zx");
}

rocke_value_t* rocke_b_sext(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target)
{
    rocke_value_t* operands[1];
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "sext NULL value");
    if(!target)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "sext NULL target");
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_SEXT, operands, 1, target, NULL, "sx");
}

rocke_value_t* rocke_b_trunc(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target)
{
    rocke_value_t* operands[1];
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "trunc NULL value");
    if(!target)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "trunc NULL target");
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_TRUNC, operands, 1, target, NULL, "tr");
}

rocke_value_t* rocke_b_bitcast(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target)
{
    rocke_value_t* operands[1];
    rocke_attr_map_t at;
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "bitcast NULL value");
    if(!target)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "bitcast NULL target");
    operands[0] = v;
    at = rocke_i_attrs(b);
    rocke_attr_set_str(b, &at, "target", target->name);
    return rocke_i_op1(b, ROCKE_OP_ARITH_BITCAST, operands, 1, target, &at, "bc");
}

rocke_value_t* rocke_b_select(rocke_ir_builder_t* b,
                              rocke_value_t* cond,
                              rocke_value_t* lhs,
                              rocke_value_t* rhs)
{
    rocke_value_t* operands[3];
    if(!rocke_i_live(b))
        return NULL;
    if(!cond || !lhs || !rhs)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "select NULL operand");
    operands[0] = cond;
    operands[1] = lhs;
    operands[2] = rhs;
    return rocke_i_op1(b, ROCKE_OP_ARITH_SELECT, operands, 3, lhs->type, NULL, "sel");
}

rocke_value_t* rocke_b_masked_select(rocke_ir_builder_t* b,
                                     rocke_value_t* cond,
                                     rocke_value_t* lhs,
                                     rocke_value_t* rhs)
{
    /* Python masked_select simply delegates to select. */
    return rocke_b_select(b, cond, lhs, rhs);
}

rocke_value_t* rocke_b_trunc_f32_to_f16(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* operands[1];
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "trunc_f32_to_f16 NULL value");
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_TRUNC_F32_TO_F16, operands, 1, rocke_f16(), NULL, "t");
}

rocke_value_t* rocke_b_rint_f32(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* operands[1];
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "rint_f32 NULL value");
    if(!rocke_i_type_is(v->type, "f32"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "rint_f32 expects f32 input, got %s", v->type->name);
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_RINT_F32, operands, 1, rocke_f32(), NULL, "rint");
}

rocke_value_t* rocke_b_cast_to_f32(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* operands[1];
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cast_to_f32 NULL value");
    if(rocke_i_type_is(v->type, "f32"))
        return v;
    if(!rocke_i_type_is(v->type, "f16") && !rocke_i_type_is(v->type, "bf16"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "cast_to_f32 unsupported from %s", v->type->name);
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_CAST_TO_F32, operands, 1, rocke_f32(), NULL, "f32");
}

rocke_value_t*
    rocke_b_cast_f32_to(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target)
{
    rocke_value_t* operands[1];
    rocke_attr_map_t at;
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cast_f32_to NULL value");
    if(!target)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cast_f32_to NULL target");
    if(!rocke_i_type_is(v->type, "f32"))
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cast_f32_to expects f32 input");
    if(rocke_i_type_is(target, "f32"))
        return v;
    if(!rocke_i_type_is(target, "f16") && !rocke_i_type_is(target, "bf16"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "cast_f32_to unsupported to %s", target->name);
    operands[0] = v;
    at = rocke_i_attrs(b);
    rocke_attr_set_str(b, &at, "target", target->name);
    return rocke_i_op1(b, ROCKE_OP_ARITH_CAST_F32_TO, operands, 1, target, &at, "cast");
}

rocke_value_t* rocke_b_sitofp_f32(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* operands[1];
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "sitofp_f32 NULL value");
    if(!rocke_i_type_is(v->type, "i32"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "sitofp_f32 expects i32 input, got %s", v->type->name);
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_SITOFP_F32, operands, 1, rocke_f32(), NULL, "sitof");
}

rocke_value_t* rocke_b_cvt_fp8_to_f32(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* operands[1];
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cvt_fp8_to_f32 NULL value");
    if(!rocke_i_type_is(v->type, "fp8e4m3"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "cvt_fp8_to_f32 expects fp8e4m3 input, got %s", v->type->name);
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_CVT_FP8_TO_F32, operands, 1, rocke_f32(), NULL, "dq8");
}

rocke_value_t* rocke_b_cvt_bf8_to_f32(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* operands[1];
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cvt_bf8_to_f32 NULL value");
    if(!rocke_i_type_is(v->type, "bf8e5m2"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "cvt_bf8_to_f32 expects bf8e5m2 input, got %s", v->type->name);
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_CVT_BF8_TO_F32, operands, 1, rocke_f32(), NULL, "dqb8");
}

rocke_value_t* rocke_b_cvt_pk_f32_fp8x4(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* operands[1];
    const rocke_type_t* vt;
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cvt_pk_f32_fp8x4 NULL value");
    if(!rocke_i_is_vector(v->type, "fp8e4m3", 4))
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "cvt_pk_f32_fp8x4 expects vec<fp8e4m3x4> input, got %s",
            v->type->name);
    vt = rocke_vector_type(b, rocke_f32(), 4);
    if(!vt)
        return NULL;
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_CVT_PK_F32_FP8X4, operands, 1, vt, NULL, "dq8x4");
}

rocke_value_t* rocke_b_cvt_pk_f32_bf8x4(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* operands[1];
    const rocke_type_t* vt;
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cvt_pk_f32_bf8x4 NULL value");
    if(!rocke_i_is_vector(v->type, "bf8e5m2", 4))
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "cvt_pk_f32_bf8x4 expects vec<bf8e5m2x4> input, got %s",
            v->type->name);
    vt = rocke_vector_type(b, rocke_f32(), 4);
    if(!vt)
        return NULL;
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_CVT_PK_F32_BF8X4, operands, 1, vt, NULL, "dqb8x4");
}

rocke_value_t*
    rocke_b_cvt_scalef32_pk_f32_fp8x4(rocke_ir_builder_t* b, rocke_value_t* v, rocke_value_t* scale)
{
    rocke_value_t* operands[2];
    const rocke_type_t* vt;
    if(!rocke_i_live(b))
        return NULL;
    if(!v || !scale)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "cvt_scalef32_pk_f32_fp8x4 NULL operand");
    if(!rocke_i_is_vector(v->type, "fp8e4m3", 4))
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "cvt_scalef32_pk_f32_fp8x4 expects vec<fp8e4m3x4>, got %s",
            v->type->name);
    if(!rocke_i_type_is(scale->type, "f32"))
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "cvt_scalef32_pk_f32_fp8x4 scale must be f32, got %s",
            scale->type->name);
    vt = rocke_vector_type(b, rocke_f32(), 4);
    if(!vt)
        return NULL;
    operands[0] = v;
    operands[1] = scale;
    return rocke_i_op1(b, ROCKE_OP_ARITH_CVT_SCALEF32_PK_F32_FP8, operands, 2, vt, NULL, "sdq8x4");
}

rocke_value_t*
    rocke_b_cvt_scalef32_pk_f32_bf8x4(rocke_ir_builder_t* b, rocke_value_t* v, rocke_value_t* scale)
{
    rocke_value_t* operands[2];
    const rocke_type_t* vt;
    if(!rocke_i_live(b))
        return NULL;
    if(!v || !scale)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "cvt_scalef32_pk_f32_bf8x4 NULL operand");
    if(!rocke_i_is_vector(v->type, "bf8e5m2", 4))
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "cvt_scalef32_pk_f32_bf8x4 expects vec<bf8e5m2x4>, got %s",
            v->type->name);
    if(!rocke_i_type_is(scale->type, "f32"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "cvt_scalef32_pk_f32_bf8x4 scale must be f32");
    vt = rocke_vector_type(b, rocke_f32(), 4);
    if(!vt)
        return NULL;
    operands[0] = v;
    operands[1] = scale;
    return rocke_i_op1(b, ROCKE_OP_ARITH_CVT_SCALEF32_PK_F32_BF8, operands, 2, vt, NULL, "sdqb8x4");
}

rocke_value_t* rocke_b_cvt_f32_to_fp8(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* operands[1];
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cvt_f32_to_fp8 NULL value");
    if(!rocke_i_type_is(v->type, "f32"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "cvt_f32_to_fp8 expects f32 input, got %s", v->type->name);
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_CVT_F32_TO_FP8, operands, 1, rocke_fp8e4m3(), NULL, "q8");
}

rocke_value_t* rocke_b_cvt_f32_to_bf8(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* operands[1];
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cvt_f32_to_bf8 NULL value");
    if(!rocke_i_type_is(v->type, "f32"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "cvt_f32_to_bf8 expects f32 input, got %s", v->type->name);
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_CVT_F32_TO_BF8, operands, 1, rocke_bf8e5m2(), NULL, "qb8");
}

rocke_value_t* rocke_b_cvt_f32_to_i8_sat(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* operands[1];
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cvt_f32_to_i8_sat NULL value");
    if(!rocke_i_type_is(v->type, "f32"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "cvt_f32_to_i8_sat expects f32 input, got %s", v->type->name);
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_CVT_F32_TO_I8_SAT, operands, 1, rocke_i8(), NULL, "qi8");
}

rocke_value_t* rocke_b_cvt_pk_fp8_f32x4(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* operands[1];
    const rocke_type_t* vt;
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cvt_pk_fp8_f32x4 NULL value");
    if(!rocke_i_is_vector(v->type, "f32", 4))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "cvt_pk_fp8_f32x4 expects vec<f32x4> input, got %s", v->type->name);
    vt = rocke_vector_type(b, rocke_fp8e4m3(), 4);
    if(!vt)
        return NULL;
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_CVT_PK_FP8_F32X4, operands, 1, vt, NULL, "q8x4");
}

rocke_value_t* rocke_b_cvt_pk_bf8_f32x4(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* operands[1];
    const rocke_type_t* vt;
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cvt_pk_bf8_f32x4 NULL value");
    if(!rocke_i_is_vector(v->type, "f32", 4))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "cvt_pk_bf8_f32x4 expects vec<f32x4> input, got %s", v->type->name);
    vt = rocke_vector_type(b, rocke_bf8e5m2(), 4);
    if(!vt)
        return NULL;
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_CVT_PK_BF8_F32X4, operands, 1, vt, NULL, "qb8x4");
}

rocke_value_t* rocke_b_cvt_pk_i8_f32x4(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* operands[1];
    const rocke_type_t* vt;
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cvt_pk_i8_f32x4 NULL value");
    if(!rocke_i_is_vector(v->type, "f32", 4))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "cvt_pk_i8_f32x4 expects vec<f32x4> input, got %s", v->type->name);
    vt = rocke_vector_type(b, rocke_i8(), 4);
    if(!vt)
        return NULL;
    operands[0] = v;
    return rocke_i_op1(b, ROCKE_OP_ARITH_CVT_PK_I8_F32X4, operands, 1, vt, NULL, "qi8x4");
}
