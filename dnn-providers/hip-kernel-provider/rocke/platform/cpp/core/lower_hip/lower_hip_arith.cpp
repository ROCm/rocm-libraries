// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_hip_lower_hip_arith.c -- C99 port of rocke.core.lower_hip, BUCKET 0
 * (arith): scalar/int/float/cmp/select/bitwise ARITH ops + the transcendental
 * MATH ops (exp2/log2/rcp/rcp_fast/sqrt/rsqrt/tanh).
 *
 * Each `_op_*` Python method becomes a static `rocke_h_op_*` handler with the
 * (lw, op) signature from lower_hip_internal.h. Shared helpers (rocke_h_emit /
 * rocke_h_emitf / rocke_h_name / rocke_h_type_to_hip / rocke_h_hip_scalar /
 * rocke_h_f32_literal / rocke_attr_get_*) are DEFINED in lower_hip_core.c and only
 * called here. The registration table is exported via rocke_h_handlers_arith(),
 * which the core bucket stitches into the dispatch table.
 *
 * Output text is byte-identical to the Python lowerer: every _emit() format
 * string is reproduced exactly (Python self._emit adds the indent prefix; here
 * rocke_h_emit/rocke_h_emitf does the same).
 */
#include <stdio.h> /* snprintf */
#include <string.h> /* strcmp   */

#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_hip_internal.h"

namespace ckc
{

/* Convenience: the single result Value of `op` (Python op.result). Every handler
 * in this bucket produces exactly one result, mirroring the Python @property
 * which asserts a single result. */
static const rocke_value_t* h_res(const rocke_op_t* op)
{
    return op->results[0];
}

/* The Python `_binary` helper:
 *   def _binary(self, op, c_op):
 *       a, b = op.operands
 *       self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
 *                  f"{_name(a)} {c_op} {_name(b)};")
 * Shared by add/sub/mul/div/mod and the fadd/fsub/fmul/fdiv/xor/shl floats. */
static void h_binary(rocke_h_lowerer_t* lw, const rocke_op_t* op, const char* c_op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw,
                  "%s %s = %s %s %s;",
                  rocke_h_type_to_hip(lw, r->type),
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, a),
                  c_op,
                  rocke_h_name(lw, b));
}

/* ----------------------------- arith: constants ---------------------------- */

/* def _op_arith_constant(self, op):
 *     res = op.result
 *     ity = op.attrs.get("ity", "i32")
 *     val = op.attrs["value"]
 *     cpp_t = _HIP_TYPE[ity]
 *     if ity in ("f16", "f32"):
 *         literal = _f32_literal(float(val))
 *         if ity == "f16":
 *             self._emit(f"{cpp_t} {_name(res)} = (fp16){literal};")
 *         else:
 *             self._emit(f"{cpp_t} {_name(res)} = {literal};")
 *     else:
 *         self._emit(f"{cpp_t} {_name(res)} = {val};") */
static rocke_status_t rocke_h_op_arith_constant(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* res = h_res(op);
    const char* ity = rocke_attr_get_str(&op->attrs, "ity");
    const char* cpp_t;
    const rocke_attr_value_t* val;
    if(!ity)
    {
        ity = "i32";
    }
    cpp_t = rocke_h_hip_scalar(ity);
    if(!cpp_t)
    {
        return rocke_h_fail(lw, ROCKE_ERR_KEY, "arith.constant: unknown ity %s", ity);
    }
    val = rocke_attr_get(&op->attrs, "value");
    if(!val)
    {
        return rocke_h_fail(lw, ROCKE_ERR_KEY, "arith.constant: missing value");
    }
    if(ity[0] == 'f' && (ity[1] == '1' /* f16 */ || ity[1] == '3' /* f32 */))
    {
        /* float-valued constant: emit through _f32_literal. The attr stores
         * the value either as a double (ROCKE_ATTR_FLOAT) or, if it was an
         * integral literal, as an int (ROCKE_ATTR_INT) -- float(val) in Python. */
        double v = (val->kind == ROCKE_ATTR_FLOAT) ? val->u.f
                   : (val->kind == ROCKE_ATTR_INT) ? (double)val->u.i
                                                   : 0.0;
        const char* literal = rocke_h_f32_literal(lw, v);
        if(ity[1] == '1')
        { /* f16 */
            rocke_h_emitf(lw, "%s %s = (fp16)%s;", cpp_t, rocke_h_name(lw, res), literal);
        }
        else
        { /* f32 */
            rocke_h_emitf(lw, "%s %s = %s;", cpp_t, rocke_h_name(lw, res), literal);
        }
    }
    else
    {
        /* integer constant: Python emits the raw `val` (an int). */
        rocke_h_emitf(lw,
                      "%s %s = %lld;",
                      cpp_t,
                      rocke_h_name(lw, res),
                      (long long)(val->kind == ROCKE_ATTR_INT     ? val->u.i
                                  : val->kind == ROCKE_ATTR_FLOAT ? (int64_t)val->u.f
                                                                  : 0));
    }
    return lw->status;
}

/* def _op_arith_constant_vec(self, op):
 *     res = op.result
 *     fill = op.attrs.get("fill", 0.0)
 *     if not isinstance(res.type, VectorType):
 *         raise NotImplementedError("constant_vec result must be a vector")
 *     count = res.type.count
 *     cpp_t = _type_to_hip(res.type)
 *     elem_name = res.type.elem.name
 *     if elem_name in ("f16", "bf16", "f32"):
 *         item = _f32_literal(float(fill))
 *     else:
 *         item = str(int(fill))
 *     items = ", ".join(item for _ in range(count))
 *     self._emit(f"{cpp_t} {_name(res)} = {{{items}}};") */
static rocke_status_t rocke_h_op_arith_constant_vec(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* res = h_res(op);
    const rocke_attr_value_t* fill_a = rocke_attr_get(&op->attrs, "fill");
    double fill = 0.0; /* Python default fill=0.0 */
    int count, i;
    const char* cpp_t;
    const char* elem_name;
    const char* item;
    /* StrBuf-free assembly: we build the "{a, b, ...}" body manually via repeated
     * appends; rocke_h_emitf builds the final statement. Use a small dynamic-ish
     * approach over the arena-backed strbuf is unavailable here, so build into a
     * fixed scratch and emit. The item text is bounded by _f32_literal / an int
     * spelling and `count` is small (vector lane counts), so a stack buffer is
     * sufficient and matches the Python join exactly. */
    char body[1024];
    size_t pos = 0;

    if(fill_a)
    {
        fill = (fill_a->kind == ROCKE_ATTR_FLOAT) ? fill_a->u.f
               : (fill_a->kind == ROCKE_ATTR_INT) ? (double)fill_a->u.i
                                                  : 0.0;
    }
    if(res->type->kind != ROCKE_TYPE_VECTOR)
    {
        return rocke_h_fail(lw, ROCKE_ERR_NOTIMPL, "constant_vec result must be a vector");
    }
    count = res->type->count;
    cpp_t = rocke_h_type_to_hip(lw, res->type);
    elem_name = res->type->elem->name;

    if(elem_name
       && (strcmp(elem_name, "f16") == 0 || strcmp(elem_name, "bf16") == 0
           || strcmp(elem_name, "f32") == 0))
    {
        item = rocke_h_f32_literal(lw, fill);
    }
    else
    {
        /* str(int(fill)) -- truncate toward zero like Python int(). */
        static char ibuf[32];
        snprintf(ibuf, sizeof(ibuf), "%lld", (long long)(int64_t)fill);
        item = ibuf;
    }

    body[0] = '\0';
    for(i = 0; i < count; i++)
    {
        int n;
        if(i > 0)
        {
            n = snprintf(body + pos, sizeof(body) - pos, ", %s", item);
        }
        else
        {
            n = snprintf(body + pos, sizeof(body) - pos, "%s", item);
        }
        if(n < 0 || (size_t)n >= sizeof(body) - pos)
        {
            return rocke_h_fail(lw, ROCKE_ERR_VALUE, "constant_vec: too many lanes to format");
        }
        pos += (size_t)n;
    }
    rocke_h_emitf(lw, "%s %s = {%s};", cpp_t, rocke_h_name(lw, res), body);
    return lw->status;
}

/* ----------------------------- arith: int binary --------------------------- */

/* def _op_arith_add(self, op): self._binary(op, "+") */
static rocke_status_t rocke_h_op_arith_add(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_binary(lw, op, "+");
    return lw->status;
}

/* def _op_arith_sub(self, op): self._binary(op, "-") */
static rocke_status_t rocke_h_op_arith_sub(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_binary(lw, op, "-");
    return lw->status;
}

/* def _op_arith_mul(self, op): self._binary(op, "*") */
static rocke_status_t rocke_h_op_arith_mul(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_binary(lw, op, "*");
    return lw->status;
}

/* def _op_arith_div(self, op): self._binary(op, "/") */
static rocke_status_t rocke_h_op_arith_div(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_binary(lw, op, "/");
    return lw->status;
}

/* def _op_arith_mod(self, op): self._binary(op, "%") */
static rocke_status_t rocke_h_op_arith_mod(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_binary(lw, op, "%");
    return lw->status;
}

/* ----------------------------- arith: cmp / select ------------------------- */

/* def _op_arith_cmp(self, op):
 *     pred = op.attrs.get("pred", "lt")
 *     c_op = {"lt":"<","le":"<=","gt":">","ge":">=","eq":"==","ne":"!="}[pred]
 *     a, b = op.operands
 *     self._emit(f"bool {_name(op.result)} = {_name(a)} {c_op} {_name(b)};") */
static rocke_status_t rocke_h_op_arith_cmp(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const char* pred = rocke_attr_get_str(&op->attrs, "pred");
    const char* c_op;
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    if(!pred)
    {
        pred = "lt";
    }
    if(strcmp(pred, "lt") == 0)
    {
        c_op = "<";
    }
    else if(strcmp(pred, "le") == 0)
    {
        c_op = "<=";
    }
    else if(strcmp(pred, "gt") == 0)
    {
        c_op = ">";
    }
    else if(strcmp(pred, "ge") == 0)
    {
        c_op = ">=";
    }
    else if(strcmp(pred, "eq") == 0)
    {
        c_op = "==";
    }
    else if(strcmp(pred, "ne") == 0)
    {
        c_op = "!=";
    }
    else
    {
        return rocke_h_fail(lw, ROCKE_ERR_KEY, "arith.cmp: unknown pred %s", pred);
    }
    rocke_h_emitf(lw,
                  "bool %s = %s %s %s;",
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, a),
                  c_op,
                  rocke_h_name(lw, b));
    return lw->status;
}

/* def _op_arith_select(self, op):
 *     cond, lhs, rhs = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
 *                f"{_name(cond)} ? {_name(lhs)} : {_name(rhs)};") */
static rocke_status_t rocke_h_op_arith_select(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* cond = op->operands[0];
    const rocke_value_t* lhs = op->operands[1];
    const rocke_value_t* rhs = op->operands[2];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw,
                  "%s %s = %s ? %s : %s;",
                  rocke_h_type_to_hip(lw, r->type),
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, cond),
                  rocke_h_name(lw, lhs),
                  rocke_h_name(lw, rhs));
    return lw->status;
}

/* ----------------------------- arith: bitwise ------------------------------ */

/* def _op_arith_and(self, op):
 *     a, b = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
 *                f"{_name(a)} & {_name(b)};") */
static rocke_status_t rocke_h_op_arith_and(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw,
                  "%s %s = %s & %s;",
                  rocke_h_type_to_hip(lw, r->type),
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, a),
                  rocke_h_name(lw, b));
    return lw->status;
}

/* def _op_arith_or(self, op):
 *     a, b = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
 *                f"{_name(a)} | {_name(b)};") */
static rocke_status_t rocke_h_op_arith_or(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw,
                  "%s %s = %s | %s;",
                  rocke_h_type_to_hip(lw, r->type),
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, a),
                  rocke_h_name(lw, b));
    return lw->status;
}

/* def _op_arith_smax(self, op):
 *     a, b = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
 *                f"({_name(a)} > {_name(b)} ? {_name(a)} : {_name(b)});") */
static rocke_status_t rocke_h_op_arith_smax(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    const char* an = rocke_h_name(lw, a);
    const char* bn = rocke_h_name(lw, b);
    rocke_h_emitf(lw,
                  "%s %s = (%s > %s ? %s : %s);",
                  rocke_h_type_to_hip(lw, r->type),
                  rocke_h_name(lw, r),
                  an,
                  bn,
                  an,
                  bn);
    return lw->status;
}

/* def _op_arith_smin(self, op):
 *     a, b = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
 *                f"({_name(a)} < {_name(b)} ? {_name(a)} : {_name(b)});") */
static rocke_status_t rocke_h_op_arith_smin(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    const char* an = rocke_h_name(lw, a);
    const char* bn = rocke_h_name(lw, b);
    rocke_h_emitf(lw,
                  "%s %s = (%s < %s ? %s : %s);",
                  rocke_h_type_to_hip(lw, r->type),
                  rocke_h_name(lw, r),
                  an,
                  bn,
                  an,
                  bn);
    return lw->status;
}

/* def _op_arith_not(self, op):
 *     (v,) = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = ~{_name(v)};") */
static rocke_status_t rocke_h_op_arith_not(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw,
                  "%s %s = ~%s;",
                  rocke_h_type_to_hip(lw, r->type),
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, v));
    return lw->status;
}

/* def _op_arith_xor(self, op): self._binary(op, "^") */
static rocke_status_t rocke_h_op_arith_xor(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_binary(lw, op, "^");
    return lw->status;
}

/* def _op_arith_shl(self, op): self._binary(op, "<<") */
static rocke_status_t rocke_h_op_arith_shl(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_binary(lw, op, "<<");
    return lw->status;
}

/* def _op_arith_lshr(self, op):
 *     a, b = op.operands
 *     self._emit(f"int {_name(op.result)} = "
 *                f"(int)((unsigned)({_name(a)}) >> {_name(b)});") */
static rocke_status_t rocke_h_op_arith_lshr(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw,
                  "int %s = (int)((unsigned)(%s) >> %s);",
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, a),
                  rocke_h_name(lw, b));
    return lw->status;
}

/* def _op_arith_umul_hi_i32(self, op):
 *     a, b = op.operands
 *     self._emit(f"int {_name(op.result)} = "
 *                f"(int)__umulhi((unsigned){_name(a)}, (unsigned){_name(b)});") */
static rocke_status_t rocke_h_op_arith_umul_hi_i32(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw,
                  "int %s = (int)__umulhi((unsigned)%s, (unsigned)%s);",
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, a),
                  rocke_h_name(lw, b));
    return lw->status;
}

/* ----------------------------- arith: float binary ------------------------- */

/* def _op_arith_fadd(self, op): self._binary(op, "+") */
static rocke_status_t rocke_h_op_arith_fadd(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_binary(lw, op, "+");
    return lw->status;
}

/* def _op_arith_fsub(self, op): self._binary(op, "-") */
static rocke_status_t rocke_h_op_arith_fsub(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_binary(lw, op, "-");
    return lw->status;
}

/* def _op_arith_fmul(self, op): self._binary(op, "*") */
static rocke_status_t rocke_h_op_arith_fmul(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_binary(lw, op, "*");
    return lw->status;
}

/* def _op_arith_fdiv(self, op): self._binary(op, "/") */
static rocke_status_t rocke_h_op_arith_fdiv(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_binary(lw, op, "/");
    return lw->status;
}

/* def _op_arith_fneg(self, op):
 *     (v,) = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = -{_name(v)};") */
static rocke_status_t rocke_h_op_arith_fneg(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw,
                  "%s %s = -%s;",
                  rocke_h_type_to_hip(lw, r->type),
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, v));
    return lw->status;
}

/* def _op_arith_fabs(self, op):
 *     (v,) = op.operands
 *     ty = _type_to_hip(op.result.type)
 *     helper = {"f32":"fabsf","f16":"__builtin_fabsf","bf16":"__builtin_fabsf"}
 *              .get(op.result.type.name, "fabsf")
 *     self._emit(f"{ty} {_name(op.result)} = ({ty}){helper}((float){_name(v)});") */
static rocke_status_t rocke_h_op_arith_fabs(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* ty = rocke_h_type_to_hip(lw, r->type);
    const char* tname = r->type->name;
    const char* helper = "fabsf";
    if(tname && (strcmp(tname, "f16") == 0 || strcmp(tname, "bf16") == 0))
    {
        helper = "__builtin_fabsf";
    }
    rocke_h_emitf(
        lw, "%s %s = (%s)%s((float)%s);", ty, rocke_h_name(lw, r), ty, helper, rocke_h_name(lw, v));
    return lw->status;
}

/* def _op_arith_fma(self, op):
 *     a, b, c = op.operands
 *     ty = _type_to_hip(op.result.type)
 *     self._emit(f"{ty} {_name(op.result)} = ({ty})fmaf("
 *                f"(float){_name(a)}, (float){_name(b)}, (float){_name(c)});") */
static rocke_status_t rocke_h_op_arith_fma(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* c = op->operands[2];
    const rocke_value_t* r = h_res(op);
    const char* ty = rocke_h_type_to_hip(lw, r->type);
    rocke_h_emitf(lw,
                  "%s %s = (%s)fmaf((float)%s, (float)%s, (float)%s);",
                  ty,
                  rocke_h_name(lw, r),
                  ty,
                  rocke_h_name(lw, a),
                  rocke_h_name(lw, b),
                  rocke_h_name(lw, c));
    return lw->status;
}

/* def _op_arith_fmax3(self, op):
 *     a, b, c = op.operands
 *     ty = _type_to_hip(op.result.type)
 *     self._emit(f"{ty} {_name(op.result)} = "
 *                f"(({_name(b)} > {_name(c)}) ? {_name(b)} : {_name(c)});")
 *     self._emit(f"{_name(op.result)} = "
 *                f"({_name(a)} > {_name(op.result)}) ? {_name(a)} : {_name(op.result)};") */
static rocke_status_t rocke_h_op_arith_fmax3(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* c = op->operands[2];
    const rocke_value_t* r = h_res(op);
    const char* ty = rocke_h_type_to_hip(lw, r->type);
    const char* rn = rocke_h_name(lw, r);
    const char* an = rocke_h_name(lw, a);
    const char* bn = rocke_h_name(lw, b);
    const char* cn = rocke_h_name(lw, c);
    rocke_h_emitf(lw, "%s %s = ((%s > %s) ? %s : %s);", ty, rn, bn, cn, bn, cn);
    rocke_h_emitf(lw, "%s = (%s > %s) ? %s : %s;", rn, an, rn, an, rn);
    return lw->status;
}

/* def _op_arith_fmin3(self, op):
 *     a, b, c = op.operands
 *     ty = _type_to_hip(op.result.type)
 *     self._emit(f"{ty} {_name(op.result)} = "
 *                f"(({_name(b)} < {_name(c)}) ? {_name(b)} : {_name(c)});")
 *     self._emit(f"{_name(op.result)} = "
 *                f"({_name(a)} < {_name(op.result)}) ? {_name(a)} : {_name(op.result)};") */
static rocke_status_t rocke_h_op_arith_fmin3(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* c = op->operands[2];
    const rocke_value_t* r = h_res(op);
    const char* ty = rocke_h_type_to_hip(lw, r->type);
    const char* rn = rocke_h_name(lw, r);
    const char* an = rocke_h_name(lw, a);
    const char* bn = rocke_h_name(lw, b);
    const char* cn = rocke_h_name(lw, c);
    rocke_h_emitf(lw, "%s %s = ((%s < %s) ? %s : %s);", ty, rn, bn, cn, bn, cn);
    rocke_h_emitf(lw, "%s = (%s < %s) ? %s : %s;", rn, an, rn, an, rn);
    return lw->status;
}

/* def _op_arith_fmax(self, op):
 *     a, b = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
 *                f"({_name(a)} > {_name(b)}) ? {_name(a)} : {_name(b)};") */
static rocke_status_t rocke_h_op_arith_fmax(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    const char* an = rocke_h_name(lw, a);
    const char* bn = rocke_h_name(lw, b);
    rocke_h_emitf(lw,
                  "%s %s = (%s > %s) ? %s : %s;",
                  rocke_h_type_to_hip(lw, r->type),
                  rocke_h_name(lw, r),
                  an,
                  bn,
                  an,
                  bn);
    return lw->status;
}

/* def _op_arith_fmin(self, op):
 *     a, b = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
 *                f"({_name(a)} < {_name(b)}) ? {_name(a)} : {_name(b)};") */
static rocke_status_t rocke_h_op_arith_fmin(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    const char* an = rocke_h_name(lw, a);
    const char* bn = rocke_h_name(lw, b);
    rocke_h_emitf(lw,
                  "%s %s = (%s < %s) ? %s : %s;",
                  rocke_h_type_to_hip(lw, r->type),
                  rocke_h_name(lw, r),
                  an,
                  bn,
                  an,
                  bn);
    return lw->status;
}

/* def _op_arith_fcmp(self, op):
 *     pred = op.attrs["pred"]
 *     a, b = op.operands
 *     op_map = {"olt":"<","ole":"<=","ogt":">","oge":">=","oeq":"==","one":"!="}
 *     if pred in op_map:
 *         self._emit(f"bool {_name(op.result)} = "
 *                    f"(!isnan(float({_name(a)})) && !isnan(float({_name(b)})) "
 *                    f"&& ({_name(a)} {op_map[pred]} {_name(b)}));")
 *     elif pred == "ord":
 *         self._emit(f"bool {_name(op.result)} = "
 *                    f"(!isnan(float({_name(a)})) && !isnan(float({_name(b)})));")
 *     elif pred == "uno":
 *         self._emit(f"bool {_name(op.result)} = "
 *                    f"(isnan(float({_name(a)})) || isnan(float({_name(b)})));")
 *     else:
 *         raise NotImplementedError(f"unknown fcmp predicate {pred!r}") */
static rocke_status_t rocke_h_op_arith_fcmp(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const char* pred = rocke_attr_get_str(&op->attrs, "pred");
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    const char* rn = rocke_h_name(lw, r);
    const char* an = rocke_h_name(lw, a);
    const char* bn = rocke_h_name(lw, b);
    const char* cop = NULL;
    if(!pred)
    {
        return rocke_h_fail(lw, ROCKE_ERR_KEY, "arith.fcmp: missing pred");
    }
    if(strcmp(pred, "olt") == 0)
    {
        cop = "<";
    }
    else if(strcmp(pred, "ole") == 0)
    {
        cop = "<=";
    }
    else if(strcmp(pred, "ogt") == 0)
    {
        cop = ">";
    }
    else if(strcmp(pred, "oge") == 0)
    {
        cop = ">=";
    }
    else if(strcmp(pred, "oeq") == 0)
    {
        cop = "==";
    }
    else if(strcmp(pred, "one") == 0)
    {
        cop = "!=";
    }
    if(cop)
    {
        rocke_h_emitf(lw,
                      "bool %s = (!isnan(float(%s)) && !isnan(float(%s)) "
                      "&& (%s %s %s));",
                      rn,
                      an,
                      bn,
                      an,
                      cop,
                      bn);
    }
    else if(strcmp(pred, "ord") == 0)
    {
        rocke_h_emitf(lw, "bool %s = (!isnan(float(%s)) && !isnan(float(%s)));", rn, an, bn);
    }
    else if(strcmp(pred, "uno") == 0)
    {
        rocke_h_emitf(lw, "bool %s = (isnan(float(%s)) || isnan(float(%s)));", rn, an, bn);
    }
    else
    {
        return rocke_h_fail(lw, ROCKE_ERR_NOTIMPL, "unknown fcmp predicate '%s'", pred);
    }
    return lw->status;
}

/* ----------------------------- math (transcendentals) --------------------- */

/* The Python `_math1` helper:
 *   def _math1(self, op, fn_f32, *, prefer_amdgcn_builtin=False):
 *       (v,) = op.operands
 *       tname = op.result.type.name
 *       cpp_t = _type_to_hip(op.result.type)
 *       if tname == "f32":
 *           self._emit(f"{cpp_t} {_name(op.result)} = {fn_f32}({_name(v)});")
 *       else:
 *           self._emit(f"{cpp_t} {_name(op.result)} = ({cpp_t}){fn_f32}((float){_name(v)});")
 * Shared by exp2/log2/sqrt/tanh. */
static void h_math1(rocke_h_lowerer_t* lw, const rocke_op_t* op, const char* fn_f32)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* tname = r->type->name;
    const char* cpp_t = rocke_h_type_to_hip(lw, r->type);
    const char* rn = rocke_h_name(lw, r);
    const char* vn = rocke_h_name(lw, v);
    if(tname && strcmp(tname, "f32") == 0)
    {
        rocke_h_emitf(lw, "%s %s = %s(%s);", cpp_t, rn, fn_f32, vn);
    }
    else
    {
        rocke_h_emitf(lw, "%s %s = (%s)%s((float)%s);", cpp_t, rn, cpp_t, fn_f32, vn);
    }
}

/* An amdgcn-builtin reciprocal-style math op (rcp / rcp_fast / rsqrt). Same
 * promote-compute-demote shape as _math1 but the f32 path drops the (cpp_t)
 * cast since the builtin already returns float:
 *   if tname == "f32":
 *       self._emit(f"{cpp_t} {_name} = {builtin}({_name(v)});")
 *   else:
 *       self._emit(f"{cpp_t} {_name} = ({cpp_t}){builtin}((float){_name(v)});") */
static void h_amdgcn_unary(rocke_h_lowerer_t* lw, const rocke_op_t* op, const char* builtin)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* tname = r->type->name;
    const char* cpp_t = rocke_h_type_to_hip(lw, r->type);
    const char* rn = rocke_h_name(lw, r);
    const char* vn = rocke_h_name(lw, v);
    if(tname && strcmp(tname, "f32") == 0)
    {
        rocke_h_emitf(lw, "%s %s = %s(%s);", cpp_t, rn, builtin, vn);
    }
    else
    {
        rocke_h_emitf(lw, "%s %s = (%s)%s((float)%s);", cpp_t, rn, cpp_t, builtin, vn);
    }
}

/* def _op_math_exp2(self, op): self._math1(op, "exp2f") */
static rocke_status_t rocke_h_op_math_exp2(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_math1(lw, op, "exp2f");
    return lw->status;
}

/* def _op_math_log2(self, op): self._math1(op, "log2f") */
static rocke_status_t rocke_h_op_math_log2(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_math1(lw, op, "log2f");
    return lw->status;
}

/* def _op_math_rcp(self, op): -- __builtin_amdgcn_rcpf, promote/demote else. */
static rocke_status_t rocke_h_op_math_rcp(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_amdgcn_unary(lw, op, "__builtin_amdgcn_rcpf");
    return lw->status;
}

/* def _op_math_rcp_fast(self, op): -- identical to math.rcp on HIP. */
static rocke_status_t rocke_h_op_math_rcp_fast(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_amdgcn_unary(lw, op, "__builtin_amdgcn_rcpf");
    return lw->status;
}

/* def _op_math_sqrt(self, op): self._math1(op, "sqrtf") */
static rocke_status_t rocke_h_op_math_sqrt(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_math1(lw, op, "sqrtf");
    return lw->status;
}

/* def _op_math_rsqrt(self, op): -- __builtin_amdgcn_rsqf, promote/demote else. */
static rocke_status_t rocke_h_op_math_rsqrt(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_amdgcn_unary(lw, op, "__builtin_amdgcn_rsqf");
    return lw->status;
}

/* def _op_math_tanh(self, op): self._math1(op, "tanhf") */
static rocke_status_t rocke_h_op_math_tanh(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    h_math1(lw, op, "tanhf");
    return lw->status;
}

/* ----------------------------- registration table -------------------------- */

const rocke_h_handler_entry_t* rocke_h_handlers_arith(void)
{
    static const rocke_h_handler_entry_t table[] = {
        {ROCKE_OP_ARITH_CONSTANT, rocke_h_op_arith_constant},
        {ROCKE_OP_ARITH_CONSTANT_VEC, rocke_h_op_arith_constant_vec},
        {ROCKE_OP_ARITH_ADD, rocke_h_op_arith_add},
        {ROCKE_OP_ARITH_SUB, rocke_h_op_arith_sub},
        {ROCKE_OP_ARITH_MUL, rocke_h_op_arith_mul},
        {ROCKE_OP_ARITH_DIV, rocke_h_op_arith_div},
        {ROCKE_OP_ARITH_MOD, rocke_h_op_arith_mod},
        {ROCKE_OP_ARITH_CMP, rocke_h_op_arith_cmp},
        {ROCKE_OP_ARITH_SELECT, rocke_h_op_arith_select},
        {ROCKE_OP_ARITH_AND, rocke_h_op_arith_and},
        {ROCKE_OP_ARITH_OR, rocke_h_op_arith_or},
        {ROCKE_OP_ARITH_SMAX, rocke_h_op_arith_smax},
        {ROCKE_OP_ARITH_SMIN, rocke_h_op_arith_smin},
        {ROCKE_OP_ARITH_NOT, rocke_h_op_arith_not},
        {ROCKE_OP_ARITH_XOR, rocke_h_op_arith_xor},
        {ROCKE_OP_ARITH_SHL, rocke_h_op_arith_shl},
        {ROCKE_OP_ARITH_LSHR, rocke_h_op_arith_lshr},
        {ROCKE_OP_ARITH_UMUL_HI_I32, rocke_h_op_arith_umul_hi_i32},
        {ROCKE_OP_ARITH_FADD, rocke_h_op_arith_fadd},
        {ROCKE_OP_ARITH_FSUB, rocke_h_op_arith_fsub},
        {ROCKE_OP_ARITH_FMUL, rocke_h_op_arith_fmul},
        {ROCKE_OP_ARITH_FDIV, rocke_h_op_arith_fdiv},
        {ROCKE_OP_ARITH_FNEG, rocke_h_op_arith_fneg},
        {ROCKE_OP_ARITH_FABS, rocke_h_op_arith_fabs},
        {ROCKE_OP_ARITH_FMA, rocke_h_op_arith_fma},
        {ROCKE_OP_ARITH_FMAX3, rocke_h_op_arith_fmax3},
        {ROCKE_OP_ARITH_FMIN3, rocke_h_op_arith_fmin3},
        {ROCKE_OP_ARITH_FMAX, rocke_h_op_arith_fmax},
        {ROCKE_OP_ARITH_FMIN, rocke_h_op_arith_fmin},
        {ROCKE_OP_ARITH_FCMP, rocke_h_op_arith_fcmp},
        {ROCKE_OP_MATH_EXP2, rocke_h_op_math_exp2},
        {ROCKE_OP_MATH_LOG2, rocke_h_op_math_log2},
        {ROCKE_OP_MATH_RCP, rocke_h_op_math_rcp},
        {ROCKE_OP_MATH_RCP_FAST, rocke_h_op_math_rcp_fast},
        {ROCKE_OP_MATH_SQRT, rocke_h_op_math_sqrt},
        {ROCKE_OP_MATH_RSQRT, rocke_h_op_math_rsqrt},
        {ROCKE_OP_MATH_TANH, rocke_h_op_math_tanh},
        {ROCKE_OP_INVALID, NULL}, /* terminator */
    };
    return table;
}

} /* namespace ckc */
