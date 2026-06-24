// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_hip_lower_hip_arith.c -- C99 port of ck_dsl.core.lower_hip, BUCKET 0
 * (arith): scalar/int/float/cmp/select/bitwise ARITH ops + the transcendental
 * MATH ops (exp2/log2/rcp/rcp_fast/sqrt/rsqrt/tanh).
 *
 * Each `_op_*` Python method becomes a static `ckc_h_op_*` handler with the
 * (lw, op) signature from lower_hip_internal.h. Shared helpers (ckc_h_emit /
 * ckc_h_emitf / ckc_h_name / ckc_h_type_to_hip / ckc_h_hip_scalar /
 * ckc_h_f32_literal / ckc_attr_get_*) are DEFINED in lower_hip_core.c and only
 * called here. The registration table is exported via ckc_h_handlers_arith(),
 * which the core bucket stitches into the dispatch table.
 *
 * Output text is byte-identical to the Python lowerer: every _emit() format
 * string is reproduced exactly (Python self._emit adds the indent prefix; here
 * ckc_h_emit/ckc_h_emitf does the same).
 */
#include <stdio.h>  /* snprintf */
#include <string.h> /* strcmp   */

#include "ckc/ir.h"
#include "ckc/lower_hip.h"
#include "ckc/lower_hip_internal.h"

namespace ckc {

/* Convenience: the single result Value of `op` (Python op.result). Every handler
 * in this bucket produces exactly one result, mirroring the Python @property
 * which asserts a single result. */
static const ckc_value_t* h_res(const ckc_op_t* op) { return op->results[0]; }

/* The Python `_binary` helper:
 *   def _binary(self, op, c_op):
 *       a, b = op.operands
 *       self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
 *                  f"{_name(a)} {c_op} {_name(b)};")
 * Shared by add/sub/mul/div/mod and the fadd/fsub/fmul/fdiv/xor/shl floats. */
static void h_binary(ckc_h_lowerer_t* lw, const ckc_op_t* op, const char* c_op)
{
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    const ckc_value_t* r = h_res(op);
    ckc_h_emitf(lw,
                "%s %s = %s %s %s;",
                ckc_h_type_to_hip(lw, r->type),
                ckc_h_name(lw, r),
                ckc_h_name(lw, a),
                c_op,
                ckc_h_name(lw, b));
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
static ckc_status_t ckc_h_op_arith_constant(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* res = h_res(op);
    const char* ity        = ckc_attr_get_str(&op->attrs, "ity");
    const char* cpp_t;
    const ckc_attr_value_t* val;
    if(!ity)
    {
        ity = "i32";
    }
    cpp_t = ckc_h_hip_scalar(ity);
    if(!cpp_t)
    {
        return ckc_h_fail(lw, CKC_ERR_KEY, "arith.constant: unknown ity %s", ity);
    }
    val = ckc_attr_get(&op->attrs, "value");
    if(!val)
    {
        return ckc_h_fail(lw, CKC_ERR_KEY, "arith.constant: missing value");
    }
    if(ity[0] == 'f' && (ity[1] == '1' /* f16 */ || ity[1] == '3' /* f32 */))
    {
        /* float-valued constant: emit through _f32_literal. The attr stores
         * the value either as a double (CKC_ATTR_FLOAT) or, if it was an
         * integral literal, as an int (CKC_ATTR_INT) -- float(val) in Python. */
        double v            = (val->kind == CKC_ATTR_FLOAT) ? val->u.f
                              : (val->kind == CKC_ATTR_INT) ? (double)val->u.i
                                                            : 0.0;
        const char* literal = ckc_h_f32_literal(lw, v);
        if(ity[1] == '1')
        { /* f16 */
            ckc_h_emitf(lw, "%s %s = (fp16)%s;", cpp_t, ckc_h_name(lw, res), literal);
        }
        else
        { /* f32 */
            ckc_h_emitf(lw, "%s %s = %s;", cpp_t, ckc_h_name(lw, res), literal);
        }
    }
    else
    {
        /* integer constant: Python emits the raw `val` (an int). */
        ckc_h_emitf(lw,
                    "%s %s = %lld;",
                    cpp_t,
                    ckc_h_name(lw, res),
                    (long long)(val->kind == CKC_ATTR_INT     ? val->u.i
                                : val->kind == CKC_ATTR_FLOAT ? (int64_t)val->u.f
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
static ckc_status_t ckc_h_op_arith_constant_vec(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* res         = h_res(op);
    const ckc_attr_value_t* fill_a = ckc_attr_get(&op->attrs, "fill");
    double fill                    = 0.0; /* Python default fill=0.0 */
    int count, i;
    const char* cpp_t;
    const char* elem_name;
    const char* item;
    /* StrBuf-free assembly: we build the "{a, b, ...}" body manually via repeated
     * appends; ckc_h_emitf builds the final statement. Use a small dynamic-ish
     * approach over the arena-backed strbuf is unavailable here, so build into a
     * fixed scratch and emit. The item text is bounded by _f32_literal / an int
     * spelling and `count` is small (vector lane counts), so a stack buffer is
     * sufficient and matches the Python join exactly. */
    char body[1024];
    size_t pos = 0;

    if(fill_a)
    {
        fill = (fill_a->kind == CKC_ATTR_FLOAT) ? fill_a->u.f
               : (fill_a->kind == CKC_ATTR_INT) ? (double)fill_a->u.i
                                                : 0.0;
    }
    if(res->type->kind != CKC_TYPE_VECTOR)
    {
        return ckc_h_fail(lw, CKC_ERR_NOTIMPL, "constant_vec result must be a vector");
    }
    count     = res->type->count;
    cpp_t     = ckc_h_type_to_hip(lw, res->type);
    elem_name = res->type->elem->name;

    if(elem_name && (strcmp(elem_name, "f16") == 0 || strcmp(elem_name, "bf16") == 0 ||
                     strcmp(elem_name, "f32") == 0))
    {
        item = ckc_h_f32_literal(lw, fill);
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
            return ckc_h_fail(lw, CKC_ERR_VALUE, "constant_vec: too many lanes to format");
        }
        pos += (size_t)n;
    }
    ckc_h_emitf(lw, "%s %s = {%s};", cpp_t, ckc_h_name(lw, res), body);
    return lw->status;
}

/* ----------------------------- arith: int binary --------------------------- */

/* def _op_arith_add(self, op): self._binary(op, "+") */
static ckc_status_t ckc_h_op_arith_add(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_binary(lw, op, "+");
    return lw->status;
}

/* def _op_arith_sub(self, op): self._binary(op, "-") */
static ckc_status_t ckc_h_op_arith_sub(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_binary(lw, op, "-");
    return lw->status;
}

/* def _op_arith_mul(self, op): self._binary(op, "*") */
static ckc_status_t ckc_h_op_arith_mul(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_binary(lw, op, "*");
    return lw->status;
}

/* def _op_arith_div(self, op): self._binary(op, "/") */
static ckc_status_t ckc_h_op_arith_div(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_binary(lw, op, "/");
    return lw->status;
}

/* def _op_arith_mod(self, op): self._binary(op, "%") */
static ckc_status_t ckc_h_op_arith_mod(ckc_h_lowerer_t* lw, const ckc_op_t* op)
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
static ckc_status_t ckc_h_op_arith_cmp(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const char* pred = ckc_attr_get_str(&op->attrs, "pred");
    const char* c_op;
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    const ckc_value_t* r = h_res(op);
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
        return ckc_h_fail(lw, CKC_ERR_KEY, "arith.cmp: unknown pred %s", pred);
    }
    ckc_h_emitf(
        lw, "bool %s = %s %s %s;", ckc_h_name(lw, r), ckc_h_name(lw, a), c_op, ckc_h_name(lw, b));
    return lw->status;
}

/* def _op_arith_select(self, op):
 *     cond, lhs, rhs = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
 *                f"{_name(cond)} ? {_name(lhs)} : {_name(rhs)};") */
static ckc_status_t ckc_h_op_arith_select(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* cond = op->operands[0];
    const ckc_value_t* lhs  = op->operands[1];
    const ckc_value_t* rhs  = op->operands[2];
    const ckc_value_t* r    = h_res(op);
    ckc_h_emitf(lw,
                "%s %s = %s ? %s : %s;",
                ckc_h_type_to_hip(lw, r->type),
                ckc_h_name(lw, r),
                ckc_h_name(lw, cond),
                ckc_h_name(lw, lhs),
                ckc_h_name(lw, rhs));
    return lw->status;
}

/* ----------------------------- arith: bitwise ------------------------------ */

/* def _op_arith_and(self, op):
 *     a, b = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
 *                f"{_name(a)} & {_name(b)};") */
static ckc_status_t ckc_h_op_arith_and(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    const ckc_value_t* r = h_res(op);
    ckc_h_emitf(lw,
                "%s %s = %s & %s;",
                ckc_h_type_to_hip(lw, r->type),
                ckc_h_name(lw, r),
                ckc_h_name(lw, a),
                ckc_h_name(lw, b));
    return lw->status;
}

/* def _op_arith_or(self, op):
 *     a, b = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
 *                f"{_name(a)} | {_name(b)};") */
static ckc_status_t ckc_h_op_arith_or(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    const ckc_value_t* r = h_res(op);
    ckc_h_emitf(lw,
                "%s %s = %s | %s;",
                ckc_h_type_to_hip(lw, r->type),
                ckc_h_name(lw, r),
                ckc_h_name(lw, a),
                ckc_h_name(lw, b));
    return lw->status;
}

/* def _op_arith_smax(self, op):
 *     a, b = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
 *                f"({_name(a)} > {_name(b)} ? {_name(a)} : {_name(b)});") */
static ckc_status_t ckc_h_op_arith_smax(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    const ckc_value_t* r = h_res(op);
    const char* an       = ckc_h_name(lw, a);
    const char* bn       = ckc_h_name(lw, b);
    ckc_h_emitf(lw,
                "%s %s = (%s > %s ? %s : %s);",
                ckc_h_type_to_hip(lw, r->type),
                ckc_h_name(lw, r),
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
static ckc_status_t ckc_h_op_arith_smin(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    const ckc_value_t* r = h_res(op);
    const char* an       = ckc_h_name(lw, a);
    const char* bn       = ckc_h_name(lw, b);
    ckc_h_emitf(lw,
                "%s %s = (%s < %s ? %s : %s);",
                ckc_h_type_to_hip(lw, r->type),
                ckc_h_name(lw, r),
                an,
                bn,
                an,
                bn);
    return lw->status;
}

/* def _op_arith_not(self, op):
 *     (v,) = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = ~{_name(v)};") */
static ckc_status_t ckc_h_op_arith_not(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* v = op->operands[0];
    const ckc_value_t* r = h_res(op);
    ckc_h_emitf(
        lw, "%s %s = ~%s;", ckc_h_type_to_hip(lw, r->type), ckc_h_name(lw, r), ckc_h_name(lw, v));
    return lw->status;
}

/* def _op_arith_xor(self, op): self._binary(op, "^") */
static ckc_status_t ckc_h_op_arith_xor(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_binary(lw, op, "^");
    return lw->status;
}

/* def _op_arith_shl(self, op): self._binary(op, "<<") */
static ckc_status_t ckc_h_op_arith_shl(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_binary(lw, op, "<<");
    return lw->status;
}

/* def _op_arith_lshr(self, op):
 *     a, b = op.operands
 *     self._emit(f"int {_name(op.result)} = "
 *                f"(int)((unsigned)({_name(a)}) >> {_name(b)});") */
static ckc_status_t ckc_h_op_arith_lshr(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    const ckc_value_t* r = h_res(op);
    ckc_h_emitf(lw,
                "int %s = (int)((unsigned)(%s) >> %s);",
                ckc_h_name(lw, r),
                ckc_h_name(lw, a),
                ckc_h_name(lw, b));
    return lw->status;
}

/* def _op_arith_umul_hi_i32(self, op):
 *     a, b = op.operands
 *     self._emit(f"int {_name(op.result)} = "
 *                f"(int)__umulhi((unsigned){_name(a)}, (unsigned){_name(b)});") */
static ckc_status_t ckc_h_op_arith_umul_hi_i32(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    const ckc_value_t* r = h_res(op);
    ckc_h_emitf(lw,
                "int %s = (int)__umulhi((unsigned)%s, (unsigned)%s);",
                ckc_h_name(lw, r),
                ckc_h_name(lw, a),
                ckc_h_name(lw, b));
    return lw->status;
}

/* ----------------------------- arith: float binary ------------------------- */

/* def _op_arith_fadd(self, op): self._binary(op, "+") */
static ckc_status_t ckc_h_op_arith_fadd(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_binary(lw, op, "+");
    return lw->status;
}

/* def _op_arith_fsub(self, op): self._binary(op, "-") */
static ckc_status_t ckc_h_op_arith_fsub(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_binary(lw, op, "-");
    return lw->status;
}

/* def _op_arith_fmul(self, op): self._binary(op, "*") */
static ckc_status_t ckc_h_op_arith_fmul(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_binary(lw, op, "*");
    return lw->status;
}

/* def _op_arith_fdiv(self, op): self._binary(op, "/") */
static ckc_status_t ckc_h_op_arith_fdiv(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_binary(lw, op, "/");
    return lw->status;
}

/* def _op_arith_fneg(self, op):
 *     (v,) = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = -{_name(v)};") */
static ckc_status_t ckc_h_op_arith_fneg(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* v = op->operands[0];
    const ckc_value_t* r = h_res(op);
    ckc_h_emitf(
        lw, "%s %s = -%s;", ckc_h_type_to_hip(lw, r->type), ckc_h_name(lw, r), ckc_h_name(lw, v));
    return lw->status;
}

/* def _op_arith_fabs(self, op):
 *     (v,) = op.operands
 *     ty = _type_to_hip(op.result.type)
 *     helper = {"f32":"fabsf","f16":"__builtin_fabsf","bf16":"__builtin_fabsf"}
 *              .get(op.result.type.name, "fabsf")
 *     self._emit(f"{ty} {_name(op.result)} = ({ty}){helper}((float){_name(v)});") */
static ckc_status_t ckc_h_op_arith_fabs(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* v = op->operands[0];
    const ckc_value_t* r = h_res(op);
    const char* ty       = ckc_h_type_to_hip(lw, r->type);
    const char* tname    = r->type->name;
    const char* helper   = "fabsf";
    if(tname && (strcmp(tname, "f16") == 0 || strcmp(tname, "bf16") == 0))
    {
        helper = "__builtin_fabsf";
    }
    ckc_h_emitf(
        lw, "%s %s = (%s)%s((float)%s);", ty, ckc_h_name(lw, r), ty, helper, ckc_h_name(lw, v));
    return lw->status;
}

/* def _op_arith_fma(self, op):
 *     a, b, c = op.operands
 *     ty = _type_to_hip(op.result.type)
 *     self._emit(f"{ty} {_name(op.result)} = ({ty})fmaf("
 *                f"(float){_name(a)}, (float){_name(b)}, (float){_name(c)});") */
static ckc_status_t ckc_h_op_arith_fma(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    const ckc_value_t* c = op->operands[2];
    const ckc_value_t* r = h_res(op);
    const char* ty       = ckc_h_type_to_hip(lw, r->type);
    ckc_h_emitf(lw,
                "%s %s = (%s)fmaf((float)%s, (float)%s, (float)%s);",
                ty,
                ckc_h_name(lw, r),
                ty,
                ckc_h_name(lw, a),
                ckc_h_name(lw, b),
                ckc_h_name(lw, c));
    return lw->status;
}

/* def _op_arith_fmax3(self, op):
 *     a, b, c = op.operands
 *     ty = _type_to_hip(op.result.type)
 *     self._emit(f"{ty} {_name(op.result)} = "
 *                f"(({_name(b)} > {_name(c)}) ? {_name(b)} : {_name(c)});")
 *     self._emit(f"{_name(op.result)} = "
 *                f"({_name(a)} > {_name(op.result)}) ? {_name(a)} : {_name(op.result)};") */
static ckc_status_t ckc_h_op_arith_fmax3(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    const ckc_value_t* c = op->operands[2];
    const ckc_value_t* r = h_res(op);
    const char* ty       = ckc_h_type_to_hip(lw, r->type);
    const char* rn       = ckc_h_name(lw, r);
    const char* an       = ckc_h_name(lw, a);
    const char* bn       = ckc_h_name(lw, b);
    const char* cn       = ckc_h_name(lw, c);
    ckc_h_emitf(lw, "%s %s = ((%s > %s) ? %s : %s);", ty, rn, bn, cn, bn, cn);
    ckc_h_emitf(lw, "%s = (%s > %s) ? %s : %s;", rn, an, rn, an, rn);
    return lw->status;
}

/* def _op_arith_fmin3(self, op):
 *     a, b, c = op.operands
 *     ty = _type_to_hip(op.result.type)
 *     self._emit(f"{ty} {_name(op.result)} = "
 *                f"(({_name(b)} < {_name(c)}) ? {_name(b)} : {_name(c)});")
 *     self._emit(f"{_name(op.result)} = "
 *                f"({_name(a)} < {_name(op.result)}) ? {_name(a)} : {_name(op.result)};") */
static ckc_status_t ckc_h_op_arith_fmin3(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    const ckc_value_t* c = op->operands[2];
    const ckc_value_t* r = h_res(op);
    const char* ty       = ckc_h_type_to_hip(lw, r->type);
    const char* rn       = ckc_h_name(lw, r);
    const char* an       = ckc_h_name(lw, a);
    const char* bn       = ckc_h_name(lw, b);
    const char* cn       = ckc_h_name(lw, c);
    ckc_h_emitf(lw, "%s %s = ((%s < %s) ? %s : %s);", ty, rn, bn, cn, bn, cn);
    ckc_h_emitf(lw, "%s = (%s < %s) ? %s : %s;", rn, an, rn, an, rn);
    return lw->status;
}

/* def _op_arith_fmax(self, op):
 *     a, b = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
 *                f"({_name(a)} > {_name(b)}) ? {_name(a)} : {_name(b)};") */
static ckc_status_t ckc_h_op_arith_fmax(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    const ckc_value_t* r = h_res(op);
    const char* an       = ckc_h_name(lw, a);
    const char* bn       = ckc_h_name(lw, b);
    ckc_h_emitf(lw,
                "%s %s = (%s > %s) ? %s : %s;",
                ckc_h_type_to_hip(lw, r->type),
                ckc_h_name(lw, r),
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
static ckc_status_t ckc_h_op_arith_fmin(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    const ckc_value_t* r = h_res(op);
    const char* an       = ckc_h_name(lw, a);
    const char* bn       = ckc_h_name(lw, b);
    ckc_h_emitf(lw,
                "%s %s = (%s < %s) ? %s : %s;",
                ckc_h_type_to_hip(lw, r->type),
                ckc_h_name(lw, r),
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
static ckc_status_t ckc_h_op_arith_fcmp(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const char* pred     = ckc_attr_get_str(&op->attrs, "pred");
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    const ckc_value_t* r = h_res(op);
    const char* rn       = ckc_h_name(lw, r);
    const char* an       = ckc_h_name(lw, a);
    const char* bn       = ckc_h_name(lw, b);
    const char* cop      = NULL;
    if(!pred)
    {
        return ckc_h_fail(lw, CKC_ERR_KEY, "arith.fcmp: missing pred");
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
        ckc_h_emitf(lw,
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
        ckc_h_emitf(lw, "bool %s = (!isnan(float(%s)) && !isnan(float(%s)));", rn, an, bn);
    }
    else if(strcmp(pred, "uno") == 0)
    {
        ckc_h_emitf(lw, "bool %s = (isnan(float(%s)) || isnan(float(%s)));", rn, an, bn);
    }
    else
    {
        return ckc_h_fail(lw, CKC_ERR_NOTIMPL, "unknown fcmp predicate '%s'", pred);
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
static void h_math1(ckc_h_lowerer_t* lw, const ckc_op_t* op, const char* fn_f32)
{
    const ckc_value_t* v = op->operands[0];
    const ckc_value_t* r = h_res(op);
    const char* tname    = r->type->name;
    const char* cpp_t    = ckc_h_type_to_hip(lw, r->type);
    const char* rn       = ckc_h_name(lw, r);
    const char* vn       = ckc_h_name(lw, v);
    if(tname && strcmp(tname, "f32") == 0)
    {
        ckc_h_emitf(lw, "%s %s = %s(%s);", cpp_t, rn, fn_f32, vn);
    }
    else
    {
        ckc_h_emitf(lw, "%s %s = (%s)%s((float)%s);", cpp_t, rn, cpp_t, fn_f32, vn);
    }
}

/* An amdgcn-builtin reciprocal-style math op (rcp / rcp_fast / rsqrt). Same
 * promote-compute-demote shape as _math1 but the f32 path drops the (cpp_t)
 * cast since the builtin already returns float:
 *   if tname == "f32":
 *       self._emit(f"{cpp_t} {_name} = {builtin}({_name(v)});")
 *   else:
 *       self._emit(f"{cpp_t} {_name} = ({cpp_t}){builtin}((float){_name(v)});") */
static void h_amdgcn_unary(ckc_h_lowerer_t* lw, const ckc_op_t* op, const char* builtin)
{
    const ckc_value_t* v = op->operands[0];
    const ckc_value_t* r = h_res(op);
    const char* tname    = r->type->name;
    const char* cpp_t    = ckc_h_type_to_hip(lw, r->type);
    const char* rn       = ckc_h_name(lw, r);
    const char* vn       = ckc_h_name(lw, v);
    if(tname && strcmp(tname, "f32") == 0)
    {
        ckc_h_emitf(lw, "%s %s = %s(%s);", cpp_t, rn, builtin, vn);
    }
    else
    {
        ckc_h_emitf(lw, "%s %s = (%s)%s((float)%s);", cpp_t, rn, cpp_t, builtin, vn);
    }
}

/* def _op_math_exp2(self, op): self._math1(op, "exp2f") */
static ckc_status_t ckc_h_op_math_exp2(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_math1(lw, op, "exp2f");
    return lw->status;
}

/* def _op_math_log2(self, op): self._math1(op, "log2f") */
static ckc_status_t ckc_h_op_math_log2(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_math1(lw, op, "log2f");
    return lw->status;
}

/* def _op_math_rcp(self, op): -- __builtin_amdgcn_rcpf, promote/demote else. */
static ckc_status_t ckc_h_op_math_rcp(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_amdgcn_unary(lw, op, "__builtin_amdgcn_rcpf");
    return lw->status;
}

/* def _op_math_rcp_fast(self, op): -- identical to math.rcp on HIP. */
static ckc_status_t ckc_h_op_math_rcp_fast(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_amdgcn_unary(lw, op, "__builtin_amdgcn_rcpf");
    return lw->status;
}

/* def _op_math_sqrt(self, op): self._math1(op, "sqrtf") */
static ckc_status_t ckc_h_op_math_sqrt(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_math1(lw, op, "sqrtf");
    return lw->status;
}

/* def _op_math_rsqrt(self, op): -- __builtin_amdgcn_rsqf, promote/demote else. */
static ckc_status_t ckc_h_op_math_rsqrt(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_amdgcn_unary(lw, op, "__builtin_amdgcn_rsqf");
    return lw->status;
}

/* def _op_math_tanh(self, op): self._math1(op, "tanhf") */
static ckc_status_t ckc_h_op_math_tanh(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    h_math1(lw, op, "tanhf");
    return lw->status;
}

/* ----------------------------- registration table -------------------------- */

const ckc_h_handler_entry_t* ckc_h_handlers_arith(void)
{
    static const ckc_h_handler_entry_t table[] = {
        {CKC_OP_ARITH_CONSTANT, ckc_h_op_arith_constant},
        {CKC_OP_ARITH_CONSTANT_VEC, ckc_h_op_arith_constant_vec},
        {CKC_OP_ARITH_ADD, ckc_h_op_arith_add},
        {CKC_OP_ARITH_SUB, ckc_h_op_arith_sub},
        {CKC_OP_ARITH_MUL, ckc_h_op_arith_mul},
        {CKC_OP_ARITH_DIV, ckc_h_op_arith_div},
        {CKC_OP_ARITH_MOD, ckc_h_op_arith_mod},
        {CKC_OP_ARITH_CMP, ckc_h_op_arith_cmp},
        {CKC_OP_ARITH_SELECT, ckc_h_op_arith_select},
        {CKC_OP_ARITH_AND, ckc_h_op_arith_and},
        {CKC_OP_ARITH_OR, ckc_h_op_arith_or},
        {CKC_OP_ARITH_SMAX, ckc_h_op_arith_smax},
        {CKC_OP_ARITH_SMIN, ckc_h_op_arith_smin},
        {CKC_OP_ARITH_NOT, ckc_h_op_arith_not},
        {CKC_OP_ARITH_XOR, ckc_h_op_arith_xor},
        {CKC_OP_ARITH_SHL, ckc_h_op_arith_shl},
        {CKC_OP_ARITH_LSHR, ckc_h_op_arith_lshr},
        {CKC_OP_ARITH_UMUL_HI_I32, ckc_h_op_arith_umul_hi_i32},
        {CKC_OP_ARITH_FADD, ckc_h_op_arith_fadd},
        {CKC_OP_ARITH_FSUB, ckc_h_op_arith_fsub},
        {CKC_OP_ARITH_FMUL, ckc_h_op_arith_fmul},
        {CKC_OP_ARITH_FDIV, ckc_h_op_arith_fdiv},
        {CKC_OP_ARITH_FNEG, ckc_h_op_arith_fneg},
        {CKC_OP_ARITH_FABS, ckc_h_op_arith_fabs},
        {CKC_OP_ARITH_FMA, ckc_h_op_arith_fma},
        {CKC_OP_ARITH_FMAX3, ckc_h_op_arith_fmax3},
        {CKC_OP_ARITH_FMIN3, ckc_h_op_arith_fmin3},
        {CKC_OP_ARITH_FMAX, ckc_h_op_arith_fmax},
        {CKC_OP_ARITH_FMIN, ckc_h_op_arith_fmin},
        {CKC_OP_ARITH_FCMP, ckc_h_op_arith_fcmp},
        {CKC_OP_MATH_EXP2, ckc_h_op_math_exp2},
        {CKC_OP_MATH_LOG2, ckc_h_op_math_log2},
        {CKC_OP_MATH_RCP, ckc_h_op_math_rcp},
        {CKC_OP_MATH_RCP_FAST, ckc_h_op_math_rcp_fast},
        {CKC_OP_MATH_SQRT, ckc_h_op_math_sqrt},
        {CKC_OP_MATH_RSQRT, ckc_h_op_math_rsqrt},
        {CKC_OP_MATH_TANH, ckc_h_op_math_tanh},
        {CKC_OP_INVALID, NULL}, /* terminator */
    };
    return table;
}

} /* namespace ckc */
