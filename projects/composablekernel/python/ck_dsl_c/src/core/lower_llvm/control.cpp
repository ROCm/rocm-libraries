// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_llvm_control.c -- BUCKET 6 of the C99 port of ck_dsl.core.lower_llvm.
 *
 * The vector / control-flow / sync handler group. Faithful (or correct-shaped
 * TODO-stubbed) translation of:
 *
 *   vector.*  : add/sub/mul/and/or/shl/lshr/smax/smin/max/fma/sum/reduce_max/
 *               splat/select/cmp/trunc/sext/trunc_f32_to_f16/trunc_f32_to/
 *               bitcast/extract/insert/pack/concat
 *   tile.*    : sync / sync_half_block / sync_lds_only / s_barrier_bare /
 *               s_waitcnt / s_setprio / iglp_opt / sched_barrier /
 *               sched_group_barrier
 *   scf./cf.  : scf.for / scf.if / scf.yield / cf.return
 *
 * Plus the bucket-shared scf builders (ckc_ll_lower_normal_for /
 * ckc_ll_lower_unrolled_for), the shared horizontal reduce
 * (ckc_ll_lower_vector_reduce), the yield-recording stack, and the two CDNA/RDNA
 * waitcnt encoders.
 *
 * Every shared helper (ckc_ll_emit*, ckc_ll_operand, ckc_ll_llvm_type,
 * ckc_ll_need, ckc_ll_fresh, ckc_ll_vector_binop, ckc_ll_fail, ...) lives in
 * BUCKET 0; this file only calls them through the internal header.
 */
#include "ckc/lower_llvm_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace ckc {

/* ------------------------------------------------------------------ helpers */

/* Python: op.result -- exactly one result (the producing builder guaranteed
 * it). Returns op->results[0] or NULL. */
static const ckc_value_t* ll_result(const ckc_op_t* op)
{
    return (op && op->num_results > 0) ? op->results[0] : NULL;
}

/* Replace every occurrence of `tok` with `repl` across `blk`'s lines, in place.
 * Defined in the scf.* section; forward-declared here for the for-CFG builders
 * that back-patch deferred %FOR_LATCH / %FOR_EXIT labels. */
static void
ll_block_replace(ckc_lower_t* L, ckc_ll_block_t* blk, const char* tok, const char* repl);

/* Element type name of a vector type, or NULL. */
static const char* ll_vec_elem_name(const ckc_type_t* t)
{
    if(t && t->kind == CKC_TYPE_VECTOR && t->elem)
    {
        return t->elem->name;
    }
    return NULL;
}

/* True if `name` is one of the floating element names f16/bf16/f32. */
static bool ll_is_fp_elem(const char* name)
{
    return name &&
           (strcmp(name, "f16") == 0 || strcmp(name, "bf16") == 0 || strcmp(name, "f32") == 0);
}

/* ======================================================================== */
/* waitcnt encoders (Python _encode_waitcnt_gfx9_10 / _encode_waitcnt_gfx11) */
/* ======================================================================== */

static int ll_clamp(int v, int lo, int hi)
{
    if(v < lo)
    {
        return lo;
    }
    if(v > hi)
    {
        return hi;
    }
    return v;
}

/* Python _encode_waitcnt_gfx9_10: split VMCNT across [3:0] and [15:14]. -1
 * means "no wait" (architectural max). */
int ckc_ll_encode_waitcnt_gfx9_10(int vmcnt, int expcnt, int lgkmcnt)
{
    int vm_b  = (vmcnt < 0) ? 0x3F : ll_clamp(vmcnt, 0, 0x3F);
    int ec_b  = (expcnt < 0) ? 0x7 : ll_clamp(expcnt, 0, 0x7);
    int lk_b  = (lgkmcnt < 0) ? 0xF : ll_clamp(lgkmcnt, 0, 0xF);
    int vm_lo = vm_b & 0xF;
    int vm_hi = (vm_b >> 4) & 0x3;
    return vm_lo | (ec_b << 4) | (lk_b << 8) | (vm_hi << 14);
}

/* Python _encode_waitcnt_gfx11: contiguous fields, 6-bit LGKMCNT, no split
 * VMCNT. -1 means "no wait" (architectural max). */
int ckc_ll_encode_waitcnt_gfx11(int vmcnt, int expcnt, int lgkmcnt)
{
    int vm_b = (vmcnt < 0) ? 0x3F : ll_clamp(vmcnt, 0, 0x3F);
    int ec_b = (expcnt < 0) ? 0x7 : ll_clamp(expcnt, 0, 0x7);
    int lk_b = (lgkmcnt < 0) ? 0x3F : ll_clamp(lgkmcnt, 0, 0x3F);
    return (ec_b & 0x7) | ((lk_b & 0x3F) << 4) | ((vm_b & 0x3F) << 10);
}

/* Convenience: invoke the backend encoder (Python self._backend.encode_waitcnt)
 * falling back to the gfx9_10 encoder when no backend is bound. */
static int ll_backend_waitcnt(ckc_lower_t* L, int vmcnt, int expcnt, int lgkmcnt)
{
    if(L && L->backend && L->backend->encode_waitcnt)
    {
        return L->backend->encode_waitcnt(vmcnt, expcnt, lgkmcnt);
    }
    return ckc_ll_encode_waitcnt_gfx9_10(vmcnt, expcnt, lgkmcnt);
}

/* ======================================================================== */
/* yield-stack helpers (Python _yield_stack: list of list[str])             */
/* ======================================================================== */

typedef CKC_VEC(const char*) ll_yield_frame_t;

/* Python self._yield_stack.append([]). */
void ckc_ll_yield_push(ckc_lower_t* L)
{
    if(!ckc_ll_live(L))
    {
        return;
    }
    ll_yield_frame_t* frame = (ll_yield_frame_t*)ckc_arena_alloc(&L->arena, sizeof *frame);
    if(!frame)
    {
        ckc_ll_fail(L, CKC_ERR_OOM, "yield_push: arena OOM");
    }
    ckc_vec_init(frame);
    int rc = 0;
    /* yield_stack stores a distinct anonymous-struct pointer type; the in-place
     * layout is identical, so launder through void* to satisfy the compiler. */
    void* frame_v = frame;
    /* WS3 C++ build: the element type is a distinct anonymous-struct pointer, so
     * the void* must be cast back explicitly (C allowed the implicit conversion).
     * __typeof__ recovers the exact element type; layout is identical. */
    ckc_vec_push(&L->arena, &L->yield_stack, (__typeof__(*L->yield_stack.data))frame_v, rc);
    if(rc != 0)
    {
        ckc_ll_fail(L, CKC_ERR_OOM, "yield_push: arena OOM");
    }
}

/* Python yielded = self._yield_stack.pop(). Returns the frame's operand
 * strings via out params; empty/NULL on an empty stack. */
void ckc_ll_yield_pop(ckc_lower_t* L, const char* const** out_items, int* out_count)
{
    if(out_items)
    {
        *out_items = NULL;
    }
    if(out_count)
    {
        *out_count = 0;
    }
    if(!L || L->yield_stack.len == 0)
    {
        return;
    }
    L->yield_stack.len -= 1;
    ll_yield_frame_t* frame = (ll_yield_frame_t*)L->yield_stack.data[L->yield_stack.len];
    if(frame)
    {
        if(out_items)
        {
            *out_items = frame->data;
        }
        if(out_count)
        {
            *out_count = (int)frame->len;
        }
    }
}

/* Python self._yield_stack[-1].append(operand_str). */
void ckc_ll_yield_record(ckc_lower_t* L, const char* operand_str)
{
    if(!ckc_ll_live(L) || L->yield_stack.len == 0)
    {
        return;
    }
    ll_yield_frame_t* frame = (ll_yield_frame_t*)L->yield_stack.data[L->yield_stack.len - 1];
    if(!frame)
    {
        return;
    }
    const char* dup = ckc_arena_strdup(&L->arena, operand_str ? operand_str : "");
    int rc          = 0;
    ckc_vec_push(&L->arena, frame, dup, rc);
    if(rc != 0)
    {
        ckc_ll_fail(L, CKC_ERR_OOM, "yield_record: arena OOM");
    }
}

/* Python len(self._yield_stack). */
int ckc_ll_yield_depth(const ckc_lower_t* L) { return L ? (int)L->yield_stack.len : 0; }

/* ======================================================================== */
/* shared horizontal vector reduce (Python _lower_vector_reduce)            */
/* ======================================================================== */

/* Extract every lane and fold with `llvm_op` starting from `init`. */
void ckc_ll_lower_vector_reduce(ckc_lower_t* L,
                                const ckc_op_t* op,
                                const char* llvm_op,
                                const char* init)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 1)
    {
        return;
    }
    const ckc_value_t* v     = op->operands[0];
    const ckc_type_t* vec_ty = v->type;
    if(!vec_ty || vec_ty->kind != CKC_TYPE_VECTOR)
    {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "vector reduce: not a vector operand");
    }
    int count                 = vec_ty->count;
    const ckc_type_t* elem_ty = vec_ty->elem;
    const char* vec_llvm      = ckc_ll_llvm_type(L, vec_ty);
    const char* elem_llvm     = ckc_ll_llvm_type(L, elem_ty);
    const char* acc           = init;
    for(int i = 0; i < count; i++)
    {
        const char* e = ckc_ll_fresh(L, "vred.e");
        ckc_ll_emitf(
            L, "  %s = extractelement %s %s, i32 %d", e, vec_llvm, ckc_ll_operand(L, v), i);
        const char* name = (i == count - 1) ? res->name : ckc_ll_fresh(L, "vred");
        ckc_ll_emitf(L, "  %s = %s %s %s, %s", name, llvm_op, elem_llvm, acc, e);
        acc = name;
    }
}

/* ======================================================================== */
/* vector.* per-op handlers                                                  */
/* ======================================================================== */

/* Python _op_vector_add. */
static void _op_vector_add(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    const char* elem       = res ? ll_vec_elem_name(res->type) : NULL;
    ckc_ll_vector_binop(L, op, ll_is_fp_elem(elem) ? "fadd" : "add");
}
/* Python _op_vector_sub. */
static void _op_vector_sub(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    const char* elem       = res ? ll_vec_elem_name(res->type) : NULL;
    ckc_ll_vector_binop(L, op, ll_is_fp_elem(elem) ? "fsub" : "sub");
}
/* Python _op_vector_mul. */
static void _op_vector_mul(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    const char* elem       = res ? ll_vec_elem_name(res->type) : NULL;
    ckc_ll_vector_binop(L, op, ll_is_fp_elem(elem) ? "fmul" : "mul");
}
/* Python _op_vector_and. */
static void _op_vector_and(ckc_lower_t* L, const ckc_op_t* op)
{
    ckc_ll_vector_binop(L, op, "and");
}
/* Python _op_vector_or. */
static void _op_vector_or(ckc_lower_t* L, const ckc_op_t* op) { ckc_ll_vector_binop(L, op, "or"); }
/* Python _op_vector_shl. */
static void _op_vector_shl(ckc_lower_t* L, const ckc_op_t* op)
{
    ckc_ll_vector_binop(L, op, "shl");
}
/* Python _op_vector_lshr. */
static void _op_vector_lshr(ckc_lower_t* L, const ckc_op_t* op)
{
    ckc_ll_vector_binop(L, op, "lshr");
}

/* Python _op_vector_cmp: packed icmp with a pred map. */
static void _op_vector_cmp(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 2)
    {
        return;
    }
    const char* pred = ckc_attr_get_str(&op->attrs, "pred");
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
        ckc_ll_fail(L, CKC_ERR_KEY, "vector.cmp: unknown pred %s", pred);
    }
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    ckc_ll_emitf(L,
                 "  %s = icmp %s %s %s, %s",
                 res->name,
                 llvm_pred,
                 ckc_ll_llvm_type(L, a->type),
                 ckc_ll_operand(L, a),
                 ckc_ll_operand(L, b));
}

/* Python _op_vector_smax: dynamic llvm.smax.v<N>i<W> intrinsic. */
static void _op_vector_smax(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 2)
    {
        return;
    }
    const ckc_value_t* a     = op->operands[0];
    const ckc_value_t* b     = op->operands[1];
    const ckc_type_t* vec_ty = a->type;
    if(!vec_ty || vec_ty->kind != CKC_TYPE_VECTOR || !vec_ty->elem || !vec_ty->elem->name)
    {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "vector.smax: not an int vector");
    }
    int count            = vec_ty->count;
    const char* ename    = vec_ty->elem->name; /* "i16" -> width "16" */
    const char* width    = (ename[0] == 'i') ? ename + 1 : ename;
    const char* vec_llvm = ckc_ll_llvm_type(L, vec_ty);
    char intrin[64];
    snprintf(intrin, sizeof intrin, "llvm.smax.v%di%s", count, width);
    const char* decl =
        ckc_arena_printf(&L->arena, "declare %s @%s(%s, %s)", vec_llvm, intrin, vec_llvm, vec_llvm);
    ckc_ll_need_dynamic(L, intrin, decl);
    ckc_ll_emitf(L,
                 "  %s = call %s @%s(%s %s, %s %s)",
                 res->name,
                 vec_llvm,
                 intrin,
                 vec_llvm,
                 ckc_ll_operand(L, a),
                 vec_llvm,
                 ckc_ll_operand(L, b));
}

/* Python _op_vector_smin: icmp slt + vselect. */
static void _op_vector_smin(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 2)
    {
        return;
    }
    const ckc_value_t* a = op->operands[0];
    const ckc_value_t* b = op->operands[1];
    int count            = (res->type && res->type->kind == CKC_TYPE_VECTOR) ? res->type->count : 0;
    const char* cmp      = ckc_ll_fresh(L, "vsmin.cmp");
    ckc_ll_emitf(L,
                 "  %s = icmp slt %s %s, %s",
                 cmp,
                 ckc_ll_llvm_type(L, a->type),
                 ckc_ll_operand(L, a),
                 ckc_ll_operand(L, b));
    ckc_ll_emitf(L,
                 "  %s = select <%d x i1> %s, %s %s, %s %s",
                 res->name,
                 count,
                 cmp,
                 ckc_ll_llvm_type(L, a->type),
                 ckc_ll_operand(L, a),
                 ckc_ll_llvm_type(L, b->type),
                 ckc_ll_operand(L, b));
}

/* Python _op_vector_trunc: packed trunc. */
static void _op_vector_trunc(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 1)
    {
        return;
    }
    const ckc_value_t* v = op->operands[0];
    ckc_ll_emitf(L,
                 "  %s = trunc %s %s to %s",
                 res->name,
                 ckc_ll_llvm_type(L, v->type),
                 ckc_ll_operand(L, v),
                 ckc_ll_llvm_type(L, res->type));
}

/* Python _op_vector_sext: packed sext. */
static void _op_vector_sext(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 1)
    {
        return;
    }
    const ckc_value_t* v = op->operands[0];
    ckc_ll_emitf(L,
                 "  %s = sext %s %s to %s",
                 res->name,
                 ckc_ll_llvm_type(L, v->type),
                 ckc_ll_operand(L, v),
                 ckc_ll_llvm_type(L, res->type));
}

/* Python _op_vector_fma: packed llvm.fmuladd.v<N><elem>.
 *
 *     a, b, c = op.operands
 *     vec_ty = a.type; count = vec_ty.count; elem_ty = vec_ty.elem
 *     elem_name = elem_ty.name
 *     if elem_name not in ("f16","bf16","f32"): raise NotImplementedError
 *     intrin_key = f"fmuladd.v{count}{elem_name}"; self._need(intrin_key)
 *     vec_llvm = _llvm_type(vec_ty)
 *     emit f"  {res} = call {vec_llvm} @llvm.fmuladd.v{count}{elem_name}("
 *          f"{vec_llvm} {a}, {vec_llvm} {b}, {vec_llvm} {c})"
 *
 * The intrin key (e.g. "fmuladd.v4f32") is a static row of the
 * _INTRINSIC_DECLS table, so ckc_ll_need() resolves the canonical declare. */
static void _op_vector_fma(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 3)
    {
        return;
    }
    const ckc_value_t* a     = op->operands[0];
    const ckc_value_t* b     = op->operands[1];
    const ckc_value_t* c     = op->operands[2];
    const ckc_type_t* vec_ty = a->type;
    if(!vec_ty || vec_ty->kind != CKC_TYPE_VECTOR || !vec_ty->elem || !vec_ty->elem->name)
    {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "vector_fma: not a vector");
    }
    int count             = vec_ty->count;
    const char* elem_name = vec_ty->elem->name;
    if(!(strcmp(elem_name, "f16") == 0 || strcmp(elem_name, "bf16") == 0 ||
         strcmp(elem_name, "f32") == 0))
    {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "vector_fma: unsupported element type %s", elem_name);
    }
    const char* intrin_key = ckc_arena_printf(&L->arena, "fmuladd.v%d%s", count, elem_name);
    ckc_ll_need(L, intrin_key);
    const char* vec_llvm = ckc_ll_llvm_type(L, vec_ty);
    ckc_ll_emitf(L,
                 "  %s = call %s @llvm.fmuladd.v%d%s(%s %s, %s %s, %s %s)",
                 res->name,
                 vec_llvm,
                 count,
                 elem_name,
                 vec_llvm,
                 ckc_ll_operand(L, a),
                 vec_llvm,
                 ckc_ll_operand(L, b),
                 vec_llvm,
                 ckc_ll_operand(L, c));
}

/* Python _op_vector_max: per-element extract + fcmp ogt + select, then
 * reassemble via an insertelement chain.
 *
 *     a, b = op.operands
 *     vec_ty = a.type; count = vec_ty.count; elem_ty = vec_ty.elem
 *     vals = []
 *     for i in range(count):
 *         ea = fresh("vmax.a"); eb = fresh("vmax.b")
 *         cmp = fresh("vmax.cmp"); sel = fresh("vmax.sel")
 *         emit f"  {ea} = extractelement {vec_ty} {a}, i32 {i}"
 *         emit f"  {eb} = extractelement {vec_ty} {b}, i32 {i}"
 *         emit f"  {cmp} = fcmp ogt {elem_ty} {ea}, {eb}"
 *         emit f"  {sel} = select i1 {cmp}, {elem_ty} {ea}, {elem_ty} {eb}"
 *         vals.append(sel)
 *     prev = "undef"
 *     for i, v in enumerate(vals):
 *         name = res.name if i==count-1 else fresh("vmax")
 *         emit f"  {name} = insertelement {vec_ty} {prev}, {elem_ty} {v}, i32 {i}"
 *         prev = name
 */
static void _op_vector_max(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 2)
    {
        return;
    }
    const ckc_value_t* a     = op->operands[0];
    const ckc_value_t* b     = op->operands[1];
    const ckc_type_t* vec_ty = a->type;
    if(!vec_ty || vec_ty->kind != CKC_TYPE_VECTOR)
    {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "vector.max: not a vector");
    }
    int count                 = vec_ty->count;
    const ckc_type_t* elem_ty = vec_ty->elem;
    const char* vec_llvm      = ckc_ll_llvm_type(L, vec_ty);
    const char* elem_llvm     = ckc_ll_llvm_type(L, elem_ty);
    const char* a_op          = ckc_ll_operand(L, a);
    const char* b_op          = ckc_ll_operand(L, b);
    /* Collect per-lane select results so the insertelement chain matches the
     * Python two-loop structure (and thus the fresh-name ordering). */
    const char** vals = (const char**)ckc_arena_alloc(&L->arena, sizeof(const char*) * count);
    if(!vals && count > 0)
    {
        ckc_ll_fail(L, CKC_ERR_OOM, "vector.max: arena OOM");
    }
    for(int i = 0; i < count; i++)
    {
        const char* ea  = ckc_ll_fresh(L, "vmax.a");
        const char* eb  = ckc_ll_fresh(L, "vmax.b");
        const char* cmp = ckc_ll_fresh(L, "vmax.cmp");
        const char* sel = ckc_ll_fresh(L, "vmax.sel");
        ckc_ll_emitf(L, "  %s = extractelement %s %s, i32 %d", ea, vec_llvm, a_op, i);
        ckc_ll_emitf(L, "  %s = extractelement %s %s, i32 %d", eb, vec_llvm, b_op, i);
        ckc_ll_emitf(L, "  %s = fcmp ogt %s %s, %s", cmp, elem_llvm, ea, eb);
        ckc_ll_emitf(
            L, "  %s = select i1 %s, %s %s, %s %s", sel, cmp, elem_llvm, ea, elem_llvm, eb);
        vals[i] = sel;
    }
    const char* prev = "undef";
    for(int i = 0; i < count; i++)
    {
        const char* name = (i == count - 1) ? res->name : ckc_ll_fresh(L, "vmax");
        ckc_ll_emitf(L,
                     "  %s = insertelement %s %s, %s %s, i32 %d",
                     name,
                     vec_llvm,
                     prev,
                     elem_llvm,
                     vals[i],
                     i);
        prev = name;
    }
}

/* Python _op_vector_select: packed vselect. */
static void _op_vector_select(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 3)
    {
        return;
    }
    const ckc_value_t* cond = op->operands[0];
    const ckc_value_t* lhs  = op->operands[1];
    const ckc_value_t* rhs  = op->operands[2];
    ckc_ll_emitf(L,
                 "  %s = select %s %s, %s %s, %s %s",
                 res->name,
                 ckc_ll_llvm_type(L, cond->type),
                 ckc_ll_operand(L, cond),
                 ckc_ll_llvm_type(L, lhs->type),
                 ckc_ll_operand(L, lhs),
                 ckc_ll_llvm_type(L, rhs->type),
                 ckc_ll_operand(L, rhs));
}

/* Python _op_vector_sum: horizontal fadd from 0.0. */
static void _op_vector_sum(ckc_lower_t* L, const ckc_op_t* op)
{
    ckc_ll_lower_vector_reduce(L, op, "fadd", "0.000000e+00");
}

/* Python _op_vector_reduce_max: horizontal max via fcmp/select. */
static void _op_vector_reduce_max(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 1)
    {
        return;
    }
    const ckc_value_t* v     = op->operands[0];
    const ckc_type_t* vec_ty = v->type;
    if(!vec_ty || vec_ty->kind != CKC_TYPE_VECTOR)
    {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "vector.reduce_max: not a vector");
    }
    int count                 = vec_ty->count;
    const ckc_type_t* elem_ty = vec_ty->elem;
    const char* vec_llvm      = ckc_ll_llvm_type(L, vec_ty);
    const char* elem_llvm     = ckc_ll_llvm_type(L, elem_ty);
    const char* acc           = NULL;
    for(int i = 0; i < count; i++)
    {
        const char* e = ckc_ll_fresh(L, "vred.e");
        ckc_ll_emitf(
            L, "  %s = extractelement %s %s, i32 %d", e, vec_llvm, ckc_ll_operand(L, v), i);
        if(acc == NULL)
        {
            acc = e;
        }
        else
        {
            const char* cmp = ckc_ll_fresh(L, "vred.cmp");
            const char* nxt = (i == count - 1) ? res->name : ckc_ll_fresh(L, "vred.max");
            ckc_ll_emitf(L, "  %s = fcmp ogt %s %s, %s", cmp, elem_llvm, acc, e);
            ckc_ll_emitf(
                L, "  %s = select i1 %s, %s %s, %s %s", nxt, cmp, elem_llvm, acc, elem_llvm, e);
            acc = nxt;
        }
    }
    if(count == 1)
    {
        ckc_ll_emitf(
            L, "  %s = fadd %s %s, 0.000000e+00", res->name, elem_llvm, acc ? acc : "0.000000e+00");
    }
}

/* Python _op_vector_splat: broadcast a scalar via an insertelement chain.
 *
 *     (scalar,) = op.operands
 *     vec_ty = _llvm_type(op.result.type); elem_ty = _llvm_type(scalar.type)
 *     prev = "undef"; count = op.result.type.count
 *     for i in range(count):
 *         name = res.name if i==count-1 else fresh("splat")
 *         emit f"  {name} = insertelement {vec_ty} {prev}, {elem_ty} {scalar}, i32 {i}"
 *         prev = name
 */
static void _op_vector_splat(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 1)
    {
        return;
    }
    const ckc_value_t* scalar = op->operands[0];
    const char* vec_ty        = ckc_ll_llvm_type(L, res->type);
    const char* elem_ty       = ckc_ll_llvm_type(L, scalar->type);
    const char* scal          = ckc_ll_operand(L, scalar);
    int count        = (res->type && res->type->kind == CKC_TYPE_VECTOR) ? res->type->count : 0;
    const char* prev = "undef";
    for(int i = 0; i < count; i++)
    {
        const char* name = (i == count - 1) ? res->name : ckc_ll_fresh(L, "splat");
        ckc_ll_emitf(
            L, "  %s = insertelement %s %s, %s %s, i32 %d", name, vec_ty, prev, elem_ty, scal, i);
        prev = name;
    }
}

/* Python _op_vector_extract. */
static void _op_vector_extract(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 1)
    {
        return;
    }
    const ckc_value_t* v = op->operands[0];
    int64_t idx          = 0;
    ckc_attr_get_int(&op->attrs, "index", &idx);
    ckc_ll_emitf(L,
                 "  %s = extractelement %s %s, i32 %lld",
                 res->name,
                 ckc_ll_llvm_type(L, v->type),
                 ckc_ll_operand(L, v),
                 (long long)idx);
}

/* Python _op_vector_insert. */
static void _op_vector_insert(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 2)
    {
        return;
    }
    const ckc_value_t* vec  = op->operands[0];
    const ckc_value_t* elem = op->operands[1];
    int64_t idx             = 0;
    ckc_attr_get_int(&op->attrs, "index", &idx);
    ckc_ll_emitf(L,
                 "  %s = insertelement %s %s, %s %s, i32 %lld",
                 res->name,
                 ckc_ll_llvm_type(L, vec->type),
                 ckc_ll_operand(L, vec),
                 ckc_ll_llvm_type(L, elem->type),
                 ckc_ll_operand(L, elem),
                 (long long)idx);
}

/* Python _op_vector_pack: pack N scalars into <N x elem> via an insertelement
 * chain.
 *
 *     result_ty = op.result.type
 *     vec_ty = _llvm_type(result_ty); elem_ty = _llvm_type(result_ty.elem)
 *     prev = "undef"; count = result_ty.count
 *     for i, comp in enumerate(op.operands):
 *         name = res.name if i==count-1 else fresh("vpk")
 *         emit f"  {name} = insertelement {vec_ty} {prev}, {elem_ty} {comp}, i32 {i}"
 *         prev = name
 */
static void _op_vector_pack(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res)
    {
        return;
    }
    const ckc_type_t* result_ty = res->type;
    if(!result_ty || result_ty->kind != CKC_TYPE_VECTOR)
    {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "vector.pack: result not a vector");
    }
    const char* vec_ty  = ckc_ll_llvm_type(L, result_ty);
    const char* elem_ty = ckc_ll_llvm_type(L, result_ty->elem);
    int count           = result_ty->count;
    const char* prev    = "undef";
    for(int i = 0; i < op->num_operands; i++)
    {
        const char* name = (i == count - 1) ? res->name : ckc_ll_fresh(L, "vpk");
        ckc_ll_emitf(L,
                     "  %s = insertelement %s %s, %s %s, i32 %d",
                     name,
                     vec_ty,
                     prev,
                     elem_ty,
                     ckc_ll_operand(L, op->operands[i]),
                     i);
        prev = name;
    }
}

/* Python _op_vector_concat: concatenate two equal-typed vectors into a
 * double-width vector via per-lane extract + insert.
 *
 *     a, b_op = op.operands
 *     a_n = a.type.count; b_n = b_op.type.count
 *     elem_ll = _llvm_type(a.type.elem); out_ty_ll = _llvm_type(op.result.type)
 *     prev = "undef"
 *     for i in range(a_n):
 *         ext = fresh("vc.a")
 *         emit f"  {ext} = extractelement {a.type} {a}, i32 {i}"
 *         nxt = fresh("vc.ins")
 *         emit f"  {nxt} = insertelement {out_ty_ll} {prev}, {elem_ll} {ext}, i32 {i}"
 *         prev = nxt
 *     for i in range(b_n):
 *         ext = fresh("vc.b")
 *         emit f"  {ext} = extractelement {b.type} {b_op}, i32 {i}"
 *         nxt = res.name if i==b_n-1 else fresh("vc.ins")
 *         emit f"  {nxt} = insertelement {out_ty_ll} {prev}, {elem_ll} {ext}, i32 {a_n+i}"
 *         prev = nxt
 */
static void _op_vector_concat(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 2)
    {
        return;
    }
    const ckc_value_t* a    = op->operands[0];
    const ckc_value_t* b_op = op->operands[1];
    const ckc_type_t* a_ty  = a->type;
    const ckc_type_t* b_ty  = b_op->type;
    if(!a_ty || a_ty->kind != CKC_TYPE_VECTOR || !b_ty || b_ty->kind != CKC_TYPE_VECTOR)
    {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "vector.concat: not a vector");
    }
    int a_n               = a_ty->count;
    int b_n               = b_ty->count;
    const char* elem_ll   = ckc_ll_llvm_type(L, a_ty->elem);
    const char* out_ty_ll = ckc_ll_llvm_type(L, res->type);
    const char* a_ty_ll   = ckc_ll_llvm_type(L, a_ty);
    const char* b_ty_ll   = ckc_ll_llvm_type(L, b_ty);
    const char* a_op      = ckc_ll_operand(L, a);
    const char* b_op_str  = ckc_ll_operand(L, b_op);
    const char* prev      = "undef";
    for(int i = 0; i < a_n; i++)
    {
        const char* ext = ckc_ll_fresh(L, "vc.a");
        ckc_ll_emitf(L, "  %s = extractelement %s %s, i32 %d", ext, a_ty_ll, a_op, i);
        const char* nxt = ckc_ll_fresh(L, "vc.ins");
        ckc_ll_emitf(
            L, "  %s = insertelement %s %s, %s %s, i32 %d", nxt, out_ty_ll, prev, elem_ll, ext, i);
        prev = nxt;
    }
    for(int i = 0; i < b_n; i++)
    {
        const char* ext = ckc_ll_fresh(L, "vc.b");
        ckc_ll_emitf(L, "  %s = extractelement %s %s, i32 %d", ext, b_ty_ll, b_op_str, i);
        const char* nxt = (i == b_n - 1) ? res->name : ckc_ll_fresh(L, "vc.ins");
        ckc_ll_emitf(L,
                     "  %s = insertelement %s %s, %s %s, i32 %d",
                     nxt,
                     out_ty_ll,
                     prev,
                     elem_ll,
                     ext,
                     a_n + i);
        prev = nxt;
    }
}

/* Python _op_vector_bitcast. */
static void _op_vector_bitcast(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 1)
    {
        return;
    }
    const ckc_value_t* v = op->operands[0];
    ckc_ll_emitf(L,
                 "  %s = bitcast %s %s to %s",
                 res->name,
                 ckc_ll_llvm_type(L, v->type),
                 ckc_ll_operand(L, v),
                 ckc_ll_llvm_type(L, res->type));
}

/* Python _op_vector_trunc_f32_to_f16: delegates to _op_vector_trunc_f32_to.
 *
 *     def _op_vector_trunc_f32_to_f16(self, op):
 *         self._op_vector_trunc_f32_to(op)
 */
static void _op_vector_trunc_f32_to(ckc_lower_t* L, const ckc_op_t* op);
static void _op_vector_trunc_f32_to_f16(ckc_lower_t* L, const ckc_op_t* op)
{
    _op_vector_trunc_f32_to(L, op);
}

/* Python _op_vector_trunc_f32_to: plain packed fptrunc to the result type.
 *
 *     (v,) = op.operands
 *     in_ty = _llvm_type(v.type); out_ty = _llvm_type(op.result.type)
 *     emit f"  {res} = fptrunc {in_ty} {v} to {out_ty}"
 */
static void _op_vector_trunc_f32_to(ckc_lower_t* L, const ckc_op_t* op)
{
    const ckc_value_t* res = ll_result(op);
    if(!ckc_ll_live(L) || !res || op->num_operands < 1)
    {
        return;
    }
    const ckc_value_t* v = op->operands[0];
    ckc_ll_emitf(L,
                 "  %s = fptrunc %s %s to %s",
                 res->name,
                 ckc_ll_llvm_type(L, v->type),
                 ckc_ll_operand(L, v),
                 ckc_ll_llvm_type(L, res->type));
}

/* ======================================================================== */
/* tile.* barriers / sync / scheduling                                      */
/* ======================================================================== */

/* Python _op_tile_sync: s_waitcnt(vmcnt0,lgkmcnt0) + s_barrier, with the
 * unroll trailing-sync elision check. */
static void _op_tile_sync(ckc_lower_t* L, const ckc_op_t* op)
{
    if(!ckc_ll_live(L))
    {
        return;
    }
    if(L->unroll_elide_sync_op && L->unroll_elide_sync_op == op)
    {
        return; /* skip the trailing sync in a non-final unrolled iteration */
    }
    int mask = ll_backend_waitcnt(L, 0, -1, 0);
    ckc_ll_need(L, "s.waitcnt");
    ckc_ll_need(L, "s.barrier");
    ckc_ll_emitf(L, "  call void @llvm.amdgcn.s.waitcnt(i32 %d)", mask);
    ckc_ll_emit(L, " call void @llvm.amdgcn.s.barrier()");
}

/* Python _op_tile_s_barrier_bare: bare s_barrier, no implicit waitcnt. */
static void _op_tile_s_barrier_bare(ckc_lower_t* L, const ckc_op_t* op)
{
    (void)op;
    if(!ckc_ll_live(L))
    {
        return;
    }
    ckc_ll_need(L, "s.barrier");
    ckc_ll_emit(L, " call void @llvm.amdgcn.s.barrier()");
}

/* Python _op_tile_sync_half_block: cond-branch so only the then-branch hits
 * the s_barrier. */
static void _op_tile_sync_half_block(ckc_lower_t* L, const ckc_op_t* op)
{
    if(!ckc_ll_live(L) || op->num_operands < 1)
    {
        return;
    }
    const ckc_value_t* sel = op->operands[0];
    ckc_ll_need(L, "s.barrier");
    const char* i1_name = ckc_ll_fresh(L, "half_pred");
    ckc_ll_emitf(L, "  %s = icmp ne i32 %s, 0", i1_name, ckc_ll_operand(L, sel));
    ckc_ll_block_t* then_blk = ckc_ll_new_block(L, "hb_then");
    ckc_ll_block_t* join_blk = ckc_ll_new_block(L, "hb_join");
    /* The block before the two we just pushed is the predecessor; terminate it
     * with the conditional branch. */
    int prev_idx             = ckc_ll_block_count(L) - 3;
    ckc_ll_block_t* prev_blk = ckc_ll_block_at(L, prev_idx);
    if(prev_blk && then_blk && join_blk)
    {
        ckc_ll_block_emitf(L,
                           prev_blk,
                           "  br i1 %s, label %%%s, label %%%s",
                           i1_name,
                           then_blk->label,
                           join_blk->label);
        prev_blk->terminated = true;
        ckc_ll_block_emit(L, then_blk, " call void @llvm.amdgcn.s.barrier()");
        ckc_ll_block_emitf(L, then_blk, "  br label %%%s", join_blk->label);
        then_blk->terminated = true;
    }
    /* Subsequent ops fall into join_blk (now _current). */
}

/* Python _op_tile_sync_lds_only: drain LDS (lgkmcnt0) but not VMEM. */
static void _op_tile_sync_lds_only(ckc_lower_t* L, const ckc_op_t* op)
{
    (void)op;
    if(!ckc_ll_live(L))
    {
        return;
    }
    int mask = ll_backend_waitcnt(L, -1, -1, 0);
    ckc_ll_need(L, "s.waitcnt");
    ckc_ll_need(L, "s.barrier");
    ckc_ll_emitf(L, "  call void @llvm.amdgcn.s.waitcnt(i32 %d)", mask);
    ckc_ll_emit(L, " call void @llvm.amdgcn.s.barrier()");
}

/* Python _op_tile_s_waitcnt: explicit s_waitcnt from attrs. */
static void _op_tile_s_waitcnt(ckc_lower_t* L, const ckc_op_t* op)
{
    if(!ckc_ll_live(L))
    {
        return;
    }
    ckc_ll_need(L, "s.waitcnt");
    int64_t vm = -1, lk = -1, ec = -1;
    ckc_attr_get_int(&op->attrs, "vmcnt", &vm);
    ckc_attr_get_int(&op->attrs, "lgkmcnt", &lk);
    ckc_attr_get_int(&op->attrs, "expcnt", &ec);
    int mask = ll_backend_waitcnt(L, (int)vm, (int)ec, (int)lk);
    ckc_ll_emitf(L, "  call void @llvm.amdgcn.s.waitcnt(i32 %d)", mask);
}

/* Python _op_tile_iglp_opt. */
static void _op_tile_iglp_opt(ckc_lower_t* L, const ckc_op_t* op)
{
    if(!ckc_ll_live(L))
    {
        return;
    }
    ckc_ll_need(L, "iglp.opt");
    int64_t level = 0;
    ckc_attr_get_int(&op->attrs, "level", &level);
    ckc_ll_emitf(L, "  call void @llvm.amdgcn.iglp.opt(i32 %lld)", (long long)level);
}

/* Python _op_tile_sched_barrier. */
static void _op_tile_sched_barrier(ckc_lower_t* L, const ckc_op_t* op)
{
    if(!ckc_ll_live(L))
    {
        return;
    }
    ckc_ll_need(L, "sched.barrier");
    int64_t mask = 0;
    ckc_attr_get_int(&op->attrs, "mask", &mask);
    ckc_ll_emitf(L, "  call void @llvm.amdgcn.sched.barrier(i32 %lld)", (long long)mask);
}

/* Python _op_tile_sched_group_barrier. */
static void _op_tile_sched_group_barrier(ckc_lower_t* L, const ckc_op_t* op)
{
    if(!ckc_ll_live(L))
    {
        return;
    }
    ckc_ll_need(L, "sched.group.barrier");
    int64_t mask = 0, count = 0, group = 0;
    ckc_attr_get_int(&op->attrs, "mask", &mask);
    ckc_attr_get_int(&op->attrs, "count", &count);
    ckc_attr_get_int(&op->attrs, "group", &group);
    ckc_ll_emitf(L,
                 "  call void @llvm.amdgcn.sched.group.barrier("
                 "i32 %lld, i32 %lld, i32 %lld)",
                 (long long)mask,
                 (long long)count,
                 (long long)group);
}

/* Python _op_tile_s_setprio. */
static void _op_tile_s_setprio(ckc_lower_t* L, const ckc_op_t* op)
{
    if(!ckc_ll_live(L))
    {
        return;
    }
    ckc_ll_need(L, "s.setprio");
    int64_t level = 0;
    ckc_attr_get_int(&op->attrs, "level", &level);
    ckc_ll_emitf(L, "  call void @llvm.amdgcn.s.setprio(i16 %lld)", (long long)level);
}

/* ======================================================================== */
/* scf.* / cf.* control flow                                                */
/* ======================================================================== */

/* Fetch the i-th iter_args metadata map's "name"/"type" string fields. The
 * Python iter_meta is op.attrs["iter_args"], a list of {"name","type"} dicts;
 * here it is a CKC_ATTR_LIST whose items are small attr maps. Returns false if
 * the list is absent or i is out of range. */
static bool ll_iter_meta(const ckc_op_t* op, int i, const char** out_name, const char** out_type)
{
    const ckc_attr_value_t* v = ckc_attr_get(&op->attrs, "iter_args");
    if(!v || v->kind != CKC_ATTR_LIST || i < 0 || i >= v->u.list.count)
    {
        return false;
    }
    const ckc_attr_map_t* m = v->u.list.items[i];
    if(!m)
    {
        return false;
    }
    if(out_name)
    {
        *out_name = ckc_attr_get_str(m, "name");
    }
    if(out_type)
    {
        *out_type = ckc_attr_get_str(m, "type");
    }
    return true;
}

/* Python _lower_normal_for: header / body / latch / exit CFG with phi nodes.
 *
 *     num_iter = int(attrs.get("num_iter_args", 0))
 *     lower, upper, step = operands[:3]; iter_inits = operands[3:3+num_iter]
 *     iter_meta = attrs.get("iter_args", []); iv_name = attrs["iv"]
 *     iv_ty = _llvm_type(lower.type)
 *     pred_block = self._current().label
 *     header = self._new_block("for.header")
 *     self._blocks[-2].emit(f"  br label %{header.label}"); blocks[-2].terminated = True
 *     header.emit(f"  {iv_name} = phi {iv_ty} [ {lower}, %{pred_block} ], "
 *                 f"[ %iv.next.{header.label}, %FOR_LATCH ]")
 *     for meta, init in zip(iter_meta, iter_inits):
 *         ll_ty = _llvm_type_from_name(meta["type"])
 *         header.emit(f"  {meta['name']} = phi {ll_ty} [ {init}, %{pred_block} ], "
 *                     f"[ {meta['name']}.next.{header.label}, %FOR_LATCH ]")
 *     cmp = fresh("cmp")
 *     header.emit(f"  {cmp} = icmp slt {iv_ty} {iv_name}, {upper}")
 *     body = self._new_block("for.body")
 *     header.emit(f"  br i1 {cmp}, label %{body.label}, label %FOR_EXIT")
 *     header.terminated = True
 *     self._yield_stack.append([]); self.lower_region(op.regions[0])
 *     last_body = self._current(); latch = self._new_block("for.latch")
 *     last_body.emit(f"  br label %{latch.label}"); last_body.terminated = True
 *     yielded = self._yield_stack.pop()  # len must == num_iter
 *     iv_next = f"%iv.next.{header.label}"
 *     latch.emit(f"  {iv_next} = add nsw {iv_ty} {iv_name}, {step}")
 *     for meta, yld in zip(iter_meta, yielded):
 *         ll_ty = _llvm_type_from_name(meta["type"])
 *         latch.emit(f"  {meta['name']}.next.{header.label} = bitcast {ll_ty} {yld} to {ll_ty}")
 *     latch.emit(f"  br label %{header.label}"); latch.terminated = True
 *     exit_blk = self._new_block("for.exit")
 *     for line in header.lines: replace %FOR_LATCH->%latch ; %FOR_EXIT->%exit
 *     for meta, result in zip(iter_meta, op.results):
 *         ll_ty = _llvm_type_from_name(meta["type"])
 *         exit_blk.emit(f"  {result.name} = bitcast {ll_ty} {meta['name']} to {ll_ty}")
 */
void ckc_ll_lower_normal_for(ckc_lower_t* L, const ckc_op_t* op)
{
    if(!ckc_ll_live(L) || op->num_operands < 3)
    {
        return;
    }
    int64_t num_iter = 0;
    ckc_attr_get_int(&op->attrs, "num_iter_args", &num_iter);
    const ckc_value_t* lower = op->operands[0];
    const ckc_value_t* upper = op->operands[1];
    const ckc_value_t* step  = op->operands[2];
    const char* iv_name      = ckc_attr_get_str(&op->attrs, "iv");
    if(!iv_name)
    {
        ckc_ll_fail(L, CKC_ERR_KEY, "scf.for: missing iv attr");
    }
    const char* iv_ty = ckc_ll_llvm_type(L, lower->type);

    ckc_ll_block_t* pred   = ckc_ll_current(L);
    const char* pred_block = pred ? pred->label : "";

    ckc_ll_block_t* header = ckc_ll_new_block(L, "for.header");
    /* self._blocks[-2] is the block just before the freshly-pushed header. */
    ckc_ll_block_t* prev_blk = ckc_ll_block_at(L, ckc_ll_block_count(L) - 2);
    if(prev_blk && header)
    {
        ckc_ll_block_emitf(L, prev_blk, "  br label %%%s", header->label);
        prev_blk->terminated = true;
    }

    ckc_ll_block_emitf(L,
                       header,
                       "  %s = phi %s [ %s, %%%s ], "
                       "[ %%iv.next.%s, %%FOR_LATCH ]",
                       iv_name,
                       iv_ty,
                       ckc_ll_operand(L, lower),
                       pred_block,
                       header->label);
    for(int i = 0; i < (int)num_iter; i++)
    {
        const char *mname = NULL, *mtype = NULL;
        if(!ll_iter_meta(op, i, &mname, &mtype) || !mname || !mtype)
        {
            break;
        }
        const ckc_value_t* init = op->operands[3 + i];
        const char* ll_ty       = ckc_ll_llvm_type_from_name(L, mtype);
        ckc_ll_block_emitf(L,
                           header,
                           "  %s = phi %s [ %s, %%%s ], "
                           "[ %s.next.%s, %%FOR_LATCH ]",
                           mname,
                           ll_ty,
                           ckc_ll_operand(L, init),
                           pred_block,
                           mname,
                           header->label);
    }

    const char* cmp = ckc_ll_fresh(L, "cmp");
    ckc_ll_block_emitf(
        L, header, "  %s = icmp slt %s %s, %s", cmp, iv_ty, iv_name, ckc_ll_operand(L, upper));

    ckc_ll_block_t* body = ckc_ll_new_block(L, "for.body");
    ckc_ll_block_emitf(L, header, "  br i1 %s, label %%%s, label %%FOR_EXIT", cmp, body->label);
    header->terminated = true;

    ckc_ll_yield_push(L);
    if(op->num_regions > 0 && op->regions[0])
    {
        ckc_ll_lower_region(L, op->regions[0]);
    }

    ckc_ll_block_t* last_body = ckc_ll_current(L);
    ckc_ll_block_t* latch     = ckc_ll_new_block(L, "for.latch");
    if(last_body && latch)
    {
        ckc_ll_block_emitf(L, last_body, "  br label %%%s", latch->label);
        last_body->terminated = true;
    }

    const char* const* yielded = NULL;
    int n_yield                = 0;
    ckc_ll_yield_pop(L, &yielded, &n_yield);
    if(n_yield != (int)num_iter)
    {
        ckc_ll_fail(
            L, CKC_ERR_VALUE, "scf.for expected %d yielded values, got %d", (int)num_iter, n_yield);
    }

    ckc_ll_block_emitf(L,
                       latch,
                       "  %%iv.next.%s = add nsw %s %s, %s",
                       header->label,
                       iv_ty,
                       iv_name,
                       ckc_ll_operand(L, step));
    for(int i = 0; i < (int)num_iter; i++)
    {
        const char *mname = NULL, *mtype = NULL;
        if(!ll_iter_meta(op, i, &mname, &mtype) || !mname || !mtype)
        {
            break;
        }
        const char* ll_ty = ckc_ll_llvm_type_from_name(L, mtype);
        ckc_ll_block_emitf(L,
                           latch,
                           "  %s.next.%s = bitcast %s %s to %s",
                           mname,
                           header->label,
                           ll_ty,
                           yielded ? yielded[i] : "",
                           ll_ty);
    }
    ckc_ll_block_emitf(L, latch, "  br label %%%s", header->label);
    latch->terminated = true;

    ckc_ll_block_t* exit_blk = ckc_ll_new_block(L, "for.exit");
    if(exit_blk)
    {
        const char* latch_repl = ckc_arena_printf(&L->arena, "%%%s", latch->label);
        const char* exit_repl  = ckc_arena_printf(&L->arena, "%%%s", exit_blk->label);
        ll_block_replace(L, header, "%FOR_LATCH", latch_repl);
        ll_block_replace(L, header, "%FOR_EXIT", exit_repl);
        for(int i = 0; i < (int)num_iter && i < op->num_results; i++)
        {
            const char *mname = NULL, *mtype = NULL;
            if(!ll_iter_meta(op, i, &mname, &mtype) || !mname || !mtype)
            {
                break;
            }
            const char* ll_ty         = ckc_ll_llvm_type_from_name(L, mtype);
            const ckc_value_t* result = op->results[i];
            ckc_ll_block_emitf(
                L, exit_blk, "  %s = bitcast %s %s to %s", result->name, ll_ty, mname, ll_ty);
        }
    }
}

/* One (Value*, saved-name) record for the unrolled-for name renaming. */
typedef struct ll_rename
{
    ckc_value_t* val;
    const char* saved;
} ll_rename_t;

/* Python _lower_unrolled_for: straight-line replication of the body for
 * constant bounds. See the Python source for the full strategy comment; the
 * mechanics ported here are:
 *
 *   - eval constant lower/upper/step -> trip_count = (upper-lower)//step
 *   - current_iter_values[meta.name] = operand(init)
 *   - find the IV + iter-arg Value objects by scanning region operands for
 *     `operand.name == iv_name/meta.name and operand.op == op`
 *   - detect a trailing tile.sync (second-last region op) when
 *     attrs["elide_trailing_barrier"] (default True)
 *   - per iteration: emit `iv_const = add iv_ty 0, iv_value`, rename the IV +
 *     iter-arg objects + ALL body op results to `%base.unrollN`, set the
 *     elide marker on non-final trips, lower the region under a yield frame,
 *     thread yielded values into current_iter_values, restore names
 *   - bind op.results to the final current_iter_values via bitcast
 *
 * The Value.name fields are `const char*`; the Python rebinds them in place, so
 * we cast away const for the duration of each trip and restore afterwards. */
void ckc_ll_lower_unrolled_for(ckc_lower_t* L, const ckc_op_t* op)
{
    if(!ckc_ll_live(L) || op->num_operands < 3 || op->num_regions < 1 || !op->regions[0])
    {
        return;
    }
    int64_t num_iter = 0;
    ckc_attr_get_int(&op->attrs, "num_iter_args", &num_iter);
    const ckc_value_t* lower = op->operands[0];
    const ckc_value_t* upper = op->operands[1];
    const ckc_value_t* step  = op->operands[2];
    const char* iv_name      = ckc_attr_get_str(&op->attrs, "iv");
    if(!iv_name)
    {
        ckc_ll_fail(L, CKC_ERR_KEY, "scf.for: missing iv attr");
    }
    const ckc_region_t* body = op->regions[0];

    /* current_iter_values: name -> current LLVM operand string. Indexed
     * parallel to iter_meta (0..num_iter-1). */
    const char** current_iter_values = NULL;
    if(num_iter > 0)
    {
        current_iter_values =
            (const char**)ckc_arena_alloc(&L->arena, sizeof(const char*) * (size_t)num_iter);
        if(!current_iter_values)
        {
            ckc_ll_fail(L, CKC_ERR_OOM, "unrolled_for: arena OOM");
        }
        for(int i = 0; i < (int)num_iter; i++)
        {
            current_iter_values[i] = ckc_ll_operand(L, op->operands[3 + i]);
        }
    }

    int64_t lower_val = ckc_ll_eval_constant(L, lower);
    int64_t upper_val = ckc_ll_eval_constant(L, upper);
    int64_t step_val  = ckc_ll_eval_constant(L, step);
    if(step_val == 0)
    {
        ckc_ll_fail(L, CKC_ERR_VALUE, "unrolled_for: zero step");
    }
    int64_t trip_count = (upper_val - lower_val) / step_val;

    /* Find the IV + iter-arg Value objects referenced by body operands. */
    ckc_value_t* iv_value_obj = NULL;
    ckc_value_t** iter_value_objs =
        num_iter > 0
            ? (ckc_value_t**)ckc_arena_alloc(&L->arena, sizeof(ckc_value_t*) * (size_t)num_iter)
            : NULL;
    int iter_value_count = 0;
    for(int bi = 0; bi < body->num_ops; bi++)
    {
        const ckc_op_t* bop = body->ops[bi];
        for(int oi = 0; oi < bop->num_operands; oi++)
        {
            ckc_value_t* operand = bop->operands[oi];
            if(!operand || !operand->name)
            {
                continue;
            }
            if(operand->op == op && strcmp(operand->name, iv_name) == 0)
            {
                iv_value_obj = operand;
            }
            for(int mi = 0; mi < (int)num_iter; mi++)
            {
                const char* mname = NULL;
                if(!ll_iter_meta(op, mi, &mname, NULL) || !mname)
                {
                    continue;
                }
                if(operand->op == op && strcmp(operand->name, mname) == 0)
                {
                    bool present = false;
                    for(int k = 0; k < iter_value_count; k++)
                    {
                        if(iter_value_objs[k] == operand)
                        {
                            present = true;
                            break;
                        }
                    }
                    if(!present && iter_value_objs)
                    {
                        iter_value_objs[iter_value_count++] = operand;
                    }
                }
            }
        }
    }

    /* Trailing tile.sync detection (second-last body op). */
    bool elide_enabled = ckc_attr_get_bool(&op->attrs, "elide_trailing_barrier", true);
    const ckc_op_t* trailing_sync_op = NULL;
    if(elide_enabled && body->num_ops >= 2)
    {
        const ckc_op_t* second_last = body->ops[body->num_ops - 2];
        if(second_last->opcode == CKC_OP_TILE_SYNC)
        {
            trailing_sync_op = second_last;
        }
    }

    const char* iv_ty = ckc_ll_llvm_type(L, lower->type);

    for(int64_t iteration = 0; iteration < trip_count; iteration++)
    {
        int64_t iv_value          = lower_val + iteration * step_val;
        const char* iv_const_name = ckc_ll_fresh(L, "iv_const");
        ckc_ll_emitf(L, "  %s = add %s 0, %lld", iv_const_name, iv_ty, (long long)iv_value);

        /* Collect (Value*, saved-name) for restoration after this trip. */
        int max_renames = 1 + iter_value_count;
        for(int bi = 0; bi < body->num_ops; bi++)
        {
            max_renames += body->ops[bi]->num_results;
        }
        ll_rename_t* renames = (ll_rename_t*)ckc_arena_alloc(
            &L->arena, sizeof(ll_rename_t) * (size_t)(max_renames > 0 ? max_renames : 1));
        if(!renames)
        {
            ckc_ll_fail(L, CKC_ERR_OOM, "unrolled_for: arena OOM");
        }
        int n_renames = 0;

        if(iv_value_obj)
        {
            renames[n_renames].val   = iv_value_obj;
            renames[n_renames].saved = iv_value_obj->name;
            n_renames++;
            iv_value_obj->name = iv_const_name;
        }
        /* Rename iter-arg objects to their current operand string. */
        for(int k = 0; k < iter_value_count; k++)
        {
            ckc_value_t* vo          = iter_value_objs[k];
            renames[n_renames].val   = vo;
            renames[n_renames].saved = vo->name;
            n_renames++;
            for(int mi = 0; mi < (int)num_iter; mi++)
            {
                const char* mname = NULL;
                if(!ll_iter_meta(op, mi, &mname, NULL) || !mname)
                {
                    continue;
                }
                if(strcmp(vo->name, mname) == 0)
                {
                    vo->name = current_iter_values[mi];
                    break;
                }
            }
        }
        /* Rename all body op results to %base.unrollN. */
        for(int bi = 0; bi < body->num_ops; bi++)
        {
            const ckc_op_t* bop = body->ops[bi];
            for(int ri = 0; ri < bop->num_results; ri++)
            {
                ckc_value_t* result      = bop->results[ri];
                renames[n_renames].val   = result;
                renames[n_renames].saved = result->name;
                n_renames++;
                const char* base = result->name;
                if(base && base[0] == '%')
                {
                    base++;
                }
                result->name = ckc_arena_printf(
                    &L->arena, "%%%s.unroll%lld", base ? base : "", (long long)iteration);
            }
        }

        bool is_final           = (iteration == trip_count - 1);
        L->unroll_elide_sync_op = (trailing_sync_op && !is_final) ? trailing_sync_op : NULL;

        ckc_ll_yield_push(L);
        ckc_ll_lower_region(L, body);
        const char* const* yielded = NULL;
        int n_yield                = 0;
        ckc_ll_yield_pop(L, &yielded, &n_yield);

        L->unroll_elide_sync_op = NULL;

        /* Restore original names. */
        for(int k = 0; k < n_renames; k++)
        {
            renames[k].val->name = renames[k].saved;
        }

        if(n_yield != (int)num_iter)
        {
            ckc_ll_fail(L,
                        CKC_ERR_VALUE,
                        "scf.for expected %d yielded values, got %d",
                        (int)num_iter,
                        n_yield);
        }
        for(int i = 0; i < (int)num_iter; i++)
        {
            current_iter_values[i] = ckc_arena_strdup(&L->arena, yielded ? yielded[i] : "");
        }
    }

    for(int i = 0; i < (int)num_iter && i < op->num_results; i++)
    {
        const char *mname = NULL, *mtype = NULL;
        if(!ll_iter_meta(op, i, &mname, &mtype) || !mtype)
        {
            break;
        }
        const char* ll_ty         = ckc_ll_llvm_type_from_name(L, mtype);
        const ckc_value_t* result = op->results[i];
        ckc_ll_emitf(
            L, "  %s = bitcast %s %s to %s", result->name, ll_ty, current_iter_values[i], ll_ty);
    }
}

/* Python _op_scf_for: pick unrolled vs normal lowering. */
static void _op_scf_for(ckc_lower_t* L, const ckc_op_t* op)
{
    if(!ckc_ll_live(L) || op->num_operands < 3)
    {
        return;
    }
    bool unroll              = ckc_attr_get_bool(&op->attrs, "unroll", false);
    const ckc_value_t* lower = op->operands[0];
    const ckc_value_t* upper = op->operands[1];
    const ckc_value_t* step  = op->operands[2];
    if(unroll && ckc_ll_is_constant(lower) && ckc_ll_is_constant(upper) && ckc_ll_is_constant(step))
    {
        ckc_ll_lower_unrolled_for(L, op);
    }
    else
    {
        ckc_ll_lower_normal_for(L, op);
    }
}

/* Replace every occurrence of `tok` with `repl` across `blk`'s lines, in place
 * (arena-allocating the rewritten lines). Mirrors the Python
 * `cur.lines[i] = line.replace(tok, repl)` back-patch loop. */
static void ll_block_replace(ckc_lower_t* L, ckc_ll_block_t* blk, const char* tok, const char* repl)
{
    if(!blk || !tok || !repl)
    {
        return;
    }
    size_t tok_len  = strlen(tok);
    size_t repl_len = strlen(repl);
    for(size_t li = 0; li < blk->lines.len; li++)
    {
        const char* line = blk->lines.data[li];
        if(!line)
        {
            continue;
        }
        /* Count occurrences first to size the output. */
        int n = 0;
        for(const char* p = strstr(line, tok); p; p = strstr(p + tok_len, tok))
        {
            n++;
        }
        if(n == 0)
        {
            continue;
        }
        size_t out_len = strlen(line) + (size_t)n * (repl_len - tok_len);
        char* out      = (char*)ckc_arena_alloc(&L->arena, out_len + 1);
        if(!out)
        {
            ckc_ll_fail(L, CKC_ERR_OOM, "scf.if: back-patch arena OOM");
        }
        char* w       = out;
        const char* s = line;
        const char* p;
        while((p = strstr(s, tok)) != NULL)
        {
            size_t pre = (size_t)(p - s);
            memcpy(w, s, pre);
            w += pre;
            memcpy(w, repl, repl_len);
            w += repl_len;
            s = p + tok_len;
        }
        strcpy(w, s);
        blk->lines.data[li] = out;
    }
}

/* Python _op_scf_if: i1 cond branch into a fresh then-block, lower the
 * then-region, then a join block, with a deferred %IF_END placeholder.
 *
 *     (cond,) = op.operands; then_region = op.regions[0]
 *     cur = self._current()
 *     then_blk = self._new_block("if.then")
 *     cur.emit(f"  br i1 {cond}, label %{then_blk.label}, label %IF_END")
 *     cur.terminated = True
 *     self.lower_region(then_region)
 *     then_last = self._current()
 *     end_blk = self._new_block("if.end")
 *     if not then_last.terminated:
 *         then_last.emit(f"  br label %{end_blk.label}"); then_last.terminated = True
 *     for i, line in enumerate(cur.lines):
 *         cur.lines[i] = line.replace("%IF_END", f"%{end_blk.label}")
 *
 * Note: the Python scf.if lowers ONLY regions[0]; there is no else region. */
static void _op_scf_if(ckc_lower_t* L, const ckc_op_t* op)
{
    if(!ckc_ll_live(L) || op->num_operands < 1 || op->num_regions < 1)
    {
        return;
    }
    const ckc_value_t* cond  = op->operands[0];
    ckc_ll_block_t* cur      = ckc_ll_current(L);
    ckc_ll_block_t* then_blk = ckc_ll_new_block(L, "if.then");
    if(cur && then_blk)
    {
        ckc_ll_block_emitf(L,
                           cur,
                           "  br i1 %s, label %%%s, label %%IF_END",
                           ckc_ll_operand(L, cond),
                           then_blk->label);
        cur->terminated = true;
    }
    ckc_ll_lower_region(L, op->regions[0]);
    ckc_ll_block_t* then_last = ckc_ll_current(L);
    ckc_ll_block_t* end_blk   = ckc_ll_new_block(L, "if.end");
    if(then_last && !then_last->terminated && end_blk)
    {
        ckc_ll_block_emitf(L, then_last, "  br label %%%s", end_blk->label);
        then_last->terminated = true;
    }
    if(cur && end_blk)
    {
        const char* repl = ckc_arena_printf(&L->arena, "%%%s", end_blk->label);
        ll_block_replace(L, cur, "%IF_END", repl);
    }
}

/* Python _op_scf_yield: record yielded operand strings into the top frame. */
static void _op_scf_yield(ckc_lower_t* L, const ckc_op_t* op)
{
    if(!ckc_ll_live(L) || L->yield_stack.len == 0)
    {
        return;
    }
    for(int i = 0; i < op->num_operands; i++)
    {
        ckc_ll_yield_record(L, ckc_ll_operand(L, op->operands[i]));
    }
}

/* Python _op_cf_return: terminate the current block with `ret void`.
 *
 *     def _op_cf_return(self, op):
 *         self._current().emit(" ret void")
 *         self._current().terminated = True
 *
 * Note the SINGLE leading space in " ret void" -- it matches the Python text
 * byte-for-byte (the no-op finalize() terminator uses the same spelling). */
static void _op_cf_return(ckc_lower_t* L, const ckc_op_t* op)
{
    (void)op;
    if(!ckc_ll_live(L))
    {
        return;
    }
    ckc_ll_emit(L, " ret void");
    ckc_ll_block_t* cur = ckc_ll_current(L);
    if(cur)
    {
        cur->terminated = true;
    }
}

/* ----------------------------------------------------------- registration */

void ckc_ll_register_vector(void)
{
    /* vector.* */
    ckc_ll_set_handler(CKC_OP_VECTOR_ADD, _op_vector_add);
    ckc_ll_set_handler(CKC_OP_VECTOR_SUB, _op_vector_sub);
    ckc_ll_set_handler(CKC_OP_VECTOR_MUL, _op_vector_mul);
    ckc_ll_set_handler(CKC_OP_VECTOR_AND, _op_vector_and);
    ckc_ll_set_handler(CKC_OP_VECTOR_OR, _op_vector_or);
    ckc_ll_set_handler(CKC_OP_VECTOR_SHL, _op_vector_shl);
    ckc_ll_set_handler(CKC_OP_VECTOR_LSHR, _op_vector_lshr);
    ckc_ll_set_handler(CKC_OP_VECTOR_SMAX, _op_vector_smax);
    ckc_ll_set_handler(CKC_OP_VECTOR_SMIN, _op_vector_smin);
    ckc_ll_set_handler(CKC_OP_VECTOR_MAX, _op_vector_max);
    ckc_ll_set_handler(CKC_OP_VECTOR_FMA, _op_vector_fma);
    ckc_ll_set_handler(CKC_OP_VECTOR_SUM, _op_vector_sum);
    ckc_ll_set_handler(CKC_OP_VECTOR_REDUCE_MAX, _op_vector_reduce_max);
    ckc_ll_set_handler(CKC_OP_VECTOR_SPLAT, _op_vector_splat);
    ckc_ll_set_handler(CKC_OP_VECTOR_SELECT, _op_vector_select);
    ckc_ll_set_handler(CKC_OP_VECTOR_CMP, _op_vector_cmp);
    ckc_ll_set_handler(CKC_OP_VECTOR_TRUNC, _op_vector_trunc);
    ckc_ll_set_handler(CKC_OP_VECTOR_SEXT, _op_vector_sext);
    ckc_ll_set_handler(CKC_OP_VECTOR_TRUNC_F32_TO_F16, _op_vector_trunc_f32_to_f16);
    ckc_ll_set_handler(CKC_OP_VECTOR_TRUNC_F32_TO, _op_vector_trunc_f32_to);
    ckc_ll_set_handler(CKC_OP_VECTOR_BITCAST, _op_vector_bitcast);
    ckc_ll_set_handler(CKC_OP_VECTOR_EXTRACT, _op_vector_extract);
    ckc_ll_set_handler(CKC_OP_VECTOR_INSERT, _op_vector_insert);
    ckc_ll_set_handler(CKC_OP_VECTOR_PACK, _op_vector_pack);
    ckc_ll_set_handler(CKC_OP_VECTOR_CONCAT, _op_vector_concat);

    /* tile.* -- barriers / scheduling */
    ckc_ll_set_handler(CKC_OP_TILE_SYNC, _op_tile_sync);
    ckc_ll_set_handler(CKC_OP_TILE_SYNC_HALF_BLOCK, _op_tile_sync_half_block);
    ckc_ll_set_handler(CKC_OP_TILE_SYNC_LDS_ONLY, _op_tile_sync_lds_only);
    ckc_ll_set_handler(CKC_OP_TILE_S_BARRIER_BARE, _op_tile_s_barrier_bare);
    ckc_ll_set_handler(CKC_OP_TILE_S_WAITCNT, _op_tile_s_waitcnt);
    ckc_ll_set_handler(CKC_OP_TILE_S_SETPRIO, _op_tile_s_setprio);
    ckc_ll_set_handler(CKC_OP_TILE_IGLP_OPT, _op_tile_iglp_opt);
    ckc_ll_set_handler(CKC_OP_TILE_SCHED_BARRIER, _op_tile_sched_barrier);
    ckc_ll_set_handler(CKC_OP_TILE_SCHED_GROUP_BARRIER, _op_tile_sched_group_barrier);

    /* scf.* / cf.* control flow */
    ckc_ll_set_handler(CKC_OP_SCF_FOR, _op_scf_for);
    ckc_ll_set_handler(CKC_OP_SCF_IF, _op_scf_if);
    ckc_ll_set_handler(CKC_OP_SCF_YIELD, _op_scf_yield);
    ckc_ll_set_handler(CKC_OP_CF_RETURN, _op_cf_return);
}

} /* namespace ckc */
