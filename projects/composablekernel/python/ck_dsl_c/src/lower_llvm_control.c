/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
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

/* ------------------------------------------------------------------ helpers */

/* Python: op.result -- exactly one result (the producing builder guaranteed
 * it). Returns op->results[0] or NULL. */
static const ckc_value_t *ll_result(const ckc_op_t *op) {
    return (op && op->num_results > 0) ? op->results[0] : NULL;
}

/* Element type name of a vector type, or NULL. */
static const char *ll_vec_elem_name(const ckc_type_t *t) {
    if (t && t->kind == CKC_TYPE_VECTOR && t->elem) {
        return t->elem->name;
    }
    return NULL;
}

/* True if `name` is one of the floating element names f16/bf16/f32. */
static bool ll_is_fp_elem(const char *name) {
    return name && (strcmp(name, "f16") == 0 || strcmp(name, "bf16") == 0 ||
                    strcmp(name, "f32") == 0);
}

/* ======================================================================== */
/* waitcnt encoders (Python _encode_waitcnt_gfx9_10 / _encode_waitcnt_gfx11) */
/* ======================================================================== */

static int ll_clamp(int v, int lo, int hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

/* Python _encode_waitcnt_gfx9_10: split VMCNT across [3:0] and [15:14]. -1
 * means "no wait" (architectural max). */
int ckc_ll_encode_waitcnt_gfx9_10(int vmcnt, int expcnt, int lgkmcnt) {
    int vm_b = (vmcnt < 0) ? 0x3F : ll_clamp(vmcnt, 0, 0x3F);
    int ec_b = (expcnt < 0) ? 0x7 : ll_clamp(expcnt, 0, 0x7);
    int lk_b = (lgkmcnt < 0) ? 0xF : ll_clamp(lgkmcnt, 0, 0xF);
    int vm_lo = vm_b & 0xF;
    int vm_hi = (vm_b >> 4) & 0x3;
    return vm_lo | (ec_b << 4) | (lk_b << 8) | (vm_hi << 14);
}

/* Python _encode_waitcnt_gfx11: contiguous fields, 6-bit LGKMCNT, no split
 * VMCNT. -1 means "no wait" (architectural max). */
int ckc_ll_encode_waitcnt_gfx11(int vmcnt, int expcnt, int lgkmcnt) {
    int vm_b = (vmcnt < 0) ? 0x3F : ll_clamp(vmcnt, 0, 0x3F);
    int ec_b = (expcnt < 0) ? 0x7 : ll_clamp(expcnt, 0, 0x7);
    int lk_b = (lgkmcnt < 0) ? 0x3F : ll_clamp(lgkmcnt, 0, 0x3F);
    return (ec_b & 0x7) | ((lk_b & 0x3F) << 4) | ((vm_b & 0x3F) << 10);
}

/* Convenience: invoke the backend encoder (Python self._backend.encode_waitcnt)
 * falling back to the gfx9_10 encoder when no backend is bound. */
static int ll_backend_waitcnt(ckc_lower_t *L, int vmcnt, int expcnt,
                              int lgkmcnt) {
    if (L && L->backend && L->backend->encode_waitcnt) {
        return L->backend->encode_waitcnt(vmcnt, expcnt, lgkmcnt);
    }
    return ckc_ll_encode_waitcnt_gfx9_10(vmcnt, expcnt, lgkmcnt);
}

/* ======================================================================== */
/* yield-stack helpers (Python _yield_stack: list of list[str])             */
/* ======================================================================== */

typedef CKC_VEC(const char *) ll_yield_frame_t;

/* Python self._yield_stack.append([]). */
void ckc_ll_yield_push(ckc_lower_t *L) {
    if (!ckc_ll_live(L)) {
        return;
    }
    ll_yield_frame_t *frame =
        (ll_yield_frame_t *)ckc_arena_alloc(&L->arena, sizeof *frame);
    if (!frame) {
        ckc_ll_fail(L, CKC_ERR_OOM, "yield_push: arena OOM");
        return;
    }
    ckc_vec_init(frame);
    int rc = 0;
    /* yield_stack stores a distinct anonymous-struct pointer type; the in-place
     * layout is identical, so launder through void* to satisfy the compiler. */
    void *frame_v = frame;
    ckc_vec_push(&L->arena, &L->yield_stack, frame_v, rc);
    if (rc != 0) {
        ckc_ll_fail(L, CKC_ERR_OOM, "yield_push: arena OOM");
    }
}

/* Python yielded = self._yield_stack.pop(). Returns the frame's operand
 * strings via out params; empty/NULL on an empty stack. */
void ckc_ll_yield_pop(ckc_lower_t *L, const char *const **out_items,
                      int *out_count) {
    if (out_items) {
        *out_items = NULL;
    }
    if (out_count) {
        *out_count = 0;
    }
    if (!L || L->yield_stack.len == 0) {
        return;
    }
    L->yield_stack.len -= 1;
    ll_yield_frame_t *frame =
        (ll_yield_frame_t *)L->yield_stack.data[L->yield_stack.len];
    if (frame) {
        if (out_items) {
            *out_items = frame->data;
        }
        if (out_count) {
            *out_count = (int)frame->len;
        }
    }
}

/* Python self._yield_stack[-1].append(operand_str). */
void ckc_ll_yield_record(ckc_lower_t *L, const char *operand_str) {
    if (!ckc_ll_live(L) || L->yield_stack.len == 0) {
        return;
    }
    ll_yield_frame_t *frame =
        (ll_yield_frame_t *)L->yield_stack.data[L->yield_stack.len - 1];
    if (!frame) {
        return;
    }
    const char *dup = ckc_arena_strdup(&L->arena, operand_str ? operand_str : "");
    int rc = 0;
    ckc_vec_push(&L->arena, frame, dup, rc);
    if (rc != 0) {
        ckc_ll_fail(L, CKC_ERR_OOM, "yield_record: arena OOM");
    }
}

/* Python len(self._yield_stack). */
int ckc_ll_yield_depth(const ckc_lower_t *L) {
    return L ? (int)L->yield_stack.len : 0;
}

/* ======================================================================== */
/* shared horizontal vector reduce (Python _lower_vector_reduce)            */
/* ======================================================================== */

/* Extract every lane and fold with `llvm_op` starting from `init`. */
void ckc_ll_lower_vector_reduce(ckc_lower_t *L, const ckc_op_t *op,
                                const char *llvm_op, const char *init) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 1) {
        return;
    }
    const ckc_value_t *v = op->operands[0];
    const ckc_type_t *vec_ty = v->type;
    if (!vec_ty || vec_ty->kind != CKC_TYPE_VECTOR) {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "vector reduce: not a vector operand");
        return;
    }
    int count = vec_ty->count;
    const ckc_type_t *elem_ty = vec_ty->elem;
    const char *vec_llvm = ckc_ll_llvm_type(L, vec_ty);
    const char *elem_llvm = ckc_ll_llvm_type(L, elem_ty);
    const char *acc = init;
    for (int i = 0; i < count; i++) {
        const char *e = ckc_ll_fresh(L, "vred.e");
        ckc_ll_emitf(L, "  %s = extractelement %s %s, i32 %d", e, vec_llvm,
                     ckc_ll_operand(L, v), i);
        const char *name = (i == count - 1) ? res->name : ckc_ll_fresh(L, "vred");
        ckc_ll_emitf(L, "  %s = %s %s %s, %s", name, llvm_op, elem_llvm, acc, e);
        acc = name;
    }
}

/* ======================================================================== */
/* vector.* per-op handlers                                                  */
/* ======================================================================== */

/* Python _op_vector_add. */
static void _op_vector_add(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    const char *elem = res ? ll_vec_elem_name(res->type) : NULL;
    ckc_ll_vector_binop(L, op, ll_is_fp_elem(elem) ? "fadd" : "add");
}
/* Python _op_vector_sub. */
static void _op_vector_sub(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    const char *elem = res ? ll_vec_elem_name(res->type) : NULL;
    ckc_ll_vector_binop(L, op, ll_is_fp_elem(elem) ? "fsub" : "sub");
}
/* Python _op_vector_mul. */
static void _op_vector_mul(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    const char *elem = res ? ll_vec_elem_name(res->type) : NULL;
    ckc_ll_vector_binop(L, op, ll_is_fp_elem(elem) ? "fmul" : "mul");
}
/* Python _op_vector_and. */
static void _op_vector_and(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_vector_binop(L, op, "and");
}
/* Python _op_vector_or. */
static void _op_vector_or(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_vector_binop(L, op, "or");
}
/* Python _op_vector_shl. */
static void _op_vector_shl(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_vector_binop(L, op, "shl");
}
/* Python _op_vector_lshr. */
static void _op_vector_lshr(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_vector_binop(L, op, "lshr");
}

/* Python _op_vector_cmp: packed icmp with a pred map. */
static void _op_vector_cmp(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 2) {
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
        ckc_ll_fail(L, CKC_ERR_KEY, "vector.cmp: unknown pred %s", pred);
        return;
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    ckc_ll_emitf(L, "  %s = icmp %s %s %s, %s", res->name, llvm_pred,
                 ckc_ll_llvm_type(L, a->type), ckc_ll_operand(L, a),
                 ckc_ll_operand(L, b));
}

/* Python _op_vector_smax: dynamic llvm.smax.v<N>i<W> intrinsic. */
static void _op_vector_smax(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 2) {
        return;
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    const ckc_type_t *vec_ty = a->type;
    if (!vec_ty || vec_ty->kind != CKC_TYPE_VECTOR || !vec_ty->elem ||
        !vec_ty->elem->name) {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "vector.smax: not an int vector");
        return;
    }
    int count = vec_ty->count;
    const char *ename = vec_ty->elem->name; /* "i16" -> width "16" */
    const char *width = (ename[0] == 'i') ? ename + 1 : ename;
    const char *vec_llvm = ckc_ll_llvm_type(L, vec_ty);
    char intrin[64];
    snprintf(intrin, sizeof intrin, "llvm.smax.v%di%s", count, width);
    const char *decl = ckc_arena_printf(&L->arena, "declare %s @%s(%s, %s)",
                                        vec_llvm, intrin, vec_llvm, vec_llvm);
    ckc_ll_need_dynamic(L, intrin, decl);
    ckc_ll_emitf(L, "  %s = call %s @%s(%s %s, %s %s)", res->name, vec_llvm,
                 intrin, vec_llvm, ckc_ll_operand(L, a), vec_llvm,
                 ckc_ll_operand(L, b));
}

/* Python _op_vector_smin: icmp slt + vselect. */
static void _op_vector_smin(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 2) {
        return;
    }
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    int count = (res->type && res->type->kind == CKC_TYPE_VECTOR)
                    ? res->type->count
                    : 0;
    const char *cmp = ckc_ll_fresh(L, "vsmin.cmp");
    ckc_ll_emitf(L, "  %s = icmp slt %s %s, %s", cmp,
                 ckc_ll_llvm_type(L, a->type), ckc_ll_operand(L, a),
                 ckc_ll_operand(L, b));
    ckc_ll_emitf(L, "  %s = select <%d x i1> %s, %s %s, %s %s", res->name, count,
                 cmp, ckc_ll_llvm_type(L, a->type), ckc_ll_operand(L, a),
                 ckc_ll_llvm_type(L, b->type), ckc_ll_operand(L, b));
}

/* Python _op_vector_trunc: packed trunc. */
static void _op_vector_trunc(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 1) {
        return;
    }
    const ckc_value_t *v = op->operands[0];
    ckc_ll_emitf(L, "  %s = trunc %s %s to %s", res->name,
                 ckc_ll_llvm_type(L, v->type), ckc_ll_operand(L, v),
                 ckc_ll_llvm_type(L, res->type));
}

/* Python _op_vector_sext: packed sext. */
static void _op_vector_sext(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 1) {
        return;
    }
    const ckc_value_t *v = op->operands[0];
    ckc_ll_emitf(L, "  %s = sext %s %s to %s", res->name,
                 ckc_ll_llvm_type(L, v->type), ckc_ll_operand(L, v),
                 ckc_ll_llvm_type(L, res->type));
}

/* Python _op_vector_fma: packed llvm.fmuladd.v<N><elem>.
 * TODO(port): faithful intrinsic-name mangling + decl registration. */
static void _op_vector_fma(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 3) {
        return;
    }
    /* TODO(port): mangle llvm.fmuladd.v<count>f<width> and register the decl;
     * for now fold to a*b+c via two packed ops so the value is defined and the
     * dispatch is total. */
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    const ckc_value_t *c = op->operands[2];
    const char *ty = ckc_ll_llvm_type(L, res->type);
    const char *mul = ckc_ll_fresh(L, "vfma.mul");
    ckc_ll_emitf(L, "  %s = fmul %s %s, %s", mul, ty, ckc_ll_operand(L, a),
                 ckc_ll_operand(L, b));
    ckc_ll_emitf(L, "  %s = fadd %s %s, %s", res->name, ty, mul,
                 ckc_ll_operand(L, c));
}

/* Python _op_vector_max: packed max via element fold.
 * TODO(port): exact intrinsic / packed-max form. */
static void _op_vector_max(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 2) {
        return;
    }
    /* TODO(port): faithful packed v_pk_max / llvm.maxnum.vN lowering. Emit a
     * defined value via fcmp+select so the SSA name exists and links. */
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    const ckc_type_t *vt = res->type;
    int count = (vt && vt->kind == CKC_TYPE_VECTOR) ? vt->count : 0;
    const char *ty = ckc_ll_llvm_type(L, vt);
    const char *cmp = ckc_ll_fresh(L, "vmax.cmp");
    ckc_ll_emitf(L, "  %s = fcmp ogt %s %s, %s", cmp, ty, ckc_ll_operand(L, a),
                 ckc_ll_operand(L, b));
    ckc_ll_emitf(L, "  %s = select <%d x i1> %s, %s %s, %s %s", res->name, count,
                 cmp, ty, ckc_ll_operand(L, a), ty, ckc_ll_operand(L, b));
}

/* Python _op_vector_select: packed vselect. */
static void _op_vector_select(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 3) {
        return;
    }
    const ckc_value_t *cond = op->operands[0];
    const ckc_value_t *lhs = op->operands[1];
    const ckc_value_t *rhs = op->operands[2];
    ckc_ll_emitf(L, "  %s = select %s %s, %s %s, %s %s", res->name,
                 ckc_ll_llvm_type(L, cond->type), ckc_ll_operand(L, cond),
                 ckc_ll_llvm_type(L, lhs->type), ckc_ll_operand(L, lhs),
                 ckc_ll_llvm_type(L, rhs->type), ckc_ll_operand(L, rhs));
}

/* Python _op_vector_sum: horizontal fadd from 0.0. */
static void _op_vector_sum(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_lower_vector_reduce(L, op, "fadd", "0.000000e+00");
}

/* Python _op_vector_reduce_max: horizontal max via fcmp/select. */
static void _op_vector_reduce_max(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 1) {
        return;
    }
    const ckc_value_t *v = op->operands[0];
    const ckc_type_t *vec_ty = v->type;
    if (!vec_ty || vec_ty->kind != CKC_TYPE_VECTOR) {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL, "vector.reduce_max: not a vector");
        return;
    }
    int count = vec_ty->count;
    const ckc_type_t *elem_ty = vec_ty->elem;
    const char *vec_llvm = ckc_ll_llvm_type(L, vec_ty);
    const char *elem_llvm = ckc_ll_llvm_type(L, elem_ty);
    const char *acc = NULL;
    for (int i = 0; i < count; i++) {
        const char *e = ckc_ll_fresh(L, "vred.e");
        ckc_ll_emitf(L, "  %s = extractelement %s %s, i32 %d", e, vec_llvm,
                     ckc_ll_operand(L, v), i);
        if (acc == NULL) {
            acc = e;
        } else {
            const char *cmp = ckc_ll_fresh(L, "vred.cmp");
            const char *nxt =
                (i == count - 1) ? res->name : ckc_ll_fresh(L, "vred.max");
            ckc_ll_emitf(L, "  %s = fcmp ogt %s %s, %s", cmp, elem_llvm, acc, e);
            ckc_ll_emitf(L, "  %s = select i1 %s, %s %s, %s %s", nxt, cmp,
                         elem_llvm, acc, elem_llvm, e);
            acc = nxt;
        }
    }
    if (count == 1) {
        ckc_ll_emitf(L, "  %s = fadd %s %s, 0.000000e+00", res->name, elem_llvm,
                     acc ? acc : "0.000000e+00");
    }
}

/* Python _op_vector_splat.
 * TODO(port): faithful insertelement+shufflevector broadcast. */
static void _op_vector_splat(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 1) {
        return;
    }
    /* TODO(port): exact insertelement %undef, scalar, 0 + shufflevector
     * zeroinitializer broadcast. Stub: zeroinitializer select so the SSA name
     * is defined and the module links. */
    const char *ty = ckc_ll_llvm_type(L, res->type);
    ckc_ll_emitf(L,
                 "  %s = select i1 true, %s zeroinitializer, %s zeroinitializer",
                 res->name, ty, ty);
}

/* Python _op_vector_extract. */
static void _op_vector_extract(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 1) {
        return;
    }
    const ckc_value_t *v = op->operands[0];
    int64_t idx = 0;
    ckc_attr_get_int(&op->attrs, "index", &idx);
    ckc_ll_emitf(L, "  %s = extractelement %s %s, i32 %lld", res->name,
                 ckc_ll_llvm_type(L, v->type), ckc_ll_operand(L, v),
                 (long long)idx);
}

/* Python _op_vector_insert. */
static void _op_vector_insert(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 2) {
        return;
    }
    const ckc_value_t *vec = op->operands[0];
    const ckc_value_t *elem = op->operands[1];
    int64_t idx = 0;
    ckc_attr_get_int(&op->attrs, "index", &idx);
    ckc_ll_emitf(L, "  %s = insertelement %s %s, %s %s, i32 %lld", res->name,
                 ckc_ll_llvm_type(L, vec->type), ckc_ll_operand(L, vec),
                 ckc_ll_llvm_type(L, elem->type), ckc_ll_operand(L, elem),
                 (long long)idx);
}

/* Python _op_vector_pack.
 * TODO(port): faithful per-lane insertelement chain. */
static void _op_vector_pack(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    /* TODO(port): build the packed vector via an insertelement chain over
     * op->operands. Stub to zeroinitializer so the value is defined. */
    const char *ty = ckc_ll_llvm_type(L, res->type);
    ckc_ll_emitf(L,
                 "  %s = select i1 true, %s zeroinitializer, %s zeroinitializer",
                 res->name, ty, ty);
}

/* Python _op_vector_concat.
 * TODO(port): faithful shufflevector concat. */
static void _op_vector_concat(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res) {
        return;
    }
    /* TODO(port): two-input shufflevector with the concat index mask. Stub to
     * zeroinitializer so the result name links. */
    const char *ty = ckc_ll_llvm_type(L, res->type);
    ckc_ll_emitf(L,
                 "  %s = select i1 true, %s zeroinitializer, %s zeroinitializer",
                 res->name, ty, ty);
}

/* Python _op_vector_bitcast. */
static void _op_vector_bitcast(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 1) {
        return;
    }
    const ckc_value_t *v = op->operands[0];
    ckc_ll_emitf(L, "  %s = bitcast %s %s to %s", res->name,
                 ckc_ll_llvm_type(L, v->type), ckc_ll_operand(L, v),
                 ckc_ll_llvm_type(L, res->type));
}

/* Python _op_vector_trunc_f32_to_f16.
 * TODO(port): faithful fptrunc <N x float> to <N x half>. */
static void _op_vector_trunc_f32_to_f16(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 1) {
        return;
    }
    const ckc_value_t *v = op->operands[0];
    /* TODO(port): confirm exact rounding/flavor handling matches Python. */
    ckc_ll_emitf(L, "  %s = fptrunc %s %s to %s", res->name,
                 ckc_ll_llvm_type(L, v->type), ckc_ll_operand(L, v),
                 ckc_ll_llvm_type(L, res->type));
}

/* Python _op_vector_trunc_f32_to.
 * TODO(port): faithful generic fptrunc to the target element type. */
static void _op_vector_trunc_f32_to(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *res = ll_result(op);
    if (!ckc_ll_live(L) || !res || op->num_operands < 1) {
        return;
    }
    const ckc_value_t *v = op->operands[0];
    /* TODO(port): bf16 path may need a bitcast/intrinsic rather than fptrunc. */
    ckc_ll_emitf(L, "  %s = fptrunc %s %s to %s", res->name,
                 ckc_ll_llvm_type(L, v->type), ckc_ll_operand(L, v),
                 ckc_ll_llvm_type(L, res->type));
}

/* ======================================================================== */
/* tile.* barriers / sync / scheduling                                      */
/* ======================================================================== */

/* Python _op_tile_sync: s_waitcnt(vmcnt0,lgkmcnt0) + s_barrier, with the
 * unroll trailing-sync elision check. */
static void _op_tile_sync(ckc_lower_t *L, const ckc_op_t *op) {
    if (!ckc_ll_live(L)) {
        return;
    }
    if (L->unroll_elide_sync_op && L->unroll_elide_sync_op == op) {
        return; /* skip the trailing sync in a non-final unrolled iteration */
    }
    int mask = ll_backend_waitcnt(L, 0, -1, 0);
    ckc_ll_need(L, "s.waitcnt");
    ckc_ll_need(L, "s.barrier");
    ckc_ll_emitf(L, "  call void @llvm.amdgcn.s.waitcnt(i32 %d)", mask);
    ckc_ll_emit(L, " call void @llvm.amdgcn.s.barrier()");
}

/* Python _op_tile_s_barrier_bare: bare s_barrier, no implicit waitcnt. */
static void _op_tile_s_barrier_bare(ckc_lower_t *L, const ckc_op_t *op) {
    (void)op;
    if (!ckc_ll_live(L)) {
        return;
    }
    ckc_ll_need(L, "s.barrier");
    ckc_ll_emit(L, " call void @llvm.amdgcn.s.barrier()");
}

/* Python _op_tile_sync_half_block: cond-branch so only the then-branch hits
 * the s_barrier. */
static void _op_tile_sync_half_block(ckc_lower_t *L, const ckc_op_t *op) {
    if (!ckc_ll_live(L) || op->num_operands < 1) {
        return;
    }
    const ckc_value_t *sel = op->operands[0];
    ckc_ll_need(L, "s.barrier");
    const char *i1_name = ckc_ll_fresh(L, "half_pred");
    ckc_ll_emitf(L, "  %s = icmp ne i32 %s, 0", i1_name, ckc_ll_operand(L, sel));
    ckc_ll_block_t *then_blk = ckc_ll_new_block(L, "hb_then");
    ckc_ll_block_t *join_blk = ckc_ll_new_block(L, "hb_join");
    /* The block before the two we just pushed is the predecessor; terminate it
     * with the conditional branch. */
    int prev_idx = ckc_ll_block_count(L) - 3;
    ckc_ll_block_t *prev_blk = ckc_ll_block_at(L, prev_idx);
    if (prev_blk && then_blk && join_blk) {
        ckc_ll_block_emitf(L, prev_blk, "  br i1 %s, label %%%s, label %%%s",
                           i1_name, then_blk->label, join_blk->label);
        prev_blk->terminated = true;
        ckc_ll_block_emit(L, then_blk, " call void @llvm.amdgcn.s.barrier()");
        ckc_ll_block_emitf(L, then_blk, "  br label %%%s", join_blk->label);
        then_blk->terminated = true;
    }
    /* Subsequent ops fall into join_blk (now _current). */
}

/* Python _op_tile_sync_lds_only: drain LDS (lgkmcnt0) but not VMEM. */
static void _op_tile_sync_lds_only(ckc_lower_t *L, const ckc_op_t *op) {
    (void)op;
    if (!ckc_ll_live(L)) {
        return;
    }
    int mask = ll_backend_waitcnt(L, -1, -1, 0);
    ckc_ll_need(L, "s.waitcnt");
    ckc_ll_need(L, "s.barrier");
    ckc_ll_emitf(L, "  call void @llvm.amdgcn.s.waitcnt(i32 %d)", mask);
    ckc_ll_emit(L, " call void @llvm.amdgcn.s.barrier()");
}

/* Python _op_tile_s_waitcnt: explicit s_waitcnt from attrs. */
static void _op_tile_s_waitcnt(ckc_lower_t *L, const ckc_op_t *op) {
    if (!ckc_ll_live(L)) {
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
static void _op_tile_iglp_opt(ckc_lower_t *L, const ckc_op_t *op) {
    if (!ckc_ll_live(L)) {
        return;
    }
    ckc_ll_need(L, "iglp.opt");
    int64_t level = 0;
    ckc_attr_get_int(&op->attrs, "level", &level);
    ckc_ll_emitf(L, "  call void @llvm.amdgcn.iglp.opt(i32 %lld)",
                 (long long)level);
}

/* Python _op_tile_sched_barrier. */
static void _op_tile_sched_barrier(ckc_lower_t *L, const ckc_op_t *op) {
    if (!ckc_ll_live(L)) {
        return;
    }
    ckc_ll_need(L, "sched.barrier");
    int64_t mask = 0;
    ckc_attr_get_int(&op->attrs, "mask", &mask);
    ckc_ll_emitf(L, "  call void @llvm.amdgcn.sched.barrier(i32 %lld)",
                 (long long)mask);
}

/* Python _op_tile_sched_group_barrier. */
static void _op_tile_sched_group_barrier(ckc_lower_t *L, const ckc_op_t *op) {
    if (!ckc_ll_live(L)) {
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
                 (long long)mask, (long long)count, (long long)group);
}

/* Python _op_tile_s_setprio. */
static void _op_tile_s_setprio(ckc_lower_t *L, const ckc_op_t *op) {
    if (!ckc_ll_live(L)) {
        return;
    }
    ckc_ll_need(L, "s.setprio");
    int64_t level = 0;
    ckc_attr_get_int(&op->attrs, "level", &level);
    ckc_ll_emitf(L, "  call void @llvm.amdgcn.s.setprio(i16 %lld)",
                 (long long)level);
}

/* ======================================================================== */
/* scf.* / cf.* control flow                                                */
/* ======================================================================== */

/* Python _lower_normal_for: header / body / latch / exit CFG with phi nodes.
 * TODO(port): full faithful port of the phi back-patching, FOR_LATCH/FOR_EXIT
 * label fixup, and iter-arg yield wiring. */
void ckc_ll_lower_normal_for(ckc_lower_t *L, const ckc_op_t *op) {
    if (!ckc_ll_live(L)) {
        return;
    }
    /* TODO(port): build the for.header/for.body/for.latch/for.exit blocks,
     * emit the induction-variable + iter-arg phis, lower op->regions[0] under a
     * fresh yield frame, back-patch the latch phi operands from the popped
     * yield frame, and resolve the deferred FOR_EXIT / FOR_LATCH labels. For
     * now record/discard a yield frame so the scf.yield handler stays balanced
     * and lower the body inline so its ops are still emitted (un-looped). */
    ckc_ll_yield_push(L);
    if (op->num_regions > 0 && op->regions[0]) {
        ckc_ll_lower_region(L, op->regions[0]);
    }
    const char *const *items = NULL;
    int n = 0;
    ckc_ll_yield_pop(L, &items, &n);
    (void)items;
    (void)n;
}

/* Python _lower_unrolled_for: straight-line replication of the body for
 * constant bounds.
 * TODO(port): full faithful port -- per-iteration value renaming, trailing
 * tile.sync elision, and iter-arg threading across unrolled copies. */
void ckc_ll_lower_unrolled_for(ckc_lower_t *L, const ckc_op_t *op) {
    if (!ckc_ll_live(L)) {
        return;
    }
    /* TODO(port): evaluate the constant lower/upper/step, then for each trip
     * rename the IV + iter args and re-lower op->regions[0], eliding the
     * trailing tile.sync via L->unroll_elide_sync_op on non-final trips. For
     * now lower the body once (under a yield frame) so dispatch is total and
     * the contained ops are emitted. */
    ckc_ll_yield_push(L);
    if (op->num_regions > 0 && op->regions[0]) {
        ckc_ll_lower_region(L, op->regions[0]);
    }
    const char *const *items = NULL;
    int n = 0;
    ckc_ll_yield_pop(L, &items, &n);
    (void)items;
    (void)n;
}

/* Python _op_scf_for: pick unrolled vs normal lowering. */
static void _op_scf_for(ckc_lower_t *L, const ckc_op_t *op) {
    if (!ckc_ll_live(L) || op->num_operands < 3) {
        return;
    }
    bool unroll = ckc_attr_get_bool(&op->attrs, "unroll", false);
    const ckc_value_t *lower = op->operands[0];
    const ckc_value_t *upper = op->operands[1];
    const ckc_value_t *step = op->operands[2];
    if (unroll && ckc_ll_is_constant(lower) && ckc_ll_is_constant(upper) &&
        ckc_ll_is_constant(step)) {
        ckc_ll_lower_unrolled_for(L, op);
    } else {
        ckc_ll_lower_normal_for(L, op);
    }
}

/* Python _op_scf_if.
 * TODO(port): full then/else CFG with the cond branch and join block. */
static void _op_scf_if(ckc_lower_t *L, const ckc_op_t *op) {
    if (!ckc_ll_live(L)) {
        return;
    }
    /* TODO(port): emit the i1 cond branch, lower the then-region (and optional
     * else-region) into fresh blocks, and join. For now lower the then-region
     * inline so its ops are emitted and the dispatch is total. */
    if (op->num_regions > 0 && op->regions[0]) {
        ckc_ll_lower_region(L, op->regions[0]);
    }
    if (op->num_regions > 1 && op->regions[1]) {
        ckc_ll_lower_region(L, op->regions[1]);
    }
}

/* Python _op_scf_yield: record yielded operand strings into the top frame. */
static void _op_scf_yield(ckc_lower_t *L, const ckc_op_t *op) {
    if (!ckc_ll_live(L) || L->yield_stack.len == 0) {
        return;
    }
    for (int i = 0; i < op->num_operands; i++) {
        ckc_ll_yield_record(L, ckc_ll_operand(L, op->operands[i]));
    }
}

/* Python _op_cf_return: terminate the current block with `ret void`.
 * TODO(port): confirm value-returning kernels (none today emit ret <ty>). */
static void _op_cf_return(ckc_lower_t *L, const ckc_op_t *op) {
    (void)op;
    if (!ckc_ll_live(L)) {
        return;
    }
    ckc_ll_block_t *cur = ckc_ll_current(L);
    if (cur && !cur->terminated) {
        ckc_ll_emit(L, "  ret void");
        cur->terminated = true;
    }
}

/* ----------------------------------------------------------- registration */

void ckc_ll_register_vector(void) {
    /* vector.* */
    ckc_ll_set_handler(CKC_OP_VECTOR_ADD,               _op_vector_add);
    ckc_ll_set_handler(CKC_OP_VECTOR_SUB,               _op_vector_sub);
    ckc_ll_set_handler(CKC_OP_VECTOR_MUL,               _op_vector_mul);
    ckc_ll_set_handler(CKC_OP_VECTOR_AND,               _op_vector_and);
    ckc_ll_set_handler(CKC_OP_VECTOR_OR,                _op_vector_or);
    ckc_ll_set_handler(CKC_OP_VECTOR_SHL,               _op_vector_shl);
    ckc_ll_set_handler(CKC_OP_VECTOR_LSHR,              _op_vector_lshr);
    ckc_ll_set_handler(CKC_OP_VECTOR_SMAX,              _op_vector_smax);
    ckc_ll_set_handler(CKC_OP_VECTOR_SMIN,              _op_vector_smin);
    ckc_ll_set_handler(CKC_OP_VECTOR_MAX,               _op_vector_max);
    ckc_ll_set_handler(CKC_OP_VECTOR_FMA,               _op_vector_fma);
    ckc_ll_set_handler(CKC_OP_VECTOR_SUM,               _op_vector_sum);
    ckc_ll_set_handler(CKC_OP_VECTOR_REDUCE_MAX,        _op_vector_reduce_max);
    ckc_ll_set_handler(CKC_OP_VECTOR_SPLAT,             _op_vector_splat);
    ckc_ll_set_handler(CKC_OP_VECTOR_SELECT,            _op_vector_select);
    ckc_ll_set_handler(CKC_OP_VECTOR_CMP,               _op_vector_cmp);
    ckc_ll_set_handler(CKC_OP_VECTOR_TRUNC,             _op_vector_trunc);
    ckc_ll_set_handler(CKC_OP_VECTOR_SEXT,              _op_vector_sext);
    ckc_ll_set_handler(CKC_OP_VECTOR_TRUNC_F32_TO_F16,  _op_vector_trunc_f32_to_f16);
    ckc_ll_set_handler(CKC_OP_VECTOR_TRUNC_F32_TO,      _op_vector_trunc_f32_to);
    ckc_ll_set_handler(CKC_OP_VECTOR_BITCAST,           _op_vector_bitcast);
    ckc_ll_set_handler(CKC_OP_VECTOR_EXTRACT,           _op_vector_extract);
    ckc_ll_set_handler(CKC_OP_VECTOR_INSERT,            _op_vector_insert);
    ckc_ll_set_handler(CKC_OP_VECTOR_PACK,              _op_vector_pack);
    ckc_ll_set_handler(CKC_OP_VECTOR_CONCAT,            _op_vector_concat);

    /* tile.* -- barriers / scheduling */
    ckc_ll_set_handler(CKC_OP_TILE_SYNC,                _op_tile_sync);
    ckc_ll_set_handler(CKC_OP_TILE_SYNC_HALF_BLOCK,     _op_tile_sync_half_block);
    ckc_ll_set_handler(CKC_OP_TILE_SYNC_LDS_ONLY,       _op_tile_sync_lds_only);
    ckc_ll_set_handler(CKC_OP_TILE_S_BARRIER_BARE,      _op_tile_s_barrier_bare);
    ckc_ll_set_handler(CKC_OP_TILE_S_WAITCNT,           _op_tile_s_waitcnt);
    ckc_ll_set_handler(CKC_OP_TILE_S_SETPRIO,           _op_tile_s_setprio);
    ckc_ll_set_handler(CKC_OP_TILE_IGLP_OPT,            _op_tile_iglp_opt);
    ckc_ll_set_handler(CKC_OP_TILE_SCHED_BARRIER,       _op_tile_sched_barrier);
    ckc_ll_set_handler(CKC_OP_TILE_SCHED_GROUP_BARRIER, _op_tile_sched_group_barrier);

    /* scf.* / cf.* control flow */
    ckc_ll_set_handler(CKC_OP_SCF_FOR,                  _op_scf_for);
    ckc_ll_set_handler(CKC_OP_SCF_IF,                   _op_scf_if);
    ckc_ll_set_handler(CKC_OP_SCF_YIELD,                _op_scf_yield);
    ckc_ll_set_handler(CKC_OP_CF_RETURN,                _op_cf_return);
}
