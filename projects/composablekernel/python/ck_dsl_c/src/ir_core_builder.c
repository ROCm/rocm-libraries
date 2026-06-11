/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ir_core_builder.c -- bucket 0 of the C99 port of ck_dsl.core.ir.
 *
 * This translation unit owns the IRBuilder lifecycle, the public low-level
 * plumbing (ckc_b_op / ckc_b_fresh / ckc_b_emit / region stack / params), and
 * the shared internal helpers (the ckc_i_* family declared in
 * ckc/ir_internal.h) that every other ir_*.c bucket funnels through.
 *
 * Mirrors the Python IRBuilder (ck_dsl/core/ir.py):
 *   - _fresh  -> ckc_b_fresh / ckc_i_new_value
 *   - _emit   -> ckc_b_emit  / ckc_i_emit
 *   - _op     -> ckc_b_op    / ckc_i_op (+ ckc_i_op0 / ckc_i_op1 shorthands)
 *   - param / get_param -> ckc_b_param / ckc_b_get_param
 *
 * Lifetime: every node lives in the builder's arena and is bulk-freed by
 * ckc_ir_builder_free, exactly as Python relies on the GC.
 */

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/ir.h"
#include "ckc/ir_internal.h"
#include "ckc/vec.h"

/* ----------------------------------------------------------- error model */

bool ckc_i_live(const ckc_ir_builder_t *b) {
    return b != NULL && b->status == CKC_OK;
}

void *ckc_i_set_err(ckc_ir_builder_t *b, ckc_status_t st, const char *fmt, ...) {
    if (b == NULL) {
        return NULL;
    }
    /* First failure wins: preserve an existing sticky error. */
    if (b->status != CKC_OK) {
        return NULL;
    }
    b->status = (st == CKC_OK) ? CKC_ERR_VALUE : st;

    if (fmt != NULL) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(b->err, sizeof(b->err), fmt, ap);
        va_end(ap);
        b->err[sizeof(b->err) - 1] = '\0';
    } else {
        b->err[0] = '\0';
    }
    return NULL;
}

/* ------------------------------------------------------- region plumbing */

ckc_region_t *ckc_i_new_region(ckc_ir_builder_t *b, const char *label) {
    ckc_region_t *r;
    if (!ckc_i_live(b)) {
        return NULL;
    }
    r = (ckc_region_t *)ckc_arena_calloc(&b->arena, sizeof(*r));
    if (!r) {
        return (ckc_region_t *)ckc_i_set_err(b, CKC_ERR_OOM, "new_region: OOM");
    }
    r->label = ckc_arena_strdup(&b->arena, label ? label : "");
    if (!r->label) {
        return (ckc_region_t *)ckc_i_set_err(b, CKC_ERR_OOM, "new_region: OOM label");
    }
    r->ops = NULL;
    r->num_ops = 0;
    r->cap_ops = 0;
    return r;
}

/* Append an op to a region, growing its arena-backed ops array as needed.
 * The previous block is abandoned to the arena (bulk-freed later). */
static int ckc_region_append(ckc_ir_builder_t *b, ckc_region_t *r, ckc_op_t *op) {
    if (r->num_ops >= r->cap_ops) {
        int nc = r->cap_ops ? r->cap_ops * 2 : 4;
        ckc_op_t **np =
            (ckc_op_t **)ckc_arena_alloc(&b->arena, sizeof(ckc_op_t *) * (size_t)nc);
        if (!np) {
            ckc_i_set_err(b, CKC_ERR_OOM, "region append: OOM");
            return -1;
        }
        if (r->ops && r->num_ops) {
            memcpy(np, r->ops, sizeof(ckc_op_t *) * (size_t)r->num_ops);
        }
        r->ops = np;
        r->cap_ops = nc;
    }
    r->ops[r->num_ops++] = op;
    return 0;
}

void ckc_i_emit(ckc_ir_builder_t *b, ckc_op_t *op) {
    ckc_region_t *cur;
    if (!ckc_i_live(b) || op == NULL) {
        return;
    }
    if (b->region_depth <= 0) {
        ckc_i_set_err(b, CKC_ERR_VALUE, "emit: no current region");
        return;
    }
    cur = b->region_stack[b->region_depth - 1];
    (void)ckc_region_append(b, cur, op);
}

void ckc_b_emit(ckc_ir_builder_t *b, ckc_op_t *op) {
    ckc_i_emit(b, op);
}

void ckc_b_region_enter(ckc_ir_builder_t *b, ckc_region_t *r) {
    if (!ckc_i_live(b)) {
        return;
    }
    if (r == NULL) {
        ckc_i_set_err(b, CKC_ERR_VALUE, "region_enter: NULL region");
        return;
    }
    if (b->region_depth >= CKC_REGION_STACK_MAX) {
        ckc_i_set_err(b, CKC_ERR_VALUE, "region_enter: stack overflow");
        return;
    }
    b->region_stack[b->region_depth++] = r;
}

void ckc_b_region_leave(ckc_ir_builder_t *b) {
    if (!ckc_i_live(b)) {
        return;
    }
    if (b->region_depth <= 0) {
        ckc_i_set_err(b, CKC_ERR_VALUE, "region_leave: stack underflow");
        return;
    }
    b->region_depth--;
}

ckc_region_t *ckc_b_current_region(ckc_ir_builder_t *b) {
    if (b == NULL || b->region_depth <= 0) {
        return NULL;
    }
    return b->region_stack[b->region_depth - 1];
}

/* ------------------------------------------------------------- naming */

const char *ckc_b_fresh(ckc_ir_builder_t *b, const char *prefix) {
    char *out;
    if (!ckc_i_live(b)) {
        return NULL;
    }
    b->counter += 1;
    out = ckc_arena_printf(&b->arena, "%%%s%d", prefix ? prefix : "v", b->counter);
    if (!out) {
        return (const char *)ckc_i_set_err(b, CKC_ERR_OOM, "fresh: OOM");
    }
    return out;
}

ckc_value_t *ckc_i_new_value(ckc_ir_builder_t *b, const char *prefix,
                             const ckc_type_t *type) {
    ckc_value_t *v;
    const char *nm;
    if (!ckc_i_live(b)) {
        return NULL;
    }
    nm = ckc_b_fresh(b, prefix);
    if (!nm) {
        return NULL;
    }
    v = (ckc_value_t *)ckc_arena_calloc(&b->arena, sizeof(*v));
    if (!v) {
        return (ckc_value_t *)ckc_i_set_err(b, CKC_ERR_OOM, "new_value: OOM");
    }
    v->name = nm;
    v->type = type;
    v->op = NULL;
    return v;
}

ckc_value_t *ckc_i_value_named(ckc_ir_builder_t *b, const char *name,
                               const ckc_type_t *type) {
    ckc_value_t *v;
    if (!ckc_i_live(b)) {
        return NULL;
    }
    v = (ckc_value_t *)ckc_arena_calloc(&b->arena, sizeof(*v));
    if (!v) {
        return (ckc_value_t *)ckc_i_set_err(b, CKC_ERR_OOM, "value_named: OOM");
    }
    v->name = ckc_arena_strdup(&b->arena, name ? name : "");
    if (!v->name) {
        return (ckc_value_t *)ckc_i_set_err(b, CKC_ERR_OOM, "value_named: OOM name");
    }
    v->type = type;
    v->op = NULL;
    return v;
}

/* ------------------------------------------------------------- attr helpers */

ckc_attr_map_t ckc_i_attrs(ckc_ir_builder_t *b) {
    ckc_attr_map_t m;
    (void)b;
    ckc_attr_map_init(&m);
    return m;
}

void ckc_i_attrs_copy(ckc_ir_builder_t *b, ckc_attr_map_t *dst,
                      const ckc_attr_map_t *src) {
    int i;
    if (dst == NULL) {
        return;
    }
    ckc_attr_map_init(dst);
    if (!ckc_i_live(b) || src == NULL) {
        return;
    }
    /* Deep-copy entries via the public setters so keys/strings are arena-owned
     * by the destination (Op.attrs = dict(attrs or {}) in Python). */
    for (i = 0; i < src->count; i++) {
        const ckc_attr_entry_t *e = &src->entries[i];
        switch (e->value.kind) {
            case CKC_ATTR_INT:
                ckc_attr_set_int(b, dst, e->key, e->value.u.i);
                break;
            case CKC_ATTR_FLOAT:
                ckc_attr_set_float(b, dst, e->key, e->value.u.f);
                break;
            case CKC_ATTR_STR:
                ckc_attr_set_str(b, dst, e->key, e->value.u.s);
                break;
            case CKC_ATTR_BOOL:
                ckc_attr_set_bool(b, dst, e->key, e->value.u.b);
                break;
            case CKC_ATTR_LIST:
            default:
                /* TODO(port): nested attr-list deep copy (scf.for iter_args
                 * metadata). For the linking milestone, shallow-copy the entry
                 * directly into the destination so list-valued attrs survive.
                 * This shares the arena-owned items array with the source,
                 * which is safe under the single-arena lifetime. */
                if (dst->count >= dst->cap) {
                    int nc = dst->cap ? dst->cap * 2 : 4;
                    ckc_attr_entry_t *ne = (ckc_attr_entry_t *)ckc_arena_alloc(
                        &b->arena, sizeof(ckc_attr_entry_t) * (size_t)nc);
                    if (!ne) {
                        ckc_i_set_err(b, CKC_ERR_OOM, "attrs_copy: OOM");
                        return;
                    }
                    if (dst->entries && dst->count) {
                        memcpy(ne, dst->entries,
                               sizeof(ckc_attr_entry_t) * (size_t)dst->count);
                    }
                    dst->entries = ne;
                    dst->cap = nc;
                }
                dst->entries[dst->count].key = ckc_arena_strdup(&b->arena, e->key);
                dst->entries[dst->count].value = e->value;
                dst->count++;
                break;
        }
    }
}

/* --------------------------------------------------------- generic op build */

/* Shared implementation behind ckc_b_op / ckc_i_op. */
ckc_op_t *ckc_i_op(ckc_ir_builder_t *b, ckc_opcode_t opcode,
                   ckc_value_t *const *operands, int num_operands,
                   const ckc_type_t *const *result_types, int num_results,
                   const ckc_attr_map_t *attrs,
                   ckc_region_t *const *regions, int num_regions,
                   const char *result_name_hint, const char *loc) {
    ckc_op_t *op;
    int i;

    if (!ckc_i_live(b)) {
        return NULL;
    }
    if (num_operands < 0) num_operands = 0;
    if (num_results < 0) num_results = 0;
    if (num_regions < 0) num_regions = 0;

    op = (ckc_op_t *)ckc_arena_calloc(&b->arena, sizeof(*op));
    if (!op) {
        return (ckc_op_t *)ckc_i_set_err(b, CKC_ERR_OOM, "op: OOM");
    }
    op->opcode = opcode;
    op->name = ckc_opcode_name(opcode);
    op->loc = loc ? ckc_arena_strdup(&b->arena, loc) : NULL;

    /* operands: copy into an arena array */
    if (num_operands > 0) {
        op->operands = (ckc_value_t **)ckc_arena_alloc(
            &b->arena, sizeof(ckc_value_t *) * (size_t)num_operands);
        if (!op->operands) {
            return (ckc_op_t *)ckc_i_set_err(b, CKC_ERR_OOM, "op: OOM operands");
        }
        for (i = 0; i < num_operands; i++) {
            op->operands[i] = operands ? operands[i] : NULL;
        }
        op->num_operands = num_operands;
    }

    /* results: one fresh Value per result type, named with result_name_hint */
    if (num_results > 0) {
        op->results = (ckc_value_t **)ckc_arena_alloc(
            &b->arena, sizeof(ckc_value_t *) * (size_t)num_results);
        if (!op->results) {
            return (ckc_op_t *)ckc_i_set_err(b, CKC_ERR_OOM, "op: OOM results");
        }
        for (i = 0; i < num_results; i++) {
            const ckc_type_t *rt = result_types ? result_types[i] : NULL;
            ckc_value_t *r = ckc_i_new_value(b, result_name_hint ? result_name_hint : "v", rt);
            if (!r) {
                return NULL; /* error already set */
            }
            op->results[i] = r;
        }
        op->num_results = num_results;
    }

    /* attrs: deep-copy borrowed map (Python dict(attrs or {})) */
    ckc_i_attrs_copy(b, &op->attrs, attrs);
    if (!ckc_i_live(b)) {
        return NULL;
    }

    /* regions: copy the borrowed region pointers into an arena array */
    if (num_regions > 0) {
        op->regions = (ckc_region_t **)ckc_arena_alloc(
            &b->arena, sizeof(ckc_region_t *) * (size_t)num_regions);
        if (!op->regions) {
            return (ckc_op_t *)ckc_i_set_err(b, CKC_ERR_OOM, "op: OOM regions");
        }
        for (i = 0; i < num_regions; i++) {
            op->regions[i] = regions ? regions[i] : NULL;
        }
        op->num_regions = num_regions;
    }

    /* link results back to the producing op */
    for (i = 0; i < op->num_results; i++) {
        if (op->results[i]) {
            op->results[i]->op = op;
        }
    }

    /* emit into the current region */
    ckc_i_emit(b, op);
    if (!ckc_i_live(b)) {
        return NULL;
    }
    return op;
}

ckc_op_t *ckc_b_op(ckc_ir_builder_t *b, ckc_opcode_t opcode,
                   ckc_value_t *const *operands, int num_operands,
                   const ckc_type_t *const *result_types, int num_results,
                   const ckc_attr_map_t *attrs,
                   ckc_region_t *const *regions, int num_regions,
                   const char *result_name_hint, const char *loc) {
    return ckc_i_op(b, opcode, operands, num_operands, result_types, num_results,
                    attrs, regions, num_regions, result_name_hint, loc);
}

/* --------------------------------------------------- emission shorthands */

ckc_value_t *ckc_i_op1(ckc_ir_builder_t *b, ckc_opcode_t opcode,
                       ckc_value_t *const *operands, int num_operands,
                       const ckc_type_t *result_type,
                       const ckc_attr_map_t *attrs,
                       const char *result_name_hint) {
    const ckc_type_t *rts[1];
    ckc_op_t *op;
    if (!ckc_i_live(b)) {
        return NULL;
    }
    rts[0] = result_type;
    op = ckc_i_op(b, opcode, operands, num_operands, rts, 1, attrs, NULL, 0,
                  result_name_hint, NULL);
    if (!op) {
        return NULL;
    }
    return op->results[0];
}

ckc_op_t *ckc_i_op0(ckc_ir_builder_t *b, ckc_opcode_t opcode,
                    ckc_value_t *const *operands, int num_operands,
                    const ckc_attr_map_t *attrs) {
    return ckc_i_op(b, opcode, operands, num_operands, NULL, 0, attrs, NULL, 0,
                    "v", NULL);
}

ckc_value_t *ckc_i_binop(ckc_ir_builder_t *b, ckc_opcode_t opcode,
                         ckc_value_t *a, ckc_value_t *bb,
                         const char *result_name_hint) {
    ckc_value_t *operands[2];
    if (!ckc_i_live(b)) {
        return NULL;
    }
    if (a == NULL || bb == NULL) {
        return (ckc_value_t *)ckc_i_set_err(b, CKC_ERR_VALUE,
                                            "binop: NULL operand");
    }
    operands[0] = a;
    operands[1] = bb;
    return ckc_i_op1(b, opcode, operands, 2, a->type, NULL, result_name_hint);
}

ckc_value_t *ckc_i_unop(ckc_ir_builder_t *b, ckc_opcode_t opcode,
                        ckc_value_t *a, const char *result_name_hint) {
    ckc_value_t *operands[1];
    if (!ckc_i_live(b)) {
        return NULL;
    }
    if (a == NULL) {
        return (ckc_value_t *)ckc_i_set_err(b, CKC_ERR_VALUE,
                                            "unop: NULL operand");
    }
    operands[0] = a;
    return ckc_i_op1(b, opcode, operands, 1, a->type, NULL, result_name_hint);
}

/* ----------------------------------------------------- type-system helpers */

bool ckc_i_type_is(const ckc_type_t *t, const char *name) {
    if (t == NULL || name == NULL || t->name == NULL) {
        return false;
    }
    return strcmp(t->name, name) == 0;
}

bool ckc_i_is_vector(const ckc_type_t *t, const char *elem_name, int count) {
    if (t == NULL || t->kind != CKC_TYPE_VECTOR) {
        return false;
    }
    if (count >= 0 && t->count != count) {
        return false;
    }
    if (elem_name != NULL) {
        if (t->elem == NULL || t->elem->name == NULL) {
            return false;
        }
        if (strcmp(t->elem->name, elem_name) != 0) {
            return false;
        }
    }
    return true;
}

const ckc_type_t *ckc_i_elem_of(const ckc_type_t *t) {
    if (t != NULL && t->kind == CKC_TYPE_VECTOR && t->elem != NULL) {
        return t->elem;
    }
    return t;
}

int ckc_i_count_of(const ckc_type_t *t) {
    if (t != NULL && t->kind == CKC_TYPE_VECTOR) {
        return t->count;
    }
    return 1;
}

/* ============================== BUILDER ================================= */

ckc_status_t ckc_ir_builder_init(ckc_ir_builder_t *b, const char *kernel_name) {
    ckc_kernel_def_t *k;
    ckc_region_t *entry;

    if (b == NULL) {
        return CKC_ERR_VALUE;
    }
    memset(b, 0, sizeof(*b));

    if (ckc_arena_init(&b->arena, 0) != 0) {
        b->status = CKC_ERR_OOM;
        snprintf(b->err, sizeof(b->err), "builder_init: arena OOM");
        return CKC_ERR_OOM;
    }

    b->status = CKC_OK;
    b->err[0] = '\0';
    b->counter = 0;
    b->region_depth = 0;

    b->param_names = NULL;
    b->param_values = NULL;
    b->num_param_lookup = 0;
    b->cap_param_lookup = 0;

    k = (ckc_kernel_def_t *)ckc_arena_calloc(&b->arena, sizeof(*k));
    if (!k) {
        ckc_i_set_err(b, CKC_ERR_OOM, "builder_init: OOM kernel");
        return b->status;
    }
    k->name = ckc_arena_strdup(&b->arena, kernel_name ? kernel_name : "");
    if (!k->name) {
        ckc_i_set_err(b, CKC_ERR_OOM, "builder_init: OOM name");
        return b->status;
    }
    k->params = NULL;
    k->num_params = 0;
    k->cap_params = 0;
    ckc_attr_map_init(&k->attrs);

    entry = ckc_i_new_region(b, "entry");
    if (!entry) {
        return b->status;
    }
    k->body = entry;
    b->kernel = k;

    /* current region == kernel body (Python pushes self.kernel.body) */
    b->region_stack[b->region_depth++] = entry;

    return CKC_OK;
}

void ckc_ir_builder_free(ckc_ir_builder_t *b) {
    if (b == NULL) {
        return;
    }
    ckc_arena_destroy(&b->arena);
    memset(b, 0, sizeof(*b));
}

bool ckc_ir_builder_ok(const ckc_ir_builder_t *b) {
    return b != NULL && b->status == CKC_OK;
}

ckc_status_t ckc_ir_builder_status(const ckc_ir_builder_t *b) {
    return b ? b->status : CKC_ERR_VALUE;
}

const char *ckc_ir_builder_error(const ckc_ir_builder_t *b) {
    if (b == NULL) {
        return "";
    }
    return b->err;
}

ckc_kernel_def_t *ckc_ir_builder_kernel(ckc_ir_builder_t *b) {
    return b ? b->kernel : NULL;
}

/* ------------------------------------------------------------- params */

/* Register a param Value in the builder's name->value lookup. */
static int ckc_param_lookup_add(ckc_ir_builder_t *b, const char *name,
                                ckc_value_t *v) {
    if (b->num_param_lookup >= b->cap_param_lookup) {
        int nc = b->cap_param_lookup ? b->cap_param_lookup * 2 : 8;
        const char **nn =
            (const char **)ckc_arena_alloc(&b->arena, sizeof(const char *) * (size_t)nc);
        ckc_value_t **nv =
            (ckc_value_t **)ckc_arena_alloc(&b->arena, sizeof(ckc_value_t *) * (size_t)nc);
        if (!nn || !nv) {
            ckc_i_set_err(b, CKC_ERR_OOM, "param: OOM lookup");
            return -1;
        }
        if (b->param_names && b->num_param_lookup) {
            memcpy(nn, b->param_names, sizeof(const char *) * (size_t)b->num_param_lookup);
            memcpy(nv, b->param_values, sizeof(ckc_value_t *) * (size_t)b->num_param_lookup);
        }
        b->param_names = nn;
        b->param_values = nv;
        b->cap_param_lookup = nc;
    }
    b->param_names[b->num_param_lookup] = name;
    b->param_values[b->num_param_lookup] = v;
    b->num_param_lookup++;
    return 0;
}

static int ckc_kernel_params_add(ckc_ir_builder_t *b, ckc_param_t *p) {
    ckc_kernel_def_t *k = b->kernel;
    if (k->num_params >= k->cap_params) {
        int nc = k->cap_params ? k->cap_params * 2 : 8;
        ckc_param_t **np =
            (ckc_param_t **)ckc_arena_alloc(&b->arena, sizeof(ckc_param_t *) * (size_t)nc);
        if (!np) {
            ckc_i_set_err(b, CKC_ERR_OOM, "param: OOM params");
            return -1;
        }
        if (k->params && k->num_params) {
            memcpy(np, k->params, sizeof(ckc_param_t *) * (size_t)k->num_params);
        }
        k->params = np;
        k->cap_params = nc;
    }
    k->params[k->num_params++] = p;
    return 0;
}

ckc_value_t *ckc_b_param(ckc_ir_builder_t *b, const char *name,
                         const ckc_type_t *t, const ckc_param_opts_t *opts) {
    ckc_value_t *v;
    ckc_param_t *p;
    char *full_name;
    int i;

    if (!ckc_i_live(b)) {
        return NULL;
    }
    if (name == NULL) {
        return (ckc_value_t *)ckc_i_set_err(b, CKC_ERR_VALUE, "param: NULL name");
    }

    /* duplicate kernel parameter check (Python ValueError) */
    for (i = 0; i < b->num_param_lookup; i++) {
        if (b->param_names[i] && strcmp(b->param_names[i], name) == 0) {
            return (ckc_value_t *)ckc_i_set_err(
                b, CKC_ERR_VALUE, "duplicate kernel parameter '%s'", name);
        }
    }

    /* Value name carries the leading '%'. */
    full_name = ckc_arena_printf(&b->arena, "%%%s", name);
    if (!full_name) {
        return (ckc_value_t *)ckc_i_set_err(b, CKC_ERR_OOM, "param: OOM name");
    }
    v = ckc_i_value_named(b, full_name, t);
    if (!v) {
        return NULL;
    }

    /* Param record (name WITHOUT leading '%'). */
    p = (ckc_param_t *)ckc_arena_calloc(&b->arena, sizeof(*p));
    if (!p) {
        return (ckc_value_t *)ckc_i_set_err(b, CKC_ERR_OOM, "param: OOM record");
    }
    p->name = ckc_arena_strdup(&b->arena, name);
    if (!p->name) {
        return (ckc_value_t *)ckc_i_set_err(b, CKC_ERR_OOM, "param: OOM record name");
    }
    p->type = t;
    ckc_attr_map_init(&p->attrs);

    /* Materialise ABI attrs from the opts struct (only set fields, mirroring
     * Python dict(**attrs) which only carries the kwargs actually passed). */
    if (opts != NULL) {
        if (opts->noalias_set) {
            ckc_attr_set_bool(b, &p->attrs, "noalias", opts->noalias);
        }
        if (opts->readonly_set) {
            ckc_attr_set_bool(b, &p->attrs, "readonly", opts->readonly);
        }
        if (opts->writeonly_set) {
            ckc_attr_set_bool(b, &p->attrs, "writeonly", opts->writeonly);
        }
        if (opts->align_set) {
            ckc_attr_set_int(b, &p->attrs, "align", (int64_t)opts->align);
        }
        if (opts->addr_space != NULL) {
            ckc_attr_set_str(b, &p->attrs, "addr_space", opts->addr_space);
        }
    }

    if (ckc_kernel_params_add(b, p) != 0) {
        return NULL;
    }
    if (ckc_param_lookup_add(b, p->name, v) != 0) {
        return NULL;
    }
    return v;
}

ckc_value_t *ckc_b_get_param(ckc_ir_builder_t *b, const char *name) {
    int i;
    if (!ckc_i_live(b)) {
        return NULL;
    }
    if (name == NULL) {
        return (ckc_value_t *)ckc_i_set_err(b, CKC_ERR_KEY, "get_param: NULL name");
    }
    for (i = 0; i < b->num_param_lookup; i++) {
        if (b->param_names[i] && strcmp(b->param_names[i], name) == 0) {
            return b->param_values[i];
        }
    }
    return (ckc_value_t *)ckc_i_set_err(b, CKC_ERR_KEY, "unknown param '%s'", name);
}
