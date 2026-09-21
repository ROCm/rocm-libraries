// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * ir_core_builder.c -- bucket 0 of the C99 port of rocke.core.ir.
 *
 * This translation unit owns the IRBuilder lifecycle, the public low-level
 * plumbing (rocke_b_op / rocke_b_fresh / rocke_b_emit / region stack / params), and
 * the shared internal helpers (the rocke_i_* family declared in
 * rocke/ir_internal.h) that every other ir_*.c bucket funnels through.
 *
 * Mirrors the Python IRBuilder (rocke/core/ir.py):
 *   - _fresh  -> rocke_b_fresh / rocke_i_new_value
 *   - _emit   -> rocke_b_emit  / rocke_i_emit
 *   - _op     -> rocke_b_op    / rocke_i_op (+ rocke_i_op0 / rocke_i_op1 shorthands)
 *   - param / get_param -> rocke_b_param / rocke_b_get_param
 *
 * Lifetime: every node lives in the builder's arena and is bulk-freed by
 * rocke_ir_builder_free, exactly as Python relies on the GC.
 */

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/vec.h"

/* ----------------------------------------------------------- error model */

bool rocke_i_live(const rocke_ir_builder_t* b)
{
    return b != NULL && b->status == ROCKE_OK;
}

/* Raise the failure as a ckc::Error (mirroring the Python `raise`). The throw
 * unwinds to the public entry boundary (rocke_build_*_new / the build workers /
 * rocke_ir_parse / the core lowerer), which catches it via ckc::guard_* and records
 * status + message on the builder, so the extern "C" ABI is unchanged. The
 * builder's arena is bulk-freed by the entry's rocke_ir_builder_free, so unwinding
 * past in-progress builder calls leaks nothing (every node is arena-owned).
 * [[noreturn]] with the void* return type keeps every existing call site valid:
 * both `return (T*)rocke_i_set_err(...)` and the bare `rocke_i_set_err(...);` forms
 * compile unchanged -- the cast/return that follows is simply never reached. */
[[noreturn]] void* rocke_i_set_err(rocke_ir_builder_t* b, rocke_status_t st, const char* fmt, ...)
{
    (void)b;
    char msg[ROCKE_ERR_MSG_CAP];
    if(fmt != NULL)
    {
        va_list ap;
        va_start(ap, fmt);
        (void)vsnprintf(msg, sizeof(msg), fmt, ap);
        va_end(ap);
        msg[sizeof(msg) - 1] = '\0';
    }
    else
    {
        msg[0] = '\0';
    }
    ckc::raise_status((st == ROCKE_OK) ? ROCKE_ERR_VALUE : st, msg);
}

void* rocke_i_set_err_msg(rocke_ir_builder_t* b, rocke_status_t code, const char* msg)
{
    if(b == NULL)
    {
        return NULL;
    }
    /* A thrown ckc::Error is the authoritative failure: record it even if a
     * sticky status was already set, so the reported message matches the throw
     * site (the throw aborts before any later set_err could overwrite it). */
    b->status = (code == ROCKE_OK) ? ROCKE_ERR_VALUE : code;
    if(msg != NULL)
    {
        size_t n = strlen(msg);
        if(n >= sizeof(b->err))
        {
            n = sizeof(b->err) - 1;
        }
        memcpy(b->err, msg, n);
        b->err[n] = '\0';
    }
    else
    {
        b->err[0] = '\0';
    }
    return NULL;
}

/* ------------------------------------------------------- region plumbing */

rocke_region_t* rocke_i_new_region(rocke_ir_builder_t* b, const char* label)
{
    rocke_region_t* r;
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    r = (rocke_region_t*)rocke_arena_calloc(&b->arena, sizeof(*r));
    if(!r)
    {
        return (rocke_region_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "new_region: OOM");
    }
    r->label = rocke_arena_strdup(&b->arena, label ? label : "");
    if(!r->label)
    {
        return (rocke_region_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "new_region: OOM label");
    }
    r->ops = NULL;
    r->num_ops = 0;
    r->cap_ops = 0;
    return r;
}

/* Append an op to a region, growing its arena-backed ops array as needed.
 * The previous block is abandoned to the arena (bulk-freed later). */
static int rocke_region_append(rocke_ir_builder_t* b, rocke_region_t* r, rocke_op_t* op)
{
    if(r->num_ops >= r->cap_ops)
    {
        int nc = r->cap_ops ? r->cap_ops * 2 : 4;
        rocke_op_t** np
            = (rocke_op_t**)rocke_arena_alloc(&b->arena, sizeof(rocke_op_t*) * (size_t)nc);
        if(!np)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "region append: OOM");
            return -1;
        }
        if(r->ops && r->num_ops)
        {
            memcpy(np, r->ops, sizeof(rocke_op_t*) * (size_t)r->num_ops);
        }
        r->ops = np;
        r->cap_ops = nc;
    }
    r->ops[r->num_ops++] = op;
    return 0;
}

void rocke_i_emit(rocke_ir_builder_t* b, rocke_op_t* op)
{
    rocke_region_t* cur;
    if(!rocke_i_live(b) || op == NULL)
    {
        return;
    }
    if(b->region_depth <= 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "emit: no current region");
        return;
    }
    cur = b->region_stack[b->region_depth - 1];
    (void)rocke_region_append(b, cur, op);
}

void rocke_b_emit(rocke_ir_builder_t* b, rocke_op_t* op)
{
    rocke_i_emit(b, op);
}

void rocke_b_region_enter(rocke_ir_builder_t* b, rocke_region_t* r)
{
    if(!rocke_i_live(b))
    {
        return;
    }
    if(r == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "region_enter: NULL region");
        return;
    }
    if(b->region_depth >= ROCKE_REGION_STACK_MAX)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "region_enter: stack overflow");
        return;
    }
    b->region_stack[b->region_depth++] = r;
}

void rocke_b_region_leave(rocke_ir_builder_t* b)
{
    if(!rocke_i_live(b))
    {
        return;
    }
    if(b->region_depth <= 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "region_leave: stack underflow");
        return;
    }
    b->region_depth--;
}

rocke_region_t* rocke_b_current_region(rocke_ir_builder_t* b)
{
    if(b == NULL || b->region_depth <= 0)
    {
        return NULL;
    }
    return b->region_stack[b->region_depth - 1];
}

/* ------------------------------------------------------------- naming */

const char* rocke_b_fresh(rocke_ir_builder_t* b, const char* prefix)
{
    char* out;
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    b->counter += 1;
    out = rocke_arena_printf(&b->arena, "%%%s%d", prefix ? prefix : "v", b->counter);
    if(!out)
    {
        return (const char*)rocke_i_set_err(b, ROCKE_ERR_OOM, "fresh: OOM");
    }
    return out;
}

rocke_value_t*
    rocke_i_new_value(rocke_ir_builder_t* b, const char* prefix, const rocke_type_t* type)
{
    rocke_value_t* v;
    const char* nm;
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    nm = rocke_b_fresh(b, prefix);
    if(!nm)
    {
        return NULL;
    }
    v = (rocke_value_t*)rocke_arena_calloc(&b->arena, sizeof(*v));
    if(!v)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "new_value: OOM");
    }
    v->name = nm;
    v->type = type;
    v->op = NULL;
    return v;
}

rocke_value_t*
    rocke_i_value_named(rocke_ir_builder_t* b, const char* name, const rocke_type_t* type)
{
    rocke_value_t* v;
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    v = (rocke_value_t*)rocke_arena_calloc(&b->arena, sizeof(*v));
    if(!v)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "value_named: OOM");
    }
    v->name = rocke_arena_strdup(&b->arena, name ? name : "");
    if(!v->name)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "value_named: OOM name");
    }
    v->type = type;
    v->op = NULL;
    return v;
}

/* ------------------------------------------------------------- attr helpers */

rocke_attr_map_t rocke_i_attrs(rocke_ir_builder_t* b)
{
    rocke_attr_map_t m;
    (void)b;
    rocke_attr_map_init(&m);
    return m;
}

void rocke_i_attrs_copy(rocke_ir_builder_t* b, rocke_attr_map_t* dst, const rocke_attr_map_t* src)
{
    int i;
    if(dst == NULL)
    {
        return;
    }
    rocke_attr_map_init(dst);
    if(!rocke_i_live(b) || src == NULL)
    {
        return;
    }
    /* Deep-copy entries via the public setters so keys/strings are arena-owned
     * by the destination (Op.attrs = dict(attrs or {}) in Python). */
    for(i = 0; i < src->count; i++)
    {
        const rocke_attr_entry_t* e = &src->entries[i];
        switch(e->value.kind)
        {
        case ROCKE_ATTR_INT:
            rocke_attr_set_int(b, dst, e->key, e->value.u.i);
            break;
        case ROCKE_ATTR_FLOAT:
            rocke_attr_set_float(b, dst, e->key, e->value.u.f);
            break;
        case ROCKE_ATTR_STR:
            rocke_attr_set_str(b, dst, e->key, e->value.u.s);
            break;
        case ROCKE_ATTR_BOOL:
            rocke_attr_set_bool(b, dst, e->key, e->value.u.b);
            break;
        case ROCKE_ATTR_LIST:
        default:
            /* List-valued attrs (scf.for ``iter_args`` metadata) are
             * copied by sharing the source's arena-owned ``items`` array.
             * This is the FAITHFUL port of Python's attr copy, not a
             * shortcut: IRBuilder builds an Op with ``attrs=dict(attrs or
             * {})`` (ir.py: IRBuilder.op / IRBuilder.scf_for), and
             * ``dict(...)`` is a SHALLOW copy -- it duplicates the
             * key->value mapping but shares each value object by reference.
             * The list value here is the ``iter_meta`` list of dicts, built
             * once in scf_for and never mutated afterwards, so the shared
             * reference is observationally identical to a deep copy. Sharing
             * the same ``items`` pointer reproduces that reference-sharing
             * exactly and is safe under the single-arena lifetime (the list
             * and the destination map share one arena). The scalar cases
             * above route through the public setters because those values
             * are stored inline (no shared backing object), matching
             * dict()'s by-value copy of immutable scalars. */
            if(dst->count >= dst->cap)
            {
                int nc = dst->cap ? dst->cap * 2 : 4;
                rocke_attr_entry_t* ne = (rocke_attr_entry_t*)rocke_arena_alloc(
                    &b->arena, sizeof(rocke_attr_entry_t) * (size_t)nc);
                if(!ne)
                {
                    rocke_i_set_err(b, ROCKE_ERR_OOM, "attrs_copy: OOM");
                    return;
                }
                if(dst->entries && dst->count)
                {
                    memcpy(ne, dst->entries, sizeof(rocke_attr_entry_t) * (size_t)dst->count);
                }
                dst->entries = ne;
                dst->cap = nc;
            }
            dst->entries[dst->count].key = rocke_arena_strdup(&b->arena, e->key);
            dst->entries[dst->count].value = e->value;
            dst->count++;
            break;
        }
    }
}

/* --------------------------------------------------------- generic op build */

/* Shared implementation behind rocke_b_op / rocke_i_op. */
rocke_op_t* rocke_i_op(rocke_ir_builder_t* b,
                       rocke_opcode_t opcode,
                       rocke_value_t* const* operands,
                       int num_operands,
                       const rocke_type_t* const* result_types,
                       int num_results,
                       const rocke_attr_map_t* attrs,
                       rocke_region_t* const* regions,
                       int num_regions,
                       const char* result_name_hint,
                       const char* loc)
{
    rocke_op_t* op;
    int i;

    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(num_operands < 0)
        num_operands = 0;
    if(num_results < 0)
        num_results = 0;
    if(num_regions < 0)
        num_regions = 0;

    op = (rocke_op_t*)rocke_arena_calloc(&b->arena, sizeof(*op));
    if(!op)
    {
        return (rocke_op_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "op: OOM");
    }
    op->opcode = opcode;
    op->name = rocke_opcode_name(opcode);
    op->loc = loc ? rocke_arena_strdup(&b->arena, loc) : NULL;

    /* operands: copy into an arena array */
    if(num_operands > 0)
    {
        op->operands = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, sizeof(rocke_value_t*) * (size_t)num_operands);
        if(!op->operands)
        {
            return (rocke_op_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "op: OOM operands");
        }
        for(i = 0; i < num_operands; i++)
        {
            op->operands[i] = operands ? operands[i] : NULL;
        }
        op->num_operands = num_operands;
    }

    /* results: one fresh Value per result type, named with result_name_hint */
    if(num_results > 0)
    {
        op->results = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, sizeof(rocke_value_t*) * (size_t)num_results);
        if(!op->results)
        {
            return (rocke_op_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "op: OOM results");
        }
        for(i = 0; i < num_results; i++)
        {
            const rocke_type_t* rt = result_types ? result_types[i] : NULL;
            rocke_value_t* r = rocke_i_new_value(b, result_name_hint ? result_name_hint : "v", rt);
            if(!r)
            {
                return NULL; /* error already set */
            }
            op->results[i] = r;
        }
        op->num_results = num_results;
    }

    /* attrs: deep-copy borrowed map (Python dict(attrs or {})) */
    rocke_i_attrs_copy(b, &op->attrs, attrs);
    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* regions: copy the borrowed region pointers into an arena array */
    if(num_regions > 0)
    {
        op->regions = (rocke_region_t**)rocke_arena_alloc(
            &b->arena, sizeof(rocke_region_t*) * (size_t)num_regions);
        if(!op->regions)
        {
            return (rocke_op_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "op: OOM regions");
        }
        for(i = 0; i < num_regions; i++)
        {
            op->regions[i] = regions ? regions[i] : NULL;
        }
        op->num_regions = num_regions;
    }

    /* link results back to the producing op */
    for(i = 0; i < op->num_results; i++)
    {
        if(op->results[i])
        {
            op->results[i]->op = op;
        }
    }

    /* emit into the current region */
    rocke_i_emit(b, op);
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    return op;
}

rocke_op_t* rocke_b_op(rocke_ir_builder_t* b,
                       rocke_opcode_t opcode,
                       rocke_value_t* const* operands,
                       int num_operands,
                       const rocke_type_t* const* result_types,
                       int num_results,
                       const rocke_attr_map_t* attrs,
                       rocke_region_t* const* regions,
                       int num_regions,
                       const char* result_name_hint,
                       const char* loc)
{
    return rocke_i_op(b,
                      opcode,
                      operands,
                      num_operands,
                      result_types,
                      num_results,
                      attrs,
                      regions,
                      num_regions,
                      result_name_hint,
                      loc);
}

/* --------------------------------------------------- emission shorthands */

rocke_value_t* rocke_i_op1(rocke_ir_builder_t* b,
                           rocke_opcode_t opcode,
                           rocke_value_t* const* operands,
                           int num_operands,
                           const rocke_type_t* result_type,
                           const rocke_attr_map_t* attrs,
                           const char* result_name_hint)
{
    const rocke_type_t* rts[1];
    rocke_op_t* op;
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    rts[0] = result_type;
    op = rocke_i_op(
        b, opcode, operands, num_operands, rts, 1, attrs, NULL, 0, result_name_hint, NULL);
    if(!op)
    {
        return NULL;
    }
    return op->results[0];
}

rocke_op_t* rocke_i_op0(rocke_ir_builder_t* b,
                        rocke_opcode_t opcode,
                        rocke_value_t* const* operands,
                        int num_operands,
                        const rocke_attr_map_t* attrs)
{
    return rocke_i_op(b, opcode, operands, num_operands, NULL, 0, attrs, NULL, 0, "v", NULL);
}

rocke_value_t* rocke_i_binop(rocke_ir_builder_t* b,
                             rocke_opcode_t opcode,
                             rocke_value_t* a,
                             rocke_value_t* bb,
                             const char* result_name_hint)
{
    rocke_value_t* operands[2];
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(a == NULL || bb == NULL)
    {
#ifdef ROCKE_TRACE_BINOP
        {
            extern int backtrace(void**, int);
            extern char** backtrace_symbols(void* const*, int);
            void* bt[40];
            int n = backtrace(bt, 40);
            char** syms = backtrace_symbols(bt, n);
            fprintf(stderr,
                    "[binop NULL] hint=%s a=%p bb=%p\n",
                    result_name_hint ? result_name_hint : "(null)",
                    (void*)a,
                    (void*)bb);
            for(int _i = 0; _i < n; _i++)
                fprintf(stderr, "  %s\n", syms[_i]);
        }
#endif
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "binop: NULL operand");
    }
    operands[0] = a;
    operands[1] = bb;
    return rocke_i_op1(b, opcode, operands, 2, a->type, NULL, result_name_hint);
}

rocke_value_t* rocke_i_unop(rocke_ir_builder_t* b,
                            rocke_opcode_t opcode,
                            rocke_value_t* a,
                            const char* result_name_hint)
{
    rocke_value_t* operands[1];
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(a == NULL)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "unop: NULL operand");
    }
    operands[0] = a;
    return rocke_i_op1(b, opcode, operands, 1, a->type, NULL, result_name_hint);
}

/* ----------------------------------------------------- type-system helpers */

bool rocke_i_type_is(const rocke_type_t* t, const char* name)
{
    if(t == NULL || name == NULL || t->name == NULL)
    {
        return false;
    }
    return strcmp(t->name, name) == 0;
}

bool rocke_i_is_vector(const rocke_type_t* t, const char* elem_name, int count)
{
    if(t == NULL || t->kind != ROCKE_TYPE_VECTOR)
    {
        return false;
    }
    if(count >= 0 && t->count != count)
    {
        return false;
    }
    if(elem_name != NULL)
    {
        if(t->elem == NULL || t->elem->name == NULL)
        {
            return false;
        }
        if(strcmp(t->elem->name, elem_name) != 0)
        {
            return false;
        }
    }
    return true;
}

const rocke_type_t* rocke_i_elem_of(const rocke_type_t* t)
{
    if(t != NULL && t->kind == ROCKE_TYPE_VECTOR && t->elem != NULL)
    {
        return t->elem;
    }
    return t;
}

int rocke_i_count_of(const rocke_type_t* t)
{
    if(t != NULL && t->kind == ROCKE_TYPE_VECTOR)
    {
        return t->count;
    }
    return 1;
}

/* ============================== BUILDER ================================= */

rocke_status_t rocke_ir_builder_init(rocke_ir_builder_t* b, const char* kernel_name)
{
    rocke_kernel_def_t* k;
    rocke_region_t* entry;

    if(b == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    memset(b, 0, sizeof(*b));

    if(rocke_arena_init(&b->arena, 0) != 0)
    {
        b->status = ROCKE_ERR_OOM;
        snprintf(b->err, sizeof(b->err), "builder_init: arena OOM");
        return ROCKE_ERR_OOM;
    }

    b->status = ROCKE_OK;
    b->err[0] = '\0';
    b->counter = 0;
    b->region_depth = 0;

    b->param_names = NULL;
    b->param_values = NULL;
    b->num_param_lookup = 0;
    b->cap_param_lookup = 0;

    k = (rocke_kernel_def_t*)rocke_arena_calloc(&b->arena, sizeof(*k));
    if(!k)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "builder_init: OOM kernel");
        return b->status;
    }
    k->name = rocke_arena_strdup(&b->arena, kernel_name ? kernel_name : "");
    if(!k->name)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "builder_init: OOM name");
        return b->status;
    }
    k->params = NULL;
    k->num_params = 0;
    k->cap_params = 0;
    rocke_attr_map_init(&k->attrs);

    entry = rocke_i_new_region(b, "entry");
    if(!entry)
    {
        return b->status;
    }
    k->body = entry;
    b->kernel = k;

    /* current region == kernel body (Python pushes self.kernel.body) */
    b->region_stack[b->region_depth++] = entry;

    return ROCKE_OK;
}

void rocke_ir_builder_free(rocke_ir_builder_t* b)
{
    if(b == NULL)
    {
        return;
    }
    rocke_arena_destroy(&b->arena);
    memset(b, 0, sizeof(*b));
}

bool rocke_ir_builder_ok(const rocke_ir_builder_t* b)
{
    return b != NULL && b->status == ROCKE_OK;
}

rocke_status_t rocke_ir_builder_status(const rocke_ir_builder_t* b)
{
    return b ? b->status : ROCKE_ERR_VALUE;
}

const char* rocke_ir_builder_error(const rocke_ir_builder_t* b)
{
    if(b == NULL)
    {
        return "";
    }
    return b->err;
}

rocke_kernel_def_t* rocke_ir_builder_kernel(rocke_ir_builder_t* b)
{
    return b ? b->kernel : NULL;
}

/* ------------------------------------------------------------- params */

/* Register a param Value in the builder's name->value lookup. */
static int rocke_param_lookup_add(rocke_ir_builder_t* b, const char* name, rocke_value_t* v)
{
    if(b->num_param_lookup >= b->cap_param_lookup)
    {
        int nc = b->cap_param_lookup ? b->cap_param_lookup * 2 : 8;
        const char** nn
            = (const char**)rocke_arena_alloc(&b->arena, sizeof(const char*) * (size_t)nc);
        rocke_value_t** nv
            = (rocke_value_t**)rocke_arena_alloc(&b->arena, sizeof(rocke_value_t*) * (size_t)nc);
        if(!nn || !nv)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "param: OOM lookup");
            return -1;
        }
        if(b->param_names && b->num_param_lookup)
        {
            memcpy(nn, b->param_names, sizeof(const char*) * (size_t)b->num_param_lookup);
            memcpy(nv, b->param_values, sizeof(rocke_value_t*) * (size_t)b->num_param_lookup);
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

static int rocke_kernel_params_add(rocke_ir_builder_t* b, rocke_param_t* p)
{
    rocke_kernel_def_t* k = b->kernel;
    if(k->num_params >= k->cap_params)
    {
        int nc = k->cap_params ? k->cap_params * 2 : 8;
        rocke_param_t** np
            = (rocke_param_t**)rocke_arena_alloc(&b->arena, sizeof(rocke_param_t*) * (size_t)nc);
        if(!np)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "param: OOM params");
            return -1;
        }
        if(k->params && k->num_params)
        {
            memcpy(np, k->params, sizeof(rocke_param_t*) * (size_t)k->num_params);
        }
        k->params = np;
        k->cap_params = nc;
    }
    k->params[k->num_params++] = p;
    return 0;
}

rocke_value_t* rocke_b_param(rocke_ir_builder_t* b,
                             const char* name,
                             const rocke_type_t* t,
                             const rocke_param_opts_t* opts)
{
    rocke_value_t* v;
    rocke_param_t* p;
    char* full_name;
    int i;

    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(name == NULL)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "param: NULL name");
    }

    /* duplicate kernel parameter check (Python ValueError) */
    for(i = 0; i < b->num_param_lookup; i++)
    {
        if(b->param_names[i] && strcmp(b->param_names[i], name) == 0)
        {
            return (rocke_value_t*)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "duplicate kernel parameter '%s'", name);
        }
    }

    /* Value name carries the leading '%'. */
    full_name = rocke_arena_printf(&b->arena, "%%%s", name);
    if(!full_name)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "param: OOM name");
    }
    v = rocke_i_value_named(b, full_name, t);
    if(!v)
    {
        return NULL;
    }

    /* Param record (name WITHOUT leading '%'). */
    p = (rocke_param_t*)rocke_arena_calloc(&b->arena, sizeof(*p));
    if(!p)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "param: OOM record");
    }
    p->name = rocke_arena_strdup(&b->arena, name);
    if(!p->name)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "param: OOM record name");
    }
    p->type = t;
    rocke_attr_map_init(&p->attrs);

    /* Materialise ABI attrs from the opts struct (only set fields, mirroring
     * Python dict(**attrs) which only carries the kwargs actually passed). */
    if(opts != NULL)
    {
        if(opts->noalias_set)
        {
            rocke_attr_set_bool(b, &p->attrs, "noalias", opts->noalias);
        }
        if(opts->readonly_set)
        {
            rocke_attr_set_bool(b, &p->attrs, "readonly", opts->readonly);
        }
        if(opts->writeonly_set)
        {
            rocke_attr_set_bool(b, &p->attrs, "writeonly", opts->writeonly);
        }
        if(opts->align_set)
        {
            rocke_attr_set_int(b, &p->attrs, "align", (int64_t)opts->align);
        }
        if(opts->addr_space != NULL)
        {
            rocke_attr_set_str(b, &p->attrs, "addr_space", opts->addr_space);
        }
    }

    if(rocke_kernel_params_add(b, p) != 0)
    {
        return NULL;
    }
    if(rocke_param_lookup_add(b, p->name, v) != 0)
    {
        return NULL;
    }
    return v;
}

rocke_value_t* rocke_b_get_param(rocke_ir_builder_t* b, const char* name)
{
    int i;
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(name == NULL)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_KEY, "get_param: NULL name");
    }
    for(i = 0; i < b->num_param_lookup; i++)
    {
        if(b->param_names[i] && strcmp(b->param_names[i], name) == 0)
        {
            return b->param_values[i];
        }
    }
    return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_KEY, "unknown param '%s'", name);
}
