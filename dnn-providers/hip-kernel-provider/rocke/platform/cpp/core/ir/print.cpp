// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * ir_print.c -- MLIR-style textual printer for the C99 CK DSL IR.
 *
 * Faithful port of rocke/core/ir_print.py. Helper-for-helper mapping:
 *
 *   Python                       C99 (this file)
 *   --------------------------   -----------------------------------------------
 *   _format_operand(v)           emit_operand()      -> v.name
 *   _attr_value(v)               emit_attr_value()   -> str()/quoted
 *   _format_attrs(attrs)         emit_attrs()        -> sorted " {k = v, ...}"
 *   _format_results(results)     emit_results()      -> "a, b = "
 *   _format_types(results)       emit_types()        -> " : t0, t1"
 *   _print_op(op, indent)        emit_op()           -> recursive, regions
 *   print_ir(kernel)             rocke_print_ir()      -> "kernel @name(...) {"
 *
 * All emission goes through rocke_strbuf (the C stand-in for the Python list of
 * lines joined with "\n"). No arena allocation is needed: the printer only
 * reads the (arena-owned) graph and writes into the caller's strbuf, except for
 * sorting attrs where a small fixed/stack copy of pointers is used.
 */
#include "rocke/ir_print.h"

#include "rocke/float_format.h"

#include <stdlib.h>
#include <string.h>

#include "rocke/ir_internal.h"

/* Python repr() of an Optional[str] appended to `out`: "None", or
 * rocke_py_str_repr's text. */
static void emit_repr_str(rocke_strbuf_t* out, const char* s)
{
    if(!s)
    {
        rocke_strbuf_append(out, "None");
        return;
    }
    char small[128];
    const size_t n = rocke_py_str_repr(s, small, sizeof(small));
    if(n < sizeof(small))
    {
        rocke_strbuf_append(out, small);
        return;
    }
    char* big = (char*)malloc(n + 1);
    if(!big)
    {
        out->oom = 1;
        return;
    }
    rocke_py_str_repr(s, big, n + 1);
    rocke_strbuf_append(out, big);
    free(big);
}

/* --------------------------------------------------------------- operands */

/* Python _format_operand: returns v.name (already includes the leading '%'). */
static void emit_operand(rocke_strbuf_t* out, const rocke_value_t* v)
{
    rocke_strbuf_append(out, v && v->name ? v->name : "");
}

/* ------------------------------------------------------------ attr values */

/* Forward declarations for the repr-based renderers used by the ROCKE_ATTR_LIST
 * case below. A list attr is a Python List[Dict[str, Any]] (scf.for iter_args
 * metadata), and str(list) renders every contained element with repr(), which
 * differs from the top-level _attr_value (repr single-quotes strings, whereas
 * the top level double-quotes them). Python repr(str) itself is emit_repr_str
 * (rocke_py_str_repr). */
static void emit_repr_value(rocke_strbuf_t* out, const rocke_attr_value_t* v);
static void emit_repr_map(rocke_strbuf_t* out, const struct rocke_attr_map* m);

/* Python repr() of a single attr value as it appears *inside* a printed
 * container (list/dict). repr(str) single-quotes; repr(bool/int/float) == str.
 * Mirrors the recursive str(...) walk CPython performs on a list of dicts. */
static void emit_repr_value(rocke_strbuf_t* out, const rocke_attr_value_t* v)
{
    switch(v->kind)
    {
    case ROCKE_ATTR_STR:
        emit_repr_str(out, v->u.s);
        break;
    case ROCKE_ATTR_BOOL:
        rocke_strbuf_append(out, v->u.b ? "True" : "False");
        break;
    case ROCKE_ATTR_INT:
        rocke_strbuf_appendf(out, "%lld", (long long)v->u.i);
        break;
    case ROCKE_ATTR_FLOAT:
        rocke_append_python_float_repr(out, v->u.f);
        break;
    case ROCKE_ATTR_LIST:
    {
        /* Nested list: str(list) -> "[e0, e1]" with repr'd elements. */
        rocke_strbuf_append_char(out, '[');
        for(int i = 0; i < v->u.list.count; ++i)
        {
            if(i > 0)
            {
                rocke_strbuf_append(out, ", ");
            }
            emit_repr_map(out, v->u.list.items[i]);
        }
        rocke_strbuf_append_char(out, ']');
        break;
    }
    case ROCKE_ATTR_INT_LIST:
    {
        /* a list of bare ints (e.g. agpr_alloc pair). */
        rocke_strbuf_append_char(out, '[');
        for(int i = 0; i < v->u.ilist.count; ++i)
        {
            if(i > 0)
            {
                rocke_strbuf_append(out, ", ");
            }
            rocke_strbuf_appendf(out, "%lld", (long long)v->u.ilist.ints[i]);
        }
        rocke_strbuf_append_char(out, ']');
        break;
    }
    default:
        break;
    }
}

/* Python str(dict) -> "{'k0': v0, 'k1': v1}" in *insertion* order (Python 3.7+
 * dicts preserve insertion order; CPython does NOT sort dict keys for str/repr).
 * The C attr map stores entries in insertion order, matching the order in which
 * the Python builder populated the dict ({"name": ..., "type": ...}). Keys are
 * always Python str -> repr-quoted; values are repr'd. */
static void emit_repr_map(rocke_strbuf_t* out, const struct rocke_attr_map* m)
{
    rocke_strbuf_append_char(out, '{');
    if(m)
    {
        for(int i = 0; i < m->count; ++i)
        {
            if(i > 0)
            {
                rocke_strbuf_append(out, ", ");
            }
            emit_repr_str(out, m->entries[i].key);
            rocke_strbuf_append(out, ": ");
            emit_repr_value(out, &m->entries[i].value);
        }
    }
    rocke_strbuf_append_char(out, '}');
}

/* Python _attr_value:
 *   str  -> '"' + value + '"'
 *   else -> str(value)   (bool -> "True"/"False", int -> decimal, float -> ...)
 */
static void emit_attr_value(rocke_strbuf_t* out, const rocke_attr_value_t* v)
{
    switch(v->kind)
    {
    case ROCKE_ATTR_STR:
        rocke_strbuf_append_char(out, '"');
        rocke_strbuf_append(out, v->u.s ? v->u.s : "");
        rocke_strbuf_append_char(out, '"');
        break;
    case ROCKE_ATTR_BOOL:
        /* Python str(bool) -> "True" / "False". */
        rocke_strbuf_append(out, v->u.b ? "True" : "False");
        break;
    case ROCKE_ATTR_INT:
        rocke_strbuf_appendf(out, "%lld", (long long)v->u.i);
        break;
    case ROCKE_ATTR_FLOAT:
        rocke_append_python_float_repr(out, v->u.f);
        break;
    case ROCKE_ATTR_LIST:
        /* Python _attr_value falls through to str(value); for a list attr
         * (scf.for "iter_args" -> List[Dict[str, Any]]) that is str(list),
         * which reprs each element: "[{'name': '%x', 'type': 't'}, ...]".
         * Empty list -> "[]". */
        rocke_strbuf_append_char(out, '[');
        for(int i = 0; i < v->u.list.count; ++i)
        {
            if(i > 0)
            {
                rocke_strbuf_append(out, ", ");
            }
            emit_repr_map(out, v->u.list.items[i]);
        }
        rocke_strbuf_append_char(out, ']');
        break;
    case ROCKE_ATTR_INT_LIST:
        /* a list of bare ints (e.g. agpr_alloc pair). */
        rocke_strbuf_append_char(out, '[');
        for(int i = 0; i < v->u.ilist.count; ++i)
        {
            if(i > 0)
            {
                rocke_strbuf_append(out, ", ");
            }
            rocke_strbuf_appendf(out, "%lld", (long long)v->u.ilist.ints[i]);
        }
        rocke_strbuf_append_char(out, ']');
        break;
    default:
        rocke_strbuf_append(out, "");
        break;
    }
}

/* ------------------------------------------------------------------ attrs */

/* Comparator for sorting attr entry pointers by key (Python sorted(items())
 * sorts by (key, value); keys are unique within a map so key order suffices). */
static int attr_entry_cmp(const void* pa, const void* pb)
{
    const rocke_attr_entry_t* a = *(const rocke_attr_entry_t* const*)pa;
    const rocke_attr_entry_t* b = *(const rocke_attr_entry_t* const*)pb;
    const char* ka = a->key ? a->key : "";
    const char* kb = b->key ? b->key : "";
    return strcmp(ka, kb);
}

/* Python _format_attrs: "" if empty, else " {k0 = v0, k1 = v1}" sorted by key. */
static void emit_attrs(rocke_strbuf_t* out, const rocke_attr_map_t* attrs)
{
    if(!attrs || attrs->count <= 0)
    {
        return;
    }
    int count = attrs->count;

    /* Sort a copy of the entry pointers; the map itself is left untouched
     * (Python sorts a copy too). Small heap alloc keeps recursion stack flat. */
    const rocke_attr_entry_t** order
        = (const rocke_attr_entry_t**)malloc((size_t)count * sizeof(*order));
    if(!order)
    {
        out->oom = 1;
        return;
    }
    for(int i = 0; i < count; ++i)
    {
        order[i] = &attrs->entries[i];
    }
    qsort(order, (size_t)count, sizeof(*order), attr_entry_cmp);

    rocke_strbuf_append(out, " {");
    for(int i = 0; i < count; ++i)
    {
        if(i > 0)
        {
            rocke_strbuf_append(out, ", ");
        }
        rocke_strbuf_append(out, order[i]->key ? order[i]->key : "");
        rocke_strbuf_append(out, " = ");
        emit_attr_value(out, &order[i]->value);
    }
    rocke_strbuf_append_char(out, '}');

    free(order);
}

/* ----------------------------------------------------------- results/types */

/* Python _format_results: "" if none, else "r0, r1 = ". */
static void emit_results(rocke_strbuf_t* out, rocke_value_t* const* results, int num_results)
{
    if(num_results <= 0)
    {
        return;
    }
    for(int i = 0; i < num_results; ++i)
    {
        if(i > 0)
        {
            rocke_strbuf_append(out, ", ");
        }
        rocke_strbuf_append(out, results[i] && results[i]->name ? results[i]->name : "");
    }
    rocke_strbuf_append(out, " = ");
}

/* Python _format_types: "" if none, else " : t0, t1". */
static void emit_types(rocke_strbuf_t* out, rocke_value_t* const* results, int num_results)
{
    if(num_results <= 0)
    {
        return;
    }
    rocke_strbuf_append(out, " : ");
    for(int i = 0; i < num_results; ++i)
    {
        if(i > 0)
        {
            rocke_strbuf_append(out, ", ");
        }
        const rocke_type_t* t = results[i] ? results[i]->type : NULL;
        rocke_strbuf_append(out, (t && t->name) ? t->name : "");
    }
}

/* ------------------------------------------------------------- indentation */

static void emit_pad(rocke_strbuf_t* out, int indent)
{
    for(int i = 0; i < indent; ++i)
    {
        rocke_strbuf_append_char(out, ' ');
    }
}

/* ---------------------------------------------------------------- ops */

/* Python _print_op(op, indent): emits one line for the op (plus nested region
 * lines) into `out`, each line followed by '\n'. The caller is responsible for
 * not appending a trailing newline at the very end (see rocke_print_ir). */
static void emit_op(rocke_strbuf_t* out, const rocke_op_t* op, int indent)
{
    emit_pad(out, indent);
    emit_results(out, op->results, op->num_results);
    rocke_strbuf_append(out, op->name ? op->name : "");

    if(op->num_operands > 0)
    {
        rocke_strbuf_append_char(out, ' ');
        for(int i = 0; i < op->num_operands; ++i)
        {
            if(i > 0)
            {
                rocke_strbuf_append(out, ", ");
            }
            emit_operand(out, op->operands[i]);
        }
    }

    emit_attrs(out, &op->attrs);
    emit_types(out, op->results, op->num_results);
    rocke_strbuf_append_char(out, '\n');

    for(int r = 0; r < op->num_regions; ++r)
    {
        const rocke_region_t* region = op->regions[r];
        emit_pad(out, indent);
        rocke_strbuf_append(out, "  region ");
        emit_repr_str(out, region ? region->label : NULL);
        rocke_strbuf_append(out, " {\n");
        if(region)
        {
            for(int i = 0; i < region->num_ops; ++i)
            {
                emit_op(out, region->ops[i], indent + 4);
            }
        }
        emit_pad(out, indent);
        rocke_strbuf_append(out, "  }\n");
    }
}

/* ---------------------------------------------------------------- kernel */

void rocke_print_ir(const rocke_kernel_def_t* kernel, rocke_strbuf_t* out)
{
    if(!kernel || !out)
    {
        return;
    }

    /* Header: "kernel @name(%p0: t0, %p1: t1) {" */
    rocke_strbuf_append(out, "kernel @");
    rocke_strbuf_append(out, kernel->name ? kernel->name : "");
    rocke_strbuf_append_char(out, '(');
    for(int i = 0; i < kernel->num_params; ++i)
    {
        const rocke_param_t* p = kernel->params[i];
        if(i > 0)
        {
            rocke_strbuf_append(out, ", ");
        }
        /* Python _format_param: f"%{p.name}: {p.type.name}", then the facts the
         * declaration states. Param.name has no leading '%'. */
        rocke_strbuf_append_char(out, '%');
        rocke_strbuf_append(out, (p && p->name) ? p->name : "");
        rocke_strbuf_append(out, ": ");
        const rocke_type_t* t = p ? p->type : NULL;
        rocke_strbuf_append(out, (t && t->name) ? t->name : "");
        rocke_param_facts_print(p, out);
    }
    rocke_strbuf_append(out, ") {\n");

    /* Body ops at indent 2. */
    if(kernel->body)
    {
        for(int i = 0; i < kernel->body->num_ops; ++i)
        {
            emit_op(out, kernel->body->ops[i], 2);
        }
    }

    /* Closing brace. Python joins lines with "\n" and the final line is "}" with
     * no trailing newline, so we close without an extra '\n'. */
    rocke_strbuf_append_char(out, '}');
}

char* rocke_print_ir_alloc(const rocke_kernel_def_t* kernel)
{
    rocke_strbuf_t sb;
    if(rocke_strbuf_init(&sb, 256) != 0)
    {
        return NULL;
    }
    rocke_print_ir(kernel, &sb);
    if(sb.oom)
    {
        rocke_strbuf_free(&sb);
        return NULL;
    }
    char* s = rocke_strbuf_detach(&sb);
    rocke_strbuf_free(&sb);
    if(s)
    {
        return s;
    }
    /* Empty kernel never happens (header is always emitted), but detach can
     * return NULL for a never-allocated buffer; hand back an empty string. */
    char* empty = (char*)malloc(1);
    if(empty)
    {
        empty[0] = '\0';
    }
    return empty;
}
