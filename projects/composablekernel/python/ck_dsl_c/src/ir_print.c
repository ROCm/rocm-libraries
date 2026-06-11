/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ir_print.c -- MLIR-style textual printer for the C99 CK DSL IR.
 *
 * Faithful port of ck_dsl/core/ir_print.py. Helper-for-helper mapping:
 *
 *   Python                       C99 (this file)
 *   --------------------------   -----------------------------------------------
 *   _format_operand(v)           emit_operand()      -> v.name
 *   _attr_value(v)               emit_attr_value()   -> str()/quoted
 *   _format_attrs(attrs)         emit_attrs()        -> sorted " {k = v, ...}"
 *   _format_results(results)     emit_results()      -> "a, b = "
 *   _format_types(results)       emit_types()        -> " : t0, t1"
 *   _print_op(op, indent)        emit_op()           -> recursive, regions
 *   print_ir(kernel)             ckc_print_ir()      -> "kernel @name(...) {"
 *
 * All emission goes through ckc_strbuf (the C stand-in for the Python list of
 * lines joined with "\n"). No arena allocation is needed: the printer only
 * reads the (arena-owned) graph and writes into the caller's strbuf, except for
 * sorting attrs where a small fixed/stack copy of pointers is used.
 */
#include "ckc/ir_print.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------- operands */

/* Python _format_operand: returns v.name (already includes the leading '%'). */
static void emit_operand(ckc_strbuf_t *out, const ckc_value_t *v) {
    ckc_strbuf_append(out, v && v->name ? v->name : "");
}

/* ------------------------------------------------------------ attr values */

/* Python str(float). CPython prints the shortest decimal string that round-trips
 * (e.g. "1.0", "0.5", "3.14"), and always includes a decimal point or exponent
 * for finite floats. C99 has no built-in shortest-round-trip formatter, so this
 * is a best-effort approximation: we use %.17g (guaranteed round-trippable for
 * IEEE-754 double) and append ".0" when the result has no '.', 'e', 'n' (nan),
 * or 'i' (inf), so integral values print like Python's "1.0".
 *
 * TODO(port): exact byte-for-byte parity with CPython repr/str(float) (shortest
 * round-trip via Grisu/Ryu) is out of scope; %.17g can emit more digits than
 * CPython for some values (e.g. 0.1). Float-valued attrs are rare in printed IR. */
static void emit_float(ckc_strbuf_t *out, double f) {
    char buf[64];
    int n = snprintf(buf, sizeof(buf), "%.17g", f);
    if (n < 0 || n >= (int)sizeof(buf)) {
        ckc_strbuf_append(out, "0.0");
        return;
    }
    int has_point = 0;
    for (int i = 0; i < n; ++i) {
        char c = buf[i];
        if (c == '.' || c == 'e' || c == 'E' || c == 'n' || c == 'i') {
            has_point = 1;
            break;
        }
    }
    ckc_strbuf_append(out, buf);
    if (!has_point) {
        ckc_strbuf_append(out, ".0");
    }
}

/* Python _attr_value:
 *   str  -> '"' + value + '"'
 *   else -> str(value)   (bool -> "True"/"False", int -> decimal, float -> ...)
 */
static void emit_attr_value(ckc_strbuf_t *out, const ckc_attr_value_t *v) {
    switch (v->kind) {
        case CKC_ATTR_STR:
            ckc_strbuf_append_char(out, '"');
            ckc_strbuf_append(out, v->u.s ? v->u.s : "");
            ckc_strbuf_append_char(out, '"');
            break;
        case CKC_ATTR_BOOL:
            /* Python str(bool) -> "True" / "False". */
            ckc_strbuf_append(out, v->u.b ? "True" : "False");
            break;
        case CKC_ATTR_INT:
            ckc_strbuf_appendf(out, "%lld", (long long)v->u.i);
            break;
        case CKC_ATTR_FLOAT:
            emit_float(out, v->u.f);
            break;
        case CKC_ATTR_LIST:
            /* Python str(list) of nested attr maps has no stable textual form in
             * the frozen contract; the original printer only ever sees scalar
             * attr values in practice (scf.for iter_args metadata is consumed by
             * lowerers, not printed). */
            /* TODO(port): render CKC_ATTR_LIST to match Python str(list). */
            ckc_strbuf_append(out, "[...]");
            break;
        default:
            ckc_strbuf_append(out, "");
            break;
    }
}

/* ------------------------------------------------------------------ attrs */

/* Comparator for sorting attr entry pointers by key (Python sorted(items())
 * sorts by (key, value); keys are unique within a map so key order suffices). */
static int attr_entry_cmp(const void *pa, const void *pb) {
    const ckc_attr_entry_t *a = *(const ckc_attr_entry_t *const *)pa;
    const ckc_attr_entry_t *b = *(const ckc_attr_entry_t *const *)pb;
    const char *ka = a->key ? a->key : "";
    const char *kb = b->key ? b->key : "";
    return strcmp(ka, kb);
}

/* Python _format_attrs: "" if empty, else " {k0 = v0, k1 = v1}" sorted by key. */
static void emit_attrs(ckc_strbuf_t *out, const ckc_attr_map_t *attrs) {
    if (!attrs || attrs->count <= 0) {
        return;
    }
    int count = attrs->count;

    /* Sort a copy of the entry pointers; the map itself is left untouched
     * (Python sorts a copy too). Small heap alloc keeps recursion stack flat. */
    const ckc_attr_entry_t **order =
        (const ckc_attr_entry_t **)malloc((size_t)count * sizeof(*order));
    if (!order) {
        out->oom = 1;
        return;
    }
    for (int i = 0; i < count; ++i) {
        order[i] = &attrs->entries[i];
    }
    qsort(order, (size_t)count, sizeof(*order), attr_entry_cmp);

    ckc_strbuf_append(out, " {");
    for (int i = 0; i < count; ++i) {
        if (i > 0) {
            ckc_strbuf_append(out, ", ");
        }
        ckc_strbuf_append(out, order[i]->key ? order[i]->key : "");
        ckc_strbuf_append(out, " = ");
        emit_attr_value(out, &order[i]->value);
    }
    ckc_strbuf_append_char(out, '}');

    free(order);
}

/* ----------------------------------------------------------- results/types */

/* Python _format_results: "" if none, else "r0, r1 = ". */
static void emit_results(ckc_strbuf_t *out, ckc_value_t *const *results,
                         int num_results) {
    if (num_results <= 0) {
        return;
    }
    for (int i = 0; i < num_results; ++i) {
        if (i > 0) {
            ckc_strbuf_append(out, ", ");
        }
        ckc_strbuf_append(out, results[i] && results[i]->name ? results[i]->name : "");
    }
    ckc_strbuf_append(out, " = ");
}

/* Python _format_types: "" if none, else " : t0, t1". */
static void emit_types(ckc_strbuf_t *out, ckc_value_t *const *results,
                       int num_results) {
    if (num_results <= 0) {
        return;
    }
    ckc_strbuf_append(out, " : ");
    for (int i = 0; i < num_results; ++i) {
        if (i > 0) {
            ckc_strbuf_append(out, ", ");
        }
        const ckc_type_t *t = results[i] ? results[i]->type : NULL;
        ckc_strbuf_append(out, (t && t->name) ? t->name : "");
    }
}

/* ------------------------------------------------------------- indentation */

static void emit_pad(ckc_strbuf_t *out, int indent) {
    for (int i = 0; i < indent; ++i) {
        ckc_strbuf_append_char(out, ' ');
    }
}

/* Python repr(label) for a str: single-quoted. Handles the common identifier
 * labels ("body","then","entry"). CPython switches to double-quotes when the
 * string contains a single quote but no double quote, and escapes backslashes /
 * non-printables; those cases never occur for region labels in the engine. */
static void emit_repr_str(ckc_strbuf_t *out, const char *s) {
    if (!s) {
        /* Python repr(None) -> "None"; labels are always strings, but be safe. */
        ckc_strbuf_append(out, "None");
        return;
    }
    int has_single = 0, has_double = 0;
    for (const char *p = s; *p; ++p) {
        if (*p == '\'') has_single = 1;
        else if (*p == '"') has_double = 1;
    }
    char quote = (has_single && !has_double) ? '"' : '\'';
    ckc_strbuf_append_char(out, quote);
    for (const char *p = s; *p; ++p) {
        char c = *p;
        if (c == '\\' || c == quote) {
            ckc_strbuf_append_char(out, '\\');
            ckc_strbuf_append_char(out, c);
        } else if (c == '\n') {
            ckc_strbuf_append(out, "\\n");
        } else if (c == '\t') {
            ckc_strbuf_append(out, "\\t");
        } else if (c == '\r') {
            ckc_strbuf_append(out, "\\r");
        } else {
            /* TODO(port): non-printable bytes are emitted verbatim; CPython repr
             * would render them as \xNN escapes. Region labels are ASCII idents. */
            ckc_strbuf_append_char(out, c);
        }
    }
    ckc_strbuf_append_char(out, quote);
}

/* ---------------------------------------------------------------- ops */

/* Python _print_op(op, indent): emits one line for the op (plus nested region
 * lines) into `out`, each line followed by '\n'. The caller is responsible for
 * not appending a trailing newline at the very end (see ckc_print_ir). */
static void emit_op(ckc_strbuf_t *out, const ckc_op_t *op, int indent) {
    emit_pad(out, indent);
    emit_results(out, op->results, op->num_results);
    ckc_strbuf_append(out, op->name ? op->name : "");

    if (op->num_operands > 0) {
        ckc_strbuf_append_char(out, ' ');
        for (int i = 0; i < op->num_operands; ++i) {
            if (i > 0) {
                ckc_strbuf_append(out, ", ");
            }
            emit_operand(out, op->operands[i]);
        }
    }

    emit_attrs(out, &op->attrs);
    emit_types(out, op->results, op->num_results);
    ckc_strbuf_append_char(out, '\n');

    for (int r = 0; r < op->num_regions; ++r) {
        const ckc_region_t *region = op->regions[r];
        emit_pad(out, indent);
        ckc_strbuf_append(out, "  region ");
        emit_repr_str(out, region ? region->label : NULL);
        ckc_strbuf_append(out, " {\n");
        if (region) {
            for (int i = 0; i < region->num_ops; ++i) {
                emit_op(out, region->ops[i], indent + 4);
            }
        }
        emit_pad(out, indent);
        ckc_strbuf_append(out, "  }\n");
    }
}

/* ---------------------------------------------------------------- kernel */

void ckc_print_ir(const ckc_kernel_def_t *kernel, ckc_strbuf_t *out) {
    if (!kernel || !out) {
        return;
    }

    /* Header: "kernel @name(%p0: t0, %p1: t1) {" */
    ckc_strbuf_append(out, "kernel @");
    ckc_strbuf_append(out, kernel->name ? kernel->name : "");
    ckc_strbuf_append_char(out, '(');
    for (int i = 0; i < kernel->num_params; ++i) {
        const ckc_param_t *p = kernel->params[i];
        if (i > 0) {
            ckc_strbuf_append(out, ", ");
        }
        /* Python: f"%{p.name}: {p.type.name}". Param.name has no leading '%'. */
        ckc_strbuf_append_char(out, '%');
        ckc_strbuf_append(out, (p && p->name) ? p->name : "");
        ckc_strbuf_append(out, ": ");
        const ckc_type_t *t = p ? p->type : NULL;
        ckc_strbuf_append(out, (t && t->name) ? t->name : "");
    }
    ckc_strbuf_append(out, ") {\n");

    /* Body ops at indent 2. */
    if (kernel->body) {
        for (int i = 0; i < kernel->body->num_ops; ++i) {
            emit_op(out, kernel->body->ops[i], 2);
        }
    }

    /* Closing brace. Python joins lines with "\n" and the final line is "}" with
     * no trailing newline, so we close without an extra '\n'. */
    ckc_strbuf_append_char(out, '}');
}

char *ckc_print_ir_alloc(const ckc_kernel_def_t *kernel) {
    ckc_strbuf_t sb;
    if (ckc_strbuf_init(&sb, 256) != 0) {
        return NULL;
    }
    ckc_print_ir(kernel, &sb);
    if (sb.oom) {
        ckc_strbuf_free(&sb);
        return NULL;
    }
    char *s = ckc_strbuf_detach(&sb);
    ckc_strbuf_free(&sb);
    if (s) {
        return s;
    }
    /* Empty kernel never happens (header is always emitted), but detach can
     * return NULL for a never-allocated buffer; hand back an empty string. */
    char *empty = (char *)malloc(1);
    if (empty) {
        empty[0] = '\0';
    }
    return empty;
}
