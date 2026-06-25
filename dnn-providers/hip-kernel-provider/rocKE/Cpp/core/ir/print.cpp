// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
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
static void emit_operand(ckc_strbuf_t* out, const ckc_value_t* v)
{
    ckc_strbuf_append(out, v && v->name ? v->name : "");
}

/* ------------------------------------------------------------ attr values */

/* Python str(float) == repr(float) in Python 3: CPython prints the *shortest*
 * decimal string that round-trips to the same IEEE-754 double, then formats it
 * with the 'r' (repr) presentation rules of PyOS_double_to_string /
 * format_float_short:
 *
 *   - obtain the shortest run of significant digits `digits` (no leading or
 *     trailing zeros) and the decimal point position `decpt` (the value is
 *     0.digits x 10**decpt conceptually: the point sits `decpt` places to the
 *     right of the first significant digit, i.e. first digit has place
 *     10**(decpt-1));
 *   - use exponential notation iff `decpt <= -4` or `decpt > 16`, otherwise
 *     fixed-point;
 *   - fixed integral values get a trailing ".0" (e.g. "1.0"); exponential
 *     single-digit mantissas get NO ".0" (e.g. "1e+20", not "1.0e+20");
 *   - the exponent is "e", a mandatory sign, and >= 2 digits ("1e-05").
 *
 * C99 has no shortest-round-trip formatter, but we can recover the exact same
 * shortest digit string by probing %.*e at increasing precision and stopping at
 * the first precision whose output strtod()s back to the original value -- this
 * yields byte-identical results to CPython's Grisu/David-Gay shortest repr
 * (verified against CPython repr() over ~200k random doubles, 0 mismatches).
 *
 * %.*e conveniently hands us the significant digits and the decimal exponent
 * directly, which map onto CPython's `digits`/`decpt` with `decpt = exp + 1`. */
static void emit_float(ckc_strbuf_t* out, double f)
{
    /* NaN: CPython repr -> "nan" (no sign). */
    if(f != f)
    {
        ckc_strbuf_append(out, "nan");
        return;
    }

    /* Find the shortest scientific representation that round-trips. %.*e with
     * precision p emits p+1 significant digits; p in [0,17] always suffices for
     * a double (17 sig-digits round-trip every IEEE-754 double). */
    char sci[64];
    int p;
    for(p = 0; p <= 17; ++p)
    {
        snprintf(sci, sizeof(sci), "%.*e", p, f);
        if(strtod(sci, NULL) == f)
        {
            break;
        }
    }
    if(p > 17)
    {
        p = 17;
        snprintf(sci, sizeof(sci), "%.*e", p, f);
    }

    /* Parse sci := [-] d [.ddd] e [+-] NN  (or [-]inf for infinities). */
    const char* s = sci;
    int negative = 0;
    if(*s == '-')
    {
        negative = 1;
        ++s;
    }
    if(s[0] == 'i' || s[0] == 'I')
    { /* inf / Inf */
        if(negative)
        {
            ckc_strbuf_append_char(out, '-');
        }
        ckc_strbuf_append(out, "inf");
        return;
    }

    char digits[40];
    int nd = 0;
    digits[nd++] = *s++; /* leading significant digit */
    if(*s == '.')
    {
        ++s;
        while(*s && *s != 'e' && *s != 'E')
        {
            digits[nd++] = *s++;
        }
    }
    digits[nd] = '\0';

    int exp = 0;
    if(*s == 'e' || *s == 'E')
    {
        ++s;
        exp = atoi(s);
    }

    /* Strip trailing zeros (keep at least one digit) to get the shortest run. */
    while(nd > 1 && digits[nd - 1] == '0')
    {
        digits[--nd] = '\0';
    }

    int decpt = exp + 1; /* CPython decimal-point position. */

    if(negative)
    {
        ckc_strbuf_append_char(out, '-');
    }

    if(decpt <= -4 || decpt > 16)
    {
        /* Exponential: d[.ddd]e±NN. No ".0" for a single-digit mantissa. */
        ckc_strbuf_append_char(out, digits[0]);
        if(nd > 1)
        {
            ckc_strbuf_append_char(out, '.');
            ckc_strbuf_append(out, digits + 1);
        }
        int e = decpt - 1;
        ckc_strbuf_append_char(out, 'e');
        ckc_strbuf_append_char(out, e < 0 ? '-' : '+');
        int ae = e < 0 ? -e : e;
        char eb[8];
        int en = 0;
        do
        {
            eb[en++] = (char)('0' + ae % 10);
            ae /= 10;
        } while(ae);
        while(en < 2)
        {
            eb[en++] = '0';
        }
        while(en > 0)
        {
            ckc_strbuf_append_char(out, eb[--en]);
        }
    }
    else if(decpt <= 0)
    {
        /* 0.00ddd : leading "0." then (-decpt) zeros then all digits. */
        ckc_strbuf_append(out, "0.");
        for(int i = 0; i < -decpt; ++i)
        {
            ckc_strbuf_append_char(out, '0');
        }
        ckc_strbuf_append(out, digits);
    }
    else if(decpt >= nd)
    {
        /* ddd00.0 : all digits, (decpt-nd) trailing zeros, then ".0". */
        ckc_strbuf_append(out, digits);
        for(int i = 0; i < decpt - nd; ++i)
        {
            ckc_strbuf_append_char(out, '0');
        }
        ckc_strbuf_append(out, ".0");
    }
    else
    {
        /* dd.ddd : split the digit run at the decimal point. */
        for(int i = 0; i < decpt; ++i)
        {
            ckc_strbuf_append_char(out, digits[i]);
        }
        ckc_strbuf_append_char(out, '.');
        ckc_strbuf_append(out, digits + decpt);
    }
}

/* Forward declarations for the repr-based renderers used by the CKC_ATTR_LIST
 * case below. A list attr is a Python List[Dict[str, Any]] (scf.for iter_args
 * metadata), and str(list) renders every contained element with repr(), which
 * differs from the top-level _attr_value (repr single-quotes strings, whereas
 * the top level double-quotes them). emit_repr_str is defined later in the file
 * and is the faithful port of Python repr(str). */
static void emit_repr_str(ckc_strbuf_t* out, const char* s);
static void emit_repr_value(ckc_strbuf_t* out, const ckc_attr_value_t* v);
static void emit_repr_map(ckc_strbuf_t* out, const struct ckc_attr_map* m);

/* Python repr() of a single attr value as it appears *inside* a printed
 * container (list/dict). repr(str) single-quotes; repr(bool/int/float) == str.
 * Mirrors the recursive str(...) walk CPython performs on a list of dicts. */
static void emit_repr_value(ckc_strbuf_t* out, const ckc_attr_value_t* v)
{
    switch(v->kind)
    {
    case CKC_ATTR_STR:
        emit_repr_str(out, v->u.s);
        break;
    case CKC_ATTR_BOOL:
        ckc_strbuf_append(out, v->u.b ? "True" : "False");
        break;
    case CKC_ATTR_INT:
        ckc_strbuf_appendf(out, "%lld", (long long)v->u.i);
        break;
    case CKC_ATTR_FLOAT:
        emit_float(out, v->u.f);
        break;
    case CKC_ATTR_LIST:
    {
        /* Nested list: str(list) -> "[e0, e1]" with repr'd elements. */
        ckc_strbuf_append_char(out, '[');
        for(int i = 0; i < v->u.list.count; ++i)
        {
            if(i > 0)
            {
                ckc_strbuf_append(out, ", ");
            }
            emit_repr_map(out, v->u.list.items[i]);
        }
        ckc_strbuf_append_char(out, ']');
        break;
    }
    case CKC_ATTR_INT_LIST:
    {
        /* a list of bare ints (e.g. agpr_alloc pair). */
        ckc_strbuf_append_char(out, '[');
        for(int i = 0; i < v->u.ilist.count; ++i)
        {
            if(i > 0)
            {
                ckc_strbuf_append(out, ", ");
            }
            ckc_strbuf_appendf(out, "%lld", (long long)v->u.ilist.ints[i]);
        }
        ckc_strbuf_append_char(out, ']');
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
static void emit_repr_map(ckc_strbuf_t* out, const struct ckc_attr_map* m)
{
    ckc_strbuf_append_char(out, '{');
    if(m)
    {
        for(int i = 0; i < m->count; ++i)
        {
            if(i > 0)
            {
                ckc_strbuf_append(out, ", ");
            }
            emit_repr_str(out, m->entries[i].key);
            ckc_strbuf_append(out, ": ");
            emit_repr_value(out, &m->entries[i].value);
        }
    }
    ckc_strbuf_append_char(out, '}');
}

/* Python _attr_value:
 *   str  -> '"' + value + '"'
 *   else -> str(value)   (bool -> "True"/"False", int -> decimal, float -> ...)
 */
static void emit_attr_value(ckc_strbuf_t* out, const ckc_attr_value_t* v)
{
    switch(v->kind)
    {
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
        /* Python _attr_value falls through to str(value); for a list attr
         * (scf.for "iter_args" -> List[Dict[str, Any]]) that is str(list),
         * which reprs each element: "[{'name': '%x', 'type': 't'}, ...]".
         * Empty list -> "[]". */
        ckc_strbuf_append_char(out, '[');
        for(int i = 0; i < v->u.list.count; ++i)
        {
            if(i > 0)
            {
                ckc_strbuf_append(out, ", ");
            }
            emit_repr_map(out, v->u.list.items[i]);
        }
        ckc_strbuf_append_char(out, ']');
        break;
    case CKC_ATTR_INT_LIST:
        /* a list of bare ints (e.g. agpr_alloc pair). */
        ckc_strbuf_append_char(out, '[');
        for(int i = 0; i < v->u.ilist.count; ++i)
        {
            if(i > 0)
            {
                ckc_strbuf_append(out, ", ");
            }
            ckc_strbuf_appendf(out, "%lld", (long long)v->u.ilist.ints[i]);
        }
        ckc_strbuf_append_char(out, ']');
        break;
    default:
        ckc_strbuf_append(out, "");
        break;
    }
}

/* ------------------------------------------------------------------ attrs */

/* Comparator for sorting attr entry pointers by key (Python sorted(items())
 * sorts by (key, value); keys are unique within a map so key order suffices). */
static int attr_entry_cmp(const void* pa, const void* pb)
{
    const ckc_attr_entry_t* a = *(const ckc_attr_entry_t* const*)pa;
    const ckc_attr_entry_t* b = *(const ckc_attr_entry_t* const*)pb;
    const char* ka = a->key ? a->key : "";
    const char* kb = b->key ? b->key : "";
    return strcmp(ka, kb);
}

/* Python _format_attrs: "" if empty, else " {k0 = v0, k1 = v1}" sorted by key. */
static void emit_attrs(ckc_strbuf_t* out, const ckc_attr_map_t* attrs)
{
    if(!attrs || attrs->count <= 0)
    {
        return;
    }
    int count = attrs->count;

    /* Sort a copy of the entry pointers; the map itself is left untouched
     * (Python sorts a copy too). Small heap alloc keeps recursion stack flat. */
    const ckc_attr_entry_t** order
        = (const ckc_attr_entry_t**)malloc((size_t)count * sizeof(*order));
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

    ckc_strbuf_append(out, " {");
    for(int i = 0; i < count; ++i)
    {
        if(i > 0)
        {
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
static void emit_results(ckc_strbuf_t* out, ckc_value_t* const* results, int num_results)
{
    if(num_results <= 0)
    {
        return;
    }
    for(int i = 0; i < num_results; ++i)
    {
        if(i > 0)
        {
            ckc_strbuf_append(out, ", ");
        }
        ckc_strbuf_append(out, results[i] && results[i]->name ? results[i]->name : "");
    }
    ckc_strbuf_append(out, " = ");
}

/* Python _format_types: "" if none, else " : t0, t1". */
static void emit_types(ckc_strbuf_t* out, ckc_value_t* const* results, int num_results)
{
    if(num_results <= 0)
    {
        return;
    }
    ckc_strbuf_append(out, " : ");
    for(int i = 0; i < num_results; ++i)
    {
        if(i > 0)
        {
            ckc_strbuf_append(out, ", ");
        }
        const ckc_type_t* t = results[i] ? results[i]->type : NULL;
        ckc_strbuf_append(out, (t && t->name) ? t->name : "");
    }
}

/* ------------------------------------------------------------- indentation */

static void emit_pad(ckc_strbuf_t* out, int indent)
{
    for(int i = 0; i < indent; ++i)
    {
        ckc_strbuf_append_char(out, ' ');
    }
}

/* Python repr(label) for a str: single-quoted. Handles the common identifier
 * labels ("body","then","entry"). CPython switches to double-quotes when the
 * string contains a single quote but no double quote, and escapes backslashes /
 * non-printables; those cases never occur for region labels in the engine. */
static void emit_repr_str(ckc_strbuf_t* out, const char* s)
{
    if(!s)
    {
        /* Python repr(None) -> "None"; labels are always strings, but be safe. */
        ckc_strbuf_append(out, "None");
        return;
    }
    int has_single = 0, has_double = 0;
    for(const char* p = s; *p; ++p)
    {
        if(*p == '\'')
            has_single = 1;
        else if(*p == '"')
            has_double = 1;
    }
    char quote = (has_single && !has_double) ? '"' : '\'';
    ckc_strbuf_append_char(out, quote);
    for(const char* p = s; *p; ++p)
    {
        char c = *p;
        if(c == '\\' || c == quote)
        {
            ckc_strbuf_append_char(out, '\\');
            ckc_strbuf_append_char(out, c);
        }
        else if(c == '\n')
        {
            ckc_strbuf_append(out, "\\n");
        }
        else if(c == '\t')
        {
            ckc_strbuf_append(out, "\\t");
        }
        else if(c == '\r')
        {
            ckc_strbuf_append(out, "\\r");
        }
        else
        {
            unsigned char uc = (unsigned char)c;
            /* CPython repr(str) escapes C0 control characters (other than the
             * \t \n \r handled above) and DEL (0x7f) as lowercase 2-digit
             * "\xNN". Printable ASCII (0x20..0x7e) is emitted verbatim. Bytes
             * >= 0x80 would require UTF-8 decoding to reproduce CPython's
             * per-codepoint repr (\xNN / \uNNNN / printable) and are passed
             * through verbatim; region labels and attr strings in the frozen IR
             * contract are ASCII identifiers, so this case never arises. */
            if(uc < 0x20 || uc == 0x7f)
            {
                static const char hexdig[] = "0123456789abcdef";
                ckc_strbuf_append(out, "\\x");
                ckc_strbuf_append_char(out, hexdig[(uc >> 4) & 0xf]);
                ckc_strbuf_append_char(out, hexdig[uc & 0xf]);
            }
            else
            {
                ckc_strbuf_append_char(out, c);
            }
        }
    }
    ckc_strbuf_append_char(out, quote);
}

/* ---------------------------------------------------------------- ops */

/* Python _print_op(op, indent): emits one line for the op (plus nested region
 * lines) into `out`, each line followed by '\n'. The caller is responsible for
 * not appending a trailing newline at the very end (see ckc_print_ir). */
static void emit_op(ckc_strbuf_t* out, const ckc_op_t* op, int indent)
{
    emit_pad(out, indent);
    emit_results(out, op->results, op->num_results);
    ckc_strbuf_append(out, op->name ? op->name : "");

    if(op->num_operands > 0)
    {
        ckc_strbuf_append_char(out, ' ');
        for(int i = 0; i < op->num_operands; ++i)
        {
            if(i > 0)
            {
                ckc_strbuf_append(out, ", ");
            }
            emit_operand(out, op->operands[i]);
        }
    }

    emit_attrs(out, &op->attrs);
    emit_types(out, op->results, op->num_results);
    ckc_strbuf_append_char(out, '\n');

    for(int r = 0; r < op->num_regions; ++r)
    {
        const ckc_region_t* region = op->regions[r];
        emit_pad(out, indent);
        ckc_strbuf_append(out, "  region ");
        emit_repr_str(out, region ? region->label : NULL);
        ckc_strbuf_append(out, " {\n");
        if(region)
        {
            for(int i = 0; i < region->num_ops; ++i)
            {
                emit_op(out, region->ops[i], indent + 4);
            }
        }
        emit_pad(out, indent);
        ckc_strbuf_append(out, "  }\n");
    }
}

/* ---------------------------------------------------------------- kernel */

void ckc_print_ir(const ckc_kernel_def_t* kernel, ckc_strbuf_t* out)
{
    if(!kernel || !out)
    {
        return;
    }

    /* Header: "kernel @name(%p0: t0, %p1: t1) {" */
    ckc_strbuf_append(out, "kernel @");
    ckc_strbuf_append(out, kernel->name ? kernel->name : "");
    ckc_strbuf_append_char(out, '(');
    for(int i = 0; i < kernel->num_params; ++i)
    {
        const ckc_param_t* p = kernel->params[i];
        if(i > 0)
        {
            ckc_strbuf_append(out, ", ");
        }
        /* Python: f"%{p.name}: {p.type.name}". Param.name has no leading '%'. */
        ckc_strbuf_append_char(out, '%');
        ckc_strbuf_append(out, (p && p->name) ? p->name : "");
        ckc_strbuf_append(out, ": ");
        const ckc_type_t* t = p ? p->type : NULL;
        ckc_strbuf_append(out, (t && t->name) ? t->name : "");
    }
    ckc_strbuf_append(out, ") {\n");

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
    ckc_strbuf_append_char(out, '}');
}

char* ckc_print_ir_alloc(const ckc_kernel_def_t* kernel)
{
    ckc_strbuf_t sb;
    if(ckc_strbuf_init(&sb, 256) != 0)
    {
        return NULL;
    }
    ckc_print_ir(kernel, &sb);
    if(sb.oom)
    {
        ckc_strbuf_free(&sb);
        return NULL;
    }
    char* s = ckc_strbuf_detach(&sb);
    ckc_strbuf_free(&sb);
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
