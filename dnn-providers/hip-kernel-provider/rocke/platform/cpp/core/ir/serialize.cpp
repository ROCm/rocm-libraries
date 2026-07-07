// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * serialize.c -- `ck.dsl.ir/v1` round-trippable IR serialization.
 *
 * Faithful C99 port of rocke/core/ir_serialize.py. Helper-for-helper map:
 *
 *   Python                       C99 (this file)
 *   --------------------------   -----------------------------------------------
 *   _escape / _unescape          ser_escape() / ser_unescape()
 *   _encode_type / _parse_type   (type uses Type.name directly) / parse_type()
 *   _encode_attr_value           emit_attr_value()
 *   _encode_attr_map             emit_attr_map()
 *   _Scanner / _parse_attr_*     ser_scanner_t / parse_attr_map/parse_attr_value
 *   serialize                    rocke_ir_serialize()
 *   _Lines / parse / _parse_*    line_reader_t / rocke_ir_parse() + helpers
 *   _walk_values_in_def_order    walk_def_order()
 *   canonicalize                 rocke_ir_canonicalize()
 *
 * The serializer reads the (arena-owned) graph and writes into a rocke_strbuf.
 * The parser builds nodes directly in the supplied builder's arena (it needs
 * explicit SSA ids on every result/param/iv/iter-arg, which the public
 * rocke_b_* helpers do not allow -- they mint fresh %vN names). Operands resolve
 * through a value table (name -> rocke_value_t*), exactly as the Python parser.
 */

#include "rocke/ir_serialize.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_status boundary shim */
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/strbuf.h"

#define SER_FORMAT_NAME "ckdsl.ir"
#define SER_FORMAT_VERSION "v1"
#define SER_INDENT "  "

/* ====================================================================== *
 *                       String escaping (spec 1.1)                        *
 * ====================================================================== */

/* Python _escape: \\ -> \\, " -> \", \n -> \n, \t -> \t; every other byte
 * literal. */
static void ser_escape(rocke_strbuf_t* out, const char* s)
{
    if(!s)
    {
        return;
    }
    for(const char* p = s; *p; ++p)
    {
        char ch = *p;
        if(ch == '\\')
        {
            rocke_strbuf_append(out, "\\\\");
        }
        else if(ch == '"')
        {
            rocke_strbuf_append(out, "\\\"");
        }
        else if(ch == '\n')
        {
            rocke_strbuf_append(out, "\\n");
        }
        else if(ch == '\t')
        {
            rocke_strbuf_append(out, "\\t");
        }
        else
        {
            rocke_strbuf_append_char(out, ch);
        }
    }
}

/* Python _unescape: the closed escape set. Writes into an arena-owned buffer
 * (the result string outlives the call). Returns NULL on OOM. */
static char* ser_unescape(rocke_arena_t* arena, const char* s, size_t n)
{
    char* buf = (char*)rocke_arena_alloc(arena, n + 1);
    if(!buf)
    {
        return NULL;
    }
    size_t o = 0;
    size_t i = 0;
    while(i < n)
    {
        char ch = s[i];
        if(ch == '\\' && i + 1 < n)
        {
            char nxt = s[i + 1];
            if(nxt == '\\')
            {
                buf[o++] = '\\';
            }
            else if(nxt == '"')
            {
                buf[o++] = '"';
            }
            else if(nxt == 'n')
            {
                buf[o++] = '\n';
            }
            else if(nxt == 't')
            {
                buf[o++] = '\t';
            }
            else
            {
                buf[o++] = nxt;
            }
            i += 2;
            continue;
        }
        buf[o++] = ch;
        i += 1;
    }
    buf[o] = '\0';
    return buf;
}

/* ====================================================================== *
 *      Shortest-round-trip float formatter (== Python repr(float))        *
 * ====================================================================== *
 *
 * Byte-identical to CPython repr(float): the shortest decimal that round-trips
 * to the same IEEE-754 double, formatted with CPython's fixed/exponential
 * switch rules. This is the exact algorithm in src/core/ir/print.c (emit_float),
 * verified there against ~200k CPython repr() samples; duplicated here so the
 * serializer has no cross-file dependency on the printer. */
static void emit_float_repr(rocke_strbuf_t* out, double f)
{
    if(f != f)
    {
        rocke_strbuf_append(out, "nan");
        return;
    }

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

    const char* s = sci;
    int negative = 0;
    if(*s == '-')
    {
        negative = 1;
        ++s;
    }
    if(s[0] == 'i' || s[0] == 'I')
    {
        if(negative)
        {
            rocke_strbuf_append_char(out, '-');
        }
        rocke_strbuf_append(out, "inf");
        return;
    }

    char digits[40];
    int nd = 0;
    digits[nd++] = *s++;
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

    while(nd > 1 && digits[nd - 1] == '0')
    {
        digits[--nd] = '\0';
    }

    int decpt = exp + 1;

    if(negative)
    {
        rocke_strbuf_append_char(out, '-');
    }

    if(decpt <= -4 || decpt > 16)
    {
        rocke_strbuf_append_char(out, digits[0]);
        if(nd > 1)
        {
            rocke_strbuf_append_char(out, '.');
            rocke_strbuf_append(out, digits + 1);
        }
        int e = decpt - 1;
        rocke_strbuf_append_char(out, 'e');
        rocke_strbuf_append_char(out, e < 0 ? '-' : '+');
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
            rocke_strbuf_append_char(out, eb[--en]);
        }
    }
    else if(decpt <= 0)
    {
        rocke_strbuf_append(out, "0.");
        for(int i = 0; i < -decpt; ++i)
        {
            rocke_strbuf_append_char(out, '0');
        }
        rocke_strbuf_append(out, digits);
    }
    else if(decpt >= nd)
    {
        rocke_strbuf_append(out, digits);
        for(int i = 0; i < decpt - nd; ++i)
        {
            rocke_strbuf_append_char(out, '0');
        }
        rocke_strbuf_append(out, ".0");
    }
    else
    {
        for(int i = 0; i < decpt; ++i)
        {
            rocke_strbuf_append_char(out, digits[i]);
        }
        rocke_strbuf_append_char(out, '.');
        rocke_strbuf_append(out, digits + decpt);
    }
}

/* ====================================================================== *
 *                       Attr encoding (spec 5)                            *
 * ====================================================================== */

static void emit_attr_value(rocke_strbuf_t* out, const rocke_attr_value_t* v);
static void emit_attr_map(rocke_strbuf_t* out, const rocke_attr_map_t* m);

/* Comparator: sort attr entry pointers ascending by key (Unicode code point ==
 * byte value for the ASCII identifier keys the format uses). */
static int ser_attr_cmp(const void* pa, const void* pb)
{
    const rocke_attr_entry_t* a = *(const rocke_attr_entry_t* const*)pa;
    const rocke_attr_entry_t* b = *(const rocke_attr_entry_t* const*)pb;
    const char* ka = a->key ? a->key : "";
    const char* kb = b->key ? b->key : "";
    return strcmp(ka, kb);
}

/* Python _encode_attr_value:
 *   bool  -> "b:true"/"b:false"   (checked before int)
 *   int   -> "i:<dec>"
 *   float -> "f:<repr>"
 *   str   -> 's:"<escaped>"'
 *   dict  -> {...}                 (list element -> nested attr map)
 *   list  -> "l:[ e0, e1 ]"
 */
static void emit_attr_value(rocke_strbuf_t* out, const rocke_attr_value_t* v)
{
    switch(v->kind)
    {
    case ROCKE_ATTR_BOOL:
        rocke_strbuf_append(out, v->u.b ? "b:true" : "b:false");
        break;
    case ROCKE_ATTR_INT:
        rocke_strbuf_appendf(out, "i:%lld", (long long)v->u.i);
        break;
    case ROCKE_ATTR_FLOAT:
        rocke_strbuf_append(out, "f:");
        emit_float_repr(out, v->u.f);
        break;
    case ROCKE_ATTR_STR:
        rocke_strbuf_append(out, "s:\"");
        ser_escape(out, v->u.s);
        rocke_strbuf_append_char(out, '"');
        break;
    case ROCKE_ATTR_LIST:
        rocke_strbuf_append(out, "l:[ ");
        for(int i = 0; i < v->u.list.count; ++i)
        {
            if(i > 0)
            {
                rocke_strbuf_append(out, ", ");
            }
            /* a list element is itself an attr map (the iter_args dict). */
            emit_attr_map(out, v->u.list.items[i]);
        }
        rocke_strbuf_append(out, " ]");
        break;
    case ROCKE_ATTR_INT_LIST:
        /* a list whose elements are bare ints, e.g. agpr_alloc (0,0). */
        rocke_strbuf_append(out, "l:[ ");
        for(int i = 0; i < v->u.ilist.count; ++i)
        {
            if(i > 0)
            {
                rocke_strbuf_append(out, ", ");
            }
            rocke_strbuf_appendf(out, "i:%lld", (long long)v->u.ilist.ints[i]);
        }
        rocke_strbuf_append(out, " ]");
        break;
    default:
        break;
    }
}

/* Python _encode_attr_map: "{ }" if empty, else "{ k = v, ... }" sorted by
 * key. */
static void emit_attr_map(rocke_strbuf_t* out, const rocke_attr_map_t* m)
{
    int count = m ? m->count : 0;
    if(count <= 0)
    {
        rocke_strbuf_append(out, "{ }");
        return;
    }
    const rocke_attr_entry_t** order
        = (const rocke_attr_entry_t**)malloc((size_t)count * sizeof(*order));
    if(!order)
    {
        out->oom = 1;
        return;
    }
    for(int i = 0; i < count; ++i)
    {
        order[i] = &m->entries[i];
    }
    qsort(order, (size_t)count, sizeof(*order), ser_attr_cmp);

    rocke_strbuf_append(out, "{ ");
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
    rocke_strbuf_append(out, " }");
    free(order);
}

/* ====================================================================== *
 *                         Serialize (spec 3, 4)                           *
 * ====================================================================== */

static void serialize_region(rocke_strbuf_t* out, const rocke_region_t* region, int depth);

static void emit_indent(rocke_strbuf_t* out, int depth)
{
    for(int i = 0; i < depth; ++i)
    {
        rocke_strbuf_append(out, SER_INDENT);
    }
}

/* Python _serialize_op. */
static void serialize_op(rocke_strbuf_t* out, const rocke_op_t* op, int depth)
{
    emit_indent(out, depth);
    if(op->num_results > 0)
    {
        for(int i = 0; i < op->num_results; ++i)
        {
            if(i > 0)
            {
                rocke_strbuf_append(out, ", ");
            }
            const rocke_value_t* r = op->results[i];
            rocke_strbuf_append(out, (r && r->name) ? r->name : "");
            rocke_strbuf_append(out, " : ");
            const rocke_type_t* t = r ? r->type : NULL;
            rocke_strbuf_append(out, (t && t->name) ? t->name : "");
        }
        rocke_strbuf_append(out, " = ");
    }
    rocke_strbuf_append(out, op->name ? op->name : "");
    rocke_strbuf_append(out, " ( ");
    for(int i = 0; i < op->num_operands; ++i)
    {
        if(i > 0)
        {
            rocke_strbuf_append(out, ", ");
        }
        const rocke_value_t* o = op->operands[i];
        rocke_strbuf_append(out, (o && o->name) ? o->name : "");
    }
    rocke_strbuf_append(out, " )");
    if(op->attrs.count > 0)
    {
        rocke_strbuf_append_char(out, ' ');
        emit_attr_map(out, &op->attrs);
    }
    if(op->loc != NULL)
    {
        rocke_strbuf_append(out, " @loc \"");
        ser_escape(out, op->loc);
        rocke_strbuf_append_char(out, '"');
    }
    rocke_strbuf_append_char(out, '\n');

    for(int r = 0; r < op->num_regions; ++r)
    {
        serialize_region(out, op->regions[r], depth + 1);
    }
}

/* Python _serialize_region. */
static void serialize_region(rocke_strbuf_t* out, const rocke_region_t* region, int depth)
{
    emit_indent(out, depth);
    rocke_strbuf_append(out, "region @");
    rocke_strbuf_append(out, (region && region->label) ? region->label : "");
    rocke_strbuf_append(out, " {\n");
    if(region)
    {
        for(int i = 0; i < region->num_ops; ++i)
        {
            serialize_op(out, region->ops[i], depth + 1);
        }
    }
    emit_indent(out, depth);
    rocke_strbuf_append(out, "}\n");
}

rocke_status_t rocke_ir_serialize(const rocke_kernel_def_t* k, char** out_text)
{
    if(out_text)
    {
        *out_text = NULL;
    }
    if(!k || !out_text)
    {
        return ROCKE_ERR_VALUE;
    }

    rocke_strbuf_t sb;
    if(rocke_strbuf_init(&sb, 1024) != 0)
    {
        return ROCKE_ERR_OOM;
    }

    rocke_strbuf_append(&sb, SER_FORMAT_NAME " " SER_FORMAT_VERSION "\n");
    rocke_strbuf_append(&sb, "kernel @");
    rocke_strbuf_append(&sb, k->name ? k->name : "");
    rocke_strbuf_append(&sb, " {\n");

    if(k->attrs.count > 0)
    {
        emit_indent(&sb, 1);
        rocke_strbuf_append(&sb, "attrs ");
        emit_attr_map(&sb, &k->attrs);
        rocke_strbuf_append_char(&sb, '\n');
    }

    emit_indent(&sb, 1);
    rocke_strbuf_append(&sb, "params {\n");
    for(int i = 0; i < k->num_params; ++i)
    {
        const rocke_param_t* p = k->params[i];
        emit_indent(&sb, 2);
        rocke_strbuf_append_char(&sb, '%');
        rocke_strbuf_append(&sb, (p && p->name) ? p->name : "");
        rocke_strbuf_append(&sb, " : ");
        const rocke_type_t* t = p ? p->type : NULL;
        rocke_strbuf_append(&sb, (t && t->name) ? t->name : "");
        if(p && p->attrs.count > 0)
        {
            rocke_strbuf_append_char(&sb, ' ');
            emit_attr_map(&sb, &p->attrs);
        }
        rocke_strbuf_append_char(&sb, '\n');
    }
    emit_indent(&sb, 1);
    rocke_strbuf_append(&sb, "}\n");

    serialize_region(&sb, k->body, 1);
    rocke_strbuf_append(&sb, "}\n");

    if(sb.oom)
    {
        rocke_strbuf_free(&sb);
        return ROCKE_ERR_OOM;
    }
    char* s = rocke_strbuf_detach(&sb);
    rocke_strbuf_free(&sb);
    if(!s)
    {
        return ROCKE_ERR_OOM;
    }
    *out_text = s;
    return ROCKE_OK;
}

/* ====================================================================== *
 *                  Type parsing (spec 2; inverse of Type.name)           *
 * ====================================================================== */

static const rocke_type_t* parse_type(rocke_ir_builder_t* b, const char* s, size_t n);

/* Find the rightmost 'x' in [0,n) of `s` that is followed only by digits up to
 * `n`. Returns the index, or -1. Mirrors the Python rfind-loop in _parse_type
 * for vector types. */
static long find_vec_split(const char* s, size_t start, size_t n)
{
    /* scan all 'x' positions; pick the rightmost whose suffix is all digits. */
    long best = -1;
    for(size_t i = start; i < n; ++i)
    {
        if(s[i] != 'x')
        {
            continue;
        }
        if(i + 1 >= n)
        {
            continue;
        }
        int all_digit = 1;
        for(size_t j = i + 1; j < n; ++j)
        {
            if(s[j] < '0' || s[j] > '9')
            {
                all_digit = 0;
                break;
            }
        }
        if(all_digit)
        {
            best = (long)i;
        }
    }
    return best;
}

static int str_has_prefix(const char* s, size_t n, const char* pre)
{
    size_t pl = strlen(pre);
    return n >= pl && memcmp(s, pre, pl) == 0;
}

/* Parse a canonical type string s[0,n). Reconstructs scalar singletons by name,
 * or arena-allocates a vec/ptr/smem composite via the public constructors. */
static const rocke_type_t* parse_type(rocke_ir_builder_t* b, const char* s, size_t n)
{
    /* strip surrounding whitespace */
    while(n > 0 && (s[0] == ' ' || s[0] == '\t'))
    {
        ++s;
        --n;
    }
    while(n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t'))
    {
        --n;
    }

    if(str_has_prefix(s, n, "vec<") && n > 0 && s[n - 1] == '>')
    {
        const char* inner = s + 4;
        size_t in = n - 5; /* between "vec<" and ">" */
        long idx = find_vec_split(inner, 0, in);
        if(idx < 0)
        {
            return (const rocke_type_t*)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "malformed vector type");
        }
        const rocke_type_t* elem = parse_type(b, inner, (size_t)idx);
        if(!elem)
        {
            return NULL;
        }
        int count = atoi(inner + idx + 1);
        return rocke_vector_type(b, elem, count);
    }
    if(str_has_prefix(s, n, "ptr<") && n > 0 && s[n - 1] == '>')
    {
        const char* inner = s + 4;
        size_t in = n - 5;
        /* pointee may contain commas (vec<...>): split at the LAST comma. */
        long comma = -1;
        for(size_t i = 0; i < in; ++i)
        {
            if(inner[i] == ',')
            {
                comma = (long)i;
            }
        }
        if(comma < 0)
        {
            return (const rocke_type_t*)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "malformed pointer type");
        }
        const rocke_type_t* pointee = parse_type(b, inner, (size_t)comma);
        if(!pointee)
        {
            return NULL;
        }
        const char* sp = inner + comma + 1;
        size_t spn = in - (size_t)comma - 1;
        while(spn > 0 && (sp[0] == ' ' || sp[0] == '\t'))
        {
            ++sp;
            --spn;
        }
        while(spn > 0 && (sp[spn - 1] == ' ' || sp[spn - 1] == '\t'))
        {
            --spn;
        }
        char* space = (char*)rocke_arena_alloc(&b->arena, spn + 1);
        if(!space)
        {
            return (const rocke_type_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "ptr type OOM");
        }
        memcpy(space, sp, spn);
        space[spn] = '\0';
        return rocke_ptr_type(b, pointee, space);
    }
    if(str_has_prefix(s, n, "smem<") && n > 0 && s[n - 1] == '>')
    {
        const char* inner = s + 5;
        size_t in = n - 6;
        /* form: "<elem>, [d0xd1x...]" */
        long lb = -1, rb = -1;
        for(size_t i = 0; i < in; ++i)
        {
            if(inner[i] == '[')
            {
                lb = (long)i;
            }
            if(inner[i] == ']')
            {
                rb = (long)i;
            }
        }
        if(lb < 0 || rb < 0)
        {
            return (const rocke_type_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "malformed smem type");
        }
        /* elem head = everything before '[', rstripped, drop trailing comma. */
        size_t hn = (size_t)lb;
        while(hn > 0 && (inner[hn - 1] == ' ' || inner[hn - 1] == '\t'))
        {
            --hn;
        }
        if(hn > 0 && inner[hn - 1] == ',')
        {
            --hn;
            while(hn > 0 && (inner[hn - 1] == ' ' || inner[hn - 1] == '\t'))
            {
                --hn;
            }
        }
        const rocke_type_t* elem = parse_type(b, inner, hn);
        if(!elem)
        {
            return NULL;
        }
        /* shape between '[' and ']': "d0xd1x..." */
        const char* sh = inner + lb + 1;
        size_t shn = (size_t)(rb - lb - 1);
        while(shn > 0 && (sh[0] == ' ' || sh[0] == '\t'))
        {
            ++sh;
            --shn;
        }
        while(shn > 0 && (sh[shn - 1] == ' ' || sh[shn - 1] == '\t'))
        {
            --shn;
        }
        int shape[16];
        int rank = 0;
        if(shn > 0)
        {
            size_t i = 0;
            while(i < shn && rank < 16)
            {
                int val = 0;
                int any = 0;
                while(i < shn && sh[i] >= '0' && sh[i] <= '9')
                {
                    val = val * 10 + (sh[i] - '0');
                    ++i;
                    any = 1;
                }
                if(any)
                {
                    shape[rank++] = val;
                }
                if(i < shn && sh[i] == 'x')
                {
                    ++i;
                }
            }
        }
        return rocke_smem_type(b, elem, shape, rank);
    }

    /* scalar: reconstruct the singleton by name. */
    char nm[64];
    if(n >= sizeof(nm))
    {
        return (const rocke_type_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "type name too long");
    }
    memcpy(nm, s, n);
    nm[n] = '\0';
    const rocke_type_t* sc = rocke_scalar_by_name(nm);
    if(sc)
    {
        return sc;
    }
    /* Unknown scalar name: build a bare arena Type with that name (mirrors
     * Python Type(s) fallback). Reuse vector_type-style construction. */
    {
        rocke_type_t* t = (rocke_type_t*)rocke_arena_calloc(&b->arena, sizeof(*t));
        if(!t)
        {
            return (const rocke_type_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "type OOM");
        }
        t->kind = ROCKE_TYPE_SCALAR;
        t->scalar = ROCKE_SCALAR__COUNT; /* unknown */
        t->name = rocke_arena_strdup(&b->arena, nm);
        if(!t->name)
        {
            return (const rocke_type_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "type OOM");
        }
        return t;
    }
}

/* ====================================================================== *
 *                Attr-map parsing (recursive-descent scanner)            *
 * ====================================================================== */

typedef struct ser_scanner
{
    const char* text;
    size_t i;
    size_t n;
    rocke_ir_builder_t* b;
} ser_scanner_t;

static void sc_skip_ws(ser_scanner_t* sc)
{
    while(sc->i < sc->n && (sc->text[sc->i] == ' ' || sc->text[sc->i] == '\t'))
    {
        ++sc->i;
    }
}

static char sc_peek(ser_scanner_t* sc)
{
    return sc->i < sc->n ? sc->text[sc->i] : '\0';
}

static int sc_match2(ser_scanner_t* sc, const char* two)
{
    return sc->i + 2 <= sc->n && sc->text[sc->i] == two[0] && sc->text[sc->i + 1] == two[1];
}

/* Forward */
static int parse_attr_value(ser_scanner_t* sc, rocke_attr_value_t* out);

/* Parse an attr map into `m` (already init'd in the arena). Returns 0 / -1. */
static int parse_attr_map(ser_scanner_t* sc, rocke_attr_map_t* m)
{
    rocke_ir_builder_t* b = sc->b;
    sc_skip_ws(sc);
    if(sc_peek(sc) != '{')
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "attr map: expected '{'");
        return -1;
    }
    ++sc->i;
    sc_skip_ws(sc);
    if(sc_peek(sc) == '}')
    {
        ++sc->i;
        return 0;
    }
    for(;;)
    {
        sc_skip_ws(sc);
        /* key: identifier */
        size_t start = sc->i;
        while(sc->i < sc->n)
        {
            char c = sc->text[sc->i];
            if((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
               || c == '_')
            {
                ++sc->i;
            }
            else
            {
                break;
            }
        }
        size_t klen = sc->i - start;
        if(klen == 0)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "attr map: empty key");
            return -1;
        }
        char keybuf[128];
        if(klen >= sizeof(keybuf))
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "attr key too long");
            return -1;
        }
        memcpy(keybuf, sc->text + start, klen);
        keybuf[klen] = '\0';
        sc_skip_ws(sc);
        if(sc_peek(sc) != '=')
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "attr map: expected '='");
            return -1;
        }
        ++sc->i;
        rocke_attr_value_t val;
        memset(&val, 0, sizeof(val));
        if(parse_attr_value(sc, &val) != 0)
        {
            return -1;
        }
        /* store into m by kind via the public setters */
        switch(val.kind)
        {
        case ROCKE_ATTR_INT:
            rocke_attr_set_int(b, m, keybuf, val.u.i);
            break;
        case ROCKE_ATTR_FLOAT:
            rocke_attr_set_float(b, m, keybuf, val.u.f);
            break;
        case ROCKE_ATTR_STR:
            rocke_attr_set_str(b, m, keybuf, val.u.s);
            break;
        case ROCKE_ATTR_BOOL:
            rocke_attr_set_bool(b, m, keybuf, val.u.b);
            break;
        case ROCKE_ATTR_INT_LIST:
        case ROCKE_ATTR_LIST:
        {
            /* Append a list-valued entry directly (no public setter). */
            if(m->count >= m->cap)
            {
                int nc = m->cap ? m->cap * 2 : 4;
                rocke_attr_entry_t* ne = (rocke_attr_entry_t*)rocke_arena_alloc(
                    &b->arena, sizeof(rocke_attr_entry_t) * (size_t)nc);
                if(!ne)
                {
                    rocke_i_set_err(b, ROCKE_ERR_OOM, "attr list OOM");
                    return -1;
                }
                if(m->entries && m->count)
                {
                    memcpy(ne, m->entries, sizeof(rocke_attr_entry_t) * (size_t)m->count);
                }
                m->entries = ne;
                m->cap = nc;
            }
            m->entries[m->count].key = rocke_arena_strdup(&b->arena, keybuf);
            m->entries[m->count].value = val;
            m->count++;
            break;
        }
        default:
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "attr map: bad value kind");
            return -1;
        }
        if(!rocke_i_live(b))
        {
            return -1;
        }
        sc_skip_ws(sc);
        char ch = sc_peek(sc);
        if(ch == ',')
        {
            ++sc->i;
            continue;
        }
        if(ch == '}')
        {
            ++sc->i;
            break;
        }
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "attr map: expected ',' or '}'");
        return -1;
    }
    return 0;
}

/* Read a bare scalar token (int/float payload) up to a delimiter. */
static size_t sc_read_scalar(ser_scanner_t* sc, char* buf, size_t cap)
{
    size_t o = 0;
    while(sc->i < sc->n)
    {
        char c = sc->text[sc->i];
        if(c == ',' || c == ' ' || c == '}' || c == ']' || c == '\t')
        {
            break;
        }
        if(o + 1 < cap)
        {
            buf[o++] = c;
        }
        ++sc->i;
    }
    buf[o] = '\0';
    return o;
}

/* Read a quoted string body (sc positioned just after the opening quote);
 * unescape into an arena buffer; returns NULL on error. */
static const char* sc_read_quoted(ser_scanner_t* sc)
{
    rocke_ir_builder_t* b = sc->b;
    size_t start = sc->i;
    /* find the closing quote, honoring backslash escapes (skip the next char) */
    while(sc->i < sc->n)
    {
        char ch = sc->text[sc->i];
        if(ch == '\\' && sc->i + 1 < sc->n)
        {
            sc->i += 2;
            continue;
        }
        if(ch == '"')
        {
            break;
        }
        ++sc->i;
    }
    if(sc->i >= sc->n)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "unterminated string literal");
        return NULL;
    }
    size_t len = sc->i - start;
    char* res = ser_unescape(&b->arena, sc->text + start, len);
    ++sc->i; /* consume closing quote */
    if(!res)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "string OOM");
        return NULL;
    }
    return res;
}

static int parse_attr_value(ser_scanner_t* sc, rocke_attr_value_t* out)
{
    rocke_ir_builder_t* b = sc->b;
    sc_skip_ws(sc);
    char ch = sc_peek(sc);
    if(ch == '{')
    {
        /* a nested attr-map (list element dict). Store as a 1-element... no:
         * the Python form for an attr that IS a dict is rare; but list elements
         * are dicts. We expose it as an attr map wrapped in a list-item; the
         * caller (list path) handles it. Here we materialise a ROCKE_ATTR_LIST
         * with a single map item is NOT correct -- instead, a bare dict value
         * should never appear at top level in v1; only inside l:[...]. We thus
         * parse it into a fresh map and signal via a special kind. */
        rocke_attr_map_t* child = (rocke_attr_map_t*)rocke_arena_calloc(&b->arena, sizeof(*child));
        if(!child)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "attr dict OOM");
            return -1;
        }
        rocke_attr_map_init(child);
        if(parse_attr_map(sc, child) != 0)
        {
            return -1;
        }
        /* Represent a dict as a single-item list carrying the map, so the list
         * path can splice it. But the Python format only nests dicts inside
         * lists; emit_attr_value handles ROCKE_ATTR_LIST element as a map. We
         * encode this dict as a list with count==-1 marker is ugly. Instead we
         * use a dedicated path: store the map pointer in a 1-cap list and the
         * caller in the list loop will pull items[0]. To keep things simple we
         * wrap: kind=LIST, count = -(1) is invalid. Use a cleaner scheme: */
        out->kind = ROCKE_ATTR_LIST;
        out->u.list.items
            = (rocke_attr_map_t**)rocke_arena_alloc(&b->arena, sizeof(rocke_attr_map_t*));
        if(!out->u.list.items)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "attr dict OOM");
            return -1;
        }
        out->u.list.items[0] = child;
        out->u.list.count = -1; /* marker: this LIST is actually a bare dict */
        return 0;
    }
    if(sc_match2(sc, "l:"))
    {
        sc->i += 2;
        sc_skip_ws(sc);
        if(sc_peek(sc) != '[')
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "list: expected '['");
            return -1;
        }
        ++sc->i;
        /* gather child maps (each list element is a dict) */
        rocke_attr_map_t** items = NULL;
        int count = 0, cap = 0;
        /* a list of bare ints (e.g. agpr_alloc l:[ i:0, i:0 ]). */
        int64_t* ints = NULL;
        int icount = 0, icap = 0;
        sc_skip_ws(sc);
        if(sc_peek(sc) == ']')
        {
            ++sc->i;
            out->kind = ROCKE_ATTR_LIST;
            out->u.list.items = NULL;
            out->u.list.count = 0;
            return 0;
        }
        for(;;)
        {
            rocke_attr_value_t elem;
            memset(&elem, 0, sizeof(elem));
            if(parse_attr_value(sc, &elem) != 0)
            {
                return -1;
            }
            /* bare-int element: accumulate into an int list. The element kinds
             * within a v1 list are homogeneous (all dicts, or all bare ints). */
            if(elem.kind == ROCKE_ATTR_INT)
            {
                if(count > 0)
                {
                    rocke_i_set_err(b, ROCKE_ERR_VALUE, "list: mixed int/dict elements");
                    return -1;
                }
                if(icount >= icap)
                {
                    int nc = icap ? icap * 2 : 4;
                    int64_t* ni
                        = (int64_t*)rocke_arena_alloc(&b->arena, sizeof(int64_t) * (size_t)nc);
                    if(!ni)
                    {
                        rocke_i_set_err(b, ROCKE_ERR_OOM, "list OOM");
                        return -1;
                    }
                    if(ints && icount)
                        memcpy(ni, ints, sizeof(int64_t) * (size_t)icount);
                    ints = ni;
                    icap = nc;
                }
                ints[icount++] = elem.u.i;
                sc_skip_ws(sc);
                char ic = sc_peek(sc);
                if(ic == ',')
                {
                    ++sc->i;
                    continue;
                }
                if(ic == ']')
                {
                    ++sc->i;
                    out->kind = ROCKE_ATTR_INT_LIST;
                    out->u.ilist.ints = ints;
                    out->u.ilist.count = icount;
                    return 0;
                }
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "list: expected ',' or ']'");
                return -1;
            }
            if(icount > 0)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "list: mixed int/dict elements");
                return -1;
            }
            /* each element is a dict, encoded as the bare-dict marker above. */
            rocke_attr_map_t* child;
            if(elem.kind == ROCKE_ATTR_LIST && elem.u.list.count == -1)
            {
                child = elem.u.list.items[0];
            }
            else if(elem.kind == ROCKE_ATTR_LIST)
            {
                /* nested list -- not used by v1 iter_args, but be tolerant. */
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "list: nested non-dict element");
                return -1;
            }
            else
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "list: element must be a dict");
                return -1;
            }
            if(count >= cap)
            {
                int nc = cap ? cap * 2 : 4;
                rocke_attr_map_t** ni = (rocke_attr_map_t**)rocke_arena_alloc(
                    &b->arena, sizeof(rocke_attr_map_t*) * (size_t)nc);
                if(!ni)
                {
                    rocke_i_set_err(b, ROCKE_ERR_OOM, "list OOM");
                    return -1;
                }
                if(items && count)
                {
                    memcpy(ni, items, sizeof(rocke_attr_map_t*) * (size_t)count);
                }
                items = ni;
                cap = nc;
            }
            items[count++] = child;
            sc_skip_ws(sc);
            char c = sc_peek(sc);
            if(c == ',')
            {
                ++sc->i;
                continue;
            }
            if(c == ']')
            {
                ++sc->i;
                break;
            }
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "list: expected ',' or ']'");
            return -1;
        }
        out->kind = ROCKE_ATTR_LIST;
        out->u.list.items = items;
        out->u.list.count = count;
        return 0;
    }
    if(sc_match2(sc, "b:"))
    {
        sc->i += 2;
        if(sc->i + 4 <= sc->n && memcmp(sc->text + sc->i, "true", 4) == 0)
        {
            sc->i += 4;
            out->kind = ROCKE_ATTR_BOOL;
            out->u.b = true;
            return 0;
        }
        if(sc->i + 5 <= sc->n && memcmp(sc->text + sc->i, "false", 5) == 0)
        {
            sc->i += 5;
            out->kind = ROCKE_ATTR_BOOL;
            out->u.b = false;
            return 0;
        }
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "malformed bool");
        return -1;
    }
    if(sc_match2(sc, "i:"))
    {
        sc->i += 2;
        char tok[64];
        sc_read_scalar(sc, tok, sizeof(tok));
        out->kind = ROCKE_ATTR_INT;
        out->u.i = (int64_t)strtoll(tok, NULL, 10);
        return 0;
    }
    if(sc_match2(sc, "f:"))
    {
        sc->i += 2;
        char tok[64];
        sc_read_scalar(sc, tok, sizeof(tok));
        out->kind = ROCKE_ATTR_FLOAT;
        out->u.f = strtod(tok, NULL);
        return 0;
    }
    if(sc_match2(sc, "s:"))
    {
        sc->i += 2;
        if(sc_peek(sc) != '"')
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "string: expected '\"'");
            return -1;
        }
        ++sc->i;
        const char* str = sc_read_quoted(sc);
        if(!str)
        {
            return -1;
        }
        out->kind = ROCKE_ATTR_STR;
        out->u.s = str;
        return 0;
    }
    rocke_i_set_err(b, ROCKE_ERR_VALUE, "unknown attr value tag");
    return -1;
}

/* ====================================================================== *
 *                      Line reader (Python _Lines)                       *
 * ====================================================================== */

typedef struct line_reader
{
    char* buf; /* arena copy of the text, '\n'-split in place */
    char** lines; /* arena array of trimmed line pointers        */
    int count;
    int idx;
} line_reader_t;

/* Left/right strip in place: returns a pointer into the same buffer with the
 * trailing whitespace NUL-terminated. */
static char* strip_line(char* s)
{
    while(*s == ' ' || *s == '\t' || *s == '\r')
    {
        ++s;
    }
    size_t n = strlen(s);
    while(n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\r'))
    {
        s[--n] = '\0';
    }
    return s;
}

static int lr_init(rocke_ir_builder_t* b, line_reader_t* lr, const char* text)
{
    size_t tn = strlen(text);
    lr->buf = (char*)rocke_arena_alloc(&b->arena, tn + 1);
    if(!lr->buf)
    {
        return -1;
    }
    memcpy(lr->buf, text, tn + 1);
    /* count lines */
    int nlines = 1;
    for(size_t i = 0; i < tn; ++i)
    {
        if(lr->buf[i] == '\n')
        {
            ++nlines;
        }
    }
    lr->lines = (char**)rocke_arena_alloc(&b->arena, sizeof(char*) * (size_t)nlines);
    if(!lr->lines)
    {
        return -1;
    }
    lr->count = 0;
    char* start = lr->buf;
    for(size_t i = 0; i <= tn; ++i)
    {
        if(lr->buf[i] == '\n' || lr->buf[i] == '\0')
        {
            lr->buf[i] = '\0';
            lr->lines[lr->count++] = start;
            start = lr->buf + i + 1;
        }
    }
    lr->idx = 0;
    return 0;
}

static int lr_is_skippable(const char* ln)
{
    /* stripped == "" or starts with '#'. We strip a copy-view: skip leading ws */
    const char* p = ln;
    while(*p == ' ' || *p == '\t' || *p == '\r')
    {
        ++p;
    }
    return *p == '\0' || *p == '#';
}

/* Peek the next non-skippable line (stripped). Returns NULL at EOF. */
static char* lr_peek(line_reader_t* lr)
{
    while(lr->idx < lr->count && lr_is_skippable(lr->lines[lr->idx]))
    {
        ++lr->idx;
    }
    if(lr->idx >= lr->count)
    {
        return NULL;
    }
    return strip_line(lr->lines[lr->idx]);
}

static char* lr_next(line_reader_t* lr)
{
    char* ln = lr_peek(lr);
    if(ln)
    {
        ++lr->idx;
    }
    return ln;
}

/* ====================================================================== *
 *                       Value table (name -> Value)                      *
 * ====================================================================== */

typedef struct val_entry
{
    const char* name;
    rocke_value_t* value;
} val_entry_t;

typedef struct val_table
{
    val_entry_t* entries;
    int count;
    int cap;
    rocke_ir_builder_t* b;
} val_table_t;

static rocke_value_t* vt_get(val_table_t* vt, const char* name)
{
    for(int i = 0; i < vt->count; ++i)
    {
        if(strcmp(vt->entries[i].name, name) == 0)
        {
            return vt->entries[i].value;
        }
    }
    return NULL;
}

static int vt_put(val_table_t* vt, const char* name, rocke_value_t* v)
{
    /* overwrite if present (later defs shadow within block scope handling). */
    for(int i = 0; i < vt->count; ++i)
    {
        if(strcmp(vt->entries[i].name, name) == 0)
        {
            vt->entries[i].value = v;
            return 0;
        }
    }
    if(vt->count >= vt->cap)
    {
        int nc = vt->cap ? vt->cap * 2 : 32;
        val_entry_t* ne
            = (val_entry_t*)rocke_arena_alloc(&vt->b->arena, sizeof(val_entry_t) * (size_t)nc);
        if(!ne)
        {
            return -1;
        }
        if(vt->entries && vt->count)
        {
            memcpy(ne, vt->entries, sizeof(val_entry_t) * (size_t)vt->count);
        }
        vt->entries = ne;
        vt->cap = nc;
    }
    vt->entries[vt->count].name = name;
    vt->entries[vt->count].value = v;
    vt->count++;
    return 0;
}

static void vt_remove(val_table_t* vt, const char* name)
{
    for(int i = 0; i < vt->count; ++i)
    {
        if(strcmp(vt->entries[i].name, name) == 0)
        {
            for(int j = i; j + 1 < vt->count; ++j)
            {
                vt->entries[j] = vt->entries[j + 1];
            }
            vt->count--;
            return;
        }
    }
}

/* ====================================================================== *
 *                       Parse (spec 3, 4)                                 *
 * ====================================================================== */

/* split a result/operand string on top-level `sep`, ignoring <>[]{}"" nesting.
 * Writes start/len pairs into out arrays (caller supplies capacity). Returns
 * the number of fields. */
static int
    split_top(const char* s, size_t n, char sep, const char** starts, size_t* lens, int max_fields)
{
    int cnt = 0;
    int depth = 0;
    int in_str = 0;
    size_t start = 0;
    size_t i = 0;
    while(i < n)
    {
        char ch = s[i];
        if(in_str)
        {
            if(ch == '\\')
            {
                i += 2;
                continue;
            }
            if(ch == '"')
            {
                in_str = 0;
            }
        }
        else
        {
            if(ch == '"')
            {
                in_str = 1;
            }
            else if(ch == '<' || ch == '[' || ch == '{' || ch == '(')
            {
                ++depth;
            }
            else if(ch == '>' || ch == ']' || ch == '}' || ch == ')')
            {
                --depth;
            }
            else if(ch == sep && depth == 0)
            {
                if(cnt < max_fields)
                {
                    starts[cnt] = s + start;
                    lens[cnt] = i - start;
                    ++cnt;
                }
                start = i + 1;
            }
        }
        ++i;
    }
    if(cnt < max_fields)
    {
        starts[cnt] = s + start;
        lens[cnt] = n - start;
        ++cnt;
    }
    return cnt;
}

/* Count top-level fields (separated by `sep` at depth 0, ignoring quoted strings
 * and bracketed groups), mirroring split_top's scan. Used to size result/operand
 * buffers to the actual arity instead of a fixed cap. Returns >= 1. */
static int count_top(const char* s, size_t n, char sep)
{
    int cnt = 1;
    int depth = 0;
    int in_str = 0;
    size_t i = 0;
    while(i < n)
    {
        char ch = s[i];
        if(in_str)
        {
            if(ch == '\\')
            {
                i += 2;
                continue;
            }
            if(ch == '"')
            {
                in_str = 0;
            }
        }
        else
        {
            if(ch == '"')
            {
                in_str = 1;
            }
            else if(ch == '<' || ch == '[' || ch == '{' || ch == '(')
            {
                ++depth;
            }
            else if(ch == '>' || ch == ']' || ch == '}' || ch == ')')
            {
                --depth;
            }
            else if(ch == sep && depth == 0)
            {
                ++cnt;
            }
        }
        ++i;
    }
    return cnt;
}

static rocke_region_t* parse_region(rocke_ir_builder_t* b, line_reader_t* lr, val_table_t* vt);

/* Register block-defined SSA values for an scf.for op (iv + iter-args), so the
 * body operands resolve. Mirrors Python _register_block_values. Returns the
 * names added via out_added (arena array) / out_n. */
static void register_block_values(
    rocke_ir_builder_t* b, rocke_op_t* op, val_table_t* vt, const char*** out_added, int* out_n)
{
    *out_added = NULL;
    *out_n = 0;
    if(op->opcode != ROCKE_OP_SCF_FOR)
    {
        return;
    }
    /* Capacity = the induction var (at most 1) + every iter-arg; an scf.for can
     * carry dozens, so size to the actual count rather than a fixed cap. */
    const rocke_attr_value_t* ia = rocke_attr_get(&op->attrs, "iter_args");
    int added_cap = 1 + ((ia && ia->kind == ROCKE_ATTR_LIST) ? ia->u.list.count : 0);
    const char** added
        = (const char**)rocke_arena_alloc(&b->arena, sizeof(const char*) * (size_t)added_cap);
    int na = 0;
    if(!added)
    {
        return;
    }

    const char* iv = rocke_attr_get_str(&op->attrs, "iv");
    const char* iv_type = rocke_attr_get_str(&op->attrs, "iv_type");
    if(iv && iv_type)
    {
        const rocke_type_t* t = parse_type(b, iv_type, strlen(iv_type));
        if(t)
        {
            rocke_value_t* v = rocke_i_value_named(b, iv, t);
            if(v)
            {
                v->op = op;
                vt_put(vt, v->name, v);
                added[na++] = v->name;
            }
        }
    }
    if(ia && ia->kind == ROCKE_ATTR_LIST && ia->u.list.count > 0)
    {
        for(int i = 0; i < ia->u.list.count; ++i)
        {
            const rocke_attr_map_t* entry = ia->u.list.items[i];
            const char* nm = rocke_attr_get_str(entry, "name");
            const char* ty = rocke_attr_get_str(entry, "type");
            if(nm && ty)
            {
                const rocke_type_t* t = parse_type(b, ty, strlen(ty));
                if(t)
                {
                    rocke_value_t* v = rocke_i_value_named(b, nm, t);
                    if(v)
                    {
                        v->op = op;
                        vt_put(vt, v->name, v);
                        added[na++] = v->name;
                    }
                }
            }
        }
    }
    if(na > 0)
    {
        const char** arr = (const char**)rocke_arena_alloc(&b->arena, sizeof(char*) * (size_t)na);
        if(arr)
        {
            memcpy(arr, added, sizeof(char*) * (size_t)na);
            *out_added = arr;
            *out_n = na;
        }
    }
}

/* Parse a single op (the current line is its head); appends nested regions. */
static rocke_op_t* parse_op(rocke_ir_builder_t* b, line_reader_t* lr, val_table_t* vt)
{
    char* ln = lr_next(lr);
    if(!ln)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "unexpected EOF in region");
        return NULL;
    }
    size_t lnlen = strlen(ln);

    /* results: detect ' = ' occurring before ' ( '. */
    const char* paren = strstr(ln, " ( ");
    const char* eq = strstr(ln, " = ");
    const char* rest = ln;

    /* result Values, collected before op build. Sized to the actual arity:
     * scf.for loops can yield dozens of results (online-softmax / multi-
     * accumulator MoE), so a fixed cap would silently drop the overflow. */
    const rocke_type_t** result_types = NULL;
    const char** result_names = NULL;
    int num_results = 0;

    if(eq && (!paren || eq < paren))
    {
        size_t reslen = (size_t)(eq - ln);
        int maxf = count_top(ln, reslen, ',');
        const char** fstart
            = (const char**)rocke_arena_alloc(&b->arena, sizeof(const char*) * (size_t)maxf);
        size_t* flen = (size_t*)rocke_arena_alloc(&b->arena, sizeof(size_t) * (size_t)maxf);
        result_types = (const rocke_type_t**)rocke_arena_alloc(
            &b->arena, sizeof(const rocke_type_t*) * (size_t)maxf);
        result_names
            = (const char**)rocke_arena_alloc(&b->arena, sizeof(const char*) * (size_t)maxf);
        if(!fstart || !flen || !result_types || !result_names)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "result buffer OOM");
            return NULL;
        }
        int nf = split_top(ln, reslen, ',', fstart, flen, maxf);
        for(int i = 0; i < nf; ++i)
        {
            const char* fs = fstart[i];
            size_t fl = flen[i];
            /* strip */
            while(fl > 0 && (fs[0] == ' ' || fs[0] == '\t'))
            {
                ++fs;
                --fl;
            }
            while(fl > 0 && (fs[fl - 1] == ' ' || fs[fl - 1] == '\t'))
            {
                --fl;
            }
            /* split on " : " */
            const char* col = NULL;
            for(size_t k = 0; k + 3 <= fl; ++k)
            {
                if(fs[k] == ' ' && fs[k + 1] == ':' && fs[k + 2] == ' ')
                {
                    col = fs + k;
                    break;
                }
            }
            if(!col)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "result missing ' : '");
                return NULL;
            }
            size_t rn_len = (size_t)(col - fs);
            char* rname = (char*)rocke_arena_alloc(&b->arena, rn_len + 1);
            if(!rname)
            {
                rocke_i_set_err(b, ROCKE_ERR_OOM, "result OOM");
                return NULL;
            }
            memcpy(rname, fs, rn_len);
            rname[rn_len] = '\0';
            const char* ts = col + 3;
            size_t tn = fl - rn_len - 3;
            const rocke_type_t* rt = parse_type(b, ts, tn);
            if(!rt)
            {
                return NULL;
            }
            result_names[num_results] = rname;
            result_types[num_results] = rt;
            num_results++;
        }
        rest = eq + 3;
    }

    /* rest: "<opname> ( <operands> ) [<attr-map>] [@loc \"...\"]" */
    const char* rparen = strstr(rest, " ( ");
    if(!rparen)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "op line missing ' ( '");
        return NULL;
    }
    size_t opname_len = (size_t)(rparen - rest);
    /* rstrip opname */
    while(opname_len > 0 && (rest[opname_len - 1] == ' '))
    {
        --opname_len;
    }
    char opname[64];
    if(opname_len >= sizeof(opname))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "opname too long");
        return NULL;
    }
    memcpy(opname, rest, opname_len);
    opname[opname_len] = '\0';
    rocke_opcode_t opcode = rocke_opcode_from_name(opname);
    if(opcode == ROCKE_OP_INVALID)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "unknown opcode '%s'", opname);
        return NULL;
    }

    const char* after = rparen + 3;
    /* find the first ')' closing the operand list (operands are %ids + commas) */
    const char* close = strchr(after, ')');
    if(!close)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "operand list missing ')'");
        return NULL;
    }
    size_t op_str_len = (size_t)(close - after);
    /* rstrip operand str */
    while(op_str_len > 0 && (after[op_str_len - 1] == ' '))
    {
        --op_str_len;
    }

    rocke_value_t** operands = NULL;
    int num_operands = 0;
    if(op_str_len > 0)
    {
        int maxf = count_top(after, op_str_len, ',');
        const char** fstart
            = (const char**)rocke_arena_alloc(&b->arena, sizeof(const char*) * (size_t)maxf);
        size_t* flen = (size_t*)rocke_arena_alloc(&b->arena, sizeof(size_t) * (size_t)maxf);
        operands
            = (rocke_value_t**)rocke_arena_alloc(&b->arena, sizeof(rocke_value_t*) * (size_t)maxf);
        if(!fstart || !flen || !operands)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "operand buffer OOM");
            return NULL;
        }
        int nf = split_top(after, op_str_len, ',', fstart, flen, maxf);
        for(int i = 0; i < nf; ++i)
        {
            const char* fs = fstart[i];
            size_t fl = flen[i];
            while(fl > 0 && (fs[0] == ' ' || fs[0] == '\t'))
            {
                ++fs;
                --fl;
            }
            while(fl > 0 && (fs[fl - 1] == ' ' || fs[fl - 1] == '\t'))
            {
                --fl;
            }
            if(fl == 0)
            {
                continue;
            }
            char oid[128];
            if(fl >= sizeof(oid))
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "operand id too long");
                return NULL;
            }
            memcpy(oid, fs, fl);
            oid[fl] = '\0';
            rocke_value_t* ov = vt_get(vt, oid);
            if(!ov)
            {
                rocke_i_set_err(b,
                                ROCKE_ERR_VALUE,
                                "operand '%s' used before definition in op '%s'",
                                oid,
                                opname);
                return NULL;
            }
            operands[num_operands++] = ov;
        }
    }

    /* tail after ')' : optional attr-map then optional @loc */
    const char* tail = close + 1;
    while(*tail == ' ')
    {
        ++tail;
    }
    rocke_attr_map_t attrs;
    rocke_attr_map_init(&attrs);
    const char* loc = NULL;
    if(*tail)
    {
        if(*tail == '{')
        {
            ser_scanner_t sc;
            sc.text = tail;
            sc.i = 0;
            sc.n = strlen(tail);
            sc.b = b;
            if(parse_attr_map(&sc, &attrs) != 0)
            {
                return NULL;
            }
            tail = tail + sc.i;
            while(*tail == ' ')
            {
                ++tail;
            }
        }
        if(strncmp(tail, "@loc", 4) == 0)
        {
            const char* q = strchr(tail, '"');
            if(q)
            {
                ser_scanner_t sc;
                sc.text = q + 1;
                sc.i = 0;
                sc.n = strlen(q + 1);
                sc.b = b;
                loc = sc_read_quoted(&sc);
                if(!loc)
                {
                    return NULL;
                }
            }
        }
    }
    (void)lnlen;

    /* Build the op manually so result Values carry the EXACT parsed names. We
     * cannot use rocke_b_op (it mints fresh %vN names). Allocate the op + arrays
     * in the arena, mirroring rocke_i_op's layout. */
    rocke_op_t* op = (rocke_op_t*)rocke_arena_calloc(&b->arena, sizeof(*op));
    if(!op)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "op OOM");
        return NULL;
    }
    op->opcode = opcode;
    op->name = rocke_opcode_name(opcode);
    op->loc = loc;

    if(num_operands > 0)
    {
        op->operands = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, sizeof(rocke_value_t*) * (size_t)num_operands);
        if(!op->operands)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "op operands OOM");
            return NULL;
        }
        memcpy(op->operands, operands, sizeof(rocke_value_t*) * (size_t)num_operands);
        op->num_operands = num_operands;
    }

    if(num_results > 0)
    {
        op->results = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, sizeof(rocke_value_t*) * (size_t)num_results);
        if(!op->results)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "op results OOM");
            return NULL;
        }
        for(int i = 0; i < num_results; ++i)
        {
            rocke_value_t* r = rocke_i_value_named(b, result_names[i], result_types[i]);
            if(!r)
            {
                return NULL;
            }
            r->op = op;
            op->results[i] = r;
        }
        op->num_results = num_results;
    }

    /* attrs: move the parsed map into the op (deep-share; arena lifetime). */
    op->attrs = attrs;

    /* register results in the value table */
    for(int i = 0; i < num_results; ++i)
    {
        if(vt_get(vt, op->results[i]->name))
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "SSA id '%s' redefined", op->results[i]->name);
            return NULL;
        }
        if(vt_put(vt, op->results[i]->name, op->results[i]) != 0)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "value table OOM");
            return NULL;
        }
    }

    /* register block-defined values (iv, iter-args) before parsing regions. */
    const char** added = NULL;
    int n_added = 0;
    register_block_values(b, op, vt, &added, &n_added);
    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* nested regions follow while next line opens a region. */
    int nregions = 0;
    int cap_regions = 0;
    rocke_region_t** regions = NULL;
    for(;;)
    {
        char* nxt = lr_peek(lr);
        if(nxt && strncmp(nxt, "region @", 8) == 0)
        {
            rocke_region_t* reg = parse_region(b, lr, vt);
            if(!reg)
            {
                return NULL;
            }
            if(nregions >= cap_regions)
            {
                int nc = cap_regions ? cap_regions * 2 : 2;
                rocke_region_t** nr = (rocke_region_t**)rocke_arena_alloc(
                    &b->arena, sizeof(rocke_region_t*) * (size_t)nc);
                if(!nr)
                {
                    rocke_i_set_err(b, ROCKE_ERR_OOM, "regions OOM");
                    return NULL;
                }
                if(regions && nregions)
                {
                    memcpy(nr, regions, sizeof(rocke_region_t*) * (size_t)nregions);
                }
                regions = nr;
                cap_regions = nc;
            }
            regions[nregions++] = reg;
        }
        else
        {
            break;
        }
    }
    if(nregions > 0)
    {
        op->regions = regions;
        op->num_regions = nregions;
    }

    /* retire block-scope values (not visible to siblings). */
    for(int i = 0; i < n_added; ++i)
    {
        vt_remove(vt, added[i]);
    }

    return op;
}

static rocke_region_t* parse_region(rocke_ir_builder_t* b, line_reader_t* lr, val_table_t* vt)
{
    char* rline = lr_next(lr);
    if(!rline || strncmp(rline, "region @", 8) != 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "expected 'region @<label> {'");
        return NULL;
    }
    size_t rl = strlen(rline);
    if(rl == 0 || rline[rl - 1] != '{')
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "region line missing '{'");
        return NULL;
    }
    /* label = between "region @" and trailing " {" */
    const char* ls = rline + 8;
    size_t ln = rl - 8 - 1; /* drop the '{' */
    while(ln > 0 && (ls[ln - 1] == ' ' || ls[ln - 1] == '\t'))
    {
        --ln;
    }
    char label[64];
    if(ln >= sizeof(label))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "region label too long");
        return NULL;
    }
    memcpy(label, ls, ln);
    label[ln] = '\0';

    rocke_region_t* region = rocke_i_new_region(b, label);
    if(!region)
    {
        return NULL;
    }
    for(;;)
    {
        char* peek = lr_peek(lr);
        if(!peek)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "unterminated region");
            return NULL;
        }
        if(strcmp(peek, "}") == 0)
        {
            lr_next(lr);
            break;
        }
        rocke_op_t* op = parse_op(b, lr, vt);
        if(!op)
        {
            return NULL;
        }
        /* append to region directly (we are not using the region stack). */
        if(region->num_ops >= region->cap_ops)
        {
            int nc = region->cap_ops ? region->cap_ops * 2 : 8;
            rocke_op_t** no
                = (rocke_op_t**)rocke_arena_alloc(&b->arena, sizeof(rocke_op_t*) * (size_t)nc);
            if(!no)
            {
                rocke_i_set_err(b, ROCKE_ERR_OOM, "region ops OOM");
                return NULL;
            }
            if(region->ops && region->num_ops)
            {
                memcpy(no, region->ops, sizeof(rocke_op_t*) * (size_t)region->num_ops);
            }
            region->ops = no;
            region->cap_ops = nc;
        }
        region->ops[region->num_ops++] = op;
    }
    return region;
}

/* Parse a param line: "%<name> : <type> [<attr-map>]" */
static rocke_param_t* parse_param(rocke_ir_builder_t* b, const char* ln, rocke_value_t** out_val)
{
    if(ln[0] != '%')
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "bad param line");
        return NULL;
    }
    /* attr-map starts at first '{' */
    const char* brace = strchr(ln, '{');
    rocke_attr_map_t attrs;
    rocke_attr_map_init(&attrs);
    size_t headlen = brace ? (size_t)(brace - ln) : strlen(ln);
    if(brace)
    {
        ser_scanner_t sc;
        sc.text = brace;
        sc.i = 0;
        sc.n = strlen(brace);
        sc.b = b;
        if(parse_attr_map(&sc, &attrs) != 0)
        {
            return NULL;
        }
    }
    /* head = ln[:headlen], rstrip */
    while(headlen > 0 && (ln[headlen - 1] == ' ' || ln[headlen - 1] == '\t'))
    {
        --headlen;
    }
    /* split on " : " */
    const char* col = NULL;
    for(size_t k = 0; k + 3 <= headlen; ++k)
    {
        if(ln[k] == ' ' && ln[k + 1] == ':' && ln[k + 2] == ' ')
        {
            col = ln + k;
            break;
        }
    }
    if(!col)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "param missing ' : '");
        return NULL;
    }
    /* name = ln[1 : col] (without '%') */
    size_t name_len = (size_t)(col - ln) - 1;
    char* name = (char*)rocke_arena_alloc(&b->arena, name_len + 1);
    if(!name)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "param OOM");
        return NULL;
    }
    memcpy(name, ln + 1, name_len);
    name[name_len] = '\0';
    const char* ts = col + 3;
    size_t tn = headlen - (size_t)(col - ln) - 3;
    const rocke_type_t* t = parse_type(b, ts, tn);
    if(!t)
    {
        return NULL;
    }

    rocke_param_t* p = (rocke_param_t*)rocke_arena_calloc(&b->arena, sizeof(*p));
    if(!p)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "param OOM");
        return NULL;
    }
    p->name = name;
    p->type = t;
    p->attrs = attrs;

    /* param value (name carries leading '%') */
    char* full = (char*)rocke_arena_alloc(&b->arena, name_len + 2);
    if(!full)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "param OOM");
        return NULL;
    }
    full[0] = '%';
    memcpy(full + 1, name, name_len + 1);
    rocke_value_t* v = rocke_i_value_named(b, full, t);
    if(!v)
    {
        return NULL;
    }
    *out_val = v;
    return p;
}

static rocke_status_t
    ir_parse_impl(const char* text, rocke_ir_builder_t* b, rocke_kernel_def_t** out)
{
    if(out)
    {
        *out = NULL;
    }
    if(!text || !b)
    {
        return ROCKE_ERR_VALUE;
    }
    if(!rocke_i_live(b))
    {
        return b->status;
    }

    line_reader_t lr;
    if(lr_init(b, &lr, text) != 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "parse: line reader OOM");
        return b->status;
    }

    /* header */
    char* header = lr_next(&lr);
    if(!header)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "empty input");
        return b->status;
    }
    {
        /* split into two tokens */
        char* sp = strchr(header, ' ');
        if(!sp)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "bad header '%s'", header);
            return b->status;
        }
        *sp = '\0';
        char* ver = sp + 1;
        while(*ver == ' ')
        {
            ++ver;
        }
        if(strcmp(header, SER_FORMAT_NAME) != 0)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "bad header format name");
            return b->status;
        }
        if(strcmp(ver, SER_FORMAT_VERSION) != 0)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "unsupported IR format version '%s'", ver);
            return b->status;
        }
    }

    /* kernel line */
    char* kline = lr_next(&lr);
    if(!kline || strncmp(kline, "kernel @", 8) != 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "bad kernel line");
        return b->status;
    }
    size_t kl = strlen(kline);
    if(kl == 0 || kline[kl - 1] != '{')
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "kernel line missing '{'");
        return b->status;
    }
    const char* ns = kline + 8;
    size_t nn = kl - 8 - 1;
    while(nn > 0 && (ns[nn - 1] == ' ' || ns[nn - 1] == '\t'))
    {
        --nn;
    }
    /* (re)build the kernel def in the arena with the parsed name. The builder
     * already created a default kernel in init; we replace it. */
    rocke_kernel_def_t* k = (rocke_kernel_def_t*)rocke_arena_calloc(&b->arena, sizeof(*k));
    if(!k)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "kernel OOM");
        return b->status;
    }
    char* kname = (char*)rocke_arena_alloc(&b->arena, nn + 1);
    if(!kname)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "kernel name OOM");
        return b->status;
    }
    memcpy(kname, ns, nn);
    kname[nn] = '\0';
    k->name = kname;
    rocke_attr_map_init(&k->attrs);

    val_table_t vt;
    vt.entries = NULL;
    vt.count = 0;
    vt.cap = 0;
    vt.b = b;

    /* optional kernel attrs line */
    char* nxt = lr_peek(&lr);
    if(nxt && strncmp(nxt, "attrs ", 6) == 0)
    {
        lr_next(&lr);
        ser_scanner_t sc;
        sc.text = nxt + 6;
        sc.i = 0;
        sc.n = strlen(nxt + 6);
        sc.b = b;
        if(parse_attr_map(&sc, &k->attrs) != 0)
        {
            return b->status;
        }
    }

    /* params block */
    char* pline = lr_next(&lr);
    if(!pline || strcmp(pline, "params {") != 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "expected 'params {'");
        return b->status;
    }
    for(;;)
    {
        char* ln = lr_next(&lr);
        if(!ln)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "unterminated params block");
            return b->status;
        }
        if(strcmp(ln, "}") == 0)
        {
            break;
        }
        rocke_value_t* pv = NULL;
        rocke_param_t* p = parse_param(b, ln, &pv);
        if(!p)
        {
            return b->status;
        }
        /* append to kernel */
        if(k->num_params >= k->cap_params)
        {
            int nc = k->cap_params ? k->cap_params * 2 : 8;
            rocke_param_t** np = (rocke_param_t**)rocke_arena_alloc(
                &b->arena, sizeof(rocke_param_t*) * (size_t)nc);
            if(!np)
            {
                rocke_i_set_err(b, ROCKE_ERR_OOM, "params OOM");
                return b->status;
            }
            if(k->params && k->num_params)
            {
                memcpy(np, k->params, sizeof(rocke_param_t*) * (size_t)k->num_params);
            }
            k->params = np;
            k->cap_params = nc;
        }
        k->params[k->num_params++] = p;
        if(vt_put(&vt, pv->name, pv) != 0)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "value table OOM");
            return b->status;
        }
    }

    /* body region */
    rocke_region_t* body = parse_region(b, &lr, &vt);
    if(!body)
    {
        return b->status;
    }
    k->body = body;

    /* closing kernel brace */
    char* closing = lr_next(&lr);
    if(!closing || strcmp(closing, "}") != 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "expected closing kernel '}'");
        return b->status;
    }

    b->kernel = k;
    if(out)
    {
        *out = k;
    }
    return ROCKE_OK;
}

/* ====================================================================== *
 *                    Canonicalization (spec 7)                           *
 * ====================================================================== */

typedef struct rename_table
{
    const char** olds; /* arena array */
    int count;
    int cap;
    rocke_ir_builder_t* b;
} rename_table_t;

static int rn_index(rename_table_t* rt, const char* name)
{
    for(int i = 0; i < rt->count; ++i)
    {
        if(strcmp(rt->olds[i], name) == 0)
        {
            return i;
        }
    }
    return -1;
}

static void rn_add(rename_table_t* rt, const char* name)
{
    if(rn_index(rt, name) >= 0)
    {
        return;
    }
    if(rt->count >= rt->cap)
    {
        int nc = rt->cap ? rt->cap * 2 : 64;
        const char** na
            = (const char**)rocke_arena_alloc(&rt->b->arena, sizeof(char*) * (size_t)nc);
        if(!na)
        {
            return;
        }
        if(rt->olds && rt->count)
        {
            memcpy(na, rt->olds, sizeof(char*) * (size_t)rt->count);
        }
        rt->olds = na;
        rt->cap = nc;
    }
    rt->olds[rt->count++] = name;
}

static void walk_region_defs(rename_table_t* rt, const rocke_region_t* region);

static void walk_op_defs(rename_table_t* rt, const rocke_op_t* op)
{
    for(int i = 0; i < op->num_results; ++i)
    {
        if(op->results[i] && op->results[i]->name)
        {
            rn_add(rt, op->results[i]->name);
        }
    }
    if(op->opcode == ROCKE_OP_SCF_FOR)
    {
        const char* iv = rocke_attr_get_str(&op->attrs, "iv");
        if(iv)
        {
            rn_add(rt, iv);
        }
        const rocke_attr_value_t* ia = rocke_attr_get(&op->attrs, "iter_args");
        if(ia && ia->kind == ROCKE_ATTR_LIST)
        {
            for(int i = 0; i < ia->u.list.count; ++i)
            {
                const char* nm = rocke_attr_get_str(ia->u.list.items[i], "name");
                if(nm)
                {
                    rn_add(rt, nm);
                }
            }
        }
    }
    for(int i = 0; i < op->num_regions; ++i)
    {
        walk_region_defs(rt, op->regions[i]);
    }
}

static void walk_region_defs(rename_table_t* rt, const rocke_region_t* region)
{
    if(!region)
    {
        return;
    }
    for(int i = 0; i < region->num_ops; ++i)
    {
        walk_op_defs(rt, region->ops[i]);
    }
}

/* Build the rename mapping over a kernel: params (ABI order) then pre-order op
 * walk. Returns the renamed id for `old` (arena %N string), or `old` if unmapped.
 */
static const char* rn_get(rename_table_t* rt, const char* old)
{
    int idx = rn_index(rt, old);
    if(idx < 0)
    {
        return old;
    }
    return rocke_arena_printf(&rt->b->arena, "%%%d", idx);
}

/* Recursively rebuild a kernel with renamed SSA ids and loc stripped, then
 * serialize. We mutate a parsed copy in place (the arena copy is disposable). */
static void canon_rename_region(rename_table_t* rt, rocke_region_t* region);

static void canon_rename_attrs(rename_table_t* rt, rocke_op_t* op)
{
    /* rename iv / iter_args[*].name in attrs (the only id-bearing attrs). */
    if(op->opcode != ROCKE_OP_SCF_FOR)
    {
        return;
    }
    rocke_attr_entry_t* iv_e = NULL;
    for(int i = 0; i < op->attrs.count; ++i)
    {
        if(strcmp(op->attrs.entries[i].key, "iv") == 0
           && op->attrs.entries[i].value.kind == ROCKE_ATTR_STR)
        {
            iv_e = &op->attrs.entries[i];
        }
    }
    if(iv_e)
    {
        iv_e->value.u.s = rn_get(rt, iv_e->value.u.s);
    }
    const rocke_attr_value_t* ia = rocke_attr_get(&op->attrs, "iter_args");
    if(ia && ia->kind == ROCKE_ATTR_LIST)
    {
        for(int i = 0; i < ia->u.list.count; ++i)
        {
            rocke_attr_map_t* entry = ia->u.list.items[i];
            for(int j = 0; j < entry->count; ++j)
            {
                if(strcmp(entry->entries[j].key, "name") == 0
                   && entry->entries[j].value.kind == ROCKE_ATTR_STR)
                {
                    entry->entries[j].value.u.s = rn_get(rt, entry->entries[j].value.u.s);
                }
            }
        }
    }
}

static void canon_rename_op(rename_table_t* rt, rocke_op_t* op)
{
    for(int i = 0; i < op->num_operands; ++i)
    {
        if(op->operands[i] && op->operands[i]->name)
        {
            op->operands[i]->name = rn_get(rt, op->operands[i]->name);
        }
    }
    for(int i = 0; i < op->num_results; ++i)
    {
        if(op->results[i] && op->results[i]->name)
        {
            op->results[i]->name = rn_get(rt, op->results[i]->name);
        }
    }
    canon_rename_attrs(rt, op);
    op->loc = NULL; /* strip */
    for(int i = 0; i < op->num_regions; ++i)
    {
        canon_rename_region(rt, op->regions[i]);
    }
}

static void canon_rename_region(rename_table_t* rt, rocke_region_t* region)
{
    if(!region)
    {
        return;
    }
    for(int i = 0; i < region->num_ops; ++i)
    {
        canon_rename_op(rt, region->ops[i]);
    }
}

/* Boundary shim: a ckc::Error thrown while parsing is caught here and recorded
 * on the builder, returning the exception's status code (the C ABI is
 * unchanged). */
rocke_status_t rocke_ir_parse(const char* text, rocke_ir_builder_t* b, rocke_kernel_def_t** out)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t { return ir_parse_impl(text, b, out); });
}

rocke_status_t rocke_ir_canonicalize(const char* text, char** out_text)
{
    if(out_text)
    {
        *out_text = NULL;
    }
    if(!text || !out_text)
    {
        return ROCKE_ERR_VALUE;
    }
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "canon") != ROCKE_OK)
    {
        return ROCKE_ERR_OOM;
    }
    rocke_kernel_def_t* k = NULL;
    rocke_status_t st = rocke_ir_parse(text, &b, &k);
    if(st != ROCKE_OK || !k)
    {
        rocke_ir_builder_free(&b);
        return st == ROCKE_OK ? ROCKE_ERR_VALUE : st;
    }

    rename_table_t rt;
    rt.olds = NULL;
    rt.count = 0;
    rt.cap = 0;
    rt.b = &b;

    /* def order: params first (ABI), then pre-order op walk. */
    for(int i = 0; i < k->num_params; ++i)
    {
        /* param id is "%name" */
        const char* nm = rocke_arena_printf(&b.arena, "%%%s", k->params[i]->name);
        if(nm)
        {
            rn_add(&rt, nm);
        }
    }
    walk_region_defs(&rt, k->body);

    /* rename params (drop the '%' for the Param.name field). */
    for(int i = 0; i < k->num_params; ++i)
    {
        const char* oldid = rocke_arena_printf(&b.arena, "%%%s", k->params[i]->name);
        const char* newid = rn_get(&rt, oldid); /* "%N" */
        if(newid && newid[0] == '%')
        {
            k->params[i]->name = rocke_arena_strdup(&b.arena, newid + 1);
        }
    }
    canon_rename_region(&rt, k->body);

    char* ser = NULL;
    st = rocke_ir_serialize(k, &ser);
    rocke_ir_builder_free(&b);
    if(st != ROCKE_OK)
    {
        return st;
    }
    *out_text = ser;
    return ROCKE_OK;
}
