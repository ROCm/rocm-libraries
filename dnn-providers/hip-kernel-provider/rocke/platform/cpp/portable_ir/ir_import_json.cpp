/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ir_import_json.c -- importer for the portable CK-DSL IR artifact
 * (schema "rocke.ir/v1", produced by ck_dsl.core.ir_export). See
 * rocke/ir_import.h for the contract.
 *
 * Two parts:
 *   1. a small dependency-free recursive-descent JSON parser -> tagged DOM;
 *   2. a walker that re-drives the DOM through the public rocke_b_* builder API.
 *
 * SSA values are resolved by their exported names ("%vN" / "%A" / "%k0") via a
 * name->value table with proper region scoping (body-local values, including
 * scf.for induction/iter-arg values, are popped when the region closes). Every
 * region-bearing op (scf.for / scf.if) is built via its real control-flow
 * builder so induction/iter values and the iter_args list attr are reconstructed
 * exactly as a native C build would; all other ops go through the generic
 * rocke_b_op path keyed by rocke_opcode_from_name.
 */
#include "rocke/ir_import.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"

/* ====================================================================== */
/* Minimal JSON DOM + parser                                              */
/* ====================================================================== */

typedef enum
{
    J_NULL,
    J_BOOL,
    J_NUM,
    J_STR,
    J_ARR,
    J_OBJ
} jkind_t;

typedef struct jval jval_t;
typedef struct
{
    char* key;
    jval_t* val;
} jmember_t;

struct jval
{
    jkind_t kind;
    bool b;
    double num;
    char* str; /* J_STR (owned) */
    jval_t** arr; /* J_ARR */
    int arr_len;
    jmember_t* obj; /* J_OBJ */
    int obj_len;
};

typedef struct
{
    const char* p;
    const char* end;
    rocke_arena_t* arena;
    char err[256];
    bool failed;
} jparser_t;

static void jp_fail(jparser_t* jp, const char* msg)
{
    if(!jp->failed)
    {
        snprintf(jp->err, sizeof jp->err, "json: %s", msg);
        jp->failed = true;
    }
}

static void jp_ws(jparser_t* jp)
{
    while(jp->p < jp->end)
    {
        char c = *jp->p;
        if(c == ' ' || c == '\t' || c == '\n' || c == '\r')
            jp->p++;
        else
            break;
    }
}

static jval_t* jval_new(jparser_t* jp, jkind_t k)
{
    jval_t* v = (jval_t*)rocke_arena_calloc(jp->arena, sizeof(jval_t));
    if(v)
        v->kind = k;
    return v;
}

static void jval_free(jval_t* v)
{
    /* DOM storage is owned by the parser arena; the whole arena is destroyed
     * at once after import. Keep the helper so error paths can remain simple. */
    (void)v;
}

static jval_t* jp_value(jparser_t* jp);

/* Parse a JSON string literal (leading '"' already detected). Returns malloc'd
 * NUL-terminated decoded string, or NULL on failure. */
static char* jp_string_raw(jparser_t* jp)
{
    if(jp->p >= jp->end || *jp->p != '"')
    {
        jp_fail(jp, "expected string");
        return NULL;
    }
    jp->p++; /* consume opening quote */
    const char* start = jp->p;
    const char* q = start;
    bool closed = false;
    while(q < jp->end)
    {
        if(*q == '"')
        {
            closed = true;
            break;
        }
        if(*q == '\\')
        {
            q++;
            if(q >= jp->end)
                break;
        }
        q++;
    }
    if(!closed)
    {
        jp_fail(jp, "unterminated string");
        return NULL;
    }
    /* The decoded UTF-8 string is never longer than the raw JSON payload between
     * quotes: simple escapes shrink two bytes to one, and \uXXXX shrinks six
     * bytes to at most three UTF-8 bytes for the BMP subset we support. */
    size_t cap = (size_t)(q - start) + 1;
    char* out = (char*)rocke_arena_alloc(jp->arena, cap);
    if(!out)
    {
        jp_fail(jp, "oom string");
        return NULL;
    }
    size_t n = 0;
    while(jp->p < jp->end)
    {
        char c = *jp->p++;
        if(c == '"')
        {
            out[n] = '\0';
            return out;
        }
        if(c == '\\')
        {
            if(jp->p >= jp->end)
                break;
            char e = *jp->p++;
            switch(e)
            {
            case '"':
                out[n++] = '"';
                break;
            case '\\':
                out[n++] = '\\';
                break;
            case '/':
                out[n++] = '/';
                break;
            case 'n':
                out[n++] = '\n';
                break;
            case 't':
                out[n++] = '\t';
                break;
            case 'r':
                out[n++] = '\r';
                break;
            case 'b':
                out[n++] = '\b';
                break;
            case 'f':
                out[n++] = '\f';
                break;
            case 'u':
            {
                /* Minimal BMP handling: decode \uXXXX to UTF-8. */
                if(jp->end - jp->p < 4)
                {
                    jp_fail(jp, "bad \\u escape");
                    return NULL;
                }
                unsigned cp = 0;
                for(int i = 0; i < 4; i++)
                {
                    char h = *jp->p++;
                    cp <<= 4;
                    if(h >= '0' && h <= '9')
                        cp |= (unsigned)(h - '0');
                    else if(h >= 'a' && h <= 'f')
                        cp |= (unsigned)(h - 'a' + 10);
                    else if(h >= 'A' && h <= 'F')
                        cp |= (unsigned)(h - 'A' + 10);
                    else
                    {
                        jp_fail(jp, "bad hex in \\u");
                        return NULL;
                    }
                }
                if(cp < 0x80)
                {
                    out[n++] = (char)cp;
                }
                else if(cp < 0x800)
                {
                    out[n++] = (char)(0xC0 | (cp >> 6));
                    out[n++] = (char)(0x80 | (cp & 0x3F));
                }
                else
                {
                    out[n++] = (char)(0xE0 | (cp >> 12));
                    out[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    out[n++] = (char)(0x80 | (cp & 0x3F));
                }
                break;
            }
            default:
                jp_fail(jp, "bad escape");
                return NULL;
            }
        }
        else
        {
            out[n++] = c;
        }
    }
    jp_fail(jp, "unterminated string");
    return NULL;
}

static jval_t* jp_number(jparser_t* jp)
{
    char* endp = NULL;
    double d = strtod(jp->p, &endp);
    if(endp == jp->p)
    {
        jp_fail(jp, "bad number");
        return NULL;
    }
    jp->p = endp;
    jval_t* v = jval_new(jp, J_NUM);
    if(!v)
    {
        jp_fail(jp, "oom num");
        return NULL;
    }
    v->num = d;
    return v;
}

static bool jp_lit(jparser_t* jp, const char* lit)
{
    size_t n = strlen(lit);
    if((size_t)(jp->end - jp->p) < n || strncmp(jp->p, lit, n) != 0)
        return false;
    jp->p += n;
    return true;
}

static jval_t* jp_array(jparser_t* jp)
{
    jp->p++; /* '[' */
    jval_t* v = jval_new(jp, J_ARR);
    if(!v)
    {
        jp_fail(jp, "oom arr");
        return NULL;
    }
    int cap = 0;
    jp_ws(jp);
    if(jp->p < jp->end && *jp->p == ']')
    {
        jp->p++;
        return v;
    }
    while(jp->p < jp->end)
    {
        jval_t* item = jp_value(jp);
        if(jp->failed)
        {
            jval_free(item);
            jval_free(v);
            return NULL;
        }
        if(v->arr_len == cap)
        {
            cap = cap ? cap * 2 : 8;
            jval_t** na = (jval_t**)rocke_arena_alloc(jp->arena, (size_t)cap * sizeof(jval_t*));
            if(!na)
            {
                jval_free(item);
                jval_free(v);
                jp_fail(jp, "oom arr grow");
                return NULL;
            }
            if(v->arr && v->arr_len > 0)
                memcpy(na, v->arr, (size_t)v->arr_len * sizeof(jval_t*));
            v->arr = na;
        }
        v->arr[v->arr_len++] = item;
        jp_ws(jp);
        if(jp->p < jp->end && *jp->p == ',')
        {
            jp->p++;
            jp_ws(jp);
            continue;
        }
        if(jp->p < jp->end && *jp->p == ']')
        {
            jp->p++;
            return v;
        }
        jval_free(v);
        jp_fail(jp, "expected ',' or ']'");
        return NULL;
    }
    jval_free(v);
    jp_fail(jp, "unterminated array");
    return NULL;
}

static jval_t* jp_object(jparser_t* jp)
{
    jp->p++; /* '{' */
    jval_t* v = jval_new(jp, J_OBJ);
    if(!v)
    {
        jp_fail(jp, "oom obj");
        return NULL;
    }
    int cap = 0;
    jp_ws(jp);
    if(jp->p < jp->end && *jp->p == '}')
    {
        jp->p++;
        return v;
    }
    while(jp->p < jp->end)
    {
        jp_ws(jp);
        char* key = jp_string_raw(jp);
        if(jp->failed)
        {
            jval_free(v);
            return NULL;
        }
        jp_ws(jp);
        if(jp->p >= jp->end || *jp->p != ':')
        {
            jval_free(v);
            jp_fail(jp, "expected ':'");
            return NULL;
        }
        jp->p++;
        jp_ws(jp);
        jval_t* val = jp_value(jp);
        if(jp->failed)
        {
            jval_free(val);
            jval_free(v);
            return NULL;
        }
        if(v->obj_len == cap)
        {
            cap = cap ? cap * 2 : 8;
            jmember_t* nm
                = (jmember_t*)rocke_arena_alloc(jp->arena, (size_t)cap * sizeof(jmember_t));
            if(!nm)
            {
                jval_free(val);
                jval_free(v);
                jp_fail(jp, "oom obj grow");
                return NULL;
            }
            if(v->obj && v->obj_len > 0)
                memcpy(nm, v->obj, (size_t)v->obj_len * sizeof(jmember_t));
            v->obj = nm;
        }
        v->obj[v->obj_len].key = key;
        v->obj[v->obj_len].val = val;
        v->obj_len++;
        jp_ws(jp);
        if(jp->p < jp->end && *jp->p == ',')
        {
            jp->p++;
            continue;
        }
        if(jp->p < jp->end && *jp->p == '}')
        {
            jp->p++;
            return v;
        }
        jval_free(v);
        jp_fail(jp, "expected ',' or '}'");
        return NULL;
    }
    jval_free(v);
    jp_fail(jp, "unterminated object");
    return NULL;
}

static jval_t* jp_value(jparser_t* jp)
{
    jp_ws(jp);
    if(jp->p >= jp->end)
    {
        jp_fail(jp, "unexpected end");
        return NULL;
    }
    char c = *jp->p;
    if(c == '"')
    {
        char* s = jp_string_raw(jp);
        if(!s)
            return NULL;
        jval_t* v = jval_new(jp, J_STR);
        if(!v)
        {
            jp_fail(jp, "oom str node");
            return NULL;
        }
        v->str = s;
        return v;
    }
    if(c == '{')
        return jp_object(jp);
    if(c == '[')
        return jp_array(jp);
    if(c == '-' || (c >= '0' && c <= '9'))
        return jp_number(jp);
    if(jp_lit(jp, "true"))
    {
        jval_t* v = jval_new(jp, J_BOOL);
        if(v)
            v->b = true;
        return v;
    }
    if(jp_lit(jp, "false"))
    {
        jval_t* v = jval_new(jp, J_BOOL);
        if(v)
            v->b = false;
        return v;
    }
    if(jp_lit(jp, "null"))
        return jval_new(jp, J_NULL);
    jp_fail(jp, "unexpected token");
    return NULL;
}

static jval_t* json_parse(const char* text, rocke_arena_t* arena, char* err, size_t err_cap)
{
    jparser_t jp;
    jp.p = text;
    jp.end = text + strlen(text);
    jp.arena = arena;
    jp.err[0] = '\0';
    jp.failed = false;
    jval_t* root = jp_value(&jp);
    if(jp.failed)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "%s", jp.err);
        jval_free(root);
        return NULL;
    }
    return root;
}

/* ---- DOM accessors ---- */

static jval_t* jobj_get(const jval_t* o, const char* key)
{
    if(!o || o->kind != J_OBJ)
        return NULL;
    for(int i = 0; i < o->obj_len; i++)
        if(strcmp(o->obj[i].key, key) == 0)
            return o->obj[i].val;
    return NULL;
}

static const char* jstr(const jval_t* v)
{
    return (v && v->kind == J_STR) ? v->str : NULL;
}

/* ====================================================================== */
/* Importer                                                               */
/* ====================================================================== */

/* SSA name -> value resolution with O(1) lookup AND correct region scoping.
 *
 * `binds` is an append-only insertion log (one entry per vmap_put) that gives
 * the region-scoping primitive: capture mark = len on region entry, and on exit
 * vmap_truncate(mark) pops every binding made inside the region (restoring any
 * value it shadowed). `slots` holds one permanent record per distinct name with
 * the index of its CURRENT (latest, innermost) binding in `binds` -- a per-name
 * shadow stack threaded through binds[i].prev. `buckets` is a hash index
 * (name -> slot, separate chaining) so both put and get are O(1) amortized,
 * replacing the previous O(n^2) linear scan. */
typedef struct
{
    rocke_value_t* val;
    int prev; /* previous binds-index for this name (shadow stack), or -1   */
    int slot; /* owning slot index (avoids a re-hash on truncate)           */
} vbind_t;

typedef struct
{
    const char* name; /* JSON SSA id (points into the DOM; stable on import) */
    int head; /* latest binds-index for this name, or -1             */
    int next; /* next slot in the same hash bucket, or -1            */
} vslot_t;

typedef struct
{
    rocke_ir_builder_t* b;
    vbind_t* binds;
    int len;
    int cap;
    vslot_t* slots;
    int nslots;
    int scap;
    int* buckets; /* nbuckets entries, each a slot index or -1               */
    int nbuckets;
    char err[ROCKE_ERR_MSG_CAP];
    bool failed;
} importer_t;

static void imp_fail(importer_t* im, const char* fmt, ...)
{
    if(im->failed)
        return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(im->err, sizeof im->err, fmt, ap);
    va_end(ap);
    im->failed = true;
}

static unsigned vmap_hash(const char* s)
{
    /* FNV-1a */
    unsigned h = 2166136261u;
    for(; *s; s++)
    {
        h ^= (unsigned char)*s;
        h *= 16777619u;
    }
    return h;
}

static bool vmap_grow_buckets(importer_t* im, int want)
{
    int n = im->nbuckets ? im->nbuckets : 1024;
    while(n < want)
        n *= 2;
    int* nb = (int*)malloc((size_t)n * sizeof(int));
    if(!nb)
    {
        imp_fail(im, "oom vmap buckets");
        return false;
    }
    for(int i = 0; i < n; i++)
        nb[i] = -1;
    /* Re-link every existing slot into the new bucket array. */
    for(int s = 0; s < im->nslots; s++)
    {
        unsigned h = vmap_hash(im->slots[s].name) & (unsigned)(n - 1);
        im->slots[s].next = nb[h];
        nb[h] = s;
    }
    free(im->buckets);
    im->buckets = nb;
    im->nbuckets = n;
    return true;
}

/* Find the slot for `name`, or -1 if absent. */
static int vmap_find_slot(importer_t* im, const char* name)
{
    if(im->nbuckets == 0)
        return -1;
    unsigned h = vmap_hash(name) & (unsigned)(im->nbuckets - 1);
    for(int s = im->buckets[h]; s != -1; s = im->slots[s].next)
        if(strcmp(im->slots[s].name, name) == 0)
            return s;
    return -1;
}

static int vmap_intern_slot(importer_t* im, const char* name)
{
    int s = vmap_find_slot(im, name);
    if(s != -1)
        return s;
    if(im->nbuckets == 0 && !vmap_grow_buckets(im, 1024))
        return -1;
    if(im->nslots + 1 > (im->nbuckets * 3) / 4 && !vmap_grow_buckets(im, im->nbuckets * 2))
        return -1;
    if(im->nslots == im->scap)
    {
        int nc = im->scap ? im->scap * 2 : 256;
        vslot_t* ns = (vslot_t*)realloc(im->slots, (size_t)nc * sizeof(vslot_t));
        if(!ns)
        {
            imp_fail(im, "oom vmap slots");
            return -1;
        }
        im->slots = ns;
        im->scap = nc;
    }
    s = im->nslots++;
    im->slots[s].name = name;
    im->slots[s].head = -1;
    unsigned h = vmap_hash(name) & (unsigned)(im->nbuckets - 1);
    im->slots[s].next = im->buckets[h];
    im->buckets[h] = s;
    return s;
}

static void vmap_put(importer_t* im, const char* name, rocke_value_t* val)
{
    if(im->failed)
        return;
    int slot = vmap_intern_slot(im, name);
    if(slot < 0)
        return;
    if(im->len == im->cap)
    {
        int nc = im->cap ? im->cap * 2 : 64;
        vbind_t* nb = (vbind_t*)realloc(im->binds, (size_t)nc * sizeof(vbind_t));
        if(!nb)
        {
            imp_fail(im, "oom vmap");
            return;
        }
        im->binds = nb;
        im->cap = nc;
    }
    im->binds[im->len].val = val;
    im->binds[im->len].prev = im->slots[slot].head;
    im->binds[im->len].slot = slot;
    im->slots[slot].head = im->len;
    im->len++;
}

static rocke_value_t* vmap_get(importer_t* im, const char* name)
{
    int slot = vmap_find_slot(im, name);
    if(slot < 0 || im->slots[slot].head < 0)
        return NULL;
    return im->binds[im->slots[slot].head].val;
}

/* Pop every binding made since `mark`, restoring shadowed bindings. This is the
 * region-scope exit primitive (replaces the old `im->len = mark`). */
static void vmap_truncate(importer_t* im, int mark)
{
    for(int i = im->len - 1; i >= mark; i--)
        im->slots[im->binds[i].slot].head = im->binds[i].prev;
    im->len = mark;
}

/* The lowerer emits SSA value names verbatim into the .ll, so a value's name
 * must equal its exported id for byte-identical lowering. The generic builder
 * mints fresh "%vN" names; overwrite them with the exported id (arena-owned so
 * it outlives the import). */
static void rename_value(importer_t* im, rocke_value_t* v, const char* exported_id)
{
    if(!v || !exported_id)
        return;
    char* copy = rocke_arena_strdup(&im->b->arena, exported_id);
    if(copy)
        v->name = copy;
}

/* Build a rocke_type_t from a JSON type node: scalar -> name string; composite
 * -> {"kind": ...}. */
static const rocke_type_t* import_type(importer_t* im, const jval_t* t)
{
    if(!t)
    {
        imp_fail(im, "missing type");
        return NULL;
    }
    if(t->kind == J_STR)
    {
        const rocke_type_t* st = rocke_scalar_by_name(t->str);
        if(!st)
            imp_fail(im, "unknown scalar type '%s'", t->str);
        return st;
    }
    if(t->kind == J_OBJ)
    {
        const char* kind = jstr(jobj_get(t, "kind"));
        if(!kind)
        {
            imp_fail(im, "type object missing 'kind'");
            return NULL;
        }
        if(strcmp(kind, "vector") == 0)
        {
            const rocke_type_t* elem = import_type(im, jobj_get(t, "elem"));
            jval_t* cnt = jobj_get(t, "count");
            if(!elem || !cnt || cnt->kind != J_NUM)
                return im->failed ? nullptr : (imp_fail(im, "bad vector type"), nullptr);
            return rocke_vector_type(im->b, elem, (int)cnt->num);
        }
        if(strcmp(kind, "ptr") == 0)
        {
            const rocke_type_t* pointee = import_type(im, jobj_get(t, "pointee"));
            const char* space = jstr(jobj_get(t, "space"));
            if(!pointee || !space)
                return im->failed ? nullptr : (imp_fail(im, "bad ptr type"), nullptr);
            return rocke_ptr_type(im->b, pointee, space);
        }
        if(strcmp(kind, "smem") == 0)
        {
            const rocke_type_t* elem = import_type(im, jobj_get(t, "elem"));
            jval_t* shp = jobj_get(t, "shape");
            if(!elem || !shp || shp->kind != J_ARR)
                return im->failed ? nullptr : (imp_fail(im, "bad smem type"), nullptr);
            int rank = shp->arr_len;
            int dims[16];
            if(rank > 16)
            {
                imp_fail(im, "smem rank %d too large", rank);
                return NULL;
            }
            for(int i = 0; i < rank; i++)
                dims[i] = (int)shp->arr[i]->num;
            /* exclusive is reconstructed from the op attr in import_op (the
             * type node, like the canonical type name, omits it). */
            return rocke_smem_type(im->b, elem, dims, rank, 0);
        }
        imp_fail(im, "unknown type kind '%s'", kind);
        return NULL;
    }
    imp_fail(im, "bad type node");
    return NULL;
}

/* Populate a rocke_attr_map_t from a JSON typed-attr object (keys -> {"t","v"}).
 * List attrs ("l") are only used by scf.for and are handled by its dedicated
 * importer (which uses the real builder), so they are rejected here. */
static void import_attrs(importer_t* im, const jval_t* attrs, rocke_attr_map_t* m)
{
    rocke_attr_map_init(m);
    if(!attrs)
        return;
    if(attrs->kind != J_OBJ)
    {
        imp_fail(im, "attrs not an object");
        return;
    }
    for(int i = 0; i < attrs->obj_len && !im->failed; i++)
    {
        const char* key = attrs->obj[i].key;
        const jval_t* tv = attrs->obj[i].val;
        const char* t = jstr(jobj_get(tv, "t"));
        const jval_t* v = jobj_get(tv, "v");
        if(!t || !v)
        {
            imp_fail(im, "attr '%s' missing t/v", key);
            return;
        }
        if(strcmp(t, "i") == 0)
            rocke_attr_set_int(im->b, m, key, (int64_t)v->num);
        else if(strcmp(t, "f") == 0)
            rocke_attr_set_float(im->b, m, key, v->num);
        else if(strcmp(t, "b") == 0)
            rocke_attr_set_bool(im->b, m, key, v->b);
        else if(strcmp(t, "s") == 0)
            rocke_attr_set_str(im->b, m, key, v->str ? v->str : "");
        else
            imp_fail(im, "attr '%s' has unsupported kind '%s' (generic op)", key, t);
    }
}

static const char* strip_pct(const char* name)
{
    return (name && name[0] == '%') ? name + 1 : name;
}

static void import_region_ops(importer_t* im, const jval_t* region);

/* scf.for / scf.for_iter: drive the real control-flow builder so iv/iter values
 * and the iter_args list attr reconstruct exactly as a native build. */
static void import_scf_for(importer_t* im, const jval_t* op)
{
    const jval_t* operands = jobj_get(op, "operands");
    const jval_t* attrs = jobj_get(op, "attrs");
    const jval_t* results = jobj_get(op, "results");
    const jval_t* regions = jobj_get(op, "regions");
    if(!operands || operands->kind != J_ARR || operands->arr_len < 3)
    {
        imp_fail(im, "scf.for needs >=3 operands");
        return;
    }
    rocke_value_t* lo = vmap_get(im, jstr(operands->arr[0]));
    rocke_value_t* hi = vmap_get(im, jstr(operands->arr[1]));
    rocke_value_t* step = vmap_get(im, jstr(operands->arr[2]));
    if(!lo || !hi || !step)
    {
        imp_fail(im, "scf.for unresolved bound operand");
        return;
    }

    const char* iv_attr = NULL;
    bool unroll = false, elide = true;
    const jval_t* iv_tv = jobj_get(attrs, "iv");
    if(iv_tv)
        iv_attr = jstr(jobj_get(iv_tv, "v"));
    const jval_t* un_tv = jobj_get(attrs, "unroll");
    if(un_tv)
    {
        const jval_t* uv = jobj_get(un_tv, "v");
        if(uv)
            unroll = uv->b;
    }
    const jval_t* el_tv = jobj_get(attrs, "elide_trailing_barrier");
    if(el_tv)
    {
        const jval_t* ev = jobj_get(el_tv, "v");
        if(ev)
            elide = ev->b;
    }

    /* iter_args metadata (list of {name,type}); inits are operands[3..]. */
    const jval_t* ia_tv = jobj_get(attrs, "iter_args");
    const jval_t* ia_list = ia_tv ? jobj_get(ia_tv, "v") : NULL;
    int num_iter = ia_list && ia_list->kind == J_ARR ? ia_list->arr_len : 0;

    int mark = im->len;
    rocke_for_t f;
    if(num_iter > 0)
    {
        if(operands->arr_len < 3 + num_iter)
        {
            imp_fail(im, "scf.for missing iter init operands");
            return;
        }
        rocke_iter_arg_t* ia
            = (rocke_iter_arg_t*)calloc((size_t)num_iter, sizeof(rocke_iter_arg_t));
        const char** ia_names = (const char**)calloc((size_t)num_iter, sizeof(char*));
        if(!ia || !ia_names)
        {
            free(ia);
            free(ia_names);
            imp_fail(im, "oom iter_args");
            return;
        }
        for(int i = 0; i < num_iter; i++)
        {
            const jval_t* meta = ia_list->arr[i];
            const jval_t* nm_tv = jobj_get(meta, "name");
            const char* nm = nm_tv ? jstr(jobj_get(nm_tv, "v")) : NULL;
            ia_names[i] = nm; /* keep "%acc" form for vmap registration */
            ia[i].name = strip_pct(nm);
            ia[i].init = vmap_get(im, jstr(operands->arr[3 + i]));
            if(!ia[i].init)
            {
                free(ia);
                free(ia_names);
                imp_fail(im, "scf.for unresolved iter init");
                return;
            }
        }
        f = rocke_b_scf_for_iter(
            im->b, lo, hi, step, ia, num_iter, strip_pct(iv_attr), unroll, elide);
        if(iv_attr)
        {
            rename_value(im, f.iv, iv_attr);
            vmap_put(im, iv_attr, f.iv);
        }
        for(int i = 0; i < num_iter; i++)
        {
            rename_value(im, f.iter_vars[i], ia_names[i]);
            vmap_put(im, ia_names[i], f.iter_vars[i]);
        }
        free(ia);
        free(ia_names);
    }
    else
    {
        f = rocke_b_scf_for(im->b, lo, hi, step, strip_pct(iv_attr));
        if(iv_attr)
        {
            rename_value(im, f.iv, iv_attr);
            vmap_put(im, iv_attr, f.iv);
        }
    }
    if(!rocke_ir_builder_ok(im->b))
    {
        imp_fail(im, "scf.for build: %s", rocke_ir_builder_error(im->b));
        return;
    }

    /* Import the body region. */
    if(!regions || regions->kind != J_ARR || regions->arr_len < 1)
    {
        imp_fail(im, "scf.for missing body region");
        return;
    }
    rocke_b_region_enter(im->b, f.body);
    import_region_ops(im, regions->arr[0]);
    rocke_b_region_leave(im->b);
    vmap_truncate(im, mark); /* pop body-local + iv/iter bindings */
    if(im->failed)
        return;

    /* Register loop results in the parent scope. */
    if(results && results->kind == J_ARR)
    {
        for(int i = 0; i < results->arr_len; i++)
        {
            const char* rid = jstr(jobj_get(results->arr[i], "id"));
            if(rid && i < f.op->num_results)
            {
                rename_value(im, f.op->results[i], rid);
                vmap_put(im, rid, f.op->results[i]);
            }
        }
    }
}

static void import_scf_if(importer_t* im, const jval_t* op)
{
    const jval_t* operands = jobj_get(op, "operands");
    const jval_t* regions = jobj_get(op, "regions");
    if(!operands || operands->kind != J_ARR || operands->arr_len < 1)
    {
        imp_fail(im, "scf.if needs a condition operand");
        return;
    }
    rocke_value_t* cond = vmap_get(im, jstr(operands->arr[0]));
    if(!cond)
    {
        imp_fail(im, "scf.if unresolved condition");
        return;
    }
    if(regions && regions->kind == J_ARR && regions->arr_len > 1)
    {
        imp_fail(im, "scf.if with else region not supported yet");
        return;
    }
    int mark = im->len;
    rocke_if_t s = rocke_b_scf_if(im->b, cond);
    if(regions && regions->kind == J_ARR && regions->arr_len >= 1)
    {
        rocke_b_region_enter(im->b, s.then_region);
        import_region_ops(im, regions->arr[0]);
        rocke_b_region_leave(im->b);
    }
    vmap_truncate(im, mark);
}

/* Resolve an opcode name, applying portable-IR aliases. The Python builder names
 * some vectorized fp16 buffer ops without the dtype suffix the engine's opcode
 * registry uses ("tile.buffer_load_vN" vs "tile.buffer_load_vN_f16"); the op is
 * otherwise identical, so normalize on the round-trip. Engine core unchanged. */
/* Portable-IR opcode alias.
 *
 * A few Python IRBuilder ops are dtype-GENERIC -- their spelling carries no
 * dtype and the element type rides in the `elem_type` attr (tile.buffer_load,
 * tile.buffer_load_vN, tile.buffer_store...). The engine's opcode registry
 * instead keys those by a dtype suffix (tile.buffer_load_f16 / _bf16). Resolve
 * the exact name first, then "<name>_<elem_type>", then fall back to the _f16
 * entry -- which is what cpp/helpers/loads.cpp emits natively for this path, so
 * the imported graph matches a native C++ build. An unrepresentable dtype
 * resolves to nothing and the caller reports it, rather than silently lowering
 * as f16. */
static rocke_opcode_t import_opcode_from_name(const char* name, const char* elem_type)
{
    if(!name)
        return ROCKE_OP_INVALID;
    rocke_opcode_t op = rocke_opcode_from_name(name);
    if(op != ROCKE_OP_INVALID)
        return op;
    char buf[128];
    if(elem_type && *elem_type)
    {
        snprintf(buf, sizeof buf, "%s_%s", name, elem_type);
        op = rocke_opcode_from_name(buf);
        if(op != ROCKE_OP_INVALID)
            return op;
    }
    snprintf(buf, sizeof buf, "%s_f16", name);
    return rocke_opcode_from_name(buf);
}

/* The `elem_type` string attr of a typed-attr JSON object, or NULL. */
static const char* attr_elem_type(const jval_t* attrs)
{
    if(!attrs || attrs->kind != J_OBJ)
        return NULL;
    const jval_t* tv = jobj_get(attrs, "elem_type");
    return tv ? jstr(jobj_get(tv, "v")) : NULL;
}

/* Generic op: resolve operands, build attrs + result types, call rocke_b_op,
 * register results by their exported ids. */
static void import_generic_op(importer_t* im, const jval_t* op, const char* opcode_name)
{
    const jval_t* operands = jobj_get(op, "operands");
    const jval_t* results = jobj_get(op, "results");
    const jval_t* attrs = jobj_get(op, "attrs");
    const jval_t* regions = jobj_get(op, "regions");

    rocke_opcode_t opcode = import_opcode_from_name(opcode_name, attr_elem_type(attrs));
    if(opcode == ROCKE_OP_INVALID)
    {
        imp_fail(im, "unknown opcode '%s'", opcode_name);
        return;
    }

    if(regions && regions->kind == J_ARR && regions->arr_len > 0)
    {
        imp_fail(im, "opcode '%s' carries regions but is not a known control-flow op", opcode_name);
        return;
    }

    int n_ops = operands && operands->kind == J_ARR ? operands->arr_len : 0;
    int n_res = results && results->kind == J_ARR ? results->arr_len : 0;

    rocke_value_t** ops = NULL;
    const rocke_type_t** rtypes = NULL;
    if(n_ops)
    {
        ops = (rocke_value_t**)calloc((size_t)n_ops, sizeof(rocke_value_t*));
        if(!ops)
        {
            imp_fail(im, "oom operands");
            return;
        }
        for(int i = 0; i < n_ops; i++)
        {
            const char* nm = jstr(operands->arr[i]);
            ops[i] = vmap_get(im, nm);
            if(!ops[i])
            {
                imp_fail(im, "op '%s' unresolved operand '%s'", opcode_name, nm ? nm : "?");
                free(ops);
                return;
            }
        }
    }
    if(n_res)
    {
        rtypes = (const rocke_type_t**)calloc((size_t)n_res, sizeof(rocke_type_t*));
        if(!rtypes)
        {
            free(ops);
            imp_fail(im, "oom rtypes");
            return;
        }
        for(int i = 0; i < n_res; i++)
        {
            rtypes[i] = import_type(im, jobj_get(results->arr[i], "type"));
            if(im->failed)
            {
                free(ops);
                free(rtypes);
                return;
            }
        }
    }

    rocke_attr_map_t m;
    import_attrs(im, attrs, &m);
    if(im->failed)
    {
        free(ops);
        free(rtypes);
        return;
    }

    /* The smem type node deliberately omits the `exclusive` bit; it rides on
     * the smem_alloc op as an attr. Rebuild the result SmemType with exclusive
     * set so the packer's no-alias behavior round-trips (lower_llvm reads it
     * off the SmemType, not the attr). */
    if(opcode == ROCKE_OP_TILE_SMEM_ALLOC && n_res > 0 && rtypes && rtypes[0]
       && rtypes[0]->kind == ROCKE_TYPE_SMEM && rocke_attr_get_bool(&m, "exclusive", false))
    {
        rtypes[0] = rocke_smem_type(im->b, rtypes[0]->elem, rtypes[0]->shape, rtypes[0]->rank, 1);
        if(!rtypes[0])
        {
            free(ops);
            free(rtypes);
            imp_fail(im, "smem_alloc: exclusive type rebuild failed");
            return;
        }
    }

    rocke_op_t* built = rocke_b_op(im->b,
                                   opcode,
                                   ops,
                                   n_ops,
                                   rtypes,
                                   n_res,
                                   &m,
                                   NULL,
                                   0,
                                   /*result_name_hint=*/NULL,
                                   /*loc=*/NULL);
    free(ops);
    free(rtypes);
    if(!built || !rocke_ir_builder_ok(im->b))
    {
        imp_fail(im,
                 "op '%s' build failed: %s",
                 opcode_name,
                 rocke_ir_builder_ok(im->b) ? "null op" : rocke_ir_builder_error(im->b));
        return;
    }

    for(int i = 0; i < n_res && i < built->num_results; i++)
    {
        const char* rid = jstr(jobj_get(results->arr[i], "id"));
        if(rid)
        {
            rename_value(im, built->results[i], rid);
            vmap_put(im, rid, built->results[i]);
        }
    }
}

static void import_op(importer_t* im, const jval_t* op)
{
    if(im->failed)
        return;
    const char* opcode_name = jstr(jobj_get(op, "opcode"));
    if(!opcode_name)
    {
        imp_fail(im, "op missing opcode");
        return;
    }
    if(strcmp(opcode_name, "scf.for") == 0)
        import_scf_for(im, op);
    else if(strcmp(opcode_name, "scf.if") == 0)
        import_scf_if(im, op);
    else
        import_generic_op(im, op, opcode_name);
}

static void import_region_ops(importer_t* im, const jval_t* region)
{
    const jval_t* ops = jobj_get(region, "ops");
    if(!ops || ops->kind != J_ARR)
    {
        imp_fail(im, "region missing ops array");
        return;
    }
    for(int i = 0; i < ops->arr_len && !im->failed; i++)
        import_op(im, ops->arr[i]);
}

static void import_params(importer_t* im, const jval_t* params)
{
    if(!params)
        return;
    if(params->kind != J_ARR)
    {
        imp_fail(im, "params not an array");
        return;
    }
    for(int i = 0; i < params->arr_len && !im->failed; i++)
    {
        const jval_t* p = params->arr[i];
        const char* name = jstr(jobj_get(p, "name"));
        const rocke_type_t* type = import_type(im, jobj_get(p, "type"));
        if(!name || !type)
        {
            imp_fail(im, "bad param at index %d", i);
            return;
        }
        rocke_param_opts_t opts;
        memset(&opts, 0, sizeof opts);
        const jval_t* pa = jobj_get(p, "attrs");
        if(pa && pa->kind == J_OBJ)
        {
            for(int k = 0; k < pa->obj_len; k++)
            {
                const char* key = pa->obj[k].key;
                const jval_t* v = pa->obj[k].val;
                if(strcmp(key, "noalias") == 0)
                {
                    opts.noalias = v->b;
                    opts.noalias_set = true;
                }
                else if(strcmp(key, "readonly") == 0)
                {
                    opts.readonly = v->b;
                    opts.readonly_set = true;
                }
                else if(strcmp(key, "writeonly") == 0)
                {
                    opts.writeonly = v->b;
                    opts.writeonly_set = true;
                }
                else if(strcmp(key, "align") == 0)
                {
                    opts.align = (int)v->num;
                    opts.align_set = true;
                }
                else if(strcmp(key, "addr_space") == 0)
                {
                    opts.addr_space = v->str;
                }
            }
        }
        rocke_value_t* pv = rocke_b_param(im->b, name, type, &opts);
        if(!pv)
        {
            imp_fail(im, "param '%s' build failed", name);
            return;
        }
        /* Register under the SSA name form ("%name") used by operand refs. */
        char keybuf[256];
        snprintf(keybuf, sizeof keybuf, "%%%s", name);
        /* keybuf is stack-local; persist a copy via the value's own name which
         * the builder set to "%name". Use that pointer instead. */
        vmap_put(im, pv->name, pv);
    }
}

static void import_kernel_attrs(importer_t* im, const jval_t* attrs)
{
    if(!attrs || attrs->kind != J_OBJ)
        return;
    rocke_kernel_def_t* k = rocke_ir_builder_kernel(im->b);
    if(!k)
        return;
    for(int i = 0; i < attrs->obj_len && !im->failed; i++)
    {
        const char* key = attrs->obj[i].key;
        const jval_t* tv = attrs->obj[i].val;
        const char* t = jstr(jobj_get(tv, "t"));
        const jval_t* v = jobj_get(tv, "v");
        if(!t || !v)
            continue;
        if(strcmp(t, "i") == 0)
            rocke_attr_set_int(im->b, &k->attrs, key, (int64_t)v->num);
        else if(strcmp(t, "f") == 0)
            rocke_attr_set_float(im->b, &k->attrs, key, v->num);
        else if(strcmp(t, "b") == 0)
            rocke_attr_set_bool(im->b, &k->attrs, key, v->b);
        else if(strcmp(t, "s") == 0)
            rocke_attr_set_str(im->b, &k->attrs, key, v->str ? v->str : "");
    }
}

rocke_status_t rocke_import_kernel_from_json(const char* text,
                                             const rocke_import_options_t* opts,
                                             rocke_ir_builder_t* out_builder,
                                             rocke_kernel_def_t** out_kernel,
                                             char* err,
                                             size_t err_cap)
{
    if(out_kernel)
        *out_kernel = NULL;
    if(!text || !out_builder)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "null text/builder");
        return ROCKE_ERR_VALUE;
    }

    rocke_arena_t parse_arena;
    if(rocke_arena_init(&parse_arena, 0) != 0)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "parse arena init failed");
        return ROCKE_ERR_OOM;
    }

    char perr[256];
    jval_t* root = json_parse(text, &parse_arena, perr, sizeof perr);
    if(!root)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "parse: %s", perr);
        rocke_arena_destroy(&parse_arena);
        return ROCKE_ERR_VALUE;
    }

    const char* schema = jstr(jobj_get(root, "schema"));
    if(!schema || strcmp(schema, "rocke.ir/v1") != 0)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "bad/missing schema (want rocke.ir/v1)");
        jval_free(root);
        rocke_arena_destroy(&parse_arena);
        return ROCKE_ERR_VALUE;
    }

    const jval_t* kernel = jobj_get(root, "kernel");
    const char* kname = jstr(jobj_get(kernel, "name"));
    if(!kernel || !kname)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "missing kernel/name");
        jval_free(root);
        rocke_arena_destroy(&parse_arena);
        return ROCKE_ERR_VALUE;
    }
    if(opts && opts->expected_kernel_name && strcmp(opts->expected_kernel_name, kname) != 0)
    {
        if(err && err_cap)
            snprintf(err,
                     err_cap,
                     "kernel name '%s' != expected '%s'",
                     kname,
                     opts->expected_kernel_name);
        jval_free(root);
        rocke_arena_destroy(&parse_arena);
        return ROCKE_ERR_VALUE;
    }

    rocke_status_t st = rocke_ir_builder_init(out_builder, kname);
    if(st != ROCKE_OK)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "builder init failed (%d)", (int)st);
        jval_free(root);
        rocke_arena_destroy(&parse_arena);
        return st;
    }

    importer_t im;
    memset(&im, 0, sizeof im);
    im.b = out_builder;

    import_kernel_attrs(&im, jobj_get(kernel, "attrs"));
    import_params(&im, jobj_get(kernel, "params"));
    if(!im.failed)
    {
        const jval_t* body = jobj_get(kernel, "body");
        if(!body)
            imp_fail(&im, "kernel missing body");
        else
            import_region_ops(&im, body);
    }

    free(im.binds);
    free(im.slots);
    free(im.buckets);
    jval_free(root);
    rocke_arena_destroy(&parse_arena);

    if(im.failed)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "%s", im.err);
        rocke_ir_builder_free(out_builder);
        return ROCKE_ERR_VALUE;
    }
    if(!rocke_ir_builder_ok(out_builder))
    {
        if(err && err_cap)
            snprintf(err, err_cap, "builder error: %s", rocke_ir_builder_error(out_builder));
        rocke_ir_builder_free(out_builder);
        return rocke_ir_builder_status(out_builder);
    }

    *out_kernel = rocke_ir_builder_kernel(out_builder);
    return ROCKE_OK;
}
