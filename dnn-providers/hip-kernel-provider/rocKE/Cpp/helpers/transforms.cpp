// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.transforms.c -- C99 port of a SUBSET of
 * rocke.helpers.transforms (the CK Tile coordinate-transform DAG).
 *
 * Ported symbols (see the header for the exact list): calculate_magic_numbers,
 * do_magic_division, CoordVar, Embed/PassThrough/Unmerge/UnmergeMagicDiv
 * transforms + their constructors (embed/pass_through/unmerge/unmerge_magic),
 * and TensorDescriptor.{naive,transform,offset,unmerge_lower}.
 *
 * The builder-call sequence in every emitting function is byte-faithful to the
 * Python so the downstream IR op stream is identical.
 */

#include "rocke/helper_rocke.helpers.transforms.h"

#include <string.h>

#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live */

/* ====================================================================== */
/* Small i1-predicate / compare helpers (Python _and / _ge / _lt).         */
/* ====================================================================== */

/* Python _and(b, p, q): conjunction of two optional i1 predicates.
 *   if p is None: return q
 *   if q is None: return p
 *   return b.land(p, q)
 */
static rocke_value_t* rocke_i_and(rocke_ir_builder_t* b, rocke_value_t* p, rocke_value_t* q)
{
    if(p == NULL)
    {
        return q;
    }
    if(q == NULL)
    {
        return p;
    }
    return rocke_b_land(b, p, q);
}

/* Python _ge(b, lhs, rhs): signed lhs >= rhs -> i1. */
static rocke_value_t* rocke_i_ge(rocke_ir_builder_t* b, rocke_value_t* lhs, rocke_value_t* rhs)
{
    return rocke_b_cmp_ge(b, lhs, rhs);
}

/* Python _lt(b, lhs, rhs): signed lhs < rhs -> i1. */
static rocke_value_t* rocke_i_lt(rocke_ir_builder_t* b, rocke_value_t* lhs, rocke_value_t* rhs)
{
    return rocke_b_cmp_lt(b, lhs, rhs);
}

/* ====================================================================== */
/* Magic-number division.                                                  */
/* ====================================================================== */

bool rocke_calculate_magic_numbers(rocke_ir_builder_t* b,
                                   int divisor,
                                   uint64_t* out_multiplier,
                                   int* out_shift)
{
    int shift;
    uint64_t multiplier;

    /* Python: if divisor < 1: raise ValueError(...) */
    if(divisor < 1)
    {
        if(b != NULL)
        {
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "magic division requires divisor >= 1, got %d", divisor);
        }
        return false;
    }

    /* shift = smallest s with (1 << s) >= divisor */
    shift = 0;
    while(((uint64_t)1 << shift) < (uint64_t)divisor)
    {
        shift += 1;
    }

    /* multiplier = (((1 << shift) - divisor) << 32) // divisor + 1.
     * Computed in 64-bit unsigned to match the Python arbitrary-precision int
     * for the documented 31-bit range; the bit pattern is what matters. */
    multiplier = (((((uint64_t)1 << shift) - (uint64_t)divisor) << 32) / (uint64_t)divisor) + 1u;

    if(out_multiplier != NULL)
    {
        *out_multiplier = multiplier;
    }
    if(out_shift != NULL)
    {
        *out_shift = shift;
    }
    return true;
}

rocke_value_t* rocke_do_magic_division(rocke_ir_builder_t* b,
                                       rocke_value_t* dividend,
                                       uint64_t multiplier,
                                       int shift)
{
    int64_t mult_i32;
    rocke_value_t* tmp;
    rocke_value_t* summed;

    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* Python:
     *   mult_i32 = multiplier - (1 << 32) if multiplier >= (1 << 31) else multiplier
     * Bake the uint32 magic as its two's-complement i32 bit pattern. */
    if(multiplier >= ((uint64_t)1 << 31))
    {
        mult_i32 = (int64_t)multiplier - ((int64_t)1 << 32);
    }
    else
    {
        mult_i32 = (int64_t)multiplier;
    }

    tmp = rocke_b_umul_hi_i32(b, dividend, rocke_b_const_i32(b, mult_i32));
    summed = rocke_b_add(b, tmp, dividend);
    if(shift == 0)
    {
        return summed;
    }
    return rocke_b_lshr(b, summed, rocke_b_const_i32(b, (int64_t)shift));
}

/* ====================================================================== */
/* Coord map: a small ordered (name -> CoordVar) association list.         */
/* The Python uses a dict; we use an insertion-ordered array. Lookups and  */
/* "name in coords" both scan by string equality, matching dict semantics. */
/* (Re-inserting an existing name overwrites in place, like dict[name]=v.)  */
/* ====================================================================== */

typedef struct rocke_i_coord_map
{
    rocke_coord_var_t* items;
    int count;
    int cap;
} rocke_i_coord_map_t;

static bool rocke_i_map_init(rocke_ir_builder_t* b, rocke_i_coord_map_t* m, int cap)
{
    if(cap < 1)
    {
        cap = 1;
    }
    m->items
        = (rocke_coord_var_t*)rocke_arena_alloc(&b->arena, (size_t)cap * sizeof(rocke_coord_var_t));
    if(m->items == NULL)
    {
        return false;
    }
    m->count = 0;
    m->cap = cap;
    return true;
}

static int rocke_i_map_find(const rocke_i_coord_map_t* m, const char* name)
{
    int i;
    for(i = 0; i < m->count; ++i)
    {
        if(strcmp(m->items[i].name, name) == 0)
        {
            return i;
        }
    }
    return -1;
}

static bool rocke_i_map_has(const rocke_i_coord_map_t* m, const char* name)
{
    return rocke_i_map_find(m, name) >= 0;
}

/* dict-style set: overwrite if present, else append (grow if needed). */
static bool rocke_i_map_set(rocke_ir_builder_t* b, rocke_i_coord_map_t* m, rocke_coord_var_t cv)
{
    int idx = rocke_i_map_find(m, cv.name);
    if(idx >= 0)
    {
        m->items[idx] = cv;
        return true;
    }
    if(m->count == m->cap)
    {
        int new_cap = m->cap * 2;
        rocke_coord_var_t* grown = (rocke_coord_var_t*)rocke_arena_alloc(
            &b->arena, (size_t)new_cap * sizeof(rocke_coord_var_t));
        if(grown == NULL)
        {
            return false;
        }
        memcpy(grown, m->items, (size_t)m->count * sizeof(rocke_coord_var_t));
        m->items = grown;
        m->cap = new_cap;
    }
    m->items[m->count] = cv;
    m->count += 1;
    return true;
}

static const rocke_coord_var_t* rocke_i_map_get(const rocke_i_coord_map_t* m, const char* name)
{
    int idx = rocke_i_map_find(m, name);
    if(idx < 0)
    {
        return NULL;
    }
    return &m->items[idx];
}

/* ====================================================================== */
/* Transform constructors.                                                 */
/* ====================================================================== */

/* Duplicate an array of name strings into the arena (each string dup'd too). */
static const char* const* rocke_i_dup_names(rocke_ir_builder_t* b, const char* const* names, int n)
{
    const char** out;
    int i;
    if(n <= 0)
    {
        return NULL;
    }
    out = (const char**)rocke_arena_alloc(&b->arena, (size_t)n * sizeof(const char*));
    if(out == NULL)
    {
        return NULL;
    }
    for(i = 0; i < n; ++i)
    {
        out[i] = rocke_arena_strdup(&b->arena, names[i]);
        if(out[i] == NULL)
        {
            return NULL;
        }
    }
    return (const char* const*)out;
}

/* Single-element name array {name}. */
static const char* const* rocke_i_dup_name1(rocke_ir_builder_t* b, const char* name)
{
    return rocke_i_dup_names(b, &name, 1);
}

static const int* rocke_i_dup_ints(rocke_ir_builder_t* b, const int* src, int n)
{
    int* out;
    if(n <= 0)
    {
        return NULL;
    }
    out = (int*)rocke_arena_alloc(&b->arena, (size_t)n * sizeof(int));
    if(out == NULL)
    {
        return NULL;
    }
    memcpy(out, src, (size_t)n * sizeof(int));
    return out;
}

static rocke_transform_t* rocke_i_new_transform(rocke_ir_builder_t* b)
{
    rocke_transform_t* t
        = (rocke_transform_t*)rocke_arena_calloc(&b->arena, sizeof(rocke_transform_t));
    return t;
}

rocke_transform_t* rocke_pass_through(rocke_ir_builder_t* b, const char* coord, const char* into)
{
    rocke_transform_t* t;
    const char* lower_name;

    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* Python: lower = (lower_name or upper_name,) */
    lower_name = (into != NULL) ? into : coord;

    t = rocke_i_new_transform(b);
    if(t == NULL)
    {
        return NULL;
    }
    t->kind = ROCKE_XFORM_PASS_THROUGH;
    t->upper = rocke_i_dup_name1(b, coord);
    t->n_upper = 1;
    t->lower = rocke_i_dup_name1(b, lower_name);
    t->n_lower = 1;
    if(t->upper == NULL || t->lower == NULL)
    {
        return NULL;
    }
    return t;
}

rocke_transform_t* rocke_embed_bounded(rocke_ir_builder_t* b,
                                       const char* const* upper,
                                       int n_upper,
                                       const char* into,
                                       const int* strides,
                                       int offset,
                                       int lo,
                                       int hi)
{
    rocke_transform_t* t;

    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* Python: if len(upper) != len(strides): raise ValueError(...) */
    /* (strides count == n_upper by this API; guard the obvious misuse.) */
    if(n_upper < 0)
    {
        return (rocke_transform_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "Embed expects len(upper) == len(strides)");
    }

    t = rocke_i_new_transform(b);
    if(t == NULL)
    {
        return NULL;
    }
    t->kind = ROCKE_XFORM_EMBED;
    t->upper = rocke_i_dup_names(b, upper, n_upper);
    t->n_upper = n_upper;
    t->lower = rocke_i_dup_name1(b, into);
    t->n_lower = 1;
    t->strides = rocke_i_dup_ints(b, strides, n_upper);
    t->offset = offset;
    t->lo = lo;
    t->hi = hi;
    if(t->lower == NULL || (n_upper > 0 && (t->upper == NULL || t->strides == NULL)))
    {
        return NULL;
    }
    return t;
}

rocke_transform_t* rocke_embed(rocke_ir_builder_t* b,
                               const char* const* upper,
                               int n_upper,
                               const char* into,
                               const int* strides,
                               int offset)
{
    /* Python None-sentinels: lo=-(1<<30), hi=(1<<30). */
    return rocke_embed_bounded(b, upper, n_upper, into, strides, offset, -(1 << 30), (1 << 30));
}

static rocke_transform_t* rocke_i_new_unmerge(rocke_ir_builder_t* b,
                                              rocke_xform_kind_t kind,
                                              const char* upper,
                                              const char* const* into,
                                              int n_lower,
                                              const int* dims,
                                              const char* who)
{
    rocke_transform_t* t;

    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* Python: if len(lowers) != len(dims): raise ValueError(...) */
    if(n_lower < 0)
    {
        return (rocke_transform_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "%s expects len(lowers) == len(dims)", who);
    }

    t = rocke_i_new_transform(b);
    if(t == NULL)
    {
        return NULL;
    }
    t->kind = kind;
    t->upper = rocke_i_dup_name1(b, upper);
    t->n_upper = 1;
    t->lower = rocke_i_dup_names(b, into, n_lower);
    t->n_lower = n_lower;
    t->dims = rocke_i_dup_ints(b, dims, n_lower);
    if(t->upper == NULL || (n_lower > 0 && (t->lower == NULL || t->dims == NULL)))
    {
        return NULL;
    }
    return t;
}

rocke_transform_t* rocke_unmerge(
    rocke_ir_builder_t* b, const char* upper, const char* const* into, int n_lower, const int* dims)
{
    return rocke_i_new_unmerge(b, ROCKE_XFORM_UNMERGE, upper, into, n_lower, dims, "Unmerge");
}

rocke_transform_t* rocke_unmerge_magic(
    rocke_ir_builder_t* b, const char* upper, const char* const* into, int n_lower, const int* dims)
{
    return rocke_i_new_unmerge(
        b, ROCKE_XFORM_UNMERGE_MAGIC, upper, into, n_lower, dims, "UnmergeMagicDiv");
}

rocke_transform_t* rocke_pad(rocke_ir_builder_t* b, const char* coord, int lo, int hi)
{
    rocke_transform_t* t;

    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* Python __init__: upper == lower == (coord_name,); lo/hi int. */
    t = rocke_i_new_transform(b);
    if(t == NULL)
    {
        return NULL;
    }
    t->kind = ROCKE_XFORM_PAD;
    t->upper = rocke_i_dup_name1(b, coord);
    t->n_upper = 1;
    t->lower = rocke_i_dup_name1(b, coord);
    t->n_lower = 1;
    t->lo = lo;
    t->hi = hi;
    if(t->upper == NULL || t->lower == NULL)
    {
        return NULL;
    }
    return t;
}

rocke_transform_t* rocke_indirect(rocke_ir_builder_t* b,
                                  const char* upper,
                                  const char* into,
                                  rocke_value_t* table,
                                  rocke_value_t* base,
                                  rocke_value_t* max_idx,
                                  int default_value)
{
    rocke_transform_t* t;

    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* Python __init__: upper == (upper_name,); lower == (into,). */
    t = rocke_i_new_transform(b);
    if(t == NULL)
    {
        return NULL;
    }
    t->kind = ROCKE_XFORM_INDIRECT;
    t->upper = rocke_i_dup_name1(b, upper);
    t->n_upper = 1;
    t->lower = rocke_i_dup_name1(b, into);
    t->n_lower = 1;
    t->table = table;
    t->base = base;
    t->max_idx = max_idx;
    t->default_value = default_value;
    if(t->upper == NULL || t->lower == NULL)
    {
        return NULL;
    }
    return t;
}

/* ====================================================================== */
/* Transform.apply -- emit lowers from uppers for one transform.           */
/* Writes produced CoordVars into `out` (dict-style set). Returns false on  */
/* builder failure. Each branch reproduces the Python apply() op order.     */
/* ====================================================================== */

static bool rocke_i_apply_pass_through(rocke_ir_builder_t* b,
                                       const rocke_transform_t* t,
                                       const rocke_i_coord_map_t* coords,
                                       rocke_i_coord_map_t* out)
{
    /* Python: u = coords[upper[0]]; return {lower[0]: replace(u, name=lower[0])} */
    const rocke_coord_var_t* u = rocke_i_map_get(coords, t->upper[0]);
    rocke_coord_var_t cv;
    cv.name = t->lower[0];
    cv.value = u->value;
    cv.valid = u->valid;
    return rocke_i_map_set(b, out, cv);
}

static bool rocke_i_apply_embed(rocke_ir_builder_t* b,
                                const rocke_transform_t* t,
                                const rocke_i_coord_map_t* coords,
                                rocke_i_coord_map_t* out)
{
    /* Python apply():
     *   acc = None; valid_acc = None
     *   for name, s in zip(upper, strides):
     *       u = coords[name]; valid_acc = _and(b, valid_acc, u.valid)
     *       term = u.value if s == 1 else b.mul(u.value, b.const_i32(s))
     *       acc = term if acc is None else b.add(acc, term)
     *   if offset != 0: acc = b.add(acc, b.const_i32(offset))
     *   if acc is None: acc = b.const_i32(offset)
     *   bounds = _and(b, _ge(acc, lo), _lt(acc, hi))
     *   valid = _and(b, valid_acc, bounds)
     */
    rocke_value_t* acc = NULL;
    rocke_value_t* valid_acc = NULL;
    rocke_value_t* bounds;
    rocke_value_t* valid;
    rocke_coord_var_t cv;
    int i;

    for(i = 0; i < t->n_upper; ++i)
    {
        const rocke_coord_var_t* u = rocke_i_map_get(coords, t->upper[i]);
        int s = t->strides[i];
        rocke_value_t* term;
        valid_acc = rocke_i_and(b, valid_acc, u->valid);
        if(s == 1)
        {
            term = u->value;
        }
        else
        {
            term = rocke_b_mul(b, u->value, rocke_b_const_i32(b, (int64_t)s));
        }
        acc = (acc == NULL) ? term : rocke_b_add(b, acc, term);
    }
    if(t->offset != 0)
    {
        acc = rocke_b_add(b, acc, rocke_b_const_i32(b, (int64_t)t->offset));
    }
    if(acc == NULL)
    {
        acc = rocke_b_const_i32(b, (int64_t)t->offset);
    }
    /* bounds: lo <= acc < hi (the inner _ge is evaluated before _lt). */
    {
        rocke_value_t* ge = rocke_i_ge(b, acc, rocke_b_const_i32(b, (int64_t)t->lo));
        rocke_value_t* lt = rocke_i_lt(b, acc, rocke_b_const_i32(b, (int64_t)t->hi));
        bounds = rocke_i_and(b, ge, lt);
    }
    valid = rocke_i_and(b, valid_acc, bounds);

    cv.name = t->lower[0];
    cv.value = acc;
    cv.valid = valid;
    return rocke_i_map_set(b, out, cv);
}

static bool rocke_i_apply_unmerge(rocke_ir_builder_t* b,
                                  const rocke_transform_t* t,
                                  const rocke_i_coord_map_t* coords,
                                  rocke_i_coord_map_t* out)
{
    /* Python apply():
     *   u = coords[upper[0]]
     *   for i, name in enumerate(lower):
     *       stride = product(dims[i+1:])
     *       quot = u.value if stride == 1 else b.div(u.value, b.const_i32(stride))
     *       val  = quot if i == 0 else b.mod(quot, b.const_i32(dims[i]))
     *       out[name] = CoordVar(name, val, u.valid)
     */
    const rocke_coord_var_t* u = rocke_i_map_get(coords, t->upper[0]);
    /* Cache u's fields: rocke_i_map_set on the shared map may relocate items. */
    rocke_value_t* u_value = u->value;
    rocke_value_t* u_valid = u->valid;
    int i, j;

    for(i = 0; i < t->n_lower; ++i)
    {
        int stride = 1;
        rocke_value_t* quot;
        rocke_value_t* val;
        rocke_coord_var_t cv;
        for(j = i + 1; j < t->n_lower; ++j)
        {
            stride *= t->dims[j];
        }
        if(stride == 1)
        {
            quot = u_value;
        }
        else
        {
            quot = rocke_b_div(b, u_value, rocke_b_const_i32(b, (int64_t)stride));
        }
        if(i == 0)
        {
            val = quot;
        }
        else
        {
            val = rocke_b_mod(b, quot, rocke_b_const_i32(b, (int64_t)t->dims[i]));
        }
        cv.name = t->lower[i];
        cv.value = val;
        cv.valid = u_valid;
        if(!rocke_i_map_set(b, out, cv))
        {
            return false;
        }
    }
    return true;
}

static bool rocke_i_apply_unmerge_magic(rocke_ir_builder_t* b,
                                        const rocke_transform_t* t,
                                        const rocke_i_coord_map_t* coords,
                                        rocke_i_coord_map_t* out)
{
    /* Python apply():
     *   u = coords[upper[0]]; n = len(lower); tmp = u.value
     *   for i in range(n-1, 0, -1):
     *       d = dims[i]
     *       if d == 1: rem = b.const_i32(0); quot = tmp
     *       else:
     *           mult, shift = calculate_magic_numbers(d)
     *           quot = do_magic_division(b, tmp, mult, shift)
     *           rem  = b.sub(tmp, b.mul(quot, b.const_i32(d)))
     *       out[lower[i]] = CoordVar(lower[i], rem, u.valid)
     *       tmp = quot
     *   out[lower[0]] = CoordVar(lower[0], tmp, u.valid)
     */
    const rocke_coord_var_t* u = rocke_i_map_get(coords, t->upper[0]);
    /* Cache u's fields: rocke_i_map_set on the shared map may relocate items. */
    rocke_value_t* u_valid = u->valid;
    int n = t->n_lower;
    rocke_value_t* tmp = u->value;
    int i;
    rocke_coord_var_t cv;

    for(i = n - 1; i > 0; --i)
    {
        int d = t->dims[i];
        rocke_value_t* rem;
        rocke_value_t* quot;
        if(d == 1)
        {
            rem = rocke_b_const_i32(b, 0);
            quot = tmp;
        }
        else
        {
            uint64_t mult;
            int shift;
            if(!rocke_calculate_magic_numbers(b, d, &mult, &shift))
            {
                return false;
            }
            quot = rocke_do_magic_division(b, tmp, mult, shift);
            rem = rocke_b_sub(b, tmp, rocke_b_mul(b, quot, rocke_b_const_i32(b, (int64_t)d)));
        }
        cv.name = t->lower[i];
        cv.value = rem;
        cv.valid = u_valid;
        if(!rocke_i_map_set(b, out, cv))
        {
            return false;
        }
        tmp = quot;
    }
    cv.name = t->lower[0];
    cv.value = tmp;
    cv.valid = u_valid;
    return rocke_i_map_set(b, out, cv);
}

static bool rocke_i_apply_pad(rocke_ir_builder_t* b,
                              const rocke_transform_t* t,
                              const rocke_i_coord_map_t* coords,
                              rocke_i_coord_map_t* out)
{
    /* Python apply():
     *   u = coords[upper[0]]
     *   c_lo = b.const_i32(lo); c_hi = b.const_i32(hi)
     *   valid = _and(b, _ge(b, u.value, c_lo), _lt(b, u.value, c_hi))
     *   merged_valid = _and(b, u.valid, valid)
     *   return {lower[0]: CoordVar(lower[0], u.value, merged_valid)}
     */
    const rocke_coord_var_t* u = rocke_i_map_get(coords, t->upper[0]);
    rocke_value_t* u_value = u->value;
    rocke_value_t* u_valid = u->valid;
    rocke_value_t* c_lo = rocke_b_const_i32(b, (int64_t)t->lo);
    rocke_value_t* c_hi = rocke_b_const_i32(b, (int64_t)t->hi);
    rocke_value_t* ge = rocke_i_ge(b, u_value, c_lo);
    rocke_value_t* lt = rocke_i_lt(b, u_value, c_hi);
    rocke_value_t* valid = rocke_i_and(b, ge, lt);
    rocke_value_t* merged = rocke_i_and(b, u_valid, valid);
    rocke_coord_var_t cv;
    cv.name = t->lower[0];
    cv.value = u_value;
    cv.valid = merged;
    return rocke_i_map_set(b, out, cv);
}

static bool rocke_i_apply_indirect(rocke_ir_builder_t* b,
                                   const rocke_transform_t* t,
                                   const rocke_i_coord_map_t* coords,
                                   rocke_i_coord_map_t* out)
{
    /* Python apply():
     *   u   = coords[upper[0]]
     *   idx = b.add(base, u.value)
     *   if max_idx is None:
     *       physical = b.global_load_i32(table, idx)
     *   else:
     *       mask     = b.cmp_lt(idx, max_idx)
     *       physical = b.masked_global_load(table, idx, mask,
     *                                       b.const_i32(default_value),
     *                                       dtype=I32, align=4)
     *   return {lower[0]: CoordVar(lower[0], physical, u.valid)}
     */
    const rocke_coord_var_t* u = rocke_i_map_get(coords, t->upper[0]);
    rocke_value_t* u_valid = u->valid;
    rocke_value_t* idx = rocke_b_add(b, t->base, u->value);
    rocke_value_t* physical;
    rocke_coord_var_t cv;

    if(t->max_idx == NULL)
    {
        physical = rocke_b_global_load_i32(b, t->table, idx, 4);
    }
    else
    {
        rocke_value_t* mask = rocke_i_lt(b, idx, t->max_idx);
        physical = rocke_b_masked_global_load(b,
                                              t->table,
                                              idx,
                                              mask,
                                              rocke_b_const_i32(b, (int64_t)t->default_value),
                                              rocke_i32(),
                                              4);
    }
    cv.name = t->lower[0];
    cv.value = physical;
    cv.valid = u_valid;
    return rocke_i_map_set(b, out, cv);
}

/* Dispatch one transform's apply onto the coord map (in place). */
static bool rocke_i_transform_apply(rocke_ir_builder_t* b,
                                    const rocke_transform_t* t,
                                    rocke_i_coord_map_t* coords)
{
    switch(t->kind)
    {
    case ROCKE_XFORM_PASS_THROUGH:
        return rocke_i_apply_pass_through(b, t, coords, coords);
    case ROCKE_XFORM_EMBED:
        return rocke_i_apply_embed(b, t, coords, coords);
    case ROCKE_XFORM_UNMERGE:
        return rocke_i_apply_unmerge(b, t, coords, coords);
    case ROCKE_XFORM_UNMERGE_MAGIC:
        return rocke_i_apply_unmerge_magic(b, t, coords, coords);
    case ROCKE_XFORM_PAD:
        return rocke_i_apply_pad(b, t, coords, coords);
    case ROCKE_XFORM_INDIRECT:
        return rocke_i_apply_indirect(b, t, coords, coords);
    default:
        return false;
    }
}

/* All of a transform's uppers present in the coord map? */
static bool rocke_i_uppers_ready(const rocke_transform_t* t, const rocke_i_coord_map_t* coords)
{
    int i;
    for(i = 0; i < t->n_upper; ++i)
    {
        if(!rocke_i_map_has(coords, t->upper[i]))
        {
            return false;
        }
    }
    return true;
}

/* ====================================================================== */
/* TensorDescriptor.naive                                                  */
/* ====================================================================== */

rocke_tensor_descriptor_t* rocke_tensor_descriptor_naive(rocke_ir_builder_t* b,
                                                         const char* name,
                                                         const int* lengths,
                                                         int n_lengths,
                                                         const int* strides,
                                                         const char* const* coord_names,
                                                         int n_coord_names)
{
    rocke_tensor_descriptor_t* d;
    const int* base_lengths;
    int* base_strides;
    const char* const* base_names;

    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* Python: if not lengths: raise ValueError("naive descriptor needs ...") */
    if(n_lengths < 1 || lengths == NULL)
    {
        return (rocke_tensor_descriptor_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "naive descriptor needs at least one dim");
    }

    base_lengths = rocke_i_dup_ints(b, lengths, n_lengths);
    if(base_lengths == NULL)
    {
        return NULL;
    }

    /* strides: row-major when not supplied.
     *   ss = [1]; for d in reversed(lengths[1:]): ss.insert(0, ss[0]*d)
     *   strides = ss[:len(lengths)]
     * Concretely strides[i] = product(lengths[i+1:]), strides[last] = 1. */
    base_strides = (int*)rocke_arena_alloc(&b->arena, (size_t)n_lengths * sizeof(int));
    if(base_strides == NULL)
    {
        return NULL;
    }
    if(strides == NULL)
    {
        int i;
        int acc = 1;
        base_strides[n_lengths - 1] = 1;
        for(i = n_lengths - 1; i >= 1; --i)
        {
            acc *= lengths[i];
            base_strides[i - 1] = acc;
        }
    }
    else
    {
        memcpy(base_strides, strides, (size_t)n_lengths * sizeof(int));
    }

    /* coord_names: default ("d0", "d1", ...). */
    if(coord_names == NULL)
    {
        const char** names
            = (const char**)rocke_arena_alloc(&b->arena, (size_t)n_lengths * sizeof(const char*));
        int i;
        if(names == NULL)
        {
            return NULL;
        }
        for(i = 0; i < n_lengths; ++i)
        {
            names[i] = rocke_arena_printf(&b->arena, "d%d", i);
            if(names[i] == NULL)
            {
                return NULL;
            }
        }
        base_names = (const char* const*)names;
    }
    else
    {
        /* Python: if len(coord_names) != len(lengths): raise ValueError(...) */
        if(n_coord_names != n_lengths)
        {
            return (rocke_tensor_descriptor_t*)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "coord_names length mismatch");
        }
        base_names = rocke_i_dup_names(b, coord_names, n_lengths);
        if(base_names == NULL)
        {
            return NULL;
        }
    }

    d = (rocke_tensor_descriptor_t*)rocke_arena_calloc(&b->arena,
                                                       sizeof(rocke_tensor_descriptor_t));
    if(d == NULL)
    {
        return NULL;
    }
    d->name = rocke_arena_strdup(&b->arena, name);
    d->base_names = base_names;
    d->base_lengths = base_lengths;
    d->base_strides = base_strides;
    d->n_base = n_lengths;
    d->chain = NULL;
    d->n_chain = 0;
    /* upper_names = coord_names (the naive coords are all user-facing). */
    d->upper_names = base_names;
    d->n_upper = n_lengths;
    if(d->name == NULL)
    {
        return NULL;
    }
    return d;
}

/* ====================================================================== */
/* TensorDescriptor.transform                                              */
/* ====================================================================== */

/* Name-membership in a name array. */
static bool rocke_i_name_in(const char* const* arr, int n, const char* name)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        if(strcmp(arr[i], name) == 0)
        {
            return true;
        }
    }
    return false;
}

/* Is `name` a lower of any transform in the chain? (the subtraction set). */
static bool
    rocke_i_is_lower_of_any(const rocke_transform_t* const* chain, int n_chain, const char* name)
{
    int ci, li;
    for(ci = 0; ci < n_chain; ++ci)
    {
        for(li = 0; li < chain[ci]->n_lower; ++li)
        {
            if(strcmp(chain[ci]->lower[li], name) == 0)
            {
                return true;
            }
        }
    }
    return false;
}

rocke_tensor_descriptor_t*
    rocke_tensor_descriptor_transform(rocke_ir_builder_t* b,
                                      const rocke_tensor_descriptor_t* desc,
                                      const rocke_transform_t* const* transforms,
                                      int n_transforms)
{
    rocke_tensor_descriptor_t* d;
    const rocke_transform_t** new_chain;
    int new_n_chain;
    int ti, k;

    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* Python: if not transforms: return self */
    if(n_transforms <= 0)
    {
        return (rocke_tensor_descriptor_t*)desc;
    }

    new_n_chain = desc->n_chain + n_transforms;
    new_chain = (const rocke_transform_t**)rocke_arena_alloc(
        &b->arena, (size_t)new_n_chain * sizeof(const rocke_transform_t*));
    if(new_chain == NULL)
    {
        return NULL;
    }
    for(k = 0; k < desc->n_chain; ++k)
    {
        new_chain[k] = desc->chain[k];
    }
    for(ti = 0; ti < n_transforms; ++ti)
    {
        new_chain[desc->n_chain + ti] = transforms[ti];
    }

    /* upper_set = (base_names | all_uppers) - all_lowers, then ordered:
     *   base_names first (kept if in upper_set), then transform uppers in
     *   appearance order (kept if in upper_set and not yet seen).
     * We compute membership in upper_set by:
     *   in_upper_set(name) := (name in base_names OR name in any t.upper)
     *                          AND name not in any t.lower.
     * The ordered walk visits exactly the candidate names, so checking
     * "is this a lower of any transform" suffices for the subtraction. */
    {
        /* Upper bound on result count = n_base + sum(n_upper). */
        int cap = desc->n_base;
        const char** ordered;
        int n_ordered = 0;

        for(k = 0; k < new_n_chain; ++k)
        {
            cap += new_chain[k]->n_upper;
        }
        if(cap < 1)
        {
            cap = 1;
        }
        ordered = (const char**)rocke_arena_alloc(&b->arena, (size_t)cap * sizeof(const char*));
        if(ordered == NULL)
        {
            return NULL;
        }

        const rocke_transform_t* const* chain_view = (const rocke_transform_t* const*)new_chain;

        /* base_names first. */
        for(k = 0; k < desc->n_base; ++k)
        {
            const char* nm = desc->base_names[k];
            if(!rocke_i_is_lower_of_any(chain_view, new_n_chain, nm)
               && !rocke_i_name_in(ordered, n_ordered, nm))
            {
                ordered[n_ordered++] = nm;
            }
        }
        /* then transform uppers in appearance order. */
        for(k = 0; k < new_n_chain; ++k)
        {
            int u;
            for(u = 0; u < new_chain[k]->n_upper; ++u)
            {
                const char* nm = new_chain[k]->upper[u];
                if(!rocke_i_is_lower_of_any(chain_view, new_n_chain, nm)
                   && !rocke_i_name_in(ordered, n_ordered, nm))
                {
                    ordered[n_ordered++] = nm;
                }
            }
        }

        d = (rocke_tensor_descriptor_t*)rocke_arena_calloc(&b->arena,
                                                           sizeof(rocke_tensor_descriptor_t));
        if(d == NULL)
        {
            return NULL;
        }
        /* replace(self, chain=new_chain, upper_names=tuple(ordered)) -- all
         * other fields copied verbatim from desc (they share arena storage). */
        d->name = desc->name;
        d->base_names = desc->base_names;
        d->base_lengths = desc->base_lengths;
        d->base_strides = desc->base_strides;
        d->n_base = desc->n_base;
        d->chain = (const rocke_transform_t* const*)new_chain;
        d->n_chain = new_n_chain;
        d->upper_names = (const char* const*)ordered;
        d->n_upper = n_ordered;
    }
    return d;
}

/* ====================================================================== */
/* Topological chain runner shared by unmerge_lower / offset.              */
/* ====================================================================== */

/* Run the chain over `coords`, resolving applicable transforms until either
 * all are consumed (success) or no progress is made.
 *
 * `require_all` selects the two Python behaviours:
 *   unmerge_lower: break on no-progress (partial result OK) -> require_all=false
 *   _run_chain   : raise on no-progress (unresolved deps)   -> require_all=true
 *
 * Returns 1 on full resolution, 0 on a clean partial stop (require_all=false),
 * -1 on error (builder failure, or unresolved deps when require_all=true). */
static int rocke_i_run_chain(rocke_ir_builder_t* b,
                             const rocke_tensor_descriptor_t* desc,
                             rocke_i_coord_map_t* coords,
                             bool require_all)
{
    /* `remaining` is the worklist of not-yet-applied transforms. */
    const rocke_transform_t** remaining;
    int n_remaining;
    int i;

    if(desc->n_chain == 0)
    {
        return 1;
    }
    remaining = (const rocke_transform_t**)rocke_arena_alloc(
        &b->arena, (size_t)desc->n_chain * sizeof(const rocke_transform_t*));
    if(remaining == NULL)
    {
        return -1;
    }
    n_remaining = desc->n_chain;
    for(i = 0; i < desc->n_chain; ++i)
    {
        remaining[i] = desc->chain[i];
    }

    while(n_remaining > 0)
    {
        bool progress = false;
        int next_n = 0;
        int j;
        for(j = 0; j < n_remaining; ++j)
        {
            const rocke_transform_t* t = remaining[j];
            if(rocke_i_uppers_ready(t, coords))
            {
                if(!rocke_i_transform_apply(b, t, coords))
                {
                    return -1;
                }
                progress = true;
            }
            else
            {
                remaining[next_n++] = t; /* compact in place (order preserved) */
            }
        }
        n_remaining = next_n;
        if(!progress)
        {
            if(require_all)
            {
                rocke_i_set_err(b,
                                ROCKE_ERR_VALUE,
                                "transform chain has unresolved deps (descriptor %s)",
                                desc->name ? desc->name : "");
                return -1;
            }
            /* Python unmerge_lower: break on no progress, keep partial map. */
            return 0;
        }
    }
    return 1;
}

/* ====================================================================== */
/* TensorDescriptor.unmerge_lower                                          */
/* ====================================================================== */

int rocke_tensor_descriptor_unmerge_lower(rocke_ir_builder_t* b,
                                          const rocke_tensor_descriptor_t* desc,
                                          const char* const* in_names,
                                          rocke_value_t* const* in_values,
                                          int n_in,
                                          const char** out_names,
                                          rocke_value_t** out_values,
                                          int out_cap)
{
    rocke_i_coord_map_t coords;
    int cap;
    int i;
    int r;

    if(!rocke_i_live(b))
    {
        return -1;
    }

    /* Pre-size the map generously: inputs + every transform's lowers. */
    cap = n_in;
    for(i = 0; i < desc->n_chain; ++i)
    {
        cap += desc->chain[i]->n_lower;
    }
    if(!rocke_i_map_init(b, &coords, cap > 0 ? cap : 1))
    {
        return -1;
    }

    /* Seed with the supplied upper coords (valid omitted -> NULL/None). */
    for(i = 0; i < n_in; ++i)
    {
        rocke_coord_var_t cv;
        cv.name = in_names[i];
        cv.value = in_values[i];
        cv.valid = NULL;
        if(!rocke_i_map_set(b, &coords, cv))
        {
            return -1;
        }
    }

    /* Run topologically; partial stop is OK (require_all=false). */
    r = rocke_i_run_chain(b, desc, &coords, /*require_all=*/false);
    if(r < 0)
    {
        return -1;
    }

    /* Emit {name: value} for every coord produced, in insertion order. */
    if(coords.count > out_cap)
    {
        return -1;
    }
    for(i = 0; i < coords.count; ++i)
    {
        if(out_names != NULL)
        {
            out_names[i] = coords.items[i].name;
        }
        if(out_values != NULL)
        {
            out_values[i] = coords.items[i].value;
        }
    }
    return coords.count;
}

/* ====================================================================== */
/* TensorDescriptor.offset                                                 */
/* ====================================================================== */

bool rocke_transforms_descriptor_offset(rocke_ir_builder_t* b,
                                        const rocke_tensor_descriptor_t* desc,
                                        const char* const* in_names,
                                        rocke_value_t* const* in_values,
                                        int n_in,
                                        rocke_value_t** out_offset,
                                        rocke_value_t** out_valid)
{
    rocke_i_coord_map_t coords;
    int cap;
    int i;
    int r;
    rocke_value_t* offset = NULL;
    rocke_value_t* valid = NULL;

    if(!rocke_i_live(b))
    {
        return false;
    }

    /* Python _run_chain prologue: every upper_name must be supplied. */
    for(i = 0; i < desc->n_upper; ++i)
    {
        if(!rocke_i_name_in(in_names, n_in, desc->upper_names[i]))
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "offset() missing upper coords for descriptor %s: %s",
                            desc->name ? desc->name : "",
                            desc->upper_names[i]);
            return false;
        }
    }

    cap = n_in;
    for(i = 0; i < desc->n_chain; ++i)
    {
        cap += desc->chain[i]->n_lower;
    }
    if(!rocke_i_map_init(b, &coords, cap > 0 ? cap : 1))
    {
        return false;
    }
    for(i = 0; i < n_in; ++i)
    {
        rocke_coord_var_t cv;
        cv.name = in_names[i];
        cv.value = in_values[i];
        cv.valid = NULL;
        if(!rocke_i_map_set(b, &coords, cv))
        {
            return false;
        }
    }

    /* Full resolution required (Python raises on unresolved deps). */
    r = rocke_i_run_chain(b, desc, &coords, /*require_all=*/true);
    if(r < 0)
    {
        return false;
    }

    /* Reduce base coords with base_strides:
     *   for name, stride in zip(base_names, base_strides):
     *       c = coords[name]   (KeyError -> ValueError if absent)
     *       valid = _and(valid, c.valid)
     *       term = c.value if stride == 1 else b.mul(c.value, b.const_i32(stride))
     *       offset = term if offset is None else b.add(offset, term)
     *   if offset is None: offset = b.const_i32(0)
     */
    for(i = 0; i < desc->n_base; ++i)
    {
        const char* name = desc->base_names[i];
        int stride = desc->base_strides[i];
        const rocke_coord_var_t* c = rocke_i_map_get(&coords, name);
        rocke_value_t* term;
        if(c == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "after chain, base coord %s not present", name);
            return false;
        }
        valid = rocke_i_and(b, valid, c->valid);
        if(stride == 1)
        {
            term = c->value;
        }
        else
        {
            term = rocke_b_mul(b, c->value, rocke_b_const_i32(b, (int64_t)stride));
        }
        offset = (offset == NULL) ? term : rocke_b_add(b, offset, term);
    }
    if(offset == NULL)
    {
        offset = rocke_b_const_i32(b, 0);
    }

    if(out_offset != NULL)
    {
        *out_offset = offset;
    }
    if(out_valid != NULL)
    {
        *out_valid = valid;
    }
    return true;
}

/* Faithful port of TensorDescriptor.offset_i64_split (transforms.py 1463-1505).
 * Returns (base_i64, within_i32, valid): the base_coord term computed in i64
 * (scalarised via to_sgpr_u32 before widening) and all other base terms summed
 * as a small i32 within-block offset. */
bool rocke_transforms_descriptor_offset_i64_split(rocke_ir_builder_t* b,
                                                  const rocke_tensor_descriptor_t* desc,
                                                  const char* base_coord,
                                                  const char* const* in_names,
                                                  rocke_value_t* const* in_values,
                                                  int n_in,
                                                  rocke_value_t** out_base_i64,
                                                  rocke_value_t** out_within,
                                                  rocke_value_t** out_valid)
{
    rocke_i_coord_map_t coords;
    int cap;
    int i;
    int r;
    rocke_value_t* base_i64 = NULL;
    rocke_value_t* within = NULL;
    rocke_value_t* valid = NULL;

    if(!rocke_i_live(b))
    {
        return false;
    }

    for(i = 0; i < desc->n_upper; ++i)
    {
        if(!rocke_i_name_in(in_names, n_in, desc->upper_names[i]))
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "offset_i64_split() missing upper coords for descriptor %s: %s",
                            desc->name ? desc->name : "",
                            desc->upper_names[i]);
            return false;
        }
    }

    cap = n_in;
    for(i = 0; i < desc->n_chain; ++i)
    {
        cap += desc->chain[i]->n_lower;
    }
    if(!rocke_i_map_init(b, &coords, cap > 0 ? cap : 1))
    {
        return false;
    }
    for(i = 0; i < n_in; ++i)
    {
        rocke_coord_var_t cv;
        cv.name = in_names[i];
        cv.value = in_values[i];
        cv.valid = NULL;
        if(!rocke_i_map_set(b, &coords, cv))
        {
            return false;
        }
    }

    r = rocke_i_run_chain(b, desc, &coords, /*require_all=*/true);
    if(r < 0)
    {
        return false;
    }

    for(i = 0; i < desc->n_base; ++i)
    {
        const char* name = desc->base_names[i];
        int stride = desc->base_strides[i];
        const rocke_coord_var_t* c = rocke_i_map_get(&coords, name);
        if(c == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "after chain, base coord %s not present", name);
            return false;
        }
        valid = rocke_i_and(b, valid, c->valid);
        if(strcmp(name, base_coord) == 0)
        {
            /* i64 term: pin the wave-uniform block id to an SGPR before widening
             * (Python b.mul(b.zext(b.to_sgpr_u32(c.value), I64), const_i64(stride))).
             * Bind the zext to a temp so C's right-to-left arg eval does not create
             * the const_i64 ahead of the zext and shift the SSA ids. */
            rocke_value_t* base_val = rocke_b_to_sgpr_u32(b, c->value);
            rocke_value_t* base_w = rocke_b_zext(b, base_val, rocke_i64());
            base_i64 = rocke_b_mul(b, base_w, rocke_b_const_i64(b, (int64_t)stride));
        }
        else
        {
            rocke_value_t* term
                = (stride == 1) ? c->value
                                : rocke_b_mul(b, c->value, rocke_b_const_i32(b, (int64_t)stride));
            within = (within == NULL) ? term : rocke_b_add(b, within, term);
        }
    }
    if(base_i64 == NULL)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "offset_i64_split: base_coord %s not among base coords",
                        base_coord);
        return false;
    }
    if(within == NULL)
    {
        within = rocke_b_const_i32(b, 0);
    }

    if(out_base_i64 != NULL)
    {
        *out_base_i64 = base_i64;
    }
    if(out_within != NULL)
    {
        *out_within = within;
    }
    if(out_valid != NULL)
    {
        *out_valid = valid;
    }
    return true;
}
