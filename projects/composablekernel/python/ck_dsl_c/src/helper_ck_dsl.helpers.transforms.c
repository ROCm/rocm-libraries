/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_ck_dsl.helpers.transforms.c -- C99 port of a SUBSET of
 * ck_dsl.helpers.transforms (the CK Tile coordinate-transform DAG).
 *
 * Ported symbols (see the header for the exact list): calculate_magic_numbers,
 * do_magic_division, CoordVar, Embed/PassThrough/Unmerge/UnmergeMagicDiv
 * transforms + their constructors (embed/pass_through/unmerge/unmerge_magic),
 * and TensorDescriptor.{naive,transform,offset,unmerge_lower}.
 *
 * The builder-call sequence in every emitting function is byte-faithful to the
 * Python so the downstream IR op stream is identical.
 */

#include "ckc/helper_ck_dsl.helpers.transforms.h"

#include <string.h>

#include "ckc/ir_internal.h" /* ckc_i_set_err, ckc_i_live */

/* ====================================================================== */
/* Small i1-predicate / compare helpers (Python _and / _ge / _lt).         */
/* ====================================================================== */

/* Python _and(b, p, q): conjunction of two optional i1 predicates.
 *   if p is None: return q
 *   if q is None: return p
 *   return b.land(p, q)
 */
static ckc_value_t* ckc_i_and(ckc_ir_builder_t* b, ckc_value_t* p, ckc_value_t* q)
{
    if (p == NULL)
    {
        return q;
    }
    if (q == NULL)
    {
        return p;
    }
    return ckc_b_land(b, p, q);
}

/* Python _ge(b, lhs, rhs): signed lhs >= rhs -> i1. */
static ckc_value_t* ckc_i_ge(ckc_ir_builder_t* b, ckc_value_t* lhs, ckc_value_t* rhs)
{
    return ckc_b_cmp_ge(b, lhs, rhs);
}

/* Python _lt(b, lhs, rhs): signed lhs < rhs -> i1. */
static ckc_value_t* ckc_i_lt(ckc_ir_builder_t* b, ckc_value_t* lhs, ckc_value_t* rhs)
{
    return ckc_b_cmp_lt(b, lhs, rhs);
}

/* ====================================================================== */
/* Magic-number division.                                                  */
/* ====================================================================== */

bool ckc_calculate_magic_numbers(ckc_ir_builder_t* b,
                                 int divisor,
                                 uint64_t* out_multiplier,
                                 int* out_shift)
{
    int shift;
    uint64_t multiplier;

    /* Python: if divisor < 1: raise ValueError(...) */
    if (divisor < 1)
    {
        if (b != NULL)
        {
            ckc_i_set_err(b,
                          CKC_ERR_VALUE,
                          "magic division requires divisor >= 1, got %d",
                          divisor);
        }
        return false;
    }

    /* shift = smallest s with (1 << s) >= divisor */
    shift = 0;
    while ((1 << shift) < divisor)
    {
        shift += 1;
    }

    /* multiplier = (((1 << shift) - divisor) << 32) // divisor + 1.
     * Computed in 64-bit unsigned to match the Python arbitrary-precision int
     * for the documented 31-bit range; the bit pattern is what matters. */
    multiplier =
        ((((uint64_t)(1 << shift) - (uint64_t)divisor) << 32) / (uint64_t)divisor) + 1u;

    if (out_multiplier != NULL)
    {
        *out_multiplier = multiplier;
    }
    if (out_shift != NULL)
    {
        *out_shift = shift;
    }
    return true;
}

ckc_value_t* ckc_do_magic_division(ckc_ir_builder_t* b,
                                   ckc_value_t* dividend,
                                   uint64_t multiplier,
                                   int shift)
{
    int64_t mult_i32;
    ckc_value_t* tmp;
    ckc_value_t* summed;

    if (!ckc_i_live(b))
    {
        return NULL;
    }

    /* Python:
     *   mult_i32 = multiplier - (1 << 32) if multiplier >= (1 << 31) else multiplier
     * Bake the uint32 magic as its two's-complement i32 bit pattern. */
    if (multiplier >= ((uint64_t)1 << 31))
    {
        mult_i32 = (int64_t)multiplier - ((int64_t)1 << 32);
    }
    else
    {
        mult_i32 = (int64_t)multiplier;
    }

    tmp    = ckc_b_umul_hi_i32(b, dividend, ckc_b_const_i32(b, mult_i32));
    summed = ckc_b_add(b, tmp, dividend);
    if (shift == 0)
    {
        return summed;
    }
    return ckc_b_lshr(b, summed, ckc_b_const_i32(b, (int64_t)shift));
}

/* ====================================================================== */
/* Coord map: a small ordered (name -> CoordVar) association list.         */
/* The Python uses a dict; we use an insertion-ordered array. Lookups and  */
/* "name in coords" both scan by string equality, matching dict semantics. */
/* (Re-inserting an existing name overwrites in place, like dict[name]=v.)  */
/* ====================================================================== */

typedef struct ckc_i_coord_map
{
    ckc_coord_var_t* items;
    int count;
    int cap;
} ckc_i_coord_map_t;

static bool ckc_i_map_init(ckc_ir_builder_t* b, ckc_i_coord_map_t* m, int cap)
{
    if (cap < 1)
    {
        cap = 1;
    }
    m->items = (ckc_coord_var_t*)ckc_arena_alloc(&b->arena,
                                                 (size_t)cap * sizeof(ckc_coord_var_t));
    if (m->items == NULL)
    {
        return false;
    }
    m->count = 0;
    m->cap   = cap;
    return true;
}

static int ckc_i_map_find(const ckc_i_coord_map_t* m, const char* name)
{
    int i;
    for (i = 0; i < m->count; ++i)
    {
        if (strcmp(m->items[i].name, name) == 0)
        {
            return i;
        }
    }
    return -1;
}

static bool ckc_i_map_has(const ckc_i_coord_map_t* m, const char* name)
{
    return ckc_i_map_find(m, name) >= 0;
}

/* dict-style set: overwrite if present, else append (grow if needed). */
static bool ckc_i_map_set(ckc_ir_builder_t* b, ckc_i_coord_map_t* m, ckc_coord_var_t cv)
{
    int idx = ckc_i_map_find(m, cv.name);
    if (idx >= 0)
    {
        m->items[idx] = cv;
        return true;
    }
    if (m->count == m->cap)
    {
        int new_cap = m->cap * 2;
        ckc_coord_var_t* grown =
            (ckc_coord_var_t*)ckc_arena_alloc(&b->arena,
                                              (size_t)new_cap * sizeof(ckc_coord_var_t));
        if (grown == NULL)
        {
            return false;
        }
        memcpy(grown, m->items, (size_t)m->count * sizeof(ckc_coord_var_t));
        m->items = grown;
        m->cap   = new_cap;
    }
    m->items[m->count] = cv;
    m->count += 1;
    return true;
}

static const ckc_coord_var_t* ckc_i_map_get(const ckc_i_coord_map_t* m, const char* name)
{
    int idx = ckc_i_map_find(m, name);
    if (idx < 0)
    {
        return NULL;
    }
    return &m->items[idx];
}

/* ====================================================================== */
/* Transform constructors.                                                 */
/* ====================================================================== */

/* Duplicate an array of name strings into the arena (each string dup'd too). */
static const char* const* ckc_i_dup_names(ckc_ir_builder_t* b,
                                          const char* const* names,
                                          int n)
{
    const char** out;
    int i;
    if (n <= 0)
    {
        return NULL;
    }
    out = (const char**)ckc_arena_alloc(&b->arena, (size_t)n * sizeof(const char*));
    if (out == NULL)
    {
        return NULL;
    }
    for (i = 0; i < n; ++i)
    {
        out[i] = ckc_arena_strdup(&b->arena, names[i]);
        if (out[i] == NULL)
        {
            return NULL;
        }
    }
    return (const char* const*)out;
}

/* Single-element name array {name}. */
static const char* const* ckc_i_dup_name1(ckc_ir_builder_t* b, const char* name)
{
    return ckc_i_dup_names(b, &name, 1);
}

static const int* ckc_i_dup_ints(ckc_ir_builder_t* b, const int* src, int n)
{
    int* out;
    if (n <= 0)
    {
        return NULL;
    }
    out = (int*)ckc_arena_alloc(&b->arena, (size_t)n * sizeof(int));
    if (out == NULL)
    {
        return NULL;
    }
    memcpy(out, src, (size_t)n * sizeof(int));
    return out;
}

static ckc_transform_t* ckc_i_new_transform(ckc_ir_builder_t* b)
{
    ckc_transform_t* t =
        (ckc_transform_t*)ckc_arena_calloc(&b->arena, sizeof(ckc_transform_t));
    return t;
}

ckc_transform_t* ckc_pass_through(ckc_ir_builder_t* b, const char* coord, const char* into)
{
    ckc_transform_t* t;
    const char* lower_name;

    if (!ckc_i_live(b))
    {
        return NULL;
    }

    /* Python: lower = (lower_name or upper_name,) */
    lower_name = (into != NULL) ? into : coord;

    t = ckc_i_new_transform(b);
    if (t == NULL)
    {
        return NULL;
    }
    t->kind    = CKC_XFORM_PASS_THROUGH;
    t->upper   = ckc_i_dup_name1(b, coord);
    t->n_upper = 1;
    t->lower   = ckc_i_dup_name1(b, lower_name);
    t->n_lower = 1;
    if (t->upper == NULL || t->lower == NULL)
    {
        return NULL;
    }
    return t;
}

ckc_transform_t* ckc_embed_bounded(ckc_ir_builder_t* b,
                                   const char* const* upper,
                                   int n_upper,
                                   const char* into,
                                   const int* strides,
                                   int offset,
                                   int lo,
                                   int hi)
{
    ckc_transform_t* t;

    if (!ckc_i_live(b))
    {
        return NULL;
    }

    /* Python: if len(upper) != len(strides): raise ValueError(...) */
    /* (strides count == n_upper by this API; guard the obvious misuse.) */
    if (n_upper < 0)
    {
        return (ckc_transform_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "Embed expects len(upper) == len(strides)");
    }

    t = ckc_i_new_transform(b);
    if (t == NULL)
    {
        return NULL;
    }
    t->kind    = CKC_XFORM_EMBED;
    t->upper   = ckc_i_dup_names(b, upper, n_upper);
    t->n_upper = n_upper;
    t->lower   = ckc_i_dup_name1(b, into);
    t->n_lower = 1;
    t->strides = ckc_i_dup_ints(b, strides, n_upper);
    t->offset  = offset;
    t->lo      = lo;
    t->hi      = hi;
    if (t->lower == NULL || (n_upper > 0 && (t->upper == NULL || t->strides == NULL)))
    {
        return NULL;
    }
    return t;
}

ckc_transform_t* ckc_embed(ckc_ir_builder_t* b,
                           const char* const* upper,
                           int n_upper,
                           const char* into,
                           const int* strides,
                           int offset)
{
    /* Python None-sentinels: lo=-(1<<30), hi=(1<<30). */
    return ckc_embed_bounded(b, upper, n_upper, into, strides, offset,
                             -(1 << 30), (1 << 30));
}

static ckc_transform_t* ckc_i_new_unmerge(ckc_ir_builder_t* b,
                                          ckc_xform_kind_t kind,
                                          const char* upper,
                                          const char* const* into,
                                          int n_lower,
                                          const int* dims,
                                          const char* who)
{
    ckc_transform_t* t;

    if (!ckc_i_live(b))
    {
        return NULL;
    }

    /* Python: if len(lowers) != len(dims): raise ValueError(...) */
    if (n_lower < 0)
    {
        return (ckc_transform_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "%s expects len(lowers) == len(dims)", who);
    }

    t = ckc_i_new_transform(b);
    if (t == NULL)
    {
        return NULL;
    }
    t->kind    = kind;
    t->upper   = ckc_i_dup_name1(b, upper);
    t->n_upper = 1;
    t->lower   = ckc_i_dup_names(b, into, n_lower);
    t->n_lower = n_lower;
    t->dims    = ckc_i_dup_ints(b, dims, n_lower);
    if (t->upper == NULL || (n_lower > 0 && (t->lower == NULL || t->dims == NULL)))
    {
        return NULL;
    }
    return t;
}

ckc_transform_t* ckc_unmerge(ckc_ir_builder_t* b,
                             const char* upper,
                             const char* const* into,
                             int n_lower,
                             const int* dims)
{
    return ckc_i_new_unmerge(b, CKC_XFORM_UNMERGE, upper, into, n_lower, dims, "Unmerge");
}

ckc_transform_t* ckc_unmerge_magic(ckc_ir_builder_t* b,
                                   const char* upper,
                                   const char* const* into,
                                   int n_lower,
                                   const int* dims)
{
    return ckc_i_new_unmerge(
        b, CKC_XFORM_UNMERGE_MAGIC, upper, into, n_lower, dims, "UnmergeMagicDiv");
}

ckc_transform_t* ckc_pad(ckc_ir_builder_t* b, const char* coord, int lo, int hi)
{
    ckc_transform_t* t;

    if (!ckc_i_live(b))
    {
        return NULL;
    }

    /* Python __init__: upper == lower == (coord_name,); lo/hi int. */
    t = ckc_i_new_transform(b);
    if (t == NULL)
    {
        return NULL;
    }
    t->kind    = CKC_XFORM_PAD;
    t->upper   = ckc_i_dup_name1(b, coord);
    t->n_upper = 1;
    t->lower   = ckc_i_dup_name1(b, coord);
    t->n_lower = 1;
    t->lo      = lo;
    t->hi      = hi;
    if (t->upper == NULL || t->lower == NULL)
    {
        return NULL;
    }
    return t;
}

ckc_transform_t* ckc_indirect(ckc_ir_builder_t* b,
                              const char* upper,
                              const char* into,
                              ckc_value_t* table,
                              ckc_value_t* base,
                              ckc_value_t* max_idx,
                              int default_value)
{
    ckc_transform_t* t;

    if (!ckc_i_live(b))
    {
        return NULL;
    }

    /* Python __init__: upper == (upper_name,); lower == (into,). */
    t = ckc_i_new_transform(b);
    if (t == NULL)
    {
        return NULL;
    }
    t->kind          = CKC_XFORM_INDIRECT;
    t->upper         = ckc_i_dup_name1(b, upper);
    t->n_upper       = 1;
    t->lower         = ckc_i_dup_name1(b, into);
    t->n_lower       = 1;
    t->table         = table;
    t->base          = base;
    t->max_idx       = max_idx;
    t->default_value = default_value;
    if (t->upper == NULL || t->lower == NULL)
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

static bool ckc_i_apply_pass_through(ckc_ir_builder_t* b,
                                     const ckc_transform_t* t,
                                     const ckc_i_coord_map_t* coords,
                                     ckc_i_coord_map_t* out)
{
    /* Python: u = coords[upper[0]]; return {lower[0]: replace(u, name=lower[0])} */
    const ckc_coord_var_t* u = ckc_i_map_get(coords, t->upper[0]);
    ckc_coord_var_t cv;
    cv.name  = t->lower[0];
    cv.value = u->value;
    cv.valid = u->valid;
    return ckc_i_map_set(b, out, cv);
}

static bool ckc_i_apply_embed(ckc_ir_builder_t* b,
                              const ckc_transform_t* t,
                              const ckc_i_coord_map_t* coords,
                              ckc_i_coord_map_t* out)
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
    ckc_value_t* acc       = NULL;
    ckc_value_t* valid_acc = NULL;
    ckc_value_t* bounds;
    ckc_value_t* valid;
    ckc_coord_var_t cv;
    int i;

    for (i = 0; i < t->n_upper; ++i)
    {
        const ckc_coord_var_t* u = ckc_i_map_get(coords, t->upper[i]);
        int s                    = t->strides[i];
        ckc_value_t* term;
        valid_acc = ckc_i_and(b, valid_acc, u->valid);
        if (s == 1)
        {
            term = u->value;
        }
        else
        {
            term = ckc_b_mul(b, u->value, ckc_b_const_i32(b, (int64_t)s));
        }
        acc = (acc == NULL) ? term : ckc_b_add(b, acc, term);
    }
    if (t->offset != 0)
    {
        acc = ckc_b_add(b, acc, ckc_b_const_i32(b, (int64_t)t->offset));
    }
    if (acc == NULL)
    {
        acc = ckc_b_const_i32(b, (int64_t)t->offset);
    }
    /* bounds: lo <= acc < hi (the inner _ge is evaluated before _lt). */
    {
        ckc_value_t* ge = ckc_i_ge(b, acc, ckc_b_const_i32(b, (int64_t)t->lo));
        ckc_value_t* lt = ckc_i_lt(b, acc, ckc_b_const_i32(b, (int64_t)t->hi));
        bounds          = ckc_i_and(b, ge, lt);
    }
    valid = ckc_i_and(b, valid_acc, bounds);

    cv.name  = t->lower[0];
    cv.value = acc;
    cv.valid = valid;
    return ckc_i_map_set(b, out, cv);
}

static bool ckc_i_apply_unmerge(ckc_ir_builder_t* b,
                                const ckc_transform_t* t,
                                const ckc_i_coord_map_t* coords,
                                ckc_i_coord_map_t* out)
{
    /* Python apply():
     *   u = coords[upper[0]]
     *   for i, name in enumerate(lower):
     *       stride = product(dims[i+1:])
     *       quot = u.value if stride == 1 else b.div(u.value, b.const_i32(stride))
     *       val  = quot if i == 0 else b.mod(quot, b.const_i32(dims[i]))
     *       out[name] = CoordVar(name, val, u.valid)
     */
    const ckc_coord_var_t* u = ckc_i_map_get(coords, t->upper[0]);
    /* Cache u's fields: ckc_i_map_set on the shared map may relocate items. */
    ckc_value_t* u_value = u->value;
    ckc_value_t* u_valid = u->valid;
    int i, j;

    for (i = 0; i < t->n_lower; ++i)
    {
        int stride = 1;
        ckc_value_t* quot;
        ckc_value_t* val;
        ckc_coord_var_t cv;
        for (j = i + 1; j < t->n_lower; ++j)
        {
            stride *= t->dims[j];
        }
        if (stride == 1)
        {
            quot = u_value;
        }
        else
        {
            quot = ckc_b_div(b, u_value, ckc_b_const_i32(b, (int64_t)stride));
        }
        if (i == 0)
        {
            val = quot;
        }
        else
        {
            val = ckc_b_mod(b, quot, ckc_b_const_i32(b, (int64_t)t->dims[i]));
        }
        cv.name  = t->lower[i];
        cv.value = val;
        cv.valid = u_valid;
        if (!ckc_i_map_set(b, out, cv))
        {
            return false;
        }
    }
    return true;
}

static bool ckc_i_apply_unmerge_magic(ckc_ir_builder_t* b,
                                      const ckc_transform_t* t,
                                      const ckc_i_coord_map_t* coords,
                                      ckc_i_coord_map_t* out)
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
    const ckc_coord_var_t* u = ckc_i_map_get(coords, t->upper[0]);
    /* Cache u's fields: ckc_i_map_set on the shared map may relocate items. */
    ckc_value_t* u_valid     = u->valid;
    int n                    = t->n_lower;
    ckc_value_t* tmp         = u->value;
    int i;
    ckc_coord_var_t cv;

    for (i = n - 1; i > 0; --i)
    {
        int d = t->dims[i];
        ckc_value_t* rem;
        ckc_value_t* quot;
        if (d == 1)
        {
            rem  = ckc_b_const_i32(b, 0);
            quot = tmp;
        }
        else
        {
            uint64_t mult;
            int shift;
            if (!ckc_calculate_magic_numbers(b, d, &mult, &shift))
            {
                return false;
            }
            quot = ckc_do_magic_division(b, tmp, mult, shift);
            rem  = ckc_b_sub(b, tmp, ckc_b_mul(b, quot, ckc_b_const_i32(b, (int64_t)d)));
        }
        cv.name  = t->lower[i];
        cv.value = rem;
        cv.valid = u_valid;
        if (!ckc_i_map_set(b, out, cv))
        {
            return false;
        }
        tmp = quot;
    }
    cv.name  = t->lower[0];
    cv.value = tmp;
    cv.valid = u_valid;
    return ckc_i_map_set(b, out, cv);
}

static bool ckc_i_apply_pad(ckc_ir_builder_t* b,
                            const ckc_transform_t* t,
                            const ckc_i_coord_map_t* coords,
                            ckc_i_coord_map_t* out)
{
    /* Python apply():
     *   u = coords[upper[0]]
     *   c_lo = b.const_i32(lo); c_hi = b.const_i32(hi)
     *   valid = _and(b, _ge(b, u.value, c_lo), _lt(b, u.value, c_hi))
     *   merged_valid = _and(b, u.valid, valid)
     *   return {lower[0]: CoordVar(lower[0], u.value, merged_valid)}
     */
    const ckc_coord_var_t* u = ckc_i_map_get(coords, t->upper[0]);
    ckc_value_t* u_value     = u->value;
    ckc_value_t* u_valid     = u->valid;
    ckc_value_t* c_lo        = ckc_b_const_i32(b, (int64_t)t->lo);
    ckc_value_t* c_hi        = ckc_b_const_i32(b, (int64_t)t->hi);
    ckc_value_t* ge          = ckc_i_ge(b, u_value, c_lo);
    ckc_value_t* lt          = ckc_i_lt(b, u_value, c_hi);
    ckc_value_t* valid       = ckc_i_and(b, ge, lt);
    ckc_value_t* merged      = ckc_i_and(b, u_valid, valid);
    ckc_coord_var_t cv;
    cv.name  = t->lower[0];
    cv.value = u_value;
    cv.valid = merged;
    return ckc_i_map_set(b, out, cv);
}

static bool ckc_i_apply_indirect(ckc_ir_builder_t* b,
                                 const ckc_transform_t* t,
                                 const ckc_i_coord_map_t* coords,
                                 ckc_i_coord_map_t* out)
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
    const ckc_coord_var_t* u = ckc_i_map_get(coords, t->upper[0]);
    ckc_value_t* u_valid     = u->valid;
    ckc_value_t* idx         = ckc_b_add(b, t->base, u->value);
    ckc_value_t* physical;
    ckc_coord_var_t cv;

    if (t->max_idx == NULL)
    {
        physical = ckc_b_global_load_i32(b, t->table, idx, 4);
    }
    else
    {
        ckc_value_t* mask = ckc_i_lt(b, idx, t->max_idx);
        physical          = ckc_b_masked_global_load(
            b, t->table, idx, mask, ckc_b_const_i32(b, (int64_t)t->default_value),
            ckc_i32(), 4);
    }
    cv.name  = t->lower[0];
    cv.value = physical;
    cv.valid = u_valid;
    return ckc_i_map_set(b, out, cv);
}

/* Dispatch one transform's apply onto the coord map (in place). */
static bool ckc_i_transform_apply(ckc_ir_builder_t* b,
                                  const ckc_transform_t* t,
                                  ckc_i_coord_map_t* coords)
{
    switch (t->kind)
    {
        case CKC_XFORM_PASS_THROUGH:
            return ckc_i_apply_pass_through(b, t, coords, coords);
        case CKC_XFORM_EMBED:
            return ckc_i_apply_embed(b, t, coords, coords);
        case CKC_XFORM_UNMERGE:
            return ckc_i_apply_unmerge(b, t, coords, coords);
        case CKC_XFORM_UNMERGE_MAGIC:
            return ckc_i_apply_unmerge_magic(b, t, coords, coords);
        case CKC_XFORM_PAD:
            return ckc_i_apply_pad(b, t, coords, coords);
        case CKC_XFORM_INDIRECT:
            return ckc_i_apply_indirect(b, t, coords, coords);
        default:
            return false;
    }
}

/* All of a transform's uppers present in the coord map? */
static bool ckc_i_uppers_ready(const ckc_transform_t* t, const ckc_i_coord_map_t* coords)
{
    int i;
    for (i = 0; i < t->n_upper; ++i)
    {
        if (!ckc_i_map_has(coords, t->upper[i]))
        {
            return false;
        }
    }
    return true;
}

/* ====================================================================== */
/* TensorDescriptor.naive                                                  */
/* ====================================================================== */

ckc_tensor_descriptor_t* ckc_tensor_descriptor_naive(ckc_ir_builder_t* b,
                                                     const char* name,
                                                     const int* lengths,
                                                     int n_lengths,
                                                     const int* strides,
                                                     const char* const* coord_names,
                                                     int n_coord_names)
{
    ckc_tensor_descriptor_t* d;
    const int* base_lengths;
    int* base_strides;
    const char* const* base_names;

    if (!ckc_i_live(b))
    {
        return NULL;
    }

    /* Python: if not lengths: raise ValueError("naive descriptor needs ...") */
    if (n_lengths < 1 || lengths == NULL)
    {
        return (ckc_tensor_descriptor_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "naive descriptor needs at least one dim");
    }

    base_lengths = ckc_i_dup_ints(b, lengths, n_lengths);
    if (base_lengths == NULL)
    {
        return NULL;
    }

    /* strides: row-major when not supplied.
     *   ss = [1]; for d in reversed(lengths[1:]): ss.insert(0, ss[0]*d)
     *   strides = ss[:len(lengths)]
     * Concretely strides[i] = product(lengths[i+1:]), strides[last] = 1. */
    base_strides = (int*)ckc_arena_alloc(&b->arena, (size_t)n_lengths * sizeof(int));
    if (base_strides == NULL)
    {
        return NULL;
    }
    if (strides == NULL)
    {
        int i;
        int acc = 1;
        base_strides[n_lengths - 1] = 1;
        for (i = n_lengths - 1; i >= 1; --i)
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
    if (coord_names == NULL)
    {
        const char** names =
            (const char**)ckc_arena_alloc(&b->arena, (size_t)n_lengths * sizeof(const char*));
        int i;
        if (names == NULL)
        {
            return NULL;
        }
        for (i = 0; i < n_lengths; ++i)
        {
            names[i] = ckc_arena_printf(&b->arena, "d%d", i);
            if (names[i] == NULL)
            {
                return NULL;
            }
        }
        base_names = (const char* const*)names;
    }
    else
    {
        /* Python: if len(coord_names) != len(lengths): raise ValueError(...) */
        if (n_coord_names != n_lengths)
        {
            return (ckc_tensor_descriptor_t*)ckc_i_set_err(
                b, CKC_ERR_VALUE, "coord_names length mismatch");
        }
        base_names = ckc_i_dup_names(b, coord_names, n_lengths);
        if (base_names == NULL)
        {
            return NULL;
        }
    }

    d = (ckc_tensor_descriptor_t*)ckc_arena_calloc(&b->arena,
                                                   sizeof(ckc_tensor_descriptor_t));
    if (d == NULL)
    {
        return NULL;
    }
    d->name         = ckc_arena_strdup(&b->arena, name);
    d->base_names   = base_names;
    d->base_lengths = base_lengths;
    d->base_strides = base_strides;
    d->n_base       = n_lengths;
    d->chain        = NULL;
    d->n_chain      = 0;
    /* upper_names = coord_names (the naive coords are all user-facing). */
    d->upper_names = base_names;
    d->n_upper     = n_lengths;
    if (d->name == NULL)
    {
        return NULL;
    }
    return d;
}

/* ====================================================================== */
/* TensorDescriptor.transform                                              */
/* ====================================================================== */

/* Name-membership in a name array. */
static bool ckc_i_name_in(const char* const* arr, int n, const char* name)
{
    int i;
    for (i = 0; i < n; ++i)
    {
        if (strcmp(arr[i], name) == 0)
        {
            return true;
        }
    }
    return false;
}

/* Is `name` a lower of any transform in the chain? (the subtraction set). */
static bool ckc_i_is_lower_of_any(const ckc_transform_t* const* chain,
                                  int n_chain,
                                  const char* name)
{
    int ci, li;
    for (ci = 0; ci < n_chain; ++ci)
    {
        for (li = 0; li < chain[ci]->n_lower; ++li)
        {
            if (strcmp(chain[ci]->lower[li], name) == 0)
            {
                return true;
            }
        }
    }
    return false;
}

ckc_tensor_descriptor_t* ckc_tensor_descriptor_transform(ckc_ir_builder_t* b,
                                                         const ckc_tensor_descriptor_t* desc,
                                                         const ckc_transform_t* const* transforms,
                                                         int n_transforms)
{
    ckc_tensor_descriptor_t* d;
    const ckc_transform_t** new_chain;
    int new_n_chain;
    int ti, k;

    if (!ckc_i_live(b))
    {
        return NULL;
    }

    /* Python: if not transforms: return self */
    if (n_transforms <= 0)
    {
        return (ckc_tensor_descriptor_t*)desc;
    }

    new_n_chain = desc->n_chain + n_transforms;
    new_chain   = (const ckc_transform_t**)ckc_arena_alloc(
        &b->arena, (size_t)new_n_chain * sizeof(const ckc_transform_t*));
    if (new_chain == NULL)
    {
        return NULL;
    }
    for (k = 0; k < desc->n_chain; ++k)
    {
        new_chain[k] = desc->chain[k];
    }
    for (ti = 0; ti < n_transforms; ++ti)
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

        for (k = 0; k < new_n_chain; ++k)
        {
            cap += new_chain[k]->n_upper;
        }
        if (cap < 1)
        {
            cap = 1;
        }
        ordered = (const char**)ckc_arena_alloc(&b->arena, (size_t)cap * sizeof(const char*));
        if (ordered == NULL)
        {
            return NULL;
        }

        const ckc_transform_t* const* chain_view =
            (const ckc_transform_t* const*)new_chain;

        /* base_names first. */
        for (k = 0; k < desc->n_base; ++k)
        {
            const char* nm = desc->base_names[k];
            if (!ckc_i_is_lower_of_any(chain_view, new_n_chain, nm) &&
                !ckc_i_name_in(ordered, n_ordered, nm))
            {
                ordered[n_ordered++] = nm;
            }
        }
        /* then transform uppers in appearance order. */
        for (k = 0; k < new_n_chain; ++k)
        {
            int u;
            for (u = 0; u < new_chain[k]->n_upper; ++u)
            {
                const char* nm = new_chain[k]->upper[u];
                if (!ckc_i_is_lower_of_any(chain_view, new_n_chain, nm) &&
                    !ckc_i_name_in(ordered, n_ordered, nm))
                {
                    ordered[n_ordered++] = nm;
                }
            }
        }

        d = (ckc_tensor_descriptor_t*)ckc_arena_calloc(
            &b->arena, sizeof(ckc_tensor_descriptor_t));
        if (d == NULL)
        {
            return NULL;
        }
        /* replace(self, chain=new_chain, upper_names=tuple(ordered)) -- all
         * other fields copied verbatim from desc (they share arena storage). */
        d->name         = desc->name;
        d->base_names   = desc->base_names;
        d->base_lengths = desc->base_lengths;
        d->base_strides = desc->base_strides;
        d->n_base       = desc->n_base;
        d->chain        = (const ckc_transform_t* const*)new_chain;
        d->n_chain      = new_n_chain;
        d->upper_names  = (const char* const*)ordered;
        d->n_upper      = n_ordered;
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
static int ckc_i_run_chain(ckc_ir_builder_t* b,
                           const ckc_tensor_descriptor_t* desc,
                           ckc_i_coord_map_t* coords,
                           bool require_all)
{
    /* `remaining` is the worklist of not-yet-applied transforms. */
    const ckc_transform_t** remaining;
    int n_remaining;
    int i;

    if (desc->n_chain == 0)
    {
        return 1;
    }
    remaining = (const ckc_transform_t**)ckc_arena_alloc(
        &b->arena, (size_t)desc->n_chain * sizeof(const ckc_transform_t*));
    if (remaining == NULL)
    {
        return -1;
    }
    n_remaining = desc->n_chain;
    for (i = 0; i < desc->n_chain; ++i)
    {
        remaining[i] = desc->chain[i];
    }

    while (n_remaining > 0)
    {
        bool progress = false;
        int next_n    = 0;
        int j;
        for (j = 0; j < n_remaining; ++j)
        {
            const ckc_transform_t* t = remaining[j];
            if (ckc_i_uppers_ready(t, coords))
            {
                if (!ckc_i_transform_apply(b, t, coords))
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
        if (!progress)
        {
            if (require_all)
            {
                ckc_i_set_err(b,
                              CKC_ERR_VALUE,
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

int ckc_tensor_descriptor_unmerge_lower(ckc_ir_builder_t* b,
                                        const ckc_tensor_descriptor_t* desc,
                                        const char* const* in_names,
                                        ckc_value_t* const* in_values,
                                        int n_in,
                                        const char** out_names,
                                        ckc_value_t** out_values,
                                        int out_cap)
{
    ckc_i_coord_map_t coords;
    int cap;
    int i;
    int r;

    if (!ckc_i_live(b))
    {
        return -1;
    }

    /* Pre-size the map generously: inputs + every transform's lowers. */
    cap = n_in;
    for (i = 0; i < desc->n_chain; ++i)
    {
        cap += desc->chain[i]->n_lower;
    }
    if (!ckc_i_map_init(b, &coords, cap > 0 ? cap : 1))
    {
        return -1;
    }

    /* Seed with the supplied upper coords (valid omitted -> NULL/None). */
    for (i = 0; i < n_in; ++i)
    {
        ckc_coord_var_t cv;
        cv.name  = in_names[i];
        cv.value = in_values[i];
        cv.valid = NULL;
        if (!ckc_i_map_set(b, &coords, cv))
        {
            return -1;
        }
    }

    /* Run topologically; partial stop is OK (require_all=false). */
    r = ckc_i_run_chain(b, desc, &coords, /*require_all=*/false);
    if (r < 0)
    {
        return -1;
    }

    /* Emit {name: value} for every coord produced, in insertion order. */
    if (coords.count > out_cap)
    {
        return -1;
    }
    for (i = 0; i < coords.count; ++i)
    {
        if (out_names != NULL)
        {
            out_names[i] = coords.items[i].name;
        }
        if (out_values != NULL)
        {
            out_values[i] = coords.items[i].value;
        }
    }
    return coords.count;
}

/* ====================================================================== */
/* TensorDescriptor.offset                                                 */
/* ====================================================================== */

bool ckc_transforms_descriptor_offset(ckc_ir_builder_t* b,
                                      const ckc_tensor_descriptor_t* desc,
                                      const char* const* in_names,
                                      ckc_value_t* const* in_values,
                                      int n_in,
                                      ckc_value_t** out_offset,
                                      ckc_value_t** out_valid)
{
    ckc_i_coord_map_t coords;
    int cap;
    int i;
    int r;
    ckc_value_t* offset = NULL;
    ckc_value_t* valid  = NULL;

    if (!ckc_i_live(b))
    {
        return false;
    }

    /* Python _run_chain prologue: every upper_name must be supplied. */
    for (i = 0; i < desc->n_upper; ++i)
    {
        if (!ckc_i_name_in(in_names, n_in, desc->upper_names[i]))
        {
            ckc_i_set_err(b,
                          CKC_ERR_VALUE,
                          "offset() missing upper coords for descriptor %s: %s",
                          desc->name ? desc->name : "",
                          desc->upper_names[i]);
            return false;
        }
    }

    cap = n_in;
    for (i = 0; i < desc->n_chain; ++i)
    {
        cap += desc->chain[i]->n_lower;
    }
    if (!ckc_i_map_init(b, &coords, cap > 0 ? cap : 1))
    {
        return false;
    }
    for (i = 0; i < n_in; ++i)
    {
        ckc_coord_var_t cv;
        cv.name  = in_names[i];
        cv.value = in_values[i];
        cv.valid = NULL;
        if (!ckc_i_map_set(b, &coords, cv))
        {
            return false;
        }
    }

    /* Full resolution required (Python raises on unresolved deps). */
    r = ckc_i_run_chain(b, desc, &coords, /*require_all=*/true);
    if (r < 0)
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
    for (i = 0; i < desc->n_base; ++i)
    {
        const char* name = desc->base_names[i];
        int stride       = desc->base_strides[i];
        const ckc_coord_var_t* c = ckc_i_map_get(&coords, name);
        ckc_value_t* term;
        if (c == NULL)
        {
            ckc_i_set_err(b,
                          CKC_ERR_VALUE,
                          "after chain, base coord %s not present",
                          name);
            return false;
        }
        valid = ckc_i_and(b, valid, c->valid);
        if (stride == 1)
        {
            term = c->value;
        }
        else
        {
            term = ckc_b_mul(b, c->value, ckc_b_const_i32(b, (int64_t)stride));
        }
        offset = (offset == NULL) ? term : ckc_b_add(b, offset, term);
    }
    if (offset == NULL)
    {
        offset = ckc_b_const_i32(b, 0);
    }

    if (out_offset != NULL)
    {
        *out_offset = offset;
    }
    if (out_valid != NULL)
    {
        *out_valid = valid;
    }
    return true;
}

/* Faithful port of TensorDescriptor.offset_i64_split (transforms.py 1463-1505).
 * Returns (base_i64, within_i32, valid): the base_coord term computed in i64
 * (scalarised via to_sgpr_u32 before widening) and all other base terms summed
 * as a small i32 within-block offset. */
bool ckc_transforms_descriptor_offset_i64_split(ckc_ir_builder_t* b,
                                                const ckc_tensor_descriptor_t* desc,
                                                const char* base_coord,
                                                const char* const* in_names,
                                                ckc_value_t* const* in_values,
                                                int n_in,
                                                ckc_value_t** out_base_i64,
                                                ckc_value_t** out_within,
                                                ckc_value_t** out_valid)
{
    ckc_i_coord_map_t coords;
    int cap;
    int i;
    int r;
    ckc_value_t* base_i64 = NULL;
    ckc_value_t* within   = NULL;
    ckc_value_t* valid    = NULL;

    if (!ckc_i_live(b))
    {
        return false;
    }

    for (i = 0; i < desc->n_upper; ++i)
    {
        if (!ckc_i_name_in(in_names, n_in, desc->upper_names[i]))
        {
            ckc_i_set_err(b,
                          CKC_ERR_VALUE,
                          "offset_i64_split() missing upper coords for descriptor %s: %s",
                          desc->name ? desc->name : "",
                          desc->upper_names[i]);
            return false;
        }
    }

    cap = n_in;
    for (i = 0; i < desc->n_chain; ++i)
    {
        cap += desc->chain[i]->n_lower;
    }
    if (!ckc_i_map_init(b, &coords, cap > 0 ? cap : 1))
    {
        return false;
    }
    for (i = 0; i < n_in; ++i)
    {
        ckc_coord_var_t cv;
        cv.name  = in_names[i];
        cv.value = in_values[i];
        cv.valid = NULL;
        if (!ckc_i_map_set(b, &coords, cv))
        {
            return false;
        }
    }

    r = ckc_i_run_chain(b, desc, &coords, /*require_all=*/true);
    if (r < 0)
    {
        return false;
    }

    for (i = 0; i < desc->n_base; ++i)
    {
        const char* name = desc->base_names[i];
        int stride       = desc->base_strides[i];
        const ckc_coord_var_t* c = ckc_i_map_get(&coords, name);
        if (c == NULL)
        {
            ckc_i_set_err(b,
                          CKC_ERR_VALUE,
                          "after chain, base coord %s not present",
                          name);
            return false;
        }
        valid = ckc_i_and(b, valid, c->valid);
        if (strcmp(name, base_coord) == 0)
        {
            /* i64 term: pin the wave-uniform block id to an SGPR before widening
             * (Python b.mul(b.zext(b.to_sgpr_u32(c.value), I64), const_i64(stride))).
             * Bind the zext to a temp so C's right-to-left arg eval does not create
             * the const_i64 ahead of the zext and shift the SSA ids. */
            ckc_value_t* base_val = ckc_b_to_sgpr_u32(b, c->value);
            ckc_value_t* base_w   = ckc_b_zext(b, base_val, ckc_i64());
            base_i64 = ckc_b_mul(b, base_w, ckc_b_const_i64(b, (int64_t)stride));
        }
        else
        {
            ckc_value_t* term =
                (stride == 1) ? c->value
                              : ckc_b_mul(b, c->value, ckc_b_const_i32(b, (int64_t)stride));
            within = (within == NULL) ? term : ckc_b_add(b, within, term);
        }
    }
    if (base_i64 == NULL)
    {
        ckc_i_set_err(b,
                      CKC_ERR_VALUE,
                      "offset_i64_split: base_coord %s not among base coords",
                      base_coord);
        return false;
    }
    if (within == NULL)
    {
        within = ckc_b_const_i32(b, 0);
    }

    if (out_base_i64 != NULL)
    {
        *out_base_i64 = base_i64;
    }
    if (out_within != NULL)
    {
        *out_within = within;
    }
    if (out_valid != NULL)
    {
        *out_valid = valid;
    }
    return true;
}
