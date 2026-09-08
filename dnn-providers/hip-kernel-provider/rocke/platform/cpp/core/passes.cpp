// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * passes.c -- C99 port of rocke.core.passes.
 *
 * Conservative IR canonicalization passes operating in place over the FROZEN
 * IR (rocke/ir.h): integer constant folding, common-subexpression elimination of
 * pure single-result ops, and dead-pure-op elimination. A faithful translation
 * of passes.py.
 *
 * The Python code relies on dict/tuple keys for CSE and on the GC for storage.
 * Here:
 *   - the "replacements" map (old Value name -> new Value) is a small
 *     linear-probed parallel array allocated in a scratch arena;
 *   - the CSE table is a list of (key-op, result-Value) entries compared
 *     structurally by rocke_i_ops_cse_equal (the C analog of _cse_key equality);
 *   - region op lists are rebuilt into fresh arena arrays;
 *   - fresh operand arrays for rewriting are arena-allocated.
 *
 * All allocation comes from the builder arena (b->arena), which owns the kernel
 * graph for its whole lifetime -- abandoned intermediate arrays are reclaimed in
 * bulk when the arena is destroyed, exactly mirroring Python GC lifetime.
 */
#include <stdlib.h>
#include <string.h>

#include "rocke/passes.h"
#include "rocke/vec.h"

/* ------------------------------------------------------------- pass stats */

rocke_pass_stats_t rocke_pass_stats_add(rocke_pass_stats_t a, rocke_pass_stats_t b)
{
    rocke_pass_stats_t r;
    r.constants_folded = a.constants_folded + b.constants_folded;
    r.common_subexpressions = a.common_subexpressions + b.common_subexpressions;
    r.dead_ops_removed = a.dead_ops_removed + b.dead_ops_removed;
    return r;
}

bool rocke_pass_stats_is_zero(rocke_pass_stats_t s)
{
    return s.constants_folded == 0 && s.common_subexpressions == 0 && s.dead_ops_removed == 0;
}

static rocke_pass_stats_t pass_stats_zero(void)
{
    rocke_pass_stats_t z;
    z.constants_folded = 0;
    z.common_subexpressions = 0;
    z.dead_ops_removed = 0;
    return z;
}

/* ----------------------------------------------------- replacement map ---
 * Python: Dict[str, Value]  (old value name -> replacement value).
 * Small, append-only, linear search. Arena-backed. */

typedef struct repl_entry
{
    const char* name; /* old value name (with leading '%') */
    rocke_value_t* value; /* replacement value */
} repl_entry_t;

typedef struct repl_map
{
    rocke_ir_builder_t* b;
    repl_entry_t* entries;
    int count;
    int cap;
} repl_map_t;

static void repl_init(repl_map_t* m, rocke_ir_builder_t* b)
{
    m->b = b;
    m->entries = NULL;
    m->count = 0;
    m->cap = 0;
}

static void repl_set(repl_map_t* m, const char* name, rocke_value_t* value)
{
    int i;
    for(i = 0; i < m->count; i++)
    {
        if(strcmp(m->entries[i].name, name) == 0)
        {
            m->entries[i].value = value;
            return;
        }
    }
    if(m->count == m->cap)
    {
        int nc = m->cap ? m->cap * 2 : 8;
        repl_entry_t* ne = (repl_entry_t*)rocke_arena_alloc(&m->b->arena, (size_t)nc * sizeof(*ne));
        if(!ne)
            return; /* OOM: silently keep old map; arena sets no err here */
        if(m->entries && m->count)
            memcpy(ne, m->entries, (size_t)m->count * sizeof(*ne));
        m->entries = ne;
        m->cap = nc;
    }
    m->entries[m->count].name = name;
    m->entries[m->count].value = value;
    m->count++;
}

/* Python: replacements.get(v.name, v) */
static rocke_value_t* repl_get(const repl_map_t* m, rocke_value_t* v)
{
    int i;
    if(!v || !v->name)
        return v;
    for(i = 0; i < m->count; i++)
    {
        if(strcmp(m->entries[i].name, v->name) == 0)
            return m->entries[i].value;
    }
    return v;
}

/* ------------------------------------------------------ operand rewriting */

static void
    rewrite_region_operands(rocke_ir_builder_t* b, rocke_region_t* region, const repl_map_t* repl);

/* Python _rewrite_operands: op.operands = [repl.get(v.name, v) ...]; recurse. */
static void rewrite_operands(rocke_ir_builder_t* b, rocke_op_t* op, const repl_map_t* repl)
{
    int i;
    for(i = 0; i < op->num_operands; i++)
        op->operands[i] = repl_get(repl, op->operands[i]);
    for(i = 0; i < op->num_regions; i++)
        rewrite_region_operands(b, op->regions[i], repl);
}

static void
    rewrite_region_operands(rocke_ir_builder_t* b, rocke_region_t* region, const repl_map_t* repl)
{
    int i;
    if(!region)
        return;
    for(i = 0; i < region->num_ops; i++)
    {
        rewrite_operands(b, region->ops[i], repl);
        /* rewrite_operands already recurses into op->regions; mirror Python
         * _rewrite_region_operands which also recurses (idempotent). */
    }
}

/* --------------------------------------------------------------- use count
 * Python _count_uses: Dict[str, int] of operand name -> use count, summed
 * across nested regions. We need only "is the count zero", so a small
 * arena-backed name->count map suffices. */

typedef struct use_entry
{
    const char* name;
    int count;
} use_entry_t;

typedef struct use_map
{
    rocke_ir_builder_t* b;
    use_entry_t* entries;
    int count;
    int cap;
} use_map_t;

static void use_init(use_map_t* m, rocke_ir_builder_t* b)
{
    m->b = b;
    m->entries = NULL;
    m->count = 0;
    m->cap = 0;
}

static void use_add(use_map_t* m, const char* name, int delta)
{
    int i;
    if(!name)
        return;
    for(i = 0; i < m->count; i++)
    {
        if(strcmp(m->entries[i].name, name) == 0)
        {
            m->entries[i].count += delta;
            return;
        }
    }
    if(m->count == m->cap)
    {
        int nc = m->cap ? m->cap * 2 : 16;
        use_entry_t* ne = (use_entry_t*)rocke_arena_alloc(&m->b->arena, (size_t)nc * sizeof(*ne));
        if(!ne)
            return;
        if(m->entries && m->count)
            memcpy(ne, m->entries, (size_t)m->count * sizeof(*ne));
        m->entries = ne;
        m->cap = nc;
    }
    m->entries[m->count].name = name;
    m->entries[m->count].count = delta;
    m->count++;
}

static int use_get(const use_map_t* m, const char* name)
{
    int i;
    if(!name)
        return 0;
    for(i = 0; i < m->count; i++)
    {
        if(strcmp(m->entries[i].name, name) == 0)
            return m->entries[i].count;
    }
    return 0;
}

static void count_uses_region(rocke_region_t* region, use_map_t* uses)
{
    int i, j;
    if(!region)
        return;
    for(i = 0; i < region->num_ops; i++)
    {
        rocke_op_t* op = region->ops[i];
        for(j = 0; j < op->num_operands; j++)
        {
            rocke_value_t* operand = op->operands[j];
            if(operand)
                use_add(uses, operand->name, 1);
        }
        for(j = 0; j < op->num_regions; j++)
            count_uses_region(op->regions[j], uses);
    }
}

/* ------------------------------------------------------------- attr equality
 * Python _cse_key sorts attrs (excluding "loc") and _freeze_attr makes
 * lists/dicts hashable. We compare two attr maps for set-equality ignoring
 * "loc": equal iff each non-loc entry in `a` has a matching entry in `b` with an
 * equal value, and the non-loc counts agree. */

static bool attr_value_eq(const rocke_attr_value_t* x, const rocke_attr_value_t* y);

static bool attr_map_eq_no_loc(const rocke_attr_map_t* a, const rocke_attr_map_t* b);

static bool attr_value_eq(const rocke_attr_value_t* x, const rocke_attr_value_t* y)
{
    if(x->kind != y->kind)
        return false;
    switch(x->kind)
    {
    case ROCKE_ATTR_INT:
        return x->u.i == y->u.i;
    case ROCKE_ATTR_FLOAT:
        /* Bitwise equality (Python freezes the float as-is into the key). */
        return memcmp(&x->u.f, &y->u.f, sizeof(double)) == 0;
    case ROCKE_ATTR_STR:
        if(x->u.s == y->u.s)
            return true;
        if(!x->u.s || !y->u.s)
            return false;
        return strcmp(x->u.s, y->u.s) == 0;
    case ROCKE_ATTR_BOOL:
        return x->u.b == y->u.b;
    case ROCKE_ATTR_LIST:
    {
        int i;
        if(x->u.list.count != y->u.list.count)
            return false;
        /* _freeze_attr preserves list ORDER (tuple), so compare positionally. */
        for(i = 0; i < x->u.list.count; i++)
        {
            if(!attr_map_eq_no_loc(x->u.list.items[i], y->u.list.items[i]))
                return false;
        }
        return true;
    }
    case ROCKE_ATTR_INT_LIST:
    {
        int i;
        if(x->u.ilist.count != y->u.ilist.count)
            return false;
        for(i = 0; i < x->u.ilist.count; i++)
        {
            if(x->u.ilist.ints[i] != y->u.ilist.ints[i])
                return false;
        }
        return true;
    }
    }
    return false;
}

/* Count non-"loc" entries. */
static int attr_count_no_loc(const rocke_attr_map_t* m)
{
    int i, n = 0;
    for(i = 0; i < m->count; i++)
    {
        if(strcmp(m->entries[i].key, "loc") != 0)
            n++;
    }
    return n;
}

static bool attr_map_eq_no_loc(const rocke_attr_map_t* a, const rocke_attr_map_t* b)
{
    int i, j;
    if(attr_count_no_loc(a) != attr_count_no_loc(b))
        return false;
    for(i = 0; i < a->count; i++)
    {
        const char* key = a->entries[i].key;
        bool found = false;
        if(strcmp(key, "loc") == 0)
            continue;
        for(j = 0; j < b->count; j++)
        {
            if(strcmp(b->entries[j].key, key) == 0)
            {
                if(!attr_value_eq(&a->entries[i].value, &b->entries[j].value))
                    return false;
                found = true;
                break;
            }
        }
        if(!found)
            return false;
    }
    return true;
}

/* Structural equality of CSE keys: (name, operand names, attrs\loc, result
 * type names). Python _cse_key tuple equality. */
static bool ops_cse_equal(const rocke_op_t* a, const rocke_op_t* b)
{
    int i;
    if(a->opcode != b->opcode)
        return false;
    /* a->name is the dotted name; opcode equality already implies name match,
     * but compare defensively to mirror op.name in the key. */
    if(a->num_operands != b->num_operands)
        return false;
    if(a->num_results != b->num_results)
        return false;
    for(i = 0; i < a->num_operands; i++)
    {
        const char* an = a->operands[i] ? a->operands[i]->name : NULL;
        const char* bn = b->operands[i] ? b->operands[i]->name : NULL;
        if(an == bn)
            continue;
        if(!an || !bn)
            return false;
        if(strcmp(an, bn) != 0)
            return false;
    }
    for(i = 0; i < a->num_results; i++)
    {
        const rocke_type_t* at = a->results[i] ? a->results[i]->type : NULL;
        const rocke_type_t* bt = b->results[i] ? b->results[i]->type : NULL;
        const char* an = at ? at->name : NULL;
        const char* bn = bt ? bt->name : NULL;
        if(an == bn)
            continue;
        if(!an || !bn)
            return false;
        if(strcmp(an, bn) != 0)
            return false;
    }
    return attr_map_eq_no_loc(&a->attrs, &b->attrs);
}

/* --------------------------------------------------------------- CSE table
 * Python: Dict[Tuple, Value]. List of (op, result) entries searched by
 * ops_cse_equal. Only pure single-result ops are inserted. */

typedef struct cse_entry
{
    rocke_op_t* op; /* representative op (for key comparison) */
    rocke_value_t* result; /* its single result */
} cse_entry_t;

typedef struct cse_table
{
    rocke_ir_builder_t* b;
    cse_entry_t* entries;
    int count;
    int cap;
} cse_table_t;

static void cse_init(cse_table_t* t, rocke_ir_builder_t* b)
{
    t->b = b;
    t->entries = NULL;
    t->count = 0;
    t->cap = 0;
}

/* Returns the cached result Value if an equal op is present, else NULL. */
static rocke_value_t* cse_lookup(const cse_table_t* t, const rocke_op_t* op)
{
    int i;
    for(i = 0; i < t->count; i++)
    {
        if(ops_cse_equal(t->entries[i].op, op))
            return t->entries[i].result;
    }
    return NULL;
}

static void cse_insert(cse_table_t* t, rocke_op_t* op, rocke_value_t* result)
{
    if(t->count == t->cap)
    {
        int nc = t->cap ? t->cap * 2 : 16;
        cse_entry_t* ne = (cse_entry_t*)rocke_arena_alloc(&t->b->arena, (size_t)nc * sizeof(*ne));
        if(!ne)
            return;
        if(t->entries && t->count)
            memcpy(ne, t->entries, (size_t)t->count * sizeof(*ne));
        t->entries = ne;
        t->cap = nc;
    }
    t->entries[t->count].op = op;
    t->entries[t->count].result = result;
    t->count++;
}

/* ----------------------------------------------------------- constant fold */

/* Python _constant_ity(t). */
static const char* constant_ity(const rocke_type_t* t)
{
    if(t && t->name)
    {
        if(strcmp(t->name, "i1") == 0)
            return "i1";
        if(strcmp(t->name, "i64") == 0)
            return "i64";
    }
    return "i32";
}

/* Python _const_int(v): returns the int value of a constant operand iff its op
 * is "arith.constant" with ity in {i1,i32,i64}. *out is set; returns true. */
static bool const_int(const rocke_value_t* v, int64_t* out)
{
    rocke_op_t* op;
    const char* ity;
    const rocke_attr_value_t* val;
    if(!v)
        return false;
    op = v->op;
    if(op == NULL || op->opcode != ROCKE_OP_ARITH_CONSTANT)
        return false;
    ity = rocke_attr_get_str(&op->attrs, "ity");
    if(ity == NULL)
        ity = "i32"; /* op.attrs.get("ity", "i32") */
    if(strcmp(ity, "i1") != 0 && strcmp(ity, "i32") != 0 && strcmp(ity, "i64") != 0)
        return false;
    val = rocke_attr_get(&op->attrs, "value");
    if(!val)
        return false;
    /* Python int(op.attrs["value"]). Constant ints are stored as INT; tolerate
     * a FLOAT just in case (truncate toward zero). */
    if(val->kind == ROCKE_ATTR_INT)
    {
        *out = val->u.i;
        return true;
    }
    if(val->kind == ROCKE_ATTR_FLOAT)
    {
        *out = (int64_t)val->u.f;
        return true;
    }
    if(val->kind == ROCKE_ATTR_BOOL)
    {
        *out = val->u.b ? 1 : 0;
        return true;
    }
    return false;
}

/* Python _try_fold(op): returns true and writes *out for a foldable integer op
 * whose operands are all constant ints; false otherwise. */
static bool try_fold(const rocke_op_t* op, int64_t* out)
{
    int64_t ints[3];
    int i;
    if(op->num_results != 1)
        return false;
    /* Every foldable op takes 1..3 operands; a different arity cannot be one. */
    if(op->num_operands < 1 || op->num_operands > 3)
        return false;
    /* Python: ints = [_const_int(v) for v in op.operands]; if any None -> skip */
    for(i = 0; i < op->num_operands; i++)
    {
        if(!const_int(op->operands[i], &ints[i]))
            return false;
    }

    switch(op->opcode)
    {
    case ROCKE_OP_ARITH_ADD:
        if(op->num_operands != 2)
            return false;
        *out = ints[0] + ints[1];
        return true;
    case ROCKE_OP_ARITH_SUB:
        if(op->num_operands != 2)
            return false;
        *out = ints[0] - ints[1];
        return true;
    case ROCKE_OP_ARITH_MUL:
        if(op->num_operands != 2)
            return false;
        *out = ints[0] * ints[1];
        return true;
    case ROCKE_OP_ARITH_DIV:
        if(op->num_operands != 2 || ints[1] == 0)
            return false;
        /* Python int(ints[0] / ints[1]): float division truncated toward 0. */
        *out = (int64_t)((double)ints[0] / (double)ints[1]);
        return true;
    case ROCKE_OP_ARITH_MOD:
        if(op->num_operands != 2 || ints[1] == 0)
            return false;
        /* Python % is floored; mirror it (C99 % truncates toward zero). */
        {
            int64_t a = ints[0], bm = ints[1];
            int64_t r = a % bm;
            if(r != 0 && ((r < 0) != (bm < 0)))
                r += bm;
            *out = r;
        }
        return true;
    case ROCKE_OP_ARITH_AND:
        if(op->num_operands != 2)
            return false;
        *out = ints[0] & ints[1];
        return true;
    case ROCKE_OP_ARITH_OR:
        if(op->num_operands != 2)
            return false;
        *out = ints[0] | ints[1];
        return true;
    case ROCKE_OP_ARITH_ZEXT:
    case ROCKE_OP_ARITH_SEXT:
        if(op->num_operands != 1)
            return false;
        *out = ints[0];
        return true;
    case ROCKE_OP_ARITH_CMP:
    {
        const char* pred;
        int64_t a, bc;
        if(op->num_operands != 2)
            return false;
        pred = rocke_attr_get_str(&op->attrs, "pred");
        if(pred == NULL)
            pred = "lt"; /* op.attrs.get("pred", "lt") */
        a = ints[0];
        bc = ints[1];
        if(strcmp(pred, "lt") == 0)
            *out = (a < bc);
        else if(strcmp(pred, "le") == 0)
            *out = (a <= bc);
        else if(strcmp(pred, "gt") == 0)
            *out = (a > bc);
        else if(strcmp(pred, "ge") == 0)
            *out = (a >= bc);
        else if(strcmp(pred, "eq") == 0)
            *out = (a == bc);
        else if(strcmp(pred, "ne") == 0)
            *out = (a != bc);
        else
            return false; /* Python would KeyError; conservatively skip fold */
        return true;
    }
    case ROCKE_OP_ARITH_SELECT:
        if(op->num_operands != 3)
            return false;
        *out = ints[0] ? ints[1] : ints[2];
        return true;
    default:
        return false;
    }
}

/* Rewrite `op` in place into an arith.constant of integer value `value`.
 * Python:
 *   op.name = "arith.constant"; op.operands = []
 *   op.attrs = {"value": value, "ity": _constant_ity(op.result.type)}
 *   op.regions = []
 */
static void fold_to_constant(rocke_ir_builder_t* b, rocke_op_t* op, int64_t value)
{
    const rocke_type_t* rty = op->results[0] ? op->results[0]->type : NULL;
    op->opcode = ROCKE_OP_ARITH_CONSTANT;
    op->name = rocke_opcode_name(ROCKE_OP_ARITH_CONSTANT);
    op->operands = NULL;
    op->num_operands = 0;
    op->regions = NULL;
    op->num_regions = 0;
    rocke_attr_map_init(&op->attrs);
    rocke_attr_set_int(b, &op->attrs, "value", value);
    rocke_attr_set_str(b, &op->attrs, "ity", constant_ity(rty));
}

/* ----------------------------------------------------------- region rebuild */

/* Replace region->ops with the contents of a ROCKE_VEC of rocke_op_t* (arena-owned
 * backing). */
static void region_set_ops(rocke_region_t* region, rocke_op_t** data, int count)
{
    region->ops = data;
    region->num_ops = count;
    /* cap_ops tracks the builder's growth capacity; set it to the new count so
     * the array is treated as exactly full (any later builder append would
     * re-grow). */
    region->cap_ops = count;
}

/* ------------------------------------------------- eliminate_dead_pure_ops */

int rocke_eliminate_dead_pure_ops(rocke_ir_builder_t* b, rocke_region_t* region)
{
    use_map_t uses;
    int removed = 0;
    int i, j;
    int rc;
    ROCKE_VEC(rocke_op_t*) kept;

    if(!region)
        return 0;

    use_init(&uses, b);
    count_uses_region(region, &uses);

    rocke_vec_init(&kept);
    for(i = 0; i < region->num_ops; i++)
    {
        rocke_op_t* op = region->ops[i];
        bool all_unused;

        for(j = 0; j < op->num_regions; j++)
            removed += rocke_eliminate_dead_pure_ops(b, op->regions[j]);

        all_unused = (op->num_results > 0);
        for(j = 0; j < op->num_results; j++)
        {
            const char* rn = op->results[j] ? op->results[j]->name : NULL;
            if(use_get(&uses, rn) != 0)
            {
                all_unused = false;
                break;
            }
        }
        if(rocke_op_is_pure(op) && op->num_results > 0 && all_unused)
        {
            removed += 1;
            continue;
        }
        rocke_vec_push(&b->arena, &kept, op, rc);
        if(rc != 0)
        {
            /* Arena OOM: bail out, leaving the region as-is for this op onward.
             * The public passes API has no sticky-error setter; the builder
             * arena rarely OOMs in practice. */
            return removed;
        }
    }
    region_set_ops(region, kept.data, (int)kept.len);
    return removed;
}

/* ------------------------------------------------------- canonicalize_region */

rocke_pass_stats_t rocke_canonicalize_region(rocke_ir_builder_t* b, rocke_region_t* region)
{
    rocke_pass_stats_t stats = pass_stats_zero();
    repl_map_t repl;
    cse_table_t cse;
    int folded = 0, cse_count = 0, dead;
    int i, j;
    int rc;
    ROCKE_VEC(rocke_op_t*) new_ops;

    if(!region)
        return stats;

    /* Python: recurse into nested regions of every op first, accumulating. */
    for(i = 0; i < region->num_ops; i++)
    {
        rocke_op_t* op = region->ops[i];
        for(j = 0; j < op->num_regions; j++)
        {
            rocke_pass_stats_t nested = rocke_canonicalize_region(b, op->regions[j]);
            stats = rocke_pass_stats_add(stats, nested);
        }
    }

    repl_init(&repl, b);
    cse_init(&cse, b);
    rocke_vec_init(&new_ops);

    for(i = 0; i < region->num_ops; i++)
    {
        rocke_op_t* op = region->ops[i];
        int64_t folded_value;

        rewrite_operands(b, op, &repl);

        if(try_fold(op, &folded_value) && op->num_results == 1)
        {
            fold_to_constant(b, op, folded_value);
            folded += 1;
        }

        if(rocke_op_is_pure(op) && op->num_results == 1)
        {
            rocke_value_t* cached = cse_lookup(&cse, op);
            if(cached != NULL)
            {
                if(op->results[0])
                    repl_set(&repl, op->results[0]->name, cached);
                cse_count += 1;
                continue; /* drop this op */
            }
            cse_insert(&cse, op, op->results[0]);
        }

        rocke_vec_push(&b->arena, &new_ops, op, rc);
        if(rc != 0)
        {
            /* Arena OOM: stop rebuilding. Commit whatever was kept so the
             * region stays self-consistent, then return current stats. */
            region_set_ops(region, new_ops.data, (int)new_ops.len);
            return stats;
        }
    }

    region_set_ops(region, new_ops.data, (int)new_ops.len);

    /* Python: _rewrite_region_operands(region, replacements). */
    rewrite_region_operands(b, region, &repl);

    dead = rocke_eliminate_dead_pure_ops(b, region);

    {
        rocke_pass_stats_t local;
        local.constants_folded = stats.constants_folded + folded;
        local.common_subexpressions = stats.common_subexpressions + cse_count;
        local.dead_ops_removed = stats.dead_ops_removed + dead;
        return local;
    }
}

/* ------------------------------------------------------------ optimize_kernel */

rocke_pass_stats_t
    rocke_optimize_kernel(rocke_ir_builder_t* b, rocke_kernel_def_t* kernel, int max_iter)
{
    rocke_pass_stats_t total = pass_stats_zero();
    int iter;

    if(max_iter <= 0)
        max_iter = 3; /* Python default max_iter=3 */
    if(!kernel || !kernel->body)
        return total;

    for(iter = 0; iter < max_iter; iter++)
    {
        rocke_pass_stats_t stats = rocke_canonicalize_region(b, kernel->body);
        total = rocke_pass_stats_add(total, stats);
        if(rocke_pass_stats_is_zero(stats))
            break;
    }
    return total;
}
