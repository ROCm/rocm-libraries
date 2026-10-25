// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_llvm/index_facts.cpp -- the global address split of the C++ port of
 * rocke.core.lower_llvm: Python _IndexFacts (which values are wave-uniform,
 * bounds on i32 values) and _Lowerer._collect_index_facts / _global_gep /
 * _split_global_index. Each rule mirrors its Python counterpart so both engines
 * split exactly the same accesses into exactly the same GEP chains.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm_internal.h"

namespace ckc
{

static const int64_t LL_I32_MIN = -INT64_C(2147483647) - 1;
static const int64_t LL_I32_MAX = INT64_C(2147483647);

/* Bounds of an i32 value. Operands of every bound computation are i32 bounds,
 * so sums, products and shifts of them fit int64. */
typedef struct ll_bounds
{
    int64_t lo;
    int64_t hi;
} ll_bounds_t;

/* One value's facts; `value` NULL marks an empty table slot. `bounded` false is
 * Python's None (no recorded bounds). */
typedef struct ll_fact
{
    const rocke_value_t* value;
    bool uniform;
    bool bounded;
    ll_bounds_t bounds;
} ll_fact_t;

/* An scf.for counter or carried value (an entry of Python _block_values). */
typedef struct ll_block_value
{
    const rocke_op_t* loop;
    const rocke_value_t* value;
} ll_block_value_t;

struct rocke_ll_index_facts
{
    ll_fact_t* slots; /* open addressing keyed by value pointer; cap is 2^k */
    size_t cap;
    size_t len;
    ROCKE_VEC(ll_block_value_t) block_values;
    int64_t thread_id_max;
    int64_t lane_id_max;
};

/* Python _GEP_ELEM_BYTES; 0 for a type not listed. */
static int64_t ll_gep_elem_bytes(const char* elem_ty)
{
    static const struct
    {
        const char* ty;
        int64_t bytes;
    } table[] = {{"i8", 1},
                 {"i16", 2},
                 {"half", 2},
                 {"bfloat", 2},
                 {"i32", 4},
                 {"float", 4},
                 {"i64", 8},
                 {"double", 8}};
    for(const auto& row : table)
        if(strcmp(elem_ty, row.ty) == 0)
            return row.bytes;
    return 0;
}

static int64_t ll_min(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

static int64_t ll_max(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

static int64_t ll_abs(int64_t a)
{
    return a < 0 ? -a : a;
}

static int ll_bit_length(int64_t x)
{
    int n = 0;
    for(; x > 0; x >>= 1)
        n++;
    return n;
}

static bool ll_is_i32(const rocke_type_t* t)
{
    return t != NULL && t->kind == ROCKE_TYPE_SCALAR && t->scalar == ROCKE_SCALAR_I32;
}

static bool ll_is_arith(const rocke_op_t* op)
{
    const char* name = op->name ? op->name : rocke_opcode_name(op->opcode);
    return name != NULL && strncmp(name, "arith.", 6) == 0;
}

static void ll_push_value(rocke_lower_t* L, ROCKE_VEC(const rocke_value_t*) * v, const rocke_value_t* x)
{
    int rc = 0;
    rocke_vec_push(&L->arena, v, x, rc);
    if(rc != 0)
        rocke_ll_fail(L, ROCKE_ERR_OOM, "index facts: arena OOM");
}

/* Python _has_nsw: the lowered add / sub / mul carries nsw. */
static bool ll_has_nsw(const rocke_op_t* op)
{
    const char* wrap = rocke_attr_get_str(&op->attrs, "wrap");
    return op->num_results >= 1 && !rocke_type_is_unsigned(op->results[0]->type)
           && !(wrap != NULL && strcmp(wrap, "nuw") == 0);
}

/* Python _i32_constant. */
static bool ll_i32_constant(const rocke_value_t* v, int64_t* out)
{
    const rocke_op_t* op = v->op;
    if(op == NULL || op->opcode != ROCKE_OP_ARITH_CONSTANT || !ll_is_i32(v->type))
        return false;
    if(!rocke_attr_get_int(&op->attrs, "value", out))
    {
        double f = 0.0;
        rocke_attr_get_float(&op->attrs, "value", &f);
        *out = (int64_t)f;
    }
    return true;
}

static size_t ll_slot(const rocke_ll_index_facts_t* F, const rocke_value_t* v)
{
    const uint64_t h = (uint64_t)(uintptr_t)v * UINT64_C(0x9E3779B97F4A7C15);
    size_t i = (size_t)(h >> 32) & (F->cap - 1);
    while(F->slots[i].value != NULL && F->slots[i].value != v)
        i = (i + 1) & (F->cap - 1);
    return i;
}

static const ll_fact_t* ll_find(const rocke_ll_index_facts_t* F, const rocke_value_t* v)
{
    const ll_fact_t* f = &F->slots[ll_slot(F, v)];
    return f->value != NULL ? f : NULL;
}

static void ll_alloc_slots(rocke_lower_t* L, rocke_ll_index_facts_t* F, size_t cap)
{
    F->cap = cap;
    F->slots = (ll_fact_t*)rocke_arena_calloc(&L->arena, cap * sizeof(ll_fact_t));
    if(F->slots == NULL)
        rocke_ll_fail(L, ROCKE_ERR_OOM, "index facts: arena OOM");
}

/* Python _IndexFacts._record; `bounds` NULL records no bounds. */
static void ll_record(rocke_lower_t* L,
                      rocke_ll_index_facts_t* F,
                      const rocke_value_t* v,
                      bool uniform,
                      const ll_bounds_t* bounds)
{
    if(2 * (F->len + 1) > F->cap)
    {
        const ll_fact_t* old = F->slots;
        const size_t old_cap = F->cap;
        ll_alloc_slots(L, F, 2 * old_cap);
        for(size_t i = 0; i < old_cap; i++)
            if(old[i].value != NULL)
                F->slots[ll_slot(F, old[i].value)] = old[i];
    }
    ll_fact_t* f = &F->slots[ll_slot(F, v)];
    if(f->value == NULL)
    {
        f->value = v;
        F->len++;
    }
    f->uniform = uniform;
    f->bounded = bounds != NULL;
    f->bounds = bounds != NULL ? *bounds : ll_bounds_t{0, 0};
}

static bool ll_is_param(const rocke_lower_t* L, const rocke_value_t* v)
{
    if(v->name == NULL || v->name[0] != '%')
        return false;
    for(int i = 0; i < L->kernel->num_params; i++)
    {
        const rocke_param_t* p = L->kernel->params[i];
        if(p != NULL && p->name != NULL && strcmp(v->name + 1, p->name) == 0)
            return true;
    }
    return false;
}

/* Python _IndexFacts.uniform. */
static bool ll_uniform(const rocke_lower_t* L, const rocke_ll_index_facts_t* F, const rocke_value_t* v)
{
    const ll_fact_t* f = ll_find(F, v);
    if(f != NULL)
        return f->uniform;
    if(v->op == NULL)
        return ll_is_param(L, v);
    return v->op->opcode == ROCKE_OP_ARITH_CONSTANT;
}

/* Python _IndexFacts.bounds; false (Python None) for a non-i32 value. */
static bool ll_bounds(const rocke_ll_index_facts_t* F, const rocke_value_t* v, ll_bounds_t* out)
{
    if(!ll_is_i32(v->type))
        return false;
    const ll_fact_t* f = ll_find(F, v);
    if(f != NULL && f->bounded)
    {
        *out = f->bounds;
        return true;
    }
    int64_t c = 0;
    if(ll_i32_constant(v, &c) && c >= LL_I32_MIN && c <= LL_I32_MAX)
        *out = {c, c};
    else
        *out = {LL_I32_MIN, LL_I32_MAX};
    return true;
}

/* Python _IndexFacts._binary_bounds; false for Python's None. */
static bool ll_binary_bounds(const rocke_op_t* op, ll_bounds_t a, ll_bounds_t b, ll_bounds_t* out)
{
    const int64_t alo = a.lo, ahi = a.hi, blo = b.lo, bhi = b.hi;
    switch(op->opcode)
    {
    case ROCKE_OP_ARITH_SELECT:
        *out = {ll_min(alo, blo), ll_max(ahi, bhi)};
        return true;
    case ROCKE_OP_ARITH_ADD:
        if(!ll_has_nsw(op) && (alo < 0 || blo < 0 || ahi + bhi > LL_I32_MAX))
            return false;
        *out = {alo + blo, ahi + bhi};
        return true;
    case ROCKE_OP_ARITH_SUB:
        *out = {alo - bhi, ahi - blo};
        return true;
    case ROCKE_OP_ARITH_MUL:
    {
        const int64_t c0 = alo * blo, c1 = alo * bhi, c2 = ahi * blo, c3 = ahi * bhi;
        *out = {ll_min(ll_min(c0, c1), ll_min(c2, c3)), ll_max(ll_max(c0, c1), ll_max(c2, c3))};
        return true;
    }
    case ROCKE_OP_ARITH_DIV:
        /* C++ division truncates toward zero, as sdiv does. */
        if(blo == bhi && blo != 0)
        {
            const int64_t e0 = alo / blo, e1 = ahi / blo;
            *out = {ll_min(e0, e1), ll_max(e0, e1)};
            return true;
        }
        if(alo >= 0 && blo >= 1)
        {
            *out = {alo / bhi, ahi / blo};
            return true;
        }
        return false;
    case ROCKE_OP_ARITH_MOD:
    {
        /* The remainder takes the dividend's sign and is smaller in magnitude
         * than the divisor. */
        const int64_t m = ll_max(ll_abs(blo), ll_abs(bhi)) - 1;
        if(m < 0)
            return false;
        *out = {alo < 0 ? ll_max(alo, -m) : 0, ahi > 0 ? ll_min(ahi, m) : 0};
        return true;
    }
    case ROCKE_OP_ARITH_AND:
        if(alo >= 0 && blo >= 0)
        {
            *out = {0, ll_min(ahi, bhi)};
            return true;
        }
        if(alo >= 0 || blo >= 0)
        {
            *out = {0, alo >= 0 ? ahi : bhi};
            return true;
        }
        return false;
    case ROCKE_OP_ARITH_OR:
    case ROCKE_OP_ARITH_XOR:
        if(alo >= 0 && blo >= 0)
        {
            *out = {0, (INT64_C(1) << ll_bit_length(ll_max(ahi, bhi))) - 1};
            return true;
        }
        return false;
    case ROCKE_OP_ARITH_SHL:
        if(blo == bhi && blo >= 0 && blo < 32 && alo >= 0 && (ahi << blo) <= LL_I32_MAX)
        {
            *out = {alo << blo, ahi << blo};
            return true;
        }
        return false;
    case ROCKE_OP_ARITH_LSHR:
        if(blo == bhi && blo >= 0 && blo < 32)
        {
            if(alo >= 0)
            {
                *out = {alo >> blo, ahi >> blo};
                return true;
            }
            if(blo >= 1)
            {
                *out = {0, ((INT64_C(1) << 32) - 1) >> blo};
                return true;
            }
            return false;
        }
        if(alo >= 0)
        {
            *out = {0, ahi};
            return true;
        }
        return false;
    case ROCKE_OP_ARITH_SMAX:
        *out = {ll_max(alo, blo), ll_max(ahi, bhi)};
        return true;
    case ROCKE_OP_ARITH_SMIN:
        *out = {ll_min(alo, blo), ll_min(ahi, bhi)};
        return true;
    case ROCKE_OP_ARITH_UMAX:
        if(alo >= 0 && blo >= 0)
        {
            *out = {ll_max(alo, blo), ll_max(ahi, bhi)};
            return true;
        }
        return false;
    case ROCKE_OP_ARITH_UMIN:
        if(alo >= 0 && blo >= 0)
        {
            *out = {ll_min(alo, blo), ll_min(ahi, bhi)};
            return true;
        }
        if(alo >= 0 || blo >= 0)
        {
            *out = {0, alo >= 0 ? ahi : bhi};
            return true;
        }
        return false;
    default:
        return false;
    }
}

/* Python _IndexFacts._op_bounds; false for Python's None. */
static bool ll_op_bounds(const rocke_ll_index_facts_t* F, const rocke_op_t* op, ll_bounds_t* out)
{
    if(!ll_is_i32(op->results[0]->type))
        return false;
    if(op->opcode == ROCKE_OP_GPU_THREAD_ID)
    {
        *out = {0, F->thread_id_max};
        return true;
    }
    if(op->opcode == ROCKE_OP_TILE_LANE_ID)
    {
        *out = {0, F->lane_id_max};
        return true;
    }
    if(op->opcode == ROCKE_OP_ARITH_ASSUME_RANGE)
        return op->num_operands >= 1 && ll_bounds(F, op->operands[0], out);
    if(op->opcode == ROCKE_OP_ARITH_ZEXT && op->num_operands == 1
       && op->operands[0]->type->kind == ROCKE_TYPE_SCALAR
       && op->operands[0]->type->scalar == ROCKE_SCALAR_I1)
    {
        *out = {0, 1};
        return true;
    }
    ll_bounds_t a = {0, 0}, b = {0, 0};
    if(op->opcode == ROCKE_OP_ARITH_SELECT && op->num_operands == 3)
    {
        if(!ll_bounds(F, op->operands[1], &a) || !ll_bounds(F, op->operands[2], &b))
            return false;
    }
    else if(ll_is_arith(op) && op->num_operands == 2)
    {
        if(!ll_bounds(F, op->operands[0], &a) || !ll_bounds(F, op->operands[1], &b))
            return false;
    }
    else
    {
        return false;
    }
    ll_bounds_t got = {0, 0};
    if(!ll_binary_bounds(op, a, b, &got))
        return false;
    const int64_t lo = ll_max(got.lo, LL_I32_MIN), hi = ll_min(got.hi, LL_I32_MAX);
    /* An empty range means the op always overflows into poison. */
    if(lo > hi)
        return false;
    *out = {lo, hi};
    return true;
}

/* Python _IndexFacts._op_uniform. */
static bool ll_op_uniform(const rocke_lower_t* L, const rocke_ll_index_facts_t* F, const rocke_op_t* op)
{
    switch(op->opcode)
    {
    case ROCKE_OP_GPU_BLOCK_ID:
    case ROCKE_OP_GPU_CLUSTER_ID:
    case ROCKE_OP_GPU_CLUSTER_WORKGROUP_ID:
    case ROCKE_OP_GPU_CLUSTER_WORKGROUP_MAX_ID:
    case ROCKE_OP_GPU_CLUSTER_WORKGROUP_FLAT_ID:
    case ROCKE_OP_GPU_CLUSTER_WORKGROUP_MAX_FLAT_ID:
    case ROCKE_OP_TILE_READFIRSTLANE:
    case ROCKE_OP_TILE_READLANE:
    case ROCKE_OP_TILE_WAVE_ALL:
    case ROCKE_OP_TILE_WAVE_ANY:
    case ROCKE_OP_TILE_WAVE_BALLOT:
    case ROCKE_OP_TILE_WAVE_REDUCE:
        return true;
    case ROCKE_OP_TILE_GLOBAL_PTR_ADD:
    case ROCKE_OP_TILE_GLOBAL_PTR_TO_I64:
    case ROCKE_OP_TILE_PIN_SGPR:
        break;
    default:
        if(!ll_is_arith(op))
            return false;
        break;
    }
    for(int i = 0; i < op->num_operands; i++)
        if(!ll_uniform(L, F, op->operands[i]))
            return false;
    return true;
}

/* Python _IndexFacts._counter_bounds: the body runs for lower <= iv < upper
 * when the step is positive. */
static bool ll_counter_bounds(const rocke_ll_index_facts_t* F,
                              const rocke_value_t* iv,
                              const rocke_value_t* lower,
                              const rocke_value_t* upper,
                              const rocke_value_t* step,
                              ll_bounds_t* out)
{
    ll_bounds_t lo_b = {0, 0}, hi_b = {0, 0}, step_b = {0, 0};
    if(!ll_is_i32(iv->type) || !ll_bounds(F, lower, &lo_b) || !ll_bounds(F, upper, &hi_b)
       || !ll_bounds(F, step, &step_b) || step_b.lo < 1)
        return false;
    *out = {lo_b.lo, ll_max(lo_b.lo, hi_b.hi - 1)};
    return true;
}

static bool ll_is_result(const rocke_op_t* op, const rocke_value_t* v)
{
    for(int i = 0; i < op->num_results; i++)
        if(op->results[i] == v)
            return true;
    return false;
}

/* Python _IndexFacts._find_block_values. */
static void ll_find_block_values(rocke_lower_t* L, rocke_ll_index_facts_t* F, const rocke_region_t* region)
{
    for(int i = 0; i < region->num_ops; i++)
    {
        const rocke_op_t* op = region->ops[i];
        for(int j = 0; j < op->num_operands; j++)
        {
            const rocke_value_t* v = op->operands[j];
            const rocke_op_t* loop = v->op;
            if(loop == NULL || loop->opcode != ROCKE_OP_SCF_FOR || ll_is_result(loop, v))
                continue;
            bool known = false;
            for(size_t k = 0; k < F->block_values.len && !known; k++)
                known = F->block_values.data[k].value == v;
            if(known)
                continue;
            int rc = 0;
            rocke_vec_push(&L->arena, &F->block_values, (ll_block_value_t{loop, v}), rc);
            if(rc != 0)
                rocke_ll_fail(L, ROCKE_ERR_OOM, "index facts: arena OOM");
        }
        for(int r = 0; r < op->num_regions; r++)
            if(op->regions[r] != NULL)
                ll_find_block_values(L, F, op->regions[r]);
    }
}

/* The counter or carried value of `loop` named `name`, or NULL. */
static const rocke_value_t*
    ll_block_value(const rocke_ll_index_facts_t* F, const rocke_op_t* loop, const char* name)
{
    if(name == NULL)
        return NULL;
    for(size_t k = 0; k < F->block_values.len; k++)
    {
        const ll_block_value_t* bv = &F->block_values.data[k];
        if(bv->loop == loop && bv->value->name != NULL && strcmp(bv->value->name, name) == 0)
            return bv->value;
    }
    return NULL;
}

static void ll_visit(rocke_lower_t* L, rocke_ll_index_facts_t* F, const rocke_region_t* region);

/* Python _IndexFacts._visit_for. */
static void ll_visit_for(rocke_lower_t* L, rocke_ll_index_facts_t* F, const rocke_op_t* op)
{
    if(op->num_operands < 3 || op->num_regions < 1 || op->regions[0] == NULL)
        return;
    const rocke_value_t* lower = op->operands[0];
    const rocke_value_t* upper = op->operands[1];
    const rocke_value_t* step = op->operands[2];
    int64_t num_iter = 0;
    rocke_attr_get_int(&op->attrs, "num_iter_args", &num_iter);
    const int num_inits = (int)ll_max(0, ll_min(num_iter, op->num_operands - 3));
    const rocke_region_t* body = op->regions[0];
    const bool counted
        = ll_uniform(L, F, lower) && ll_uniform(L, F, upper) && ll_uniform(L, F, step);
    const rocke_value_t* iv = ll_block_value(F, op, rocke_attr_get_str(&op->attrs, "iv"));
    if(iv != NULL)
    {
        ll_bounds_t b = {0, 0};
        const bool bounded = ll_counter_bounds(F, iv, lower, upper, step, &b);
        ll_record(L, F, iv, counted, bounded ? &b : NULL);
    }
    const rocke_attr_value_t* meta = rocke_attr_get(&op->attrs, "iter_args");
    const int num_meta = (meta != NULL && meta->kind == ROCKE_ATTR_LIST) ? meta->u.list.count : 0;
    const int num_carried = num_meta < num_inits ? num_meta : num_inits;
    const rocke_op_t* yield = NULL;
    for(int i = body->num_ops - 1; i >= 0 && yield == NULL; i--)
        if(body->ops[i]->opcode == ROCKE_OP_SCF_YIELD)
            yield = body->ops[i];
    const int num_yielded = yield != NULL ? yield->num_operands : 0;
    bool* assumed = (bool*)rocke_arena_calloc(&L->arena, sizeof(bool) * (size_t)(num_inits + 1));
    if(assumed == NULL)
        rocke_ll_fail(L, ROCKE_ERR_OOM, "index facts: arena OOM");
    for(int i = 0; i < num_inits; i++)
        assumed[i] = counted && ll_uniform(L, F, op->operands[3 + i]);
    /* A carried value is uniform if its initial value is and every yield is;
     * assume so, then drop the assumption where a yield disagrees. */
    for(;;)
    {
        for(int k = 0; k < num_carried; k++)
        {
            const rocke_attr_map_t* m = meta->u.list.items[k];
            const rocke_value_t* v
                = m != NULL ? ll_block_value(F, op, rocke_attr_get_str(m, "name")) : NULL;
            if(v != NULL)
                ll_record(L, F, v, assumed[k], NULL);
        }
        ll_visit(L, F, body);
        bool changed = false;
        for(int i = 0; i < num_inits; i++)
        {
            const bool held
                = assumed[i] && i < num_yielded && ll_uniform(L, F, yield->operands[i]);
            changed = changed || held != assumed[i];
            assumed[i] = held;
        }
        if(!changed)
            break;
    }
    for(int i = 0; i < num_inits && i < op->num_results; i++)
        ll_record(L, F, op->results[i], assumed[i], NULL);
}

/* Python _IndexFacts._visit. */
static void ll_visit(rocke_lower_t* L, rocke_ll_index_facts_t* F, const rocke_region_t* region)
{
    for(int i = 0; i < region->num_ops; i++)
    {
        const rocke_op_t* op = region->ops[i];
        if(op->opcode == ROCKE_OP_SCF_FOR)
        {
            ll_visit_for(L, F, op);
            continue;
        }
        for(int r = 0; r < op->num_regions; r++)
            if(op->regions[r] != NULL)
                ll_visit(L, F, op->regions[r]);
        if(op->num_results == 1)
        {
            ll_bounds_t b = {0, 0};
            const bool bounded = ll_op_bounds(F, op, &b);
            ll_record(L, F, op->results[0], ll_op_uniform(L, F, op), bounded ? &b : NULL);
        }
    }
}

void rocke_ll_collect_index_facts(rocke_lower_t* L)
{
    if(!L->backend->splits_global_address)
        return;
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(L->backend->gfx);
    if(target == NULL)
        rocke_ll_fail(L, ROCKE_ERR_KEY, "arch metadata not present for %s", L->backend->gfx);
    rocke_ll_index_facts_t* F
        = (rocke_ll_index_facts_t*)rocke_arena_calloc(&L->arena, sizeof(rocke_ll_index_facts_t));
    if(F == NULL)
        rocke_ll_fail(L, ROCKE_ERR_OOM, "index facts: arena OOM");
    ll_alloc_slots(L, F, 1024);
    rocke_vec_init(&F->block_values);
    F->thread_id_max = rocke_kernel_max_workgroup_size(L->kernel) - 1;
    F->lane_id_max = target->wave_size - 1;
    ll_find_block_values(L, F, L->kernel->body);
    ll_visit(L, F, L->kernel->body);
    L->index_facts = F;
}

/* Python _IndexFacts.split_index: appends the wave-uniform terms of `idx` to
 * `uniform` and returns its one per-lane term, or NULL. */
static const rocke_value_t* ll_split_index(rocke_lower_t* L,
                                           const rocke_ll_index_facts_t* F,
                                           const rocke_value_t* idx,
                                           ROCKE_VEC(const rocke_value_t*) * uniform)
{
    ROCKE_VEC(const rocke_value_t*) stack;
    rocke_vec_init(&stack);
    ll_push_value(L, &stack, idx);
    const rocke_value_t* lane = NULL;
    while(stack.len > 0)
    {
        const rocke_value_t* v = stack.data[--stack.len];
        const rocke_op_t* op = v->op;
        if(ll_uniform(L, F, v))
            ll_push_value(L, uniform, v);
        else if(op != NULL && op->opcode == ROCKE_OP_ARITH_ASSUME_RANGE && op->num_operands >= 1)
            ll_push_value(L, &stack, op->operands[0]);
        else if(op != NULL && op->opcode == ROCKE_OP_ARITH_ADD && op->num_operands == 2
                && ll_has_nsw(op))
        {
            ll_push_value(L, &stack, op->operands[1]);
            ll_push_value(L, &stack, op->operands[0]);
        }
        else if(lane == NULL)
            lane = v;
        else
            return NULL;
    }
    return lane;
}

/* Python _Lowerer._split_global_index: on success fills the non-constant
 * uniform terms, the sum of the constant ones, and the per-lane term. */
static bool ll_split_global_index(rocke_lower_t* L,
                                  const char* elem_ty,
                                  const rocke_value_t* ptr,
                                  const rocke_value_t* idx,
                                  const char* idx_ty,
                                  ROCKE_VEC(const rocke_value_t*) * terms,
                                  int64_t* constant,
                                  const rocke_value_t** lane)
{
    const rocke_ll_index_facts_t* F = L->index_facts;
    const int64_t elem_bytes = F != NULL ? ll_gep_elem_bytes(elem_ty) : 0;
    if(F == NULL || elem_bytes == 0 || strcmp(idx_ty, "i32") != 0 || !ll_is_i32(idx->type)
       || strcmp(rocke_ll_value_ptr_type(L, ptr), "ptr addrspace(1)") != 0
       || !ll_uniform(L, F, ptr))
        return false;
    ROCKE_VEC(const rocke_value_t*) uniform;
    rocke_vec_init(&uniform);
    *lane = ll_split_index(L, F, idx, &uniform);
    if(*lane == NULL)
        return false;
    *constant = 0;
    for(size_t i = 0; i < uniform.len; i++)
    {
        int64_t c = 0;
        if(ll_i32_constant(uniform.data[i], &c))
            *constant += c;
        else
            ll_push_value(L, terms, uniform.data[i]);
    }
    ll_bounds_t lane_bounds = {0, 0};
    return terms->len > 0 && ll_bounds(F, *lane, &lane_bounds) && lane_bounds.lo >= 0
           && lane_bounds.hi * elem_bytes < (INT64_C(1) << 32);
}

void rocke_ll_global_gep(rocke_lower_t* L,
                         const char* gep,
                         const char* elem_ty,
                         const rocke_value_t* ptr,
                         const rocke_value_t* idx,
                         const char* idx_ty)
{
    ROCKE_VEC(const rocke_value_t*) terms;
    rocke_vec_init(&terms);
    int64_t constant = 0;
    const rocke_value_t* lane = NULL;
    if(!ll_split_global_index(L, elem_ty, ptr, idx, idx_ty, &terms, &constant, &lane))
    {
        rocke_ll_emitf(L,
                       "  %s = getelementptr inbounds %s, ptr addrspace(1) %s, %s %s",
                       gep,
                       elem_ty,
                       rocke_ll_operand(L, ptr),
                       idx_ty,
                       rocke_ll_operand(L, idx));
        return;
    }
    const char* base = rocke_ll_operand(L, ptr);
    for(size_t i = 0; i <= terms.len; i++)
    {
        const char* index;
        if(i < terms.len)
            index = rocke_arena_printf(&L->arena, "i32 %s", rocke_ll_operand(L, terms.data[i]));
        else if(constant != 0)
            index = rocke_arena_printf(&L->arena, "i64 %lld", (long long)constant);
        else
            break;
        const char* term_base = rocke_ll_fresh(L, "gbase");
        rocke_ll_emitf(
            L, "  %s = getelementptr %s, ptr addrspace(1) %s, %s", term_base, elem_ty, base, index);
        base = term_base;
    }
    rocke_ll_emitf(L,
                   "  %s = getelementptr %s, ptr addrspace(1) %s, i32 %s",
                   gep,
                   elem_ty,
                   base,
                   rocke_ll_operand(L, lane));
}

} /* namespace ckc */
