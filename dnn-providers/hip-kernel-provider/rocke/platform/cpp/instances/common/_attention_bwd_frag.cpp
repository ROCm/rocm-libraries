// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_bwd_frag.py: lane-map facts, the
 * checked C-to-A register relabel, the K-permuted operand load, the LDS
 * fragment loaders and the LDS views of the attention backward.
 *
 * See rocke/helper_rocke.instances.common._attention_bwd_frag.h for the mapping
 * and the byte-identity rules. Every op is created in the Python evaluation
 * order; each comment `py:` quotes the Python expression a block reproduces.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_frag.h"

#include <stdio.h>
#include <string.h>

#include <map>
#include <vector>

#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ====================================================================== *
 * Host integer helpers (Python semantics)
 * ====================================================================== */

static int64_t py_floordiv(int64_t a, int64_t b)
{
    int64_t q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        --q;
    return q;
}

static int64_t py_mod(int64_t a, int64_t b)
{
    int64_t r = a % b;
    if(r != 0 && ((r < 0) != (b < 0)))
        r += b;
    return r;
}

static int64_t py_gcd(int64_t a, int64_t b)
{
    if(a < 0)
        a = -a;
    if(b < 0)
        b = -b;
    while(b != 0)
    {
        int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static int bit_length(int64_t n)
{
    int r = 0;
    while(n > 0)
    {
        ++r;
        n >>= 1;
    }
    return r;
}

static bool live(rocke_ir_builder_t* b)
{
    return b != NULL && b->status == ROCKE_OK;
}

/* ====================================================================== *
 * Integer evaluation of the catalog lane maps (IntLaneBuilder)
 * ====================================================================== */

/* Fold the integer program that produced `v` (the ops IntLaneBuilder
 * implements). */
static bool int_eval(const rocke_value_t* v, int64_t* out)
{
    if(v == NULL || v->op == NULL)
        return false;
    const rocke_op_t* op = v->op;
    if(op->opcode == ROCKE_OP_ARITH_CONSTANT)
        return rocke_attr_get_int(&op->attrs, "value", out);
    if(op->num_operands != 2)
        return false;
    int64_t a, c;
    if(!int_eval(op->operands[0], &a) || !int_eval(op->operands[1], &c))
        return false;
    switch(op->opcode)
    {
    case ROCKE_OP_ARITH_ADD:
        *out = a + c;
        return true;
    case ROCKE_OP_ARITH_SUB:
        *out = a - c;
        return true;
    case ROCKE_OP_ARITH_MUL:
        *out = a * c;
        return true;
    case ROCKE_OP_ARITH_DIV:
        if(c == 0)
            return false;
        *out = py_floordiv(a, c);
        return true;
    case ROCKE_OP_ARITH_MOD:
        if(c == 0)
            return false;
        *out = py_mod(a, c);
        return true;
    case ROCKE_OP_ARITH_XOR:
        *out = a ^ c;
        return true;
    case ROCKE_OP_ARITH_SHL:
        *out = a << c;
        return true;
    case ROCKE_OP_ARITH_LSHR:
        *out = a >> c;
        return true;
    case ROCKE_OP_ARITH_AND:
        *out = a & c;
        return true;
    default:
        return false;
    }
}

/* Python @cache of _eval_cached, keyed on the layout map's address. Sound
 * because every rocke_layout_map_t an atom points at is static catalog
 * storage (arch_target data), never freed or reused, as Python's cache key
 * (the MmaOp and role) is immutable. */
static thread_local std::map<const rocke_layout_map_t*, std::vector<int>> g_eval_cache;

static const std::vector<int>* eval_cached(rocke_ir_builder_t* b, const rocke_layout_map_t* m)
{
    if(!live(b))
        return NULL;
    if(m == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "no lane map to evaluate");
        return NULL;
    }
    auto it = g_eval_cache.find(m);
    if(it != g_eval_cache.end())
        return &it->second;
    if(m->wave_size > ROCKE_ATTN_BWD_FRAG_MAX_LANES || m->frag_len > ROCKE_ATTN_BWD_FRAG_MAX_SLOTS
       || m->wave_size < 1 || m->frag_len < 1)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "lane map of %d lanes x %d slots exceeds the backward fragment "
                        "tables (%d x %d)",
                        m->wave_size,
                        m->frag_len,
                        ROCKE_ATTN_BWD_FRAG_MAX_LANES,
                        ROCKE_ATTN_BWD_FRAG_MAX_SLOTS);
        return NULL;
    }
    std::vector<int> t((size_t)m->wave_size * m->frag_len * 2);
    rocke_ir_builder_t sb;
    if(rocke_ir_builder_init(&sb, "int_lane_eval") != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "lane-map evaluation builder failed");
        return NULL;
    }
    /* Freed on every exit, including a ckc::Error thrown by the layout map
     * or by rocke_i_set_err below. */
    struct ScratchGuard
    {
        rocke_ir_builder_t* p;
        ~ScratchGuard()
        {
            rocke_ir_builder_free(p);
        }
    } guard{&sb};
    bool ok = true;
    for(int lane = 0; lane < m->wave_size && ok; ++lane)
    {
        rocke_value_t* lv = rocke_b_const_i32(&sb, lane);
        for(int s = 0; s < m->frag_len && ok; ++s)
        {
            rocke_value_t *c0 = NULL, *c1 = NULL;
            int64_t x0, x1;
            if(!rocke_layout_map_coord(m, &sb, lv, s, &c0, &c1) || !int_eval(c0, &x0)
               || !int_eval(c1, &x1))
            {
                ok = false;
                break;
            }
            t[((size_t)lane * m->frag_len + s) * 2] = (int)x0;
            t[((size_t)lane * m->frag_len + s) * 2 + 1] = (int)x1;
        }
    }
    if(!ok)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "lane map is not an integer program of the lane");
        return NULL;
    }
    return &(g_eval_cache[m] = std::move(t));
}

bool rocke_attn_bwd_eval_layout(rocke_ir_builder_t* b,
                                const rocke_layout_map_t* layout,
                                int* out,
                                int cap)
{
    const std::vector<int>* t = eval_cached(b, layout);
    if(t == NULL)
        return false;
    if(out == NULL || cap < (int)t->size())
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "eval_layout output holds %d ints, needs %d", cap, (int)t->size());
        return false;
    }
    memcpy(out, t->data(), t->size() * sizeof(int));
    return true;
}

/* _layout(op, role) */
static const rocke_layout_map_t*
    op_layout(rocke_ir_builder_t* b, const rocke_mma_op_t* op, char role)
{
    if(!live(b))
        return NULL;
    if(op == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "no MMA atom");
        return NULL;
    }
    if(role == 'a')
        return rocke_mma_op_a_layout(op, b);
    if(role == 'b')
        return rocke_mma_op_b_layout(op, b);
    return rocke_mma_op_c_layout(op, b);
}

/* A [lane][slot] -> (c0, c1) table. */
struct Ev
{
    const std::vector<int>* t = NULL;
    int ws = 0;
    int fl = 0;
    int at(int lane, int s, int i) const
    {
        return (*t)[((size_t)lane * fl + s) * 2 + i];
    }
};

/* _eval(op, role) */
static bool eval_op(rocke_ir_builder_t* b, const rocke_mma_op_t* op, char role, Ev* ev)
{
    const rocke_layout_map_t* m = op_layout(b, op, role);
    if(m == NULL)
    {
        if(live(b))
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "missing lane map");
        return false;
    }
    ev->t = eval_cached(b, m);
    if(ev->t == NULL)
        return false;
    ev->ws = m->wave_size;
    ev->fl = m->frag_len;
    return true;
}

/* _mn_k(role, coord) */
static void mn_k(char role, int c0, int c1, int* mn, int* k)
{
    if(role == 'a')
    {
        *mn = c0;
        *k = c1;
    }
    else
    {
        *mn = c1;
        *k = c0;
    }
}

/* ====================================================================== *
 * Atom facts
 * ====================================================================== */

static bool run_check(const Ev& ev, int c_frag_len, int run)
{
    if(run == 0 || c_frag_len % run)
        return false;
    for(int lane = 0; lane < ev.ws; ++lane)
    {
        int r0 = ev.at(lane, 0, 0), c0 = ev.at(lane, 0, 1);
        for(int s = 0; s < ev.fl; ++s)
            if(ev.at(lane, s, 0) != r0 + s || ev.at(lane, s, 1) != c0)
                return false;
    }
    return true;
}

bool rocke_attn_bwd_c_row_runs(rocke_ir_builder_t* b, const rocke_mma_op_t* op, int run, bool* out)
{
    *out = false;
    if(!live(b))
        return false;
    if(op == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "no MMA atom");
        return false;
    }
    if(run == 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "integer modulo by zero");
        return false;
    }
    /* py: if op.c_frag_len % run: return False */
    if(py_mod(op->c_frag_len, run))
        return true;
    Ev ev;
    if(!eval_op(b, op, 'c', &ev))
        return false;
    *out = run_check(ev, op->c_frag_len, run);
    return true;
}

bool rocke_attn_bwd_c_quad_transposable(rocke_ir_builder_t* b, const rocke_mma_op_t* op, bool* out)
{
    *out = false;
    if(!live(b))
        return false;
    if(op == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "no MMA atom");
        return false;
    }
    if(op->c_frag_len != 4)
        return true;
    bool runs = false;
    if(!rocke_attn_bwd_c_row_runs(b, op, 4, &runs))
        return false;
    if(!runs || op->wave_size % 4)
        return true;
    Ev ev;
    if(!eval_op(b, op, 'c', &ev))
        return false;
    for(int q = 0; q < op->wave_size; q += 4)
    {
        int r0 = ev.at(q, 0, 0), c0 = ev.at(q, 0, 1);
        if(py_mod(c0, 4))
            return true;
        for(int j = 0; j < 4; ++j)
            if(ev.at(q + j, 0, 0) != r0 || ev.at(q + j, 0, 1) != c0 + j)
                return true;
    }
    *out = true;
    return true;
}

static bool contig(const Ev& ev, char role)
{
    for(int lane = 0; lane < ev.ws; ++lane)
    {
        int mn0, k0;
        mn_k(role, ev.at(lane, 0, 0), ev.at(lane, 0, 1), &mn0, &k0);
        for(int s = 0; s < ev.fl; ++s)
        {
            int mn, k;
            mn_k(role, ev.at(lane, s, 0), ev.at(lane, s, 1), &mn, &k);
            if(mn != mn0 || k != k0 + s)
                return false;
        }
    }
    return true;
}

static bool affine(const Ev& ev)
{
    for(int lane = 1; lane < ev.ws; ++lane)
        for(int s = 0; s < ev.fl; ++s)
            for(int i = 0; i < 2; ++i)
                if(ev.at(lane, s, i) - ev.at(lane, 0, i) != ev.at(0, s, i) - ev.at(0, 0, i))
                    return false;
    return true;
}

bool rocke_attn_bwd_atom_facts(rocke_ir_builder_t* b,
                               const rocke_mma_op_t* op,
                               rocke_attn_bwd_atom_facts_t* out)
{
    memset(out, 0, sizeof(*out));
    Ev a, bb, c;
    if(!eval_op(b, op, 'a', &a) || !eval_op(b, op, 'b', &bb) || !eval_op(b, op, 'c', &c))
        return false;
    bool c_is_aT = op->c_frag_len == op->a_frag_len;
    for(int lane = 0; c_is_aT && lane < op->wave_size; ++lane)
        for(int s = 0; s < op->a_frag_len; ++s)
            if(c.at(lane, s, 0) != a.at(lane, s, 1) || c.at(lane, s, 1) != a.at(lane, s, 0))
            {
                c_is_aT = false;
                break;
            }
    bool ab_same_k = op->a_frag_len == op->b_frag_len;
    for(int lane = 0; ab_same_k && lane < op->wave_size; ++lane)
        for(int s = 0; s < op->a_frag_len; ++s)
            if(a.at(lane, s, 1) != bb.at(lane, s, 0))
            {
                ab_same_k = false;
                break;
            }
    out->op_id = op->op_id;
    out->a_contig = contig(a, 'a');
    out->b_contig = contig(bb, 'b');
    out->affine = affine(a) && affine(bb) && affine(c);
    out->c_is_aT = c_is_aT;
    out->ab_same_k = ab_same_k;
    return true;
}

/* ====================================================================== *
 * C-to-A relabel plan
 * ====================================================================== */

bool rocke_attn_bwd_relabel_plan_needs_k_perm(const rocke_attn_bwd_relabel_plan_t* plan)
{
    for(int k = 0; k < plan->n_k_perm; ++k)
        if(plan->k_perm[k] != k)
            return true;
    return false;
}

/* _logical_a: C (row, col) of tile `tile` -> consumer A (col, tile*rows+row). */
static void logical_a(int row, int col, int tile, int rows, int* r, int* k)
{
    *r = col;
    *k = tile * rows + row;
}

/* Simulation without a builder error: false is Python's ValueError (caught by
 * _proved). out[(lane * len + s) * 2 + i]. */
static bool simulate(const rocke_attn_bwd_relabel_plan_t* plan,
                     const Ev& c,
                     int ws,
                     std::vector<int>* out,
                     int* n_slots)
{
    out->clear();
    if(plan->kind == ROCKE_ATTN_BWD_RELABEL_DIRECT)
    {
        *n_slots = plan->n_slot_src;
        for(int lane = 0; lane < ws; ++lane)
            for(int s = 0; s < plan->n_slot_src; ++s)
            {
                int t = plan->slot_src[s][0], cs = plan->slot_src[s][1];
                int r, k;
                logical_a(c.at(lane, cs, 0), c.at(lane, cs, 1), t, plan->c_tile_rows, &r, &k);
                out->push_back(r);
                out->push_back(k);
            }
        return true;
    }
    if(plan->kind != ROCKE_ATTN_BWD_RELABEL_XLANE16)
        return false;
    /* bytes: each 16-bit element contributes (elem, 0) and (elem, 1); a byte
     * is (row, k, half). */
    struct Byte
    {
        int r, k, h;
    };
    int nvals = c.fl;
    auto dwords = [&](int lane, int j, Byte d[4]) {
        int r0, k0, r1, k1;
        logical_a(c.at(lane, 2 * j, 0), c.at(lane, 2 * j, 1), 0, plan->c_tile_rows, &r0, &k0);
        logical_a(
            c.at(lane, 2 * j + 1, 0), c.at(lane, 2 * j + 1, 1), 0, plan->c_tile_rows, &r1, &k1);
        d[0] = {r0, k0, 0};
        d[1] = {r0, k0, 1};
        d[2] = {r1, k1, 0};
        d[3] = {r1, k1, 1};
    };
    *n_slots = plan->n_xlane * 2;
    for(int lane = 0; lane < ws; ++lane)
    {
        for(int m = 0; m < plan->n_xlane; ++m)
        {
            int j = (int)plan->xlane[m][0];
            if(j < 0 || 2 * j + 1 >= nvals)
                return false;
            int64_t sel = lane / 16 == 0 ? plan->xlane[m][1] : plan->xlane[m][2];
            Byte own[4], partner[4], pool[8], d[4];
            dwords(lane, j, own);
            dwords(lane ^ 16, j, partner);
            /* _perm_sim(src0=own, src1=partner): bytes 0..3 from src1, 4..7 from src0 */
            for(int i = 0; i < 4; ++i)
            {
                pool[i] = partner[i];
                pool[4 + i] = own[i];
            }
            for(int i = 0; i < 4; ++i)
            {
                int idx = (int)((sel >> (8 * i)) & 0xFF);
                if(idx > 7)
                    return false;
                d[i] = pool[idx];
            }
            for(int half = 0; half < 2; ++half)
            {
                Byte lo = d[2 * half], hi = d[2 * half + 1];
                if(lo.r != hi.r || lo.k != hi.k || lo.h != 0 || hi.h != 1)
                    return false;
                out->push_back(lo.r);
                out->push_back(lo.k);
            }
        }
    }
    return true;
}

static bool proved(const rocke_attn_bwd_relabel_plan_t* plan, const Ev& c, const Ev& a)
{
    std::vector<int> got;
    int n_slots = 0;
    if(!simulate(plan, c, a.ws, &got, &n_slots))
        return false;
    if(n_slots < a.fl)
        return false;
    for(int lane = 0; lane < a.ws; ++lane)
        for(int s = 0; s < a.fl; ++s)
        {
            int row = a.at(lane, s, 0), k = a.at(lane, s, 1);
            if(k < 0 || k >= plan->n_k_perm)
                return false;
            if(got[((size_t)lane * n_slots + s) * 2] != row
               || got[((size_t)lane * n_slots + s) * 2 + 1] != plan->k_perm[k])
                return false;
        }
    /* Python compares got[lane][s] for s < a_frag_len only: extra simulated
     * slots are ignored. (Fewer slots, or a k outside k_perm, is an
     * IndexError in Python; both are refused here. Neither is reachable: the
     * plans come from try_direct / try_xlane16.) */
    return true;
}

static bool try_direct(const rocke_mma_op_t* c_op,
                       const rocke_mma_op_t* a_op,
                       int tiles,
                       const Ev& a,
                       const Ev& c,
                       rocke_attn_bwd_relabel_plan_t* plan)
{
    int cf = c_op->c_frag_len;
    if(a_op->a_frag_len != tiles * cf)
        return false;
    memset(plan, 0, sizeof(*plan));
    plan->n_slot_src = a_op->a_frag_len;
    for(int s = 0; s < a_op->a_frag_len; ++s)
    {
        plan->slot_src[s][0] = s / cf;
        plan->slot_src[s][1] = s % cf;
    }
    std::map<int, int> perm;
    for(int lane = 0; lane < a_op->wave_size; ++lane)
        for(int s = 0; s < plan->n_slot_src; ++s)
        {
            int t = plan->slot_src[s][0], cs = plan->slot_src[s][1];
            int row, k_log;
            logical_a(c.at(lane, cs, 0), c.at(lane, cs, 1), t, c_op->m, &row, &k_log);
            int a_row = a.at(lane, s, 0), a_k = a.at(lane, s, 1);
            if(row != a_row)
                return false;
            auto it = perm.find(a_k);
            if(it == perm.end())
                perm[a_k] = k_log;
            else if(it->second != k_log)
                return false;
        }
    /* sorted(perm) == range(k) and sorted(values) == range(k) */
    if((int)perm.size() != a_op->k)
        return false;
    {
        int i = 0;
        for(auto& kv : perm)
            if(kv.first != i++)
                return false;
        std::vector<int> vals;
        for(auto& kv : perm)
            vals.push_back(kv.second);
        std::vector<bool> seen(a_op->k, false);
        for(int v : vals)
        {
            if(v < 0 || v >= a_op->k || seen[v])
                return false;
            seen[v] = true;
        }
    }
    plan->c_op_id = c_op->op_id;
    plan->a_op_id = a_op->op_id;
    plan->kind = ROCKE_ATTN_BWD_RELABEL_DIRECT;
    plan->c_tiles = tiles;
    plan->c_tile_rows = c_op->m;
    plan->n_k_perm = a_op->k;
    for(int k = 0; k < a_op->k; ++k)
        plan->k_perm[k] = perm[k];
    return true;
}

static bool try_xlane16(const rocke_mma_op_t* c_op,
                        const rocke_mma_op_t* a_op,
                        int tiles,
                        const Ev& a,
                        const Ev& c,
                        rocke_attn_bwd_relabel_plan_t* plan)
{
    int ws = a_op->wave_size;
    if(ws != 32 || tiles != 1 || a_op->a_frag_len != 2 * c_op->c_frag_len)
        return false;
    if(a_op->a_frag_len % 2 || c_op->c_frag_len % 2)
        return false;
    /* held[lane][(row, k)] -> cslot (a dict: the last slot wins) */
    auto held = [&](int lane, int row, int k) -> int {
        int found = -1;
        for(int cs = 0; cs < c_op->c_frag_len; ++cs)
        {
            int r, kk;
            logical_a(c.at(lane, cs, 0), c.at(lane, cs, 1), 0, c_op->m, &r, &kk);
            if(r == row && kk == k)
                found = cs;
        }
        return found;
    };
    std::vector<std::pair<int, int>> per_half[2];
    bool have[2] = {false, false};
    for(int lane = 0; lane < ws; ++lane)
    {
        int h = lane / 16;
        std::vector<std::pair<int, int>> srcs;
        for(int s = 0; s < a_op->a_frag_len; ++s)
        {
            int nr = a.at(lane, s, 0), nk = a.at(lane, s, 1);
            int cs = held(lane, nr, nk);
            if(cs >= 0)
                srcs.push_back({0, cs});
            else if((cs = held(lane ^ 16, nr, nk)) >= 0)
                srcs.push_back({1, cs});
            else
                return false;
        }
        if(!have[h])
        {
            per_half[h] = srcs;
            have[h] = true;
        }
        else if(per_half[h] != srcs)
            return false;
    }
    memset(plan, 0, sizeof(*plan));
    plan->n_xlane = a_op->a_frag_len / 2;
    for(int m = 0; m < plan->n_xlane; ++m)
    {
        std::vector<int> js;
        int64_t sels[2];
        for(int h = 0; h < 2; ++h)
        {
            int64_t sel = 0;
            int ss[2] = {2 * m, 2 * m + 1};
            for(int pos = 0; pos < 2; ++pos)
            {
                int partner = per_half[h][ss[pos]].first, cs = per_half[h][ss[pos]].second;
                int j = cs / 2;
                bool dup = false;
                for(int x : js)
                    dup = dup || x == j;
                if(!dup)
                    js.push_back(j);
                int64_t byte0 = (partner ? 0 : 4) + 2 * (cs % 2);
                sel |= byte0 << (16 * pos);
                sel |= (byte0 + 1) << (16 * pos + 8);
            }
            sels[h] = sel;
        }
        if(js.size() != 1)
            return false;
        plan->xlane[m][0] = js[0];
        plan->xlane[m][1] = sels[0];
        plan->xlane[m][2] = sels[1];
    }
    plan->c_op_id = c_op->op_id;
    plan->a_op_id = a_op->op_id;
    plan->kind = ROCKE_ATTN_BWD_RELABEL_XLANE16;
    plan->c_tiles = 1;
    plan->c_tile_rows = c_op->m;
    plan->n_k_perm = a_op->k;
    for(int k = 0; k < a_op->k; ++k)
        plan->k_perm[k] = k;
    return proved(plan, c, a);
}

bool rocke_attn_bwd_plan_c_to_a(rocke_ir_builder_t* b,
                                const rocke_mma_op_t* c_op,
                                const rocke_mma_op_t* a_op,
                                rocke_attn_bwd_relabel_plan_t* out)
{
    if(!live(b))
        return false;
    if(c_op == NULL || a_op == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "no MMA atom");
        return false;
    }
    /* _check_pair */
    if(c_op->wave_size != a_op->wave_size)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "producer and consumer atoms have different wave sizes");
        return false;
    }
    if(c_op->n != a_op->m)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "C columns (%d) must equal the consumer A rows (%d)",
                        c_op->n,
                        a_op->m);
        return false;
    }
    if(a_op->k % c_op->m)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "consumer K (%d) is not a multiple of the C tile rows (%d)",
                        a_op->k,
                        c_op->m);
        return false;
    }
    int tiles = a_op->k / c_op->m;
    if(a_op->a_frag_len > ROCKE_ATTN_BWD_FRAG_MAX_SLOTS || a_op->k > ROCKE_ATTN_BWD_FRAG_MAX_K)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s exceeds the relabel plan tables", a_op->op_id);
        return false;
    }
    Ev a, c;
    if(!eval_op(b, a_op, 'a', &a) || !eval_op(b, c_op, 'c', &c))
        return false;
    rocke_attn_bwd_relabel_plan_t plan;
    bool ok = try_direct(c_op, a_op, tiles, a, c, &plan);
    if(!ok)
        ok = try_xlane16(c_op, a_op, tiles, a, c, &plan);
    if(!ok || !proved(&plan, c, a))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "no register C-to-A relabel from %s to %s: the lane maps do not match; "
                        "route the hand-off through LDS",
                        c_op->op_id,
                        a_op->op_id);
        return false;
    }
    *out = plan;
    return true;
}

bool rocke_attn_bwd_simulate_relabel(rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_relabel_plan_t* plan,
                                     const rocke_mma_op_t* c_op,
                                     const rocke_mma_op_t* a_op,
                                     int* out,
                                     int cap)
{
    Ev c;
    if(!live(b))
        return false;
    if(plan->kind != ROCKE_ATTN_BWD_RELABEL_DIRECT && plan->kind != ROCKE_ATTN_BWD_RELABEL_XLANE16)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "unknown relabel kind %d", (int)plan->kind);
        return false;
    }
    if(!eval_op(b, c_op, 'c', &c))
        return false;
    std::vector<int> got;
    int n_slots = 0;
    if(!simulate(plan, c, a_op->wave_size, &got, &n_slots))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "relabel program splits a 16-bit element");
        return false;
    }
    if(out == NULL || cap < (int)got.size())
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "simulate_relabel output too small");
        return false;
    }
    memcpy(out, got.data(), got.size() * sizeof(int));
    return true;
}

/* ====================================================================== *
 * Emission
 * ====================================================================== */

const rocke_type_t* rocke_attn_bwd_ir_type(rocke_ir_builder_t* b, const char* dtype)
{
    if(!live(b))
        return NULL;
    if(dtype != NULL && (strcmp(dtype, "fp16") == 0 || strcmp(dtype, "f16") == 0))
        return rocke_f16();
    if(dtype != NULL && strcmp(dtype, "bf16") == 0)
        return rocke_bf16();
    rocke_i_set_err(
        b, ROCKE_ERR_VALUE, "operand dtype must be fp16 or bf16, got '%s'", dtype ? dtype : "None");
    return NULL;
}

/* _concat_all */
static rocke_value_t* concat_all(rocke_ir_builder_t* b, std::vector<rocke_value_t*> parts)
{
    while(parts.size() > 1)
    {
        std::vector<rocke_value_t*> nxt;
        for(size_t i = 0; i + 1 < parts.size(); i += 2)
            nxt.push_back(rocke_b_vec_concat(b, parts[i], parts[i + 1]));
        if(parts.size() % 2)
            nxt.push_back(parts.back());
        parts = nxt;
    }
    return parts[0];
}

/* _assemble */
static rocke_value_t* assemble(rocke_ir_builder_t* b,
                               const std::vector<rocke_value_t*>& chunks,
                               const rocke_type_t* elem)
{
    if(!live(b) || chunks.empty())
        return NULL;
    for(rocke_value_t* ch : chunks)
        if(ch == NULL)
            return NULL;
    if(chunks.size() == 1)
        return chunks[0];
    bool same = true;
    for(rocke_value_t* ch : chunks)
        same = same && ch->type->count == chunks[0]->type->count;
    size_t n = chunks.size();
    if(same && (n & (n - 1)) == 0)
        return concat_all(b, chunks);
    std::vector<rocke_value_t*> elems;
    for(rocke_value_t* ch : chunks)
        for(int i = 0; i < ch->type->count; ++i)
            elems.push_back(rocke_b_vec_extract(b, ch, i));
    return rocke_b_vec_pack(b, elems.data(), (int)elems.size(), elem);
}

static int64_t wrap_i32(int64_t v)
{
    v &= 0xFFFFFFFFLL;
    return v >= (1LL << 31) ? v - (1LL << 32) : v;
}

rocke_value_t* rocke_attn_bwd_emit_c_to_a(rocke_ir_builder_t* b,
                                          const rocke_attn_bwd_relabel_plan_t* plan,
                                          rocke_value_t* const* c_tiles,
                                          int n_tiles,
                                          const char* dtype,
                                          rocke_value_t* lane,
                                          const char* bf16_cast)
{
    if(!live(b))
        return NULL;
    if(n_tiles != plan->c_tiles)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "plan needs %d C tiles, got %d", plan->c_tiles, n_tiles);
        return NULL;
    }
    const rocke_type_t* elem = rocke_attn_bwd_ir_type(b, dtype);
    if(elem == NULL)
        return NULL;
    std::vector<rocke_value_t*> cast;
    for(int t = 0; t < n_tiles; ++t)
    {
        rocke_value_t* v = rocke_attn_bwd_cast_c_frag(b, c_tiles[t], elem, bf16_cast);
        if(v == NULL)
            return NULL;
        cast.push_back(v);
    }
    if(plan->kind == ROCKE_ATTN_BWD_RELABEL_DIRECT)
    {
        int cf = cast[0]->type->count;
        bool identity = true;
        for(int s = 0; s < plan->n_slot_src; ++s)
            identity = identity && plan->slot_src[s][0] == s / cf && plan->slot_src[s][1] == s % cf;
        if(identity)
            return assemble(b, cast, elem);
        std::vector<rocke_value_t*> elems;
        for(int s = 0; s < plan->n_slot_src; ++s)
        {
            int t = plan->slot_src[s][0];
            if(t < 0 || t >= (int)cast.size())
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "relabel slot source outside the C tiles");
                return NULL;
            }
            elems.push_back(rocke_b_vec_extract(b, cast[t], plan->slot_src[s][1]));
        }
        return rocke_b_vec_pack(b, elems.data(), (int)elems.size(), elem);
    }
    if(plan->kind != ROCKE_ATTN_BWD_RELABEL_XLANE16)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "unknown relabel kind %d", (int)plan->kind);
        return NULL;
    }
    if(lane == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "the lane ^ 16 relabel needs the lane id");
        return NULL;
    }
    rocke_value_t* v = cast[0];
    int n_dw = v->type->count / 2;
    rocke_value_t* words = rocke_b_vec_bitcast(b, v, rocke_vector_type(b, rocke_i32(), n_dw));
    std::vector<rocke_value_t*> own;
    for(int j = 0; j < n_dw; ++j)
        own.push_back(rocke_b_vec_extract(b, words, j));
    std::map<int64_t, rocke_value_t*> partner;
    rocke_value_t* c16 = rocke_b_const_i32(b, 16);
    rocke_value_t* is_lo = rocke_b_cmp_lt(b, lane, c16);
    std::map<std::pair<int64_t, int64_t>, rocke_value_t*> sels;
    std::vector<rocke_value_t*> out;
    for(int m = 0; m < plan->n_xlane; ++m)
    {
        int64_t j = plan->xlane[m][0], s0 = plan->xlane[m][1], s1 = plan->xlane[m][2];
        if(j < 0 || j >= n_dw)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "relabel dword outside the fragment");
            return NULL;
        }
        if(partner.find(j) == partner.end())
            partner[j] = rocke_b_permlanex16(b, own[j]);
        std::pair<int64_t, int64_t> key(s0, s1);
        if(sels.find(key) == sels.end())
        {
            rocke_value_t* c0 = rocke_b_const_i32(b, wrap_i32(s0));
            rocke_value_t* c1 = rocke_b_const_i32(b, wrap_i32(s1));
            sels[key] = rocke_b_select(b, is_lo, c0, c1);
        }
        out.push_back(rocke_b_perm_b32(b, own[j], partner[j], sels[key]));
    }
    rocke_value_t* packed = rocke_b_vec_pack(b, out.data(), (int)out.size(), rocke_i32());
    return rocke_b_vec_bitcast(b, packed, rocke_vector_type(b, elem, 2 * (int)out.size()));
}

/* ====================================================================== *
 * Operand coordinates
 * ====================================================================== */

bool rocke_attn_bwd_frag_coords_emit_base(rocke_ir_builder_t* b,
                                          const rocke_attn_bwd_frag_coords_t* c,
                                          rocke_value_t* lane,
                                          rocke_value_t** mn,
                                          rocke_value_t** k)
{
    *mn = NULL;
    *k = NULL;
    if(!live(b))
        return false;
    if(c->base == ROCKE_ATTN_BWD_COORDS_OPERAND)
    {
        /* py: _mn_k(role, layout.coord(b, lane, 0)) */
        rocke_value_t *c0, *c1;
        if(!rocke_layout_map_coord(c->layout, b, lane, 0, &c0, &c1))
            return false;
        if(c->role == 'a')
        {
            *mn = c0;
            *k = c1;
        }
        else
        {
            *mn = c1;
            *k = c0;
        }
        return true;
    }
    /* py: k_row, _ = c_layout.coord(b, lane, cs0)
     *     k = b.add(k_row, b.const_i32(t0 * rows)) if t0 else k_row
     *     _, col = b_layout.coord(b, lane, 0) */
    rocke_value_t *k_row, *unused, *unused2, *col;
    if(!rocke_layout_map_coord(c->c_layout, b, lane, c->kperm_cs0, &k_row, &unused))
        return false;
    rocke_value_t* kk = k_row;
    if(c->kperm_t0)
    {
        rocke_value_t* off = rocke_b_const_i32(b, (int64_t)c->kperm_t0 * c->kperm_rows);
        kk = rocke_b_add(b, k_row, off);
    }
    if(!rocke_layout_map_coord(c->layout, b, lane, 0, &unused2, &col))
        return false;
    *mn = col;
    *k = kk;
    return live(b);
}

void rocke_attn_bwd_frag_coords_int_coords(const rocke_attn_bwd_frag_coords_t* c,
                                           int lane,
                                           int* out)
{
    for(int s = 0; s < c->frag_len; ++s)
    {
        out[2 * s] = c->int_base[lane][0] + c->deltas[s][0];
        out[2 * s + 1] = c->int_base[lane][1] + c->deltas[s][1];
    }
}

/* _deltas_of over per_lane[lane][slot] = (mn, k). */
static bool deltas_of(rocke_ir_builder_t* b,
                      const std::vector<int>& per_lane,
                      int ws,
                      int fl,
                      rocke_attn_bwd_frag_coords_t* out)
{
    for(int lane = 0; lane < ws; ++lane)
    {
        int m0 = per_lane[((size_t)lane * fl) * 2], k0 = per_lane[((size_t)lane * fl) * 2 + 1];
        for(int s = 0; s < fl; ++s)
        {
            int dm = per_lane[((size_t)lane * fl + s) * 2] - m0;
            int dk = per_lane[((size_t)lane * fl + s) * 2 + 1] - k0;
            if(lane == 0)
            {
                out->deltas[s][0] = dm;
                out->deltas[s][1] = dk;
            }
            else if(out->deltas[s][0] != dm || out->deltas[s][1] != dk)
            {
                rocke_i_set_err(
                    b, ROCKE_ERR_VALUE, "operand lane map is not affine in the slot index");
                return false;
            }
        }
    }
    return true;
}

bool rocke_attn_bwd_operand_coords(rocke_ir_builder_t* b,
                                   const rocke_mma_op_t* op,
                                   char role,
                                   rocke_attn_bwd_frag_coords_t* out)
{
    if(!live(b))
        return false;
    if(role != 'a' && role != 'b')
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "role must be 'a' or 'b', got '%c'", role);
        return false;
    }
    const rocke_layout_map_t* layout = op_layout(b, op, role);
    Ev ev;
    if(layout == NULL || !eval_op(b, op, role, &ev))
    {
        if(live(b))
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "missing lane map");
        return false;
    }
    memset(out, 0, sizeof(*out));
    std::vector<int> per_lane((size_t)ev.ws * ev.fl * 2);
    for(int lane = 0; lane < ev.ws; ++lane)
        for(int s = 0; s < ev.fl; ++s)
            mn_k(role,
                 ev.at(lane, s, 0),
                 ev.at(lane, s, 1),
                 &per_lane[((size_t)lane * ev.fl + s) * 2],
                 &per_lane[((size_t)lane * ev.fl + s) * 2 + 1]);
    if(!deltas_of(b, per_lane, ev.ws, ev.fl, out))
        return false;
    out->role = role;
    out->frag_len = layout->frag_len;
    out->wave_size = layout->wave_size;
    out->kperm = false;
    out->base = ROCKE_ATTN_BWD_COORDS_OPERAND;
    out->layout = layout;
    for(int lane = 0; lane < ev.ws; ++lane)
    {
        out->int_base[lane][0] = per_lane[((size_t)lane * ev.fl) * 2];
        out->int_base[lane][1] = per_lane[((size_t)lane * ev.fl) * 2 + 1];
    }
    return true;
}

bool rocke_attn_bwd_kperm_b_coords(rocke_ir_builder_t* b,
                                   const rocke_attn_bwd_relabel_plan_t* plan,
                                   const rocke_mma_op_t* c_op,
                                   const rocke_mma_op_t* b_op,
                                   rocke_attn_bwd_frag_coords_t* out)
{
    if(!live(b))
        return false;
    if(plan->kind != ROCKE_ATTN_BWD_RELABEL_DIRECT)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "a K-permuted load exists only for the direct relabel");
        return false;
    }
    if(b_op == NULL || c_op == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "no MMA atom");
        return false;
    }
    if(strcmp(b_op->op_id, plan->a_op_id) != 0)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "plan consumer is %s, not %s", plan->a_op_id, b_op->op_id);
        return false;
    }
    rocke_attn_bwd_atom_facts_t facts;
    if(!rocke_attn_bwd_atom_facts(b, b_op, &facts))
        return false;
    if(!facts.ab_same_k)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "%s: A and B K maps differ; K permutation unsafe", b_op->op_id);
        return false;
    }
    Ev bl;
    if(!eval_op(b, b_op, 'b', &bl))
        return false;
    std::vector<int> per_lane((size_t)b_op->wave_size * bl.fl * 2);
    for(int lane = 0; lane < b_op->wave_size; ++lane)
        for(int s = 0; s < bl.fl; ++s)
        {
            int k = bl.at(lane, s, 0);
            if(k < 0 || k >= plan->n_k_perm)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "B K index outside the plan permutation");
                return false;
            }
            per_lane[((size_t)lane * bl.fl + s) * 2] = bl.at(lane, s, 1);
            per_lane[((size_t)lane * bl.fl + s) * 2 + 1] = plan->k_perm[k];
        }
    memset(out, 0, sizeof(*out));
    if(!deltas_of(b, per_lane, b_op->wave_size, bl.fl, out))
        return false;
    const rocke_layout_map_t* c_layout = rocke_mma_op_c_layout(c_op, b);
    const rocke_layout_map_t* b_layout = rocke_mma_op_b_layout(b_op, b);
    if(c_layout == NULL || b_layout == NULL)
        return false;
    Ev cl;
    if(!eval_op(b, c_op, 'c', &cl))
        return false;
    int t0 = plan->slot_src[0][0], cs0 = plan->slot_src[0][1];
    out->role = 'b';
    out->frag_len = b_op->b_frag_len;
    out->wave_size = b_op->wave_size;
    out->kperm = true;
    out->base = ROCKE_ATTN_BWD_COORDS_KPERM;
    out->layout = b_layout;
    out->c_layout = c_layout;
    out->kperm_t0 = t0;
    out->kperm_cs0 = cs0;
    out->kperm_rows = plan->c_tile_rows;
    for(int lane = 0; lane < b_op->wave_size; ++lane)
    {
        int k_row = cl.at(lane, cs0, 0);
        out->int_base[lane][0] = bl.at(lane, 0, 1);
        out->int_base[lane][1] = t0 ? k_row + t0 * plan->c_tile_rows : k_row;
    }
    /* the emitted base must equal the proof */
    std::vector<int> cc((size_t)out->frag_len * 2);
    for(int lane = 0; lane < b_op->wave_size; ++lane)
    {
        rocke_attn_bwd_frag_coords_int_coords(out, lane, cc.data());
        for(int s = 0; s < out->frag_len; ++s)
            if(cc[2 * s] != per_lane[((size_t)lane * bl.fl + s) * 2]
               || cc[2 * s + 1] != per_lane[((size_t)lane * bl.fl + s) * 2 + 1])
            {
                rocke_i_set_err(
                    b, ROCKE_ERR_VALUE, "K-permuted B base does not reproduce the permutation");
                return false;
            }
    }
    return true;
}

/* ====================================================================== *
 * LDS views
 * ====================================================================== */

#define SLAB_LINE_ELEMS 64
#define SLAB_CHUNK_ELEMS 8
#define LINE_ELEMS 64

static bool base_post_init(rocke_ir_builder_t* b, const rocke_attn_bwd_lds_view_t* v)
{
    if(!live(b))
        return false;
    if(v->base < 0 || v->pitch < 1 || v->rows < 1 || (v->scale != 1 && v->scale != 2))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "illegal LDS view (base %lld, pitch %lld, rows %lld, "
                        "scale %d)",
                        (long long)v->base,
                        (long long)v->pitch,
                        (long long)v->rows,
                        v->scale);
        return false;
    }
    if(v->base % v->scale)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "an fp32 view must start on a 4-byte boundary");
        return false;
    }
    return true;
}

static void fill_view(rocke_attn_bwd_lds_view_t* v,
                      rocke_attn_bwd_view_kind_t kind,
                      rocke_value_t* smem,
                      int64_t base,
                      int64_t pitch,
                      int64_t rows,
                      int scale,
                      rocke_value_t* shift,
                      int chunk)
{
    v->kind = kind;
    v->smem = smem;
    v->base = base;
    v->pitch = pitch;
    v->rows = rows;
    v->scale = scale;
    v->shift = shift;
    v->chunk = chunk;
}

bool rocke_attn_bwd_lds_view_init(rocke_attn_bwd_lds_view_t* v,
                                  rocke_ir_builder_t* b,
                                  rocke_value_t* smem,
                                  int64_t base,
                                  int64_t pitch,
                                  int64_t rows,
                                  int scale,
                                  rocke_value_t* shift)
{
    fill_view(v, ROCKE_ATTN_BWD_VIEW_PLAIN, smem, base, pitch, rows, scale, shift, 0);
    return base_post_init(b, v);
}

bool rocke_attn_bwd_dma_slab_view_init(rocke_attn_bwd_lds_view_t* v,
                                       rocke_ir_builder_t* b,
                                       rocke_value_t* smem,
                                       int64_t base,
                                       int64_t pitch,
                                       int64_t rows,
                                       int scale,
                                       rocke_value_t* shift)
{
    fill_view(v, ROCKE_ATTN_BWD_VIEW_DMA_SLAB, smem, base, pitch, rows, scale, shift, 0);
    if(!base_post_init(b, v))
        return false;
    if(scale != 1 || base % SLAB_CHUNK_ELEMS)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "a DMA slab view is a 16-bit view on a 16-byte origin");
        return false;
    }
    int64_t row_elems = pitch;
    if(row_elems % SLAB_CHUNK_ELEMS)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "DMA slab rows must be whole 16-byte chunks");
        return false;
    }
    if(row_elems >= SLAB_LINE_ELEMS)
    {
        if(row_elems % SLAB_LINE_ELEMS)
        {
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "DMA slab rows of a line or more must be whole lines");
            return false;
        }
    }
    else if(SLAB_LINE_ELEMS % row_elems || rows % (SLAB_LINE_ELEMS / row_elems))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "folded DMA slab rows must tile whole lines");
        return false;
    }
    return true;
}

static int log2i(int64_t n)
{
    return bit_length(n) - 1;
}

bool rocke_attn_bwd_xor_lds_view_init(rocke_attn_bwd_lds_view_t* v,
                                      rocke_ir_builder_t* b,
                                      rocke_value_t* smem,
                                      int64_t base,
                                      int64_t pitch,
                                      int64_t rows,
                                      int scale,
                                      rocke_value_t* shift,
                                      int chunk)
{
    fill_view(v, ROCKE_ATTN_BWD_VIEW_XOR, smem, base, pitch, rows, scale, shift, chunk);
    if(!base_post_init(b, v))
        return false;
    if(scale != 1)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "an XOR view is a 16-bit view");
        return false;
    }
    if(shift != NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "an XOR view takes no run-time shift (ring slots)");
        return false;
    }
    if(chunk != 2 && chunk != 4 && chunk != 8)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "XOR chunk of %d elements is not 4, 8 or 16 bytes", chunk);
        return false;
    }
    if(!(pitch % LINE_ELEMS == 0 || LINE_ELEMS % pitch == 0))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "XOR view pitch %lld must divide or be a multiple of a "
                        "line",
                        (long long)pitch);
        return false;
    }
    if(pitch % chunk || base % LINE_ELEMS)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "XOR view rows must be whole chunks on a line-aligned origin");
        return false;
    }
    return true;
}

static int plain_align(const rocke_attn_bwd_lds_view_t* v)
{
    int a = 8;
    while(a > 1 && (py_mod(py_floordiv(v->base, v->scale), a) || py_mod(v->pitch, a)))
        a /= 2;
    return a;
}

int rocke_attn_bwd_lds_view_align_elems(const rocke_attn_bwd_lds_view_t* v)
{
    switch(v->kind)
    {
    case ROCKE_ATTN_BWD_VIEW_DMA_SLAB:
        return SLAB_CHUNK_ELEMS;
    case ROCKE_ATTN_BWD_VIEW_XOR:
    {
        int a = plain_align(v);
        return a < v->chunk ? a : v->chunk;
    }
    default:
        return plain_align(v);
    }
}

static int64_t rows_per_line(const rocke_attn_bwd_lds_view_t* v)
{
    int64_t r = LINE_ELEMS / v->pitch;
    return r > 1 ? r : 1;
}

int64_t rocke_attn_bwd_xor_view_row_mask(const rocke_attn_bwd_lds_view_t* v, int64_t row)
{
    int64_t g = py_floordiv(row, rows_per_line(v));
    int64_t nch = LINE_ELEMS / v->chunk;
    return py_mod(g ^ (g >> log2i(nch)), nch) * v->chunk;
}

/* _origin of DmaSlabView (also the tail of LdsView.index). */
static rocke_value_t*
    view_origin(rocke_ir_builder_t* b, const rocke_attn_bwd_lds_view_t* v, rocke_value_t* lin)
{
    if(v->shift != NULL)
        lin = rocke_b_add(b, lin, v->shift);
    if(v->base)
    {
        rocke_value_t* cb = rocke_b_const_i32(b, v->base);
        return rocke_b_add(b, lin, cb);
    }
    return lin;
}

rocke_value_t* rocke_attn_bwd_xor_view_emit_index(rocke_ir_builder_t* b,
                                                  const rocke_attn_bwd_lds_view_t* v,
                                                  rocke_value_t* row,
                                                  rocke_attn_bwd_iv_t col)
{
    if(!live(b))
        return NULL;
    /* py: r = b.mul(row, b.const_i32(self.pitch)) */
    rocke_value_t* cp = rocke_b_const_i32(b, v->pitch);
    rocke_value_t* r = rocke_b_mul(b, row, cp);
    rocke_value_t* lin = r;
    if(!rocke_attn_bwd_iv_is_int(col, 0))
    {
        rocke_value_t* cv = rocke_attn_bwd_as_i32(b, col);
        lin = rocke_b_add(b, r, cv);
    }
    rocke_value_t* g = row;
    int64_t rpl = rows_per_line(v);
    if(rpl > 1)
    {
        rocke_value_t* cs = rocke_b_const_i32(b, log2i(rpl));
        g = rocke_b_lshr(b, row, cs);
    }
    int64_t nch = LINE_ELEMS / v->chunk;
    /* py: h = b.land(b.xor(g, b.lshr(g, b.const_i32(log2(nch)))), b.const_i32(nch - 1)) */
    rocke_value_t* cl = rocke_b_const_i32(b, log2i(nch));
    rocke_value_t* gs = rocke_b_lshr(b, g, cl);
    rocke_value_t* gx = rocke_b_xor(b, g, gs);
    rocke_value_t* cm = rocke_b_const_i32(b, nch - 1);
    rocke_value_t* h = rocke_b_land(b, gx, cm);
    /* py: idx = b.xor(lin, b.shl(h, b.const_i32(log2(chunk)))) */
    rocke_value_t* cc = rocke_b_const_i32(b, log2i(v->chunk));
    rocke_value_t* hs = rocke_b_shl(b, h, cc);
    rocke_value_t* idx = rocke_b_xor(b, lin, hs);
    if(v->base)
    {
        rocke_value_t* cb = rocke_b_const_i32(b, v->base);
        return rocke_b_add(b, idx, cb);
    }
    return idx;
}

/* DmaSlabLayout(rows, cols, 2).offset(row, col) // 2 */
bool rocke_attn_bwd_slab_offset(
    rocke_ir_builder_t* b, int64_t rows, int64_t cols, int64_t row, int64_t col, int64_t* out)
{
    *out = 0;
    if(!live(b))
        return false;
    for(int64_t v : {rows, cols, row, col})
        if(v < INT32_MIN || v > INT32_MAX)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "DMA slab argument out of int range");
            return false;
        }
    rocke_attn_bwd_err_t err;
    err.code = ROCKE_OK;
    err.msg[0] = '\0';
    rocke_attn_bwd_dma_slab_layout_t l;
    if(rocke_attn_bwd_dma_slab_layout_init(&l, (int)rows, (int)cols, 2, &err) != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", err.msg);
        return false;
    }
    int off = rocke_attn_bwd_dma_slab_layout_offset(&l, (int)row, (int)col, &err);
    if(err.code != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", err.msg);
        return false;
    }
    *out = off / 2;
    return true;
}

static rocke_value_t* dma_index(rocke_ir_builder_t* b,
                                const rocke_attn_bwd_lds_view_t* v,
                                rocke_attn_bwd_iv_t row_iv,
                                rocke_attn_bwd_iv_t col_iv)
{
    if(row_iv.v == NULL && col_iv.v == NULL)
    {
        int64_t off;
        if(!rocke_attn_bwd_slab_offset(b, v->rows, v->pitch, row_iv.i, col_iv.i, &off))
            return NULL;
        rocke_value_t* c = rocke_b_const_i32(b, off);
        return view_origin(b, v, c);
    }
    rocke_value_t* row = rocke_attn_bwd_as_i32(b, row_iv);
    rocke_value_t* col = rocke_attn_bwd_as_i32(b, col_iv);
    const int64_t line_e = SLAB_LINE_ELEMS, chunk_e = SLAB_CHUNK_ELEMS;
    rocke_value_t* c7a = rocke_b_const_i32(b, chunk_e - 1);
    rocke_value_t* inner = rocke_b_land(b, col, c7a);
    rocke_value_t *chunk, *line, *lin;
    if(v->pitch >= line_e)
    {
        rocke_value_t* c6 = rocke_b_const_i32(b, 6);
        rocke_value_t* slab = rocke_b_lshr(b, col, c6);
        rocke_value_t* c3 = rocke_b_const_i32(b, 3);
        rocke_value_t* cs = rocke_b_lshr(b, col, c3);
        rocke_value_t* c7 = rocke_b_const_i32(b, 7);
        chunk = rocke_b_land(b, cs, c7);
        line = row;
        rocke_value_t* cr = rocke_b_const_i32(b, v->rows * line_e);
        rocke_value_t* sm = rocke_b_mul(b, slab, cr);
        rocke_value_t* c6b = rocke_b_const_i32(b, 6);
        rocke_value_t* ls = rocke_b_shl(b, line, c6b);
        lin = rocke_b_add(b, sm, ls);
    }
    else
    {
        int64_t fold = line_e / v->pitch;
        int64_t cpr = v->pitch / chunk_e;
        rocke_value_t* cf = rocke_b_const_i32(b, bit_length(fold) - 1);
        line = rocke_b_lshr(b, row, cf);
        rocke_value_t* cf1 = rocke_b_const_i32(b, fold - 1);
        rocke_value_t* sub = rocke_b_land(b, row, cf1);
        rocke_value_t* ccpr = rocke_b_const_i32(b, cpr);
        rocke_value_t* sc = rocke_b_mul(b, sub, ccpr);
        rocke_value_t* c3 = rocke_b_const_i32(b, 3);
        rocke_value_t* col3 = rocke_b_lshr(b, col, c3);
        chunk = rocke_b_add(b, sc, col3);
        rocke_value_t* c6 = rocke_b_const_i32(b, 6);
        lin = rocke_b_shl(b, line, c6);
    }
    /* py: sw = b.xor(chunk, b.land(line, b.const_i32(7))) */
    rocke_value_t* c7s = rocke_b_const_i32(b, 7);
    rocke_value_t* l7 = rocke_b_land(b, line, c7s);
    rocke_value_t* sw = rocke_b_xor(b, chunk, l7);
    /* py: lin = b.add(lin, b.add(b.shl(sw, b.const_i32(3)), inner)) */
    rocke_value_t* c3s = rocke_b_const_i32(b, 3);
    rocke_value_t* sws = rocke_b_shl(b, sw, c3s);
    rocke_value_t* si = rocke_b_add(b, sws, inner);
    lin = rocke_b_add(b, lin, si);
    return view_origin(b, v, lin);
}

static rocke_value_t* xor_index(rocke_ir_builder_t* b,
                                const rocke_attn_bwd_lds_view_t* v,
                                rocke_attn_bwd_iv_t row,
                                rocke_attn_bwd_iv_t col)
{
    if(row.v != NULL)
        return rocke_attn_bwd_xor_view_emit_index(b, v, row.v, col);
    int64_t mask = rocke_attn_bwd_xor_view_row_mask(v, row.i);
    rocke_value_t* r = rocke_b_const_i32(b, row.i * v->pitch);
    rocke_value_t* lin = r;
    if(!rocke_attn_bwd_iv_is_int(col, 0))
    {
        rocke_value_t* cv = rocke_attn_bwd_as_i32(b, col);
        lin = rocke_b_add(b, r, cv);
    }
    if(mask)
    {
        rocke_value_t* cm = rocke_b_const_i32(b, mask);
        lin = rocke_b_xor(b, lin, cm);
    }
    if(v->base)
    {
        rocke_value_t* cb = rocke_b_const_i32(b, v->base);
        return rocke_b_add(b, lin, cb);
    }
    return lin;
}

rocke_value_t* rocke_attn_bwd_lds_view_index(rocke_ir_builder_t* b,
                                             const rocke_attn_bwd_lds_view_t* v,
                                             rocke_attn_bwd_iv_t row,
                                             rocke_attn_bwd_iv_t col)
{
    if(!live(b))
        return NULL;
    if(v->kind == ROCKE_ATTN_BWD_VIEW_DMA_SLAB)
        return dma_index(b, v, row, col);
    if(v->kind == ROCKE_ATTN_BWD_VIEW_XOR)
        return xor_index(b, v, row, col);
    rocke_value_t* r;
    if(row.v != NULL)
    {
        if(v->pitch != 1)
        {
            rocke_value_t* cp = rocke_b_const_i32(b, v->pitch);
            r = rocke_b_mul(b, row.v, cp);
        }
        else
            r = row.v;
    }
    else
        r = rocke_b_const_i32(b, row.i * v->pitch);
    rocke_value_t* lin = r;
    if(!rocke_attn_bwd_iv_is_int(col, 0))
    {
        rocke_value_t* cv = rocke_attn_bwd_as_i32(b, col);
        lin = rocke_b_add(b, r, cv);
    }
    if(v->scale != 1)
    {
        rocke_value_t* cs = rocke_b_const_i32(b, v->scale);
        lin = rocke_b_mul(b, lin, cs);
    }
    return view_origin(b, v, lin);
}

/* _offset(b, col, k) */
static rocke_attn_bwd_iv_t iv_offset(rocke_ir_builder_t* b, rocke_attn_bwd_iv_t col, int k)
{
    if(k == 0)
        return col;
    if(col.v == NULL)
        return rocke_attn_bwd_iv_int(col.i + k);
    rocke_value_t* ck = rocke_b_const_i32(b, k);
    return rocke_attn_bwd_iv_val(rocke_b_add(b, col.v, ck));
}

rocke_value_t* rocke_attn_bwd_lds_view_load(rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_lds_view_t* v,
                                            rocke_attn_bwd_iv_t row,
                                            rocke_attn_bwd_iv_t col,
                                            const rocke_type_t* dtype,
                                            int n)
{
    if(!live(b))
        return NULL;
    int a = rocke_attn_bwd_lds_view_align_elems(v);
    if(n <= a)
    {
        rocke_value_t* idx = rocke_attn_bwd_lds_view_index(b, v, row, col);
        return rocke_b_smem_load_vN(b, v->smem, &idx, 1, dtype, n);
    }
    std::vector<rocke_value_t*> parts;
    for(int k = 0; k < n; k += a)
    {
        rocke_attn_bwd_iv_t ck = iv_offset(b, col, k);
        rocke_value_t* idx = rocke_attn_bwd_lds_view_index(b, v, row, ck);
        parts.push_back(rocke_b_smem_load_vN(b, v->smem, &idx, 1, dtype, a));
    }
    std::vector<rocke_value_t*> elems;
    for(rocke_value_t* p : parts)
        for(int i = 0; i < a; ++i)
            elems.push_back(rocke_b_vec_extract(b, p, i));
    return rocke_b_vec_pack(b, elems.data(), (int)elems.size(), dtype);
}

void rocke_attn_bwd_lds_view_store(rocke_ir_builder_t* b,
                                   const rocke_attn_bwd_lds_view_t* v,
                                   rocke_attn_bwd_iv_t row,
                                   rocke_attn_bwd_iv_t col,
                                   rocke_value_t* value,
                                   int n)
{
    if(!live(b))
        return;
    int a = rocke_attn_bwd_lds_view_align_elems(v);
    if(n <= a)
    {
        rocke_value_t* idx = rocke_attn_bwd_lds_view_index(b, v, row, col);
        rocke_b_smem_store_vN(b, v->smem, &idx, 1, value, n);
        return;
    }
    const rocke_type_t* elem = value->type->elem;
    for(int k = 0; k < n; k += a)
    {
        std::vector<rocke_value_t*> piece;
        for(int i = 0; i < a; ++i)
            piece.push_back(rocke_b_vec_extract(b, value, k + i));
        rocke_value_t* part = a == 1 ? piece[0] : rocke_b_vec_pack(b, piece.data(), a, elem);
        rocke_attn_bwd_iv_t ck = iv_offset(b, col, k);
        rocke_value_t* idx = rocke_attn_bwd_lds_view_index(b, v, row, ck);
        rocke_b_smem_store_vN(b, v->smem, &idx, 1, part, a);
    }
}

rocke_value_t* rocke_attn_bwd_lds_view_tr16_load(rocke_ir_builder_t* b,
                                                 const rocke_attn_bwd_lds_view_t* v,
                                                 rocke_attn_bwd_iv_t row,
                                                 rocke_attn_bwd_iv_t col,
                                                 const rocke_type_t* dtype)
{
    if(!live(b))
        return NULL;
    rocke_value_t* idx = rocke_attn_bwd_lds_view_index(b, v, row, col);
    return rocke_b_ds_read_tr16_b64(b, v->smem, &idx, 1, dtype);
}

/* ====================================================================== *
 * The LDS fragment loader
 * ====================================================================== */

static const int WIDTHS[4] = {8, 4, 2, 1};

/* _runs: slot runs contiguous along `along` (0 mn, 1 k). */
static std::vector<std::vector<int>> runs_of(const rocke_attn_bwd_frag_coords_t* c, int along)
{
    int other = 1 - along;
    std::vector<std::vector<int>> runs;
    for(int s = 0; s < c->frag_len; ++s)
    {
        if(!runs.empty())
        {
            int p = runs.back().back();
            if(c->deltas[s][other] == c->deltas[p][other]
               && c->deltas[s][along] == c->deltas[p][along] + 1)
            {
                runs.back().push_back(s);
                continue;
            }
        }
        runs.push_back(std::vector<int>(1, s));
    }
    return runs;
}

/* `x + origin` unless origin is the literal int 0 (const before the add). */
static rocke_value_t* add_origin(rocke_ir_builder_t* b, rocke_value_t* x, rocke_attn_bwd_iv_t o)
{
    if(rocke_attn_bwd_iv_is_int(o, 0))
        return x;
    rocke_value_t* ov = o.v != NULL ? o.v : rocke_b_const_i32(b, o.i);
    return rocke_b_add(b, x, ov);
}

rocke_value_t* rocke_attn_bwd_load_frag_lds(rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_frag_coords_t* coords,
                                            rocke_value_t* smem,
                                            const rocke_attn_bwd_lds_view_t* view,
                                            rocke_value_t* lane,
                                            const char* dtype,
                                            rocke_attn_bwd_iv_t mn0,
                                            rocke_attn_bwd_iv_t k0,
                                            rocke_attn_bwd_storage_t storage,
                                            int align)
{
    if(!live(b))
        return NULL;
    if(storage != ROCKE_ATTN_BWD_K_INNER && storage != ROCKE_ATTN_BWD_K_OUTER)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "storage must be 'k_inner' or 'k_outer'");
        return NULL;
    }
    if(view != NULL)
        align = (int)py_gcd(align, rocke_attn_bwd_lds_view_align_elems(view));
    const rocke_type_t* elem = rocke_attn_bwd_ir_type(b, dtype);
    if(elem == NULL)
        return NULL;
    int along = storage == ROCKE_ATTN_BWD_K_INNER ? 1 : 0;
    int64_t g = align;
    std::vector<std::vector<int>> runs = runs_of(coords, along);
    std::vector<int> cc((size_t)coords->frag_len * 2);
    for(int lane_i = 0; lane_i < coords->wave_size; ++lane_i)
    {
        rocke_attn_bwd_frag_coords_int_coords(coords, lane_i, cc.data());
        for(const auto& r : runs)
            g = py_gcd(g, cc[(size_t)r[0] * 2 + along]);
    }
    rocke_value_t *mn, *k;
    if(!rocke_attn_bwd_frag_coords_emit_base(b, coords, lane, &mn, &k))
        return NULL;
    mn = add_origin(b, mn, mn0);
    k = add_origin(b, k, k0);
    std::vector<rocke_value_t*> chunks;
    for(const auto& r : runs)
    {
        size_t pos = 0;
        while(pos < r.size())
        {
            int64_t rem = (int64_t)(r.size() - pos);
            int w = 1;
            for(int wi = 0; wi < 4; ++wi)
            {
                int ww = WIDTHS[wi];
                if(ww <= rem && py_mod(g, ww) == 0 && rem % ww == 0)
                {
                    w = ww;
                    break;
                }
            }
            int dm = coords->deltas[r[pos]][0], dk = coords->deltas[r[pos]][1];
            rocke_value_t* im = mn;
            if(dm)
            {
                rocke_value_t* cd = rocke_b_const_i32(b, dm);
                im = rocke_b_add(b, mn, cd);
            }
            rocke_value_t* ik = k;
            if(dk)
            {
                rocke_value_t* cd = rocke_b_const_i32(b, dk);
                ik = rocke_b_add(b, k, cd);
            }
            rocke_value_t* idx[2];
            if(storage == ROCKE_ATTN_BWD_K_INNER)
            {
                idx[0] = im;
                idx[1] = ik;
            }
            else
            {
                idx[0] = ik;
                idx[1] = im;
            }
            if(view != NULL)
                chunks.push_back(rocke_attn_bwd_lds_view_load(b,
                                                              view,
                                                              rocke_attn_bwd_iv_val(idx[0]),
                                                              rocke_attn_bwd_iv_val(idx[1]),
                                                              elem,
                                                              w));
            else
                chunks.push_back(rocke_b_smem_load_vN(b, smem, idx, 2, elem, w));
            pos += w;
        }
    }
    return assemble(b, chunks, elem);
}

/* ====================================================================== *
 * Transpose reads
 * ====================================================================== */

void rocke_attn_bwd_tr16_lane_address(int lane, int t, int kpl, int* k, int* mn)
{
    *k = (lane / 16) * kpl + (lane % 16) / 4 + 4 * t;
    *mn = (lane % 4) * 4;
}

void rocke_attn_bwd_tr16_model_coords(int lane, int kpl, int* out)
{
    int g = lane / 16, j = lane % 16, n = 0;
    for(int t = 0; t < kpl / 4; ++t)
        for(int r = 0; r < 4; ++r)
        {
            int src = 16 * g + 4 * r + j / 4, k, mn;
            rocke_attn_bwd_tr16_lane_address(src, t, kpl, &k, &mn);
            out[2 * n] = mn + j % 4;
            out[2 * n + 1] = k;
            ++n;
        }
}

int rocke_attn_bwd_check_tr16_coords(rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_frag_coords_t* coords)
{
    if(!live(b))
        return 0;
    int kpl = coords->frag_len;
    if(coords->wave_size != 64 || kpl % 4 || (kpl != 4 && kpl != 8))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "ds_read_tr16_b64 needs a wave64 fragment of 4 or 8 elements (got %d on "
                        "wave%d)",
                        kpl,
                        coords->wave_size);
        return 0;
    }
    std::vector<int> want((size_t)kpl * 2), got((size_t)kpl * 2);
    for(int lane = 0; lane < 64; ++lane)
    {
        rocke_attn_bwd_frag_coords_int_coords(coords, lane, want.data());
        rocke_attn_bwd_tr16_model_coords(lane, kpl, got.data());
        if(want != got)
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "the ds_read_tr16_b64 lane map does not deliver this operand's "
                            "coordinates; use the plain K-outer reader");
            return 0;
        }
    }
    return kpl;
}

void rocke_attn_bwd_tr16_plan_address(
    int lane, int t, const rocke_attn_bwd_tr16_read_plan_t* plan, int* k, int* mn)
{
    *k = plan->g_stride * (lane / 16) + (lane % 16) / 4 + plan->k_offsets[t];
    *mn = (lane % 4) * 4;
}

void rocke_attn_bwd_tr16_plan_coords(int lane,
                                     const rocke_attn_bwd_tr16_read_plan_t* plan,
                                     int* out)
{
    int g = lane / 16, j = lane % 16, n = 0;
    for(int t = 0; t < plan->n_offsets; ++t)
        for(int r = 0; r < 4; ++r)
        {
            int k, mn;
            rocke_attn_bwd_tr16_plan_address(16 * g + 4 * r + j / 4, t, plan, &k, &mn);
            out[2 * n] = mn + j % 4;
            out[2 * n + 1] = k;
            ++n;
        }
}

/* _tr16_plan_proved */
static bool plan_proved(const rocke_attn_bwd_tr16_read_plan_t* plan,
                        const rocke_attn_bwd_frag_coords_t* c)
{
    if(plan->n_offsets * 4 != c->frag_len)
        return false;
    std::vector<int> want((size_t)c->frag_len * 2), got((size_t)c->frag_len * 2);
    for(int lane = 0; lane < c->wave_size; ++lane)
    {
        rocke_attn_bwd_frag_coords_int_coords(c, lane, want.data());
        rocke_attn_bwd_tr16_plan_coords(lane, plan, got.data());
        if(want != got)
            return false;
    }
    return true;
}

bool rocke_attn_bwd_tr16_read_plan(rocke_ir_builder_t* b,
                                   const rocke_attn_bwd_frag_coords_t* coords,
                                   rocke_attn_bwd_tr16_read_plan_t* out)
{
    if(!live(b))
        return false;
    int n = coords->frag_len;
    if(coords->wave_size != 64 || (n != 4 && n != 8))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "ds_read_tr16_b64 needs a wave64 fragment of 4 or 8 elements (got %d on "
                        "wave%d)",
                        n,
                        coords->wave_size);
        return false;
    }
    /* _catalog_plan(n) = Tr16ReadPlan("catalog", n, (0, 4)[: n // 4]) */
    rocke_attn_bwd_tr16_read_plan_t plan;
    plan.kind = ROCKE_ATTN_BWD_TR16_CATALOG;
    plan.g_stride = n;
    plan.n_offsets = n / 4;
    plan.k_offsets[0] = 0;
    plan.k_offsets[1] = 4;
    if(plan_proved(&plan, coords))
    {
        *out = plan;
        return true;
    }
    if(coords->kperm && n == 8)
    {
        rocke_attn_bwd_tr16_read_plan_t st;
        st.kind = ROCKE_ATTN_BWD_TR16_STACKED;
        st.g_stride = 4;
        st.n_offsets = 2;
        st.k_offsets[0] = 0;
        st.k_offsets[1] = 16;
        std::vector<int> c0((size_t)n * 2);
        rocke_attn_bwd_frag_coords_int_coords(coords, 0, c0.data());
        if(c0[1] == 0 && c0[4 * 2 + 1] == 16 && plan_proved(&st, coords))
        {
            *out = st;
            return true;
        }
    }
    rocke_i_set_err(b,
                    ROCKE_ERR_VALUE,
                    "the ds_read_tr16_b64 lane map does not deliver this operand's coordinates "
                    "(neither the catalog nor the stacked K-permuted plan); use the plain K-outer "
                    "reader");
    return false;
}

rocke_value_t* rocke_attn_bwd_load_frag_lds_tr16(rocke_ir_builder_t* b,
                                                 const rocke_attn_bwd_frag_coords_t* coords,
                                                 const rocke_attn_bwd_lds_view_t* view,
                                                 rocke_value_t* lane,
                                                 const char* dtype,
                                                 bool has_tr_read_b64,
                                                 rocke_attn_bwd_iv_t mn0,
                                                 rocke_attn_bwd_iv_t k0)
{
    if(!live(b))
        return NULL;
    if(!has_tr_read_b64)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "this arch has no ds_read_tr16_b64");
        return NULL;
    }
    if(view == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "the transpose-read loader needs an LdsView");
        return NULL;
    }
    rocke_attn_bwd_tr16_read_plan_t plan;
    if(!rocke_attn_bwd_tr16_read_plan(b, coords, &plan))
        return NULL;
    if(view->scale != 1 || py_mod(view->base, 4) || py_mod(view->pitch, 4))
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "transpose reads need a 16-bit view with origin and pitch % 4");
        return NULL;
    }
    if(mn0.v == NULL && py_mod(mn0.i, 4))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "transpose-read column origin %lld is not a multiple of 4",
                        (long long)mn0.i);
        return NULL;
    }
    const rocke_type_t* elem = rocke_attn_bwd_ir_type(b, dtype);
    if(elem == NULL)
        return NULL;
    rocke_value_t* c16 = rocke_b_const_i32(b, 16);
    rocke_value_t* g = rocke_b_div(b, lane, c16);
    rocke_value_t* j = rocke_b_mod(b, lane, c16);
    /* py: row_l = b.add(b.mul(g, b.const_i32(g_stride)), b.div(j, b.const_i32(4))) */
    rocke_value_t* cgs = rocke_b_const_i32(b, plan.g_stride);
    rocke_value_t* gm = rocke_b_mul(b, g, cgs);
    rocke_value_t* c4a = rocke_b_const_i32(b, 4);
    rocke_value_t* jd = rocke_b_div(b, j, c4a);
    rocke_value_t* row_l = rocke_b_add(b, gm, jd);
    /* py: col_l = b.mul(b.mod(j, b.const_i32(4)), b.const_i32(4)) */
    rocke_value_t* c4b = rocke_b_const_i32(b, 4);
    rocke_value_t* jm = rocke_b_mod(b, j, c4b);
    rocke_value_t* c4c = rocke_b_const_i32(b, 4);
    rocke_value_t* col_l = rocke_b_mul(b, jm, c4c);
    if(!rocke_attn_bwd_iv_is_int(k0, 0))
    {
        rocke_value_t* kv = rocke_attn_bwd_as_i32(b, k0);
        row_l = rocke_b_add(b, row_l, kv);
    }
    if(!rocke_attn_bwd_iv_is_int(mn0, 0))
    {
        rocke_value_t* mv = rocke_attn_bwd_as_i32(b, mn0);
        col_l = rocke_b_add(b, col_l, mv);
    }
    std::vector<rocke_value_t*> reads;
    for(int t = 0; t < plan.n_offsets; ++t)
    {
        int off = plan.k_offsets[t];
        rocke_value_t* row = row_l;
        if(off)
        {
            rocke_value_t* co = rocke_b_const_i32(b, off);
            row = rocke_b_add(b, row_l, co);
        }
        reads.push_back(rocke_attn_bwd_lds_view_tr16_load(
            b, view, rocke_attn_bwd_iv_val(row), rocke_attn_bwd_iv_val(col_l), elem));
    }
    return assemble(b, reads, elem);
}
