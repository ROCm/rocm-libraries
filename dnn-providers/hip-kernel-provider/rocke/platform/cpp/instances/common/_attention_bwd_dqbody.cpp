// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_bwd_dqbody.py: the key-tile
 * range of a block of query rows, BwdDqGeometry and BwdDqStep (stage 0, the
 * key step and the dQ epilogue of the split dQ kernel).
 *
 * See rocke/helper_rocke.instances.common._attention_bwd_dqbody.h for the
 * mapping. Every op is created in the Python evaluation order (left to right,
 * inner call first, constants included); each `py:` comment quotes the
 * Python a block reproduces. Internal code throws ckc::Error where Python
 * raises; the extern "C" entry points catch it (ckc::guard_builder) and leave
 * it on the builder.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_dqbody.h"

#include <string.h>

#include <limits>
#include <set>
#include <string>
#include <vector>

#include "rocke/error.hpp"
#include "rocke/error_boundary.hpp"
#include "rocke/instance_attention_bwd_dq.h"
#include "rocke/ir_internal.h"

extern "C" {

const char* const ROCKE_ATTN_BWD_DQ_TRANSPOSE_SOURCES[ROCKE_ATTN_BWD_N_DQ_TRANSPOSE_SOURCES]
    = {"tr_read", "lds_plain"};

} /* extern "C" */

namespace
{

typedef rocke_value_t* V;
typedef rocke_attn_bwd_iv_t IV;
typedef rocke_ir_builder_t B;
typedef rocke_attn_bwd_dq_geometry_t Geom;
typedef rocke_attn_bwd_dq_step_t Step;

const char* const FENCE = ROCKE_ATTN_BWD_BARRIER_FENCE;
const double INF = std::numeric_limits<double>::infinity();

/* ------------------------------------------------------------ plumbing */

bool live(const B* b)
{
    return b != NULL && b->status == ROCKE_OK;
}

[[noreturn]] void raise_builder(B* b)
{
    if(b != NULL && b->status != ROCKE_OK)
        ckc::raise_status(b->status, b->err);
    throw ckc::ValueError("attention backward dQ body: a helper failed");
}

void need(B* b, bool ok)
{
    if(!ok || !live(b))
        raise_builder(b);
}

V needv(B* b, V v)
{
    if(v == NULL || !live(b))
        raise_builder(b);
    return v;
}

void need_st(rocke_status_t st, const rocke_attn_bwd_err_t& e)
{
    if(st != ROCKE_OK)
        ckc::raise_status(e.code, e.msg);
}

bool eq(const char* a, const char* b)
{
    return a != NULL && b != NULL && strcmp(a, b) == 0;
}

/* Python repr of a str (or None). */
std::string repr(const char* s)
{
    if(s == NULL)
        return "None";
    return rocke_attn_bwd::py_repr(s);
}

template <class Fn>
bool guarded(B* b, Fn&& fn)
{
    if(!live(b))
        return false;
    void* r = ckc::guard_builder(b, [&]() -> void* {
        fn();
        return (void*)1;
    });
    return r != NULL && live(b);
}

template <class T>
T* arena(B* b, size_t n)
{
    void* p = rocke_arena_calloc(&b->arena, (n ? n : 1) * sizeof(T));
    if(p == NULL)
        throw ckc::OOMError("attention backward dQ body: arena allocation failed");
    return (T*)p;
}

V ci(B* b, int64_t x)
{
    return needv(b, rocke_b_const_i32(b, x));
}

IV ivv(V v)
{
    return rocke_attn_bwd_iv_val(v);
}

void sync(B* b)
{
    rocke_b_sync_fence(b, FENCE);
    need(b, true);
}

/* An scf_if region: enter on construction, leave on destruction. */
struct IfScope
{
    B* b;
    IfScope(B* bb, V cond)
        : b(bb)
    {
        rocke_if_t f = rocke_b_scf_if(b, cond);
        if(f.op == NULL)
            raise_builder(b);
        rocke_b_region_enter(b, f.then_region);
    }
    ~IfScope()
    {
        rocke_b_region_leave(b);
    }
};

int FM(const rocke_attn_bwd_block_gemm_t* g)
{
    return rocke_attn_bwd_block_gemm_frags_m(g);
}

int FN(const rocke_attn_bwd_block_gemm_t* g)
{
    return rocke_attn_bwd_block_gemm_frags_n(g);
}

int KS(const rocke_attn_bwd_block_gemm_t* g)
{
    return rocke_attn_bwd_block_gemm_k_steps(g);
}

/* _floor_div(b, a, m): floor(a / m) for a signed i32 a, positive constant m. */
V floor_div(B* b, V a, int m)
{
    V cm = ci(b, m);
    V zero0 = ci(b, 0);
    V m1 = ci(b, (int64_t)m - 1);
    V sh = rocke_b_sub(b, m1, a);
    V q = rocke_b_div(b, sh, cm);
    V neg = rocke_b_sub(b, zero0, q);
    V zero1 = ci(b, 0);
    V is_neg = rocke_b_cmp_lt(b, a, zero1);
    V pos = rocke_b_div(b, a, cm);
    return needv(b, rocke_b_select(b, is_neg, neg, pos));
}

/* ----------------------------------------------------- the key range */

void kv_tile_range(B* b,
                   rocke_band_emitter_t* band,
                   V q_first,
                   IV n_rows,
                   int k_n0,
                   bool uniform,
                   V* out_start,
                   V* out_n)
{
    if(k_n0 < 1)
        throw ckc::ValueError("tile sizes must be positive");
    V zero = ci(b, 0);
    V cn = ci(b, k_n0);
    V rows = needv(b, rocke_attn_bwd_as_i32(b, n_rows));
    /* py: q_last = smin(sub(add(q_first, rows), 1), sub(band.sq, 1)) */
    V qa = rocke_b_add(b, q_first, rows);
    V one0 = ci(b, 1);
    V qs = rocke_b_sub(b, qa, one0);
    V one1 = ci(b, 1);
    V sq1 = rocke_b_sub(b, band->sq, one1);
    V q_last = needv(b, rocke_b_smin(b, qs, sq1));
    V off = band->off;
    V c_first = off == NULL ? q_first : rocke_b_add(b, q_first, off);
    V c_last = off == NULL ? q_last : rocke_b_add(b, q_last, off);
    V k_lo;
    if(!band->left.has)
        k_lo = zero;
    else
    {
        V lft = needv(b, rocke_band_i32(b, band->left.bound));
        k_lo = rocke_b_sub(b, c_first, lft);
        if(band->left.bound.is_value)
        {
            V neg = rocke_b_cmp_lt(b, lft, zero);
            k_lo = rocke_b_select(b, neg, zero, k_lo);
        }
        k_lo = rocke_b_smax(b, k_lo, zero);
    }
    V k_hi;
    if(!band->right.has)
        k_hi = band->sk_minus1;
    else
    {
        V r = needv(b, rocke_band_i32(b, band->right.bound));
        k_hi = rocke_b_add(b, c_last, r);
        if(band->right.bound.is_value)
        {
            V neg = rocke_b_cmp_lt(b, r, zero);
            k_hi = rocke_b_select(b, neg, band->sk_minus1, k_hi);
        }
        k_hi = rocke_b_smin(b, k_hi, band->sk_minus1);
    }
    /* Empty when no row is live or the rows' band misses every key. */
    V d0 = rocke_b_cmp_lt(b, q_last, q_first);
    V d1 = rocke_b_cmp_lt(b, k_hi, k_lo);
    V dead = rocke_b_lor(b, d0, d1);
    V m1 = ci(b, -1);
    k_hi = rocke_b_select(b, dead, m1, k_hi);
    V kt_start = rocke_b_div(b, k_lo, cn);
    V m1b = ci(b, -1);
    V hi_c = rocke_b_smax(b, k_hi, m1b);
    V hi_n = rocke_b_add(b, hi_c, cn);
    V kt_stop = rocke_b_div(b, hi_n, cn);
    V span = rocke_b_sub(b, kt_stop, kt_start);
    V n_kt = rocke_b_smax(b, span, zero);
    if(uniform)
    {
        kt_start = rocke_b_readfirstlane(b, kt_start);
        n_kt = rocke_b_readfirstlane(b, n_kt);
    }
    *out_start = needv(b, kt_start);
    *out_n = needv(b, n_kt);
}

void interior_kv_range(B* b,
                       rocke_band_emitter_t* band,
                       V q_first,
                       IV n_rows,
                       V kt_start,
                       V n_kt,
                       int k_n0,
                       bool masked,
                       bool uniform,
                       V* out_i0,
                       V* out_i1)
{
    V zero = ci(b, 0);
    V rows = needv(b, rocke_attn_bwd_as_i32(b, n_rows));
    V kt_end = rocke_b_add(b, kt_start, n_kt);
    V qa = rocke_b_add(b, q_first, rows);
    V one0 = ci(b, 1);
    V q_last = rocke_b_sub(b, qa, one0);
    V lo = kt_start;
    /* k0 + k_n0 - 1 < len_kv  <->  kt < floor(len_kv / k_n0) */
    V cn = ci(b, k_n0);
    V full = rocke_b_div(b, band->sk, cn);
    V hi = needv(b, rocke_b_smin(b, kt_end, full));
    V off = band->off;
    bool has_right = masked && band->right.has;
    bool has_left = masked && band->left.has;
    if(has_right)
    {
        /* k_last <= q_first + off + right  ->  kt <= floor((... - k_n0 + 1) / k_n0) */
        V r = needv(b, rocke_band_i32(b, band->right.bound));
        V top;
        if(off == NULL)
            top = rocke_b_add(b, q_first, r);
        else
        {
            V shifted = rocke_b_add(b, q_first, off);
            top = rocke_b_add(b, shifted, r);
        }
        V cn1 = ci(b, (int64_t)k_n0 - 1);
        V top_m = rocke_b_sub(b, top, cn1);
        V last = floor_div(b, top_m, k_n0);
        V one = ci(b, 1);
        V stop = rocke_b_add(b, last, one);
        if(band->right.bound.is_value)
        {
            V neg = rocke_b_cmp_lt(b, r, zero);
            stop = rocke_b_select(b, neg, hi, stop);
        }
        hi = rocke_b_smin(b, hi, stop);
    }
    if(has_left)
    {
        /* k_first >= q_last + off - left  ->  kt >= ceil((...) / k_n0) */
        V lft = needv(b, rocke_band_i32(b, band->left.bound));
        V need_;
        if(off == NULL)
            need_ = rocke_b_sub(b, q_last, lft);
        else
        {
            V shifted = rocke_b_add(b, q_last, off);
            need_ = rocke_b_sub(b, shifted, lft);
        }
        V neg_need = rocke_b_sub(b, zero, need_);
        V fl = floor_div(b, neg_need, k_n0);
        V first = rocke_b_sub(b, zero, fl);
        if(band->left.bound.is_value)
        {
            V neg = rocke_b_cmp_lt(b, lft, zero);
            first = rocke_b_select(b, neg, lo, first);
        }
        lo = rocke_b_smax(b, lo, first);
    }
    V row_in = rocke_b_cmp_lt(b, q_last, band->sq);
    hi = rocke_b_select(b, row_in, hi, lo); /* a row past len_q: none */
    V i0 = rocke_b_smin(b, lo, kt_end);
    V hi_c = rocke_b_smin(b, hi, kt_end);
    V i1 = rocke_b_smax(b, hi_c, i0);
    if(uniform)
    {
        i0 = rocke_b_readfirstlane(b, i0);
        i1 = rocke_b_readfirstlane(b, i1);
    }
    *out_i0 = needv(b, i0);
    *out_i1 = needv(b, i1);
}

/* ---------------------------------------------------------- geometry */

void make_gemm(B* b,
               const Geom* g,
               int m,
               int n,
               int k,
               int rows,
               int cols,
               int atom_k,
               rocke_attn_bwd_block_gemm_t* out)
{
    rocke_attn_bwd_gemm_caps_t caps;
    rocke_attn_bwd_err_t e;
    need_st(rocke_attn_bwd_gemm_caps_fill(g->arch, g->dtype, atom_k, &caps, &e), e);
    need(b,
         rocke_attn_bwd_block_gemm_init(
             out, b, g->arch, g->dtype, m, n, k, rows, cols, atom_k, &caps));
}

void check_facts(B* b, const Geom* g, const rocke_attn_bwd_block_gemm_t* gm)
{
    rocke_attn_bwd_atom_facts_t facts;
    need(b, rocke_attn_bwd_atom_facts(b, gm->op, &facts));
    if(!(facts.a_contig && facts.b_contig && facts.affine))
        throw ckc::ValueError(std::string(gm->op->op_id)
                              + ": lane maps are not K-contiguous and affine; this dQ "
                                "geometry cannot be built on "
                              + std::string(g->arch));
}

/* BwdDqGeometry.check(), building the cached g0t / g4t / relabel_plan /
 * b4_coords in Python's order. */
void geometry_check(B* b, Geom* g)
{
    const char* reader = rocke_attn_bwd_dq_reader(g->transpose_source);
    if(reader == NULL)
        throw ckc::ValueError("transpose_source=" + repr(g->transpose_source)
                              + " has no reader in the split dQ kernel (built: "
                                "('tr_read', 'lds_plain'))");
    bool known = false;
    std::vector<std::string> scheds;
    for(int i = 0; i < ROCKE_ATTN_BWD_N_DQ_SCHEDS; ++i)
    {
        scheds.push_back(ROCKE_ATTN_BWD_DQ_SCHEDS[i]);
        known = known || eq(g->sched, ROCKE_ATTN_BWD_DQ_SCHEDS[i]);
    }
    if(!known)
        throw ckc::ValueError("dq sched " + repr(g->sched) + " not in "
                              + rocke_attn_bwd::py_tuple(scheds));
    /* py: for g in (self.g0t, self.g4t): facts = atom_facts(g.op) */
    make_gemm(b, g, g->k_n0, g->k_m0, g->head_size, 1, g->waves, g->atom_g02, &g->g0t);
    check_facts(b, g, &g->g0t);
    make_gemm(b, g, g->k_m0, g->head_size, g->k_n0, g->waves, 1, g->atom_g4, &g->g4t);
    check_facts(b, g, &g->g4t);
    need(b, rocke_attn_bwd_check_relabel_grids(b, &g->g0t, &g->g4t, &g->relabel_plan));
    need(b,
         rocke_attn_bwd_kperm_b_operand_coords(
             b, &g->relabel_plan, &g->g0t, &g->g4t, &g->b4_coords));
    if(eq(reader, "tr16"))
    {
        rocke_attn_bwd_err_t e;
        need_st(rocke_attn_bwd_check_transpose_read(g->arch, 64, &e), e);
        if(rocke_attn_bwd_relabel_plan_needs_k_perm(&g->relabel_plan))
            throw ckc::ValueError(
                "transpose_source='tr_read' with a K-permuted G4' B operand (atom_g4 K = "
                + std::to_string(g->atom_g4) + ") is not built; use 'lds_plain'");
        need(b, rocke_attn_bwd_check_tr16_coords(b, &g->b4_coords) != 0);
    }
    /* one query column per lane in every C fragment (per-column stats) */
    const rocke_mma_op_t* op = g->g0t.op;
    const rocke_layout_map_t* m = op->c_layout;
    if(m == NULL)
        throw ckc::NotImplError(std::string("no verified C lane map for ") + op->op_id);
    int lanes = m->wave_size, n = m->frag_len;
    std::vector<int> ev((size_t)lanes * (size_t)n * 2 + 2);
    need(b, rocke_attn_bwd_eval_layout(b, m, ev.data(), (int)ev.size()));
    for(int lane = 0; lane < lanes; ++lane)
    {
        std::set<int> cols;
        for(int s = 0; s < n; ++s)
            cols.insert(ev[((size_t)lane * n + s) * 2 + 1]);
        if(cols.size() != 1)
            throw ckc::ValueError(std::string(op->op_id)
                                  + ": a C fragment spans several columns per lane; the "
                                    "per-column stats of the dQ kernel need one");
    }
}

int atom_k(const char* atom)
{
    std::string a = atom ? atom : "";
    size_t x = a.rfind('x');
    return std::stoi(a.substr(x + 1));
}

void from_spec(B* b, const rocke_attn_bwd_dq_spec_t* s, const char* arch, Geom* out)
{
    const int NONE = ROCKE_ATTN_BWD_NONE;
    if(s->waves == NONE || s->block_m == NONE || s->block_n == NONE)
        throw ckc::ValueError("BwdDqGeometry needs a resolved spec");
    memset(out, 0, sizeof(*out));
    out->arch = arch;
    out->dtype = s->dtype;
    out->waves = s->waves;
    out->k_m0 = s->block_m;
    out->k_n0 = s->block_n;
    out->head_size = s->head_size;
    out->atom_g02 = atom_k(s->atom_g02);
    out->atom_g4 = atom_k(s->atom_g4);
    out->transpose_source = s->transpose_source;
    out->sched = s->sched;
    out->kv_pad = s->kv_pad;
    out->kv_prefetch = s->kv_prefetch;
    out->stage_vec = s->stage_vec;
    geometry_check(b, out);
}

int wave_size_of(const Geom* g)
{
    rocke_attn_bwd_caps_t c;
    rocke_attn_bwd_err_t e;
    need_st(rocke_attn_bwd_arch_caps(g->arch, &c, &e), e);
    return c.wave_size;
}

int threads_of(const Geom* g)
{
    return g->waves * wave_size_of(g);
}

void stage_table(const Geom* g, rocke_attn_bwd_dq_stage_table_t* out)
{
    rocke_attn_bwd_dq_sched_geom_t sg;
    memset(&sg, 0, sizeof(sg));
    sg.arch = g->arch;
    sg.head_size = g->head_size;
    sg.k_m0 = g->k_m0;
    sg.k_n0 = g->k_n0;
    sg.waves = g->waves;
    sg.atom_g02 = g->atom_g02;
    sg.atom_g4 = g->atom_g4;
    sg.dtype = g->dtype;
    sg.transpose_source = g->transpose_source;
    sg.kv_pad = g->kv_pad;
    sg.has_kv_prefetch = true;
    sg.kv_prefetch = g->kv_prefetch;
    sg.stage_vec = g->stage_vec;
    sg.sched = g->sched;
    rocke_attn_bwd_err_t e;
    need_st(rocke_attn_bwd_dq_stage_table_from_geometry(&sg, out, &e), e);
}

/* ------------------------------------------------------------- the step */

rocke_attn_bwd_lds_operand_t lds_operand(const rocke_attn_bwd_lds_view_t* view,
                                         rocke_attn_bwd_storage_t storage)
{
    rocke_attn_bwd_lds_operand_t o = rocke_attn_bwd_lds_operand_default();
    o.smem = NULL;
    o.view = view;
    o.storage = storage;
    return o;
}

rocke_attn_bwd_gemm_operand_t lds_gemm_operand(const rocke_attn_bwd_lds_operand_t* o)
{
    rocke_attn_bwd_gemm_operand_t g;
    memset(&g, 0, sizeof(g));
    g.lds = o;
    return g;
}

rocke_attn_bwd_gemm_operand_t reg_gemm_operand(V* v, int rows, int cols)
{
    rocke_attn_bwd_gemm_operand_t g;
    memset(&g, 0, sizeof(g));
    g.lds = NULL;
    g.regs.v = v;
    g.regs.rows = rows;
    g.regs.cols = cols;
    return g;
}

void step_init(Step* t,
               B* b,
               const Geom* geom,
               rocke_attn_bwd_mask_policy_t* mask,
               const rocke_attn_bwd_stats_view_t* stats,
               const rocke_attn_bwd_dq_lds_views_t* lds,
               V tid,
               V lane,
               V wave,
               V scale_log2,
               V ds_mult,
               bool prescale,
               bool sentinel)
{
    memset(t, 0, sizeof(*t));
    t->geom = geom;
    t->mask = mask;
    t->stats = stats;
    t->lds = lds;
    t->tid = tid;
    t->lane = lane;
    t->wave = wave;
    t->scale_log2 = scale_log2;
    t->ds_mult = ds_mult;
    t->prescale = prescale;
    t->sentinel = sentinel;
    t->elem = rocke_attn_bwd_ir_type(b, geom->dtype);
    need(b, t->elem != NULL);
    rocke_attn_bwd_block_gemm_wave_origin(b, &geom->g0t, wave, t->o0);
    need(b, true);
    rocke_attn_bwd_block_gemm_wave_origin(b, &geom->g4t, wave, t->o4);
    need(b, true);
    t->qt_regs = NULL;
    t->dot_regs = NULL;
    t->n_cols = 0;
    stage_table(geom, &t->sched); /* the key step's hints (dq_sched) */
}

/* _column_stats(b, h, q0, rows) */
void column_stats(Step* t, B* b, V h, V q0, const rocke_attn_bwd_packed_rows_t* rows)
{
    const rocke_attn_bwd_block_gemm_t* g0t = &t->geom->g0t;
    int fn_n = FN(g0t);
    int clen = g0t->op->c_frag_len;
    std::vector<V> cols((size_t)fn_n);
    std::vector<V> coords((size_t)clen * 2);
    for(int fn = 0; fn < fn_n; ++fn)
    {
        need(b, rocke_attn_bwd_block_gemm_acc_coords(b, g0t, t->lane, t->o0, 0, fn, coords.data()));
        cols[(size_t)fn] = coords[1];
    }
    std::vector<V> lse2((size_t)fn_n), dsum((size_t)fn_n);
    if(rows == NULL)
        need(b,
             rocke_attn_bwd_stats_view_rows(
                 t->stats, b, h, q0, cols.data(), fn_n, lse2.data(), dsum.data()));
    else
        need(b,
             rocke_attn_bwd_stats_view_packed_rows(
                 t->stats, b, rows, cols.data(), fn_n, lse2.data(), dsum.data()));
    V inf = needv(b, rocke_b_const_f32(b, INF));
    t->n_cols = fn_n;
    t->col_lse2 = arena<V>(b, (size_t)fn_n);
    t->col_dsum = arena<V>(b, (size_t)fn_n);
    t->col_live = arena<V>(b, (size_t)fn_n);
    t->col_lo = arena<V>(b, (size_t)fn_n);
    t->col_hi = arena<V>(b, (size_t)fn_n);
    for(int fn = 0; fn < fn_n; ++fn)
    {
        V col = cols[(size_t)fn];
        V l2 = lse2[(size_t)fn];
        V ds = dsum[(size_t)fn];
        V lv = t->sentinel ? needv(b, rocke_b_fcmp(b, "olt", l2, inf)) : NULL;
        V q, ok;
        if(rows == NULL)
        {
            q = needv(b, rocke_b_add(b, q0, col));
            ok = NULL;
        }
        else
        {
            V hh;
            rocke_attn_bwd_packed_rows_locate(rows, b, ivv(col), &hh, &q, &ok);
            need(b, true);
        }
        if(lv != NULL)
            ok = ok == NULL ? lv : needv(b, rocke_b_land(b, ok, lv));
        V lo, hi;
        if(rocke_attn_bwd_mask_row_limits(t->mask, b, q, ok, &lo, &hi) != ROCKE_OK)
            raise_builder(b);
        need(b, true);
        if(t->prescale)
        {
            /* one fma per cell for the exp2 argument and one for dS */
            l2 = needv(b, rocke_b_fneg(b, l2));
            if(t->ds_mult != NULL)
            {
                V p = rocke_b_fmul(b, ds, t->ds_mult);
                ds = needv(b, rocke_b_fneg(b, p));
            }
        }
        t->col_lse2[fn] = l2;
        t->col_dsum[fn] = ds;
        t->col_live[fn] = lv;
        t->col_lo[fn] = lo;
        t->col_hi[fn] = hi;
    }
}

void copy_rows(B* b,
               const rocke_attn_bwd_row_slab_t* slab,
               const rocke_attn_bwd_lds_view_t* view,
               V row0,
               int rows,
               int cols,
               V tid,
               int threads,
               int vec)
{
    need(b, rocke_attn_bwd_copy_rows_to_lds(b, slab, view, row0, rows, cols, tid, threads, vec));
}

void stage_q(Step* t,
             B* b,
             V live_,
             V h,
             V q0,
             const rocke_attn_bwd_row_slab_t* q_slab,
             const rocke_attn_bwd_row_slab_t* do_slab,
             const rocke_attn_bwd_strided_tensor_addr_t* q_addr,
             const rocke_attn_bwd_strided_tensor_addr_t* do_addr,
             const rocke_attn_bwd_packed_rows_t* rows,
             int vec)
{
    const Geom* g = t->geom;
    int d = g->head_size, m0 = g->k_m0;
    int threads = threads_of(g);
    {
        IfScope f(b, live_);
        if(rows == NULL)
        {
            copy_rows(b, q_slab, &t->lds->q, q0, m0, d, t->tid, threads, vec);
            copy_rows(b, do_slab, &t->lds->do_, q0, m0, d, t->tid, threads, vec);
        }
        else
        {
            need(b,
                 rocke_attn_bwd_copy_packed_rows_to_lds(
                     b, q_addr, &t->lds->q, rows, m0, d, t->tid, threads, vec, true));
            need(b,
                 rocke_attn_bwd_copy_packed_rows_to_lds(
                     b, do_addr, &t->lds->do_, rows, m0, d, t->tid, threads, vec, true));
        }
    }
    sync(b);
    const rocke_attn_bwd_block_gemm_t* g0t = &g->g0t;
    size_t nb = (size_t)FN(g0t) * (size_t)KS(g0t);
    rocke_attn_bwd_lds_operand_t oq = lds_operand(&t->lds->q, ROCKE_ATTN_BWD_K_INNER);
    t->qt_regs = arena<V>(b, nb);
    need(b, rocke_attn_bwd_block_gemm_load_all_b(b, g0t, &oq, t->lane, t->o0[1], t->qt_regs));
    rocke_attn_bwd_lds_operand_t od = lds_operand(&t->lds->do_, ROCKE_ATTN_BWD_K_INNER);
    t->dot_regs = arena<V>(b, nb);
    need(b, rocke_attn_bwd_block_gemm_load_all_b(b, g0t, &od, t->lane, t->o0[1], t->dot_regs));
    column_stats(t, b, h, q0, rows);
    sync(b); /* stage-0 reads complete before the key loop writes K / V */
}

/* One _tile_loads entry: (r, c, value, guard). */
struct Pending
{
    V r, c, val, guard;
};

/* _tile_loads(b, slab, row0, vec) */
std::vector<Pending>
    tile_loads(Step* t, B* b, const rocke_attn_bwd_row_slab_t* slab, V row0, int vec)
{
    const Geom* g = t->geom;
    int per_row = g->head_size / vec;
    int total = g->k_n0 * per_row;
    int threads = threads_of(g);
    std::vector<Pending> out;
    int rounds = rocke_attn_bwd_deal_chunks_count(b, total, threads);
    need(b, rounds >= 0);
    for(int it = 0; it < rounds; ++it)
    {
        bool tail = false;
        V chunk = needv(b, rocke_attn_bwd_deal_chunks_round(b, t->tid, total, threads, it, &tail));
        V src = chunk;
        if(tail)
        {
            V last = ci(b, (int64_t)total - 1);
            src = needv(b, rocke_b_smin(b, chunk, last));
        }
        V r, c;
        rocke_attn_bwd_chunk_coords(b, src, per_row, vec, &r, &c);
        need(b, true);
        V guard = NULL;
        if(tail)
        {
            V ct = ci(b, total);
            guard = needv(b, rocke_b_cmp_lt(b, chunk, ct));
        }
        V row = needv(b, rocke_b_add(b, row0, r));
        V val = needv(b, rocke_attn_bwd_row_slab_load_vec(slab, row, ivv(c), vec));
        out.push_back(Pending{r, c, val, guard});
    }
    return out;
}

/* _tile_stores(b, view, pending, vec) */
void tile_stores(B* b,
                 const rocke_attn_bwd_lds_view_t* view,
                 const std::vector<Pending>& pend,
                 int vec)
{
    for(const Pending& p : pend)
    {
        if(p.guard == NULL)
        {
            rocke_attn_bwd_lds_view_store(b, view, ivv(p.r), ivv(p.c), p.val, vec);
            need(b, true);
        }
        else
        {
            IfScope f(b, p.guard);
            rocke_attn_bwd_lds_view_store(b, view, ivv(p.r), ivv(p.c), p.val, vec);
            need(b, true);
        }
    }
}

void prologue(Step* t,
              B* b,
              const rocke_attn_bwd_row_slab_t* k_slab,
              const rocke_attn_bwd_row_slab_t* v_slab,
              V k0,
              V live_)
{
    const Geom* g = t->geom;
    int threads = threads_of(g);
    {
        IfScope f(b, live_);
        copy_rows(b, k_slab, &t->lds->k, k0, g->k_n0, g->head_size, t->tid, threads, 8);
        copy_rows(b, v_slab, &t->lds->v, k0, g->k_n0, g->head_size, t->tid, threads, 8);
    }
    sync(b);
}

/* _rounding_point(b, v): v through an empty asm (fp32 materialised). */
V rounding_point(B* b, V v)
{
    rocke_inline_asm_opts_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.sideeffect = false;
    opts.sideeffect_set = true;
    const rocke_type_t* ty = rocke_f32();
    rocke_op_t* op = rocke_b_inline_asm(b, "", "=v,0", &v, 1, &ty, 1, &opts);
    if(op == NULL || op->num_results < 1)
        raise_builder(b);
    return op->results[0];
}

/* _softmax_grad_t(b, s_acc, dp_acc, k0, edge=): out[frags_m * frags_n]. */
void softmax_grad_t(Step* t, B* b, V* s_acc, V* dp_acc, V k0, bool edge, V* out)
{
    const rocke_attn_bwd_block_gemm_t* g0t = &t->geom->g0t;
    V zero = needv(b, rocke_b_const_f32(b, 0.0));
    V m = t->ds_mult;
    int fm_n = FM(g0t), fn_n = FN(g0t);
    int clen = g0t->op->c_frag_len;
    std::vector<V> coords((size_t)clen * 2);
    for(int fm = 0; fm < fm_n; ++fm)
        for(int fn = 0; fn < fn_n; ++fn)
        {
            V lse2 = t->col_lse2[fn], dsum = t->col_dsum[fn], lv = t->col_live[fn];
            V lo = t->col_lo[fn], hi = t->col_hi[fn];
            need(b,
                 rocke_attn_bwd_block_gemm_acc_coords(
                     b, g0t, t->lane, t->o0, fm, fn, coords.data()));
            std::vector<V> vals;
            size_t fi = (size_t)fm * fn_n + fn;
            for(int s = 0; s < clen; ++s)
            {
                V r = coords[(size_t)s * 2];
                V sv = needv(b, rocke_b_vec_extract(b, s_acc[fi], s));
                V dpv = needv(b, rocke_b_vec_extract(b, dp_acc[fi], s));
                V p;
                if(t->prescale) /* lse2 holds -lse2 */
                {
                    V f = rocke_b_fma(b, sv, t->scale_log2, lse2);
                    p = rocke_b_exp2(b, f);
                }
                else
                {
                    V x = rocke_b_fmul(b, sv, t->scale_log2);
                    V d = rocke_b_fsub(b, x, lse2);
                    p = rocke_b_exp2(b, d);
                }
                V cond;
                if(edge)
                {
                    V k = rocke_b_add(b, k0, r);
                    cond = needv(b, rocke_attn_bwd_mask_keep_in(b, lo, hi, k));
                    p = rocke_b_select(b, cond, p, zero);
                }
                else
                    cond = lv;
                V ds;
                if(t->prescale && m != NULL)
                {
                    /* dsum holds -ds_mult * Dsum: one fma per cell */
                    V f = rocke_b_fma(b, dpv, m, dsum);
                    ds = rocke_b_fmul(b, p, f);
                    if(cond != NULL)
                        ds = rocke_b_select(b, cond, ds, zero);
                }
                else
                {
                    V d = rocke_b_fsub(b, dpv, dsum);
                    ds = rocke_b_fmul(b, p, d);
                    if(cond != NULL)
                        ds = rocke_b_select(b, cond, ds, zero);
                    if(m != NULL) /* None: 1.0, not applied */
                        ds = rocke_b_fmul(b, m, ds);
                }
                need(b, ds != NULL);
                vals.push_back(rounding_point(b, ds));
            }
            out[fi] = needv(b, rocke_b_vec_pack(b, vals.data(), (int)vals.size(), rocke_f32()));
        }
}

void run_kv_step(Step* t,
                 B* b,
                 V const* dq,
                 const rocke_attn_bwd_row_slab_t* k_slab,
                 const rocke_attn_bwd_row_slab_t* v_slab,
                 V k0,
                 bool edge,
                 bool prefetch,
                 int vec,
                 V* out)
{
    const Geom* g = t->geom;
    const rocke_attn_bwd_block_gemm_t* g0t = &g->g0t;
    const rocke_attn_bwd_block_gemm_t* g4t = &g->g4t;
    int d = g->head_size, n0 = g->k_n0;
    V lane = t->lane;
    const rocke_attn_bwd_dq_stage_table_t* sched = &t->sched;
    int threads = threads_of(g);
    if(!prefetch)
    {
        copy_rows(b, k_slab, &t->lds->k, k0, n0, d, t->tid, threads, vec);
        copy_rows(b, v_slab, &t->lds->v, k0, n0, d, t->tid, threads, vec);
        sync(b);
    }
    need(b, rocke_attn_bwd_dq_stage_table_emit_loop_head(b, sched));
    need(b, rocke_attn_bwd_dq_stage_table_emit_stage_begin(b, sched, "A"));
    std::vector<Pending> pend_k, pend_v;
    if(prefetch)
    {
        V cn = ci(b, n0);
        V k_next = needv(b, rocke_b_add(b, k0, cn));
        pend_k = tile_loads(t, b, k_slab, k_next, vec);
        pend_v = tile_loads(t, b, v_slab, k_next, vec);
    }
    int fm0 = FM(g0t), fn0 = FN(g0t), ks0 = KS(g0t);
    size_t n_acc0 = (size_t)fm0 * fn0;
    rocke_attn_bwd_mma_opts_t opts = rocke_attn_bwd_mma_opts_default();
    /* G0': S^T = K Q^T */
    V* z0 = arena<V>(b, n_acc0);
    rocke_attn_bwd_block_gemm_zero_acc(b, g0t, z0);
    need(b, true);
    rocke_attn_bwd_frags_t acc0 = {z0, fm0, fn0};
    rocke_attn_bwd_lds_operand_t ok_ = lds_operand(&t->lds->k, ROCKE_ATTN_BWD_K_INNER);
    rocke_attn_bwd_gemm_operand_t a_k = lds_gemm_operand(&ok_);
    rocke_attn_bwd_gemm_operand_t b_q = reg_gemm_operand(t->qt_regs, fn0, ks0);
    V* s_acc = arena<V>(b, n_acc0);
    need(b, rocke_attn_bwd_block_gemm_mma(b, g0t, &acc0, &a_k, &b_q, lane, t->o0, &opts, s_acc));
    need(b, rocke_attn_bwd_dq_stage_table_emit_stage_end(b, sched, "A"));
    need(b, rocke_attn_bwd_dq_stage_table_emit_stage_begin(b, sched, "B"));
    /* G2': dP^T = V dO^T */
    V* z2 = arena<V>(b, n_acc0);
    rocke_attn_bwd_block_gemm_zero_acc(b, g0t, z2);
    need(b, true);
    rocke_attn_bwd_frags_t acc2 = {z2, fm0, fn0};
    rocke_attn_bwd_lds_operand_t ov = lds_operand(&t->lds->v, ROCKE_ATTN_BWD_K_INNER);
    rocke_attn_bwd_gemm_operand_t a_v = lds_gemm_operand(&ov);
    rocke_attn_bwd_gemm_operand_t b_do = reg_gemm_operand(t->dot_regs, fn0, ks0);
    V* dp_acc = arena<V>(b, n_acc0);
    need(b, rocke_attn_bwd_block_gemm_mma(b, g0t, &acc2, &a_v, &b_do, lane, t->o0, &opts, dp_acc));
    V* ds = arena<V>(b, n_acc0);
    softmax_grad_t(t, b, s_acc, dp_acc, k0, edge, ds);
    rocke_attn_bwd_frags_t ds_f = {ds, fm0, fn0};
    int fm4 = FM(g4t), fn4 = FN(g4t), ks4 = KS(g4t);
    V* a_ds = arena<V>(b, (size_t)fm4 * ks4);
    need(b, rocke_attn_bwd_relabel_acc_to_a(b, g0t, g4t, &ds_f, lane, "generic", a_ds, NULL));
    need(b, rocke_attn_bwd_dq_stage_table_emit_stage_end(b, sched, "B"));
    need(b, rocke_attn_bwd_dq_stage_table_emit_stage_begin(b, sched, "C"));
    /* G4': dQ += dS K (K read K-outer through the transpose source) */
    rocke_attn_bwd_lds_operand_t okt = lds_operand(&t->lds->k, ROCKE_ATTN_BWD_K_OUTER);
    okt.coords = &g->b4_coords;
    okt.reader = eq(rocke_attn_bwd_dq_reader(g->transpose_source), "tr16")
                     ? ROCKE_ATTN_BWD_READER_TR16
                     : ROCKE_ATTN_BWD_READER_PLAIN;
    need(b, rocke_attn_bwd_lds_operand_check(b, &okt));
    rocke_attn_bwd_gemm_operand_t a_dsop = reg_gemm_operand(a_ds, fm4, ks4);
    rocke_attn_bwd_gemm_operand_t b_kt = lds_gemm_operand(&okt);
    size_t n_acc4 = (size_t)fm4 * fn4;
    V* dq_in = arena<V>(b, n_acc4);
    for(size_t i = 0; i < n_acc4; ++i)
        dq_in[i] = dq[i];
    rocke_attn_bwd_frags_t acc4 = {dq_in, fm4, fn4};
    V* dq_out = arena<V>(b, n_acc4);
    need(b,
         rocke_attn_bwd_block_gemm_mma(b, g4t, &acc4, &a_dsop, &b_kt, lane, t->o4, &opts, dq_out));
    need(b, rocke_attn_bwd_dq_stage_table_emit_stage_end(b, sched, "C"));
    sync(b); /* K / V reads complete before the next step's copies */
    if(prefetch)
    {
        tile_stores(b, &t->lds->k, pend_k, vec);
        tile_stores(b, &t->lds->v, pend_v, vec);
        sync(b); /* the next tile is visible to every wave */
    }
    for(size_t i = 0; i < n_acc4; ++i)
        out[i] = dq_out[i];
}

void store_dq(Step* t,
              B* b,
              V const* dq,
              V mult,
              const rocke_attn_bwd_row_slab_t* dq_slab,
              V head_step,
              V h,
              V q0,
              const rocke_attn_bwd_packed_rows_t* rows,
              V len_q,
              V s_q_max,
              rocke_attn_bwd_seq_mode_t seq_mode)
{
    const Geom* g = t->geom;
    const rocke_attn_bwd_block_gemm_t* g4t = &g->g4t;
    int d = g->head_size, m0 = g->k_m0;
    const rocke_attn_bwd_lds_view_t* out = &t->lds->out;
    int n = g4t->op->c_frag_len;
    int fm_n = FM(g4t), fn_n = FN(g4t);
    std::vector<V> coords((size_t)n * 2);
    for(int fm = 0; fm < fm_n; ++fm)
        for(int fn = 0; fn < fn_n; ++fn)
        {
            V frag = dq[(size_t)fm * fn_n + fn];
            if(mult != NULL)
            {
                std::vector<V> xs;
                for(int i = 0; i < n; ++i)
                {
                    V e = rocke_b_vec_extract(b, frag, i);
                    xs.push_back(needv(b, rocke_b_fmul(b, mult, e)));
                }
                frag = needv(b, rocke_b_vec_pack(b, xs.data(), n, rocke_f32()));
            }
            V half = needv(b, rocke_b_vec_cast_f32_to(b, frag, t->elem));
            need(b,
                 rocke_attn_bwd_block_gemm_acc_coords(
                     b, g4t, t->lane, t->o4, fm, fn, coords.data()));
            for(int s = 0; s < n; ++s)
            {
                V e = needv(b, rocke_b_vec_extract(b, half, s));
                rocke_attn_bwd_lds_view_store(
                    b, out, ivv(coords[(size_t)s * 2]), ivv(coords[(size_t)s * 2 + 1]), e, 1);
                need(b, true);
            }
        }
    sync(b);
    const int vec = 8;
    int per_row = d / vec;
    int total = m0 * per_row;
    V zero = needv(b, rocke_b_zero_vec(b, t->elem, vec));
    if(dq_slab->last != NULL || (rows == NULL) != (head_step == NULL))
        throw ckc::ValueError("store_dq: dq_slab must be unclamped and head_step set exactly "
                              "with rows (AssertionError)");
    int threads = threads_of(g);
    int rounds = rocke_attn_bwd_deal_chunks_count(b, total, threads);
    need(b, rounds >= 0);
    rocke_int_or_value_t smax = rocke_iov_value(s_q_max);
    for(int it = 0; it < rounds; ++it)
    {
        bool tail = false;
        V chunk = needv(b, rocke_attn_bwd_deal_chunks_round(b, t->tid, total, threads, it, &tail));
        V r, c;
        rocke_attn_bwd_chunk_coords(b, chunk, per_row, vec, &r, &c);
        need(b, true);
        V q, ok;
        rocke_attn_bwd_row_slab_t row_slab = *dq_slab;
        if(rows == NULL)
        {
            q = needv(b, rocke_b_add(b, q0, r));
            ok = NULL;
        }
        else
        {
            V hh;
            rocke_attn_bwd_packed_rows_locate(rows, b, ivv(r), &hh, &q, &ok);
            need(b, true);
            /* hh >= h: the packed heads are h .. h + G - 1 */
            V dh = rocke_b_sub(b, hh, h);
            V dh64 = rocke_b_zext(b, dh, rocke_i64());
            V shift = needv(b, rocke_b_mul(b, dh64, head_step));
            row_slab.ptr = needv(b, rocke_b_global_ptr_add(b, dq_slab->ptr, shift));
        }
        V store, row_live;
        if(rocke_attn_bwd_band_dkv_store_rule(b, q, len_q, seq_mode, &smax, &store, &row_live)
           != ROCKE_OK)
            raise_builder(b);
        need(b, true);
        if(ok != NULL)
            store = needv(b, rocke_b_land(b, store, ok));
        /* Unreachable for a spec validate_attn_bwd_dq_spec accepts: block_m
         * = 16 * waves * k, head_size is 32/64/128 and both split-dQ arches
         * are wave64, so total = 2 * waves * k * head_size is a multiple of
         * threads = 64 * waves. Kept as the twin of Python's branch. */
        if(tail)
        {
            V ct = ci(b, total);
            V in = rocke_b_cmp_lt(b, chunk, ct);
            store = needv(b, rocke_b_land(b, store, in));
        }
        IfScope f(b, store);
        V ld = needv(b, rocke_attn_bwd_lds_view_load(b, out, ivv(r), ivv(c), t->elem, vec));
        V v = needv(b, rocke_b_select(b, row_live, ld, zero));
        rocke_attn_bwd_row_slab_store_vec(&row_slab, q, ivv(c), v);
        need(b, true);
    }
}

} // namespace

/* ====================================================================== *
 * C entry points
 * ====================================================================== */

extern "C" {

const char* rocke_attn_bwd_dq_reader(const char* transpose_source)
{
    if(eq(transpose_source, "tr_read"))
        return "tr16";
    if(eq(transpose_source, "lds_plain"))
        return "plain";
    return NULL;
}

bool rocke_attn_bwd_kv_tile_range(rocke_ir_builder_t* b,
                                  rocke_band_emitter_t* band,
                                  rocke_value_t* q_first,
                                  rocke_attn_bwd_iv_t n_rows,
                                  int k_n0,
                                  bool uniform,
                                  rocke_value_t** out_kt_start,
                                  rocke_value_t** out_n_kt)
{
    return guarded(b, [&]() {
        kv_tile_range(b, band, q_first, n_rows, k_n0, uniform, out_kt_start, out_n_kt);
    });
}

bool rocke_attn_bwd_interior_kv_range(rocke_ir_builder_t* b,
                                      rocke_band_emitter_t* band,
                                      rocke_value_t* q_first,
                                      rocke_attn_bwd_iv_t n_rows,
                                      rocke_value_t* kt_start,
                                      rocke_value_t* n_kt,
                                      int k_n0,
                                      bool masked,
                                      bool uniform,
                                      rocke_value_t** out_i0,
                                      rocke_value_t** out_i1)
{
    return guarded(b, [&]() {
        interior_kv_range(
            b, band, q_first, n_rows, kt_start, n_kt, k_n0, masked, uniform, out_i0, out_i1);
    });
}

bool rocke_attn_bwd_dq_geometry_from_spec(rocke_ir_builder_t* b,
                                          const struct rocke_attn_bwd_dq_spec* spec,
                                          const char* arch,
                                          rocke_attn_bwd_dq_geometry_t* out)
{
    return guarded(b, [&]() {
        if(spec == NULL || out == NULL)
            throw ckc::ValueError("BwdDqGeometry.from_spec: NULL spec or output");
        from_spec(b, spec, arch, out);
    });
}

int rocke_attn_bwd_dq_geometry_wave_size(const rocke_attn_bwd_dq_geometry_t* g)
{
    rocke_attn_bwd_caps_t c;
    rocke_attn_bwd_err_t e;
    if(g == NULL || rocke_attn_bwd_arch_caps(g->arch, &c, &e) != ROCKE_OK)
        return 0;
    return c.wave_size;
}

int rocke_attn_bwd_dq_geometry_threads(const rocke_attn_bwd_dq_geometry_t* g)
{
    return g == NULL ? 0 : g->waves * rocke_attn_bwd_dq_geometry_wave_size(g);
}

const char* rocke_attn_bwd_dq_geometry_reader(const rocke_attn_bwd_dq_geometry_t* g)
{
    return g == NULL ? NULL : rocke_attn_bwd_dq_reader(g->transpose_source);
}

bool rocke_attn_bwd_dq_geometry_stage_table(rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_dq_geometry_t* g,
                                            rocke_attn_bwd_dq_stage_table_t* out)
{
    return guarded(b, [&]() { stage_table(g, out); });
}

bool rocke_attn_bwd_dq_step_init(rocke_attn_bwd_dq_step_t* t,
                                 rocke_ir_builder_t* b,
                                 const rocke_attn_bwd_dq_geometry_t* geom,
                                 rocke_attn_bwd_mask_policy_t* mask,
                                 const rocke_attn_bwd_stats_view_t* stats,
                                 const rocke_attn_bwd_dq_lds_views_t* lds,
                                 rocke_value_t* tid,
                                 rocke_value_t* lane,
                                 rocke_value_t* wave,
                                 rocke_value_t* scale_log2,
                                 rocke_value_t* ds_mult,
                                 bool prescale,
                                 bool sentinel)
{
    return guarded(b, [&]() {
        step_init(
            t, b, geom, mask, stats, lds, tid, lane, wave, scale_log2, ds_mult, prescale, sentinel);
    });
}

bool rocke_attn_bwd_dq_step_stage_q(rocke_attn_bwd_dq_step_t* t,
                                    rocke_ir_builder_t* b,
                                    rocke_value_t* live_,
                                    rocke_value_t* h,
                                    rocke_value_t* q0,
                                    const rocke_attn_bwd_row_slab_t* q_slab,
                                    const rocke_attn_bwd_row_slab_t* do_slab,
                                    const rocke_attn_bwd_strided_tensor_addr_t* q_addr,
                                    const rocke_attn_bwd_strided_tensor_addr_t* do_addr,
                                    const rocke_attn_bwd_packed_rows_t* rows,
                                    int vec)
{
    return guarded(
        b, [&]() { stage_q(t, b, live_, h, q0, q_slab, do_slab, q_addr, do_addr, rows, vec); });
}

bool rocke_attn_bwd_dq_step_zero_dq(rocke_attn_bwd_dq_step_t* t,
                                    rocke_ir_builder_t* b,
                                    rocke_value_t** out)
{
    return guarded(b, [&]() { rocke_attn_bwd_block_gemm_zero_acc(b, &t->geom->g4t, out); });
}

bool rocke_attn_bwd_dq_step_prologue(rocke_attn_bwd_dq_step_t* t,
                                     rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_row_slab_t* k_slab,
                                     const rocke_attn_bwd_row_slab_t* v_slab,
                                     rocke_value_t* k0,
                                     rocke_value_t* live_)
{
    return guarded(b, [&]() { prologue(t, b, k_slab, v_slab, k0, live_); });
}

bool rocke_attn_bwd_dq_step_run_kv_step(rocke_attn_bwd_dq_step_t* t,
                                        rocke_ir_builder_t* b,
                                        rocke_value_t* const* dq,
                                        const rocke_attn_bwd_row_slab_t* k_slab,
                                        const rocke_attn_bwd_row_slab_t* v_slab,
                                        rocke_value_t* k0,
                                        bool edge,
                                        bool prefetch,
                                        int vec,
                                        rocke_value_t** out)
{
    return guarded(b,
                   [&]() { run_kv_step(t, b, dq, k_slab, v_slab, k0, edge, prefetch, vec, out); });
}

bool rocke_attn_bwd_dq_step_store_dq(rocke_attn_bwd_dq_step_t* t,
                                     rocke_ir_builder_t* b,
                                     rocke_value_t* const* dq,
                                     rocke_value_t* mult,
                                     const rocke_attn_bwd_row_slab_t* dq_slab,
                                     rocke_value_t* head_step,
                                     rocke_value_t* h,
                                     rocke_value_t* q0,
                                     const rocke_attn_bwd_packed_rows_t* rows,
                                     rocke_value_t* len_q,
                                     rocke_value_t* s_q_max,
                                     rocke_attn_bwd_seq_mode_t seq_mode)
{
    return guarded(b, [&]() {
        store_dq(t, b, dq, mult, dq_slab, head_step, h, q0, rows, len_q, s_q_max, seq_mode);
    });
}

} /* extern "C" */
