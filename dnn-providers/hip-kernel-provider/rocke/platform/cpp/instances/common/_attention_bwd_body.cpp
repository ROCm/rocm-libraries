// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_bwd_body.py: the shared inner
 * body of the attention backward main kernel (tile geometry, flat LDS, stats
 * view, copy helpers, BwdTileStep and its per-q-step G0..G4 program).
 *
 * See rocke/helper_rocke.instances.common._attention_bwd_body.h for the
 * mapping. Every op is created in the Python evaluation order; each `py:`
 * comment quotes the Python expression a block reproduces. Internal code
 * throws ckc::Error where Python raises; the extern "C" entry points catch it
 * (ckc::guard_builder) and leave it on the builder.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_body.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_ring.h"

#include <stdio.h>
#include <string.h>

#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "rocke/error.hpp"
#include "rocke/error_boundary.hpp"
#include "rocke/instance_attention_bwd.h"
#include "rocke/ir_internal.h"

/* ====================================================================== *
 * constants
 * ====================================================================== */

extern "C" {

const char* const ROCKE_ATTN_BWD_STAGE_ORDERS[ROCKE_ATTN_BWD_N_STAGE_ORDERS]
    = {"g1g3g4", "g3g1g4", "g4g1g3", "g1g4g3", "g3g4g1", "g4g3g1"};
const char* const ROCKE_ATTN_BWD_DEFAULT_STAGE_ORDER = "g1g3g4";
const char* const ROCKE_ATTN_BWD_G4_SLICINGS[ROCKE_ATTN_BWD_N_G4_SLICINGS]
    = {"step", "slice", "slice_db", "slice_db_g3"};
const char* const ROCKE_ATTN_BWD_DEFAULT_G4_SLICING = "step";
const char* const ROCKE_ATTN_BWD_G4_AFTER_G3_ORDERS[ROCKE_ATTN_BWD_N_G4_AFTER_G3_ORDERS]
    = {"g1g3g4", "g3g4g1"};
const char* const ROCKE_ATTN_BWD_COPY_ORDERS[ROCKE_ATTN_BWD_N_COPY_ORDERS]
    = {"split", "loads_first", "joint"};

} /* extern "C" */

namespace
{

typedef rocke_value_t* V;
typedef rocke_attn_bwd_iv_t IV;
typedef rocke_ir_builder_t B;

const char* const FENCE = ROCKE_ATTN_BWD_BARRIER_FENCE;
const double INF = std::numeric_limits<double>::infinity();
const int NONE = ROCKE_ATTN_BWD_NONE;
/* _attention_bwd_lds.DKV_EPILOGUES */
const char* const DKV_EPILOGUES[] = {"lds_staged", "lds_staged_pair", "direct"};

/* ------------------------------------------------------------ plumbing */

bool live(const B* b)
{
    return b != NULL && b->status == ROCKE_OK;
}

[[noreturn]] void raise_builder(B* b)
{
    if(b != NULL && b->status != ROCKE_OK)
        ckc::raise_status(b->status, b->err);
    throw ckc::ValueError("attention backward body: a helper failed");
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

/* BwdTileGeometry.wave_size / threads inside a guarded body: an unknown arch
 * raises bwd_arch_caps' ValueError (the C entry points return 0 instead). */
int wave_size_of(const rocke_attn_bwd_tile_geometry_t* g)
{
    rocke_attn_bwd_caps_t c;
    rocke_attn_bwd_err_t e;
    need_st(rocke_attn_bwd_arch_caps(g->arch, &c, &e), e);
    return c.wave_size;
}

int threads_of(const rocke_attn_bwd_tile_geometry_t* g)
{
    int waves = g->waves;
    return waves * wave_size_of(g);
}

[[noreturn]] void value_error(const std::string& msg)
{
    throw ckc::ValueError(msg);
}

bool eq(const char* a, const char* b)
{
    return a != NULL && b != NULL && strcmp(a, b) == 0;
}

std::string s_(const char* s)
{
    return s ? std::string(s) : std::string("None");
}

/* Python repr of a str (or None). */
std::string repr(const char* s)
{
    if(s == NULL)
        return "None";
    std::string out = "'";
    for(const char* p = s; *p; ++p)
    {
        if(*p == '\\' || *p == '\'')
            out += '\\';
        out += *p;
    }
    return out + "'";
}

std::string tuple_repr(const char* const* xs, int n)
{
    std::string out = "(";
    for(int i = 0; i < n; ++i)
    {
        if(i)
            out += ", ";
        out += repr(xs[i]);
    }
    return out + (n == 1 ? ",)" : ")");
}

bool in(const char* x, const char* const* xs, int n)
{
    for(int i = 0; i < n; ++i)
        if(eq(x, xs[i]))
            return true;
    return false;
}

template <class T>
T* arena(B* b, size_t n)
{
    void* p = rocke_arena_calloc(&b->arena, (n ? n : 1) * sizeof(T));
    if(p == NULL)
        throw ckc::OOMError("attention backward body: arena allocation failed");
    return (T*)p;
}

IV ivi(int64_t i)
{
    return rocke_attn_bwd_iv_int(i);
}

IV ivv(V v)
{
    return rocke_attn_bwd_iv_val(v);
}

V ci(B* b, int64_t x)
{
    return rocke_b_const_i32(b, x);
}

void sync(B* b)
{
    rocke_b_sync_fence(b, FENCE);
    need(b, true);
}

V vload(B* b, const rocke_attn_bwd_lds_view_t* v, IV r, IV c, const rocke_type_t* t, int n)
{
    return needv(b, rocke_attn_bwd_lds_view_load(b, v, r, c, t, n));
}

void vstore(B* b, const rocke_attn_bwd_lds_view_t* v, IV r, IV c, V val, int n)
{
    rocke_attn_bwd_lds_view_store(b, v, r, c, val, n);
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

/* ckc::guard_builder around a void body: false after an error. */
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

/* ------------------------------------------------------ gemm shortcuts */

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

int CFL(const rocke_attn_bwd_block_gemm_t* g)
{
    return g->op->c_frag_len;
}

/* acc_coords(b, lane, origin, fm, fn): out[slot * 2 + 0 / 1] = (row, col). */
std::vector<V>
    acc_coords(B* b, const rocke_attn_bwd_block_gemm_t* g, V lane, V* const origin, int fm, int fn)
{
    std::vector<V> out(2 * (size_t)CFL(g));
    need(b, rocke_attn_bwd_block_gemm_acc_coords(b, g, lane, origin, fm, fn, out.data()));
    return out;
}

rocke_attn_bwd_frags_t frags_of(V* v, int rows, int cols)
{
    rocke_attn_bwd_frags_t f;
    f.v = v;
    f.rows = rows;
    f.cols = cols;
    return f;
}

rocke_attn_bwd_frags_t no_frags()
{
    return frags_of(NULL, 0, 0);
}

/* An LdsOperand (the view is borrowed) or register fragments. */
struct Opnd
{
    bool is_lds = false;
    rocke_attn_bwd_lds_operand_t lds;
    rocke_attn_bwd_frags_t regs = no_frags();

    rocke_attn_bwd_gemm_operand_t g() const
    {
        rocke_attn_bwd_gemm_operand_t o;
        o.lds = is_lds ? &lds : NULL;
        o.regs = regs;
        return o;
    }
};

/* LdsOperand(view, storage, coords=, reader=) with its __post_init__. */
Opnd lds_opnd(B* b,
              const rocke_attn_bwd_lds_view_t* view,
              rocke_attn_bwd_storage_t storage = ROCKE_ATTN_BWD_K_INNER,
              const rocke_attn_bwd_frag_coords_t* coords = NULL,
              const char* reader = "plain")
{
    Opnd o;
    o.is_lds = true;
    o.lds = rocke_attn_bwd_lds_operand_default();
    o.lds.view = view;
    o.lds.storage = storage;
    o.lds.coords = coords;
    if(eq(reader, "plain"))
        o.lds.reader = ROCKE_ATTN_BWD_READER_PLAIN;
    else if(eq(reader, "tr16"))
        o.lds.reader = ROCKE_ATTN_BWD_READER_TR16;
    else
        value_error("reader must be 'plain' or 'tr16', got " + repr(reader));
    need(b, rocke_attn_bwd_lds_operand_check(b, &o.lds));
    return o;
}

Opnd regs_opnd(rocke_attn_bwd_frags_t f)
{
    Opnd o;
    o.regs = f;
    return o;
}

/* Python's int(str(atom).rsplit("x", 1)[1]). */
int atom_k_of(const char* atom)
{
    std::string s = s_(atom);
    size_t x = s.rfind('x');
    if(x == std::string::npos)
        value_error("list index out of range");
    std::string tail = s.substr(x + 1);
    char* end = NULL;
    long v = strtol(tail.c_str(), &end, 10);
    if(tail.empty() || end == NULL || *end != '\0')
        value_error("invalid literal for int() with base 10: " + repr(tail.c_str()));
    return (int)v;
}

bool truthy(int optional_bool)
{
    return optional_bool != NONE && optional_bool != 0;
}

const char* reader_of(const char* transpose_source)
{
    if(eq(transpose_source, "lds_plain") || eq(transpose_source, "wmma_lds"))
        return "plain";
    if(eq(transpose_source, "tr_read"))
        return "tr16";
    if(eq(transpose_source, "xt_lds"))
        return "xt";
    return NULL;
}

rocke_attn_bwd_caps_t arch_caps(const char* arch)
{
    rocke_attn_bwd_caps_t c;
    rocke_attn_bwd_err_t e;
    need_st(rocke_attn_bwd_arch_caps(arch, &c, &e), e);
    return c;
}

/* ====================================================================== *
 * geometry
 * ====================================================================== */

void make_gemm(B* b,
               const rocke_attn_bwd_tile_geometry_t* g,
               int m,
               int n,
               int k,
               const int grid[2],
               int atom_k,
               rocke_attn_bwd_block_gemm_t* out)
{
    rocke_attn_bwd_gemm_caps_t caps;
    rocke_attn_bwd_err_t e;
    need_st(rocke_attn_bwd_gemm_caps_fill(g->arch, g->dtype, atom_k, &caps, &e), e);
    need(b,
         rocke_attn_bwd_block_gemm_init(
             out, b, g->arch, g->dtype, m, n, k, grid[0], grid[1], atom_k, &caps));
}

void stage_units_(const char* order, const char* out[3])
{
    rocke_attn_bwd_err_t e;
    need_st(rocke_attn_bwd_stage_units(order, out, &e), e);
}

void check_counted(B* b, const rocke_attn_bwd_tile_geometry_t* g);
void check_g4_slicing(const rocke_attn_bwd_tile_geometry_t* g);

void geometry_check(B* b, rocke_attn_bwd_tile_geometry_t* g)
{
    /* the stage-table model reads the same set (an import-time assertion in
     * Python) */
    for(int i = 0; i < ROCKE_ATTN_BWD_N_G4_SLICINGS; ++i)
        if(ROCKE_ATTN_BWD_SCHED_N_G4_SLICINGS != ROCKE_ATTN_BWD_N_G4_SLICINGS
           || !eq(ROCKE_ATTN_BWD_G4_SLICINGS[i], ROCKE_ATTN_BWD_SCHED_G4_SLICINGS[i]))
            value_error("G4_SLICINGS differs from the stage-table model's set");
    static const char* const readers[] = {"lds_plain", "wmma_lds", "tr_read", "xt_lds"};
    if(reader_of(g->transpose_source) == NULL)
        value_error("transpose_source=" + repr(g->transpose_source)
                    + " has no operand reader in this body (built: " + tuple_repr(readers, 4)
                    + ")");
    int w = g->waves;
    if(!(g->grid_g0[0] == 1 && g->grid_g0[1] == w) || !(g->grid_g1[0] == w && g->grid_g1[1] == 1))
        if(eq(g->pt_route, "relabel"))
            value_error("the relabel route needs G0 grid (1, W), G1 (W, 1)");
    int w0 = g->grid_g0[0] * g->grid_g0[1];
    if(w0 == 0)
        value_error("integer division or modulo by zero");
    if(w0 > w || w % w0)
        value_error("G0 grid (" + std::to_string(g->grid_g0[0]) + ", "
                    + std::to_string(g->grid_g0[1]) + ") must hold W=" + std::to_string(w)
                    + " waves or a divisor of W (a replicated grid)");
    const char* gnames[2] = {"G1", "G4"};
    const int* grids[2] = {g->grid_g1, g->grid_g4};
    for(int i = 0; i < 2; ++i)
        if(grids[i][0] * grids[i][1] != w)
            value_error(std::string(gnames[i]) + " grid (" + std::to_string(grids[i][0]) + ", "
                        + std::to_string(grids[i][1]) + ") must hold W=" + std::to_string(w)
                        + " waves");
    /* the cached block GEMMs: Python builds the tuple (g0, g1, g4) before
     * the facts loop */
    rocke_attn_bwd_block_gemm_t* gs[3] = {&g->g0, &g->g1, &g->g4};
    make_gemm(b, g, g->k_m0, g->k_n0, g->head_size, g->grid_g0, g->atoms.g02, &g->g0);
    make_gemm(b, g, g->k_n0, g->head_size, g->k_m0, g->grid_g1, g->atoms.g13, &g->g1);
    make_gemm(b, g, g->k_m0, g->head_size, g->k_n0, g->grid_g4, g->atoms.g4, &g->g4);
    for(int i = 0; i < 3; ++i)
    {
        rocke_attn_bwd_atom_facts_t facts;
        need(b, rocke_attn_bwd_atom_facts(b, gs[i]->op, &facts));
        if(!(facts.a_contig && facts.b_contig && facts.affine))
        {
            /* Python's repr of the AtomFacts dataclass */
            auto tf = [](bool v) { return std::string(v ? "True" : "False"); };
            value_error(std::string(gs[i]->op->op_id)
                        + ": lane maps are not K-contiguous and affine (AtomFacts(op_id="
                        + repr(facts.op_id) + ", a_contig=" + tf(facts.a_contig)
                        + ", b_contig=" + tf(facts.b_contig) + ", affine=" + tf(facts.affine)
                        + ", c_is_aT=" + tf(facts.c_is_aT) + ", ab_same_k=" + tf(facts.ab_same_k)
                        + ")); this geometry cannot be built on " + s_(g->arch));
        }
    }
    /* relabel_plan (proves the relabel) and b13_coords. Python's b13_coords
     * is lazy; building it here is equivalent: without a K permutation it is
     * operand_coords(g1, "b"), whose only raise (_deltas_of) is the facts
     * loop's affine test above; with one, check() reads it at this point. */
    g->has_relabel_plan = eq(g->pt_route, "relabel");
    if(g->has_relabel_plan)
        need(b, rocke_attn_bwd_check_relabel_grids(b, &g->g0, &g->g1, &g->relabel_plan));
    if(!g->has_relabel_plan)
        need(b, rocke_attn_bwd_operand_coords(b, g->g1.op, 'b', &g->b13_coords));
    else
        need(b,
             rocke_attn_bwd_kperm_b_operand_coords(
                 b, &g->relabel_plan, &g->g0, &g->g1, &g->b13_coords));
    g->derived = true;
    if(g->has_relabel_plan && rocke_attn_bwd_relabel_plan_needs_k_perm(&g->relabel_plan)
       && !g->b13_coords.kperm)
        value_error("the relabel permutes the G1 / G3 K order; their B operand must be loaded "
                    "with the same permutation");
    if(eq(reader_of(g->transpose_source), "tr16"))
    {
        rocke_attn_bwd_err_t e;
        need_st(rocke_attn_bwd_check_transpose_read(g->arch, 64, &e), e);
        rocke_attn_bwd_tr16_read_plan_t plan;
        need(b, rocke_attn_bwd_tr16_read_plan(b, &g->b13_coords, &plan));
        rocke_attn_bwd_frag_coords_t c4;
        need(b, rocke_attn_bwd_operand_coords(b, g->g4.op, 'b', &c4));
        need(b, rocke_attn_bwd_check_tr16_coords(b, &c4) != 0);
        if(eq(g->pt_route, "lds"))
        {
            rocke_attn_bwd_frag_coords_t c1;
            need(b, rocke_attn_bwd_operand_coords(b, g->g1.op, 'a', &c1));
            need(b, rocke_attn_bwd_check_tr16_coords(b, &c1) != 0);
        }
    }
    if(!eq(g->dq_mode, "atomic") && !eq(g->dq_mode, "split"))
        value_error("dq_mode=" + repr(g->dq_mode) + " not in ('atomic', 'split')");
    const char* units[3];
    stage_units_(g->stage_order, units);
    if(!eq(g->stage_order, ROCKE_ATTN_BWD_DEFAULT_STAGE_ORDER)
       && in(g->sched, ROCKE_ATTN_BWD_STAGE_TABLE_SCHEDS, ROCKE_ATTN_BWD_N_STAGE_TABLE_SCHEDS))
        value_error("stage_order=" + repr(g->stage_order) + " with sched=" + repr(g->sched)
                    + ": the stage-table count model assumes the default G1, G3, G4 order");
    if(!in(g->dkv_epilogue, DKV_EPILOGUES, 3))
        value_error("dkv_epilogue must be one of " + tuple_repr(DKV_EPILOGUES, 3));
    if(g->dkv_store_vec != 8 && g->dkv_store_vec != 4 && g->dkv_store_vec != 2
       && g->dkv_store_vec != 1)
        value_error("dkv_store_vec must be 8, 4, 2 or 1");
    if(g->acc_in_lds && eq(g->dkv_epilogue, "lds_staged_pair"))
        value_error("acc_in_lds has no dK / dV staging buffer to pair");
    if(!eq(g->dq_issue, "block") && !eq(g->dq_issue, "fragment")
       && !eq(g->dq_issue, "fragment_regs"))
        value_error("dq_issue=" + repr(g->dq_issue) + " is not built in this body");
    if(g->asm_load_hoist && (!eq(g->agpr_route, "asm_g13") || g->acc_in_lds))
        value_error("asm_load_hoist needs the asm_g13 route with register accumulators");
    if(!eq(g->bf16_cast, "generic") && !eq(g->dtype, "bf16"))
        value_error("bf16_cast=" + repr(g->bf16_cast) + " needs dtype='bf16'");
    if(g->stats_vec != 1)
        need(b, rocke_attn_bwd_check_stats_vec_layout(b, g->g0.op, g->stats_vec));
    if(!eq(g->barrier_merge, "off") && !eq(g->barrier_merge, "on"))
        value_error("barrier_merge=" + repr(g->barrier_merge) + " is not built");
    if(!eq(g->ring_issue, "early") && !eq(g->ring_issue, "late"))
        value_error("ring_issue=" + repr(g->ring_issue) + " is not built");
    if(eq(g->barrier_merge, "on")
       && (g->ring_depth != 2 || !eq(g->global_path, "vgpr") || !eq(g->pt_route, "relabel")
           || g->acc_in_lds || !eq(units[2], "g4")))
        value_error("barrier_merge='on' needs ring_depth=2, global_path='vgpr', the relabel "
                    "route, register dK / dV and G4 last in stage_order");
    check_counted(b, g);
    /* _check_sched */
    if(!(eq(g->sched, "none") && !g->setprio))
        if(!eq(g->dq_mode, "atomic") && eq(g->sched, "stage_table_atomics"))
            value_error(ROCKE_ATTN_BWD_SPLIT_DEFERRED);
    check_g4_slicing(g);
}

void check_counted(B* b, const rocke_attn_bwd_tile_geometry_t* g)
{
    if(!in(g->counted_issue, ROCKE_ATTN_BWD_COUNTED_ISSUES, ROCKE_ATTN_BWD_COUNTED_N_ISSUES))
        value_error("counted_issue must be one of "
                    + tuple_repr(ROCKE_ATTN_BWD_COUNTED_ISSUES, ROCKE_ATTN_BWD_COUNTED_N_ISSUES));
    if(!eq(g->loop_sync, "counted"))
    {
        if(!eq(g->counted_issue, ROCKE_ATTN_BWD_COUNTED_ISSUES[0]))
            value_error("counted_issue applies to loop_sync='counted' only");
        return;
    }
    if(!eq(arch_caps(g->arch).matrix_path, "mfma"))
        value_error("loop_sync='counted' is CDNA-only (RDNA counts no-return atomics in vscnt, "
                    "not vmcnt)");
    if(!(eq(g->global_path, "vgpr") && g->ring_depth == 1))
        value_error("loop_sync='counted' owns the next-step loads: it needs global_path='vgpr' "
                    "and ring_depth=1");
    if(eq(g->sched, "stage_table_atomics") && eq(g->counted_issue, "late"))
        value_error(ROCKE_ATTN_BWD_COUNTED_LATE_DEFERRED);
    if(!(eq(g->qdo_stage, "split") && eq(g->stats_load, "at_use")))
        value_error("loop_sync='counted' owns the Q / dO and stats issue order: it needs "
                    "qdo_stage='split' and stats_load='at_use'");
    rocke_attn_bwd_counted_waits_t w;
    bool ok
        = rocke_attn_bwd_counted_waits_init(&w, b, rocke_attn_bwd_tile_geometry_counted_atomics(g));
    rocke_attn_bwd_counted_waits_free(&w);
    need(b, ok);
}

void check_g4_slicing(const rocke_attn_bwd_tile_geometry_t* g)
{
    if(!in(g->g4_slicing, ROCKE_ATTN_BWD_G4_SLICINGS, ROCKE_ATTN_BWD_N_G4_SLICINGS))
        value_error("g4_slicing=" + repr(g->g4_slicing) + " not in "
                    + tuple_repr(ROCKE_ATTN_BWD_G4_SLICINGS, ROCKE_ATTN_BWD_N_G4_SLICINGS));
    if(eq(g->g4_slicing, ROCKE_ATTN_BWD_DEFAULT_G4_SLICING))
        return;
    std::string why;
    if(g->acc_in_lds || !arch_caps(g->arch).has_agprs)
        why = "MFMA register accumulators (acc_in_lds=False)";
    else if(!eq(g->dq_issue, "block"))
        why = "dq_issue='block' (the per-fragment issue reads dS its own way)";
    else if(g->atoms.g4 == 0 || g->k_k4 == 0)
        value_error("integer division or modulo by zero");
    else if(g->k_k4 % g->atoms.g4 || g->k_n0 % g->k_k4)
        why = "kK4 a multiple of the G4 atom K that divides kN0";
    else if(rocke_attn_bwd_tile_geometry_g4_prefetch_before_g3(g)
            && !in(g->stage_order,
                   ROCKE_ATTN_BWD_G4_AFTER_G3_ORDERS,
                   ROCKE_ATTN_BWD_N_G4_AFTER_G3_ORDERS))
        why = "a stage_order with G3 right before G4 "
              + tuple_repr(ROCKE_ATTN_BWD_G4_AFTER_G3_ORDERS, ROCKE_ATTN_BWD_N_G4_AFTER_G3_ORDERS);
    else if(!eq(g->dq_mode, "atomic"))
        why = "dq_mode='atomic' (the split-dQ main kernel has no G4)";
    else if(!eq(g->barrier_merge, "off") || g->ring_depth != 1 || !eq(g->global_path, "vgpr")
            || eq(g->loop_sync, "counted"))
        why = "the synchronous body: barrier_merge='off', ring_depth=1, global_path='vgpr' and "
              "loop_sync != 'counted' (the slice reads and the moved dS barrier are built for "
              "its three barriers)";
    else if(!eq(g->sched, "none") || g->setprio)
        why = "sched='none' (the stage table does not model the dS key slices)";
    else if(eq(reader_of(g->transpose_source), "xt") && eq(g->kt_source, "lds"))
        why = "K^T from registers under xt_lds (no K-inner K^T image slices)";
    if(!why.empty())
        value_error("g4_slicing=" + repr(g->g4_slicing) + " needs " + why);
}

void require_derived(const rocke_attn_bwd_tile_geometry_t* g)
{
    if(g == NULL || !g->derived)
        value_error(
            "the tile geometry is not checked (rocke_attn_bwd_tile_geometry_check_in_place)");
}

/* ====================================================================== *
 * LDS views and copies
 * ====================================================================== */

/* _pad_elems(desc, elem_bytes) */
int pad_elems(const char* desc, int elem_bytes)
{
    const char* colon = strchr(desc, ':');
    std::string sw = colon ? std::string(colon + 1) : std::string("none");
    if(sw.compare(0, 3, "pad") == 0)
    {
        std::string num = sw.substr(3);
        char* end = NULL;
        long v = strtol(num.c_str(), &end, 10);
        if(num.empty() || *end != '\0')
            value_error("invalid literal for int() with base 10: " + repr(num.c_str()));
        return (int)(v / elem_bytes);
    }
    if(sw == "none" || sw == "f32")
        return 0;
    value_error("LDS buffer layout " + repr(desc) + " has no view in this body");
}

bool starts(const char* s, const char* p)
{
    return strncmp(s, p, strlen(p)) == 0;
}

const rocke_attn_bwd_lds_buffer_t*
    plan_buffer(const rocke_attn_bwd_lds_plan_t* plan, const char* name, int slot)
{
    rocke_attn_bwd_err_t e;
    const rocke_attn_bwd_lds_buffer_t* buf = rocke_attn_bwd_lds_plan_buffer(plan, name, slot, &e);
    if(buf == NULL)
        ckc::raise_status(e.code, e.msg);
    return buf;
}

int plan_offset(const rocke_attn_bwd_lds_plan_t* plan, const char* name, int slot)
{
    rocke_attn_bwd_err_t e;
    int off = rocke_attn_bwd_lds_plan_offset(plan, name, slot, &e);
    if(off < 0)
        ckc::raise_status(e.code, e.msg);
    return off;
}

rocke_attn_bwd_lds_view_t
    lds_view(B* b, const rocke_attn_bwd_bwd_lds_t* lds, const char* name, int cols, int slot)
{
    const rocke_attn_bwd_lds_buffer_t* buf = plan_buffer(lds->plan, name, slot);
    if(buf->offset % 16)
        value_error("LDS buffer " + s_(name) + " is not 16-byte aligned");
    rocke_attn_bwd_lds_view_t v;
    int64_t pitch, rows;
    int scale;
    const char* layout = buf->layout;
    if(strcmp(layout, "rowmajor:f32") == 0 || strcmp(layout, "vector:f32") == 0)
    {
        pitch = cols;
        scale = 2;
        if(cols == 0)
            value_error("integer division or modulo by zero");
        rows = buf->nbytes / (4 * cols);
    }
    else if(strcmp(layout, "rowmajor:xor") == 0 || strcmp(layout, "xt:xor") == 0)
    {
        rocke_attn_bwd_err_t e;
        const rocke_attn_bwd_lds_layout_t* lay
            = rocke_attn_bwd_lds_plan_layout(lds->plan, name, &e);
        if(lay == NULL)
            ckc::raise_status(e.code, e.msg);
        int lrows, lcols, leb, lchunk;
        if(lay->kind == ROCKE_ATTN_BWD_LAYOUT_XT)
        {
            lrows = lay->xt.rows;
            lcols = lay->xt.cols;
            leb = lay->xt.elem_bytes;
            lchunk = lay->xt.chunk_bytes;
        }
        else if(lay->kind == ROCKE_ATTN_BWD_LAYOUT_ROW_XOR)
        {
            lrows = lay->row_xor.rows;
            lcols = lay->row_xor.cols;
            leb = lay->row_xor.elem_bytes;
            lchunk = lay->row_xor.chunk_bytes;
        }
        else
            value_error("LDS buffer " + s_(name) + " has no XOR layout object");
        if(lcols != cols || leb != 2)
            value_error("LDS buffer " + s_(name) + ": XOR layout " + std::to_string(lrows) + "x"
                        + std::to_string(lcols) + " viewed with " + std::to_string(cols)
                        + " columns");
        need(b,
             rocke_attn_bwd_xor_lds_view_init(
                 &v, b, lds->smem, buf->offset / 2, cols, lrows, 1, NULL, lchunk / 2));
        return v;
    }
    else if(starts(layout, "rowmajor:") || starts(layout, "xt:"))
    {
        pitch = cols + pad_elems(layout, 2);
        scale = 1;
        if(pitch == 0)
            value_error("integer division or modulo by zero");
        rows = buf->nbytes / (2 * pitch);
    }
    else if(strcmp(layout, "dma_slab_xor") == 0)
    {
        if(cols == 0)
            value_error("integer division or modulo by zero");
        need(b,
             rocke_attn_bwd_dma_slab_view_init(
                 &v, b, lds->smem, buf->offset / 2, cols, buf->nbytes / (2 * cols), 1, NULL));
        return v;
    }
    else
        value_error("LDS buffer " + s_(name) + " layout " + repr(layout) + " has no view");
    need(b,
         rocke_attn_bwd_lds_view_init(&v, b, lds->smem, buf->offset / 2, pitch, rows, scale, NULL));
    return v;
}

/* tile loaders */

V slab_load_fn(void* ctx, V r, V c)
{
    auto* L = (rocke_attn_bwd_slab_row_loader_t*)ctx;
    /* py: slab.load_vec(b.add(row0, r), c, vec) */
    V row = rocke_b_add(L->b, L->row0, r);
    return needv(L->b, rocke_attn_bwd_row_slab_load_vec(&L->slab, row, ivv(c), L->vec));
}

V packed_load_fn(void* ctx, V r, V c)
{
    auto* L = (rocke_attn_bwd_packed_row_loader_t*)ctx;
    B* b = L->b;
    /* py: h, q, _ok = packed.locate(b, r)
     *     return addr.slab(h, clamp=clamp).load_vec(q, c, vec) */
    V h, q, ok;
    rocke_attn_bwd_packed_rows_locate(&L->packed, b, ivv(r), &h, &q, &ok);
    need(b, true);
    rocke_attn_bwd_row_slab_t slab
        = rocke_attn_bwd_strided_tensor_addr_slab(L->addr, ivv(h), ivi(0), L->clamp);
    need(b, true);
    return needv(b, rocke_attn_bwd_row_slab_load_vec(&slab, q, ivv(c), L->vec));
}

/* slab_row_loader over a duck-typed slab (rocke_attn_bwd_row_source_t). */
struct SourceRowLoader
{
    B* b;
    const rocke_attn_bwd_row_source_t* src;
    V row0;
    int vec;
};

V source_load_fn(void* ctx, V r, V c)
{
    auto* L = (SourceRowLoader*)ctx;
    /* py: slab.load_vec(b.add(row0, r), c, vec) */
    V row = rocke_b_add(L->b, L->row0, r);
    return needv(L->b, L->src->load_vec(L->src->ctx, L->b, row, ivv(c), L->vec));
}

V call_load(B* b, const rocke_attn_bwd_tile_loader_t& l, V r, V c)
{
    return needv(b, l.fn(l.ctx, r, c));
}

/* _copy_rows(b, load, view, rows=, cols=, tid=, threads=, vec=) */
void copy_rows(B* b,
               const rocke_attn_bwd_tile_loader_t& load,
               const rocke_attn_bwd_lds_view_t* view,
               int rows,
               int cols,
               V tid,
               int threads,
               int vec)
{
    if(vec == 0)
        value_error("integer division or modulo by zero");
    int per_row = cols / vec;
    int total = rows * per_row;
    int n = rocke_attn_bwd_deal_chunks_count(b, total, threads);
    need(b, n >= 0);
    for(int it = 0; it < n; ++it)
    {
        bool tail = false;
        V chunk = needv(b, rocke_attn_bwd_deal_chunks_round(b, tid, total, threads, it, &tail));
        auto body = [&]() {
            V r, c;
            rocke_attn_bwd_chunk_coords(b, chunk, per_row, vec, &r, &c);
            need(b, true);
            V val = call_load(b, load, r, c);
            vstore(b, view, ivv(r), ivv(c), val, vec);
        };
        if(!tail)
            body();
        else
        {
            V t = ci(b, total);
            V cond = rocke_b_cmp_lt(b, chunk, t);
            IfScope s(b, cond);
            body();
        }
    }
}

/* One pending LDS store of _row_loads / _ring_loads. */
struct PendingStore
{
    const rocke_attn_bwd_lds_view_t* view;
    V r;
    IV c;
    V val;
    int n;
    V guard; /* NULL: None */
};

/* _row_loads(b, sets, rows=, cols=, tid=, threads=, vec=8) */
std::vector<PendingStore> row_loads(B* b,
                                    const rocke_attn_bwd_row_set_t* sets,
                                    int n_sets,
                                    int rows,
                                    int cols,
                                    V tid,
                                    int threads,
                                    int vec)
{
    int per_row = cols / vec;
    int total = rows * per_row;
    std::vector<PendingStore> pending;
    for(int i = 0; i < n_sets; ++i)
    {
        int n = rocke_attn_bwd_deal_chunks_count(b, total, threads);
        need(b, n >= 0);
        for(int it = 0; it < n; ++it)
        {
            bool tail = false;
            V chunk = needv(b, rocke_attn_bwd_deal_chunks_round(b, tid, total, threads, it, &tail));
            V src = chunk;
            if(tail)
            {
                V t = ci(b, total - 1);
                src = rocke_b_smin(b, chunk, t);
            }
            V r, c;
            rocke_attn_bwd_chunk_coords(b, src, per_row, vec, &r, &c);
            need(b, true);
            V guard = NULL;
            if(tail)
            {
                V t = ci(b, total);
                guard = rocke_b_cmp_lt(b, chunk, t);
            }
            V val = call_load(b, sets[i].load, r, c);
            pending.push_back(PendingStore{sets[i].view, r, ivv(c), val, vec, guard});
        }
    }
    return pending;
}

void copy_row_sets(B* b,
                   const rocke_attn_bwd_row_set_t* sets,
                   int n_sets,
                   const char* order,
                   int rows,
                   int cols,
                   V tid,
                   int threads,
                   V wave,
                   int wave_size,
                   int vec)
{
    if(eq(order, "split"))
    {
        for(int i = 0; i < n_sets; ++i)
            copy_rows(b, sets[i].load, sets[i].view, rows, cols, tid, threads, vec);
        return;
    }
    if(vec == 0)
        value_error("integer division or modulo by zero");
    int per_row = cols / vec;
    int total = rows * per_row;
    if(eq(order, "loads_first"))
    {
        struct P
        {
            const rocke_attn_bwd_lds_view_t* view;
            V r, c, val, guard;
        };
        std::vector<P> pending;
        for(int i = 0; i < n_sets; ++i)
        {
            int n = rocke_attn_bwd_deal_chunks_count(b, total, threads);
            need(b, n >= 0);
            for(int it = 0; it < n; ++it)
            {
                bool tail = false;
                V chunk
                    = needv(b, rocke_attn_bwd_deal_chunks_round(b, tid, total, threads, it, &tail));
                V src = chunk;
                if(tail)
                {
                    V t = ci(b, total - 1);
                    src = rocke_b_smin(b, chunk, t);
                }
                V r, c;
                rocke_attn_bwd_chunk_coords(b, src, per_row, vec, &r, &c);
                need(b, true);
                V val = call_load(b, sets[i].load, r, c);
                pending.push_back(P{sets[i].view, r, c, val, tail ? chunk : NULL});
            }
        }
        for(const P& p : pending)
        {
            if(p.guard == NULL)
                vstore(b, p.view, ivv(p.r), ivv(p.c), p.val, vec);
            else
            {
                V t = ci(b, total);
                V cond = rocke_b_cmp_lt(b, p.guard, t);
                IfScope s(b, cond);
                vstore(b, p.view, ivv(p.r), ivv(p.c), p.val, vec);
            }
        }
        return;
    }
    if(!eq(order, "joint"))
        value_error("copy order " + repr(order) + " not in "
                    + tuple_repr(ROCKE_ATTN_BWD_COPY_ORDERS, ROCKE_ATTN_BWD_N_COPY_ORDERS));
    if(wave == NULL || !wave_size || total % wave_size || n_sets * total > threads)
        value_error("joint copy needs whole waves per tile: " + std::to_string(n_sets) + " x "
                    + std::to_string(total) + " chunks, " + std::to_string(threads)
                    + " threads, wave size " + (wave_size ? std::to_string(wave_size) : "None"));
    int per_tile = total / wave_size;
    for(int i = 0; i < n_sets; ++i)
    {
        V hi_c = ci(b, (int64_t)(i + 1) * per_tile);
        V hi = rocke_b_cmp_lt(b, wave, hi_c);
        V cond = hi;
        if(i != 0)
        {
            V lo_c = ci(b, (int64_t)i * per_tile);
            V lo = rocke_b_cmp_ge(b, wave, lo_c);
            cond = rocke_b_land(b, lo, hi);
        }
        IfScope s(b, cond);
        V chunk = tid;
        if(i != 0)
        {
            V off = ci(b, (int64_t)i * total);
            chunk = rocke_b_sub(b, tid, off);
        }
        V r, c;
        rocke_attn_bwd_chunk_coords(b, chunk, per_row, vec, &r, &c);
        need(b, true);
        V val = call_load(b, sets[i].load, r, c);
        vstore(b, sets[i].view, ivv(r), ivv(c), val, vec);
    }
}

void write_xt(B* b,
              const rocke_attn_bwd_lds_view_t* src,
              const rocke_attn_bwd_lds_view_t* dst,
              int seq,
              int depth,
              V tid,
              int threads,
              const rocke_type_t* elem)
{
    rocke_attn_bwd_xt_writer_t w;
    rocke_attn_bwd_err_t e;
    need_st(rocke_attn_bwd_default_xt_writer(&w, seq, depth, threads, 2, 16, &e), e);
    int dl = rocke_attn_bwd_xt_writer_d_lanes_eff(&w);
    int n_sg = rocke_attn_bwd_xt_writer_n_sg(&w);
    int n_dg = rocke_attn_bwd_xt_writer_n_dg(&w);
    int kpt = w.k_per_thread, vec = w.vec_d;
    int units = rocke_attn_bwd_xt_writer_units(&w);
    int iters = rocke_attn_bwd_xt_writer_iterations(&w);
    for(int it = 0; it < iters; ++it)
    {
        V u = tid;
        if(it != 0)
        {
            V off = ci(b, (int64_t)it * threads);
            u = rocke_b_add(b, tid, off);
        }
        auto body = [&]() {
            V c_dl = ci(b, dl);
            V rest = rocke_b_div(b, u, c_dl);
            V sg = rest;
            if(n_dg > dl)
            {
                V c_sg = ci(b, n_sg);
                sg = rocke_b_mod(b, rest, c_sg);
            }
            V c_dl2 = ci(b, dl);
            V dg = rocke_b_mod(b, u, c_dl2);
            if(n_dg > dl)
            {
                V c_sg = ci(b, n_sg);
                V q = rocke_b_div(b, rest, c_sg);
                V c_dl3 = ci(b, dl);
                V m = rocke_b_mul(b, q, c_dl3);
                dg = rocke_b_add(b, m, dg);
            }
            V c_kpt = ci(b, kpt);
            V s0 = rocke_b_mul(b, sg, c_kpt);
            V c_vec = ci(b, vec);
            V d0 = rocke_b_mul(b, dg, c_vec);
            std::vector<V> rows;
            for(int r = 0; r < kpt; ++r)
            {
                V row = s0;
                if(r != 0)
                {
                    V cr = ci(b, r);
                    row = rocke_b_add(b, s0, cr);
                }
                rows.push_back(vload(b, src, ivv(row), ivv(d0), elem, vec));
            }
            for(int j = 0; j < vec; ++j)
            {
                std::vector<V> parts;
                for(V x : rows)
                    parts.push_back(needv(b, rocke_b_vec_extract(b, x, j)));
                V val = parts[0];
                if(kpt != 1)
                    val = needv(b, rocke_b_vec_pack(b, parts.data(), (int)parts.size(), elem));
                V row = d0;
                if(j != 0)
                {
                    V cj = ci(b, j);
                    row = rocke_b_add(b, d0, cj);
                }
                vstore(b, dst, ivv(row), ivv(s0), val, kpt);
            }
        };
        if((int64_t)(it + 1) * threads <= units)
            body();
        else
        {
            V cu = ci(b, units);
            V cond = rocke_b_cmp_lt(b, u, cu);
            IfScope s(b, cond);
            body();
        }
    }
}

/* quad_transpose4 */
std::vector<V> quad_transpose4(B* b, const std::vector<V>& vals, V lane)
{
    if(vals.size() != 4)
        value_error("quad_transpose4 takes 4 values per lane");
    std::vector<V> x;
    for(V v : vals)
        x.push_back(needv(b, rocke_b_bitcast(b, v, rocke_i32())));
    V c4 = ci(b, 4);
    V j = rocke_b_mod(b, lane, c4);
    V zero = ci(b, 0);
    struct Stage
    {
        int bit;
        int perm[4];
        int pairs[2][2];
    };
    static const Stage stages[2]
        = {{1, {1, 0, 3, 2}, {{0, 1}, {2, 3}}}, {2, {2, 3, 0, 1}, {{0, 2}, {1, 3}}}};
    for(const Stage& st : stages)
    {
        V cb = ci(b, st.bit);
        V a = rocke_b_land(b, j, cb);
        V hi = rocke_b_cmp_ne(b, a, zero);
        for(const auto& pr : st.pairs)
        {
            int lo_s = pr[0], hi_s = pr[1];
            V send = rocke_b_select(b, hi, x[lo_s], x[hi_s]);
            V recv = needv(
                b, rocke_b_quad_perm(b, send, st.perm[0], st.perm[1], st.perm[2], st.perm[3]));
            V nlo = rocke_b_select(b, hi, recv, x[lo_s]);
            V nhi = rocke_b_select(b, hi, x[hi_s], recv);
            x[lo_s] = nlo;
            x[hi_s] = nhi;
        }
    }
    std::vector<V> out;
    for(V v : x)
        out.push_back(needv(b, rocke_b_bitcast(b, v, rocke_f32())));
    return out;
}

} // namespace

namespace
{

/* ====================================================================== *
 * StatsView
 * ====================================================================== */

typedef rocke_attn_bwd_stats_view_t SV;

rocke_attn_bwd_ws_slab_t ws_slab(const rocke_attn_bwd_workspace_addr_t* w, IV h, IV row0)
{
    return rocke_attn_bwd_workspace_addr_slab(w, h, row0);
}

/* (lse2, Dsum) lists of a stats call */
struct StatPairs
{
    std::vector<V> lse2, dsum;
};

StatPairs stats_rows(const SV* s, B* b, V h, V q0, const std::vector<V>& rows)
{
    rocke_attn_bwd_ws_slab_t sl = ws_slab(&s->lse2, ivv(h), ivv(q0));
    need(b, true);
    rocke_attn_bwd_ws_slab_t sd = ws_slab(&s->dsum, ivv(h), ivv(q0));
    need(b, true);
    StatPairs out;
    if(s->bounds == ROCKE_ATTN_BWD_STATS_CLAMP)
    {
        for(V r : rows)
        {
            out.lse2.push_back(needv(b, rocke_attn_bwd_ws_slab_load(&sl, r, ivi(0), INF)));
            out.dsum.push_back(needv(b, rocke_attn_bwd_ws_slab_load(&sd, r, ivi(0), 0.0)));
        }
        return out;
    }
    const rocke_attn_bwd_ws_slab_t* slabs[2] = {&sl, &sd};
    const double neutrals[2] = {INF, 0.0};
    std::vector<V>* outs[2] = {&out.lse2, &out.dsum};
    for(int i = 0; i < 2; ++i)
    {
        const rocke_attn_bwd_ws_slab_t* slab = slabs[i];
        V span = ci(b, s->span);
        V n = rocke_b_smin(b, slab->rows_left, span);
        V c4 = ci(b, 4);
        V nb = rocke_b_mul(b, n, c4);
        V rsrc = needv(b, rocke_b_buffer_rsrc(b, slab->ptr, nb));
        V zero = ci(b, 0);
        V fill = rocke_b_const_f32(b, neutrals[i]);
        for(V r : rows)
        {
            V c4r = ci(b, 4);
            V off = rocke_b_mul(b, r, c4r);
            V x = needv(b, rocke_b_buffer_load(b, rsrc, off, zero, rocke_f32()));
            V ok = needv(b, rocke_attn_bwd_ws_slab_row_ok(slab, r));
            outs[i]->push_back(rocke_b_select(b, ok, x, fill));
        }
    }
    need(b, true);
    return out;
}

StatPairs stats_batched_rows(const SV* s, B* b, V h, V q0, const std::vector<V>& rows, int width)
{
    if(s->n_heads == NULL)
        value_error("stats_vec > 1 needs the StatsView head count");
    StatPairs out;
    out.lse2.resize(rows.size());
    out.dsum.resize(rows.size());
    need(b,
         rocke_attn_bwd_batched_stats_rows(b,
                                           &s->lse2,
                                           &s->dsum,
                                           h,
                                           ivv(q0),
                                           rows.data(),
                                           (int)rows.size(),
                                           width,
                                           ivv(s->n_heads),
                                           out.lse2.data(),
                                           out.dsum.data()));
    return out;
}

StatPairs stats_packed_rows(const SV* s,
                            B* b,
                            const rocke_attn_bwd_packed_rows_t* packed,
                            const std::vector<V>& rows)
{
    V inf = rocke_b_const_f32(b, INF);
    V zero = rocke_b_const_f32(b, 0.0);
    StatPairs out;
    for(V r : rows)
    {
        V h, q, ok;
        rocke_attn_bwd_packed_rows_locate(packed, b, ivv(r), &h, &q, &ok);
        need(b, true);
        rocke_attn_bwd_ws_slab_t sl = ws_slab(&s->lse2, ivv(h), ivi(0));
        need(b, true);
        V lse2 = needv(b, rocke_attn_bwd_ws_slab_load(&sl, q, ivi(0), INF));
        rocke_attn_bwd_ws_slab_t sd = ws_slab(&s->dsum, ivv(h), ivi(0));
        need(b, true);
        V dsum = needv(b, rocke_attn_bwd_ws_slab_load(&sd, q, ivi(0), 0.0));
        out.lse2.push_back(rocke_b_select(b, ok, lse2, inf));
        out.dsum.push_back(rocke_b_select(b, ok, dsum, zero));
    }
    need(b, true);
    return out;
}

StatPairs stats_raw(const SV* s,
                    B* b,
                    V h,
                    V q0,
                    const std::vector<V>& rows,
                    const rocke_attn_bwd_packed_rows_t* packed)
{
    StatPairs out;
    rocke_attn_bwd_ws_slab_t sl, sd;
    if(packed == NULL)
    {
        sl = ws_slab(&s->lse2, ivv(h), ivv(q0));
        need(b, true);
        sd = ws_slab(&s->dsum, ivv(h), ivv(q0));
        need(b, true);
    }
    for(V r : rows)
    {
        V q = r;
        if(packed != NULL)
        {
            V hh, ok;
            rocke_attn_bwd_packed_rows_locate(packed, b, ivv(r), &hh, &q, &ok);
            need(b, true);
            sl = ws_slab(&s->lse2, ivv(hh), ivi(0));
            need(b, true);
            sd = ws_slab(&s->dsum, ivv(hh), ivi(0));
            need(b, true);
        }
        out.lse2.push_back(needv(b, rocke_attn_bwd_ws_slab_load_raw(&sl, q, ivi(0))));
        out.dsum.push_back(needv(b, rocke_attn_bwd_ws_slab_load_raw(&sd, q, ivi(0))));
    }
    return out;
}

StatPairs stats_finish(const SV* s,
                       B* b,
                       V h,
                       V q0,
                       const std::vector<V>& rows,
                       const StatPairs& raw,
                       const rocke_attn_bwd_packed_rows_t* packed)
{
    if(raw.lse2.size() != rows.size() || raw.dsum.size() != rows.size())
        value_error("zip() argument 2 is shorter than argument 1");
    V inf = rocke_b_const_f32(b, INF);
    V zero = rocke_b_const_f32(b, 0.0);
    StatPairs out;
    rocke_attn_bwd_ws_slab_t sl, sd;
    if(packed == NULL)
    {
        sl = ws_slab(&s->lse2, ivv(h), ivv(q0));
        need(b, true);
        sd = ws_slab(&s->dsum, ivv(h), ivv(q0));
        need(b, true);
    }
    for(size_t i = 0; i < rows.size(); ++i)
    {
        V q = rows[i], ok = NULL;
        if(packed != NULL)
        {
            V hh;
            rocke_attn_bwd_packed_rows_locate(packed, b, ivv(rows[i]), &hh, &q, &ok);
            need(b, true);
            sl = ws_slab(&s->lse2, ivv(hh), ivi(0));
            need(b, true);
            sd = ws_slab(&s->dsum, ivv(hh), ivi(0));
            need(b, true);
        }
        V lse2 = needv(b, rocke_attn_bwd_ws_slab_neutral(&sl, q, raw.lse2[i], INF));
        V dsum = needv(b, rocke_attn_bwd_ws_slab_neutral(&sd, q, raw.dsum[i], 0.0));
        if(ok != NULL)
        {
            lse2 = rocke_b_select(b, ok, lse2, inf);
            dsum = rocke_b_select(b, ok, dsum, zero);
        }
        out.lse2.push_back(lse2);
        out.dsum.push_back(dsum);
    }
    need(b, true);
    return out;
}

/* ====================================================================== *
 * BwdTileStep internals
 * ====================================================================== */

typedef rocke_attn_bwd_tile_step_t Step;
typedef rocke_attn_bwd_tile_geometry_t Geom;

/* (row_of, keys, {key: (lse2, Dsum)}) of the lane's C rows; keys are
 * (fm, slot) in sorted order, so key k is fm * c_frag_len + slot. */
struct Stats
{
    std::vector<V> row_of;
    std::vector<V> lse2, dsum;
};

std::vector<V> stats_row_list(B* b, Step* t)
{
    const rocke_attn_bwd_block_gemm_t* g0 = &t->geom->g0;
    std::vector<V> row_of;
    for(int fm = 0; fm < FM(g0); ++fm)
    {
        std::vector<V> co = acc_coords(b, g0, t->lane, t->o0, fm, 0);
        for(int s = 0; s < CFL(g0); ++s)
            row_of.push_back(co[2 * s]);
    }
    return row_of;
}

void loop_barrier(B* b, Step* t)
{
    const Geom* g = t->geom;
    if(eq(g->loop_sync, "lds_only"))
    {
        rocke_b_sync_lds_only_fence(b, FENCE);
        need(b, true);
    }
    else if(eq(g->loop_sync, "counted"))
    {
        rocke_attn_bwd_counted_waits_barrier_fence(&t->waits, b, FENCE);
        need(b, true);
    }
    else
        sync(b);
}

Opnd kt_operand(B* b, Step* t)
{
    if(t->has_v_kt)
        return lds_opnd(b, &t->v_kt);
    return lds_opnd(
        b, &t->v_k, ROCKE_ATTN_BWD_K_OUTER, NULL, rocke_attn_bwd_tile_geometry_reader(t->geom));
}

Opnd k_outer_operand(B* b, Step* t, const rocke_attn_bwd_lds_view_t* view)
{
    const char* reader = rocke_attn_bwd_tile_geometry_reader(t->geom);
    return lds_opnd(b, view, ROCKE_ATTN_BWD_K_OUTER, NULL, eq(reader, "xt") ? "plain" : reader);
}

rocke_attn_bwd_frags_t
    load_all_b(B* b, const rocke_attn_bwd_block_gemm_t* g, const Opnd& src, V lane, V col0)
{
    int n = FN(g) * KS(g);
    V* out = arena<V>(b, n);
    need(b, rocke_attn_bwd_block_gemm_load_all_b(b, g, &src.lds, lane, col0, out));
    return frags_of(out, FN(g), KS(g));
}

rocke_attn_bwd_frags_t
    load_all_a(B* b, const rocke_attn_bwd_block_gemm_t* g, const Opnd& src, V lane, V row0)
{
    int n = FM(g) * KS(g);
    V* out = arena<V>(b, n);
    need(b, rocke_attn_bwd_block_gemm_load_all_a(b, g, &src.lds, lane, row0, out));
    return frags_of(out, FM(g), KS(g));
}

V* zero_acc(B* b, const rocke_attn_bwd_block_gemm_t* g)
{
    V* out = arena<V>(b, (size_t)FM(g) * FN(g));
    rocke_attn_bwd_block_gemm_zero_acc(b, g, out);
    need(b, true);
    return out;
}

/* BlockGemm.mma(b, acc, a, bop, lane=, origin=, **opts); acc NULL is None. */
V* gemm_mma(B* b,
            const rocke_attn_bwd_block_gemm_t* g,
            V* acc,
            const Opnd& a,
            const Opnd& bop,
            V lane,
            V* const origin,
            const rocke_attn_bwd_mma_opts_t& opts)
{
    V* out = arena<V>(b, (size_t)FM(g) * FN(g));
    rocke_attn_bwd_frags_t fr = frags_of(acc, FM(g), FN(g));
    rocke_attn_bwd_gemm_operand_t ga = a.g(), gb = bop.g();
    need(b,
         rocke_attn_bwd_block_gemm_mma(b, g, acc ? &fr : NULL, &ga, &gb, lane, origin, &opts, out));
    return out;
}

rocke_attn_bwd_asm_cluster_t asm_cluster_of(const char* c)
{
    if(eq(c, "k_step"))
        return ROCKE_ATTN_BWD_CLUSTER_K_STEP;
    if(eq(c, "gemm"))
        return ROCKE_ATTN_BWD_CLUSTER_GEMM;
    value_error("asm_cluster must be 'k_step' or 'gemm', got " + repr(c));
}

/* _asm_knobs on an agpr_asm route */
rocke_attn_bwd_mma_opts_t asm_opts(const Geom* g)
{
    rocke_attn_bwd_mma_opts_t o = rocke_attn_bwd_mma_opts_default();
    o.route = ROCKE_ATTN_BWD_ROUTE_AGPR_ASM;
    o.has_asm_tail_nop = g->asm_tail_nop != NONE;
    o.asm_tail_nop = g->asm_tail_nop != NONE ? g->asm_tail_nop : 0;
    o.asm_cluster = asm_cluster_of(g->asm_cluster);
    return o;
}

bool g13_asm(const Geom* g)
{
    return eq(g->agpr_route, "asm_g13");
}

bool g02_asm(const Geom* g)
{
    return eq(g->agpr_route, "asm_g13") && g->s_in_agpr && !g->acc_in_lds;
}

int g02_agpr_base(const Geom* g, int which)
{
    const rocke_attn_bwd_block_gemm_t* g0 = &g->g0;
    int r = FM(g0) * FN(g0) * CFL(g0);
    if(g->agpr_alloc.kind == ROCKE_ATTN_BWD_AGPR_NONE)
        value_error("'NoneType' object is not subscriptable");
    if(g->agpr_alloc.kind == ROCKE_ATTN_BWD_AGPR_STR)
        value_error("invalid literal for int() with base 10: "
                    + repr(std::string(1, g->agpr_alloc.str ? g->agpr_alloc.str[0] : '?').c_str()));
    int n = g->agpr_alloc.a;
    return n - (2 - which) * r;
}

/* _acc_lds_coords(b, fm, fn) */
std::vector<V> acc_lds_coords(B* b, Step* t, int fm, int fn)
{
    return acc_coords(b, &t->geom->g1, t->lane, t->o1, fm, fn);
}

V load_lds_acc(B* b, Step* t, const rocke_attn_bwd_lds_view_t* view, int fm, int fn)
{
    std::vector<V> co = acc_lds_coords(b, t, fm, fn);
    std::vector<V> vals;
    for(size_t s = 0; s < co.size() / 2; ++s)
    {
        V x = vload(b, view, ivv(co[2 * s]), ivv(co[2 * s + 1]), rocke_f32(), 1);
        vals.push_back(needv(b, rocke_b_vec_extract(b, x, 0)));
    }
    return needv(b, rocke_b_vec_pack(b, vals.data(), (int)vals.size(), rocke_f32()));
}

void store_lds_acc(B* b, Step* t, const rocke_attn_bwd_lds_view_t* view, V frag, int fm, int fn)
{
    std::vector<V> co = acc_lds_coords(b, t, fm, fn);
    for(size_t s = 0; s < co.size() / 2; ++s)
    {
        V x = needv(b, rocke_b_vec_extract(b, frag, (int)s));
        vstore(b, view, ivv(co[2 * s]), ivv(co[2 * s + 1]), x, 1);
    }
}

void update_lds_acc(
    B* b, Step* t, const rocke_attn_bwd_lds_view_t* view, const Opnd& a, const Opnd& bop)
{
    const rocke_attn_bwd_block_gemm_t* g1 = &t->geom->g1;
    rocke_attn_bwd_gemm_operand_t ga = a.g(), gb = bop.g();
    for(int fm = 0; fm < FM(g1); ++fm)
        for(int fn = 0; fn < FN(g1); ++fn)
        {
            V frag = load_lds_acc(b, t, view, fm, fn);
            frag = needv(
                b,
                rocke_attn_bwd_block_gemm_mma_frag(b, g1, frag, &ga, &gb, fm, fn, t->lane, t->o1));
            store_lds_acc(b, t, view, frag, fm, fn);
        }
}

/* ----- counted waits helpers ----- */

std::vector<rocke_attn_bwd_tile_chunk_t> qdo_chunks(B* b, Step* t)
{
    const Geom* g = t->geom;
    int threads = threads_of(g);
    int n = rocke_attn_bwd_tile_chunks_count(b, g->k_m0, g->head_size, threads, 8);
    need(b, n >= 0);
    std::vector<rocke_attn_bwd_tile_chunk_t> out(n ? n : 1);
    need(b,
         rocke_attn_bwd_tile_chunks(b, g->k_m0, g->head_size, t->tid, threads, 8, out.data(), n));
    out.resize(n);
    return out;
}

void view_store_fn(void* ctx, B* b, V row, V col, V value, int vec)
{
    vstore(b, (const rocke_attn_bwd_lds_view_t*)ctx, ivv(row), ivv(col), value, vec);
}

rocke_attn_bwd_staged_step_t counted_loads(B* b,
                                           Step* t,
                                           V h,
                                           V q0,
                                           const rocke_attn_bwd_row_slab_t* q_slab,
                                           const rocke_attn_bwd_row_slab_t* do_slab,
                                           const rocke_attn_bwd_packed_step_t* packed,
                                           bool edge)
{
    const Geom* g = t->geom;
    std::vector<V> row_of = stats_row_list(b, t);
    StatPairs raw = stats_raw(t->stats, b, h, q0, row_of, packed ? &packed->rows : NULL);
    int n_stats = (int)row_of.size() * 2;
    V* stats = arena<V>(b, n_stats);
    for(size_t i = 0; i < row_of.size(); ++i)
    {
        stats[2 * i] = raw.lse2[i];
        stats[2 * i + 1] = raw.dsum[i];
    }
    rocke_attn_bwd_slab_row_loader_t sq, sdo;
    rocke_attn_bwd_packed_row_loader_t pq, pdo;
    rocke_attn_bwd_tile_loader_t loads[2];
    if(packed == NULL)
    {
        if(q_slab == NULL || do_slab == NULL)
            value_error("'NoneType' object has no attribute 'load_vec'");
        sq = {b, *q_slab, q0, 8};
        sdo = {b, *do_slab, q0, 8};
        loads[0] = rocke_attn_bwd_slab_row_loader(&sq);
        loads[1] = rocke_attn_bwd_slab_row_loader(&sdo);
    }
    else
    {
        bool clamp = edge || g->interior_clamp;
        pq = {b, packed->q_addr, packed->rows, 8, clamp};
        pdo = {b, packed->do_addr, packed->rows, 8, clamp};
        loads[0] = rocke_attn_bwd_packed_row_loader(&pq);
        loads[1] = rocke_attn_bwd_packed_row_loader(&pdo);
    }
    std::vector<rocke_attn_bwd_tile_chunk_t> chunks = qdo_chunks(b, t);
    int n_qdo = 2 * (int)chunks.size();
    V* qdo = arena<V>(b, n_qdo);
    need(b,
         rocke_attn_bwd_issue_tile_loads(
             b, loads, 2, chunks.data(), (int)chunks.size(), qdo, n_qdo));
    rocke_attn_bwd_staged_step_t out;
    out.stats = stats;
    out.n_stats = n_stats;
    out.qdo = qdo;
    out.n_qdo = n_qdo;
    return out;
}

Stats store_staged(B* b,
                   Step* t,
                   const rocke_attn_bwd_staged_step_t* staged,
                   V h,
                   V q0,
                   const rocke_attn_bwd_packed_step_t* packed)
{
    rocke_attn_bwd_tile_view_t views[2]
        = {{view_store_fn, (void*)&t->v_q}, {view_store_fn, (void*)&t->v_do}};
    std::vector<rocke_attn_bwd_tile_chunk_t> chunks = qdo_chunks(b, t);
    need(b,
         rocke_attn_bwd_store_tile(
             b, views, 2, chunks.data(), (int)chunks.size(), staged->qdo, staged->n_qdo, 8));
    Stats st;
    st.row_of = stats_row_list(b, t);
    if(staged->n_stats != 2 * (int)st.row_of.size())
        value_error("staged stats do not match the stats rows");
    StatPairs raw;
    for(size_t i = 0; i < st.row_of.size(); ++i)
    {
        raw.lse2.push_back(staged->stats[2 * i]);
        raw.dsum.push_back(staged->stats[2 * i + 1]);
    }
    StatPairs fin = stats_finish(t->stats, b, h, q0, st.row_of, raw, packed ? &packed->rows : NULL);
    st.lse2 = fin.lse2;
    st.dsum = fin.dsum;
    return st;
}

rocke_attn_bwd_staged_step_t issue_next(B* b, Step* t, rocke_attn_bwd_next_loads_fn fn, void* ctx)
{
    rocke_b_sched_barrier(b, 0);
    rocke_attn_bwd_staged_step_t nxt;
    memset(&nxt, 0, sizeof(nxt));
    need(b, fn(ctx, b, &nxt));
    rocke_b_sched_barrier(b, 0);
    need(b, true);
    rocke_attn_bwd_counted_waits_loads(&t->waits);
    return nxt;
}

/* ----- stats ----- */

/* _load_stats(b, h, q0, rows, source): src NULL reads the workspace. */
Stats load_stats(B* b,
                 Step* t,
                 V h,
                 V q0,
                 const rocke_attn_bwd_packed_rows_t* rows,
                 const rocke_attn_bwd_lds_stats_view_t* src = NULL)
{
    Stats st;
    st.row_of = stats_row_list(b, t);
    StatPairs p;
    if(rows == NULL && t->geom->stats_vec != 1 && src == NULL)
        p = stats_batched_rows(t->stats, b, h, q0, st.row_of, t->geom->stats_vec);
    else if(src != NULL)
    {
        p.lse2.resize(st.row_of.size());
        p.dsum.resize(st.row_of.size());
        int n = (int)st.row_of.size();
        if(rows == NULL)
            need(b,
                 rocke_attn_bwd_lds_stats_view_rows(
                     src, b, st.row_of.data(), n, p.lse2.data(), p.dsum.data()));
        else
            need(b,
                 rocke_attn_bwd_lds_stats_view_packed_rows(
                     src, b, st.row_of.data(), n, p.lse2.data(), p.dsum.data()));
    }
    else if(rows == NULL)
        p = stats_rows(t->stats, b, h, q0, st.row_of);
    else
        p = stats_packed_rows(t->stats, b, rows, st.row_of);
    st.lse2 = p.lse2;
    st.dsum = p.dsum;
    return st;
}

/* ----- barrier merge: the ring slots ----- */

struct RingViews
{
    rocke_attn_bwd_lds_view_t q, dO, ds, lse2, dsum;
    bool has_ds = false, has_stats = false;
};

/* _ring_views(b, slot): slot NULL is the int 0. */
RingViews ring_views(B* b, Step* t, V slot)
{
    const rocke_attn_bwd_lds_plan_t* plan = t->lds->plan;
    RingViews out;
    struct Item
    {
        const char* name;
        bool has;
        const rocke_attn_bwd_lds_view_t* view;
        rocke_attn_bwd_lds_view_t* dst;
    };
    Item items[5] = {{"q", true, &t->v_q, &out.q},
                     {"do", true, &t->v_do, &out.dO},
                     {"ds", t->has_v_ds, &t->v_ds, &out.ds},
                     {"lse2", t->has_v_ring_stats, &t->v_lse2, &out.lse2},
                     {"dsum", t->has_v_ring_stats, &t->v_dsum, &out.dsum}};
    for(const Item& it : items)
    {
        if(slot == NULL)
        {
            /* py: out[name] = view (None for an absent buffer) */
            if(it.has)
                *it.dst = *it.view;
            continue;
        }
        if(!it.has)
            value_error("replace() should be called on dataclass instances");
        int delta = plan_offset(plan, it.name, 1) - plan_offset(plan, it.name, 0);
        if(delta % 16)
            value_error("LDS ring buffer " + std::string(it.name) + " slots are not 16-byte apart");
        V c = ci(b, delta / 2);
        V shift = rocke_b_mul(b, slot, c);
        need(b, true);
        *it.dst = *it.view;
        it.dst->shift = shift;
    }
    out.has_ds = t->has_v_ds;
    out.has_stats = t->has_v_ring_stats;
    return out;
}

std::vector<PendingStore>
    ring_loads(B* b, Step* t, const rocke_attn_bwd_q_step_inputs_t* step, const RingViews& views)
{
    const Geom* g = t->geom;
    int m0 = g->k_m0;
    rocke_attn_bwd_slab_row_loader_t sq, sdo;
    rocke_attn_bwd_packed_row_loader_t pq, pdo;
    rocke_attn_bwd_row_set_t sets[2];
    if(step->packed == NULL)
    {
        if(!step->has_slabs)
            value_error("'NoneType' object has no attribute 'load_vec'");
        sq = {b, step->q_slab, step->q0, 8};
        sdo = {b, step->do_slab, step->q0, 8};
        sets[0].load = rocke_attn_bwd_slab_row_loader(&sq);
        sets[1].load = rocke_attn_bwd_slab_row_loader(&sdo);
    }
    else
    {
        pq = {b, step->packed->q_addr, step->packed->rows, 8, true};
        pdo = {b, step->packed->do_addr, step->packed->rows, 8, true};
        sets[0].load = rocke_attn_bwd_packed_row_loader(&pq);
        sets[1].load = rocke_attn_bwd_packed_row_loader(&pdo);
    }
    if(!views.has_stats)
        value_error("'NoneType' object has no attribute 'store'");
    sets[0].view = &views.q;
    sets[1].view = &views.dO;
    std::vector<PendingStore> pending
        = row_loads(b, sets, 2, m0, g->head_size, t->tid, threads_of(g), 8);
    V cm = ci(b, m0 - 1);
    V row = rocke_b_smin(b, t->tid, cm);
    need(b, true);
    std::vector<V> rows(1, row);
    StatPairs p = step->packed == NULL ? stats_rows(t->stats, b, step->h, step->q0, rows)
                                       : stats_packed_rows(t->stats, b, &step->packed->rows, rows);
    V cm0 = ci(b, m0);
    V guard = rocke_b_cmp_lt(b, t->tid, cm0);
    need(b, true);
    pending.push_back(PendingStore{&views.lse2, row, ivi(0), p.lse2[0], 1, guard});
    pending.push_back(PendingStore{&views.dsum, row, ivi(0), p.dsum[0], 1, guard});
    return pending;
}

void ring_stores(B* b, const std::vector<PendingStore>& pending)
{
    for(const PendingStore& p : pending)
    {
        if(p.guard == NULL)
            vstore(b, p.view, ivv(p.r), p.c, p.val, p.n);
        else
        {
            IfScope s(b, p.guard);
            vstore(b, p.view, ivv(p.r), p.c, p.val, p.n);
        }
    }
}

Stats ring_stats(B* b, Step* t, const RingViews& views)
{
    if(!views.has_stats)
        value_error("'NoneType' object has no attribute 'load'");
    Stats st;
    st.row_of = stats_row_list(b, t);
    for(V r : st.row_of)
    {
        V a = vload(b, &views.lse2, ivv(r), ivi(0), rocke_f32(), 1);
        V la = needv(b, rocke_b_vec_extract(b, a, 0));
        V d = vload(b, &views.dsum, ivv(r), ivi(0), rocke_f32(), 1);
        V ld = needv(b, rocke_b_vec_extract(b, d, 0));
        st.lse2.push_back(la);
        st.dsum.push_back(ld);
    }
    return st;
}

/* ----- the softmax gradient ----- */

/* _rounding_point / _agpr_fence / _vgpr_fence: v through an empty asm. */
V empty_asm(B* b, V v, const char* constraints, const rocke_type_t* type)
{
    rocke_inline_asm_opts_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.sideeffect = false;
    opts.sideeffect_set = true;
    rocke_op_t* op = rocke_b_inline_asm(b, "", constraints, &v, 1, &type, 1, &opts);
    if(op == NULL || op->num_results < 1)
        raise_builder(b);
    return op->results[0];
}

void softmax_grad(B* b,
                  Step* t,
                  V* s_acc,
                  V* dp_acc,
                  V q0,
                  V k0,
                  bool edge,
                  const rocke_attn_bwd_packed_rows_t* rows,
                  Stats* stats_in,
                  V h,
                  const rocke_attn_bwd_lds_stats_view_t* stats_src,
                  V* p_out,
                  V* ds_out)
{
    const Geom* g = t->geom;
    const rocke_attn_bwd_block_gemm_t* g0 = &g->g0;
    V lane = t->lane;
    V zero = rocke_b_const_f32(b, 0.0);
    V inf = rocke_b_const_f32(b, INF);
    Stats st = stats_in != NULL ? *stats_in : load_stats(b, t, h, q0, rows, stats_src);
    size_t nk = st.row_of.size();
    std::vector<V> live_(nk, (V)NULL);
    if(t->sentinel)
        for(size_t k = 0; k < nk; ++k)
            live_[k] = needv(b, rocke_b_fcmp(b, "olt", st.lse2[k], inf));
    std::vector<V> lim_lo(nk), lim_hi(nk);
    if(edge)
        for(size_t k = 0; k < nk; ++k)
        {
            V q, ok = NULL;
            if(rows == NULL)
                q = rocke_b_add(b, q0, st.row_of[k]);
            else
            {
                V hh;
                rocke_attn_bwd_packed_rows_locate(rows, b, ivv(st.row_of[k]), &hh, &q, &ok);
            }
            need(b, true);
            if(t->sentinel)
                ok = ok == NULL ? live_[k] : rocke_b_land(b, ok, live_[k]);
            rocke_status_t st
                = rocke_attn_bwd_mask_row_limits(t->mask, b, q, ok, &lim_lo[k], &lim_hi[k]);
            need(b, st == ROCKE_OK);
        }
    std::vector<V> neg;
    if(g->stats_fma)
    {
        neg.resize(nk);
        rocke_attn_bwd_stats_rows_neg(b, st.lse2.data(), (int)nk, neg.data());
        need(b, true);
    }
    int cfl = CFL(g0), fn_n = FN(g0);
    for(int fm = 0; fm < FM(g0); ++fm)
        for(int fn = 0; fn < fn_n; ++fn)
        {
            std::vector<V> co = acc_coords(b, g0, lane, t->o0, fm, fn);
            std::vector<V> ps, dss;
            for(int s = 0; s < cfl; ++s)
            {
                size_t key = (size_t)fm * cfl + s;
                V c = co[2 * s + 1];
                V lse2 = st.lse2[key], dsum = st.dsum[key];
                V sv = needv(b, rocke_b_vec_extract(b, s_acc[fm * fn_n + fn], s));
                V dpv = needv(b, rocke_b_vec_extract(b, dp_acc[fm * fn_n + fn], s));
                V p = needv(
                    b,
                    rocke_attn_bwd_softmax_p(
                        b, sv, t->scale_log2, lse2, g->stats_fma ? neg[key] : NULL, g->bwd_exp2));
                V cond;
                if(edge)
                {
                    V kk = rocke_b_add(b, k0, c);
                    cond = needv(b, rocke_attn_bwd_mask_keep_in(b, lim_lo[key], lim_hi[key], kk));
                    p = rocke_b_select(b, cond, p, zero);
                }
                else
                    cond = live_[key];
                V diff = rocke_b_fsub(b, dpv, dsum);
                V ds = rocke_b_fmul(b, p, diff);
                if(cond != NULL)
                    ds = rocke_b_select(b, cond, ds, zero);
                ps.push_back(p);
                if(t->ds_mult != NULL)
                    ds = rocke_b_fmul(b, t->ds_mult, ds);
                need(b, true);
                dss.push_back(empty_asm(b, ds, "=v,0", rocke_f32()));
            }
            p_out[fm * fn_n + fn] = needv(b, rocke_b_vec_pack(b, ps.data(), cfl, rocke_f32()));
            ds_out[fm * fn_n + fn] = needv(b, rocke_b_vec_pack(b, dss.data(), cfl, rocke_f32()));
        }
}

void store_c_tiles(B* b, Step* t, const rocke_attn_bwd_lds_view_t* view, V* frags)
{
    const rocke_attn_bwd_block_gemm_t* g0 = &t->geom->g0;
    for(int fm = 0; fm < FM(g0); ++fm)
        for(int fn = 0; fn < FN(g0); ++fn)
        {
            V half = needv(b,
                           rocke_attn_bwd_cast_c_frag(
                               b, frags[fm * FN(g0) + fn], t->elem, t->geom->bf16_cast));
            std::vector<V> co = acc_coords(b, g0, t->lane, t->o0, fm, fn);
            for(int s = 0; s < CFL(g0); ++s)
            {
                V x = needv(b, rocke_b_vec_extract(b, half, s));
                vstore(b, view, ivv(co[2 * s]), ivv(co[2 * s + 1]), x, 1);
            }
        }
}

void write_c_tiles(B* b, Step* t, const rocke_attn_bwd_lds_view_t* view, V* frags)
{
    if(t->g0_replicated)
    {
        V cw = ci(b, rocke_attn_bwd_block_gemm_waves(&t->geom->g0));
        V first = rocke_b_cmp_lt(b, t->wave, cw);
        IfScope s(b, first);
        store_c_tiles(b, t, view, frags);
        return;
    }
    store_c_tiles(b, t, view, frags);
}

/* ----- dQ ----- */

int dq_atomics_per_lane(Step* t)
{
    int n = 0;
    rocke_attn_bwd_err_t e;
    need_st(rocke_attn_bwd_stage_table_dq_atomics_per_lane(&t->sched, &n, &e), e);
    return n;
}

void flush_dq(B* b, Step* t, const rocke_attn_bwd_pending_dq_t* pending)
{
    const rocke_attn_bwd_block_gemm_t* g4 = &t->geom->g4;
    if(pending->n_values != dq_atomics_per_lane(t))
        value_error("the pending dQ partial does not match the G4 tile");
    if(t->o4[0] == NULL)
        value_error("'NoneType' object is not subscriptable");
    int next = 0;
    std::vector<rocke_attn_bwd_dq_slot_t> slots;
    for(int fm = 0; fm < FM(g4); ++fm)
        for(int fn = 0; fn < FN(g4); ++fn)
        {
            std::vector<V> co = acc_coords(b, g4, t->lane, t->o4, fm, fn);
            for(size_t s = 0; s < co.size() / 2; ++s)
            {
                if(next >= pending->n_values)
                    value_error("StopIteration");
                rocke_attn_bwd_dq_slot_t slot;
                slot.row = ivv(co[2 * s]);
                slot.col = ivv(co[2 * s + 1]);
                slot.value = pending->values[next++];
                slots.push_back(slot);
            }
        }
    rocke_attn_bwd_packed_rows_t pr;
    const rocke_attn_bwd_packed_rows_t* packed = NULL;
    V rows = NULL;
    if(pending->n_heads == NULL)
        rows = pending->q0;
    else
    {
        pr.head0 = pending->h;
        pr.n_heads = pending->n_heads;
        pr.rows_per_head = pending->rows_per_head;
        pr.row0 = pending->q0;
        packed = &pr;
    }
    need(b, rocke_attn_bwd_dq_sink_begin_head(t->sink, b, pending->h));
    need(b,
         rocke_attn_bwd_dq_sink_consume(
             t->sink, b, slots.data(), (int)slots.size(), rows, packed, pending->h, true));
    if(t->has_waits)
        rocke_attn_bwd_counted_waits_atomics(&t->waits, (int)slots.size());
}

bool g4_prefetch_before_g3(Step* t)
{
    return rocke_attn_bwd_tile_geometry_g4_prefetch_before_g3(t->geom);
}

/* _g4_operands: (LdsOperand(v_ds), K^T operand) */
Opnd kt_b_default(B* b, Step* t)
{
    if(t->kt_regs.v != NULL)
        return regs_opnd(t->kt_regs);
    return kt_operand(b, t);
}

rocke_attn_bwd_kslice_t g4_first_slice(B* b, Step* t)
{
    const Geom* g = t->geom;
    const rocke_attn_bwd_block_gemm_t* g4 = &g->g4;
    if(!t->has_v_ds)
        value_error("'NoneType' object has no attribute 'load'");
    Opnd a_ds = lds_opnd(b, &t->v_ds);
    Opnd kt_b = kt_b_default(b, t);
    int per = rocke_attn_bwd_block_gemm_check_k_slice(b, g4, g->k_k4);
    need(b, per > 0);
    rocke_attn_bwd_kslice_t sl;
    sl.a = arena<V>(b, (size_t)per * FM(g4));
    sl.b = arena<V>(b, (size_t)per * FN(g4));
    sl.n = per;
    rocke_attn_bwd_gemm_operand_t ga = a_ds.g(), gb = kt_b.g();
    need(b,
         rocke_attn_bwd_block_gemm_load_k_slice(b, g4, &ga, &gb, 0, g->k_k4, t->lane, t->o4, &sl));
    return sl;
}

void g4_dq(B* b,
           Step* t,
           V h,
           V q0,
           const rocke_attn_bwd_packed_step_t* packed,
           bool edge,
           const rocke_attn_bwd_lds_view_t* v_ds,
           const rocke_attn_bwd_frags_t* kt_ahead,
           std::vector<V>* record,
           const rocke_attn_bwd_kslice_t* first)
{
    const Geom* g = t->geom;
    const rocke_attn_bwd_block_gemm_t* g4 = &g->g4;
    V lane = t->lane;
    bool clamp = edge || g->interior_clamp;
    Opnd kt_b = kt_ahead != NULL ? regs_opnd(*kt_ahead) : kt_b_default(b, t);
    int cfl = CFL(g4);
    struct Frag
    {
        int fm, fn;
        V frag;
    };
    auto hand_off = [&](const std::vector<Frag>& frags) {
        if(record != NULL)
        {
            for(const Frag& f : frags)
                for(int s = 0; s < cfl; ++s)
                    record->push_back(needv(b, rocke_b_vec_extract(b, f.frag, s)));
            return;
        }
        std::vector<rocke_attn_bwd_dq_slot_t> slots;
        for(const Frag& f : frags)
        {
            std::vector<V> co = acc_coords(b, g4, lane, t->o4, f.fm, f.fn);
            for(int s = 0; s < cfl; ++s)
            {
                rocke_attn_bwd_dq_slot_t slot;
                slot.row = ivv(co[2 * s]);
                slot.col = ivv(co[2 * s + 1]);
                slot.value = needv(b, rocke_b_vec_extract(b, f.frag, s));
                slots.push_back(slot);
            }
        }
        need(b,
             rocke_attn_bwd_dq_sink_consume(t->sink,
                                            b,
                                            slots.data(),
                                            (int)slots.size(),
                                            packed == NULL ? q0 : NULL,
                                            packed == NULL ? NULL : &packed->rows,
                                            h,
                                            clamp));
        if(t->has_waits)
            rocke_attn_bwd_counted_waits_atomics(&t->waits, (int)slots.size());
    };
    const rocke_attn_bwd_lds_view_t* ds_view = v_ds != NULL ? v_ds : &t->v_ds;
    if(v_ds == NULL && !t->has_v_ds)
        value_error("'NoneType' object has no attribute 'load'");
    Opnd a_ds = lds_opnd(b, ds_view);
    if(g->acc_in_lds || !eq(g->dq_issue, "block"))
    {
        if(eq(g->dq_issue, "fragment_regs") && !g->acc_in_lds)
            a_ds = regs_opnd(load_all_a(b, g4, a_ds, lane, t->o4[0]));
        rocke_attn_bwd_gemm_operand_t ga = a_ds.g(), gb = kt_b.g();
        for(int fm = 0; fm < FM(g4); ++fm)
            for(int fn = 0; fn < FN(g4); ++fn)
            {
                V z = needv(b, rocke_b_zero_vec_f32(b, cfl));
                V frag = needv(
                    b, rocke_attn_bwd_block_gemm_mma_frag(b, g4, z, &ga, &gb, fm, fn, lane, t->o4));
                hand_off(std::vector<Frag>{Frag{fm, fn, frag}});
            }
        return;
    }
    V* dq;
    if(eq(g->g4_slicing, ROCKE_ATTN_BWD_DEFAULT_G4_SLICING))
        dq = gemm_mma(
            b, g4, zero_acc(b, g4), a_ds, kt_b, lane, t->o4, rocke_attn_bwd_mma_opts_default());
    else
    {
        V* z = zero_acc(b, g4);
        dq = arena<V>(b, (size_t)FM(g4) * FN(g4));
        rocke_attn_bwd_frags_t fr = frags_of(z, FM(g4), FN(g4));
        rocke_attn_bwd_gemm_operand_t ga = a_ds.g(), gb = kt_b.g();
        need(
            b,
            rocke_attn_bwd_block_gemm_mma_k_sliced(b,
                                                   g4,
                                                   &fr,
                                                   &ga,
                                                   &gb,
                                                   g->k_k4,
                                                   rocke_attn_bwd_tile_geometry_g4_slice_buffers(g),
                                                   lane,
                                                   t->o4,
                                                   first,
                                                   dq));
    }
    std::vector<Frag> all;
    for(int fm = 0; fm < FM(g4); ++fm)
        for(int fn = 0; fn < FN(g4); ++fn)
            all.push_back(Frag{fm, fn, dq[fm * FN(g4) + fn]});
    hand_off(all);
}

} // namespace

namespace
{

/* ====================================================================== *
 * BwdTileStep: staging, the q step
 * ====================================================================== */

void step_init(B* b,
               Step* t,
               const Geom* geom,
               rocke_attn_bwd_mask_policy_t* mask,
               const SV* stats,
               rocke_attn_bwd_dq_sink_t* sink,
               const rocke_attn_bwd_bwd_lds_t* lds,
               V tid,
               V lane,
               V wave,
               V scale_log2,
               V ds_mult,
               bool sentinel)
{
    require_derived(geom);
    memset(t, 0, sizeof(*t));
    t->geom = geom;
    t->mask = mask;
    t->stats = stats;
    t->sink = sink;
    t->lds = lds;
    t->tid = tid;
    t->lane = lane;
    t->wave = wave;
    t->scale_log2 = scale_log2;
    t->ds_mult = ds_mult;
    t->sentinel = sentinel;
    need(b, rocke_attn_bwd_tile_geometry_stage_table(b, geom, &t->sched));
    t->elem = rocke_attn_bwd_ir_type(b, geom->dtype);
    need(b, t->elem != NULL);
    int d = geom->head_size, m0 = geom->k_m0, n0 = geom->k_n0;
    int g0_waves = rocke_attn_bwd_block_gemm_waves(&geom->g0);
    t->g0_replicated = g0_waves != geom->waves;
    V wave0 = wave;
    if(t->g0_replicated)
    {
        V cw = ci(b, g0_waves);
        wave0 = rocke_b_mod(b, wave, cw);
    }
    rocke_attn_bwd_block_gemm_wave_origin(b, &geom->g0, wave0, t->o0);
    rocke_attn_bwd_block_gemm_wave_origin(b, &geom->g1, wave, t->o1);
    if(sink->emits_g4)
        rocke_attn_bwd_block_gemm_wave_origin(b, &geom->g4, wave, t->o4);
    need(b, true);
    t->k_regs = t->v_regs = t->kt_regs = no_frags();
    t->v_q = lds_view(b, lds, "q", d, 0);
    t->v_do = lds_view(b, lds, "do", d, 0);
    t->has_v_ds = rocke_attn_bwd_lds_plan_has(lds->plan, "ds");
    if(t->has_v_ds)
        t->v_ds = lds_view(b, lds, "ds", n0, 0);
    t->has_v_pt = eq(geom->pt_route, "lds");
    if(t->has_v_pt)
        t->v_pt = lds_view(b, lds, "pt", n0, 0);
    t->has_v_ring_stats = eq(geom->barrier_merge, "on");
    if(t->has_v_ring_stats)
    {
        t->v_lse2 = lds_view(b, lds, "lse2", 1, 0);
        t->v_dsum = lds_view(b, lds, "dsum", 1, 0);
    }
    t->v_k = lds_view(
        b, lds, rocke_attn_bwd_lds_plan_has(lds->plan, "k_res") ? "k_res" : "k_stage", d, 0);
    t->v_v = lds_view(
        b, lds, rocke_attn_bwd_lds_plan_has(lds->plan, "v_res") ? "v_res" : "v_stage", d, 0);
    if(eq(rocke_attn_bwd_tile_geometry_reader(geom), "xt"))
    {
        t->v_qt = lds_view(b, lds, "qt", m0, 0);
        t->has_v_qt = true;
        t->v_dot = lds_view(b, lds, "dot", m0, 0);
        t->has_v_dot = true;
        const char* kts[2] = {"kt_res", "kt_stage"};
        for(const char* kt : kts)
            if(rocke_attn_bwd_lds_plan_has(lds->plan, kt))
            {
                t->v_kt = lds_view(b, lds, kt, n0, 0);
                t->has_v_kt = true;
                break;
            }
    }
    t->has_v_acc = geom->acc_in_lds;
    if(geom->acc_in_lds)
    {
        t->v_dk_acc = lds_view(b, lds, "dk_acc", d, 0);
        t->v_dv_acc = lds_view(b, lds, "dv_acc", d, 0);
    }
    if(m0 % 16 || n0 % 16)
        value_error("tile extents must be multiples of 16");
    if(eq(geom->loop_sync, "counted"))
    {
        need(b,
             rocke_attn_bwd_counted_waits_init(
                 &t->waits, b, rocke_attn_bwd_tile_geometry_counted_atomics(geom)));
        t->has_waits = true;
    }
}

void copy_rows_to_lds(B* b,
                      const rocke_attn_bwd_row_slab_t* slab,
                      const rocke_attn_bwd_lds_view_t* view,
                      V row0,
                      int rows,
                      int cols,
                      V tid,
                      int threads,
                      int vec)
{
    rocke_attn_bwd_slab_row_loader_t ctx = {b, *slab, row0, vec};
    rocke_attn_bwd_tile_loader_t load = rocke_attn_bwd_slab_row_loader(&ctx);
    copy_rows(b, load, view, rows, cols, tid, threads, vec);
}

/* copy_rows_to_lds over a duck-typed slab. */
void copy_source_to_lds(B* b,
                        const rocke_attn_bwd_row_source_t* src,
                        const rocke_attn_bwd_lds_view_t* view,
                        V row0,
                        int rows,
                        int cols,
                        V tid,
                        int threads,
                        int vec)
{
    SourceRowLoader ctx = {b, src, row0, vec};
    rocke_attn_bwd_tile_loader_t load = {source_load_fn, (void*)&ctx};
    copy_rows(b, load, view, rows, cols, tid, threads, vec);
}

/* stage_kv over K / V row sources (a RowSlab or another shell's slab). */
template <class Src, class Copy>
void stage_kv_with(B* b, Step* t, const Src* k_slab, const Src* v_slab, V k0, V live, Copy copy)
{
    const Geom* g = t->geom;
    int d = g->head_size, n0 = g->k_n0;
    int threads = threads_of(g);
    V lane = t->lane;
    bool reg = eq(g->kv_residency, "reg");
    {
        IfScope s(b, live);
        copy(b, k_slab, &t->v_k, k0, n0, d, t->tid, threads, 8);
        if(!reg)
            copy(b, v_slab, &t->v_v, k0, n0, d, t->tid, threads, 8);
    }
    sync(b);
    if(reg)
        t->k_regs = load_all_b(b, &g->g0, lds_opnd(b, &t->v_k), lane, t->o0[1]);
    if(t->has_v_kt)
    {
        write_xt(b, &t->v_k, &t->v_kt, n0, d, t->tid, threads, t->elem);
        sync(b); /* the K^T image is visible to every wave */
    }
    if(eq(g->kt_source, "reg") && t->sink->emits_g4)
        t->kt_regs = load_all_b(b, &g->g4, kt_operand(b, t), lane, t->o4[1]);
    if(reg)
    {
        sync(b); /* stage-0 K reads complete before V overwrites the phase */
        {
            IfScope s(b, live);
            copy(b, v_slab, &t->v_v, k0, n0, d, t->tid, threads, 8);
        }
        sync(b);
        t->v_regs = load_all_b(b, &g->g0, lds_opnd(b, &t->v_v), lane, t->o0[1]);
        sync(b); /* stage-0 reads complete before the loop phase is written */
    }
}

void stage_kv(B* b,
              Step* t,
              const rocke_attn_bwd_row_slab_t* k_slab,
              const rocke_attn_bwd_row_slab_t* v_slab,
              V k0,
              V live)
{
    stage_kv_with(b, t, k_slab, v_slab, k0, live, copy_rows_to_lds);
}

void zero_lds_acc(B* b, Step* t)
{
    const Geom* g = t->geom;
    int d = g->head_size, n0 = g->k_n0;
    int threads = threads_of(g);
    V zero4 = needv(b, rocke_b_zero_vec_f32(b, 4));
    int per_row = d / 4;
    int total = n0 * per_row;
    const rocke_attn_bwd_lds_view_t* views[2] = {&t->v_dk_acc, &t->v_dv_acc};
    for(const rocke_attn_bwd_lds_view_t* view : views)
    {
        int n = rocke_attn_bwd_deal_chunks_count(b, total, threads);
        need(b, n >= 0);
        for(int it = 0; it < n; ++it)
        {
            bool tail = false;
            V chunk
                = needv(b, rocke_attn_bwd_deal_chunks_round(b, t->tid, total, threads, it, &tail));
            auto body = [&]() {
                V r, c;
                rocke_attn_bwd_chunk_coords(b, chunk, per_row, 4, &r, &c);
                need(b, true);
                vstore(b, view, ivv(r), ivv(c), zero4, 4);
            };
            if(!tail)
                body();
            else
            {
                V ct = ci(b, total);
                V cond = rocke_b_cmp_lt(b, chunk, ct);
                IfScope s(b, cond);
                body();
            }
        }
    }
    sync(b);
}

rocke_attn_bwd_dkdv_acc_t zero_acc_step(B* b, Step* t)
{
    const Geom* g = t->geom;
    rocke_attn_bwd_dkdv_acc_t acc;
    memset(&acc, 0, sizeof(acc));
    if(g->acc_in_lds)
    {
        zero_lds_acc(b, t);
        return acc;
    }
    acc.dk = zero_acc(b, &g->g1);
    acc.dv = zero_acc(b, &g->g1);
    acc.frags_m = FM(&g->g1);
    acc.frags_n = FN(&g->g1);
    return acc;
}

/* _stage_qdo: the synchronous staging; stats->row_of empty when None. */
bool stage_qdo(B* b,
               Step* t,
               const rocke_attn_bwd_row_slab_t* q_slab,
               const rocke_attn_bwd_row_slab_t* do_slab,
               V h,
               V q0,
               const rocke_attn_bwd_packed_step_t* packed,
               bool edge,
               Stats* stats)
{
    const Geom* g = t->geom;
    bool have = false;
    if(eq(g->stats_load, "step_top"))
    {
        *stats = load_stats(b, t, h, q0, packed ? &packed->rows : NULL);
        have = true;
    }
    rocke_attn_bwd_slab_row_loader_t sq, sdo;
    rocke_attn_bwd_packed_row_loader_t pq, pdo;
    rocke_attn_bwd_row_set_t sets[2];
    if(packed == NULL)
    {
        if(q_slab == NULL || do_slab == NULL)
            value_error("'NoneType' object has no attribute 'load_vec'");
        sq = {b, *q_slab, q0, 8};
        sdo = {b, *do_slab, q0, 8};
        sets[0].load = rocke_attn_bwd_slab_row_loader(&sq);
        sets[1].load = rocke_attn_bwd_slab_row_loader(&sdo);
    }
    else
    {
        bool clamp = edge || g->interior_clamp;
        pq = {b, packed->q_addr, packed->rows, 8, clamp};
        pdo = {b, packed->do_addr, packed->rows, 8, clamp};
        sets[0].load = rocke_attn_bwd_packed_row_loader(&pq);
        sets[1].load = rocke_attn_bwd_packed_row_loader(&pdo);
    }
    sets[0].view = &t->v_q;
    sets[1].view = &t->v_do;
    copy_row_sets(b,
                  sets,
                  2,
                  g->qdo_stage,
                  g->k_m0,
                  g->head_size,
                  t->tid,
                  threads_of(g),
                  t->wave,
                  wave_size_of(g),
                  8);
    loop_barrier(b, t);
    return have;
}

void pre_issue_trampoline(B* b, void* ctx)
{
    (void)b;
    (*(std::function<void()>*)ctx)();
}

void run_q_step(B* b,
                Step* t,
                const rocke_attn_bwd_dkdv_acc_t* acc,
                const rocke_attn_bwd_row_slab_t* q_slab,
                const rocke_attn_bwd_row_slab_t* do_slab,
                V qt,
                V h,
                V k0,
                const rocke_attn_bwd_q_step_opts_t& o,
                rocke_attn_bwd_q_step_result_t* res)
{
    const Geom* g = t->geom;
    const rocke_attn_bwd_block_gemm_t* g0 = &g->g0;
    const rocke_attn_bwd_block_gemm_t* g1 = &g->g1;
    int m0 = g->k_m0;
    V lane = t->lane;
    bool edge = o.edge;
    const rocke_attn_bwd_packed_step_t* packed = o.packed;
    const rocke_attn_bwd_packed_rows_t* prows = packed ? &packed->rows : NULL;
    bool fenced = in(g->sched, ROCKE_ATTN_BWD_FENCED_SCHEDS, ROCKE_ATTN_BWD_N_FENCED_SCHEDS);
    if((o.pending != NULL) != eq(g->sched, "stage_table_atomics"))
        value_error("a pending dQ partial is carried by stage_table_atomics");
    V cm0 = ci(b, m0);
    V q0 = rocke_b_mul(b, qt, cm0);
    need(b, true);
    bool counted = eq(g->loop_sync, "counted");
    Stats stats;
    bool have_stats = false;
    bool have_ring_loads = false;
    std::vector<PendingStore> ring_ld;
    bool have_nxt = false;
    rocke_attn_bwd_staged_step_t nxt_staged;
    memset(&nxt_staged, 0, sizeof(nxt_staged));
    if(eq(g->sched, "stage_table_atomics") && counted && eq(g->counted_issue, "late"))
        value_error(ROCKE_ATTN_BWD_COUNTED_LATE_DEFERRED);
    bool loads_in_a;
    {
        rocke_attn_bwd_ring_prefetch_t rp;
        rocke_attn_bwd_err_t e;
        need_st(rocke_attn_bwd_stage_table_ring_prefetch(&t->sched, &rp, &e), e);
        loads_in_a = rp.present && eq(rp.where, "A");
    }
    const rocke_attn_bwd_ring_step_t* ring = o.ring;
    rocke_attn_bwd_prefetch_step_t* prefetch = o.prefetch;
    RingViews cur, nxt;
    const rocke_attn_bwd_lds_view_t *v_q, *v_do, *v_ds;
    bool has_ds;
    rocke_attn_bwd_lds_view_t pf_q, pf_do;
    rocke_attn_bwd_lds_stats_view_t pf_stats;
    const rocke_attn_bwd_lds_stats_view_t* stats_src = NULL;
    /* barrier: the step's in-loop barrier (the prefetch ring's is LDS-only) */
    auto barrier = [&]() {
        if(prefetch != NULL)
            need(b, rocke_attn_bwd_prefetch_step_barrier(prefetch, b));
        else
            loop_barrier(b, t);
    };
    if(ring != NULL)
    {
        cur = ring_views(b, t, ring->slot);
        V one = ci(b, 1);
        V other = rocke_b_sub(b, one, ring->slot);
        need(b, true);
        nxt = ring_views(b, t, other);
        v_q = &cur.q;
        v_do = &cur.dO;
        v_ds = &cur.ds;
        has_ds = cur.has_ds;
        if(eq(g->ring_issue, "early") && !loads_in_a)
        {
            ring_ld = ring_loads(b, t, &ring->nxt, nxt);
            have_ring_loads = true;
        }
    }
    else if(prefetch != NULL)
    {
        need(b,
             rocke_attn_bwd_prefetch_step_begin(prefetch, b, loads_in_a, &pf_q, &pf_do, &pf_stats));
        v_q = &pf_q;
        v_do = &pf_do;
        stats_src = &pf_stats;
        v_ds = &t->v_ds;
        has_ds = t->has_v_ds;
    }
    else if(counted)
    {
        if(o.staged == NULL || o.next_loads == NULL)
            value_error("loop_sync='counted' needs staged and next_loads");
        v_q = &t->v_q;
        v_do = &t->v_do;
        v_ds = &t->v_ds;
        has_ds = t->has_v_ds;
        rocke_attn_bwd_counted_waits_step_entry(&t->waits);
        stats = store_staged(b, t, o.staged, h, q0, packed);
        have_stats = true;
        loop_barrier(b, t);
        if(eq(g->counted_issue, "early"))
        {
            nxt_staged = issue_next(b, t, o.next_loads, o.next_loads_ctx);
            have_nxt = true;
        }
    }
    else
    {
        v_q = &t->v_q;
        v_do = &t->v_do;
        v_ds = &t->v_ds;
        has_ds = t->has_v_ds;
        have_stats = stage_qdo(b, t, q_slab, do_slab, h, q0, packed, edge, &stats);
    }
    need(b, rocke_attn_bwd_stage_table_emit_loop_head(b, &t->sched));

    bool xt_unpublished = t->has_v_qt;
    int threads = threads_of(g);
    if(xt_unpublished)
    {
        if(ring != NULL || prefetch != NULL || counted)
            value_error("xt_lds runs on the synchronous body only");
        write_xt(b, &t->v_q, &t->v_qt, m0, g->head_size, t->tid, threads, t->elem);
        write_xt(b, &t->v_do, &t->v_dot, m0, g->head_size, t->tid, threads, t->elem);
    }

    const rocke_attn_bwd_frag_coords_t* coords = &g->b13_coords;
    Opnd b_dot, b_qt;
    const char* reader = rocke_attn_bwd_tile_geometry_reader(g);
    if(t->has_v_qt)
    {
        b_dot = lds_opnd(b, &t->v_dot, ROCKE_ATTN_BWD_K_INNER, coords);
        b_qt = lds_opnd(b, &t->v_qt, ROCKE_ATTN_BWD_K_INNER, coords);
    }
    else
    {
        b_dot = lds_opnd(b, v_do, ROCKE_ATTN_BWD_K_OUTER, coords, reader);
        b_qt = lds_opnd(b, v_q, ROCKE_ATTN_BWD_K_OUTER, coords, reader);
    }
    std::map<std::string, rocke_attn_bwd_frags_t> ahead;
    auto read_ahead = [&](const std::string& name) {
        if(name == "g1")
            ahead["g1"] = load_all_b(b, g1, b_dot, lane, t->o1[1]);
        else if(name == "g3")
            ahead["g3"] = load_all_b(b, g1, b_qt, lane, t->o1[1]);
        else if(t->kt_regs.v == NULL)
            ahead["g4"] = load_all_b(b, &g->g4, kt_operand(b, t), lane, t->o4[1]);
    };

    bool g4_on = t->sink->emits_g4;
    const char* units3[3];
    stage_units_(g->stage_order, units3);
    std::vector<std::string> order;
    for(const char* u : units3)
        if(g4_on || strcmp(u, "g4") != 0)
            order.push_back(u);

    /* Stage A: G0 */
    need(b, rocke_attn_bwd_stage_table_emit_stage_begin(b, &t->sched, "A"));
    if(loads_in_a)
    {
        if(ring != NULL)
        {
            ring_ld = ring_loads(b, t, &ring->nxt, nxt);
            have_ring_loads = true;
        }
        else
        {
            if(prefetch == NULL)
                value_error("'NoneType' object has no attribute 'issue_loads'");
            need(b, rocke_attn_bwd_prefetch_step_issue_loads(prefetch, b));
        }
    }
    Opnd k_b = t->k_regs.v != NULL ? regs_opnd(t->k_regs) : lds_opnd(b, &t->v_k);
    V* s_acc;
    if(g02_asm(g))
    {
        rocke_attn_bwd_mma_opts_t mo = asm_opts(g);
        mo.has_asm_agpr_base = true;
        mo.asm_agpr_base = g02_agpr_base(g, 0);
        s_acc = gemm_mma(b, g0, NULL, lds_opnd(b, v_q), k_b, lane, t->o0, mo);
    }
    else
    {
        V* z = zero_acc(b, g0);
        s_acc = gemm_mma(
            b, g0, z, lds_opnd(b, v_q), k_b, lane, t->o0, rocke_attn_bwd_mma_opts_default());
    }
    if(fenced)
    {
        if(!have_stats && ring != NULL)
            stats = ring_stats(b, t, cur);
        else if(!have_stats)
            stats = load_stats(b, t, h, q0, prows, stats_src);
        have_stats = true;
        read_ahead(order[0]);
        if(o.pending != NULL)
            flush_dq(b, t, o.pending);
    }
    need(b, rocke_attn_bwd_stage_table_emit_stage_end(b, &t->sched, "A", true, true));
    need(b, rocke_attn_bwd_stage_table_emit_stage_begin(b, &t->sched, "B"));
    Opnd v_b = t->v_regs.v != NULL ? regs_opnd(t->v_regs) : lds_opnd(b, &t->v_v);
    V* dp_acc;
    if(g02_asm(g))
    {
        rocke_attn_bwd_mma_opts_t mo = asm_opts(g);
        mo.has_asm_agpr_base = true;
        mo.asm_agpr_base = g02_agpr_base(g, 1);
        dp_acc = gemm_mma(b, g0, NULL, lds_opnd(b, v_do), v_b, lane, t->o0, mo);
    }
    else
    {
        V* z = zero_acc(b, g0);
        dp_acc = gemm_mma(
            b, g0, z, lds_opnd(b, v_do), v_b, lane, t->o0, rocke_attn_bwd_mma_opts_default());
    }

    if(ring != NULL && !have_stats)
    {
        stats = ring_stats(b, t, cur);
        have_stats = true;
    }
    int n0f = FM(g0) * FN(g0);
    V* p_f32 = arena<V>(b, n0f);
    V* ds_f32 = arena<V>(b, n0f);
    softmax_grad(b,
                 t,
                 s_acc,
                 dp_acc,
                 q0,
                 k0,
                 edge,
                 prows,
                 have_stats ? &stats : NULL,
                 h,
                 stats_src,
                 p_f32,
                 ds_f32);

    Opnd a_p, a_ds;
    if(eq(g->pt_route, "relabel"))
    {
        const char* cast = g->bf16_cast;
        int na = FM(g1) * KS(g1);
        V* ap = arena<V>(b, na);
        V* ad = arena<V>(b, na);
        rocke_attn_bwd_frags_t pf = frags_of(p_f32, FM(g0), FN(g0));
        rocke_attn_bwd_frags_t df = frags_of(ds_f32, FM(g0), FN(g0));
        need(b, rocke_attn_bwd_relabel_acc_to_a(b, g0, g1, &pf, lane, cast, ap, NULL));
        need(b, rocke_attn_bwd_relabel_acc_to_a(b, g0, g1, &df, lane, cast, ad, NULL));
        a_p = regs_opnd(frags_of(ap, FM(g1), KS(g1)));
        a_ds = regs_opnd(frags_of(ad, FM(g1), KS(g1)));
        if(g4_on)
        {
            if(!has_ds)
                value_error("'NoneType' object has no attribute 'store'");
            write_c_tiles(b, t, v_ds, ds_f32);
        }
    }
    else
    {
        if(!t->has_v_pt || !t->has_v_ds)
            value_error("'NoneType' object has no attribute 'store'");
        write_c_tiles(b, t, &t->v_pt, p_f32);
        write_c_tiles(b, t, &t->v_ds, ds_f32);
        barrier(); /* also publishes the Q^T / dO^T images */
        xt_unpublished = false;
        a_p = k_outer_operand(b, t, &t->v_pt);
        a_ds = k_outer_operand(b, t, &t->v_ds);
    }

    /* G1, G3 and G4 in stage_order */
    V* out_dv = NULL;
    V* out_dk = NULL;
    std::vector<V> record;
    bool recording = o.pending != NULL;
    std::map<std::string, std::pair<Opnd, Opnd>> operands;
    operands["g1"] = std::make_pair(a_p, b_dot);
    operands["g3"] = std::make_pair(a_ds, b_qt);
    std::map<std::string, std::function<void()>> pre_issue;
    std::string hoisted_unit;
    if(g->asm_load_hoist)
    {
        std::vector<std::string> g13;
        for(const std::string& u : order)
            if(u != "g4")
                g13.push_back(u);
        if(g13.size() != 2)
            value_error("not enough values to unpack (expected 2, got " + std::to_string(g13.size())
                        + ")");
        hoisted_unit = g13[1];
        pre_issue[g13[0]] = [&, hoisted_unit]() {
            std::pair<Opnd, Opnd>& ops = operands[hoisted_unit];
            auto it = ahead.find(hoisted_unit);
            Opnd bsrc = it != ahead.end() ? regs_opnd(it->second) : ops.second;
            int na = FM(g1) * KS(g1), nb = FN(g1) * KS(g1);
            V* a_out = arena<V>(b, na);
            V* b_out = arena<V>(b, nb);
            rocke_attn_bwd_gemm_operand_t ga = ops.first.g(), gb = bsrc.g();
            need(b,
                 rocke_attn_bwd_block_gemm_load_operands(
                     b, g1, &ga, &gb, lane, t->o1, a_out, b_out));
            ops.first = regs_opnd(frags_of(a_out, FM(g1), KS(g1)));
            ops.second = regs_opnd(frags_of(b_out, FN(g1), KS(g1)));
        };
    }
    auto g13_unit = [&](const std::string& name,
                        const rocke_attn_bwd_lds_view_t* acc_view,
                        V* acc_frags,
                        V** out) {
        std::pair<Opnd, Opnd> ops = operands[name];
        auto it = ahead.find(name);
        Opnd bop = it != ahead.end() ? regs_opnd(it->second) : ops.second;
        if(g->acc_in_lds)
        {
            update_lds_acc(b, t, acc_view, ops.first, bop);
            *out = NULL;
            return;
        }
        rocke_attn_bwd_mma_opts_t mo = g13_asm(g) ? asm_opts(g) : rocke_attn_bwd_mma_opts_default();
        auto pi = pre_issue.find(name);
        if(pi != pre_issue.end())
        {
            mo.pre_issue = pre_issue_trampoline;
            mo.pre_issue_ctx = (void*)&pi->second;
        }
        *out = gemm_mma(b, g1, acc_frags, ops.first, bop, lane, t->o1, mo);
    };
    const char* ds_reader = g4_prefetch_before_g3(t) ? "g3" : "g4";
    bool have_first = false;
    rocke_attn_bwd_kslice_t first;
    memset(&first, 0, sizeof(first));
    const char* stages = "BCD";
    for(size_t pos = 0; pos < order.size(); ++pos)
    {
        const std::string& name = order[pos];
        char stage[2] = {stages[pos], '\0'};
        if(name == ds_reader && eq(g->pt_route, "relabel"))
        {
            if(ring != NULL)
            {
                if(!have_ring_loads) /* ring_issue = "late" */
                {
                    ring_ld = ring_loads(b, t, &ring->nxt, nxt);
                    have_ring_loads = true;
                }
                ring_stores(b, ring_ld); /* the next step's slot */
            }
            barrier(); /* the dS tile above is visible to every wave */
            xt_unpublished = false;
        }
        else if(xt_unpublished)
        {
            barrier(); /* the Q^T / dO^T images are visible */
            xt_unpublished = false;
        }
        if(name == "g4" && counted && eq(g->counted_issue, "late"))
        {
            nxt_staged = issue_next(b, t, o.next_loads, o.next_loads_ctx);
            have_nxt = true;
        }
        if(pos)
            need(b, rocke_attn_bwd_stage_table_emit_stage_begin(b, &t->sched, stage));
        if(name == "g3" && g4_prefetch_before_g3(t))
        {
            first = g4_first_slice(b, t);
            have_first = true;
        }
        if(name == "g1")
            g13_unit("g1", &t->v_dv_acc, acc->dv, &out_dv);
        else if(name == "g3")
            g13_unit("g3", &t->v_dk_acc, acc->dk, &out_dk);
        else
        {
            if(!has_ds)
                value_error("'NoneType' object has no attribute 'load'");
            auto it = ahead.find("g4");
            g4_dq(b,
                  t,
                  h,
                  q0,
                  packed,
                  edge,
                  v_ds,
                  it != ahead.end() ? &it->second : NULL,
                  recording ? &record : NULL,
                  have_first ? &first : NULL);
        }
        if(fenced && pos + 1 < order.size())
            read_ahead(order[pos + 1]);
        need(b, rocke_attn_bwd_stage_table_emit_stage_end(b, &t->sched, stage, true, true));
    }
    if(prefetch != NULL)
        need(b, rocke_attn_bwd_prefetch_step_end(prefetch, b)); /* the ring's wait and barrier */
    else if(ring == NULL)
        loop_barrier(b, t); /* q / do / ds / pt reads done before the next step */
    if(counted)
        need(b, rocke_attn_bwd_counted_waits_end_step(&t->waits, b));
    memset(res, 0, sizeof(*res));
    if(counted)
    {
        res->has_staged = true;
        res->staged = nxt_staged;
        (void)have_nxt;
    }
    if(recording)
    {
        V* vals = arena<V>(b, record.size());
        for(size_t i = 0; i < record.size(); ++i)
            vals[i] = record[i];
        res->has_pending = true;
        res->pending.values = vals;
        res->pending.n_values = (int)record.size();
        res->pending.q0 = prows == NULL ? q0 : prows->row0;
        res->pending.h = prows == NULL ? h : prows->head0;
        res->pending.n_heads = prows == NULL ? NULL : prows->n_heads;
        res->pending.rows_per_head = prows == NULL ? NULL : prows->rows_per_head;
    }
    if(g->acc_in_lds)
    {
        res->acc = *acc;
        return;
    }
    int n1 = FM(g1) * FN(g1);
    if(eq(g->agpr_route, "agpr_fence"))
    {
        V* dk = arena<V>(b, n1);
        V* dv = arena<V>(b, n1);
        for(int i = 0; i < n1; ++i)
            dk[i] = empty_asm(b, out_dk[i], "=a,0", out_dk[i]->type);
        for(int i = 0; i < n1; ++i)
            dv[i] = empty_asm(b, out_dv[i], "=a,0", out_dv[i]->type);
        out_dk = dk;
        out_dv = dv;
    }
    res->acc.dk = out_dk;
    res->acc.dv = out_dv;
    res->acc.frags_m = FM(g1);
    res->acc.frags_n = FN(g1);
}

/* ====================================================================== *
 * epilogue helpers
 * ====================================================================== */

void check_which(const char* which)
{
    if(!eq(which, "dk") && !eq(which, "dv"))
        value_error("which=" + repr(which) + " must be 'dk' or 'dv'");
}

V* frags_which(const rocke_attn_bwd_dkdv_acc_t* acc, const char* which)
{
    return eq(which, "dk") ? acc->dk : acc->dv;
}

const rocke_attn_bwd_lds_view_t* acc_view_which(Step* t, const char* which)
{
    return eq(which, "dk") ? &t->v_dk_acc : &t->v_dv_acc;
}

void need_reg_frags(const rocke_attn_bwd_dkdv_acc_t* acc, Step* t)
{
    const rocke_attn_bwd_block_gemm_t* g1 = &t->geom->g1;
    if(acc == NULL || acc->dk == NULL || acc->dv == NULL || acc->frags_m != FM(g1)
       || acc->frags_n != FN(g1))
        value_error("tuple index out of range");
}

/* _rounded_frag(b, frag, mult) */
V rounded_frag(B* b, Step* t, V frag, V mult)
{
    if(mult != NULL)
    {
        int n = CFL(&t->geom->g1);
        std::vector<V> vals;
        for(int i = 0; i < n; ++i)
        {
            V x = needv(b, rocke_b_vec_extract(b, frag, i));
            vals.push_back(rocke_b_fmul(b, mult, x));
        }
        frag = needv(b, rocke_b_vec_pack(b, vals.data(), n, rocke_f32()));
    }
    return needv(b, rocke_b_vec_cast_f32_to(b, frag, t->elem));
}

std::vector<rocke_attn_bwd_dkv_row_t>
    dkv_row_slots(B* b, Step* t, const char* which, const rocke_attn_bwd_dkdv_acc_t* acc)
{
    const rocke_attn_bwd_block_gemm_t* g1 = &t->geom->g1;
    check_which(which);
    if(!t->geom->acc_in_lds)
        need_reg_frags(acc, t);
    std::vector<rocke_attn_bwd_dkv_row_t> out;
    int fn_n = FN(g1);
    for(int fm = 0; fm < FM(g1); ++fm)
    {
        std::vector<V> frags;
        std::vector<std::vector<V>> coords;
        for(int fn = 0; fn < fn_n; ++fn)
        {
            if(t->geom->acc_in_lds)
                frags.push_back(load_lds_acc(b, t, acc_view_which(t, which), fm, fn));
            else
                frags.push_back(frags_which(acc, which)[fm * fn_n + fn]);
            coords.push_back(acc_lds_coords(b, t, fm, fn));
        }
        for(int s = 0; s < CFL(g1); ++s)
        {
            rocke_attn_bwd_dkv_row_t row;
            row.row = coords[0][2 * s];
            row.n_items = fn_n;
            row.cols = arena<V>(b, fn_n);
            row.values = arena<V>(b, fn_n);
            for(int fn = 0; fn < fn_n; ++fn)
            {
                row.cols[fn] = coords[fn][2 * s + 1];
                row.values[fn] = needv(b, rocke_b_vec_extract(b, frags[fn], s));
            }
            out.push_back(row);
        }
    }
    return out;
}

std::vector<rocke_attn_bwd_dkv_row_t>
    dkv_cells(B* b, Step* t, const char* which, const rocke_attn_bwd_dkdv_acc_t* acc, V mult)
{
    const rocke_attn_bwd_block_gemm_t* g1 = &t->geom->g1;
    check_which(which);
    if(t->geom->acc_in_lds)
        value_error("dkv_cells needs register accumulators");
    need_reg_frags(acc, t);
    V* frags = frags_which(acc, which);
    std::vector<rocke_attn_bwd_dkv_row_t> out;
    int fn_n = FN(g1);
    for(int fm = 0; fm < FM(g1); ++fm)
    {
        std::vector<V> halves;
        std::vector<std::vector<V>> coords;
        for(int fn = 0; fn < fn_n; ++fn)
        {
            halves.push_back(rounded_frag(b, t, frags[fm * fn_n + fn], mult));
            coords.push_back(acc_lds_coords(b, t, fm, fn));
        }
        for(int s = 0; s < CFL(g1); ++s)
        {
            rocke_attn_bwd_dkv_row_t row;
            row.n_items = fn_n;
            row.cols = arena<V>(b, fn_n);
            row.values = arena<V>(b, fn_n);
            for(int fn = 0; fn < fn_n; ++fn)
            {
                row.cols[fn] = coords[fn][2 * s + 1];
                row.values[fn] = needv(b, rocke_b_vec_extract(b, halves[fn], s));
            }
            row.row = coords[0][2 * s];
            out.push_back(row);
        }
    }
    return out;
}

rocke_attn_bwd_dkv_reader_t dkv_rows(B* b,
                                     Step* t,
                                     const char* which,
                                     const rocke_attn_bwd_dkdv_acc_t* acc,
                                     V mult,
                                     int vec,
                                     int slot,
                                     bool do_sync)
{
    const Geom* g = t->geom;
    if(vec != 8 && vec != 4 && vec != 2 && vec != 1)
        value_error("vec=" + std::to_string(vec) + " must be 8, 4, 2 or 1");
    rocke_attn_bwd_dkv_reader_t rd;
    memset(&rd, 0, sizeof(rd));
    rd.mult = mult;
    rd.vec = vec;
    rd.elem = t->elem;
    if(g->acc_in_lds)
    {
        rd.in_lds_acc = true;
        rd.view = *acc_view_which(t, which);
        return rd;
    }
    need_reg_frags(acc, t);
    V* frags = frags_which(acc, which);
    rocke_attn_bwd_lds_view_t out = lds_view(b, t->lds, "dkv_out", g->head_size, slot);
    const rocke_attn_bwd_block_gemm_t* g1 = &g->g1;
    for(int fm = 0; fm < FM(g1); ++fm)
        for(int fn = 0; fn < FN(g1); ++fn)
        {
            V half = rounded_frag(b, t, frags[fm * FN(g1) + fn], mult);
            std::vector<V> co = acc_lds_coords(b, t, fm, fn);
            for(int s = 0; s < CFL(g1); ++s)
            {
                V x = needv(b, rocke_b_vec_extract(b, half, s));
                vstore(b, &out, ivv(co[2 * s]), ivv(co[2 * s + 1]), x, 1);
            }
        }
    if(do_sync)
        sync(b);
    rd.view = out;
    return rd;
}

V dkv_read(B* b, const rocke_attn_bwd_dkv_reader_t* rd, V r, V c)
{
    if(!rd->in_lds_acc)
        return vload(b, &rd->view, ivv(r), ivv(c), rd->elem, rd->vec);
    int step = rd->vec < 4 ? rd->vec : 4;
    std::vector<V> parts;
    for(int k = 0; k < rd->vec; k += step)
    {
        V col = c;
        if(k != 0)
        {
            V ck = ci(b, k);
            col = rocke_b_add(b, c, ck);
        }
        parts.push_back(vload(b, &rd->view, ivv(r), ivv(col), rocke_f32(), step));
    }
    std::vector<V> vals;
    for(V p : parts)
        for(int i = 0; i < step; ++i)
            vals.push_back(needv(b, rocke_b_vec_extract(b, p, i)));
    if(rd->mult != NULL)
        for(V& v : vals)
            v = rocke_b_fmul(b, rd->mult, v);
    V packed = needv(b, rocke_b_vec_pack(b, vals.data(), (int)vals.size(), rocke_f32()));
    return needv(b, rocke_b_vec_cast_f32_to(b, packed, rd->elem));
}

std::vector<rocke_attn_bwd_dkv_quad_t>
    dkv_quad_rows(B* b, Step* t, const char* which, const rocke_attn_bwd_dkdv_acc_t* acc, V mult)
{
    const rocke_attn_bwd_block_gemm_t* g1 = &t->geom->g1;
    check_which(which);
    if(t->geom->acc_in_lds)
        value_error("dkv_quad_rows needs register accumulators");
    need_reg_frags(acc, t);
    V* frags = frags_which(acc, which);
    V c4 = ci(b, 4);
    V j = rocke_b_mod(b, t->lane, c4);
    need(b, true);
    std::vector<rocke_attn_bwd_dkv_quad_t> out;
    for(int fm = 0; fm < FM(g1); ++fm)
        for(int fn = 0; fn < FN(g1); ++fn)
        {
            V frag = frags[fm * FN(g1) + fn];
            std::vector<V> vals;
            for(int s = 0; s < 4; ++s)
                vals.push_back(needv(b, rocke_b_vec_extract(b, frag, s)));
            if(mult != NULL)
                for(V& v : vals)
                    v = rocke_b_fmul(b, mult, v);
            std::vector<V> co = acc_lds_coords(b, t, fm, fn);
            rocke_attn_bwd_dkv_quad_t q;
            q.row = rocke_b_add(b, co[0], j);
            q.col0 = rocke_b_sub(b, co[1], j);
            need(b, true);
            std::vector<V> tr = quad_transpose4(b, vals, t->lane);
            for(int i = 0; i < 4; ++i)
                q.values[i] = tr[i];
            out.push_back(q);
        }
    return out;
}

rocke_attn_bwd_lds_view_t dkv_f32_part(B* b,
                                       Step* t,
                                       const char* which,
                                       const rocke_attn_bwd_dkdv_acc_t* acc,
                                       V mult,
                                       int half,
                                       int slot)
{
    const Geom* g = t->geom;
    if(g->acc_in_lds)
        value_error("LDS accumulators are read in place (dkv_f32_image)");
    if((!eq(which, "dk") && !eq(which, "dv")) || (half != 0 && half != 1))
        value_error("which=" + repr(which) + " / half=" + std::to_string(half));
    int hw = g->head_size / 2;
    const rocke_attn_bwd_lds_buffer_t* buf = plan_buffer(t->lds->plan, "dkv_out", slot);
    if(g->k_n0 == 0)
        value_error("integer division or modulo by zero");
    if(buf->offset % 16 || buf->nbytes % (4 * g->k_n0))
        value_error("dkv_out cannot hold an fp32 half image");
    int pitch = buf->nbytes / (4 * g->k_n0);
    if(pitch < hw)
        value_error("dkv_out rows are narrower than half a row in fp32");
    rocke_attn_bwd_lds_view_t view;
    need(b,
         rocke_attn_bwd_lds_view_init(
             &view, b, t->lds->smem, buf->offset / 2, pitch, g->k_n0, 2, NULL));
    const rocke_attn_bwd_block_gemm_t* g1 = &g->g1;
    if(hw % g1->op->n)
        value_error("D/2 must be a multiple of the G1 atom N");
    need_reg_frags(acc, t);
    V* frags = frags_which(acc, which);
    bool is_static = g->grid_g1[1] == 1;
    V shift = NULL;
    if(half)
        shift = ci(b, hw);
    for(int fm = 0; fm < FM(g1); ++fm)
        for(int fn = 0; fn < FN(g1); ++fn)
        {
            if(is_static && (fn * g1->op->n) / hw != half)
                continue;
            V frag = frags[fm * FN(g1) + fn];
            std::vector<V> vals;
            for(int i = 0; i < CFL(g1); ++i)
                vals.push_back(needv(b, rocke_b_vec_extract(b, frag, i)));
            if(mult != NULL)
                for(V& v : vals)
                    v = rocke_b_fmul(b, mult, v);
            std::vector<V> co = acc_lds_coords(b, t, fm, fn);
            auto put = [&]() {
                for(size_t s = 0; s < vals.size(); ++s)
                {
                    V c = co[2 * s + 1];
                    if(shift != NULL)
                        c = rocke_b_sub(b, c, shift);
                    vstore(b, &view, ivv(co[2 * s]), ivv(c), vals[s], 1);
                }
            };
            if(is_static)
            {
                put();
                continue;
            }
            V c0 = co[1];
            V hc = ci(b, hw);
            V inside = half == 0 ? rocke_b_cmp_lt(b, c0, hc) : rocke_b_cmp_ge(b, c0, hc);
            IfScope s(b, inside);
            put();
        }
    return view;
}

} // namespace

/* ====================================================================== *
 * C ABI
 * ====================================================================== */

extern "C" {

rocke_status_t
    rocke_attn_bwd_stage_units(const char* order, const char* out[3], rocke_attn_bwd_err_t* err)
{
    for(int i = 0; i < ROCKE_ATTN_BWD_N_STAGE_ORDERS; ++i)
        if(eq(order, ROCKE_ATTN_BWD_STAGE_ORDERS[i]))
        {
            static const char* const names[] = {"g1", "g3", "g4"};
            for(int k = 0; k < 3; ++k)
            {
                std::string u = std::string(order).substr(2 * k, 2);
                for(const char* n : names)
                    if(u == n)
                        out[k] = n;
            }
            if(err != NULL)
                err->code = ROCKE_OK;
            return ROCKE_OK;
        }
    if(err != NULL)
    {
        err->code = ROCKE_ERR_VALUE;
        std::string m = "stage_order=" + repr(order) + " not in "
                        + tuple_repr(ROCKE_ATTN_BWD_STAGE_ORDERS, ROCKE_ATTN_BWD_N_STAGE_ORDERS);
        snprintf(err->msg, sizeof(err->msg), "%s", m.c_str());
    }
    return ROCKE_ERR_VALUE;
}

rocke_attn_bwd_tile_geometry_t rocke_attn_bwd_tile_geometry_default(void)
{
    rocke_attn_bwd_tile_geometry_t g;
    memset(&g, 0, sizeof(g));
    g.ring_depth = 1;
    g.global_path = "vgpr";
    g.sched = "none";
    g.agpr_alloc.kind = ROCKE_ATTN_BWD_AGPR_NONE;
    g.dq_mode = "atomic";
    g.stage_vec = 8;
    g.asm_tail_nop = NONE;
    g.asm_cluster = "k_step";
    g.stage_order = ROCKE_ATTN_BWD_DEFAULT_STAGE_ORDER;
    g.loop_sync = "full";
    g.dkv_epilogue = "lds_staged";
    g.dkv_store_vec = 8;
    g.qdo_stage = "split";
    g.stats_load = "at_use";
    g.dq_issue = "block";
    g.barrier_merge = "off";
    g.ring_issue = "early";
    g.counted_issue = "early";
    g.bf16_cast = "generic";
    g.bwd_exp2 = "exact";
    g.stats_vec = 1;
    g.g4_slicing = ROCKE_ATTN_BWD_DEFAULT_G4_SLICING;
    return g;
}

bool rocke_attn_bwd_tile_geometry_from_spec(rocke_ir_builder_t* b,
                                            const struct rocke_attn_bwd_spec* spec,
                                            const char* arch,
                                            rocke_attn_bwd_tile_geometry_t* out)
{
    return guarded(b, [&]() {
        const rocke_attn_bwd_spec_t* s = spec;
        if(s->waves == NONE || s->block_m == NONE)
            value_error("BwdTileGeometry needs a resolved spec");
        const char* epilogue = s->dkv_epilogue;
        if(eq(s->dkv_mode, "atomic") && !eq(epilogue, "direct"))
            epilogue = "lds_staged";
        rocke_attn_bwd_tile_geometry_t g = rocke_attn_bwd_tile_geometry_default();
        g.arch = arch;
        g.dtype = s->dtype;
        g.waves = s->waves;
        g.k_m0 = s->block_m;
        g.k_n0 = s->block_n;
        g.k_k4 = s->block_k4;
        g.head_size = s->head_size;
        const rocke_attn_bwd_opt_grid_t* grids[3]
            = {&s->warp_grid_g02, &s->warp_grid_g13, &s->warp_grid_g4};
        int* dst[3] = {g.grid_g0, g.grid_g1, g.grid_g4};
        for(int i = 0; i < 3; ++i)
        {
            if(!grids[i]->has)
                value_error("'NoneType' object is not iterable");
            dst[i][0] = grids[i]->rows;
            dst[i][1] = grids[i]->cols;
        }
        g.atoms.g02 = atom_k_of(s->atom_g02);
        g.atoms.g13 = atom_k_of(s->atom_g13);
        g.atoms.g4 = atom_k_of(s->atom_g4);
        g.pt_route = s->pt_route;
        g.kv_residency = s->kv_residency;
        g.kt_source = s->kt_source;
        g.transpose_source = s->transpose_source;
        g.ring_depth = s->ring_depth;
        g.global_path = s->global_path;
        g.lds_swizzle = s->has_lds_swizzle ? s->lds_swizzle : NULL;
        g.sched = s->sched;
        g.setprio = truthy(s->setprio);
        g.edge_tiles = truthy(s->edge_tiles);
        g.head_pack = truthy(s->head_pack);
        g.agpr_alloc = s->agpr_alloc;
        g.s_in_agpr = truthy(s->s_in_agpr);
        g.acc_in_lds = truthy(s->acc_in_lds);
        g.dq_mode = s->dq_mode;
        g.agpr_route = s->agpr_route;
        g.stage_vec = s->stage_vec;
        g.asm_tail_nop = s->asm_tail_nop;
        g.asm_cluster = s->asm_cluster;
        g.asm_load_hoist = s->asm_load_hoist;
        g.stage_order = s->stage_order;
        g.loop_sync = s->loop_sync;
        g.dkv_epilogue = epilogue;
        g.dkv_store_vec = s->dkv_store_vec;
        g.qdo_stage = s->qdo_stage;
        g.stats_load = s->stats_load;
        g.dq_issue = s->dq_issue;
        g.interior_clamp = s->interior_clamp;
        g.barrier_merge = s->barrier_merge;
        g.ring_issue = s->ring_issue;
        g.counted_issue = s->counted_issue;
        g.bf16_cast = s->bf16_cast;
        g.bwd_exp2 = s->bwd_exp2;
        g.stats_fma = s->stats_fma;
        g.stats_vec = s->stats_vec;
        g.g4_slicing = s->g4_slicing;
        geometry_check(b, &g);
        *out = g;
    });
}

bool rocke_attn_bwd_tile_geometry_check_in_place(rocke_ir_builder_t* b,
                                                 rocke_attn_bwd_tile_geometry_t* g)
{
    return guarded(b, [&]() { geometry_check(b, g); });
}

int rocke_attn_bwd_tile_geometry_wave_size(const rocke_attn_bwd_tile_geometry_t* g)
{
    rocke_attn_bwd_caps_t c;
    if(rocke_attn_bwd_arch_caps(g->arch, &c, NULL) != ROCKE_OK)
        return 0;
    return c.wave_size;
}

int rocke_attn_bwd_tile_geometry_threads(const rocke_attn_bwd_tile_geometry_t* g)
{
    return g->waves * rocke_attn_bwd_tile_geometry_wave_size(g);
}

const char* rocke_attn_bwd_tile_geometry_reader(const rocke_attn_bwd_tile_geometry_t* g)
{
    return reader_of(g->transpose_source);
}

int rocke_attn_bwd_tile_geometry_g4_slice_buffers(const rocke_attn_bwd_tile_geometry_t* g)
{
    return (eq(g->g4_slicing, "slice_db") || eq(g->g4_slicing, "slice_db_g3")) ? 2 : 1;
}

bool rocke_attn_bwd_tile_geometry_g4_prefetch_before_g3(const rocke_attn_bwd_tile_geometry_t* g)
{
    return eq(g->g4_slicing, "slice_db_g3");
}

int rocke_attn_bwd_tile_geometry_counted_atomics(const rocke_attn_bwd_tile_geometry_t* g)
{
    if(!g->derived)
        return 0;
    return FM(&g->g4) * FN(&g->g4) * CFL(&g->g4);
}

bool rocke_attn_bwd_tile_geometry_stage_table(rocke_ir_builder_t* b,
                                              const rocke_attn_bwd_tile_geometry_t* g,
                                              rocke_attn_bwd_stage_table_t* out)
{
    return guarded(b, [&]() {
        rocke_attn_bwd_sched_geom_t sg = rocke_attn_bwd_sched_geom_empty();
        sg.head_size = g->head_size;
        sg.k_m0 = g->k_m0;
        sg.k_n0 = g->k_n0;
        sg.k_k4 = g->k_k4;
        sg.waves = g->waves;
        sg.has_atoms = true;
        sg.atom_g02 = g->atoms.g02;
        sg.atom_g13 = g->atoms.g13;
        sg.atom_g4 = g->atoms.g4;
        sg.has_grid_g4 = true;
        sg.grid_g4[0] = g->grid_g4[0];
        sg.grid_g4[1] = g->grid_g4[1];
        sg.has_grid_g0 = true;
        sg.grid_g0[0] = g->grid_g0[0];
        sg.grid_g0[1] = g->grid_g0[1];
        sg.has_grid_g1 = true;
        sg.grid_g1[0] = g->grid_g1[0];
        sg.grid_g1[1] = g->grid_g1[1];
        sg.ring_depth = g->ring_depth;
        sg.global_path = g->global_path;
        sg.barrier_merge = g->barrier_merge;
        sg.transpose_source = g->transpose_source;
        sg.kt_source = g->kt_source;
        sg.pt_route = g->pt_route;
        sg.stage_vec = g->stage_vec;
        sg.dq_mode = g->dq_mode;
        sg.sched = g->sched;
        sg.setprio = g->setprio;
        sg.kv_residency = g->kv_residency;
        sg.acc_in_lds = g->acc_in_lds ? 1 : 0;
        sg.stats_load = g->stats_load;
        sg.dq_issue = g->dq_issue;
        sg.has_lds_swizzle = true;
        sg.lds_swizzle = g->lds_swizzle;
        sg.g4_slicing = g->g4_slicing;
        sg.ring_issue = g->ring_issue;
        sg.head_pack = g->head_pack;
        sg.loop_sync = g->loop_sync;
        sg.counted_issue = g->counted_issue;
        rocke_attn_bwd_err_t e;
        need_st(rocke_attn_bwd_stage_table_from_geometry(
                    &sg, g->arch, g->dtype, g->stage_vec, g->dq_mode, g->acc_in_lds, out, &e),
                e);
    });
}

bool rocke_attn_bwd_quad_transpose4(rocke_ir_builder_t* b,
                                    rocke_value_t* const* vals,
                                    int n_vals,
                                    rocke_value_t* lane,
                                    rocke_value_t** out)
{
    return guarded(b, [&]() {
        std::vector<V> v(vals, vals + n_vals);
        std::vector<V> r = quad_transpose4(b, v, lane);
        for(int i = 0; i < 4; ++i)
            out[i] = r[i];
    });
}

bool rocke_attn_bwd_bwd_lds_init(rocke_attn_bwd_bwd_lds_t* lds,
                                 rocke_ir_builder_t* b,
                                 const rocke_attn_bwd_lds_plan_t* plan,
                                 const char* dtype)
{
    return guarded(b, [&]() {
        if(plan->total_bytes % 16)
            value_error("the LDS plan total must be a multiple of 16 bytes");
        lds->plan = plan;
        lds->elem = rocke_attn_bwd_ir_type(b, dtype);
        need(b, lds->elem != NULL);
        int shape[1] = {plan->total_bytes / 2};
        lds->smem = needv(b, rocke_b_smem_alloc(b, lds->elem, shape, 1, "bwd_lds"));
    });
}

bool rocke_attn_bwd_bwd_lds_has(const rocke_attn_bwd_bwd_lds_t* lds, const char* name)
{
    return rocke_attn_bwd_lds_plan_has(lds->plan, name);
}

bool rocke_attn_bwd_bwd_lds_view(rocke_ir_builder_t* b,
                                 const rocke_attn_bwd_bwd_lds_t* lds,
                                 const char* name,
                                 int cols,
                                 int slot,
                                 rocke_attn_bwd_lds_view_t* out)
{
    return guarded(b, [&]() { *out = lds_view(b, lds, name, cols, slot); });
}

rocke_attn_bwd_tile_loader_t rocke_attn_bwd_slab_row_loader(rocke_attn_bwd_slab_row_loader_t* ctx)
{
    rocke_attn_bwd_tile_loader_t l = {slab_load_fn, (void*)ctx};
    return l;
}

rocke_attn_bwd_tile_loader_t
    rocke_attn_bwd_packed_row_loader(rocke_attn_bwd_packed_row_loader_t* ctx)
{
    rocke_attn_bwd_tile_loader_t l = {packed_load_fn, (void*)ctx};
    return l;
}

bool rocke_attn_bwd_copy_rows_to_lds(rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_row_slab_t* slab,
                                     const rocke_attn_bwd_lds_view_t* view,
                                     rocke_value_t* row0,
                                     int rows,
                                     int cols,
                                     rocke_value_t* tid,
                                     int threads,
                                     int vec)
{
    return guarded(b,
                   [&]() { copy_rows_to_lds(b, slab, view, row0, rows, cols, tid, threads, vec); });
}

bool rocke_attn_bwd_copy_packed_rows_to_lds(rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_strided_tensor_addr_t* addr,
                                            const rocke_attn_bwd_lds_view_t* view,
                                            const rocke_attn_bwd_packed_rows_t* packed,
                                            int rows,
                                            int cols,
                                            rocke_value_t* tid,
                                            int threads,
                                            int vec,
                                            bool clamp)
{
    return guarded(b, [&]() {
        rocke_attn_bwd_packed_row_loader_t ctx = {b, addr, *packed, vec, clamp};
        copy_rows(b, rocke_attn_bwd_packed_row_loader(&ctx), view, rows, cols, tid, threads, vec);
    });
}

bool rocke_attn_bwd_copy_row_sets_to_lds(rocke_ir_builder_t* b,
                                         const rocke_attn_bwd_row_set_t* sets,
                                         int n_sets,
                                         const char* order,
                                         int rows,
                                         int cols,
                                         rocke_value_t* tid,
                                         int threads,
                                         rocke_value_t* wave,
                                         int wave_size,
                                         int vec)
{
    return guarded(b, [&]() {
        copy_row_sets(b, sets, n_sets, order, rows, cols, tid, threads, wave, wave_size, vec);
    });
}

bool rocke_attn_bwd_write_xt(rocke_ir_builder_t* b,
                             const rocke_attn_bwd_lds_view_t* src,
                             const rocke_attn_bwd_lds_view_t* dst,
                             int seq,
                             int depth,
                             rocke_value_t* tid,
                             int threads,
                             const rocke_type_t* elem)
{
    return guarded(b, [&]() { write_xt(b, src, dst, seq, depth, tid, threads, elem); });
}

/* ----- StatsView ----- */

bool rocke_attn_bwd_stats_view_init(rocke_attn_bwd_stats_view_t* s,
                                    rocke_ir_builder_t* b,
                                    const rocke_attn_bwd_workspace_addr_t* lse2,
                                    const rocke_attn_bwd_workspace_addr_t* dsum,
                                    rocke_value_t* n_heads,
                                    const char* bounds,
                                    int span)
{
    return guarded(b, [&]() {
        if(lse2->width != 1 || dsum->width != 1)
            value_error("lse2 / Dsum workspaces are one fp32 per row");
        if(!eq(bounds, "clamp") && !eq(bounds, "buffer"))
            value_error("stats bounds=" + repr(bounds) + " not in ('clamp', 'buffer')");
        if(!eq(bounds, "clamp") && span < 1)
            value_error("buffer-bounded stats need the q-step span (kM0)");
        s->lse2 = *lse2;
        s->dsum = *dsum;
        s->n_heads = n_heads;
        s->bounds = eq(bounds, "clamp") ? ROCKE_ATTN_BWD_STATS_CLAMP : ROCKE_ATTN_BWD_STATS_BUFFER;
        s->span = span;
    });
}

static void put_pairs(const StatPairs& p, rocke_value_t** out_lse2, rocke_value_t** out_dsum)
{
    for(size_t i = 0; i < p.lse2.size(); ++i)
    {
        out_lse2[i] = p.lse2[i];
        out_dsum[i] = p.dsum[i];
    }
}

bool rocke_attn_bwd_stats_view_rows(const rocke_attn_bwd_stats_view_t* s,
                                    rocke_ir_builder_t* b,
                                    rocke_value_t* h,
                                    rocke_value_t* q0,
                                    rocke_value_t* const* rows,
                                    int n_rows,
                                    rocke_value_t** out_lse2,
                                    rocke_value_t** out_dsum)
{
    return guarded(b, [&]() {
        put_pairs(stats_rows(s, b, h, q0, std::vector<V>(rows, rows + n_rows)), out_lse2, out_dsum);
    });
}

bool rocke_attn_bwd_stats_view_batched_rows(const rocke_attn_bwd_stats_view_t* s,
                                            rocke_ir_builder_t* b,
                                            rocke_value_t* h,
                                            rocke_value_t* q0,
                                            rocke_value_t* const* rows,
                                            int n_rows,
                                            int width,
                                            rocke_value_t** out_lse2,
                                            rocke_value_t** out_dsum)
{
    return guarded(b, [&]() {
        put_pairs(stats_batched_rows(s, b, h, q0, std::vector<V>(rows, rows + n_rows), width),
                  out_lse2,
                  out_dsum);
    });
}

bool rocke_attn_bwd_stats_view_packed_rows(const rocke_attn_bwd_stats_view_t* s,
                                           rocke_ir_builder_t* b,
                                           const rocke_attn_bwd_packed_rows_t* packed,
                                           rocke_value_t* const* rows,
                                           int n_rows,
                                           rocke_value_t** out_lse2,
                                           rocke_value_t** out_dsum)
{
    return guarded(b, [&]() {
        put_pairs(stats_packed_rows(s, b, packed, std::vector<V>(rows, rows + n_rows)),
                  out_lse2,
                  out_dsum);
    });
}

bool rocke_attn_bwd_stats_view_raw(const rocke_attn_bwd_stats_view_t* s,
                                   rocke_ir_builder_t* b,
                                   rocke_value_t* h,
                                   rocke_value_t* q0,
                                   rocke_value_t* const* rows,
                                   int n_rows,
                                   const rocke_attn_bwd_packed_rows_t* packed,
                                   rocke_value_t** out_lse2,
                                   rocke_value_t** out_dsum)
{
    return guarded(b, [&]() {
        put_pairs(stats_raw(s, b, h, q0, std::vector<V>(rows, rows + n_rows), packed),
                  out_lse2,
                  out_dsum);
    });
}

bool rocke_attn_bwd_stats_view_finish(const rocke_attn_bwd_stats_view_t* s,
                                      rocke_ir_builder_t* b,
                                      rocke_value_t* h,
                                      rocke_value_t* q0,
                                      rocke_value_t* const* rows,
                                      int n_rows,
                                      rocke_value_t* const* raw_lse2,
                                      rocke_value_t* const* raw_dsum,
                                      const rocke_attn_bwd_packed_rows_t* packed,
                                      rocke_value_t** out_lse2,
                                      rocke_value_t** out_dsum)
{
    return guarded(b, [&]() {
        StatPairs raw;
        raw.lse2.assign(raw_lse2, raw_lse2 + n_rows);
        raw.dsum.assign(raw_dsum, raw_dsum + n_rows);
        put_pairs(stats_finish(s, b, h, q0, std::vector<V>(rows, rows + n_rows), raw, packed),
                  out_lse2,
                  out_dsum);
    });
}

/* ----- DkDvAcc ----- */

int rocke_attn_bwd_dkdv_acc_flat_len(const rocke_attn_bwd_dkdv_acc_t* acc)
{
    return 2 * acc->frags_m * acc->frags_n;
}

void rocke_attn_bwd_dkdv_acc_flat(const rocke_attn_bwd_dkdv_acc_t* acc, rocke_value_t** out)
{
    int n = acc->frags_m * acc->frags_n;
    for(int i = 0; i < n; ++i)
        out[i] = acc->dk[i];
    for(int i = 0; i < n; ++i)
        out[n + i] = acc->dv[i];
}

bool rocke_attn_bwd_dkdv_acc_from_flat(rocke_ir_builder_t* b,
                                       rocke_value_t* const* values,
                                       int n_values,
                                       int frags_m,
                                       int frags_n,
                                       rocke_attn_bwd_dkdv_acc_t* out)
{
    return guarded(b, [&]() {
        int n = frags_m * frags_n;
        if(n_values != 2 * n)
            value_error("accumulator count does not match the geometry");
        out->dk = arena<V>(b, n);
        out->dv = arena<V>(b, n);
        for(int i = 0; i < n; ++i)
        {
            out->dk[i] = values[i];
            out->dv[i] = values[n + i];
        }
        out->frags_m = frags_m;
        out->frags_n = frags_n;
    });
}

/* ----- BwdTileStep ----- */

bool rocke_attn_bwd_tile_step_init(rocke_attn_bwd_tile_step_t* t,
                                   rocke_ir_builder_t* b,
                                   const rocke_attn_bwd_tile_geometry_t* geom,
                                   rocke_attn_bwd_mask_policy_t* mask,
                                   const rocke_attn_bwd_stats_view_t* stats,
                                   rocke_attn_bwd_dq_sink_t* dq_sink,
                                   const rocke_attn_bwd_bwd_lds_t* lds,
                                   rocke_value_t* tid,
                                   rocke_value_t* lane,
                                   rocke_value_t* wave,
                                   rocke_value_t* scale_log2,
                                   rocke_value_t* ds_mult,
                                   bool sentinel)
{
    return guarded(b, [&]() {
        step_init(
            b, t, geom, mask, stats, dq_sink, lds, tid, lane, wave, scale_log2, ds_mult, sentinel);
    });
}

void rocke_attn_bwd_tile_step_free(rocke_attn_bwd_tile_step_t* t)
{
    if(t->has_waits)
        rocke_attn_bwd_counted_waits_free(&t->waits);
    t->has_waits = false;
}

bool rocke_attn_bwd_tile_step_stage_kv(rocke_attn_bwd_tile_step_t* t,
                                       rocke_ir_builder_t* b,
                                       const rocke_attn_bwd_row_slab_t* k_slab,
                                       const rocke_attn_bwd_row_slab_t* v_slab,
                                       rocke_value_t* k0,
                                       rocke_value_t* live_)
{
    return guarded(b, [&]() { stage_kv(b, t, k_slab, v_slab, k0, live_); });
}

bool rocke_attn_bwd_tile_step_stage_kv_rows(rocke_attn_bwd_tile_step_t* t,
                                            rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_row_source_t* k_rows,
                                            const rocke_attn_bwd_row_source_t* v_rows,
                                            rocke_value_t* k0,
                                            rocke_value_t* live_)
{
    return guarded(b,
                   [&]() { stage_kv_with(b, t, k_rows, v_rows, k0, live_, copy_source_to_lds); });
}

bool rocke_attn_bwd_tile_step_zero_acc(rocke_attn_bwd_tile_step_t* t,
                                       rocke_ir_builder_t* b,
                                       rocke_attn_bwd_dkdv_acc_t* out)
{
    return guarded(b, [&]() { *out = zero_acc_step(b, t); });
}

int rocke_attn_bwd_tile_step_counted_n_stats(const rocke_attn_bwd_tile_step_t* t)
{
    const rocke_attn_bwd_block_gemm_t* g0 = &t->geom->g0;
    return 2 * FM(g0) * CFL(g0);
}

int rocke_attn_bwd_tile_step_counted_zero_len(const rocke_attn_bwd_tile_step_t* t)
{
    const Geom* g = t->geom;
    int threads = rocke_attn_bwd_tile_geometry_threads(g);
    int x = g->k_m0 * g->head_size / 8;
    int n_chunks = threads ? (x + threads - 1) / threads : 0;
    return rocke_attn_bwd_tile_step_counted_n_stats(t) + 2 * n_chunks;
}

bool rocke_attn_bwd_tile_step_counted_zero(rocke_attn_bwd_tile_step_t* t,
                                           rocke_ir_builder_t* b,
                                           rocke_value_t** out)
{
    return guarded(b, [&]() {
        const Geom* g = t->geom;
        int threads = threads_of(g);
        if(threads == 0)
            value_error("integer division or modulo by zero");
        int n_stats = rocke_attn_bwd_tile_step_counted_n_stats(t);
        int n = rocke_attn_bwd_tile_step_counted_zero_len(t);
        const rocke_type_t* v8 = rocke_vector_type(b, t->elem, 8);
        std::vector<const rocke_type_t*> types(n);
        for(int i = 0; i < n; ++i)
            types[i] = i < n_stats ? rocke_f32() : v8;
        need(b, rocke_attn_bwd_zero_like(b, types.data(), n, out));
    });
}

bool rocke_attn_bwd_tile_step_counted_loads(rocke_attn_bwd_tile_step_t* t,
                                            rocke_ir_builder_t* b,
                                            rocke_value_t* h,
                                            rocke_value_t* q0,
                                            const rocke_attn_bwd_row_slab_t* q_slab,
                                            const rocke_attn_bwd_row_slab_t* do_slab,
                                            const rocke_attn_bwd_packed_step_t* packed,
                                            bool edge,
                                            rocke_attn_bwd_staged_step_t* out)
{
    return guarded(b, [&]() { *out = counted_loads(b, t, h, q0, q_slab, do_slab, packed, edge); });
}

rocke_attn_bwd_q_step_opts_t rocke_attn_bwd_q_step_opts_default(void)
{
    rocke_attn_bwd_q_step_opts_t o;
    memset(&o, 0, sizeof(o));
    o.edge = true;
    return o;
}

bool rocke_attn_bwd_tile_step_run_q_step(rocke_attn_bwd_tile_step_t* t,
                                         rocke_ir_builder_t* b,
                                         const rocke_attn_bwd_dkdv_acc_t* acc,
                                         const rocke_attn_bwd_row_slab_t* q_slab,
                                         const rocke_attn_bwd_row_slab_t* do_slab,
                                         rocke_value_t* qt,
                                         rocke_value_t* h,
                                         rocke_value_t* k0,
                                         const rocke_attn_bwd_q_step_opts_t* opts,
                                         rocke_attn_bwd_q_step_result_t* out)
{
    return guarded(b, [&]() {
        rocke_attn_bwd_q_step_opts_t o = opts ? *opts : rocke_attn_bwd_q_step_opts_default();
        if(!t->geom->acc_in_lds)
            need_reg_frags(acc, t);
        run_q_step(b, t, acc, q_slab, do_slab, qt, h, k0, o, out);
    });
}

bool rocke_attn_bwd_tile_step_zero_pending_dq(rocke_attn_bwd_tile_step_t* t,
                                              rocke_ir_builder_t* b,
                                              rocke_value_t* h,
                                              rocke_value_t* n_heads,
                                              rocke_value_t* rows_per_head,
                                              rocke_attn_bwd_pending_dq_t* out)
{
    return guarded(b, [&]() {
        V zero = rocke_b_const_f32(b, 0.0);
        int n = dq_atomics_per_lane(t);
        bool packed = n_heads != NULL;
        V* vals = arena<V>(b, n);
        for(int i = 0; i < n; ++i)
            vals[i] = zero;
        out->values = vals;
        out->n_values = n;
        out->q0 = needv(b, ci(b, 0));
        out->h = h;
        out->n_heads = packed ? n_heads : NULL;
        out->rows_per_head = packed ? rows_per_head : NULL;
    });
}

bool rocke_attn_bwd_tile_step_flush_dq(rocke_attn_bwd_tile_step_t* t,
                                       rocke_ir_builder_t* b,
                                       const rocke_attn_bwd_pending_dq_t* pending)
{
    return guarded(b, [&]() { flush_dq(b, t, pending); });
}

bool rocke_attn_bwd_tile_step_ring_prologue(rocke_attn_bwd_tile_step_t* t,
                                            rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_q_step_inputs_t* step)
{
    return guarded(b, [&]() {
        RingViews views = ring_views(b, t, NULL);
        ring_stores(b, ring_loads(b, t, step, views));
    });
}

bool rocke_attn_bwd_tile_step_ring_publish(rocke_attn_bwd_tile_step_t* t, rocke_ir_builder_t* b)
{
    return guarded(b, [&]() { loop_barrier(b, t); });
}

bool rocke_attn_bwd_tile_step_finish(rocke_attn_bwd_tile_step_t* t,
                                     rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_dkdv_acc_t* acc,
                                     rocke_attn_bwd_dkdv_acc_t* out)
{
    return guarded(b, [&]() {
        const Geom* g = t->geom;
        if(eq(g->barrier_merge, "on"))
            sync(b);
        if(eq(g->agpr_route, "agpr_fence") && !g->acc_in_lds)
        {
            need_reg_frags(acc, t);
            int n = acc->frags_m * acc->frags_n;
            V* dk = arena<V>(b, n);
            V* dv = arena<V>(b, n);
            for(int i = 0; i < n; ++i)
                dk[i] = empty_asm(b, acc->dk[i], "=v,0", acc->dk[i]->type);
            for(int i = 0; i < n; ++i)
                dv[i] = empty_asm(b, acc->dv[i], "=v,0", acc->dv[i]->type);
            out->dk = dk;
            out->dv = dv;
            out->frags_m = acc->frags_m;
            out->frags_n = acc->frags_n;
            return;
        }
        *out = *acc;
    });
}

static void put_rows(B* b,
                     const std::vector<rocke_attn_bwd_dkv_row_t>& rows,
                     rocke_attn_bwd_dkv_row_t** out,
                     int* n_out)
{
    rocke_attn_bwd_dkv_row_t* a = arena<rocke_attn_bwd_dkv_row_t>(b, rows.size());
    for(size_t i = 0; i < rows.size(); ++i)
        a[i] = rows[i];
    *out = a;
    *n_out = (int)rows.size();
}

bool rocke_attn_bwd_tile_step_dkv_row_slots(rocke_attn_bwd_tile_step_t* t,
                                            rocke_ir_builder_t* b,
                                            const char* which,
                                            const rocke_attn_bwd_dkdv_acc_t* acc,
                                            rocke_attn_bwd_dkv_row_t** out,
                                            int* n_out)
{
    return guarded(b, [&]() { put_rows(b, dkv_row_slots(b, t, which, acc), out, n_out); });
}

bool rocke_attn_bwd_tile_step_dkv_cells(rocke_attn_bwd_tile_step_t* t,
                                        rocke_ir_builder_t* b,
                                        const char* which,
                                        const rocke_attn_bwd_dkdv_acc_t* acc,
                                        rocke_value_t* mult,
                                        rocke_attn_bwd_dkv_row_t** out,
                                        int* n_out)
{
    return guarded(b, [&]() { put_rows(b, dkv_cells(b, t, which, acc, mult), out, n_out); });
}

bool rocke_attn_bwd_tile_step_dkv_rows(rocke_attn_bwd_tile_step_t* t,
                                       rocke_ir_builder_t* b,
                                       const char* which,
                                       const rocke_attn_bwd_dkdv_acc_t* acc,
                                       rocke_value_t* mult,
                                       int vec,
                                       int slot,
                                       bool do_sync,
                                       rocke_attn_bwd_dkv_reader_t* out)
{
    return guarded(b, [&]() { *out = dkv_rows(b, t, which, acc, mult, vec, slot, do_sync); });
}

rocke_value_t* rocke_attn_bwd_dkv_reader_read(rocke_ir_builder_t* b,
                                              const rocke_attn_bwd_dkv_reader_t* rd,
                                              rocke_value_t* r,
                                              rocke_value_t* c)
{
    V out = NULL;
    if(!guarded(b, [&]() { out = dkv_read(b, rd, r, c); }))
        return NULL;
    return out;
}

bool rocke_attn_bwd_tile_step_dkv_quad_rows(rocke_attn_bwd_tile_step_t* t,
                                            rocke_ir_builder_t* b,
                                            const char* which,
                                            const rocke_attn_bwd_dkdv_acc_t* acc,
                                            rocke_value_t* mult,
                                            rocke_attn_bwd_dkv_quad_t** out,
                                            int* n_out)
{
    return guarded(b, [&]() {
        std::vector<rocke_attn_bwd_dkv_quad_t> q = dkv_quad_rows(b, t, which, acc, mult);
        rocke_attn_bwd_dkv_quad_t* a = arena<rocke_attn_bwd_dkv_quad_t>(b, q.size());
        for(size_t i = 0; i < q.size(); ++i)
            a[i] = q[i];
        *out = a;
        *n_out = (int)q.size();
    });
}

bool rocke_attn_bwd_tile_step_dkv_f32_image(rocke_attn_bwd_tile_step_t* t,
                                            rocke_ir_builder_t* b,
                                            const char* which,
                                            rocke_attn_bwd_lds_view_t* out)
{
    return guarded(b, [&]() {
        if(!t->geom->acc_in_lds)
            value_error("dkv_f32_image needs LDS accumulators");
        check_which(which);
        *out = *acc_view_which(t, which);
    });
}

bool rocke_attn_bwd_tile_step_dkv_f32_part(rocke_attn_bwd_tile_step_t* t,
                                           rocke_ir_builder_t* b,
                                           const char* which,
                                           const rocke_attn_bwd_dkdv_acc_t* acc,
                                           rocke_value_t* mult,
                                           int half,
                                           int slot,
                                           rocke_attn_bwd_lds_view_t* out)
{
    return guarded(b, [&]() { *out = dkv_f32_part(b, t, which, acc, mult, half, slot); });
}

} /* extern "C" */
