// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/attention_bwd_unified.py: the unified
 * (paged) SDPA backward main kernel (rocke.attn_bwd_unified.v1), its spec,
 * validation and kernargs (rocke/instance_attention_bwd_unified.h).
 *
 * The kernel is the general family's tile step and q loops (_kv_tile of
 * attention_bwd_main.cpp, through rocke/instance_attention_bwd_main_internal.h)
 * with the paged K / V slabs of _attention_bwd_paged and the page atomic /
 * direct dK / dV epilogues.
 *
 * Byte identity: every builder call (constants included) is bound to a local
 * in Python's left-to-right, inner-first evaluation order, and every barrier
 * carries ROCKE_ATTN_BWD_BARRIER_FENCE (the Python builder's fence_barriers).
 * A Python raise is a ckc::Error throw with the same message; the entry point
 * catches it under ckc::guard_builder.
 */
#include "rocke/instance_attention_bwd_unified.h"

#include <string.h>

#include <memory>
#include <string>
#include <vector>

#include "rocke/error.hpp"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.attention_band.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_band.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_body.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_frag.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_gemm.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_paged.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_sinks.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_worklist.h"
#include "rocke/instance_attention_bwd_main_internal.h"
#include "rocke/ir_internal.h"

namespace rocke_attn_bwd
{
namespace
{

using namespace main_kernel;
using namespace main_kernel::plumbing;

typedef rocke_attn_bwd_unified_spec_t U;
typedef std::vector<std::string> Strs;

const int NONE = ROCKE_ATTN_BWD_NONE;
const int PAGE_GRANULE = ROCKE_ATTN_BWD_UNIFIED_PAGE_GRANULE;
/* The lowering's flat-work-group-size attribute has a fixed minimum of 64. */
const int MIN_WG_ATTR = 64;

const Strs BWD_DTYPES = {"fp16", "bf16"};
const std::vector<int> BWD_HEAD_SIZES = {32, 64, 128};
const Strs PAGE_MODES = {"atomic", "direct"};
const Strs MASK_CLASSES = {"band", "none"};
const std::vector<int> STAGE_VECS = {8, 1};

std::string s_(const char* s)
{
    return s ? s : "";
}

std::string repr_s(const char* s)
{
    return s == NULL ? std::string("None") : py_repr(s);
}

std::string repr_b(bool v)
{
    return v ? "True" : "False";
}

/* _check_in(name, value, allowed) */
void check_in(const std::string& name, const char* value, const Strs& allowed)
{
    for(const std::string& a : allowed)
        if(value != NULL && a == value)
            return;
    throw ckc::ValueError(name + "=" + repr_s(value) + " not in " + py_tuple(allowed));
}

void check_in(const std::string& name, int value, const std::vector<int>& allowed)
{
    for(int a : allowed)
        if(a == value)
            return;
    throw ckc::ValueError(name + "=" + std::to_string(value) + " not in " + py_tuple(allowed));
}

/* general_spec() before AttnBwdSpec.__post_init__ */
rocke_attn_bwd_spec_t general_raw(const U& u)
{
    rocke_attn_bwd_spec_t g = u.knobs;
    const rocke_attn_bwd_spec_t d = rocke_attn_bwd_spec_default();
    g.head_size = u.head_size;
    g.dtype = u.dtype;
    g.seq_mode = "thd";
    g.dkv_mode = eq(u.page_mode, "atomic") ? "atomic" : "direct";
    g.stage_vec = u.stage_vec;
    g.mask_class = u.mask_class;
    g.dq_mode = d.dq_mode;
    g.name = u.name;
    return g;
}

/* general_spec() */
rocke_attn_bwd_spec_t general_spec(const U& u)
{
    rocke_attn_bwd_spec_t g = general_raw(u);
    spec_init(g);
    return g;
}

/* AttnBwdUnifiedSpec.__post_init__ */
void post_init(U& u)
{
    check_in("head_size", u.head_size, BWD_HEAD_SIZES);
    check_in("dtype", u.dtype, BWD_DTYPES);
    check_in("page_mode", u.page_mode, PAGE_MODES);
    check_in("mask_class", u.mask_class, MASK_CLASSES);
    check_in("stage_vec", u.stage_vec, STAGE_VECS);
    int sp = u.sub_page;
    if(sp != NONE && (sp < PAGE_GRANULE || sp % PAGE_GRANULE))
        throw ckc::ValueError("sub_page=" + std::to_string(sp)
                              + " must be None or a positive multiple of "
                              + std::to_string(PAGE_GRANULE));
    if(!rocke_attn_bwd::name_ok(u.name))
        throw ckc::ValueError("name=" + repr_s(u.name) + " must be a non-empty identifier");
    /* Knob values and arch-free couplings: the general spec's rules (it also
     * canonicalises tuples and the swizzle string). */
    u.knobs = general_spec(u);
}

std::string kernel_name(const U& u)
{
    std::string base = s_(u.name) + "_main_" + s_(u.dtype) + "_d" + std::to_string(u.head_size)
                       + "_" + s_(u.page_mode) + "_" + s_(u.mask_class) + "_sv"
                       + std::to_string(u.stage_vec);
    if(u.sub_page != NONE)
        base += "_sp" + std::to_string(u.sub_page);
    /* the unified knobs are the general spec's fields other than its problem
     * keys (_NOT_KNOBS) */
    static const Strs not_knobs
        = {"dtype", "seq_mode", "dkv_mode", "stage_vec", "mask_class", "dq_mode"};
    return salted_kernel_name(base, general_raw(u), not_knobs);
}

/* validate_attn_bwd_unified_spec(spec, arch); *g_out the resolved general
 * spec. */
U validate(const U& spec, const std::string& arch, rocke_attn_bwd_spec_t* g_out)
{
    rocke_attn_bwd_spec_t g = validate_spec(general_spec(spec), arch);
    U s = spec;
    s.knobs = g;
    const rocke_attn_bwd_spec_t& k = s.knobs;
    auto shell = [](const std::string& name, const std::string& value) {
        throw ckc::ValueError(name + "=" + value
                              + ": a knob of the general main-kernel shell (CTA decode, "
                                "kernarg attributes or strided dK / dV epilogue), not built "
                                "for the unified family");
    };
    if(k.xcd_remap)
        shell("xcd_remap", repr_b(k.xcd_remap));
    if(k.lb_interleave)
        shell("lb_interleave", repr_b(k.lb_interleave));
    if(k.ptr_noalias != NULL)
        shell("ptr_noalias", py_repr(k.ptr_noalias));
    if(k.dkv_slab_hoist)
        shell("dkv_slab_hoist", repr_b(k.dkv_slab_hoist));
    if(k.lb_mirror)
        shell("lb_mirror", repr_b(k.lb_mirror));
    if(!eq(k.dkv_sink, "atomic"))
        shell("dkv_sink", repr_s(k.dkv_sink));
    if(!eq(k.ws_bounds, "clamp"))
        shell("ws_bounds", repr_s(k.ws_bounds));
    if(k.dkv_cell_vec != 1)
        shell("dkv_cell_vec", std::to_string(k.dkv_cell_vec));
    if(k.narrow_vec != 1)
        shell("narrow_vec", std::to_string(k.narrow_vec));
    if(k.slab_carry)
        shell("slab_carry", repr_b(k.slab_carry));
    if((k.ring_depth != 1 && !eq(k.barrier_merge, "on")) || !eq(k.global_path, "vgpr"))
        /* the general shell builds the Q / dO prefetch ring (QRing) and hands
         * it to the kv-tile body; the unified shell does not */
        throw ckc::ValueError("ring_depth=" + std::to_string(k.ring_depth)
                              + ", global_path=" + repr_s(k.global_path)
                              + ": the prefetch ring of the general main-kernel shell is not "
                                "built for the unified family");
    if(s.sub_page != NONE)
    {
        if(k.block_n % s.sub_page || s.sub_page >= k.block_n)
            throw ckc::ValueError("sub_page=" + std::to_string(s.sub_page)
                                  + " must divide block_n=" + std::to_string(k.block_n)
                                  + " and be below it (otherwise the page is the 'multi' class)");
    }
    if(g_out != NULL)
        *g_out = g;
    return s;
}

/* ---------------------------------------------------------------- kernargs */

const char* const MAIN_PTRS[] = {"Q",
                                 "KC",
                                 "VC",
                                 "dO",
                                 "DKC",
                                 "DVC",
                                 "WS_LSE2",
                                 "WS_DSUM",
                                 "WS_DQ",
                                 "WS_PDK",
                                 "WS_PDV",
                                 "CU_SEQLENS_Q",
                                 "SEQ_LENS_KV",
                                 "BLOCK_TABLE",
                                 "WORKLIST"};
const char* const MAIN_F32[] = {"scale_log2", "ds_mult", "dk_mult", "dq_mult"};
const char* const MAIN_I32[] = {
    "q_t",        "q_h",         "do_t",     "do_h",         "kc_blk",        "kc_tok",
    "kc_h",       "vc_blk",      "vc_tok",   "vc_h",         "bt_stride",     "h_q",
    "h_k",        "G",           "num_seqs", "max_seqlen_q", "max_seqlen_kv", "max_blocks_per_seq",
    "page",       "left",        "right",    "bottom_right", "ws_rows_q",     "page_rows",
    "g_split",    "n_kv_tiles",  "lb_order", "pair",         "xcd_n",         "xcd_chunk",
    "pack_heads", "use_worklist"};
const char* const MAIN_READONLY[] = {"Q",
                                     "KC",
                                     "VC",
                                     "dO",
                                     "WS_LSE2",
                                     "WS_DSUM",
                                     "CU_SEQLENS_Q",
                                     "SEQ_LENS_KV",
                                     "BLOCK_TABLE",
                                     "WORKLIST"};
const char* const IO_PTRS[] = {"Q", "KC", "VC", "dO", "DKC", "DVC"};

void main_params(std::vector<const char*>* names, std::vector<const char*>* formats)
{
    for(const char* p : MAIN_PTRS)
    {
        names->push_back(p);
        formats->push_back("Q");
    }
    for(const char* p : MAIN_F32)
    {
        names->push_back(p);
        formats->push_back("f");
    }
    for(const char* p : MAIN_I32)
    {
        names->push_back(p);
        formats->push_back("i");
    }
}

/* _declare_main(b, io) */
Kargs declare_main(B* b, const rocke_type_t* io)
{
    std::vector<const char*> names, formats;
    main_params(&names, &formats);
    Kargs a;
    for(size_t i = 0; i < names.size(); ++i)
    {
        const char* name = names[i];
        const char* f = formats[i];
        V v;
        if(eq(f, "Q"))
        {
            const rocke_type_t* pointee;
            bool is_io = false;
            for(const char* x : IO_PTRS)
                is_io = is_io || eq(name, x);
            if(is_io)
                pointee = io;
            else if(strncmp(name, "WS_", 3) == 0)
                pointee = rocke_f32();
            else
                pointee = rocke_i32();
            bool ro = false;
            for(const char* x : MAIN_READONLY)
                ro = ro || eq(name, x);
            rocke_param_opts_t opts;
            memset(&opts, 0, sizeof(opts));
            opts.readonly = true;
            opts.readonly_set = true;
            v = rocke_b_param(b, name, rocke_ptr_type(b, pointee, "global"), ro ? &opts : NULL);
        }
        else if(eq(f, "f"))
            v = rocke_b_param(b, name, rocke_f32(), NULL);
        else
            v = rocke_b_param(b, name, rocke_i32(), NULL);
        needv(b, v);
        a.order.push_back(name);
        a.m[name] = v;
    }
    return a;
}

/* ------------------------------------------------------------- the kernel */

/* A PagedKv released on scope exit. */
struct PagedGuard
{
    rocke_attn_bwd_paged_kv_t p;
    bool live = false;
    PagedGuard() = default;
    PagedGuard(const PagedGuard&) = delete;
    PagedGuard& operator=(const PagedGuard&) = delete;
    ~PagedGuard()
    {
        if(live)
            rocke_attn_bwd_paged_kv_free(&p);
    }
};

/* PagedSlab.load_vec as a duck-typed K / V slab of stage_kv */
V paged_load_vec(void* ctx, B* b, V row, IV col0, int n)
{
    (void)b;
    return rocke_attn_bwd_paged_slab_load_vec(
        (const rocke_attn_bwd_paged_slab_t*)ctx, row, col0, n);
}

struct Unified
{
    Main& m;
    const U& u;
    const Kargs& a;
    Cta& c;
    rocke_attn_bwd_paged_kv_t* paged;
    rocke_attn_bwd_paged_cache_rows_t k_cache, v_cache;
    rocke_attn_bwd_paged_cache_rows_t out[2]; /* direct page mode */
    rocke_attn_bwd_paged_slab_t kv_slabs[2]; /* _kv_tile's k_slab, v_slab */

    /* PagedCacheRows.slab(hk, k0, clamp=) */
    rocke_attn_bwd_paged_slab_t
        slab(const rocke_attn_bwd_paged_cache_rows_t& cache, V k0, bool clamp)
    {
        rocke_attn_bwd_paged_slab_t sl;
        need(m.b, rocke_attn_bwd_paged_cache_rows_slab(&cache, ivv(c.hk), k0, clamp, &sl));
        return sl;
    }

    const rocke_attn_bwd_paged_kv_tile_t* tile(V k0)
    {
        const rocke_attn_bwd_paged_kv_tile_t* t = rocke_attn_bwd_paged_kv_tile(paged, k0);
        need(m.b, t != NULL);
        return t;
    }

    /* _kv_tile's k_addr.slab(hk, k0) / v_addr.slab(hv, k0) (k_addr=k_cache,
     * v_addr=v_cache): a PagedSlab, a row source of step.stage_kv */
    rocke_attn_bwd_row_source_t kv_rows(int which, IV h, IV k0, bool clamp)
    {
        int i = which == SLAB_K ? 0 : 1;
        need(m.b,
             rocke_attn_bwd_paged_cache_rows_slab(
                 i == 0 ? &k_cache : &v_cache, h, k0.v, clamp, &kv_slabs[i]));
        rocke_attn_bwd_row_source_t src = {paged_load_vec, &kv_slabs[i]};
        return src;
    }

    /* _page_atomic_epilogue */
    void atomic_epilogue(const rocke_attn_bwd_dkdv_acc_t& acc, V k0, V live_)
    {
        B* b = m.b;
        const rocke_attn_bwd_paged_kv_tile_t* t = tile(k0);
        bool select = eq(m.s.dkv_atomic_guard, "select");
        IfScope live_if(b, live_);
        struct Out
        {
            const char* which;
            const char* ptr;
            V mult;
        };
        Out outs[2] = {{"dk", "WS_PDK", a["dk_mult"]}, {"dv", "WS_PDV", NULL}};
        for(const Out& o : outs)
        {
            rocke_attn_bwd_paged_ws_row_t w;
            need(b,
                 rocke_attn_bwd_paged_kv_tile_workspace_row(
                     t, a[o.ptr], ivv(a["page_rows"]), u.head_size, ivv(c.hk), &w));
            rocke_attn_bwd_dkv_row_t* rows;
            int n_rows;
            need(b,
                 rocke_attn_bwd_tile_step_dkv_row_slots(&c.step, b, o.which, &acc, &rows, &n_rows));
            for(int r = 0; r < n_rows; ++r)
            {
                const rocke_attn_bwd_dkv_row_t& row = rows[r];
                std::vector<IV> xs;
                std::vector<V> vs;
                for(int i = 0; i < row.n_items; ++i)
                {
                    xs.push_back(ivv(row.cols[i]));
                    vs.push_back(o.mult != NULL ? needv(b, rocke_b_fmul(b, row.values[i], o.mult))
                                                : row.values[i]);
                }
                V kr = rocke_b_add(b, k0, row.row);
                V in_seq = rocke_b_cmp_lt(b, kr, paged->seq.length);
                rocke_attn_bwd_ws_slab_t sl;
                V rr;
                need(b, rocke_attn_bwd_paged_ws_row_slab(&w, row.row, &sl, &rr));
                if(select)
                    rocke_attn_bwd_ws_slab_atomic_add_row_select(
                        &sl, rr, xs.data(), vs.data(), (int)vs.size(), in_seq);
                else
                    rocke_attn_bwd_ws_slab_atomic_add_row(
                        &sl, rr, xs.data(), vs.data(), (int)vs.size(), in_seq);
                need(b, true);
            }
        }
    }

    /* store_rows(dst, read) of _page_direct_epilogue */
    void store_rows(const rocke_attn_bwd_paged_kv_tile_t* t,
                    const rocke_attn_bwd_paged_slab_t& dst,
                    const rocke_attn_bwd_dkv_reader_t& rd,
                    V k0,
                    V zero)
    {
        B* b = m.b;
        int vec = m.s.dkv_store_vec;
        int D = u.head_size, threads = threads_of(m);
        int per_row = D / vec;
        int total = m.s.block_n * per_row;
        int rounds = rocke_attn_bwd_deal_chunks_count(b, total, threads);
        need(b, rounds >= 0);
        for(int it = 0; it < rounds; ++it)
        {
            bool tail = false;
            V chunk
                = needv(b, rocke_attn_bwd_deal_chunks_round(b, c.tid, total, threads, it, &tail));
            V r, col;
            rocke_attn_bwd_chunk_coords(b, chunk, per_row, vec, &r, &col);
            V k = rocke_b_add(b, k0, r);
            V store, row_live;
            rocke_attn_bwd_paged_kv_tile_store_rule(t, k, &store, &row_live);
            if(tail)
            {
                V ct = ci(b, total);
                V lt = rocke_b_cmp_lt(b, chunk, ct);
                store = rocke_b_land(b, store, lt);
            }
            IfScope f(b, needv(b, store));
            V rv = needv(b, rocke_attn_bwd_dkv_reader_read(b, &rd, r, col));
            V v = rocke_b_select(b, row_live, rv, zero);
            rocke_attn_bwd_paged_slab_store_vec(&dst, k, ivv(col), v);
            need(b, true);
        }
    }

    /* _page_direct_epilogue */
    void direct_epilogue(const rocke_attn_bwd_dkdv_acc_t& acc, V k0)
    {
        B* b = m.b;
        const rocke_attn_bwd_spec_t& s = m.s;
        const rocke_attn_bwd_paged_kv_tile_t* t = tile(k0);
        struct Out
        {
            const char* which;
            const rocke_attn_bwd_paged_cache_rows_t* cache;
            V mult;
        };
        Out outs[2] = {{"dk", &out[0], a["dk_mult"]}, {"dv", &out[1], NULL}};
        if(eq(s.dkv_epilogue, "direct"))
        {
            V zv = rocke_b_zero_vec(b, m.io, 1);
            V e0 = needv(b, rocke_b_vec_extract(b, zv, 0));
            for(const Out& o : outs)
            {
                rocke_attn_bwd_paged_slab_t dst = slab(*o.cache, k0, false);
                rocke_attn_bwd_dkv_row_t* cells;
                int n_cells;
                need(b,
                     rocke_attn_bwd_tile_step_dkv_cells(
                         &c.step, b, o.which, &acc, o.mult, &cells, &n_cells));
                for(int i = 0; i < n_cells; ++i)
                {
                    V k = rocke_b_add(b, k0, cells[i].row);
                    V store, row_live;
                    rocke_attn_bwd_paged_kv_tile_store_rule(t, k, &store, &row_live);
                    IfScope f(b, needv(b, store));
                    for(int j = 0; j < cells[i].n_items; ++j)
                    {
                        V v = rocke_b_select(b, row_live, cells[i].values[j], e0);
                        rocke_attn_bwd_paged_slab_store(&dst, k, ivv(cells[i].cols[j]), v);
                        need(b, true);
                    }
                }
            }
            return;
        }
        int vec = s.dkv_store_vec;
        V zero = needv(b, rocke_b_zero_vec(b, m.io, vec));
        if(eq(s.dkv_epilogue, "lds_staged_pair"))
        {
            rocke_attn_bwd_dkv_reader_t rd[2];
            rocke_attn_bwd_paged_slab_t sl[2];
            for(int slot = 0; slot < 2; ++slot)
            {
                bool last = slot == 1;
                need(b,
                     rocke_attn_bwd_tile_step_dkv_rows(&c.step,
                                                       b,
                                                       outs[slot].which,
                                                       &acc,
                                                       outs[slot].mult,
                                                       vec,
                                                       slot,
                                                       last,
                                                       &rd[slot]));
                sl[slot] = slab(*outs[slot].cache, k0, false);
            }
            for(int i = 0; i < 2; ++i)
                store_rows(t, sl[i], rd[i], k0, zero);
            sync(b); /* reads of the epilogue buffers done before they are reused */
            return;
        }
        for(const Out& o : outs)
        {
            rocke_attn_bwd_dkv_reader_t rd;
            need(b,
                 rocke_attn_bwd_tile_step_dkv_rows(
                     &c.step, b, o.which, &acc, o.mult, vec, 0, true, &rd));
            rocke_attn_bwd_paged_slab_t dst = slab(*o.cache, k0, false);
            store_rows(t, dst, rd, k0, zero);
            sync(b); /* reads of the epilogue buffer done before it is reused */
        }
    }
};

/* _emit_main(spec, arch) after validation, on the initialised builder. */
rocke_kernel_def_t* emit_main(Main& m, const U& u)
{
    B* b = m.b;
    const rocke_attn_bwd_spec_t& g = m.s;
    /* _not_built(g). Its agpr_route check is not ported: AGPR_ROUTES_BUILT
     * equals AGPR_ROUTES, which validation already enforces (the general
     * main builder does the same). The dkv_sink check is unreachable here too
     * (validation refuses last_cta as a shell knob) but kept as Python has it. */
    if(eq(g.dkv_mode, "atomic") && eq(g.dkv_sink, "last_cta"))
        throw ckc::NotImplError(
            "the unified attention backward main kernel is not built for dkv_sink='last_cta'");
    m.facts = arch_facts(m.arch);
    need(b, rocke_attn_bwd_tile_geometry_from_spec(b, &m.s, m.arch.c_str(), &m.geom));
    m.plan = lds_plan(g, m.arch);
    m.io = rocke_attn_bwd_ir_type(b, u.dtype);
    need(b, m.io != NULL);
    int D = u.head_size, kn = g.block_n, km = g.block_m;

    Kargs a = declare_main(b, m.io);
    /* Multipliers that are 1.0 under the compiled placement are not applied
     * (the general family's rule; the plan asserts they are 1.0). */
    unsigned unit = 0;
    rocke_attn_bwd_err_t e;
    need_err(rocke_attn_bwd_static_unit_mults(g.static_scale, g.dkv_mode, &unit, &e), e);
    if(unit & ROCKE_ATTN_BWD_UNIT_DS_MULT)
        a.m["ds_mult"] = NULL;
    if(unit & ROCKE_ATTN_BWD_UNIT_DK_MULT)
        a.m["dk_mult"] = NULL;
    if(unit & ROCKE_ATTN_BWD_UNIT_DQ_MULT)
        a.m["dq_mult"] = NULL;
    /* _kernel_attrs(b, g, geom, facts) */
    int threads = threads_of(m);
    rocke_attr_set_int(
        b, &b->kernel->attrs, "max_workgroup_size", threads > MIN_WG_ATTR ? threads : MIN_WG_ATTR);
    if(eq(m.facts.matrix_path, "mfma") && g.agpr_alloc.kind == ROCKE_ATTN_BWD_AGPR_PAIR)
    {
        int64_t pair[2] = {g.agpr_alloc.a, g.agpr_alloc.b};
        rocke_attr_set_int_list(b, &b->kernel->attrs, "agpr_alloc", pair, 2);
    }
    if(eq(g.agpr_route, "vgpr_form") || eq(g.agpr_route, "agpr_fence")
       || eq(g.agpr_route, "asm_g13"))
        rocke_attr_set_bool(b, &b->kernel->attrs, "mfma_vgpr_form", true);
    if(g.waves_per_eu != NONE)
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", g.waves_per_eu);
    if(g.scheduler_strategy != NULL)
        rocke_attr_set_str(b, &b->kernel->attrs, "scheduler_strategy", g.scheduler_strategy);
    need(b, true);

    V tid, lane, wave;
    rocke_attn_bwd_lane_and_wave(b, m.facts.wave_size, NULL, &tid, &lane, &wave);
    need(b, true);
    bool atomic = eq(u.page_mode, "atomic");
    std::unique_ptr<Cta> cta(new Cta());
    Cta& c = *cta;
    c.b = b;
    c.tid = tid;
    /* CTA decode: kv tile fastest, then the kv head (and split), then the
     * sequence; the load-balance order permutes the kv tiles of one (head,
     * seq). */
    c.ids[0] = rocke_b_block_id_x(b);
    c.ids[1] = rocke_b_block_id_y(b);
    c.ids[2] = rocke_b_block_id_z(b);
    need(b,
         rocke_attn_bwd_decode_cta(b,
                                   c.ids[0],
                                   c.ids[1],
                                   c.ids[2],
                                   a["n_kv_tiles"],
                                   a["lb_order"],
                                   NULL,
                                   atomic ? a["g_split"] : NULL,
                                   false,
                                   NULL,
                                   NULL,
                                   &c.work));
    V s_idx = c.work.batch;
    /* q side: THD rows by token; kv side: the sequence's kv length, clamped
     * to [0, max_seqlen_kv]. */
    rocke_attn_bwd_decode_args_t qa = rocke_attn_bwd_decode_args_default();
    qa.s_max = ivv(a["max_seqlen_q"]);
    qa.off_ptr = a["CU_SEQLENS_Q"];
    c.seq_q = rocke_attn_bwd_decode_thd(b, &m.seq_guards, s_idx, &qa);
    need(b, true);
    rocke_attn_bwd_decode_args_t ka = rocke_attn_bwd_decode_args_default();
    ka.s_max = ivv(a["max_seqlen_kv"]);
    ka.seq_ptr = a["SEQ_LENS_KV"];
    ka.has_len = rocke_attn_bwd_iv_int(1);
    c.seq_kv = rocke_attn_bwd_decode_batched(b, &m.seq_guards, s_idx, &ka);
    need(b, true);
    /* The kv head of the CTA owns query heads [hk * G, hk * G + G); a split
     * (atomic page mode) owns G / g_split of them. */
    V G = a["G"];
    V hk = c.work.unit;
    if(atomic)
    {
        V per = rocke_b_div(b, G, a["g_split"]);
        V m1 = rocke_b_mul(b, hk, G);
        V m2 = rocke_b_mul(b, c.work.split, per);
        c.h0 = rocke_b_add(b, m1, m2);
        c.n_h = per;
    }
    else
    {
        c.h0 = rocke_b_mul(b, hk, G);
        c.n_h = G;
    }
    need(b, true);
    c.hk = hk;
    c.hv = hk;

    auto q_side = [&](const char* name,
                      const std::string& prefix,
                      rocke_attn_bwd_strided_tensor_addr_t* out) {
        IV strides[3] = {rocke_attn_bwd_iv_int(0), ivv(a[prefix + "_h"]), ivv(a[prefix + "_t"])};
        need(b,
             rocke_attn_bwd_strided_tensor_addr_init(
                 out, b, a[name], m.io, strides, &c.seq_q, u.stage_vec, 1));
    };
    q_side("Q", "q", &c.q_addr);
    q_side("dO", "do", &c.do_addr);
    bool token_major = eq(g.ws_layout, "token_major");
    rocke_attn_bwd_ws_layout_t layout
        = token_major ? ROCKE_ATTN_BWD_WS_TOKEN_MAJOR : ROCKE_ATTN_BWD_WS_HEAD_MAJOR;
    IV nh = ivv(a["h_q"]);
    auto ws = [&](const char* name, int width, rocke_attn_bwd_workspace_addr_t* out) {
        need(b,
             rocke_attn_bwd_workspace_addr_init(out,
                                                b,
                                                a[name],
                                                ivv(a["ws_rows_q"]),
                                                width,
                                                c.seq_q.ws_row0,
                                                layout,
                                                token_major ? &nh : NULL));
    };
    /* stats_vec > 1 reads <n x f32> windows kept inside the head-major arrays
     * (StatsView needs their query-head count, as in the general shell) */
    ws("WS_LSE2", 1, &c.ws_lse2);
    ws("WS_DSUM", 1, &c.ws_dsum);
    need(b,
         rocke_attn_bwd_stats_view_init(
             &c.stats, b, &c.ws_lse2, &c.ws_dsum, g.stats_vec != 1 ? a["h_q"] : NULL, "clamp", 0));
    ws("WS_DQ", D, &c.ws_dq);
    bool ok
        = g.head_pack == 1
              ? rocke_attn_bwd_dq_sink_init_packed_atomic(
                    &c.sink, b, &c.ws_dq, &c.seq_q, a["dq_mult"])
              : rocke_attn_bwd_dq_sink_init_atomic(&c.sink, b, &c.ws_dq, &c.seq_q, a["dq_mult"]);
    need(b, ok);
    c.has_sink = true;
    V zero = ci(b, 0);
    if(eq(u.mask_class, "band"))
    {
        /* Per-sequence diagonal: 0 (top-left) or len_kv - len_q (bottom-right). */
        V ne = rocke_b_cmp_ne(b, a["bottom_right"], zero);
        V diff = rocke_b_sub(b, c.seq_kv.length, c.seq_q.length);
        V off = rocke_b_select(b, ne, diff, zero);
        rocke_attn_runtime_bounds_t bounds = rocke_attn_runtime_bounds_make(
            rocke_iov_value(c.seq_q.length), rocke_iov_value(c.seq_kv.length));
        bounds.diag_offset = rocke_iov_value(off);
        bounds.left_bound = rocke_iov_value(a["left"]);
        bounds.right_bound = rocke_iov_value(a["right"]);
        need_st(b, rocke_attn_bwd_runtime_band_init(&c.mask, b, &bounds, km, kn));
    }
    else
        need_st(b,
                rocke_attn_bwd_no_mask_ktail_init(&c.mask,
                                                  b,
                                                  rocke_iov_value(c.seq_q.length),
                                                  rocke_iov_value(c.seq_kv.length),
                                                  km,
                                                  kn));
    rocke_attn_bwd_bwd_lds_t lds;
    need(b, rocke_attn_bwd_bwd_lds_init(&lds, b, &m.plan, u.dtype));
    c.lds = &lds;
    need(b,
         rocke_attn_bwd_tile_step_init(&c.step,
                                       b,
                                       &m.geom,
                                       &c.mask,
                                       &c.stats,
                                       &c.sink,
                                       c.lds,
                                       tid,
                                       lane,
                                       wave,
                                       a["scale_log2"],
                                       a["ds_mult"],
                                       m.sentinel));
    c.has_step = true;
    PagedGuard pg;
    IV page = u.sub_page == NONE ? ivv(a["page"]) : rocke_attn_bwd_iv_int(u.sub_page);
    IV page_rows = ivv(a["page_rows"]);
    need(b,
         rocke_attn_bwd_paged_kv_init(&pg.p,
                                      b,
                                      a["BLOCK_TABLE"],
                                      ivv(a["bt_stride"]),
                                      s_idx,
                                      &c.seq_kv,
                                      kn,
                                      page,
                                      u.sub_page == NONE ? NULL : &u.sub_page,
                                      &page_rows));
    pg.live = true;

    Unified un{m, u, a, c, &pg.p, {}, {}, {}, {}};
    auto cache = [&](const char* ptr, const std::string& prefix) {
        IV strides[3] = {ivv(a[prefix + "_blk"]), ivv(a[prefix + "_tok"]), ivv(a[prefix + "_h"])};
        rocke_attn_bwd_paged_cache_rows_t cr
            = rocke_attn_bwd_paged_kv_cache(&pg.p, a[ptr], m.io, strides, u.stage_vec);
        need(b, true);
        return cr;
    };
    un.k_cache = cache("KC", "kc");
    un.v_cache = cache("VC", "vc");
    c.kv_rows_fn
        = [&un](int which, IV h, IV row0, bool clamp) { return un.kv_rows(which, h, row0, clamp); };
    KvEpilogue epilogue;
    if(atomic)
        epilogue = [&](const rocke_attn_bwd_dkdv_acc_t& acc, V k0, V live_) {
            un.atomic_epilogue(acc, k0, live_);
        };
    else
    {
        un.out[0] = cache("DKC", "kc");
        un.out[1] = cache("DVC", "vc");
        epilogue = [&](const rocke_attn_bwd_dkdv_acc_t& acc, V k0, V live_) {
            (void)live_;
            un.direct_epilogue(acc, k0);
        };
    }
    kv_tile(m, a, c, c.work.first_kt, "", &epilogue);
    rocke_b_ret(b);
    need(b, true);
    return rocke_ir_builder_kernel(b);
}

template <class Fn>
rocke_status_t boundary(rocke_attn_bwd_err_t* err, Fn&& fn)
{
    store_err(err, ROCKE_OK, "");
    try
    {
        fn();
        return ROCKE_OK;
    }
    catch(const ckc::Error& e)
    {
        return store_err(err, e.code(), e.what());
    }
    catch(const std::bad_alloc&)
    {
        return store_err(err, ROCKE_ERR_OOM, "out of memory");
    }
}

} // namespace
} // namespace rocke_attn_bwd

using namespace rocke_attn_bwd;

extern "C" {

rocke_attn_bwd_unified_spec_t rocke_attn_bwd_unified_spec_default(void)
{
    rocke_attn_bwd_unified_spec_t s;
    memset(&s, 0, sizeof(s));
    s.head_size = 0;
    s.dtype = "bf16";
    s.page_mode = "atomic";
    s.mask_class = "band";
    s.sub_page = ROCKE_ATTN_BWD_NONE;
    s.stage_vec = 8;
    s.knobs = rocke_attn_bwd_spec_default();
    s.name = "rocke_attn_bwd_unified";
    return s;
}

rocke_status_t rocke_attn_bwd_unified_spec_init(rocke_attn_bwd_unified_spec_t* spec,
                                                rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        if(spec == NULL)
            throw ckc::ValueError("spec must not be NULL");
        post_init(*spec);
    });
}

rocke_status_t rocke_attn_bwd_unified_general_spec(const rocke_attn_bwd_unified_spec_t* spec,
                                                   rocke_attn_bwd_spec_t* out,
                                                   rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        if(spec == NULL || out == NULL)
            throw ckc::ValueError("spec and out must not be NULL");
        *out = general_spec(*spec);
    });
}

const char* rocke_attn_bwd_unified_page_class(const rocke_attn_bwd_unified_spec_t* spec)
{
    if(spec == NULL)
        return NULL;
    return spec->sub_page == ROCKE_ATTN_BWD_NONE ? "multi" : "sub";
}

rocke_status_t rocke_attn_bwd_unified_kernel_name(const rocke_attn_bwd_unified_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap,
                                                  rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        if(spec == NULL || out == NULL)
            throw ckc::ValueError("spec and out must not be NULL");
        std::string n = kernel_name(*spec);
        if(n.size() + 1 > out_cap)
            throw ckc::ValueError("kernel name does not fit in out_cap bytes");
        memcpy(out, n.c_str(), n.size() + 1);
    });
}

rocke_status_t rocke_attn_bwd_unified_validate_spec(const rocke_attn_bwd_unified_spec_t* spec,
                                                    const char* arch,
                                                    rocke_attn_bwd_unified_spec_t* out,
                                                    rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        if(spec == NULL || out == NULL)
            throw ckc::ValueError("spec and out must not be NULL");
        *out = validate(*spec, arch ? arch : "", NULL);
    });
}

rocke_status_t rocke_attn_bwd_unified_params(const char* stage,
                                             const char* names[ROCKE_ATTN_BWD_PARAMS_MAX],
                                             const char* formats[ROCKE_ATTN_BWD_PARAMS_MAX],
                                             int* n_out,
                                             rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        if(!eq(stage, "main"))
            throw ckc::ValueError("stage=" + repr_s(stage) + " not in ('main',)");
        std::vector<const char*> n, f;
        main_params(&n, &f);
        if(n.size() > ROCKE_ATTN_BWD_PARAMS_MAX)
            throw ckc::ValueError("more kernargs than ROCKE_ATTN_BWD_PARAMS_MAX");
        for(size_t i = 0; i < n.size(); ++i)
        {
            names[i] = n[i];
            formats[i] = f[i];
        }
        *n_out = (int)n.size();
    });
}

rocke_kernel_def_t* rocke_build_attn_bwd_unified_main_new(rocke_ir_builder_t* b,
                                                          const rocke_attn_bwd_unified_spec_t* spec,
                                                          const char* arch)
{
    if(b == NULL)
        return NULL;
    /* validate_attn_bwd_unified_spec raises before the Python builder exists;
     * the C builder must exist to carry the error, so a refused spec gets a
     * placeholder name. */
    std::unique_ptr<Main> m(new Main());
    U u;
    std::string name;
    rocke_status_t code = ROCKE_OK;
    std::string msg;
    try
    {
        if(spec == NULL)
            throw ckc::ValueError("spec must not be NULL");
        m->arch = arch ? arch : "";
        u = validate(*spec, m->arch, &m->s);
        name = kernel_name(u);
    }
    catch(const ckc::Error& e)
    {
        code = e.code();
        msg = e.what();
    }
    catch(const std::bad_alloc& e)
    {
        code = ROCKE_ERR_OOM;
        msg = e.what();
    }
    if(rocke_ir_builder_init(b, code == ROCKE_OK ? name.c_str() : "rocke_attn_bwd_unified_invalid")
       != ROCKE_OK)
        return NULL;
    if(code != ROCKE_OK)
    {
        rocke_i_set_err_msg(b, code, msg.c_str());
        return NULL;
    }
    m->b = b;
    m->seq_guards = 0;
    m->sentinel = true;
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* { return emit_main(*m, u); });
}

} /* extern "C" */
