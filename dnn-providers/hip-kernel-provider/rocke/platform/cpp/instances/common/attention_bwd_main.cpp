// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of the main-kernel builder of library/kernels/common/attention_bwd.py
 * (_declare_main through build_attn_bwd_main): _emit_main, _MirrorStash,
 * _opaque, _cta_setup, _kv_tile with its q loops, the atomic / slices /
 * direct dK / dV epilogues and their store helpers. The spec, validation and
 * ABI half lives in attention_bwd.cpp (rocke/instance_attention_bwd.h).
 *
 * Byte identity: every builder call (constants included) is bound to a local
 * in Python's left-to-right, inner-first evaluation order, and every barrier
 * carries ROCKE_ATTN_BWD_BARRIER_FENCE (the Python builder's fence_barriers).
 * A Python raise is a ckc::Error throw with the same message; the entry point
 * catches it under ckc::guard_builder.
 *
 * This mirrors _emit_main(spec, arch, main_kernel_refusals=False): the
 * recorded resource-gate refusals stay Python-only, so the C entry accepts
 * the RESOURCE_REFUSED specs that Python's public build_attn_bwd_main
 * refuses. _kv_tile, the plumbing and the types it shares with the dense
 * backward shell (_attention_dense_bwd_shell.cpp) and the unified main kernel
 * (attention_bwd_unified.cpp) are declared in
 * rocke/instance_attention_bwd_main_internal.h, with _kv_tile's duck-typed
 * hooks (mask.static_runs, epilogue=, the q_addr .. v_addr slab sources).
 */
#include "rocke/instance_attention_bwd.h"

#include <string.h>

#include <deque>
#include <map>
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
#include "rocke/helper_rocke.instances.common._attention_bwd_counted.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_frag.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_gemm.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_ring.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_sinks.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_worklist.h"
#include "rocke/instance_attention_bwd_main_internal.h"
#include "rocke/ir_internal.h"

namespace rocke_attn_bwd
{
using namespace main_kernel;
using namespace main_kernel::plumbing;

namespace
{

typedef rocke_attn_bwd_q_step_inputs_t Inputs;
typedef rocke_attn_bwd_prefetch_tile_t Tile;

/* The lowering's flat-work-group-size attribute has a fixed minimum of 64. */
const int MIN_WG_ATTR = 64;

bool is_int_type(V v, const rocke_type_t* t)
{
    return v != NULL && v->type != NULL && rocke_type_eq(v->type, t);
}

template <class T>
T* arena(B* b, size_t n)
{
    void* p = rocke_arena_calloc(&b->arena, (n ? n : 1) * sizeof(T));
    if(p == NULL)
        throw ckc::OOMError("attention backward main kernel: arena allocation failed");
    return (T*)p;
}

/* A copy of vals in the builder's arena (the helpers' from_flat point into
 * their input). */
V* arena_copy(B* b, const std::vector<V>& vals)
{
    V* p = arena<V>(b, vals.size());
    for(size_t i = 0; i < vals.size(); ++i)
        p[i] = vals[i];
    return p;
}

/* (name, init) iter args of scf_for_iter. */
struct IterArgs
{
    std::vector<std::string> names;
    std::vector<V> vals;
    void add(const std::string& n, V v)
    {
        names.push_back(n);
        vals.push_back(v);
    }
};

rocke_for_t
    for_iter(B* b, V lo, V hi, V step, const IterArgs& args, const std::string& iv, bool elide)
{
    std::vector<rocke_iter_arg_t> ia;
    for(size_t i = 0; i < args.vals.size(); ++i)
    {
        rocke_iter_arg_t a;
        a.name = args.names[i].c_str();
        a.init = args.vals[i];
        ia.push_back(a);
    }
    rocke_for_t f = rocke_b_scf_for_iter(
        b, lo, hi, step, ia.data(), (int)ia.size(), iv.c_str(), false, elide);
    if(f.op == NULL)
        raise_builder(b);
    return f;
}

std::vector<V> results_of(const rocke_for_t& f)
{
    std::vector<V> out;
    for(int i = 0; i < f.op->num_results; ++i)
        out.push_back(f.op->results[i]);
    return out;
}

void yield(B* b, const std::vector<V>& vals)
{
    rocke_b_scf_yield(b, vals.data(), (int)vals.size());
    need(b, true);
}

/* DkDvAcc.flat() */
std::vector<V> acc_flat(const rocke_attn_bwd_dkdv_acc_t& acc)
{
    int n = rocke_attn_bwd_dkdv_acc_flat_len(&acc);
    std::vector<V> out((size_t)n);
    if(n)
        rocke_attn_bwd_dkdv_acc_flat(&acc, out.data());
    return out;
}

/* StagedStep.flat() */
std::vector<V> staged_flat(const rocke_attn_bwd_staged_step_t& s)
{
    int n = rocke_attn_bwd_staged_step_flat_len(&s);
    std::vector<V> out((size_t)n);
    if(n)
        rocke_attn_bwd_staged_step_flat(&s, out.data());
    return out;
}

/* PendingDq.flat() / names(prefix) */
std::vector<V> pend_flat(const rocke_attn_bwd_pending_dq_t& p)
{
    int n = rocke_attn_bwd_pending_dq_flat_len(&p);
    std::vector<V> out((size_t)n);
    if(n)
        rocke_attn_bwd_pending_dq_flat(&p, out.data());
    return out;
}

std::vector<std::string>
    pend_names(B* b, const rocke_attn_bwd_pending_dq_t& p, const std::string& prefix)
{
    std::vector<std::string> out;
    int n = rocke_attn_bwd_pending_dq_flat_len(&p);
    for(int i = 0; i < n; ++i)
    {
        char buf[256];
        need(b, rocke_attn_bwd_pending_dq_name(&p, prefix.c_str(), i, buf, sizeof(buf)));
        out.push_back(buf);
    }
    return out;
}

rocke_attn_bwd_pending_dq_t
    pend_from_flat(B* b, const rocke_attn_bwd_pending_dq_t& p, const std::vector<V>& vals)
{
    rocke_attn_bwd_pending_dq_t out;
    V* keep = arena_copy(b, vals);
    need(b, rocke_attn_bwd_pending_dq_from_flat(b, &p, keep, (int)vals.size(), &out));
    return out;
}

/* ---------------------------------------------------------------- kernargs */

/* _main_noalias(spec) membership */
bool noalias_of(const rocke_attn_bwd_spec_t& s, const char* name)
{
    if(s.ptr_noalias == NULL)
        return false;
    if(eq(s.ptr_noalias, "workspace"))
        return strncmp(name, "WS_", 3) == 0;
    return true;
}

/* _declare_main(b, io, noalias) */
Kargs declare_main(B* b, const rocke_attn_bwd_spec_t& s, const rocke_type_t* io)
{
    static const char* const readonly[] = {"Q",
                                           "K",
                                           "V",
                                           "dO",
                                           "WS_LSE2",
                                           "WS_DSUM",
                                           "SEQ_Q",
                                           "SEQ_KV",
                                           "OFF_Q",
                                           "OFF_KV",
                                           "WORKLIST"};
    static const char* const io_ptrs[] = {"Q", "K", "V", "dO", "dK", "dV"};
    const char* names[ROCKE_ATTN_BWD_PARAMS_MAX];
    const char* formats[ROCKE_ATTN_BWD_PARAMS_MAX];
    int n = 0;
    rocke_attn_bwd_err_t e;
    need_err(rocke_attn_bwd_params("main", names, formats, &n, &e), e);
    Kargs a;
    for(int i = 0; i < n; ++i)
    {
        const char* name = names[i];
        const char* f = formats[i];
        V v;
        if(eq(f, "Q"))
        {
            const rocke_type_t* pointee;
            bool is_io = false;
            for(const char* x : io_ptrs)
                is_io = is_io || eq(name, x);
            if(is_io)
                pointee = io;
            else if(strncmp(name, "WS_", 3) == 0)
                pointee = rocke_f32();
            else
                pointee = rocke_i32();
            bool ro = false;
            for(const char* x : readonly)
                ro = ro || eq(name, x);
            rocke_param_opts_t opts;
            memset(&opts, 0, sizeof(opts));
            if(ro)
            {
                opts.readonly = true;
                opts.readonly_set = true;
            }
            /* _MAIN_WORKSPACE_PTRS: WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV */
            if(noalias_of(s, name))
            {
                opts.noalias = true;
                opts.noalias_set = true;
            }
            bool any = opts.readonly_set || opts.noalias_set;
            v = rocke_b_param(b, name, rocke_ptr_type(b, pointee, "global"), any ? &opts : NULL);
        }
        else if(eq(f, "q"))
            v = rocke_b_param(b, name, rocke_i64(), NULL);
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

/* _decode_side(b, s, a, bz, side) */
rocke_attn_bwd_seq_info_t decode_side(Main& m, const Kargs& a, V bz, bool q)
{
    B* b = m.b;
    rocke_attn_bwd_decode_args_t args = rocke_attn_bwd_decode_args_default();
    args.s_max = ivv(a[q ? "S_q_max" : "S_kv_max"]);
    args.seq_ptr = a[q ? "SEQ_Q" : "SEQ_KV"];
    args.has_len = ivv(a["has_len"]);
    args.len_stride = ivv(a["len_stride"]);
    rocke_attn_bwd_seq_info_t out;
    if(eq(m.s.seq_mode, "thd"))
    {
        args.off_ptr = a[q ? "OFF_Q" : "OFF_KV"];
        args.off64 = ivv(a["off64"]);
        args.mult = ivv(a[q ? "q_mult" : "kv_mult"]);
        args.div = ivv(a[q ? "q_div" : "kv_div"]);
        args.ws_seg = ivv(a["ws_seg"]);
        out = rocke_attn_bwd_decode_thd(b, &m.seq_guards, bz, &args);
    }
    else
        out = rocke_attn_bwd_decode_batched(b, &m.seq_guards, bz, &args);
    need(b, true);
    return out;
}

/* _cta_setup(b, s, a, geom, plan, io, tid=, lane=, wave=, sentinel=, ids=,
 * lds=); ids NULL / lds NULL are None. */
void cta_setup(Main& m,
               const Kargs& a,
               Cta& c,
               V tid,
               V lane,
               V wave,
               const V* ids,
               rocke_attn_bwd_bwd_lds_t* lds_in,
               rocke_attn_bwd_bwd_lds_t* lds_out)
{
    B* b = m.b;
    const rocke_attn_bwd_spec_t& s = m.s;
    c.b = b;
    c.tid = tid;
    int D = s.head_size, km = s.block_m, kn = s.block_n;
    bool atomic = eq(s.dkv_mode, "atomic");
    V split_arg = atomic ? a["g_split"] : NULL;
    rocke_attn_bwd_xcd_remap_t xcd;
    bool has_xcd = false;
    if(s.xcd_remap)
    {
        V gy;
        if(atomic)
        {
            V d = rocke_b_div(b, a["h_q"], a["G"]);
            gy = rocke_b_mul(b, d, a["g_split"]);
        }
        else
            gy = a["h_k"];
        xcd.n = a["xcd_n"];
        xcd.chunk = a["xcd_chunk"];
        xcd.gy = gy;
        has_xcd = true;
    }
    if(ids == NULL)
    {
        c.ids[0] = rocke_b_block_id_x(b);
        c.ids[1] = rocke_b_block_id_y(b);
        c.ids[2] = rocke_b_block_id_z(b);
    }
    else
        for(int i = 0; i < 3; ++i)
            c.ids[i] = ids[i];
    rocke_attn_bwd_work_list_t wl;
    if(s.worklist)
    {
        wl.ptr = a["WORKLIST"];
        wl.use = a["use_worklist"];
    }
    need(b,
         rocke_attn_bwd_decode_cta(b,
                                   c.ids[0],
                                   c.ids[1],
                                   c.ids[2],
                                   a["n_kv_tiles"],
                                   a["lb_order"],
                                   s.lb_mirror ? a["pair"] : NULL,
                                   split_arg,
                                   s.lb_interleave,
                                   has_xcd ? &xcd : NULL,
                                   s.worklist ? &wl : NULL,
                                   &c.work));
    V bz = c.work.batch;
    c.seq_q = decode_side(m, a, bz, true);
    c.seq_kv = decode_side(m, a, bz, false);
    rocke_attn_bwd_head_range_t heads = rocke_attn_bwd_head_range(
        b, c.work.unit, c.work.split, a["G"], a["gk"], a["gv"], split_arg);
    need(b, true);
    c.hk = heads.hk;
    c.hv = heads.hv;
    c.h0 = heads.h0;
    c.n_h = heads.n_heads;

    auto tensor = [&](const char* name,
                      const std::string& prefix,
                      const rocke_attn_bwd_seq_info_t* seq,
                      rocke_attn_bwd_strided_tensor_addr_t* out) {
        IV strides[3] = {ivv(a[prefix + "_b"]), ivv(a[prefix + "_h"]), ivv(a[prefix + "_t"])};
        need(b,
             rocke_attn_bwd_strided_tensor_addr_init(
                 out, b, a[name], m.io, strides, seq, s.stage_vec, s.narrow_vec));
    };
    tensor("Q", "q", &c.seq_q, &c.q_addr);
    tensor("dO", "do", &c.seq_q, &c.do_addr);
    tensor("K", "k", &c.seq_kv, &c.k_addr);
    tensor("V", "v", &c.seq_kv, &c.v_addr);
    bool token_major = eq(s.ws_layout, "token_major");
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
    ws("WS_LSE2", 1, &c.ws_lse2);
    ws("WS_DSUM", 1, &c.ws_dsum);
    need(b,
         rocke_attn_bwd_stats_view_init(&c.stats,
                                        b,
                                        &c.ws_lse2,
                                        &c.ws_dsum,
                                        s.stats_vec != 1 ? a["h_q"] : NULL,
                                        s.ws_bounds,
                                        km));
    if(eq(s.dq_mode, "split"))
        rocke_attn_bwd_dq_sink_init_null(&c.sink);
    else
    {
        ws("WS_DQ", D, &c.ws_dq);
        bool ok = s.head_pack == 1 ? rocke_attn_bwd_dq_sink_init_packed_atomic(
                                         &c.sink, b, &c.ws_dq, &c.seq_q, a["dq_mult"])
                                   : rocke_attn_bwd_dq_sink_init_atomic(
                                         &c.sink, b, &c.ws_dq, &c.seq_q, a["dq_mult"]);
        need(b, ok);
    }
    c.has_sink = true;
    V zero = ci(b, 0);
    if(eq(s.mask_class, "band"))
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
    if(lds_in == NULL)
    {
        need(b, rocke_attn_bwd_bwd_lds_init(lds_out, b, &m.plan, s.dtype));
        c.lds = lds_out;
    }
    else
        c.lds = lds_in;
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
    bool staged = !(eq(s.global_path, "vgpr") && s.ring_depth == 1);
    if(staged && !eq(s.barrier_merge, "on"))
    {
        /* Q / dO prefetch ring: the DMA wait counts the step's dQ atomics
         * (none when the main kernel emits no G4). */
        int n_atomics = km * D / (m.geom.waves * m.facts.wave_size);
        need(b,
             rocke_attn_bwd_q_ring_init(&c.ring,
                                        b,
                                        &m.geom,
                                        c.lds,
                                        tid,
                                        lane,
                                        wave,
                                        &c.stats,
                                        eq(s.dq_mode, "atomic") ? n_atomics : 0));
        c.has_ring = true;
        need(b, rocke_attn_bwd_q_ring_check_step(&c.ring, b, &c.step.sched));
    }
}

/* ------------------------------------------------------------ epilogues */

/* _dkv_slab(b, s, a, which, hk=, k0=, seq_kv=, io=) */
rocke_attn_bwd_row_slab_t
    dkv_slab(Main& m, const Kargs& a, const Cta& c, const std::string& which, V hk, V k0)
{
    B* b = m.b;
    const rocke_attn_bwd_spec_t& s = m.s;
    rocke_attn_bwd_strided_tensor_addr_t t;
    IV strides[3] = {a.iv(which + "_b"), a.iv(which + "_h"), a.iv(which + "_t")};
    need(b,
         rocke_attn_bwd_strided_tensor_addr_init(&t,
                                                 b,
                                                 a[which == "dk" ? "dK" : "dV"],
                                                 m.io,
                                                 strides,
                                                 &c.seq_kv,
                                                 s.stage_vec,
                                                 s.narrow_vec));
    rocke_attn_bwd_row_slab_t sl
        = rocke_attn_bwd_strided_tensor_addr_slab(&t, ivv(hk), ivv(k0), false);
    need(b, true);
    return sl;
}

/* _atomic_epilogue */
void atomic_epilogue(
    Main& m, const Kargs& a, Cta& c, const rocke_attn_bwd_dkdv_acc_t& acc, V k0, V live_)
{
    B* b = m.b;
    const rocke_attn_bwd_spec_t& s = m.s;
    int D = s.head_size;
    bool tok_major = eq(s.ws_layout, "token_major");
    bool select = eq(s.dkv_atomic_guard, "select");
    IfScope live_if(b, live_);
    struct Out
    {
        const char* which;
        const char* ptr;
        V head;
        V n_heads;
        V mult;
    };
    Out outs[2]
        = {{"dk", "WS_DK", c.hk, a["h_k"], a["dk_mult"]}, {"dv", "WS_DV", c.hv, a["h_v"], NULL}};
    for(const Out& o : outs)
    {
        rocke_attn_bwd_workspace_addr_t w;
        IV nh = ivv(o.n_heads);
        rocke_attn_bwd_ws_layout_t layout
            = tok_major ? ROCKE_ATTN_BWD_WS_TOKEN_MAJOR : ROCKE_ATTN_BWD_WS_HEAD_MAJOR;
        need(b,
             rocke_attn_bwd_workspace_addr_init(&w,
                                                b,
                                                a[o.ptr],
                                                a.iv("ws_rows_kv"),
                                                D,
                                                c.seq_kv.ws_row0,
                                                layout,
                                                tok_major ? &nh : NULL));
        rocke_attn_bwd_ws_slab_t slab
            = rocke_attn_bwd_workspace_addr_slab(&w, ivv(o.head), ivv(k0));
        need(b, true);
        rocke_attn_bwd_dkv_row_t* rows;
        int n_rows;
        need(b, rocke_attn_bwd_tile_step_dkv_row_slots(&c.step, b, o.which, &acc, &rows, &n_rows));
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
            V in_seq = rocke_b_cmp_lt(b, kr, c.seq_kv.length);
            if(select)
                rocke_attn_bwd_ws_slab_atomic_add_row_select(
                    &slab, row.row, xs.data(), vs.data(), (int)vs.size(), in_seq);
            else
                rocke_attn_bwd_ws_slab_atomic_add_row(
                    &slab, row.row, xs.data(), vs.data(), (int)vs.size(), in_seq);
            need(b, true);
        }
    }
}

/* _slice_slab */
rocke_attn_bwd_ws_slab_t
    slice_slab(Main& m, const Kargs& a, const Cta& c, const std::string& which, V head, V k0)
{
    B* b = m.b;
    const rocke_attn_bwd_spec_t& s = m.s;
    int D = s.head_size;
    bool dk = which == "dk";
    const char* ptr = dk ? "WS_DK" : "WS_DV";
    V n_heads = dk ? a["h_k"] : a["h_v"];
    V g = dk ? a["gk"] : a["gv"];
    V hm = rocke_b_mod(b, c.h0, g);
    V sl = rocke_b_div(b, hm, c.n_h);
    V wr = rocke_b_zext(b, a["ws_rows_kv"], rocke_i64());
    V nh = rocke_b_zext(b, n_heads, rocke_i64());
    V wn = rocke_b_mul(b, wr, nh);
    V cd = rocke_b_const_i64(b, D);
    V elems = rocke_b_mul(b, wn, cd);
    V sl64 = rocke_b_zext(b, sl, rocke_i64());
    V se = rocke_b_mul(b, sl64, elems);
    V c4 = rocke_b_const_i64(b, 4);
    V bytes = rocke_b_mul(b, se, c4);
    V base = rocke_b_global_ptr_add(b, a[ptr], bytes);
    bool tok_major = eq(s.ws_layout, "token_major");
    rocke_attn_bwd_workspace_addr_t w;
    IV nhv = ivv(n_heads);
    rocke_attn_bwd_ws_layout_t layout
        = tok_major ? ROCKE_ATTN_BWD_WS_TOKEN_MAJOR : ROCKE_ATTN_BWD_WS_HEAD_MAJOR;
    need(
        b,
        rocke_attn_bwd_workspace_addr_init(
            &w, b, base, a.iv("ws_rows_kv"), D, c.seq_kv.ws_row0, layout, tok_major ? &nhv : NULL));
    rocke_attn_bwd_ws_slab_t out = rocke_attn_bwd_workspace_addr_slab(&w, ivv(head), ivv(k0));
    need(b, true);
    return out;
}

/* _SliceTarget */
struct SliceTarget
{
    B* b;
    rocke_attn_bwd_ws_slab_t slab;
    V k0, length;
    bool buffer;
    int D;
    V rsrc = NULL;

    SliceTarget(Main& m, const rocke_attn_bwd_ws_slab_t& sl, V k0_, const Cta& c)
        : b(m.b)
        , slab(sl)
        , k0(k0_)
        , length(c.seq_kv.length)
    {
        const rocke_attn_bwd_spec_t& s = m.s;
        buffer = eq(s.ws_bounds, "buffer");
        D = s.head_size;
        if(buffer)
        {
            if(slab.row_stride.v != NULL || slab.row_stride.i != D)
                throw ckc::ValueError("buffer-bounded slices need the head-major workspace");
            V d = rocke_b_sub(b, c.seq_kv.length, k0);
            V n = rocke_b_smin(b, d, slab.rows_left);
            V cbn = ci(b, s.block_n);
            V mn = rocke_b_smin(b, n, cbn);
            V c0 = ci(b, 0);
            n = rocke_b_smax(b, mn, c0);
            V cb = ci(b, 4 * D);
            V nb = rocke_b_mul(b, n, cb);
            rsrc = needv(b, rocke_b_buffer_rsrc(b, slab.ptr, nb));
        }
    }

    V ok(V r)
    {
        V rk = rocke_attn_bwd_ws_slab_row_ok(&slab, r);
        V kr = rocke_b_add(b, k0, r);
        V in = rocke_b_cmp_lt(b, kr, length);
        return rocke_b_land(b, rk, in);
    }

    /* _offset(r, c): c is IntOrValue, as Python's isinstance(c, int). */
    V offset(V r, IV c)
    {
        V cd = ci(b, D);
        V idx = rocke_b_mul(b, r, cd);
        if(c.v == NULL)
        {
            if(c.i)
            {
                V cc = ci(b, c.i);
                idx = rocke_b_add(b, idx, cc);
            }
        }
        else
            idx = rocke_b_add(b, idx, c.v);
        V c4 = ci(b, 4);
        return rocke_b_mul(b, idx, c4);
    }

    void buffer_store(V r, IV c, V v)
    {
        int n = v->type->count;
        V zero = ci(b, 0);
        V off = offset(r, c);
        if(n == 1)
        {
            V e = rocke_b_vec_extract(b, v, 0);
            rocke_b_buffer_store_f32(b, rsrc, off, zero, e);
            need(b, true);
            return;
        }
        for(int k = 0; k < n; k += 4)
        {
            V part = v;
            if(n > 4)
            {
                V comps[4];
                for(int i = 0; i < 4; ++i)
                    comps[i] = rocke_b_vec_extract(b, v, k + i);
                part = rocke_b_vec_pack(b, comps, 4, rocke_f32());
            }
            V o = off;
            if(k != 0)
            {
                V ck = ci(b, 4 * k);
                o = rocke_b_add(b, off, ck);
            }
            rocke_b_buffer_store_vN_f32(b, rsrc, o, zero, part, n < 4 ? n : 4);
            need(b, true);
        }
    }

    void store_vec(V r, IV c, V v)
    {
        if(buffer)
        {
            buffer_store(r, c, v);
            return;
        }
        V k = ok(r);
        IfScope f(b, k);
        rocke_attn_bwd_ws_slab_store_vec_in_row(&slab, r, c, v);
        need(b, true);
    }

    void store_row(V r, const std::vector<IV>& cols, const std::vector<V>& vals)
    {
        if(!buffer)
        {
            V kr = rocke_b_add(b, k0, r);
            V in = rocke_b_cmp_lt(b, kr, length);
            rocke_attn_bwd_ws_slab_store_row(
                &slab, r, cols.data(), vals.data(), (int)vals.size(), in);
            need(b, true);
            return;
        }
        V zero = ci(b, 0);
        for(size_t i = 0; i < cols.size(); ++i)
        {
            V off = offset(r, cols[i]);
            rocke_b_buffer_store_f32(b, rsrc, off, zero, vals[i]);
            need(b, true);
        }
    }
};

/* _store_slice_rows(b, s, step, dst, view, cols=, col0=, tid=, mult=) */
void store_slice_rows(Main& m,
                      SliceTarget& dst,
                      const rocke_attn_bwd_lds_view_t& view,
                      int cols,
                      int col0,
                      V tid,
                      V mult)
{
    B* b = m.b;
    const rocke_attn_bwd_spec_t& s = m.s;
    int kn = s.block_n, vec = s.dkv_store_vec;
    if(cols % vec)
        throw ckc::ValueError("dkv_store_vec=" + std::to_string(vec) + " does not divide "
                              + std::to_string(cols) + " columns");
    int threads = threads_of(m);
    int per_row = cols / vec;
    int total = kn * per_row;
    int rounds = rocke_attn_bwd_deal_chunks_count(b, total, threads);
    need(b, rounds >= 0);
    for(int it = 0; it < rounds; ++it)
    {
        bool tail = false;
        V chunk = needv(b, rocke_attn_bwd_deal_chunks_round(b, tid, total, threads, it, &tail));
        V r, c;
        rocke_attn_bwd_chunk_coords(b, chunk, per_row, vec, &r, &c);
        V cc = c;
        if(col0 != 0)
        {
            V k = ci(b, col0);
            cc = rocke_b_add(b, c, k);
        }
        auto read = [&]() -> V {
            V v = rocke_attn_bwd_lds_view_load(b, &view, ivv(r), ivv(c), rocke_f32(), vec);
            needv(b, v);
            if(mult != NULL)
            {
                std::vector<V> comps;
                for(int i = 0; i < vec; ++i)
                {
                    V e = rocke_b_vec_extract(b, v, i);
                    comps.push_back(rocke_b_fmul(b, mult, e));
                }
                v = rocke_b_vec_pack(b, comps.data(), vec, rocke_f32());
            }
            return needv(b, v);
        };
        if(dst.buffer)
        {
            if(tail)
            {
                V ct = ci(b, total);
                V lt = rocke_b_cmp_lt(b, chunk, ct);
                IfScope f(b, lt);
                V v = read();
                dst.store_vec(r, ivv(cc), v);
            }
            else
            {
                V v = read();
                dst.store_vec(r, ivv(cc), v);
            }
            continue;
        }
        V ok = dst.ok(r);
        if(tail)
        {
            V ct = ci(b, total);
            V lt = rocke_b_cmp_lt(b, chunk, ct);
            ok = rocke_b_land(b, ok, lt);
        }
        IfScope f(b, ok);
        V v = read();
        rocke_attn_bwd_ws_slab_store_vec_in_row(&dst.slab, r, ivv(cc), v);
        need(b, true);
    }
}

/* _slices_epilogue */
void slices_epilogue(
    Main& m, const Kargs& a, Cta& c, const rocke_attn_bwd_dkdv_acc_t& acc, V k0, V live_, V tid)
{
    B* b = m.b;
    const rocke_attn_bwd_spec_t& s = m.s;
    int D = s.head_size;
    int hw = D / 2;
    struct Out
    {
        const char* which;
        V head;
        V mult;
    };
    Out outs[2] = {{"dk", c.hk, a["dk_mult"]}, {"dv", c.hv, NULL}};
    auto target = [&](const Out& o) {
        rocke_attn_bwd_ws_slab_t sl = slice_slab(m, a, c, o.which, o.head, k0);
        return std::unique_ptr<SliceTarget>(new SliceTarget(m, sl, k0, c));
    };
    IfScope live_if(b, live_);
    if(s.acc_in_lds == 1)
    {
        sync(b); /* the images' cells were last updated by other lanes */
        for(const Out& o : outs)
        {
            rocke_attn_bwd_lds_view_t view;
            need(b, rocke_attn_bwd_tile_step_dkv_f32_image(&c.step, b, o.which, &view));
            auto dst = target(o);
            store_slice_rows(m, *dst, view, D, 0, tid, o.mult);
        }
        return;
    }
    if(eq(s.dkv_epilogue, "direct"))
    {
        for(const Out& o : outs)
        {
            auto dst = target(o);
            if(s.dkv_cell_vec == 4)
            {
                rocke_attn_bwd_dkv_quad_t* quads;
                int nq;
                need(b,
                     rocke_attn_bwd_tile_step_dkv_quad_rows(
                         &c.step, b, o.which, &acc, o.mult, &quads, &nq));
                for(int i = 0; i < nq; ++i)
                {
                    V v = rocke_b_vec_pack(b, quads[i].values, 4, rocke_f32());
                    dst->store_vec(quads[i].row, ivv(quads[i].col0), needv(b, v));
                }
                continue;
            }
            rocke_attn_bwd_dkv_row_t* rows;
            int n_rows;
            need(b,
                 rocke_attn_bwd_tile_step_dkv_row_slots(&c.step, b, o.which, &acc, &rows, &n_rows));
            for(int r = 0; r < n_rows; ++r)
            {
                std::vector<IV> cols;
                std::vector<V> vals;
                for(int i = 0; i < rows[r].n_items; ++i)
                {
                    cols.push_back(ivv(rows[r].cols[i]));
                    vals.push_back(o.mult != NULL
                                       ? needv(b, rocke_b_fmul(b, rows[r].values[i], o.mult))
                                       : rows[r].values[i]);
                }
                dst->store_row(rows[r].row, cols, vals);
            }
        }
        return;
    }
    bool pair = eq(s.dkv_epilogue, "lds_staged_pair");
    for(const Out& o : outs)
    {
        auto dst = target(o);
        if(pair)
        {
            rocke_attn_bwd_lds_view_t views[2];
            for(int h = 0; h < 2; ++h)
                need(b,
                     rocke_attn_bwd_tile_step_dkv_f32_part(
                         &c.step, b, o.which, &acc, o.mult, h, h, &views[h]));
            sync(b);
            for(int h = 0; h < 2; ++h)
                store_slice_rows(m, *dst, views[h], hw, h * hw, tid, NULL);
            sync(b); /* reads done before the buffers are written again */
            continue;
        }
        for(int h = 0; h < 2; ++h)
        {
            rocke_attn_bwd_lds_view_t view;
            need(b,
                 rocke_attn_bwd_tile_step_dkv_f32_part(
                     &c.step, b, o.which, &acc, o.mult, h, 0, &view));
            sync(b);
            store_slice_rows(m, *dst, view, hw, h * hw, tid, NULL);
            sync(b); /* reads done before the buffer is written again */
        }
    }
}

rocke_attn_bwd_seq_mode_t seq_mode_of(const rocke_attn_bwd_spec_t& s)
{
    return eq(s.seq_mode, "thd") ? ROCKE_ATTN_BWD_SEQ_THD : ROCKE_ATTN_BWD_SEQ_BATCHED;
}

/* dkv_store_rule(b, k, seq_kv.length, seq_mode=, s_kv_max=a["S_kv_max"]) */
void store_rule(Main& m, const Kargs& a, const Cta& c, V k, V* store, V* row_live)
{
    rocke_int_or_value_t smax = rocke_iov_value(a["S_kv_max"]);
    need_st(m.b,
            rocke_attn_bwd_band_dkv_store_rule(
                m.b, k, c.seq_kv.length, seq_mode_of(m.s), &smax, store, row_live));
}

/* _store_dkv_rows */
void store_dkv_rows(Main& m,
                    const Kargs& a,
                    const Cta& c,
                    const rocke_attn_bwd_row_slab_t& slab,
                    const rocke_attn_bwd_dkv_reader_t& rd,
                    V k0,
                    V tid,
                    V zero)
{
    B* b = m.b;
    const rocke_attn_bwd_spec_t& s = m.s;
    int D = s.head_size, kn = s.block_n, vec = s.dkv_store_vec;
    int threads = threads_of(m);
    int per_row = D / vec;
    int total = kn * per_row;
    int rounds = rocke_attn_bwd_deal_chunks_count(b, total, threads);
    need(b, rounds >= 0);
    for(int it = 0; it < rounds; ++it)
    {
        bool tail = false;
        V chunk = needv(b, rocke_attn_bwd_deal_chunks_round(b, tid, total, threads, it, &tail));
        V r, col;
        rocke_attn_bwd_chunk_coords(b, chunk, per_row, vec, &r, &col);
        V k = rocke_b_add(b, k0, r);
        V store, row_live;
        store_rule(m, a, c, k, &store, &row_live);
        if(tail)
        {
            V ct = ci(b, total);
            V lt = rocke_b_cmp_lt(b, chunk, ct);
            store = rocke_b_land(b, store, lt);
        }
        IfScope f(b, store);
        V rv = needv(b, rocke_attn_bwd_dkv_reader_read(b, &rd, r, col));
        V v = rocke_b_select(b, row_live, rv, zero);
        rocke_attn_bwd_row_slab_store_vec(&slab, k, ivv(col), v);
        need(b, true);
    }
}

/* _store_dkv_cells */
void store_dkv_cells(Main& m,
                     const Kargs& a,
                     const Cta& c,
                     const rocke_attn_bwd_row_slab_t& slab,
                     const rocke_attn_bwd_dkv_row_t* cells,
                     int n_cells,
                     V k0)
{
    B* b = m.b;
    V zv = rocke_b_zero_vec(b, m.io, 1);
    V zero = rocke_b_vec_extract(b, zv, 0);
    for(int i = 0; i < n_cells; ++i)
    {
        V k = rocke_b_add(b, k0, cells[i].row);
        V store, row_live;
        store_rule(m, a, c, k, &store, &row_live);
        IfScope f(b, store);
        for(int j = 0; j < cells[i].n_items; ++j)
        {
            V v = rocke_b_select(b, row_live, cells[i].values[j], zero);
            rocke_attn_bwd_row_slab_store(&slab, k, ivv(cells[i].cols[j]), v);
            need(b, true);
        }
    }
}

/* _store_dkv_quads */
void store_dkv_quads(Main& m,
                     const Kargs& a,
                     const Cta& c,
                     const rocke_attn_bwd_row_slab_t& slab,
                     const rocke_attn_bwd_dkv_quad_t* quads,
                     int n_quads,
                     V k0)
{
    B* b = m.b;
    V zero = rocke_b_zero_vec(b, m.io, 4);
    for(int i = 0; i < n_quads; ++i)
    {
        V k = rocke_b_add(b, k0, quads[i].row);
        V store, row_live;
        store_rule(m, a, c, k, &store, &row_live);
        IfScope f(b, store);
        V p = rocke_b_vec_pack(b, quads[i].values, 4, rocke_f32());
        V v = rocke_b_vec_cast_f32_to(b, p, m.io);
        V sel = rocke_b_select(b, row_live, v, zero);
        rocke_attn_bwd_row_slab_store_vec(&slab, k, ivv(quads[i].col0), sel);
        need(b, true);
    }
}

/* _direct_epilogue; slabs NULL forms each slab here. */
void direct_epilogue(Main& m,
                     const Kargs& a,
                     Cta& c,
                     const rocke_attn_bwd_dkdv_acc_t& acc,
                     V k0,
                     V tid,
                     const rocke_attn_bwd_row_slab_t* slabs)
{
    B* b = m.b;
    const rocke_attn_bwd_spec_t& s = m.s;
    struct Out
    {
        const char* which;
        V mult;
    };
    Out outs[2] = {{"dk", a["dk_mult"]}, {"dv", NULL}};
    auto out_slab = [&](int i) {
        if(slabs != NULL)
            return slabs[i];
        return dkv_slab(m, a, c, outs[i].which, c.hk, k0);
    };
    if(eq(s.dkv_epilogue, "direct"))
    {
        for(int i = 0; i < 2; ++i)
        {
            rocke_attn_bwd_row_slab_t slab = out_slab(i);
            if(s.dkv_cell_vec == 4)
            {
                rocke_attn_bwd_dkv_quad_t* quads;
                int nq;
                need(b,
                     rocke_attn_bwd_tile_step_dkv_quad_rows(
                         &c.step, b, outs[i].which, &acc, outs[i].mult, &quads, &nq));
                store_dkv_quads(m, a, c, slab, quads, nq, k0);
                continue;
            }
            rocke_attn_bwd_dkv_row_t* cells;
            int n_cells;
            need(b,
                 rocke_attn_bwd_tile_step_dkv_cells(
                     &c.step, b, outs[i].which, &acc, outs[i].mult, &cells, &n_cells));
            store_dkv_cells(m, a, c, slab, cells, n_cells, k0);
        }
        return;
    }
    int vec = s.dkv_store_vec;
    V zero = needv(b, rocke_b_zero_vec(b, m.io, vec));
    if(eq(s.dkv_epilogue, "lds_staged_pair"))
    {
        /* dK in buffer 0 and dV in buffer 1, one barrier before the reads */
        rocke_attn_bwd_row_slab_t sl[2];
        rocke_attn_bwd_dkv_reader_t rd[2];
        for(int slot = 0; slot < 2; ++slot)
        {
            sl[slot] = out_slab(slot);
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
        }
        for(int i = 0; i < 2; ++i)
            store_dkv_rows(m, a, c, sl[i], rd[i], k0, tid, zero);
        sync(b); /* reads of the epilogue buffers done before they are reused */
        return;
    }
    for(int i = 0; i < 2; ++i)
    {
        rocke_attn_bwd_row_slab_t slab = out_slab(i);
        rocke_attn_bwd_dkv_reader_t rd;
        need(b,
             rocke_attn_bwd_tile_step_dkv_rows(
                 &c.step, b, outs[i].which, &acc, outs[i].mult, vec, 0, true, &rd));
        store_dkv_rows(m, a, c, slab, rd, k0, tid, zero);
        sync(b); /* reads of the epilogue buffer done before it is reused */
    }
}

/* inline_asm("", constraints, [v], result_type=v.type, sideeffect=, hint) */
V empty_asm(B* b, V v, const char* constraints, bool sideeffect, const char* hint)
{
    rocke_attr_map_t attrs = rocke_i_attrs(b);
    rocke_attr_set_str(b, &attrs, "template", "");
    rocke_attr_set_str(b, &attrs, "constraints", constraints);
    rocke_attr_set_bool(b, &attrs, "sideeffect", sideeffect);
    rocke_attr_set_bool(b, &attrs, "convergent", false);
    const rocke_type_t* ty = v->type;
    rocke_op_t* op
        = rocke_b_op(b, ROCKE_OP_TILE_INLINE_ASM, &v, 1, &ty, 1, &attrs, NULL, 0, hint, NULL);
    need(b, op != NULL && op->num_results == 1);
    return op->results[0];
}

/* ----------------------------------------------------------------- _kv_tile */

/* x_addr.slab(h, row0, clamp=) of tensor `which` (SLAB_*): the dense shell's
 * source when it set one, else the CTA's StridedTensorAddr. */
rocke_attn_bwd_row_slab_t cta_slab(Cta& c, int which, IV h, IV row0, bool clamp)
{
    if(c.slab_fn)
        return c.slab_fn(which, h, row0, clamp);
    const rocke_attn_bwd_strided_tensor_addr_t* t = which == SLAB_Q    ? &c.q_addr
                                                    : which == SLAB_DO ? &c.do_addr
                                                    : which == SLAB_K  ? &c.k_addr
                                                                       : &c.v_addr;
    return rocke_attn_bwd_strided_tensor_addr_slab(t, h, row0, clamp);
}

struct KvTile
{
    Main& m;
    const Kargs& a;
    Cta& c;
    B* b;
    V kt;
    V zero, one, k0;
    V pack_on = NULL, per = NULL, rows_per_head = NULL, n_grp;
    int n_acc;
    int fm, fn;
    rocke_attn_bwd_dkdv_acc_t acc0;
    bool counted;
    bool carry = false;
    V head_d[2], row_d[2];
    std::deque<rocke_attn_bwd_packed_step_t> packed;
    std::deque<Tile> tiles;

    KvTile(Main& mm, const Kargs& aa, Cta& cc)
        : m(mm)
        , a(aa)
        , c(cc)
        , b(mm.b)
    {
    }

    /* carried(prefix, acc, qt0, grp=, pend=) */
    IterArgs carried(const std::string& prefix,
                     const rocke_attn_bwd_dkdv_acc_t& acc,
                     V qt0,
                     bool grp,
                     const rocke_attn_bwd_pending_dq_t* pend)
    {
        std::vector<V> vals = acc_flat(acc);
        size_t half = vals.size() / 2;
        IterArgs out;
        for(size_t i = 0; i < half; ++i)
            out.add(prefix + "dk" + std::to_string(i), vals[i]);
        for(size_t i = 0; i < half; ++i)
            out.add(prefix + "dv" + std::to_string(i), vals[half + i]);
        if(qt0 != NULL)
            out.add(prefix + "qt", qt0);
        if(grp)
            out.add(prefix + "grp", zero);
        if(pend != NULL)
        {
            std::vector<std::string> pn = pend_names(b, *pend, prefix);
            std::vector<V> pv = pend_flat(*pend);
            for(size_t i = 0; i < pv.size(); ++i)
                out.add(pn[i], pv[i]);
        }
        return out;
    }

    rocke_attn_bwd_dkdv_acc_t acc_of(V const* cur)
    {
        if(!n_acc)
            return acc0;
        rocke_attn_bwd_dkdv_acc_t out;
        need(b, rocke_attn_bwd_dkdv_acc_from_flat(b, cur, n_acc, fm, fn, &out));
        return out;
    }

    /* head_of(gi): (h, first); first NULL is None. */
    void head_of(V gi, V* h, V* first)
    {
        if(m.s.head_pack == 1)
        {
            V f = rocke_b_mul(b, gi, per);
            *h = rocke_b_add(b, c.h0, f);
            *first = f;
            return;
        }
        *h = rocke_b_add(b, c.h0, gi);
        *first = NULL;
    }
};

/* One q-step range loop (q_steps of _kv_tile). */
struct QSteps
{
    KvTile& t;
    B* b;
    bool edge;
    std::string name;
    V end;
    bool has_wrap;
    V wrap_lo, wrap_hi;
    const V* group; /* (gi, h, first) or NULL */
    V start0, qt0;

    QSteps(
        KvTile& tt, bool edge_, const std::string& name_, V end_, const V* wrap_at, const V* group_)
        : t(tt)
        , b(tt.b)
        , edge(edge_)
        , name(name_)
        , end(end_)
        , group(group_)
    {
        has_wrap = wrap_at != NULL;
        wrap_lo = has_wrap ? wrap_at[0] : NULL;
        wrap_hi = has_wrap ? wrap_at[1] : NULL;
    }

    /* advance(qt, gi, with_wrap=): idx[0..n) and *wrap (with_wrap). */
    std::vector<V> advance(V qt, V gi, V* wrap_out)
    {
        const rocke_attn_bwd_spec_t& s = t.m.s;
        V one = t.one, zero = t.zero;
        if(group != NULL)
        {
            V qt_next = rocke_b_add(b, qt, one);
            if(s.q_rotate)
            {
                V ge = rocke_b_cmp_ge(b, qt_next, end);
                qt_next = rocke_b_select(b, ge, start0, qt_next);
            }
            return {qt_next};
        }
        if(!has_wrap)
        {
            V nxt = rocke_b_add(b, gi, one);
            V wrap = rocke_b_cmp_ge(b, nxt, t.n_grp);
            V inc = rocke_b_select(b, wrap, one, zero);
            V qt_next = rocke_b_add(b, qt, inc);
            if(s.q_rotate)
            {
                V ge = rocke_b_cmp_ge(b, qt_next, end);
                qt_next = rocke_b_select(b, ge, start0, qt_next);
            }
            V gi_next = rocke_b_select(b, wrap, zero, nxt);
            if(wrap_out)
                *wrap_out = wrap;
            return {qt_next, gi_next};
        }
        V nxt = rocke_b_add(b, qt, one);
        V wrap = rocke_b_cmp_ge(b, nxt, wrap_hi);
        V qt_next = rocke_b_select(b, wrap, wrap_lo, nxt);
        if(s.q_rotate)
            wrap = rocke_b_cmp_eq(b, qt_next, qt0);
        V inc = rocke_b_select(b, wrap, one, zero);
        V gn = rocke_b_add(b, gi, inc);
        if(wrap_out)
            *wrap_out = wrap;
        return {qt_next, gn};
    }

    /* inputs(qt, gi, slabs=) */
    Inputs inputs(V qt, V gi, bool slabs)
    {
        const rocke_attn_bwd_spec_t& s = t.m.s;
        V ckm = ci(b, s.block_m);
        V q0 = rocke_b_mul(b, qt, ckm);
        V h, first;
        if(group == NULL)
            t.head_of(gi, &h, &first);
        else
        {
            h = group[1];
            first = group[2];
        }
        Inputs in;
        memset(&in, 0, sizeof(in));
        in.q0 = q0;
        in.h = h;
        if(s.head_pack == 1)
        {
            V d = rocke_b_sub(b, t.c.n_h, first);
            V nh = rocke_b_smin(b, t.per, d);
            t.packed.emplace_back();
            rocke_attn_bwd_packed_step_t& ps = t.packed.back();
            ps.rows.head0 = h;
            ps.rows.n_heads = nh;
            ps.rows.rows_per_head = t.rows_per_head;
            ps.rows.row0 = q0;
            ps.q_addr = &t.c.q_addr;
            ps.do_addr = &t.c.do_addr;
            in.packed = &ps;
            need(b, true);
            return in;
        }
        if(!slabs)
        {
            need(b, true);
            return in;
        }
        bool clamp = edge || s.interior_clamp;
        in.has_slabs = true;
        in.q_slab = cta_slab(t.c, SLAB_Q, ivv(h), ivv(q0), clamp);
        in.do_slab = cta_slab(t.c, SLAB_DO, ivv(h), ivv(q0), clamp);
        need(b, true);
        return in;
    }

    Tile* tile_of(const Inputs& in)
    {
        t.tiles.emplace_back();
        Tile* p = &t.tiles.back();
        p->q0 = in.q0;
        p->h = in.h;
        p->has_slabs = in.has_slabs;
        p->q_slab = in.q_slab;
        p->do_slab = in.do_slab;
        p->packed = in.packed;
        return p;
    }

    Tile* ring_tile(V qt, V gi)
    {
        Inputs in = inputs(qt, gi, true);
        return tile_of(in);
    }

    /* ahead_of(qt, gi, k): the index k steps after (qt, gi). */
    void ahead_of(V qt, V gi, int k, V* aq, V* ag)
    {
        V iq = qt, ig = gi;
        for(int i = 0; i < k; ++i)
        {
            std::vector<V> nxt = advance(iq, ig, NULL);
            iq = nxt[0];
            ig = group != NULL ? gi : nxt[1];
        }
        *aq = iq;
        *ag = ig;
    }

    /* step_loads(qt, gi) */
    rocke_attn_bwd_staged_step_t step_loads(V qt, V gi)
    {
        Inputs in = inputs(qt, gi, true);
        rocke_attn_bwd_staged_step_t out;
        need(b,
             rocke_attn_bwd_tile_step_counted_loads(&t.c.step,
                                                    b,
                                                    in.h,
                                                    in.q0,
                                                    in.has_slabs ? &in.q_slab : NULL,
                                                    in.has_slabs ? &in.do_slab : NULL,
                                                    in.packed,
                                                    edge,
                                                    &out));
        return out;
    }

    struct NextLoads
    {
        QSteps* q;
        V it, qt, gi;
        std::vector<V> nxt_idx;
    };

    static bool next_loads_fn(void* ctx, rocke_ir_builder_t* bb, rocke_attn_bwd_staged_step_t* out)
    {
        (void)bb;
        NextLoads* n = (NextLoads*)ctx;
        QSteps* q = n->q;
        B* b = q->b;
        V one = q->t.one;
        V nit = rocke_b_add(b, n->it, one);
        V last = rocke_b_cmp_ge(b, nit, q->trip);
        V nq = rocke_b_select(b, last, n->qt, n->nxt_idx[0]);
        V ngi = q->group != NULL ? n->gi : rocke_b_select(b, last, n->gi, n->nxt_idx[1]);
        *out = q->step_loads(nq, ngi);
        return true;
    }

    V trip = NULL;

    /* q_steps(acc, qt0, trip, ...): returns the accumulators; *pend in / out
     * (has_pend: stage_table_atomics). */
    rocke_attn_bwd_dkdv_acc_t run(const rocke_attn_bwd_dkdv_acc_t& acc,
                                  V qt0_in,
                                  V trip_,
                                  bool has_pend,
                                  rocke_attn_bwd_pending_dq_t* pend)
    {
        Main& m = t.m;
        const rocke_attn_bwd_spec_t& s = m.s;
        Cta& c = t.c;
        V zero = t.zero, one = t.one;
        trip = trip_;
        start0 = qt0_in;
        qt0 = qt0_in;
        if(s.q_rotate)
        {
            V d = rocke_b_sub(b, end, start0);
            V mx = rocke_b_smax(b, d, one);
            V md = rocke_b_mod(b, t.kt, mx);
            qt0 = rocke_b_add(b, start0, md);
        }
        V gi0 = group == NULL ? zero : group[0];
        bool merge = eq(s.barrier_merge, "on");
        if(merge)
        {
            /* ring prologue: the loop's first step into slot 0, then barrier P */
            V lt = rocke_b_cmp_lt(b, zero, trip);
            {
                IfScope f(b, lt);
                Inputs in = inputs(qt0, gi0, true);
                need(b, rocke_attn_bwd_tile_step_ring_prologue(&c.step, b, &in));
            }
            need(b, rocke_attn_bwd_tile_step_ring_publish(&c.step, b));
        }
        rocke_value_t** carried_regs = NULL;
        int n_regs = 0;
        if(c.has_ring)
        {
            std::vector<Tile> pro;
            pro.push_back(*ring_tile(qt0, gi0));
            if(c.ring.depth == 3)
            {
                /* the second step of the loop, or the first again (one trip) */
                V aq, ag;
                ahead_of(qt0, gi0, 1, &aq, &ag);
                V two = rocke_b_cmp_lt(b, one, trip);
                if(group == NULL)
                    ag = rocke_b_select(b, two, ag, gi0);
                V sq = rocke_b_select(b, two, aq, qt0);
                pro.push_back(*ring_tile(sq, ag));
            }
            if(eq(c.ring.path, "dma") && c.ring.depth == 1)
                pro.clear();
            std::string pname = "bwd_" + name;
            need(b,
                 rocke_attn_bwd_q_ring_prologue(&c.ring,
                                                b,
                                                trip,
                                                pro.data(),
                                                (int)pro.size(),
                                                pname.c_str(),
                                                &carried_regs,
                                                &n_regs));
        }
        int n_idx = group != NULL ? 1 : 2;
        std::string prefix = "bwd_" + name;
        IterArgs iter_args = t.carried(prefix, acc, qt0, group == NULL, NULL);
        for(int i = 0; i < n_regs; ++i)
            iter_args.add(prefix + "r" + std::to_string(i), carried_regs[i]);
        if(t.counted)
        {
            /* the first step's loads, guarded so an empty range loads nothing */
            V gt = rocke_b_cmp_gt(b, trip, zero);
            V gtrip = rocke_b_select(b, gt, one, zero);
            int nz = rocke_attn_bwd_tile_step_counted_zero_len(&c.step);
            std::vector<V> zs((size_t)(nz ? nz : 1));
            need(b, rocke_attn_bwd_tile_step_counted_zero(&c.step, b, zs.data()));
            IterArgs gargs;
            for(int i = 0; i < nz; ++i)
                gargs.add(prefix + "pz" + std::to_string(i), zs[(size_t)i]);
            rocke_for_t guard = for_iter(b, zero, gtrip, one, gargs, prefix + "pzit", true);
            rocke_b_region_enter(b, guard.body);
            {
                rocke_attn_bwd_staged_step_t st = step_loads(qt0, gi0);
                yield(b, staged_flat(st));
            }
            rocke_b_region_leave(b);
            need(b, true);
            rocke_b_s_waitcnt(b, 0, -1, -1);
            std::vector<V> gres = results_of(guard);
            for(size_t i = 0; i < gres.size(); ++i)
                iter_args.add(prefix + "pf" + std::to_string(i), gres[i]);
        }
        int n_carry = 0;
        V wrap_d[2] = {NULL, NULL}, step_d[2] = {NULL, NULL};
        if(t.carry)
        {
            V h_first, first_unused;
            t.head_of(gi0, &h_first, &first_unused);
            V ckm = ci(b, s.block_m);
            V row_first = rocke_b_mul(b, qt0, ckm);
            V oq = needv(b,
                         rocke_attn_bwd_strided_tensor_addr_offset_bytes(
                             &c.q_addr, ivv(h_first), ivv(row_first)));
            iter_args.add(prefix + "oq", oq);
            V od = needv(b,
                         rocke_attn_bwd_strided_tensor_addr_offset_bytes(
                             &c.do_addr, ivv(h_first), ivv(row_first)));
            iter_args.add(prefix + "od", od);
            n_carry = 2;
            if(!has_wrap)
            {
                V g1 = rocke_b_sub(b, t.n_grp, one);
                V back = rocke_b_sext(b, g1, rocke_i64());
                for(int k = 0; k < 2; ++k)
                {
                    V mul = rocke_b_mul(b, back, t.head_d[k]);
                    wrap_d[k] = rocke_b_sub(b, t.row_d[k], mul);
                    step_d[k] = t.head_d[k];
                }
            }
            else
            {
                V span = rocke_b_sub(b, wrap_hi, wrap_lo);
                V s1 = rocke_b_sub(b, span, one);
                V back = rocke_b_sext(b, s1, rocke_i64());
                for(int k = 0; k < 2; ++k)
                {
                    V mul = rocke_b_mul(b, back, t.row_d[k]);
                    wrap_d[k] = rocke_b_sub(b, t.head_d[k], mul);
                    step_d[k] = t.row_d[k];
                }
            }
        }
        int n_pend = 0;
        if(has_pend)
        {
            std::vector<V> pv = pend_flat(*pend);
            std::vector<std::string> pn = pend_names(b, *pend, prefix);
            n_pend = (int)pv.size();
            for(size_t i = 0; i < pv.size(); ++i)
                iter_args.add(pn[i], pv[i]);
        }
        need(b, true);
        rocke_for_t loop = for_iter(b, zero, trip, one, iter_args, prefix + "it", false);
        V it = loop.iv;
        V* cur = loop.iter_vars;
        int n_cur = loop.num_iter_vars;
        rocke_b_region_enter(b, loop.body);
        {
            int n_acc = t.n_acc;
            V qt = cur[n_acc];
            V gi = group != NULL ? group[0] : cur[n_acc + 1];
            std::vector<V> carry_yields;
            V wrap = NULL;
            std::vector<V> nxt_idx = advance(qt, gi, t.carry ? &wrap : NULL);
            Inputs here = inputs(qt, gi, !(t.counted || t.carry));
            V h = here.h;
            int body_end = n_cur - n_pend - n_carry;
            if(t.carry)
            {
                V offs[2] = {cur[body_end], cur[body_end + 1]};
                bool clamp = edge || s.interior_clamp;
                Inputs h2;
                memset(&h2, 0, sizeof(h2));
                h2.q0 = here.q0;
                h2.h = h;
                h2.has_slabs = true;
                h2.q_slab = rocke_attn_bwd_strided_tensor_addr_slab_at(
                    &c.q_addr, offs[0], here.q0, clamp);
                h2.do_slab = rocke_attn_bwd_strided_tensor_addr_slab_at(
                    &c.do_addr, offs[1], here.q0, clamp);
                here = h2;
                for(int k = 0; k < 2; ++k)
                {
                    V sel = rocke_b_select(b, wrap, wrap_d[k], step_d[k]);
                    carry_yields.push_back(rocke_b_add(b, offs[k], sel));
                }
                need(b, true);
            }
            if(group == NULL && !has_pend)
                need(b, rocke_attn_bwd_dq_sink_begin_head(c.step.sink, b, h));
            rocke_attn_bwd_pending_dq_t cur_pend;
            if(has_pend)
            {
                std::vector<V> pv(cur + body_end + n_carry, cur + n_cur);
                cur_pend = pend_from_flat(b, *pend, pv);
            }
            rocke_attn_bwd_q_step_opts_t o = rocke_attn_bwd_q_step_opts_default();
            o.edge = edge;
            o.packed = here.packed;
            o.pending = has_pend ? &cur_pend : NULL;
            rocke_attn_bwd_q_step_result_t res;
            if(t.counted)
            {
                std::vector<V> sv(cur + n_acc + n_idx, cur + body_end);
                V* keep = arena_copy(b, sv);
                rocke_attn_bwd_staged_step_t staged = rocke_attn_bwd_staged_step_from_flat(
                    keep, (int)sv.size(), rocke_attn_bwd_tile_step_counted_n_stats(&c.step));
                NextLoads nl;
                nl.q = this;
                nl.it = it;
                nl.qt = qt;
                nl.gi = gi;
                nl.nxt_idx = nxt_idx;
                o.staged = &staged;
                o.next_loads = next_loads_fn;
                o.next_loads_ctx = &nl;
                rocke_attn_bwd_dkdv_acc_t acc_cur = t.acc_of(cur);
                need(b,
                     rocke_attn_bwd_tile_step_run_q_step(
                         &c.step, b, &acc_cur, NULL, NULL, qt, h, t.k0, &o, &res));
                std::vector<V> ys = acc_flat(res.acc);
                ys.insert(ys.end(), nxt_idx.begin(), nxt_idx.end());
                std::vector<V> sf = staged_flat(res.staged);
                ys.insert(ys.end(), sf.begin(), sf.end());
                if(has_pend)
                {
                    std::vector<V> pf = pend_flat(res.pending);
                    ys.insert(ys.end(), pf.begin(), pf.end());
                }
                yield(b, ys);
            }
            else
            {
                rocke_attn_bwd_ring_step_t merged;
                if(merge)
                {
                    /* The next step of this loop (the step itself on the last
                     * trip) and the ring slot. */
                    V nit = rocke_b_add(b, it, one);
                    V more = rocke_b_cmp_lt(b, nit, trip);
                    V nqt = rocke_b_select(b, more, nxt_idx[0], qt);
                    V ngi = group != NULL ? gi : rocke_b_select(b, more, nxt_idx[1], gi);
                    V c2 = ci(b, 2);
                    merged.slot = rocke_b_mod(b, it, c2);
                    merged.nxt = inputs(nqt, ngi, true);
                    o.ring = &merged;
                }
                rocke_attn_bwd_prefetch_step_t pf;
                if(c.has_ring)
                {
                    Tile* tile = tile_of(here);
                    Tile* ahead = tile;
                    int k = c.ring.schedule.ahead;
                    if(k)
                    {
                        V aq, ag;
                        ahead_of(qt, gi, k, &aq, &ag);
                        V ck = ci(b, k);
                        V ik = rocke_b_add(b, it, ck);
                        V more = rocke_b_cmp_lt(b, ik, trip);
                        if(group == NULL)
                            ag = rocke_b_select(b, more, ag, gi);
                        V sq = rocke_b_select(b, more, aq, qt);
                        ahead = ring_tile(sq, ag);
                    }
                    std::vector<V> cv(cur + n_acc + n_idx, cur + body_end);
                    V* keep = arena_copy(b, cv);
                    pf = rocke_attn_bwd_prefetch_step_make(
                        &c.ring, it, tile, ahead, keep, (int)cv.size());
                    o.prefetch = &pf;
                }
                rocke_attn_bwd_dkdv_acc_t acc_cur = t.acc_of(cur);
                need(b,
                     rocke_attn_bwd_tile_step_run_q_step(&c.step,
                                                         b,
                                                         &acc_cur,
                                                         here.has_slabs ? &here.q_slab : NULL,
                                                         here.has_slabs ? &here.do_slab : NULL,
                                                         qt,
                                                         h,
                                                         t.k0,
                                                         &o,
                                                         &res));
                std::vector<V> ys = acc_flat(res.acc);
                ys.insert(ys.end(), nxt_idx.begin(), nxt_idx.end());
                if(c.has_ring)
                    for(int i = 0; i < pf.n_yields; ++i)
                        ys.push_back(pf.yields[i]);
                ys.insert(ys.end(), carry_yields.begin(), carry_yields.end());
                if(has_pend)
                {
                    std::vector<V> pv = pend_flat(res.pending);
                    ys.insert(ys.end(), pv.begin(), pv.end());
                }
                yield(b, ys);
            }
        }
        rocke_b_region_leave(b);
        need(b, true);
        if(c.has_ring)
            need(b, rocke_attn_bwd_q_ring_drain(&c.ring, b));
        std::vector<V> results = results_of(loop);
        std::vector<V> res(results.begin(), results.begin() + t.n_acc);
        if(has_pend)
        {
            std::vector<V> pv(results.end() - n_pend, results.end());
            *pend = pend_from_flat(b, *pend, pv);
        }
        if(!t.n_acc)
            return t.acc0;
        rocke_attn_bwd_dkdv_acc_t out;
        V* keep = arena_copy(b, res);
        need(b, rocke_attn_bwd_dkdv_acc_from_flat(b, keep, t.n_acc, t.fm, t.fn, &out));
        return out;
    }
};

} // namespace

/* _kv_tile(b, s, a, step, mask, kt=, ..., epilogue=, ring=, tag=) */
void main_kernel::kv_tile(
    Main& m, const Kargs& a, Cta& c, V kt, const std::string& tag, const KvEpilogue* epilogue)
{
    B* b = m.b;
    const rocke_attn_bwd_spec_t& s = m.s;
    KvTile t(m, a, c);
    t.kt = kt;
    int km = s.block_m, kn = s.block_n;
    t.zero = ci(b, 0);
    t.one = ci(b, 1);
    V zero = t.zero, one = t.one;
    V ckn = ci(b, kn);
    V k0 = rocke_b_mul(b, kt, ckn);
    t.k0 = k0;
    V live_ = rocke_b_cmp_lt(b, k0, c.seq_kv.length); /* CTA-uniform */
    if(c.kv_rows_fn)
    {
        /* k_addr / v_addr whose slab is only a row source (load_vec) */
        rocke_attn_bwd_row_source_t k_rows = c.kv_rows_fn(SLAB_K, ivv(c.hk), ivv(k0), true);
        rocke_attn_bwd_row_source_t v_rows = c.kv_rows_fn(SLAB_V, ivv(c.hv), ivv(k0), true);
        need(b, true);
        need(b, rocke_attn_bwd_tile_step_stage_kv_rows(&c.step, b, &k_rows, &v_rows, k0, live_));
    }
    else
    {
        rocke_attn_bwd_row_slab_t k_slab = cta_slab(c, SLAB_K, ivv(c.hk), ivv(k0), true);
        rocke_attn_bwd_row_slab_t v_slab = cta_slab(c, SLAB_V, ivv(c.hv), ivv(k0), true);
        need(b, true);
        need(b, rocke_attn_bwd_tile_step_stage_kv(&c.step, b, &k_slab, &v_slab, k0, live_));
    }

    if(s.head_pack == 1)
    {
        V pk = a["pack_heads"];
        t.pack_on = rocke_b_cmp_ne(b, pk, zero);
        t.per = rocke_b_select(b, t.pack_on, pk, one);
        V mx = rocke_b_smax(b, c.seq_q.length, one);
        V ckm = ci(b, km);
        t.rows_per_head = rocke_b_select(b, t.pack_on, mx, ckm);
        V p1 = rocke_b_sub(b, t.per, one);
        V nh = rocke_b_add(b, c.n_h, p1);
        t.n_grp = rocke_b_div(b, nh, t.per);
    }
    else
        t.n_grp = c.n_h;

    /* Inverse band (empty band, a tile past len_kv or len_q == 0: 0 tiles). */
    V qt_start, n_qt;
    need_st(b, rocke_attn_bwd_mask_q_tile_range(&c.mask, b, k0, &qt_start, &n_qt));
    bool head_outer = eq(s.step_order, "head_outer");
    V qt_end = rocke_b_add(b, qt_start, n_qt);
    struct Range
    {
        V s0, e;
        bool edge;
    };
    std::vector<Range> ranges;
    if(s.edge_tiles == 1)
    {
        V i0, i1;
        if(c.has_static && !c.any_edge)
        {
            /* _StaticClassMixin.interior_q_range: no edge tile anywhere */
            i0 = qt_start;
            i1 = rocke_b_add(b, qt_start, n_qt);
        }
        else if(c.has_static && !c.any_interior)
        {
            i0 = qt_start;
            i1 = qt_start;
        }
        else
            need_st(b,
                    rocke_attn_bwd_mask_interior_q_range(&c.mask, b, k0, qt_start, n_qt, &i0, &i1));
        if(t.pack_on != NULL)
            i1 = rocke_b_select(b, t.pack_on, i0, i1);
        ranges = {{qt_start, i0, true}, {i0, i1, false}, {i1, qt_end, true}};
        /* A policy that classified every tile at build time (the dense
         * family's) drops the runs it proved empty (mask.static_runs). */
        if(c.has_static)
        {
            std::vector<Range> kept;
            for(const Range& r : ranges)
                if(r.edge ? c.any_edge : c.any_interior)
                    kept.push_back(r);
            ranges = kept;
        }
    }
    else
        ranges = {{qt_start, qt_end, true}};
    bool grouped = head_outer && ranges.size() > 1;
    std::vector<V> trips;
    V n_outer = NULL;
    if(grouped)
    {
        for(const Range& r : ranges)
        {
            V d = rocke_b_sub(b, r.e, r.s0);
            trips.push_back(rocke_b_readfirstlane(b, d));
        }
        V lt = rocke_b_cmp_lt(b, zero, n_qt);
        V sel = rocke_b_select(b, lt, t.n_grp, zero);
        n_outer = rocke_b_readfirstlane(b, sel);
    }
    else
        for(const Range& r : ranges)
        {
            V d = rocke_b_sub(b, r.e, r.s0);
            V p = rocke_b_mul(b, t.n_grp, d);
            trips.push_back(rocke_b_readfirstlane(b, p));
        }
    need(b, true);
    bool hoist = s.dkv_slab_hoist && eq(s.dkv_mode, "direct") && eq(s.seq_mode, "thd");
    rocke_attn_bwd_row_slab_t dkv_slabs[2];
    if(hoist)
    {
        /* THD: the dK / dV slabs before the q loops, each pointer opaque */
        const char* which[2] = {"dk", "dv"};
        for(int i = 0; i < 2; ++i)
        {
            rocke_attn_bwd_row_slab_t sl = dkv_slab(m, a, c, which[i], c.hk, k0);
            sl.ptr = empty_asm(b, sl.ptr, "=s,0", false, "dkvptr");
            sl.has_parts = false; /* parts[2] is not the new pointer */
            dkv_slabs[i] = sl;
        }
    }
    need(b, rocke_attn_bwd_tile_step_zero_acc(&c.step, b, &t.acc0));
    t.fm = rocke_attn_bwd_block_gemm_frags_m(&m.geom.g1);
    t.fn = rocke_attn_bwd_block_gemm_frags_n(&m.geom.g1);
    t.n_acc = rocke_attn_bwd_dkdv_acc_flat_len(&t.acc0);
    bool has_pend = eq(s.sched, "stage_table_atomics");
    rocke_attn_bwd_pending_dq_t pend0;
    if(has_pend)
    {
        if(s.head_pack == 1)
            need(b,
                 rocke_attn_bwd_tile_step_zero_pending_dq(
                     &c.step, b, c.h0, one, t.rows_per_head, &pend0));
        else
            need(b, rocke_attn_bwd_tile_step_zero_pending_dq(&c.step, b, c.h0, NULL, NULL, &pend0));
    }
    t.carry = s.slab_carry && s.head_pack != 1 && !grouped;
    if(t.carry)
    {
        t.head_d[0] = needv(b, rocke_attn_bwd_strided_tensor_addr_stride_bytes(&c.q_addr, 'h'));
        t.head_d[1] = needv(b, rocke_attn_bwd_strided_tensor_addr_stride_bytes(&c.do_addr, 'h'));
        const rocke_attn_bwd_strided_tensor_addr_t* addrs[2] = {&c.q_addr, &c.do_addr};
        for(int k = 0; k < 2; ++k)
        {
            V ckm = rocke_b_const_i64(b, km);
            V st = needv(b, rocke_attn_bwd_strided_tensor_addr_stride_bytes(addrs[k], 't'));
            t.row_d[k] = rocke_b_mul(b, ckm, st);
        }
    }
    t.counted = eq(s.loop_sync, "counted");

    rocke_attn_bwd_dkdv_acc_t acc = t.acc0;
    rocke_attn_bwd_pending_dq_t pend = pend0;
    const char* suffixes[3] = {"q", "qi_", "qe_"};
    if(!grouped)
    {
        for(size_t i = 0; i < ranges.size(); ++i)
        {
            V wrap_at[2] = {ranges[i].s0, ranges[i].e};
            QSteps q(t,
                     ranges[i].edge,
                     tag + suffixes[i],
                     ranges[i].e,
                     head_outer ? wrap_at : NULL,
                     NULL);
            acc = q.run(acc, ranges[i].s0, trips[i], has_pend, &pend);
        }
    }
    else
    {
        std::string gprefix = "bwd_" + tag + "g";
        IterArgs oargs = t.carried(gprefix, t.acc0, NULL, true, has_pend ? &pend0 : NULL);
        int n_p0 = has_pend ? rocke_attn_bwd_pending_dq_flat_len(&pend0) : 0;
        rocke_for_t outer = for_iter(b, zero, n_outer, one, oargs, "bwd_" + tag + "git", false);
        V* cur = outer.iter_vars;
        int n_cur = outer.num_iter_vars;
        rocke_b_region_enter(b, outer.body);
        {
            V gi = cur[t.n_acc];
            V h, first;
            t.head_of(gi, &h, &first);
            rocke_attn_bwd_pending_dq_t ipend;
            if(!has_pend)
                need(b, rocke_attn_bwd_dq_sink_begin_head(c.step.sink, b, h));
            else
            {
                std::vector<V> pv(cur + n_cur - n_p0, cur + n_cur);
                ipend = pend_from_flat(b, pend0, pv);
            }
            rocke_attn_bwd_dkdv_acc_t inner = t.acc_of(cur);
            V group[3] = {gi, h, first};
            for(size_t i = 0; i < ranges.size(); ++i)
            {
                QSteps q(t, ranges[i].edge, tag + suffixes[i], ranges[i].e, NULL, group);
                inner = q.run(inner, ranges[i].s0, trips[i], has_pend, &ipend);
            }
            std::vector<V> ys = acc_flat(inner);
            ys.push_back(rocke_b_add(b, gi, one));
            if(has_pend)
            {
                std::vector<V> pv = pend_flat(ipend);
                ys.insert(ys.end(), pv.begin(), pv.end());
            }
            yield(b, ys);
        }
        rocke_b_region_leave(b);
        need(b, true);
        std::vector<V> results = results_of(outer);
        if(t.n_acc)
        {
            std::vector<V> res(results.begin(), results.begin() + t.n_acc);
            V* keep = arena_copy(b, res);
            need(b, rocke_attn_bwd_dkdv_acc_from_flat(b, keep, t.n_acc, t.fm, t.fn, &acc));
        }
        else
            acc = t.acc0;
        if(has_pend)
        {
            std::vector<V> pv(results.end() - n_p0, results.end());
            pend = pend_from_flat(b, pend0, pv);
        }
    }
    if(has_pend)
        need(b,
             rocke_attn_bwd_tile_step_flush_dq(&c.step, b, &pend)); /* the last step's dQ atomics */
    rocke_attn_bwd_dkdv_acc_t fin;
    need(b, rocke_attn_bwd_tile_step_finish(&c.step, b, &acc, &fin));
    if(epilogue != NULL)
        (*epilogue)(fin, k0, live_);
    else if(eq(s.dkv_mode, "atomic") && eq(s.dkv_sink, "slices"))
        slices_epilogue(m, a, c, fin, k0, live_, c.tid);
    else if(eq(s.dkv_mode, "atomic"))
        atomic_epilogue(m, a, c, fin, k0, live_);
    else
        direct_epilogue(m, a, c, fin, k0, c.tid, hoist ? dkv_slabs : NULL);
}

namespace
{

/* ------------------------------------------------------ pass-1 roots */

/* _MirrorStash */
struct MirrorStash
{
    const Kargs* a;
    int wave;
    int n_ids;
    std::vector<std::string> keys;
    std::vector<V> regs;

    MirrorStash(B* b, const Kargs& aa, const V* ids, int wave_size)
        : a(&aa)
        , wave(wave_size)
        , n_ids(3)
    {
        for(const std::string& k : aa.order)
        {
            V v = aa.m.at(k);
            if(v != NULL && (is_int_type(v, rocke_i32()) || is_int_type(v, rocke_i64())))
                keys.push_back(k);
        }
        std::vector<V> words;
        for(const std::string& k : keys)
        {
            V v = aa.m.at(k);
            if(is_int_type(v, rocke_i32()))
                words.push_back(v);
            else
            {
                V lo = rocke_b_trunc(b, v, rocke_i32());
                V c32 = rocke_b_const_i64(b, 32);
                V sh = rocke_b_lshr(b, v, c32);
                V hi = rocke_b_trunc(b, sh, rocke_i32());
                words.push_back(lo);
                words.push_back(hi);
            }
        }
        for(int i = 0; i < 3; ++i)
            words.push_back(ids[i]);
        for(size_t c = 0; c < words.size(); c += (size_t)wave_size)
        {
            V reg = ci(b, 0);
            for(size_t j = 0; j < (size_t)wave_size && c + j < words.size(); ++j)
            {
                V lane = ci(b, (int64_t)j);
                reg = rocke_b_writelane(b, words[c + j], lane, reg);
            }
            regs.push_back(empty_asm(b, needv(b, reg), "=v,0", true, "stash"));
        }
    }

    /* (kernargs, block ids) read back from the stash. */
    Kargs restore(B* b, V ids_out[3])
    {
        size_t i = 0;
        auto word = [&]() -> V {
            size_t k = i++;
            if(k >= regs.size() * (size_t)wave)
                throw ckc::ValueError("StopIteration");
            V lane = ci(b, (int64_t)(k % (size_t)wave));
            return rocke_b_readlane(b, regs[k / (size_t)wave], lane);
        };
        Kargs a1 = *a;
        for(const std::string& k : keys)
        {
            if(is_int_type(a->m.at(k), rocke_i32()))
                a1.m[k] = word();
            else
            {
                V lo = word();
                V hi = word();
                V zl = rocke_b_zext(b, lo, rocke_i64());
                V zh = rocke_b_zext(b, hi, rocke_i64());
                V c32 = rocke_b_const_i64(b, 32);
                V sh = rocke_b_shl(b, zh, c32);
                a1.m[k] = rocke_b_lor(b, zl, sh);
            }
        }
        for(int j = 0; j < n_ids; ++j)
            ids_out[j] = word();
        need(b, true);
        return a1;
    }
};

/* _not_built(s) */
void check_not_built(const rocke_attn_bwd_spec_t& s)
{
    if(eq(s.dkv_mode, "atomic") && eq(s.dkv_sink, "last_cta"))
        throw ckc::NotImplError(
            "the layout-general attention backward main kernel is not built for "
            "dkv_sink='last_cta'");
}

/* _emit_main(spec, arch, main_kernel_refusals=False) after validation, on
 * the initialised builder. */
rocke_kernel_def_t* emit_main(Main& m)
{
    B* b = m.b;
    const rocke_attn_bwd_spec_t& s = m.s;
    check_not_built(s);
    m.facts = arch_facts(m.arch);
    need(b, rocke_attn_bwd_tile_geometry_from_spec(b, &m.s, m.arch.c_str(), &m.geom));
    m.plan = lds_plan(s, m.arch);
    m.io = rocke_attn_bwd_ir_type(b, s.dtype);
    need(b, m.io != NULL);

    Kargs a = declare_main(b, s, m.io);
    /* Multipliers that are 1.0 under the compiled placement are not applied. */
    unsigned unit = 0;
    rocke_attn_bwd_err_t e;
    need_err(rocke_attn_bwd_static_unit_mults(s.static_scale, s.dkv_mode, &unit, &e), e);
    if(unit & ROCKE_ATTN_BWD_UNIT_DS_MULT)
        a.m["ds_mult"] = NULL;
    if(unit & ROCKE_ATTN_BWD_UNIT_DK_MULT)
        a.m["dk_mult"] = NULL;
    if(unit & ROCKE_ATTN_BWD_UNIT_DQ_MULT)
        a.m["dq_mult"] = NULL;
    int threads = threads_of(m);
    rocke_attr_set_int(
        b, &b->kernel->attrs, "max_workgroup_size", threads > MIN_WG_ATTR ? threads : MIN_WG_ATTR);
    if(eq(m.facts.matrix_path, "mfma") && s.agpr_alloc.kind == ROCKE_ATTN_BWD_AGPR_PAIR)
    {
        int64_t pair[2] = {s.agpr_alloc.a, s.agpr_alloc.b};
        rocke_attr_set_int_list(b, &b->kernel->attrs, "agpr_alloc", pair, 2);
    }
    if(eq(s.agpr_route, "vgpr_form") || eq(s.agpr_route, "agpr_fence")
       || eq(s.agpr_route, "asm_g13"))
        rocke_attr_set_bool(b, &b->kernel->attrs, "mfma_vgpr_form", true);
    if(s.waves_per_eu != ROCKE_ATTN_BWD_NONE)
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", s.waves_per_eu);
    if(s.scheduler_strategy != NULL)
        rocke_attr_set_str(b, &b->kernel->attrs, "scheduler_strategy", s.scheduler_strategy);
    need(b, true);

    V tid, lane, wave;
    rocke_attn_bwd_lane_and_wave(b, m.facts.wave_size, NULL, &tid, &lane, &wave);
    need(b, true);
    rocke_attn_bwd_bwd_lds_t lds;
    std::unique_ptr<Cta> cta(new Cta());
    cta_setup(m, a, *cta, tid, lane, wave, NULL, NULL, &lds);
    /* pass 1's integer roots parked in VGPR lanes before pass 0's loops */
    std::unique_ptr<MirrorStash> stash;
    if(s.lb_mirror)
        stash.reset(new MirrorStash(b, a, cta->ids, m.facts.wave_size));
    kv_tile(m, a, *cta, cta->work.first_kt, "", NULL);
    if(s.lb_mirror)
    {
        V c2 = ci(b, 2);
        V two = rocke_b_cmp_eq(b, cta->work.n_pass, c2); /* CTA-uniform */
        IfScope f(b, two);
        sync(b);
        /* Pass 1 recomputes every value from fresh copies of its roots. */
        V ids1[3];
        Kargs a1 = stash->restore(b, ids1);
        V tid1 = empty_asm(b, tid, "=v,0", false, "m1");
        V tid_out, lane1, wave1;
        rocke_attn_bwd_lane_and_wave(b, m.facts.wave_size, tid1, &tid_out, &lane1, &wave1);
        need(b, true);
        std::unique_ptr<Cta> cta1(new Cta());
        cta_setup(m, a1, *cta1, tid1, lane1, wave1, ids1, cta->lds, NULL);
        kv_tile(m, a1, *cta1, cta1->work.second_kt, "m1_", NULL);
    }
    rocke_b_ret(b);
    need(b, true);
    return rocke_ir_builder_kernel(b);
}

} // namespace
} // namespace rocke_attn_bwd

using namespace rocke_attn_bwd;

extern "C" {

rocke_kernel_def_t* rocke_build_attn_bwd_main_new(rocke_ir_builder_t* b,
                                                  const rocke_attn_bwd_spec_t* spec,
                                                  const char* arch)
{
    if(b == NULL)
        return NULL;
    /* validate_attn_bwd_spec raises before the Python builder exists; the C
     * builder must exist to carry the error, so a refused spec gets a
     * placeholder name. */
    std::unique_ptr<Main> m(new Main());
    std::string name;
    rocke_status_t code = ROCKE_OK;
    std::string msg;
    try
    {
        if(spec == NULL)
            throw ckc::ValueError("spec must not be NULL");
        m->arch = arch ? arch : "";
        m->s = validate_spec(*spec, m->arch);
        name = kernel_name(m->s, "main");
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
    if(rocke_ir_builder_init(b, code == ROCKE_OK ? name.c_str() : "rocke_attn_bwd_invalid")
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
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* { return emit_main(*m); });
}

} /* extern "C" */
