// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_dense_bwd_shell.py: the
 * BufferRowSlab / DenseFixedAddr addressing, the compile-time band
 * classification (dense_static_classes, DiagonalBand, DenseNoMask) and
 * emit_dense_bwd_main, which drives the general family's per-kv-tile driver
 * (main_kernel::kv_tile, rocke/instance_attention_bwd_main_internal.h).
 *
 * Byte identity: every builder call (constants included) is bound to a local
 * in Python's left-to-right, inner-first evaluation order, and every barrier
 * carries ROCKE_ATTN_BWD_BARRIER_FENCE (the Python builder's fence_barriers).
 */
#include "rocke/helper_rocke.instances.common._attention_dense_bwd_shell.h"

#include <stdio.h>
#include <string.h>

#include <memory>
#include <string>
#include <vector>

#include "rocke/error.hpp"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.instances.common._attention_bwd_frag.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_gemm.h"
#include "rocke/instance_attention_bwd_main_internal.h"
#include "rocke/instance_attention_dense_bwd_spec.h"
#include "rocke/ir_internal.h"

namespace rocke_attn_bwd
{
using namespace main_kernel;

namespace
{

typedef rocke_ir_builder_t B;
typedef rocke_value_t* V;
typedef rocke_attn_bwd_iv_t IV;

const int ELEM_BYTES = 2;
/* The lowering's flat-work-group-size attribute has a fixed minimum of 64. */
const int MIN_WG_ATTR = 64;

bool live(const B* b)
{
    return b != NULL && b->status == ROCKE_OK;
}

[[noreturn]] void raise_builder(B* b)
{
    if(b != NULL && b->status != ROCKE_OK)
        ckc::raise_status(b->status, b->err);
    throw ckc::ValueError("dense attention backward: a helper failed");
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

void need_st(B* b, rocke_status_t st)
{
    if(st != ROCKE_OK || !live(b))
        raise_builder(b);
}

bool eq(const char* a, const char* b)
{
    if(a == NULL || b == NULL)
        return a == b;
    return strcmp(a, b) == 0;
}

V ci(B* b, int64_t x)
{
    return rocke_b_const_i32(b, x);
}

IV ivv(V v)
{
    return rocke_attn_bwd_iv_val(v);
}

void sync(B* b)
{
    rocke_b_sync_fence(b, ROCKE_ATTN_BWD_BARRIER_FENCE);
    need(b, true);
}

/* BufferRowSlab._voffset(row, col) */
V buffer_voffset(const rocke_attn_bwd_row_slab_t* s, V row, IV col)
{
    B* b = s->b;
    V ts = ci(b, s->t_stride.i);
    V idx = rocke_b_mul(b, row, ts);
    if(!rocke_attn_bwd_iv_is_int(col, 0))
    {
        V c = col.v != NULL ? col.v : ci(b, col.i);
        idx = rocke_b_add(b, idx, c);
    }
    V eb = ci(b, ELEM_BYTES);
    return rocke_b_mul(b, idx, eb);
}

} // namespace

/* ------------------------------------------------------------ the kernel */

rocke_kernel_def_t* emit_dense_bwd_main(B* b,
                                        const rocke_attn_bwd_dense_spec_t& spec,
                                        const std::string& arch,
                                        const rocke_attn_bwd_spec_t& body)
{
    Main m;
    m.b = b;
    m.s = body;
    m.arch = arch;
    m.seq_guards = 0;
    m.sentinel = true;
    const rocke_attn_bwd_spec_t& s = m.s;
    m.facts = arch_facts(arch);
    need(b, rocke_attn_bwd_tile_geometry_from_spec(b, &m.s, arch.c_str(), &m.geom));
    m.plan = lds_plan(s, arch);
    m.io = rocke_attn_bwd_ir_type(b, s.dtype);
    need(b, m.io != NULL);
    const rocke_type_t* io = m.io;
    int km = s.block_m, kn = s.block_n;
    long long s_q = spec.seqlen_q, s_kv = spec.seqlen_kv;
    int h_q = spec.num_query_heads, h_k = spec.num_kv_heads;
    int group = h_q / h_k;
    long long n_kv_tiles = (s_kv + kn - 1) / kn;
    bool atomic = eq(spec.dkv_mode, "atomic");
    bool split = eq(spec.dq_mode, "split");
    /* The Q / dO prefetch ring, by the general shell's rule: every staging
     * but the synchronous (vgpr, 1) one, unless the merged-barrier 2-slot
     * ring stages the q steps instead. */
    bool staged = !(eq(s.global_path, "vgpr") && s.ring_depth == 1);
    bool ringed = staged && !eq(s.barrier_merge, "on");

    /* _declare(b, io, _noalias(body.ptr_noalias)) */
    static const char* const readonly[] = {"Q", "K", "V", "O", "dO", "LSE", "WS_LSE2", "WS_DSUM"};
    static const char* const io_ptrs[] = {"Q", "K", "V", "O", "dO", "dK", "dV"};
    const char* names[ROCKE_ATTN_BWD_DENSE_PARAMS];
    const char* formats[ROCKE_ATTN_BWD_DENSE_PARAMS];
    int n_params = rocke_attn_bwd_dense_params(names, formats);
    Kargs a;
    for(int i = 0; i < n_params; ++i)
    {
        const char* name = names[i];
        V v;
        if(eq(formats[i], "Q"))
        {
            bool is_io = false;
            for(const char* x : io_ptrs)
                is_io = is_io || eq(name, x);
            const rocke_type_t* pointee = is_io ? io : rocke_f32();
            bool ro = false;
            for(const char* x : readonly)
                ro = ro || eq(name, x);
            bool na = false;
            if(s.ptr_noalias != NULL)
                na = eq(s.ptr_noalias, "workspace") ? strncmp(name, "WS_", 3) == 0 : true;
            rocke_param_opts_t opts;
            memset(&opts, 0, sizeof(opts));
            if(ro)
            {
                opts.readonly = true;
                opts.readonly_set = true;
            }
            if(na)
            {
                opts.noalias = true;
                opts.noalias_set = true;
            }
            v = rocke_b_param(
                b, name, rocke_ptr_type(b, pointee, "global"), (ro || na) ? &opts : NULL);
        }
        else if(eq(formats[i], "f"))
            v = rocke_b_param(b, name, rocke_f32(), NULL);
        else
            v = rocke_b_param(b, name, rocke_i32(), NULL);
        needv(b, v);
        a.order.push_back(name);
        a.m[name] = v;
    }
    /* Multipliers that are 1.0 under the compiled scale placement are not
     * applied (the kernargs stay in the ABI). */
    unsigned unit = 0;
    rocke_attn_bwd_err_t e;
    if(rocke_attn_bwd_static_unit_mults(s.static_scale, s.dkv_mode, &unit, &e) != ROCKE_OK)
        ckc::raise_status(e.code, e.msg);
    if(unit & ROCKE_ATTN_BWD_UNIT_DS_MULT)
        a.m["ds_mult"] = NULL;
    if(unit & ROCKE_ATTN_BWD_UNIT_DK_MULT)
        a.m["dk_mult"] = NULL;
    if(unit & ROCKE_ATTN_BWD_UNIT_DQ_MULT)
        a.m["dq_mult"] = NULL;
    int threads = rocke_attn_bwd_tile_geometry_threads(&m.geom);
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
    V batch = spec.batch != ROCKE_ATTN_BWD_NONE ? ci(b, spec.batch) : a["batch"];
    V g_split = atomic ? a["g_split"] : NULL;
    V gx = ci(b, n_kv_tiles);
    V gy;
    if(atomic)
    {
        V chk = ci(b, h_k);
        gy = rocke_b_mul(b, chk, g_split);
    }
    else
        gy = ci(b, h_k);
    need(b, true);
    int left, right;
    bool br;
    rocke_attn_bwd_dense_spec_band(&spec, &left, &right, &br);
    /* The CTA objects; the mask policy is built once, here (one tile() per
     * kernel in every grid mode). */
    std::unique_ptr<Cta> cta(new Cta());
    Cta& c = *cta;
    c.b = b;
    c.has_static = true;
    if(rocke_attn_bwd_dense_spec_masked(&spec))
        need(b,
             rocke_attn_bwd_diagonal_band_init(
                 &c.mask, &c.any_edge, &c.any_interior, b, s_q, s_kv, left, right, br, km, kn));
    else
        need(b,
             rocke_attn_bwd_dense_no_mask_init(
                 &c.mask, &c.any_edge, &c.any_interior, b, s_q, s_kv, km, kn));
    rocke_attn_bwd_bwd_lds_t lds;
    need(b, rocke_attn_bwd_bwd_lds_init(&lds, b, &m.plan, s.dtype));
    c.lds = &lds;
    V csq = ci(b, s_q);
    V rows_q = rocke_b_mul(b, batch, csq);
    V cskv = ci(b, s_kv);
    V rows_kv = needv(b, rocke_b_mul(b, batch, cskv));
    /* Arguments of the shared per-kv-tile driver and epilogues (compile-time
     * strides and lengths, runtime pointers and multipliers). */
    Kargs shared;
    const char* dkv[2] = {"dk", "dv"};
    for(const char* t : dkv)
    {
        long long st[3];
        rocke_attn_bwd_err_t se;
        if(rocke_attn_bwd_dense_spec_strides(&spec, t, st, &se) != ROCKE_OK)
            ckc::raise_status(se.code, se.msg);
        shared.ints[std::string(t) + "_b"] = st[0];
        shared.ints[std::string(t) + "_h"] = st[1];
        shared.ints[std::string(t) + "_t"] = st[2];
    }
    shared.m["dK"] = a["dK"];
    shared.m["dV"] = a["dV"];
    shared.m["dk_mult"] = a["dk_mult"];
    shared.m["S_kv_max"] = ci(b, s_kv);
    shared.m["WS_DK"] = a["WS_DK"];
    shared.m["WS_DV"] = a["WS_DV"];
    shared.m["ws_rows_kv"] = rows_kv;
    shared.m["h_k"] = ci(b, h_k);
    shared.m["h_v"] = ci(b, h_k);
    need(b, true);

    rocke_attn_bwd_dense_fixed_addr_t addrs[4];
    rocke_attn_bwd_workspace_addr_t ws_lse2, ws_dsum, ws_dq;

    auto tile = [&](V x, V y, V z) {
        need(b,
             rocke_attn_bwd_decode_cta(b,
                                       x,
                                       y,
                                       z,
                                       gx,
                                       a["lb_order"],
                                       NULL,
                                       g_split,
                                       s.lb_interleave,
                                       NULL,
                                       NULL,
                                       &c.work));
        V bz = c.work.batch;
        rocke_attn_bwd_decode_args_t dq_args = rocke_attn_bwd_decode_args_default();
        dq_args.s_max = rocke_attn_bwd_iv_int(s_q);
        c.seq_q = rocke_attn_bwd_decode_batched(b, &m.seq_guards, bz, &dq_args);
        need(b, true);
        rocke_attn_bwd_decode_args_t dkv_args = rocke_attn_bwd_decode_args_default();
        dkv_args.s_max = rocke_attn_bwd_iv_int(s_kv);
        c.seq_kv = rocke_attn_bwd_decode_batched(b, &m.seq_guards, bz, &dkv_args);
        need(b, true);
        V g0 = ci(b, group);
        V g1 = ci(b, group);
        V g2 = ci(b, group);
        rocke_attn_bwd_head_range_t heads
            = rocke_attn_bwd_head_range(b, c.work.unit, c.work.split, g0, g1, g2, g_split);
        need(b, true);
        c.hk = heads.hk;
        c.hv = heads.hv;
        c.h0 = heads.h0;
        c.n_h = heads.n_heads;

        auto tensor = [&](rocke_attn_bwd_dense_fixed_addr_t* out,
                          const char* name,
                          const char* role,
                          const rocke_attn_bwd_seq_info_t* seq,
                          long long seqlen,
                          bool global_view) {
            long long st[3];
            rocke_attn_bwd_err_t se;
            if(rocke_attn_bwd_dense_spec_strides(&spec, role, st, &se) != ROCKE_OK)
                ckc::raise_status(se.code, se.msg);
            need(b,
                 rocke_attn_bwd_dense_fixed_addr_init(out,
                                                      b,
                                                      a[name],
                                                      io,
                                                      st,
                                                      seq,
                                                      seqlen,
                                                      spec.head_size,
                                                      spec.oob,
                                                      global_view,
                                                      spec.ragged));
        };
        /* the prefetch ring reads Q / dO through global pointers */
        tensor(&addrs[SLAB_Q], "Q", "q", &c.seq_q, s_q, ringed);
        tensor(&addrs[SLAB_DO], "dO", "do", &c.seq_q, s_q, ringed);
        tensor(&addrs[SLAB_K], "K", "k", &c.seq_kv, s_kv, false);
        tensor(&addrs[SLAB_V], "V", "v", &c.seq_kv, s_kv, false);
        c.slab_fn = [&addrs](int which, IV h, IV row0, bool clamp) {
            return rocke_attn_bwd_dense_fixed_addr_slab(&addrs[which], h, row0, clamp);
        };
        bool token_major = eq(spec.ws_layout, "token_major");
        IV nh;
        if(token_major)
            nh = ivv(ci(b, h_q));
        rocke_attn_bwd_ws_layout_t layout
            = token_major ? ROCKE_ATTN_BWD_WS_TOKEN_MAJOR : ROCKE_ATTN_BWD_WS_HEAD_MAJOR;
        auto ws = [&](const char* name, int width, rocke_attn_bwd_workspace_addr_t* out) {
            need(b,
                 rocke_attn_bwd_workspace_addr_init(out,
                                                    b,
                                                    a[name],
                                                    ivv(rows_q),
                                                    width,
                                                    c.seq_q.ws_row0,
                                                    layout,
                                                    token_major ? &nh : NULL));
        };
        /* stats_vec > 1 reads <n x f32> windows kept inside the head-major
         * arrays (StatsView needs their query-head count) */
        ws("WS_LSE2", 1, &ws_lse2);
        ws("WS_DSUM", 1, &ws_dsum);
        V stats_nh = s.stats_vec != 1 ? ci(b, h_q) : NULL;
        need(b,
             rocke_attn_bwd_stats_view_init(&c.stats, b, &ws_lse2, &ws_dsum, stats_nh, "clamp", 0));
        if(split)
            rocke_attn_bwd_dq_sink_init_null(&c.sink);
        else
        {
            ws("WS_DQ", spec.head_size, &ws_dq);
            need(b, rocke_attn_bwd_dq_sink_init_atomic(&c.sink, b, &ws_dq, &c.seq_q, a["dq_mult"]));
        }
        c.has_sink = true;
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
        c.tid = tid;
        if(ringed)
        {
            /* The general shell's ring on the same geometry, LDS plan and
             * stats view; the DMA wait counts the step's dQ atomics (none
             * under split dQ, whose main kernel emits no G4). */
            int n_atomics = km * spec.head_size
                            / (m.geom.waves * rocke_attn_bwd_tile_geometry_wave_size(&m.geom));
            need(b,
                 rocke_attn_bwd_q_ring_init(
                     &c.ring, b, &m.geom, c.lds, tid, lane, wave, &c.stats, split ? 0 : n_atomics));
            c.has_ring = true;
        }
        kv_tile(m, shared, c, c.work.first_kt, "", NULL);
    };

    /* (x, y, z) of a linear work id, kv tile fastest. */
    auto split_linear = [&](V lin, bool unit_major_last, V out[3]) {
        V t = rocke_b_mod(b, lin, gx);
        V r = rocke_b_div(b, lin, gx);
        out[0] = t;
        if(unit_major_last)
        {
            /* kv-head-major: (kv tile, batch, unit) */
            out[1] = rocke_b_div(b, r, batch);
            out[2] = rocke_b_mod(b, r, batch);
        }
        else
        {
            out[1] = rocke_b_mod(b, r, gy);
            out[2] = rocke_b_div(b, r, gy);
        }
        need(b, true);
    };

    if(spec.persistent)
    {
        V gxy = rocke_b_mul(b, gx, gy);
        V all = rocke_b_mul(b, gxy, batch);
        V total = rocke_b_readfirstlane(b, all);
        V stride = a["num_persistent"];
        V start = rocke_b_block_id_x(b);
        V one = ci(b, 1);
        V zero = ci(b, 0);
        V left_items = rocke_b_sub(b, total, start);
        V sm1 = rocke_b_sub(b, stride, one);
        V num = rocke_b_add(b, left_items, sm1);
        V trips = rocke_b_div(b, num, stride);
        V lt = rocke_b_cmp_lt(b, start, total);
        V sel = rocke_b_select(b, lt, trips, zero);
        trips = needv(b, rocke_b_readfirstlane(b, sel));
        rocke_iter_arg_t ia;
        ia.name = "dense_item";
        ia.init = start;
        rocke_for_t loop
            = rocke_b_scf_for_iter(b, zero, trips, one, &ia, 1, "dense_it", false, false);
        if(loop.op == NULL)
            raise_builder(b);
        rocke_b_region_enter(b, loop.body);
        {
            V item = loop.iter_vars[0];
            sync(b); /* the previous item's LDS reads finish before staging */
            bool hkv_major = eq(spec.persist_decode, "hkv_major");
            V ids[3];
            split_linear(item, hkv_major, ids);
            tile(ids[0], ids[1], ids[2]);
            V nxt = needv(b, rocke_b_add(b, item, stride));
            rocke_b_scf_yield(b, &nxt, 1);
            need(b, true);
        }
        rocke_b_region_leave(b);
        need(b, true);
    }
    else if(spec.xcd_remap)
    {
        V x = rocke_b_block_id_x(b);
        V y = rocke_b_block_id_y(b);
        V z = rocke_b_block_id_z(b);
        V gyz = rocke_b_mul(b, gy, z);
        V yz = rocke_b_add(b, y, gyz);
        V gxyz = rocke_b_mul(b, gx, yz);
        V lin = rocke_b_add(b, x, gxyz);
        /* whole-grid remap (the batch is a kernarg, so the grid size is
         * known); identity when xcd_n or xcd_chunk is 0 */
        V gxy = rocke_b_mul(b, gx, gy);
        V num = rocke_b_mul(b, gxy, batch);
        V remap = needv(b, rocke_attn_bwd_xcd_remap_id(b, lin, num, a["xcd_n"], a["xcd_chunk"]));
        lin = needv(b, rocke_b_readfirstlane(b, remap));
        V ids[3];
        split_linear(lin, false, ids);
        tile(ids[0], ids[1], ids[2]);
    }
    else
    {
        V x = rocke_b_block_id_x(b);
        V y = rocke_b_block_id_y(b);
        V z = rocke_b_block_id_z(b);
        tile(x, y, z);
    }
    rocke_b_ret(b);
    need(b, true);
    return rocke_ir_builder_kernel(b);
}

} // namespace rocke_attn_bwd

using namespace rocke_attn_bwd;

extern "C" {

rocke_attn_bwd_row_slab_t rocke_attn_bwd_buffer_row_slab(rocke_ir_builder_t* b,
                                                         rocke_value_t* rsrc,
                                                         const rocke_type_t* dtype,
                                                         long long t_stride)
{
    rocke_attn_bwd_row_slab_t s;
    memset(&s, 0, sizeof(s));
    s.b = b;
    s.rsrc = rsrc;
    s.dtype = dtype;
    s.t_stride = rocke_attn_bwd_iv_int(t_stride);
    s.stage_vec = 8;
    s.narrow = 1;
    return s;
}

rocke_value_t* rocke_attn_bwd_buffer_row_slab_load_vec(const rocke_attn_bwd_row_slab_t* s,
                                                       rocke_value_t* row,
                                                       rocke_attn_bwd_iv_t col0,
                                                       int n)
{
    B* b = s->b;
    return ckc::guard_builder(b, [&]() -> V {
        if(n % 8)
            throw ckc::ValueError("buffer row loads move multiples of 8 elements");
        V zero = ci(b, 0);
        std::vector<V> parts;
        for(int i = 0; i < n / 8; ++i)
        {
            IV col = col0;
            if(i != 0)
            {
                V c0 = rocke_attn_bwd_as_i32(b, col0);
                V c8 = ci(b, 8 * i);
                col = ivv(rocke_b_add(b, c0, c8));
            }
            V voff = buffer_voffset(s, row, col);
            parts.push_back(rocke_b_buffer_load_vN(b, s->rsrc, voff, zero, s->dtype, 8));
        }
        if(parts.empty())
            throw ckc::ValueError("list index out of range");
        V out = parts[0];
        for(size_t i = 1; i < parts.size(); ++i)
            out = rocke_b_vec_concat(b, out, parts[i]);
        return needv(b, out);
    });
}

rocke_value_t* rocke_attn_bwd_buffer_row_slab_load(const rocke_attn_bwd_row_slab_t* s,
                                                   rocke_value_t* row,
                                                   rocke_attn_bwd_iv_t col)
{
    B* b = s->b;
    return ckc::guard_builder(b, [&]() -> V {
        V voff = buffer_voffset(s, row, col);
        V zero = ci(b, 0);
        return needv(b, rocke_b_buffer_load(b, s->rsrc, voff, zero, s->dtype));
    });
}

bool rocke_attn_bwd_dense_fixed_addr_init(rocke_attn_bwd_dense_fixed_addr_t* t,
                                          rocke_ir_builder_t* b,
                                          rocke_value_t* ptr,
                                          const rocke_type_t* dtype,
                                          const long long strides[3],
                                          const rocke_attn_bwd_seq_info_t* seq,
                                          long long seqlen,
                                          int head_size,
                                          const char* oob,
                                          bool global_view,
                                          bool ragged)
{
    return ckc::guard_builder(b, [&]() -> bool {
        memset(t, 0, sizeof(*t));
        t->b = b;
        t->dtype = dtype;
        for(int i = 0; i < 3; ++i)
            t->strides[i] = strides[i];
        if(!eq(oob, "buffer_oob") && !eq(oob, "clamp"))
            throw ckc::ValueError("oob=" + (oob ? py_repr(oob) : std::string("None"))
                                  + " not in ('buffer_oob', 'clamp')");
        /* The global-pointer view of a buffer_oob tensor clamps only where a
         * row can pass the sequence end (a ragged kernel). */
        t->clamp = eq(oob, "clamp") || ragged;
        if(eq(oob, "clamp") || global_view)
        {
            IV st[3] = {rocke_attn_bwd_iv_int(strides[0]),
                        rocke_attn_bwd_iv_int(strides[1]),
                        rocke_attn_bwd_iv_int(strides[2])};
            need(
                b,
                rocke_attn_bwd_strided_tensor_addr_init(&t->strided, b, ptr, dtype, st, seq, 8, 1));
            /* its slab clamps only under _clamp (the packed loaders of the
             * dQ body take this view itself) */
            t->strided.never_clamp = !t->clamp;
            t->has_strided = true;
            return true;
        }
        V bz = rocke_attn_bwd_as_i64(b, ivv(seq->batch), false);
        V sb = rocke_b_const_i64(b, strides[0] * ELEM_BYTES);
        V off = rocke_b_mul(b, bz, sb);
        t->batch_ptr = needv(b, rocke_b_global_ptr_add(b, ptr, off));
        /* The slab's bytes under the compile-time shape: the hardware range
         * check reads zeros past them. */
        t->num_records = ((seqlen - 1) * strides[2] + head_size) * ELEM_BYTES;
        if(t->num_records >= (1LL << 31))
            throw ckc::ValueError("num_records " + std::to_string(t->num_records)
                                  + " does not fit i32");
        return true;
    });
}

rocke_attn_bwd_row_slab_t
    rocke_attn_bwd_dense_fixed_addr_slab(const rocke_attn_bwd_dense_fixed_addr_t* t,
                                         rocke_attn_bwd_iv_t h,
                                         rocke_attn_bwd_iv_t row0,
                                         bool clamp)
{
    if(t->has_strided)
        return rocke_attn_bwd_strided_tensor_addr_slab(&t->strided, h, row0, clamp);
    B* b = t->b;
    V ptr = t->batch_ptr;
    if(!rocke_attn_bwd_iv_is_int(h, 0))
    {
        V h64 = rocke_attn_bwd_as_i64(b, h, false);
        V sh = rocke_b_const_i64(b, t->strides[1] * ELEM_BYTES);
        V off = rocke_b_mul(b, h64, sh);
        ptr = rocke_b_global_ptr_add(b, ptr, off);
    }
    V nrec = ci(b, t->num_records);
    V rsrc = rocke_b_buffer_rsrc(b, ptr, nrec);
    return rocke_attn_bwd_buffer_row_slab(b, rsrc, t->dtype, t->strides[2]);
}

rocke_status_t rocke_attn_bwd_dense_static_classes(long long s_q,
                                                   long long s_kv,
                                                   int k_m0,
                                                   int k_n0,
                                                   int left,
                                                   int right,
                                                   bool bottom_right,
                                                   bool masked,
                                                   bool* any_edge,
                                                   bool* any_interior,
                                                   char* err,
                                                   size_t err_cap)
{
    bool edge = false, interior = false;
    long long n_kt = k_n0 > 0 ? (s_kv + k_n0 - 1) / k_n0 : 0;
    for(long long kt = 0; kt < n_kt; ++kt)
    {
        rocke_attn_bwd_static_kv_tile_plan_t plan;
        rocke_status_t st = rocke_attn_bwd_static_kv_tile_plan(s_q,
                                                               s_kv,
                                                               kt * k_n0,
                                                               k_m0,
                                                               k_n0,
                                                               left,
                                                               right,
                                                               bottom_right,
                                                               masked,
                                                               &plan,
                                                               err,
                                                               err_cap);
        if(st != ROCKE_OK)
            return st;
        edge = edge || plan.n_edge > 0;
        interior = interior || rocke_attn_bwd_static_kv_tile_plan_n_interior(&plan) > 0;
        if(edge && interior)
            break;
    }
    *any_edge = edge;
    *any_interior = interior;
    return ROCKE_OK;
}

bool rocke_attn_bwd_diagonal_band_init(rocke_attn_bwd_mask_policy_t* self,
                                       bool* any_edge,
                                       bool* any_interior,
                                       rocke_ir_builder_t* b,
                                       long long s_q,
                                       long long s_kv,
                                       int left,
                                       int right,
                                       bool bottom_right,
                                       int k_m0,
                                       int k_n0)
{
    return ckc::guard_builder(b, [&]() -> bool {
        long long off = bottom_right ? s_kv - s_q : 0;
        rocke_attn_runtime_bounds_t bounds
            = rocke_attn_runtime_bounds_make(rocke_iov_int(s_q), rocke_iov_int(s_kv));
        bounds.diag_offset = rocke_iov_int(off);
        bounds.left_bound = rocke_iov_int(left);
        bounds.right_bound = rocke_iov_int(right);
        need_st(b, rocke_attn_bwd_runtime_band_init(self, b, &bounds, k_m0, k_n0));
        char msg[256] = "";
        rocke_status_t st = rocke_attn_bwd_dense_static_classes(s_q,
                                                                s_kv,
                                                                k_m0,
                                                                k_n0,
                                                                left,
                                                                right,
                                                                bottom_right,
                                                                true,
                                                                any_edge,
                                                                any_interior,
                                                                msg,
                                                                sizeof(msg));
        if(st != ROCKE_OK)
            ckc::raise_status(st, msg);
        return true;
    });
}

bool rocke_attn_bwd_dense_no_mask_init(rocke_attn_bwd_mask_policy_t* self,
                                       bool* any_edge,
                                       bool* any_interior,
                                       rocke_ir_builder_t* b,
                                       long long s_q,
                                       long long s_kv,
                                       int k_m0,
                                       int k_n0)
{
    return ckc::guard_builder(b, [&]() -> bool {
        need_st(b,
                rocke_attn_bwd_no_mask_ktail_init(
                    self, b, rocke_iov_int(s_q), rocke_iov_int(s_kv), k_m0, k_n0));
        char msg[256] = "";
        rocke_status_t st = rocke_attn_bwd_dense_static_classes(
            s_q, s_kv, k_m0, k_n0, -1, -1, false, false, any_edge, any_interior, msg, sizeof(msg));
        if(st != ROCKE_OK)
            ckc::raise_status(st, msg);
        return true;
    });
}

} /* extern "C" */
