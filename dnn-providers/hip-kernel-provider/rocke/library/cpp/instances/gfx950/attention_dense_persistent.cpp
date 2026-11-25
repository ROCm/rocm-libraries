// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "attention_dense_internal.hpp"
#include "rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h"
#include "rocke/helper_rocke.helpers.attention.h"
#include "rocke/helper_rocke.helpers.mfma_attention.h"
#include "rocke/helper_rocke.helpers.schedule.h"
#include "rocke/instance_lse_store_internal.h"

#include <algorithm>
#include <cstdio>
#include <string_view>

namespace rocke_dense_gfx950
{
namespace
{
// The largest admitted geometry has D=128, BN=256 (not necessarily together).
constexpr int max_d_tiles = 128 / 32, max_kk_steps = 256 / 16;
using Scores = std::array<V, 256 / 32 * 16>; // [nsub][MFMA accumulator element]
using Carry = std::array<V, 2 + max_d_tiles + max_kk_steps>;

// Braced tuple operands preserve the original, observable left-to-right node order.
#define OP DENSE_OP

struct Persistent
{
    Builder* b;
    const Spec& s;
    const rocke_attention_dense_spec_t& p = s.base;
    const Type dtype = rocke_dtype_to_ir_type(p.dtype);
    const int d = p.head_size, bm = p.block_m, bn = p.block_n, nbuf = s.lds_num_buffers;
    const int waves = rocke_attention_dense_spec_num_waves(&p), n_per = bm / bn;
    const int k_steps = d / 16, d_tiles = d / 32, kk_steps = bn / 16, n_sub = bn / 32;
    const int n_scores = n_sub * 16, n_carry = 2 + d_tiles + kk_steps;
    const int rows_per_instr = 128 / d, rows_per_wave = bn / waves;
    const int64_t n_ktiles = (int64_t(p.seqlen_kv) + (p.ragged ? bn - 1 : 0)) / bn;
    const int64_t nqb = (int64_t(p.seqlen_q) + (p.ragged ? bm - 1 : 0)) / bm;
    const int sw = p.sliding_window, right = s.right_bound;
    const bool window_empty = sw > 0 && !p.varlen && !s.emit_lse;
    const bool empty_rows = window_empty && p.seqlen_q > int64_t(p.seqlen_kv) + sw - 1;
    const bool kbound
        = p.ragged && p.seqlen_kv % bn && (!p.causal || int64_t(p.seqlen_q) + right > p.seqlen_kv);
    const bool runtime = rocke_gfx950_attention_dense_runtime_shape(&s);
    const bool wide = s.wide_lds_dma, fold_scale = sw == 0, early_k = fold_scale && d == 128;
    const int loop_vmcnt = s.partial_vmcnt_prefetch ? bn * d / (waves * 64 * 8) : 0;
    const int exp_per = rocke_gfx950_attention_dense_resolved_exp_per_pv_step(&s);
    const bool sched_group = rocke_gfx950_attention_dense_resolved_pv_sched_group_template(&s);
    const bool pv_fence = rocke_gfx950_attention_dense_resolved_pv_sched_fence(&s);
    const bool k_major
        = std::string_view(rocke_gfx950_attention_dense_resolved_pv_loop_order(&s)) == "k_major";
    const int iglp = rocke_gfx950_attention_dense_resolved_iglp_mode(&s);

    DecodeRuntime rt{};
    V stride_q_p{}, stride_k_p{}, n_ktiles_p{};
    V q{}, k{}, v{}, o{}, sinks{}, lse{}, qk_scale{}, neg_inf{}, rcp_ln2{}, one_f{};
    V wave{}, lane{}, lane_m{}, lane_h{}, d_base{}, wave_s{};
    struct DMATile
    {
        V lds{}, addr{}, rsrc{};
        int bytes_per_buf{}, group_bytes{};
    } kt, vt;
    V zero_soff{}, q_rsrc{}, k_lane_grp{}, k_sub_col{};

    // Re-established per work item, inside the grid-stride loop.
    V q_tok0{}, k_base{}, vline{}, velem_lane{};
    std::array<V, 128 / 16> q_packs{};
    int rescale_id = 0;

    Persistent(Builder* builder, const Spec& spec)
        : b(builder)
        , s(spec)
    {
    }

    V c(int64_t x)
    {
        return rocke_b_const_i32(b, x);
    }
    V c64(int64_t x)
    {
        return rocke_b_const_i64(b, x);
    }
    V f(double x)
    {
        return rocke_b_const_f32(b, x);
    }
    V sq()
    {
        return runtime ? rt.seqlen_q : c(p.seqlen_q);
    }
    V skv()
    {
        return runtime ? rt.seqlen_kv : c(p.seqlen_kv);
    }
    V hq_count()
    {
        return runtime ? rt.num_query_heads : c(p.num_query_heads);
    }
    V stride_q()
    {
        return runtime ? stride_q_p : c(int64_t(p.num_query_heads) * d);
    }
    V stride_k()
    {
        return runtime ? stride_k_p : c(int64_t(p.num_kv_heads) * d);
    }
    V tile_count()
    {
        return runtime ? n_ktiles_p : c(n_ktiles);
    }
    V exp2(V x)
    {
        return s.use_exp2_fast ? OP(exp2_fast, x) : OP(exp2, x);
    }
    V mfma(V a, V rhs, V acc)
    {
        return rocke_mfma_attn_mfma_32x32x16_for_dtype(b, dtype, a, rhs, acc);
    }

    void parameters()
    {
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", waves * 64);
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", p.waves_per_eu);
        if(d == 128)
            rocke_attr_set_bool(b, &b->kernel->attrs, "lds_alias_scopes", true);
        q = pointer_param(b, "q_ptr", dtype, false, 16);
        k = pointer_param(b, "k_ptr", dtype, false, 16);
        v = pointer_param(b, "v_ptr", dtype, false, 16);
        o = pointer_param(b, "o_ptr", dtype, true, 16);
        V scale = OP(param, "scale", rocke_f32(), nullptr);
        if(runtime)
        {
            // Use the ABI registry's ordered field list (including just the pairs
            // used by this decode). Magic and shift are host-computed kernargs.
            rocke_gfx950_dense_fields_t fields{};
            rocke_gfx950_attention_dense_shape_kernarg_fields(&s, &fields);
            for(size_t i = 0; i < fields.count; ++i)
            {
                std::string_view name = fields.names[i];
                V arg = OP(param, fields.names[i], rocke_i32(), nullptr);
                if(name.ends_with("_magic") || name.ends_with("_shift"))
                {
                    bool magic = name.ends_with("_magic");
                    name.remove_suffix(6);
                    auto& pair = name == "batch"             ? rt.batch_div
                                 : name == "num_query_heads" ? rt.heads_div
                                 : name == "gqa"             ? rt.gqa_div
                                                             : rt.nqb_div;
                    (magic ? pair.magic : pair.shift) = arg;
                }
                else if(name == "batch")
                    rt.batch = arg;
                else if(name == "seqlen_q")
                    rt.seqlen_q = arg;
                else if(name == "seqlen_kv")
                    rt.seqlen_kv = arg;
                else if(name == "num_query_heads")
                    rt.num_query_heads = arg;
                else if(name == "num_kv_heads")
                    rt.num_kv_heads = arg;
            }
            rt.gqa = OP(div, rt.num_query_heads, rt.num_kv_heads);
            stride_q_p = OP(mul, rt.num_query_heads, c(d));
            stride_k_p = OP(mul, rt.num_kv_heads, c(d));
        }
        if(p.use_sinks)
            sinks = pointer_param(b, "sink_ptr", dtype, false, 16);
        if(s.emit_lse)
            lse = pointer_param(b, "lse_ptr", rocke_f32(), true, 4);
        qk_scale = OP(fmul, scale, f(log2e));
    }

    void setup()
    {
        parameters();
        V tid = OP(thread_id_x);
        wave = OP(div, tid, c(64));
        lane = OP(mod, tid, c(64));
        lane_m = OP(mod, lane, c(32));
        lane_h = OP(div, lane, c(32));
        d_base = OP(mul, lane_h, c(8));
        neg_inf = f(-1e30);
        if(p.use_sinks)
        {
            rcp_ln2 = f(log2e);
            one_f = f(1.0);
        }

        if(wide)
        {
            auto alloc = [&](DMATile& tile, int pad, const char* name) {
                int shape[] = {nbuf, d / 64, bn / 8, 512 + pad};
                tile.lds = OP(smem_alloc, dtype, shape, 4, name);
                tile.group_bytes = shape[3] * 2;
                tile.bytes_per_buf = shape[1] * shape[2] * tile.group_bytes;
            };
            alloc(kt, s.lds_k_row_pad, "Klds");
            alloc(vt, s.lds_v_row_pad, "Vlds");
        }
        else
        {
            int k_pitch
                = rows_per_instr * d + (rows_per_instr == 1 ? s.lds_k_row_pad : p.lds_k_group_pad);
            int k_shape[] = {nbuf, bn / rows_per_instr, k_pitch};
            kt.lds = OP(smem_alloc, dtype, k_shape, 3, "Klds");
            kt.group_bytes = 2 * k_pitch;
            kt.bytes_per_buf = k_shape[1] * kt.group_bytes;
            if(rows_per_instr > 1)
            {
                k_lane_grp = OP(div, lane_m, c(rows_per_instr));
                k_sub_col = OP(mul, OP(mod, lane_m, c(rows_per_instr)), c(d));
            }
            int v_pitch = rows_per_instr == 1 ? d + s.lds_v_row_pad : d;
            int v_shape[] = {nbuf, bn, v_pitch};
            vt.lds = OP(smem_alloc, dtype, v_shape, 3, "Vlds");
            vt.bytes_per_buf = bn * v_pitch * 2;
            vt.group_bytes = rows_per_instr * v_pitch * 2;
        }

        zero_soff = c(0);
        kt.addr = OP(smem_addr_of, kt.lds);
        vt.addr = OP(smem_addr_of, vt.lds);
        auto kv_extent = [&] {
            return runtime ? OP(mul, OP(mul, rt.batch, rt.seqlen_kv), OP(mul, stride_k_p, c(2)))
                           : c(int64_t(p.batch) * p.seqlen_kv * p.num_kv_heads * d * 2);
        };
        kt.rsrc = OP(buffer_rsrc, k, kv_extent());
        vt.rsrc = OP(buffer_rsrc, v, kv_extent());
        if(p.ragged)
            q_rsrc
                = OP(buffer_rsrc, q, c(int64_t(p.batch) * p.seqlen_q * p.num_query_heads * d * 2));
        // Deliberate dead node sequence retained by the Python lowering.
        OP(zext, OP(to_sgpr_u32, OP(mul, wave, c(64 * 16))), rocke_i64());
        if(wide)
            wave_s = OP(to_sgpr_u32, wave);
        if(runtime)
        {
            rt.nqb = OP(div, rt.seqlen_q, c(bm));
            rt.w = OP(mul, OP(mul, rt.nqb, rt.num_query_heads), rt.batch);
            n_ktiles_p = OP(div, rt.seqlen_kv, c(bn));
        }
    }

    void wait_barrier(int vmcnt)
    {
        OP(s_waitcnt, vmcnt, -1, -1);
        OP(s_barrier_bare);
    }

    Decoded begin_item(V wi)
    {
        wait_barrier(0); // drain the previous item's DMA before its LDS is reused
        Decoded work = decode_persistent(b, s, rt, wi);
        if(!work.hkv)
            work.hkv = decode_hkv(b, s, rt, work.hq);
        q_tok0 = OP(add, OP(mul, work.qb, c(bm)), OP(mul, wave, c(32)));
        V q_base = OP(add, OP(mul, OP(mul, work.bt, sq()), stride_q()), OP(mul, work.hq, c(d)));
        k_base = OP(add, OP(mul, OP(mul, work.bt, skv()), stride_k()), OP(mul, work.hkv, c(d)));
        V q_tok = OP(add, q_tok0, lane_m);
        for(int ks = 0; ks < k_steps; ++ks)
        {
            V col = OP(add, c(ks * 16), d_base);
            V addr = OP(add, OP(add, q_base, OP(mul, q_tok, stride_q())), col);
            q_packs[ks] = p.ragged ? OP(buffer_load_vN, q_rsrc, OP(mul, addr, c(2)), c(0), dtype, 8)
                                   : OP(global_load_vN, q, addr, dtype, 8, 16);
        }
        return work;
    }

    // Each wave issues one dword per lane into a contiguous LDS row group.
    // K carries inter-group padding at D64; V stays packed for the tr16 read.
    void async_load(const DMATile& tile, V buf, V key0)
    {
        V buf_off = OP(mul, OP(zext, buf, rocke_i64()), c64(tile.bytes_per_buf));
        auto issue = [&](V dst, V elem, int dwords) {
            OP(async_buffer_load_lds_addr,
               tile.rsrc,
               dst,
               OP(mul, elem, c(2)),
               zero_soff,
               dwords,
               0);
        };
        if(wide)
        {
            // Each slab line holds eight rows of 64 elements plus its pad.
            V n_in_wave = OP(div, lane, c(8));
            V d_bucket = OP(mod, lane, c(8));
            for(int pass = 0; pass < 8 / waves; ++pass)
            {
                V line_id = OP(add, wave_s, c(pass * waves));
                V row = OP(add, key0, OP(add, OP(mul, n_in_wave, c(8)), line_id));
                V src;
                if(runtime)
                {
                    V lane_part
                        = OP(add,
                             OP(mul, OP(add, OP(mul, n_in_wave, c(8)), line_id), stride_k()),
                             OP(mul, d_bucket, c(8)));
                    src = OP(add,
                             lane_part,
                             OP(to_sgpr_u32, OP(add, k_base, OP(mul, key0, stride_k()))));
                }
                else
                    src = OP(
                        add, OP(add, k_base, OP(mul, row, stride_k())), OP(mul, d_bucket, c(8)));
                for(int rpt = 0; rpt < d / 64; ++rpt)
                {
                    V line = OP(add,
                                OP(mul, line_id, c(tile.group_bytes)),
                                c(rpt * (bn / 8) * tile.group_bytes));
                    V dst = OP(
                        smem_ptr_add, tile.addr, OP(add, buf_off, OP(zext, line, rocke_i64())));
                    issue(dst, OP(add, src, c(rpt * 64)), 4);
                }
            }
        }
        else if(rows_per_instr == 1)
        {
            V row0 = OP(mul, wave, c(rows_per_wave));
            V off = OP(add, buf_off, OP(zext, OP(mul, row0, c(tile.group_bytes)), rocke_i64()));
            V gcol = OP(mul, lane, c(2));
            V lane_off{}, row_u{}, elem{};
            if(runtime)
            {
                // Keep just the lane part in a VGPR, not a hoisted VGPR per row.
                lane_off = OP(add, OP(mul, row0, stride_k()), gcol);
                row_u = OP(to_sgpr_u32, OP(add, k_base, OP(mul, key0, stride_k())));
            }
            else
                elem = OP(add, OP(add, k_base, OP(mul, OP(add, key0, row0), stride_k())), gcol);
            for(int r = 0; r < rows_per_wave; ++r)
            {
                V dst = OP(smem_ptr_add, tile.addr, off);
                issue(dst, runtime ? OP(add, lane_off, row_u) : elem, 1);
                if(r + 1 < rows_per_wave)
                {
                    off = OP(add, off, c64(tile.group_bytes));
                    if(runtime)
                        row_u = OP(to_sgpr_u32, OP(add, row_u, stride_k()));
                    else
                        elem = OP(add, elem, stride_k());
                }
            }
        }
        else
        {
            V sub_row = OP(div, lane, c(d / 2));
            V col = OP(mul, OP(mod, lane, c(d / 2)), c(2));
            int groups = rows_per_wave / rows_per_instr;
            for(int it = 0; it < groups; ++it)
            {
                V row0 = OP(add, OP(mul, wave, c(rows_per_wave)), c(it * rows_per_instr));
                V grp = OP(add, OP(mul, wave, c(groups)), c(it));
                V off = OP(add, buf_off, OP(zext, OP(mul, grp, c(tile.group_bytes)), rocke_i64()));
                V dst = OP(smem_ptr_add, tile.addr, off);
                V gkey = OP(add, OP(add, key0, row0), sub_row);
                issue(dst, OP(add, OP(add, k_base, OP(mul, gkey, stride_k())), col), 1);
            }
        }
    }

    void load_tile(V buf, V idx)
    {
        V key0 = OP(mul, idx, c(bn));
        async_load(kt, buf, key0);
        async_load(vt, buf, key0);
    }

    void do_qk(Scores& scores, V kbuf)
    {
        for(int ns = 0; ns < n_sub; ++ns)
        {
            V acc = OP(zero_vec_f32, 16);
            V kline{}, base{}, row{};
            if(wide)
            {
                kline = OP(mod, lane_m, c(8));
                base = OP(add,
                          OP(add, OP(mul, OP(div, lane_m, c(8)), c(64)), OP(mul, lane_h, c(8))),
                          c(ns * 256));
            }
            else if(rows_per_instr == 1)
                row = OP(add, c(ns * 32), lane_m);
            else
                row = OP(add, c(ns * (32 / rows_per_instr)), k_lane_grp);
            for(int ks = 0; ks < k_steps; ++ks)
            {
                V pack;
                if(wide)
                {
                    V elem = OP(add, base, c((ks % 4) * 16));
                    V idx[] = {kbuf, c(ks / 4), kline, elem};
                    pack = OP(smem_load_vN, kt.lds, idx, 4, dtype, 8);
                }
                else
                {
                    V col = OP(add, c(ks * 16), d_base);
                    if(rows_per_instr > 1)
                        col = OP(add, k_sub_col, col);
                    V idx[] = {kbuf, row, col};
                    pack = OP(smem_load_vN, kt.lds, idx, 3, dtype, 8);
                }
                acc = mfma(pack, q_packs[ks], acc);
            }
            for(int i = 0; i < 16; ++i)
            {
                V x = OP(vec_extract, acc, i);
                scores[16 * ns + i] = fold_scale ? x : OP(fmul, x, qk_scale);
            }
        }
    }

    void mask(Scores& scores, V tile_idx, bool lower, bool upper)
    {
        if(!p.causal && !lower)
            return;
        V key0 = OP(mul, tile_idx, c(bn));
        V query = OP(add, q_tok0, rocke__mfma_32x32_c_col(b, lane, 0));
        V win_lo = lower ? OP(sub, query, c(sw)) : nullptr;
        V up_tok = right ? OP(add, query, c(right)) : query;
        for(int ns = 0; ns < n_sub; ++ns)
        {
            V base = OP(add, key0, c(ns * 32));
            for(int i = 0; i < 16; ++i)
            {
                V ktok = OP(add, base, rocke__mfma_32x32_c_row(b, lane, i));
                V& x = scores[ns * 16 + i];
                if(upper && p.causal)
                    x = OP(select, OP(cmp_le, ktok, up_tok), x, neg_inf);
                if(lower)
                    x = OP(select, OP(cmp_gt, ktok, win_lo), x, neg_inf);
            }
        }
    }

    void mask_keys(Scores& scores, V tile_idx)
    {
        V key0 = OP(mul, tile_idx, c(bn));
        for(int ns = 0; ns < n_sub; ++ns)
        {
            V base = OP(add, key0, c(ns * 32));
            for(int i = 0; i < 16; ++i)
            {
                V ktok = OP(add, base, rocke__mfma_32x32_c_row(b, lane, i));
                V& x = scores[ns * 16 + i];
                x = OP(select, OP(cmp_lt, ktok, c(p.seqlen_kv)), x, neg_inf);
            }
        }
    }

    void init_v_layout()
    {
        if(wide)
        {
            vline = OP(add, OP(mul, lane_h, c(4)), OP(div, OP(mod, lane, c(16)), c(4)));
            velem_lane = OP(add,
                            OP(mul, OP(mod, OP(div, lane, c(16)), c(2)), c(16)),
                            OP(mul, OP(mod, lane, c(4)), c(4)));
        }
    }

    V read_v(int dt, int kk, V buf)
    {
        if(!wide)
            return rocke_pv32_v_load_paired(b, vt.lds, buf, dt, kk, lane_h, lane_m, dtype);
        V elem = OP(add, velem_lane, c((dt % 2) * 32 + kk * 128));
        V idx0[] = {buf, c(dt / 2), vline, elem};
        V a0 = OP(ds_read_tr16_b64, vt.lds, idx0, 4, dtype);
        V idx1[] = {buf, c(dt / 2), vline, OP(add, elem, c(64))};
        V a1 = OP(ds_read_tr16_b64, vt.lds, idx1, 4, dtype);
        return OP(vec_concat, a0, a1);
    }

    V half_reduce(V x, bool maximum)
    {
        auto combine = maximum ? rocke_b_fmax : rocke_b_fadd;
        if(!fold_scale)
            return combine(b, x, OP(warp_shuffle_xor, x, 32));
        V lo{}, hi{};
        OP(permlane32_swap, OP(bitcast, x, rocke_i32()), OP(bitcast, x, rocke_i32()), &lo, &hi);
        V a = OP(bitcast, lo, rocke_f32());
        V rhs = OP(bitcast, hi, rocke_f32());
        return combine(b, a, rhs);
    }

    struct MaxUpdate
    {
        V m, alpha, skip;
    };

    MaxUpdate softmax_max(const Scores& scores, V old_m)
    {
        V local = neg_inf;
        for(int i = 0; i < n_scores; ++i)
            local = OP(fmax, local, scores[i]);
        V tile_max = half_reduce(local, true);
        if(fold_scale)
            tile_max = OP(fmul, tile_max, qk_scale);
        V within{}, skip{}, m;
        if(p.lazy_rescale)
        {
            V diff = OP(fsub, tile_max, old_m);
            within = OP(fcmp, "ole", diff, f(s.lazy_rescale_threshold));
            V below = OP(select, within, c(1), c(0));
            skip = OP(cmp_ne, OP(wave_all, below), c(0));
            m = OP(select, within, old_m, OP(fmax, old_m, tile_max));
        }
        else
            m = OP(fmax, old_m, tile_max);
        V alpha = exp2(OP(fsub, old_m, m));
        if(within)
            alpha = OP(select, within, f(1.0), alpha);
        return {m, alpha, skip};
    }

    V probability(V x, V m, V neg_m)
    {
        return exp2(fold_scale ? OP(fma, x, qk_scale, neg_m) : OP(fsub, x, m));
    }

    V row_sum(const Scores& probs)
    {
        if(!fold_scale)
        {
            V sum = f(0.0);
            for(int i = 0; i < n_scores; ++i)
                sum = OP(fadd, sum, probs[i]);
            return sum;
        }
        // Pairwise tree of <2 x f32> adds. Finish both vector lanes in scalar IR.
        std::array<V, Scores{}.size() / 2> pairs;
        for(int i = 0; i < n_scores / 2; ++i)
            pairs[i] = OP(vec_pack, probs.data() + 2 * i, 2, rocke_f32());
        for(int n = n_scores / 2; n > 1; n /= 2)
            for(int i = 0; i < n / 2; ++i)
                pairs[i] = OP(fadd, pairs[2 * i], pairs[2 * i + 1]);
        return OP(fadd, OP(vec_extract, pairs[0], 0), OP(vec_extract, pairs[0], 1));
    }

    void relayout_p(V* packs, const Scores& probs)
    {
        for(int kk = 0; kk < kk_steps; ++kk)
        {
            V elems[8];
            for(int i = 0; i < 8; ++i)
            {
                int key = kk * 16 + (i / 4) * 8 + i % 4;
                int row = key % 32, preg = (row / 8) * 4 + row % 4;
                elems[i] = OP(cast_f32_to, probs[(key / 32) * 16 + preg], dtype);
            }
            packs[kk] = OP(vec_pack, elems, 8, dtype);
        }
    }

    // PV uses the previous tile's P. In the steady state, schedule this tile's
    // exponentials into the PV cluster; in the epilogue, only PV remains.
    V pv(V* out, V const* carry, V buf, Scores* scores = nullptr, V m = nullptr)
    {
        V neg_m = scores && fold_scale ? OP(fneg, m) : nullptr;
        int slot = 0;
        for(int step = 0; step < kk_steps * d_tiles; ++step)
        {
            int dt = k_major ? step % d_tiles : step / kk_steps;
            int kk = k_major ? step / d_tiles : step % kk_steps;
            V a = read_v(dt, kk, buf);
            out[dt] = mfma(a, carry[2 + d_tiles + kk], kk == 0 ? carry[2 + dt] : out[dt]);
            if(!scores)
                continue;
            int n_emit = std::min(exp_per, n_scores - slot);
            for(int i = 0; i < n_emit; ++i, ++slot)
                (*scores)[slot] = probability((*scores)[slot], m, neg_m);
            if(sched_group)
            {
                OP(sched_group_barrier, ROCKE_SCHED_DS_READ, s.pv_sched_group_ds_read, 0);
                OP(sched_group_barrier, ROCKE_SCHED_MFMA, 1, 0);
                OP(sched_group_barrier, ROCKE_SCHED_VALU, std::max(1, n_emit), 0);
                OP(sched_group_barrier, ROCKE_SCHED_TRANS, std::max(1, n_emit), 0);
            }
        }
        if(!scores)
            return nullptr;
        for(; slot < n_scores; ++slot)
            (*scores)[slot] = probability((*scores)[slot], m, neg_m);
        return half_reduce(row_sum(*scores), false);
    }

    void rescale_o(V* out, V const* in, V alpha)
    {
        for(int dt = 0; dt < d_tiles; ++dt)
        {
            V elems[16];
            for(int i = 0; i < 16; ++i)
                elems[i] = OP(fmul, OP(vec_extract, in[dt], i), alpha);
            out[dt] = OP(vec_pack, elems, 16, rocke_f32());
        }
    }

    V lazy_rescale(V* o_acc, V l_i, const MaxUpdate& update, V l_tile)
    {
        // 0 trips passes O/l through untouched; 1 trip rescales by alpha.
        // Each phase emits its own loop, so the IV/args need unique names.
        ++rescale_id;
        V trips = OP(select, update.skip, c(0), c(1));
        std::array<rocke_iter_arg_t, max_d_tiles + 1> args;
        char names[max_d_tiles + 1][32], iv[32];
        for(int dt = 0; dt < d_tiles; ++dt)
        {
            std::snprintf(names[dt], sizeof(names[dt]), "ro%d_rs%d", dt, rescale_id);
            args[dt] = {names[dt], o_acc[dt]};
        }
        std::snprintf(names[d_tiles], sizeof(names[d_tiles]), "rl_rs%d", rescale_id);
        args[d_tiles] = {names[d_tiles], l_i};
        std::snprintf(iv, sizeof(iv), "rs%d", rescale_id);
        auto rs = OP(scf_for_iter, c(0), trips, c(1), args.data(), d_tiles + 1, iv, false, true);
        if(!rocke_ir_builder_ok(b))
            return nullptr;
        OP(region_enter, rs.body);
        V vals[max_d_tiles + 1];
        rescale_o(vals, rs.iter_vars, update.alpha);
        vals[d_tiles] = OP(fmul, rs.iter_vars[d_tiles], update.alpha);
        OP(scf_yield, vals, d_tiles + 1);
        OP(region_leave);
        std::copy_n(rs.op->results, d_tiles, o_acc);
        return OP(fadd, rs.op->results[d_tiles], l_tile);
    }

    void loop_body(V j, V const* carry, bool lower, bool upper, bool key_bound)
    {
        if(iglp != -1)
            OP(iglp_opt, iglp);
        V pbuf = OP(mod, OP(add, j, c(1)), c(nbuf));
        V kbuf = OP(mod, j, c(nbuf));
        V vbuf_prev = OP(mod, OP(add, j, c(nbuf - 1)), c(nbuf));
        wait_barrier(loop_vmcnt);
        Scores scores;
        do_qk(scores, kbuf);
        V next_key0{};
        if(early_k)
        {
            // All reads of this K tile have issued: K(j+1) can overlap PV(j-1).
            next_key0 = OP(mul, OP(add, j, c(1)), c(bn));
            async_load(kt, pbuf, next_key0);
        }
        if(lower || upper)
            mask(scores, j, lower, upper);
        if(key_bound)
            mask_keys(scores, j);
        MaxUpdate update = softmax_max(scores, carry[0]);
        if(pv_fence)
            OP(sched_barrier, s.pv_sched_fence_mask);
        if(s.pv_priority)
            OP(s_setprio, s.pv_priority);
        Carry next;
        next[0] = update.m;
        V* o_acc = next.data() + 2;
        V l_tile = pv(o_acc, carry, vbuf_prev, &scores, update.m);
        if(s.pv_priority)
            OP(s_setprio, 0);
        if(p.lazy_rescale)
            next[1] = lazy_rescale(o_acc, carry[1], update, l_tile);
        else
        {
            next[1] = OP(fadd, OP(fmul, carry[1], update.alpha), l_tile);
            rescale_o(o_acc, o_acc, update.alpha);
        }
        relayout_p(o_acc + d_tiles, scores);
        OP(s_barrier_bare);
        if(early_k)
            async_load(vt, pbuf, next_key0);
        else
            load_tile(pbuf, OP(add, j, c(1)));
        OP(scf_yield, next.data(), n_carry);
    }

    V upper_bound(V qb)
    {
        if(!p.causal)
            return tile_count();
        V upper = OP(add, OP(mul, qb, c(n_per)), c((int64_t(bm) - 1 + right) / bn + 1));
        return OP(select, OP(cmp_lt, upper, tile_count()), upper, tile_count());
    }

    V start_tile(V qb)
    {
        if(sw == 0)
            return c(0);
        V diag = OP(mul, qb, c(n_per));
        V raw = OP(sub, diag, c(sw / bn));
        V tile = OP(select, OP(cmp_gt, raw, c(0)), raw, c(0));
        if(runtime)
        {
            V last = OP(sub, n_ktiles_p, c(1));
            tile = OP(select, OP(cmp_lt, tile, last), tile, last);
        }
        else if(empty_rows)
            tile = OP(select, OP(cmp_lt, tile, c(n_ktiles - 1)), tile, c(n_ktiles - 1));
        return tile;
    }

    void prime(Carry& init, V start, V hq)
    {
        V buf = OP(mod, start, c(nbuf));
        V buf1 = OP(mod, OP(add, start, c(1)), c(nbuf));
        load_tile(buf, start);
        load_tile(buf1, OP(add, start, c(1)));
        wait_barrier(0);
        Scores scores;
        do_qk(scores, buf);
        mask(scores, start, sw > 0, p.causal);
        if(kbound)
            mask_keys(scores, start);
        V m_init = neg_inf;
        if(p.use_sinks)
        {
            V sink_h = OP(global_load, sinks, hq, dtype, 2);
            m_init = OP(fmul, OP(cast_to_f32, sink_h), rcp_ln2);
        }
        MaxUpdate update = softmax_max(scores, m_init);
        V neg_m = fold_scale ? OP(fneg, update.m) : nullptr;
        for(int i = 0; i < n_scores; ++i)
            scores[i] = probability(scores[i], update.m, neg_m);
        V l_tile = half_reduce(row_sum(scores), false);
        init[0] = update.m;
        init[1] = p.use_sinks ? OP(fadd, l_tile, OP(fmul, one_f, update.alpha)) : l_tile;
        for(int dt = 0; dt < d_tiles; ++dt)
            init[2 + dt] = OP(zero_vec_f32, 16);
        relayout_p(init.data() + 2 + d_tiles, scores);
    }

    // Each phase forwards [m, l, o[dt], pk[kk]]. Phase names are always
    // derived from the initial names, never from a previous phase's suffix.
    V const* phase(V lo,
                   V hi,
                   V const* init,
                   const char* iv,
                   const char* suffix,
                   bool lower,
                   bool upper,
                   bool key_bound)
    {
        if(!rocke_ir_builder_ok(b))
            return nullptr;
        std::array<rocke_iter_arg_t, Carry{}.size()> args;
        char names[Carry{}.size()][16];
        for(int i = 0; i < n_carry; ++i)
        {
            if(i < 2)
                std::snprintf(names[i], sizeof(names[i]), "%s%s", i == 0 ? "m" : "l", suffix);
            else if(i < 2 + d_tiles)
                std::snprintf(names[i], sizeof(names[i]), "o%d%s", i - 2, suffix);
            else
                std::snprintf(names[i], sizeof(names[i]), "pk%d%s", i - 2 - d_tiles, suffix);
            args[i] = {names[i], init[i]};
        }
        auto loop = OP(scf_for_iter, lo, hi, c(1), args.data(), n_carry, iv, false, true);
        if(!rocke_ir_builder_ok(b))
            return nullptr;
        OP(region_enter, loop.body);
        loop_body(loop.iv, loop.iter_vars, lower, upper, key_bound);
        OP(region_leave);
        return loop.op->results;
    }

    V clamp(V x, V lo, V hi)
    {
        x = OP(select, OP(cmp_lt, x, lo), lo, x);
        return OP(select, OP(cmp_lt, x, hi), x, hi);
    }

    V const* kv_phases(V qb, V start, V upper, const Carry& init)
    {
        if(sw > 0)
        {
            V diag = OP(mul, qb, c(n_per));
            V a = OP(add, start, c(1));
            V left_end = OP(add, diag, c(n_per - sw / bn));
            if(p.causal && right / bn)
                diag = OP(add, diag, c(right / bn));
            V mid_lo = clamp(left_end, a, upper);
            V mid_hi = p.causal ? clamp(diag, mid_lo, upper) : upper;
            auto left = phase(a, mid_lo, init.data(), "swl", "", true, p.causal, false);
            auto mid = phase(mid_lo, mid_hi, left, "swm", "_m", false, false, false);
            return p.causal ? phase(mid_hi, upper, mid, "swr", "_r", true, true, false) : mid;
        }
        if(p.causal && s.causal_diag_split)
        {
            V diag = OP(mul, qb, c(n_per));
            if(right / bn)
                diag = OP(add, diag, c(right / bn));
            V body_hi = OP(select, OP(cmp_lt, diag, upper), diag, upper);
            auto body = phase(c(1), body_hi, init.data(), "nb", "", false, false, kbound);
            V tail_lo = OP(select, OP(cmp_lt, diag, c(1)), c(1), diag);
            return phase(tail_lo, upper, body, "nt", "_t", false, true, kbound);
        }
        return phase(c(1), upper, init.data(), "nkt", "", false, p.causal, kbound);
    }

    void epilogue(V wi, V upper, V const* res)
    {
        wait_barrier(0);
        V last_vbuf = OP(mod, OP(add, upper, c(nbuf - 1)), c(nbuf));
        std::array<V, max_d_tiles> o_acc;
        pv(o_acc.data(), res, last_vbuf);
        V m_i = res[0], l_i = res[1], rcp_l;
        if(s.emit_lse)
        {
            V positive = OP(fcmp, "ogt", l_i, f(0.0));
            rcp_l = OP(rcp, OP(select, positive, l_i, f(1.0)));
        }
        else
            rcp_l = OP(rcp, l_i);
        if(empty_rows && !runtime)
            rcp_l = OP(select, OP(fcmp, "ogt", m_i, f(-1e29)), rcp_l, f(0.0));

        // Decode again from the live outer IV: do not carry batch/head through
        // the KV loop, or elide the repeated/dead nodes of this second decode.
        auto e = decode_persistent(b, s, rt, wi);
        V base = OP(add, OP(mul, OP(mul, e.bt, sq()), stride_q()), OP(mul, e.hq, c(d)));
        V qtok = OP(add, q_tok0, rocke__mfma_32x32_c_col(b, lane, 0));
        V row = OP(add, base, OP(mul, qtok, stride_q()));
        V d_half = OP(mul, lane_h, c(4));
        V has_key{};
        if(s.emit_lse || (window_empty && runtime))
            has_key = rocke_dense_row_has_key(b, qtok, skv(), p.causal, 0, sw, right);
        if(s.emit_lse)
        {
            V index = OP(add, OP(mul, OP(add, OP(mul, e.bt, hq_count()), e.hq), sq()), qtok);
            V exists = OP(cmp_lt, qtok, sq());
            dense_lse_output(b, lse, index, m_i, l_i, exists, has_key, lane_h, sinks, e.hq, dtype);
        }
        if(p.ragged)
        {
            auto guard = OP(scf_if, OP(cmp_lt, qtok, c(p.seqlen_q)));
            OP(region_enter, guard.then_region);
        }
        emit_o_store(b,
                     o,
                     row,
                     d_half,
                     std::span(o_acc).first(d_tiles),
                     rcp_l,
                     dtype,
                     s.o_store_width,
                     has_key);
        if(p.ragged)
            OP(region_leave);
    }

    void build()
    {
        setup();
        V cta = OP(block_id_x);
        auto outer = OP(scf_for,
                        cta,
                        runtime ? rt.w : c(nqb * p.num_query_heads * p.batch),
                        c(p.num_persistent),
                        "wi");
        if(!rocke_ir_builder_ok(b))
            return;
        OP(region_enter, outer.body);
        Decoded work = begin_item(outer.iv);
        init_v_layout();
        V upper = upper_bound(work.qb);
        V start = start_tile(work.qb);
        Carry init;
        prime(init, start, work.hq);
        auto res = kv_phases(work.qb, start, upper, init);
        if(rocke_ir_builder_ok(b))
            epilogue(outer.iv, upper, res);
        OP(region_leave);
        OP(ret);
    }
};
} // namespace

#undef OP

void build_persistent(Builder* b, const Spec& s)
{
    Persistent(b, s).build();
}
} // namespace rocke_dense_gfx950
