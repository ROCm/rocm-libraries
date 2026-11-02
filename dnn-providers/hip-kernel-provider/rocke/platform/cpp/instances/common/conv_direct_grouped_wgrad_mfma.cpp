// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_wgrad_mfma.cpp -- C++ engine port of
 * library/kernels/common/conv_direct_grouped_wgrad_mfma.py: the grouped wgrad
 * for 4 and 8 channels per group on the batched 4x4x4 MFMA
 * (build_grouped_wgrad_mfma).
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the contract in
 * rocke/instance_conv_direct_wgrad_internal.hpp.
 */
#include "rocke/instance_conv_direct_grouped_wgrad_mfma.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */
#include "rocke/instance_conv_direct_wgrad_internal.hpp"

using namespace rocke_dwg;

namespace
{

constexpr int kBC = 64; /* channels per block */
constexpr int kCH = kBC / 8; /* 16-byte chunks per staged pixel */

const char* bf16_flag[1] = {"bf16"};

int total_c(const rocke_direct_conv_problem_t* p)
{
    return p->groups * p->cpg;
}

int total_k(const rocke_direct_conv_problem_t* p)
{
    return p->groups * p->kpg;
}

int block_q(const rocke_grouped_wgrad_mfma_spec_t* s)
{
    return 4 * s->wave_q4 * s->waves_q;
}

/* 4x4 blocks along k per group: 1 for cpg 4, 2 for cpg 8. */
int halves(const rocke_grouped_wgrad_mfma_spec_t* s)
{
    return s->problem.cpg / 4;
}

int x_cols(const rocke_grouped_wgrad_mfma_spec_t* s)
{
    return block_q(s) + s->problem.KW - 1;
}

int ix(const rocke_grouped_wgrad_mfma_spec_t* s)
{
    return cdiv(cdiv(x_cols(s) * kCH, kWave), s->waves_q);
}

int iy(const rocke_grouped_wgrad_mfma_spec_t* s)
{
    return cdiv(cdiv(block_q(s) * kCH, kWave), s->waves_q);
}

int n_channel_tiles(const rocke_grouped_wgrad_mfma_spec_t* s)
{
    return cdiv(total_c(&s->problem), kBC);
}

int n_q(const rocke_grouped_wgrad_mfma_spec_t* s)
{
    return cdiv(problem_wo(&s->problem), block_q(s));
}

int n_units(const rocke_grouped_wgrad_mfma_spec_t* s)
{
    return s->problem.N * cdiv(s->problem.H, s->rows_per_chunk);
}

int n_z(const rocke_grouped_wgrad_mfma_spec_t* s)
{
    int planes = n_channel_tiles(s) * n_q(s);
    int c = cdiv(s->target_blocks, planes);
    if(c < 1)
    {
        c = 1;
    }
    int u = n_units(s);
    return u < c ? u : c;
}

int n_acc(const rocke_grouped_wgrad_mfma_spec_t* s)
{
    return s->problem.KH * s->problem.KW * halves(s);
}

} // namespace

extern "C" rocke_grouped_wgrad_mfma_spec_t rocke_grouped_wgrad_mfma_spec_default(void)
{
    rocke_grouped_wgrad_mfma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "grouped_wgrad_mfma";
    s.waves_q = 4;
    s.wave_q4 = 1;
    s.rows_per_chunk = 56;
    s.prefetch = 2;
    s.target_blocks = 1024;
    return s;
}

extern "C" int rocke_grouped_wgrad_mfma_threads(const rocke_grouped_wgrad_mfma_spec_t* s)
{
    return s->waves_q * kWave;
}

extern "C" long rocke_grouped_wgrad_mfma_lds_bytes(const rocke_grouped_wgrad_mfma_spec_t* s)
{
    long stage = (long)s->prefetch * s->waves_q * (ix(s) + iy(s)) * kWave * 16;
    long red = (long)s->waves_q * kWave * 4 * 4;
    return stage + red;
}

extern "C" int rocke_grouped_wgrad_mfma_n_partitions(const rocke_grouped_wgrad_mfma_spec_t* s)
{
    return n_q(s) * n_z(s);
}

extern "C" int rocke_grouped_wgrad_mfma_fold_taps(const rocke_grouped_wgrad_mfma_spec_t* s)
{
    return s->problem.KH * s->problem.KW * s->problem.cpg;
}

extern "C" long rocke_grouped_wgrad_mfma_workspace_bytes(const rocke_grouped_wgrad_mfma_spec_t* s)
{
    return (long)rocke_grouped_wgrad_mfma_n_partitions(s) * rocke_grouped_wgrad_mfma_fold_taps(s)
           * total_k(&s->problem) * 4;
}

extern "C" void rocke_grouped_wgrad_mfma_launch_geometry(const rocke_grouped_wgrad_mfma_spec_t* s,
                                                         int grid[3],
                                                         int block[3])
{
    grid[0] = n_channel_tiles(s);
    grid[1] = n_q(s);
    grid[2] = n_z(s);
    block[0] = rocke_grouped_wgrad_mfma_threads(s);
    block[1] = 1;
    block[2] = 1;
}

extern "C" rocke_status_t rocke_grouped_wgrad_mfma_kernel_name(
    const rocke_grouped_wgrad_mfma_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    char f_buf[64], wq_buf[24], q4_buf[24], rpc_buf[24], pf_buf[24], tb_buf[24];
    snprintf(f_buf, sizeof f_buf, "f%dx%dp%ds%dc%d", p->KH, p->KW, p->PAD, p->stride, p->cpg);
    snprintf(wq_buf, sizeof wq_buf, "wq%d", s->waves_q);
    snprintf(q4_buf, sizeof q4_buf, "q4%d", s->wave_q4);
    snprintf(rpc_buf, sizeof rpc_buf, "rpc%d", s->rows_per_chunk);
    snprintf(pf_buf, sizeof pf_buf, "pf%d", s->prefetch);
    snprintf(tb_buf, sizeof tb_buf, "tb%d", s->target_blocks);
    const char* parts[6] = {f_buf, wq_buf, q4_buf, rpc_buf, pf_buf, tb_buf};
    int on[1] = {problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(s->name, parts, 6, bf16_flag, on, 1, out, out_cap, NULL);
}

extern "C" bool rocke_grouped_wgrad_mfma_is_valid_spec(const rocke_grouped_wgrad_mfma_spec_t* s,
                                                       const char* arch,
                                                       char* reason,
                                                       size_t reason_cap)
{
    char msg[ROCKE_ERR_MSG_CAP];
    if(s == NULL)
    {
        set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(dilation_reason(p, msg, sizeof msg))
    {
        set_reason(reason, reason_cap, msg);
        return false;
    }
    const char* why = NULL;
    if(p->cpg != p->kpg || (p->cpg != 4 && p->cpg != 8))
    {
        why = "cpg = kpg in (4, 8)";
    }
    else if(!problem_dtype_ok(p))
    {
        why = "fp16/bf16 only";
    }
    else if(p->stride != 1)
    {
        why = "stride 1 only";
    }
    else if(p->PAD > p->KH - 1 || p->PAD > p->KW - 1)
    {
        why = "PAD <= K - 1";
    }
    else if(total_c(p) % 8)
    {
        why = "C must be a multiple of 8";
    }
    else if(s->waves_q != 1 && s->waves_q != 2 && s->waves_q != 4 && s->waves_q != 8)
    {
        why = "waves_q in (1, 2, 4, 8)";
    }
    else if(s->wave_q4 != 1 && s->wave_q4 != 2 && s->wave_q4 != 4)
    {
        why = "wave_q4 in (1, 2, 4)";
    }
    else if(s->prefetch < 2 || s->prefetch > 4)
    {
        why = "prefetch in 2..4";
    }
    else if(s->rows_per_chunk < 1)
    {
        why = "rows_per_chunk >= 1";
    }
    else if(s->target_blocks < 1)
    {
        why = "target_blocks >= 1";
    }
    else if(n_acc(s) * 4 > 128)
    {
        why = "too many accumulators";
    }
    else if(rocke_grouped_wgrad_mfma_lds_bytes(s) > 160 * 1024)
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "LDS %ld B > 160 KiB", rocke_grouped_wgrad_mfma_lds_bytes(s));
        why = msg;
    }
    else if(strcmp(arch, "gfx950") != 0)
    {
        why = "ds_read_tr16_b64 needs gfx950";
    }
    if(why != NULL)
    {
        set_reason(reason, reason_cap, why);
        return false;
    }
    set_reason(reason, reason_cap, "ok");
    return true;
}

namespace
{

struct Lane
{
    V col, ok, cc;
};

struct Off
{
    V off, ok;
};

/* ring[row][q4][half]: dY fragments, newest row first. */
using Ring = std::vector<std::vector<std::vector<V>>>;

rocke_kernel_def_t* build_grouped(rocke_ir_builder_t* b,
                                  const rocke_grouped_wgrad_mfma_spec_t* spec,
                                  const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_grouped_wgrad_mfma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        fail_invalid(b, "invalid GroupedWgradMfmaSpec", reason);
        return NULL;
    }
    Em e{b};
    const rocke_direct_conv_problem_t& p = spec->problem;
    const int KH = p.KH, KW = p.KW, PAD = p.PAD, CPG = p.cpg;
    const int HV = halves(spec);
    const int TAPS = rocke_grouped_wgrad_mfma_fold_taps(spec);
    const int WQ = spec->waves_q, Q4 = spec->wave_q4;
    const int BQ = block_q(spec);
    const int XC = x_cols(spec);
    const int D = spec->prefetch;
    const int IX = ix(spec), IY = iy(spec);
    const int T = rocke_grouped_wgrad_mfma_threads(spec);
    const int PRE = KH - 1; /* dY rows a unit needs before its first input row */
    const bool bf16 = problem_is_bf16(&p);
    const rocke_type_t* io_type = bf16 ? rocke_bf16() : rocke_f16();

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", T);
    rocke_attr_set_bool(b, &b->kernel->attrs, "lds_alias_scopes", true);
    rocke_dconv_params_t P;
    rocke_dconv_emit_params(b, &P, "wgrad", io_type);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    V c0 = e.ci(0);
    V c1 = e.ci(1);
    V c2 = e.ci(2);
    V oob = e.ci(((int64_t)1 << 31) - 1);
    V p_N = P.p_N;
    V p_C = P.p_total_c;
    V p_K = P.p_total_k;
    V p_H = P.p_Hi, p_W = P.p_Wi;
    V p_Ho = P.p_Ho, p_Wo = P.p_Wo;
    V sdy_n = P.p_A_stride_n, sdy_h = P.p_A_stride_hi, sdy_w = P.p_A_stride_wi;
    V sx_n = P.p_B_stride_n, sx_h = P.p_B_stride_hi, sx_w = P.p_B_stride_wi;

    V tid = rocke_b_thread_id_x(b);
    V lane = e.modi(tid, kWave);
    V wave = rocke_b_readfirstlane(b, e.divi(tid, kWave));
    V ctile = rocke_b_block_id_x(b);
    V qtile = rocke_b_block_id_y(b);
    V zslot = rocke_b_block_id_z(b);
    V c_blk0 = e.muli(ctile, kBC);
    V q0 = e.muli(qtile, BQ);
    V n_q_v = e.cdivi(p_Wo, BQ);
    V n_chunks = e.cdivi(p_H, spec->rows_per_chunk);
    V n_units_v = e.mul(p_N, n_chunks);
    /* Same formula as GroupedWgradMfmaSpec.n_z, which sizes the grid. */
    V n_ct = e.cdivi(p_C, kBC);
    V planes = e.mul(n_ct, n_q_v);
    V nz = e.ci(spec->target_blocks);
    nz = e.add(nz, e.sub(planes, c1));
    nz = e.div(nz, planes);
    nz = e.smax(nz, c1);
    V n_z_v = e.smin(n_units_v, nz);

    V a_rsrc = rocke_b_buffer_rsrc(b, P.A, P.A_bytes); /* dY */
    V b_rsrc = rocke_b_buffer_rsrc(b, P.Bp, P.B_bytes); /* X */

    /* ---- LDS: one allocation per staging buffer (one alias scope each) ---- */
    const int X_ELEMS = WQ * IX * kWave * 8;
    const int Y_ELEMS = WQ * IY * kWave * 8;
    std::vector<V> x_stage, y_stage, x_addr, y_addr;
    for(int i = 0; i < D; ++i)
    {
        std::string h = "xs" + std::to_string(i);
        x_stage.push_back(e.smem_alloc_1d(io_type, X_ELEMS, h.c_str()));
    }
    for(int i = 0; i < D; ++i)
    {
        std::string h = "ys" + std::to_string(i);
        y_stage.push_back(e.smem_alloc_1d(io_type, Y_ELEMS, h.c_str()));
    }
    for(V s : x_stage)
    {
        x_addr.push_back(rocke_b_smem_addr_of(b, s));
    }
    for(V s : y_stage)
    {
        y_addr.push_back(rocke_b_smem_addr_of(b, s));
    }

    /* DMA lane geometry: slot L = (wave * I + i) * 64 + lane -> (col, chunk). */
    auto dma_lanes = [&](int n_instr, int cols) {
        std::vector<Lane> out;
        for(int i = 0; i < n_instr; ++i)
        {
            V slot = e.muli(wave, n_instr);
            slot = e.addi(slot, i);
            slot = e.muli(slot, kWave);
            slot = e.add(slot, lane);
            Lane l;
            l.col = e.divi(slot, kCH);
            V ch = e.modi(slot, kCH);
            ch = e.muli(ch, 8);
            l.cc = e.add(c_blk0, ch);
            V a = e.lti(l.col, cols);
            l.ok = e.land(a, e.lt(l.cc, p_C));
            out.push_back(l);
        }
        return out;
    };

    std::vector<Off> x_offs, y_offs;
    {
        std::vector<Lane> xl = dma_lanes(IX, XC);
        for(const Lane& l : xl)
        {
            V wi = e.add(e.subi(q0, PAD), l.col);
            V a = e.ge(wi, c0);
            a = e.land(a, e.lt(wi, p_W));
            V okc = e.land(l.ok, a);
            V off = e.add(e.mul(wi, sx_w), l.cc);
            x_offs.push_back({off, okc});
        }
        std::vector<Lane> yl = dma_lanes(IY, BQ);
        for(const Lane& l : yl)
        {
            V wo = e.add(q0, l.col);
            V off = e.add(e.mul(wo, sdy_w), l.cc);
            V ok = e.land(l.ok, e.lt(wo, p_Wo));
            y_offs.push_back({off, ok});
        }
    }

    auto dma = [&](V rsrc, V addr, int n_instr, int i, V voff) {
        V o = e.muli(wave, n_instr);
        o = e.addi(o, i);
        o = e.muli(o, kWave * 16);
        V lds = rocke_b_smem_ptr_add(b, addr, e.zext64(o));
        rocke_b_async_buffer_load_lds(b, rsrc, lds, voff, c0, 4, 0);
    };

    /* ---- operand reads: out lane L gets channel L, four columns ---- */
    V tr_row = e.modi(e.divi(lane, 4), 4);
    V tr_cg = e.muli(e.divi(lane, 16), 4);
    tr_cg = e.add(tr_cg, e.modi(lane, 4));
    V tr_base = e.muli(tr_row, kBC);
    tr_base = e.add(tr_base, e.muli(tr_cg, 4));
    /* The partner half of an 8-channel group (channel ^ 4). */
    V tr_swap = e.muli(tr_row, kBC);
    tr_swap = e.add(tr_swap, e.muli(e.bxor(tr_cg, c1), 4));
    V col_w = e.muli(wave, 4 * Q4); /* this wave's first column in the tile */

    auto read = [&](V stage, V base, int col) {
        V idx = e.addi(col_w, col);
        idx = e.muli(idx, kBC);
        idx = e.add(base, idx);
        return e.ds_read_tr(stage, c0, idx, io_type);
    };

    V zero_acc = rocke_b_zero_vec_f32(b, 4);
    V zero_frag = rocke_b_zero_vec(b, io_type, 4);
    const int NACC = KH * KW * HV;
    /* The carried ring holds the dY fragments of the KH - 1 rows before the
     * current one, newest first: [row][q4][half]. */
    const int n_ring = (KH - 1) * Q4 * HV;
    auto acc_i = [&](int r, int s, int h) { return (r * KW + s) * HV + h; };

    auto pack = [&](const std::vector<V>& accs, const Ring& ring) {
        std::vector<V> out(accs);
        for(const auto& row : ring)
        {
            for(const auto& q : row)
            {
                for(V v : q)
                {
                    out.push_back(v);
                }
            }
        }
        return out;
    };
    auto unpack = [&](const std::vector<V>& vals, std::vector<V>* accs, Ring* ring) {
        accs->assign(vals.begin(), vals.begin() + NACC);
        ring->assign(KH - 1, std::vector<std::vector<V>>(Q4));
        for(int j = 0; j < KH - 1; ++j)
        {
            for(int q = 0; q < Q4; ++q)
            {
                int o = NACC + (j * Q4 + q) * HV;
                (*ring)[j][q].assign(vals.begin() + o, vals.begin() + o + HV);
            }
        }
    };

    std::vector<std::string> unit_names = iter_names("acc", NACC);
    std::vector<std::string> ring_names = iter_names("ring", n_ring);
    unit_names.insert(unit_names.end(), ring_names.begin(), ring_names.end());
    std::vector<V> unit_init(NACC, zero_acc);
    unit_init.insert(unit_init.end(), n_ring, zero_frag);

    std::vector<V> unit_out = e.for_iter(
        zslot,
        n_units_v,
        n_z_v,
        unit_names,
        unit_init,
        "unit",
        [&](V unit, std::vector<V> carried) {
            std::vector<V> accs;
            Ring ring;
            unpack(carried, &accs, &ring);
            V n_img = e.mod(unit, p_N);
            V chunk = e.div(unit, p_N);
            V y0 = e.muli(chunk, spec->rows_per_chunk);
            V k_rpc = e.ci(spec->rows_per_chunk);
            V n_rows = e.smin(k_rpc, e.sub(p_H, y0));
            V n_steps = e.addi(n_rows, PRE);
            V x_img = e.mul(n_img, sx_n);
            V y_img = e.mul(n_img, sdy_n);
            const int PER_STEP = IX + IY;

            /* DMA step t: X row y0 - PRE + t (steps >= PRE), dY row y + PAD. */
            auto issue = [&](int buf, V t) {
                V y = e.add(e.subi(y0, PRE), t);
                V x_ok = e.gei(t, PRE);
                x_ok = e.land(x_ok, e.lt(t, n_steps));
                V x_row = e.add(x_img, e.mul(y, sx_h));
                for(int i = 0; i < (int)x_offs.size(); ++i)
                {
                    V ok = e.land(x_offs[i].ok, x_ok);
                    V voff = e.mul(e.add(x_offs[i].off, x_row), c2);
                    voff = e.select(ok, voff, oob);
                    dma(b_rsrc, x_addr[buf], IX, i, voff);
                }
                V pr = e.addi(y, PAD);
                V y_ok = e.ge(pr, c0);
                y_ok = e.land(y_ok, e.lt(pr, p_Ho));
                y_ok = e.land(y_ok, e.lt(t, n_steps));
                V y_row = e.add(y_img, e.mul(pr, sdy_h));
                for(int i = 0; i < (int)y_offs.size(); ++i)
                {
                    V ok = e.land(y_offs[i].ok, y_ok);
                    V voff = e.mul(e.add(y_offs[i].off, y_row), c2);
                    voff = e.select(ok, voff, oob);
                    dma(a_rsrc, y_addr[buf], IY, i, voff);
                }
            };

            /* Barrier, DMA step t + D - 1, read and MFMA step t. */
            auto step = [&](V t, int buf, std::vector<V>& accs_s, Ring& ring_s, bool do_compute) {
                rocke_b_s_waitcnt(b, (D - 2) * PER_STEP, 0, -1);
                rocke_b_s_barrier_bare(b);
                issue((buf + D - 1) % D, e.addi(t, D - 1));
                std::vector<std::vector<V>> newest;
                for(int q = 0; q < Q4; ++q)
                {
                    std::vector<V> h;
                    h.push_back(read(y_stage[buf], tr_base, 4 * q));
                    if(HV == 2)
                    {
                        h.push_back(read(y_stage[buf], tr_swap, 4 * q));
                    }
                    newest.push_back(h);
                }
                Ring full;
                full.push_back(newest);
                full.insert(full.end(), ring_s.begin(), ring_s.end());
                if(do_compute)
                {
                    for(int s = 0; s < KW; ++s)
                    {
                        for(int q = 0; q < Q4; ++q)
                        {
                            V a = read(x_stage[buf], tr_base, 4 * q + s);
                            for(int r = 0; r < KH; ++r)
                            {
                                for(int h = 0; h < HV; ++h)
                                {
                                    int i = acc_i(r, s, h);
                                    accs_s[i] = bf16 ? rocke_b_mfma_f32_4x4x4_bf16(
                                                           b, a, full[r][q][h], accs_s[i])
                                                     : rocke_b_mfma_f32_4x4x4_f16(
                                                           b, a, full[r][q][h], accs_s[i]);
                                }
                            }
                        }
                    }
                }
                ring_s.assign(full.begin(), full.begin() + (KH - 1));
            };

            for(int t = 0; t < D - 1; ++t)
            {
                issue(t, e.ci(t));
            }
            /* Prologue: the PRE dY rows the first input row pairs with. */
            for(int t = 0; t < PRE; ++t)
            {
                step(e.ci(t), t % D, accs, ring, false);
            }
            /* Main: one input row per step, unrolled by D. */
            V trips = e.cdivi(n_rows, D);
            trips = e.smax(trips, c1);
            std::vector<V> init = pack(accs, ring);
            std::vector<V> res = e.for_iter(c0,
                                            trips,
                                            c1,
                                            iter_names("a", init.size()),
                                            init,
                                            "rb",
                                            [&](V rb, std::vector<V> la) {
                                                std::vector<V> l_acc;
                                                Ring l_ring;
                                                unpack(la, &l_acc, &l_ring);
                                                for(int u = 0; u < D; ++u)
                                                {
                                                    V t = e.muli(rb, D);
                                                    t = e.addi(t, PRE);
                                                    t = e.addi(t, u);
                                                    step(t, (PRE + u) % D, l_acc, l_ring, true);
                                                }
                                                return pack(l_acc, l_ring);
                                            });
            /* Drain the tail DMAs before the next unit reuses the buffers. */
            rocke_b_s_waitcnt(b, 0, 0, -1);
            rocke_b_s_barrier_bare(b);
            return res;
        });
    if((int)unit_out.size() != NACC + n_ring)
    {
        return NULL;
    }
    std::vector<V> accs(unit_out.begin(), unit_out.begin() + NACC);

    /* ---- epilogue: sum the waves in LDS, store ws[part][tap][o] ---- */
    V red = e.smem_alloc_1d(rocke_f32(), WQ * kWave * 4, "red");
    V part = e.add(e.mul(zslot, n_q_v), qtile);
    V lane4 = e.modi(lane, 4);
    V quad = e.muli(e.divi(lane, 4), 4);
    for(int i_acc = 0; i_acc < NACC; ++i_acc)
    {
        int r = i_acc / (KW * HV), s = (i_acc / HV) % KW, h = i_acc % HV;
        e.smem_store_f32(red, c0, e.muli(tid, 4), accs[i_acc], 4);
        rocke_b_sync(b);
        e.if_(e.lti(tid, kWave), [&]() {
            V tot = e.smem_load_f32(red, c0, e.muli(lane, 4), 4);
            for(int w = 1; w < WQ; ++w)
            {
                V idx = e.addi(lane, w * kWave);
                idx = e.muli(idx, 4);
                V v = e.smem_load_f32(red, c0, idx, 4);
                tot = rocke_b_vector_add(b, tot, v);
            }
            V o_loc = e.bxor(quad, e.ci(4 * h));
            o_loc = e.add(o_loc, lane4);
            V o_glb = e.add(c_blk0, o_loc);
            /* Channel c of the group: (4 * (L // 4) + i) % cpg. */
            e.if_(e.lt(o_glb, p_K), [&]() {
                for(int el = 0; el < 4; ++el)
                {
                    V c_in = e.modi(e.addi(quad, el), CPG);
                    V tap = e.ci((r * KW + s) * CPG);
                    tap = e.add(tap, c_in);
                    V off = e.muli(part, TAPS);
                    off = e.add(off, tap);
                    off = e.mul(off, p_K);
                    off = e.add(off, o_glb);
                    rocke_b_global_store(b, P.D, off, rocke_b_vec_extract(b, tot, el), 4);
                }
            });
        });
        rocke_b_sync(b);
    }
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_grouped_wgrad_mfma(
    rocke_ir_builder_t* b, const rocke_grouped_wgrad_mfma_spec_t* spec, const char* arch)
{
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    return build_grouped(b, spec, arch ? arch : "gfx950");
}

extern "C" rocke_kernel_def_t* rocke_build_grouped_wgrad_mfma_new(
    rocke_ir_builder_t* b, const rocke_grouped_wgrad_mfma_spec_t* spec, const char* arch)
{
    return build_new(
        b,
        spec != NULL,
        "build_grouped_wgrad_mfma",
        [&](char* out, size_t cap) { return rocke_grouped_wgrad_mfma_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_grouped_wgrad_mfma(b, spec, arch); });
}

extern "C" rocke_status_t
    rocke_grouped_wgrad_mfma_lower_to_llvm(const rocke_grouped_wgrad_mfma_spec_t* spec,
                                           const char* arch,
                                           rocke_llvm_flavor_t flavor,
                                           char** out_ll,
                                           char* err,
                                           size_t err_cap)
{
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    return lower_to_llvm(
        spec != NULL,
        arch,
        flavor,
        out_ll,
        err,
        err_cap,
        "build_grouped_wgrad_mfma failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_grouped_wgrad_mfma_new(b, spec, arch); });
}
