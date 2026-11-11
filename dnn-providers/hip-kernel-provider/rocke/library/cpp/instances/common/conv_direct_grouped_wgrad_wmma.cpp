// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_wgrad_wmma.cpp -- C++ engine port of
 * library/kernels/common/conv_direct_grouped_wgrad_wmma.py: the gfx1250
 * grouped wgrad for 4 and 8 channels per group, packed block-diagonally into
 * the wave32 WMMA 16x16x32 (build_grouped_wgrad_wmma).
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the contract in
 * rocke/instance_conv_direct_wgrad_internal.hpp.
 */
#include "rocke/instance_conv_direct_grouped_wgrad_wmma.h"

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

constexpr int kWave32 = 32;
constexpr int kTile = 16; /* channels per wave (WMMA M = N) */
constexpr int kKStep = 32; /* output columns per WMMA (WMMA K) */
constexpr int kPadCh = 8; /* LDS row padding (channels) against bank conflicts */
constexpr int kDummy = 8; /* elements past each buffer: the padding lanes' copy target */
/* The kernel's LDS budget, not the arch capacity: the ceiling its knob grid
 * was tuned under. The arch's own capacity is checked as well. */
constexpr long long kLdsBudget = 160 * 1024;

const char* bf16_flag[1] = {"bf16"};

int block_c(const rocke_grouped_wgrad_wmma_spec_t* s)
{
    return kTile * s->waves_c;
}

int block_q(const rocke_grouped_wgrad_wmma_spec_t* s)
{
    return kKStep * s->q_steps;
}

int x_cols(const rocke_grouped_wgrad_wmma_spec_t* s)
{
    return block_q(s) + s->problem.KW - 1;
}

int pitch(const rocke_grouped_wgrad_wmma_spec_t* s)
{
    return block_c(s) + kPadCh;
}

int n_channel_tiles(const rocke_grouped_wgrad_wmma_spec_t* s)
{
    /* The validator bounds total_c to i32. */
    return cdiv((int)rocke_direct_conv_problem_total_c(&s->problem), block_c(s));
}

int n_q(const rocke_grouped_wgrad_wmma_spec_t* s)
{
    return cdiv(problem_wo(&s->problem), block_q(s));
}

int n_units(const rocke_grouped_wgrad_wmma_spec_t* s)
{
    return s->problem.N * cdiv(s->problem.H, s->rows_per_chunk);
}

int n_z(const rocke_grouped_wgrad_wmma_spec_t* s)
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

} // namespace

extern "C" rocke_grouped_wgrad_wmma_spec_t rocke_grouped_wgrad_wmma_spec_default(void)
{
    rocke_grouped_wgrad_wmma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "grouped_wgrad_wmma";
    s.waves_c = 4;
    s.q_steps = 2;
    s.rows_per_chunk = 32;
    s.target_blocks = 1024;
    s.stages = 2;
    return s;
}

extern "C" int rocke_grouped_wgrad_wmma_threads(const rocke_grouped_wgrad_wmma_spec_t* s)
{
    return s->waves_c * kWave32;
}

extern "C" long rocke_grouped_wgrad_wmma_lds_bytes(const rocke_grouped_wgrad_wmma_spec_t* s)
{
    return (long)s->stages * (((long)x_cols(s) + block_q(s)) * pitch(s) + 2 * kDummy) * 2;
}

extern "C" int rocke_grouped_wgrad_wmma_n_partitions(const rocke_grouped_wgrad_wmma_spec_t* s)
{
    return n_q(s) * n_z(s);
}

extern "C" int rocke_grouped_wgrad_wmma_fold_taps(const rocke_grouped_wgrad_wmma_spec_t* s)
{
    return s->problem.KH * s->problem.KW * s->problem.cpg;
}

extern "C" long long
    rocke_grouped_wgrad_wmma_fold_channels(const rocke_grouped_wgrad_wmma_spec_t* s)
{
    return rocke_direct_conv_problem_total_k(&s->problem);
}

extern "C" long rocke_grouped_wgrad_wmma_workspace_bytes(const rocke_grouped_wgrad_wmma_spec_t* s)
{
    return (long)rocke_grouped_wgrad_wmma_n_partitions(s) * rocke_grouped_wgrad_wmma_fold_taps(s)
           * rocke_direct_conv_problem_total_k(&s->problem) * 4;
}

extern "C" void rocke_grouped_wgrad_wmma_launch_geometry(const rocke_grouped_wgrad_wmma_spec_t* s,
                                                         int grid[3],
                                                         int block[3])
{
    grid[0] = n_channel_tiles(s);
    grid[1] = n_q(s);
    grid[2] = n_z(s);
    block[0] = rocke_grouped_wgrad_wmma_threads(s);
    block[1] = 1;
    block[2] = 1;
}

extern "C" rocke_status_t rocke_grouped_wgrad_wmma_kernel_name(
    const rocke_grouped_wgrad_wmma_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    char f_buf[64], wc_buf[24], qs_buf[24], rpc_buf[24], tb_buf[24], st_buf[24];
    snprintf(f_buf, sizeof f_buf, "f%dx%dp%ds%dc%d", p->KH, p->KW, p->PAD, p->stride, p->cpg);
    snprintf(wc_buf, sizeof wc_buf, "wc%d", s->waves_c);
    snprintf(qs_buf, sizeof qs_buf, "qs%d", s->q_steps);
    snprintf(rpc_buf, sizeof rpc_buf, "rpc%d", s->rows_per_chunk);
    snprintf(tb_buf, sizeof tb_buf, "tb%d", s->target_blocks);
    snprintf(st_buf, sizeof st_buf, "st%d", s->stages);
    const char* parts[6] = {f_buf, wc_buf, qs_buf, rpc_buf, tb_buf, st_buf};
    int on[1] = {problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(s->name, parts, 6, bf16_flag, on, 1, out, out_cap, NULL);
}

extern "C" bool rocke_grouped_wgrad_wmma_is_valid_spec(const rocke_grouped_wgrad_wmma_spec_t* s,
                                                       const char* arch,
                                                       char* reason,
                                                       size_t reason_cap)
{
    char msg[ROCKE_ERR_MSG_CAP];
    if(s == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx1250";
    }
    if(rocke_direct_channel_totals_reason(&s->problem, reason, reason_cap))
        return false;
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(rocke_direct_dilation_reason(p, msg, sizeof msg))
    {
        rocke_spec_set_reason(reason, reason_cap, msg);
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
    else if(rocke_direct_conv_one_padding_reason(p, "", msg, sizeof msg))
    {
        why = msg; /* the kernel offsets columns by PAD */
    }
    else if(pymod((int)rocke_direct_conv_problem_total_c(p), kTile))
    {
        why = "C must be a multiple of 16 (whole packed tiles)";
    }
    else if(s->waves_c != 1 && s->waves_c != 2 && s->waves_c != 4 && s->waves_c != 8)
    {
        why = "waves_c in (1, 2, 4, 8)";
    }
    else if(s->q_steps < 1 || s->q_steps > 4)
    {
        why = "q_steps in 1..4";
    }
    else if(s->rows_per_chunk < 1)
    {
        why = "rows_per_chunk >= 1";
    }
    else if(s->target_blocks < 1)
    {
        why = "target_blocks >= 1";
    }
    else if(s->stages < 2 || s->stages > 4)
    {
        why = "stages in 2..4";
    }
    if(why != NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, why);
        return false;
    }
    const rocke_archtarget_t* target
        = wgrad_wmma_arch_target(arch, problem_is_bf16(p), reason, reason_cap);
    if(target == NULL)
        return false;
    if(wgrad_wmma_lds_over(
           rocke_grouped_wgrad_wmma_lds_bytes(s), kLdsBudget, target, arch, reason, reason_cap))
        return false;
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

namespace
{

/* One copy slot of copy_slots(): lane slot i * T + tid -> (column, chunk). */
struct Slot
{
    V col, lds_i, ok_c, in_tile, cc;
};

/* A staged row's copy: LDS element, ok (incl. in_tile), in_tile, offset. */
struct Copy
{
    V lds_i, ok, in_tile, off;
};

/* ring[row][q]: dY fragments, newest row first. */
using Ring = std::vector<std::vector<V>>;

rocke_kernel_def_t* build_grouped(rocke_ir_builder_t* b,
                                  const rocke_grouped_wgrad_wmma_spec_t* spec,
                                  const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_grouped_wgrad_wmma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        fail_invalid(b, "invalid GroupedWgradWmmaSpec", reason);
        return NULL;
    }
    Em e{b};
    const rocke_direct_conv_problem_t& p = spec->problem;
    const int KH = p.KH, KW = p.KW, PAD = p.PAD, CPG = p.cpg;
    const int TAPS = rocke_grouped_wgrad_wmma_fold_taps(spec);
    const int QS = spec->q_steps;
    const int BC = block_c(spec), BQ = block_q(spec);
    const int XC = x_cols(spec);
    const int PITCH = pitch(spec);
    const int T = rocke_grouped_wgrad_wmma_threads(spec);
    const int D = spec->stages;
    const int PRE = KH - 1; /* dY rows a unit needs before its first input row */
    const int CH8 = BC / 8; /* 16-byte chunks per staged pixel */
    const bool bf16 = problem_is_bf16(&p);
    const rocke_type_t* io_type = bf16 ? rocke_bf16() : rocke_f16();
    const char* wmma_op = bf16 ? "wmma_gfx1250_f32_16x16x32_bf16" : "wmma_gfx1250_f32_16x16x32_f16";

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
    V p_N = P.p_N;
    V p_C = P.p_total_c;
    V p_K = P.p_total_k;
    V p_H = P.p_Hi, p_W = P.p_Wi;
    V p_Ho = P.p_Ho, p_Wo = P.p_Wo;
    V sdy_n = P.p_A_stride_n, sdy_h = P.p_A_stride_hi, sdy_w = P.p_A_stride_wi;
    V sx_n = P.p_B_stride_n, sx_h = P.p_B_stride_hi, sx_w = P.p_B_stride_wi;
    V dy_ptr = P.A, x_ptr = P.Bp;

    V tid = rocke_b_thread_id_x(b);
    V lane = e.modi(tid, kWave32);
    V wave = rocke_b_readfirstlane(b, e.divi(tid, kWave32));
    V ctile = rocke_b_block_id_x(b);
    V qtile = rocke_b_block_id_y(b);
    V zslot = rocke_b_block_id_z(b);
    V c_blk0 = e.muli(ctile, BC);
    V q0 = e.muli(qtile, BQ);
    V n_q_v = e.cdivi(p_Wo, BQ);
    V n_chunks = e.cdivi(p_H, spec->rows_per_chunk);
    V n_units_v = e.mul(p_N, n_chunks);
    /* Same formula as GroupedWgradWmmaSpec.n_z, which sizes the grid. */
    V n_ct = e.cdivi(p_C, BC);
    V planes = e.mul(n_ct, n_q_v);
    V nz = e.ci(spec->target_blocks);
    nz = e.add(nz, e.sub(planes, c1));
    nz = e.div(nz, planes);
    nz = e.smax(nz, c1);
    V n_z_v = e.smin(n_units_v, nz);

    /* ---- LDS: [column][channel] rows, one allocation per buffer ---- */
    const int X_DUMMY = XC * PITCH, Y_DUMMY = BQ * PITCH;
    std::vector<V> x_stage, y_stage;
    for(int i = 0; i < D; ++i)
    {
        std::string h = "xs" + std::to_string(i);
        x_stage.push_back(e.smem_alloc_1d(io_type, X_DUMMY + kDummy, h.c_str()));
    }
    for(int i = 0; i < D; ++i)
    {
        std::string h = "ys" + std::to_string(i);
        y_stage.push_back(e.smem_alloc_1d(io_type, Y_DUMMY + kDummy, h.c_str()));
    }

    /* ---- copy slots: slot = i * T + tid -> (column, 8-channel chunk) ---- */
    auto copy_slots = [&](int cols) {
        std::vector<Slot> out;
        for(int i = 0; i < cdiv(cols * CH8, T); ++i)
        {
            V k = e.ci((int64_t)i * T);
            V slot = e.add(k, tid);
            Slot sl;
            sl.col = e.divi(slot, CH8);
            V ch = e.modi(slot, CH8);
            ch = e.muli(ch, 8);
            const bool partial = (i + 1) * T > cols * CH8; /* the last pass has idle lanes */
            sl.lds_i = e.add(e.muli(sl.col, PITCH), ch);
            sl.ok_c = e.lt(e.add(c_blk0, ch), p_C);
            sl.in_tile = partial ? e.lti(sl.col, cols) : NULL;
            sl.cc = e.add(c_blk0, ch);
            out.push_back(sl);
        }
        return out;
    };
    auto slot_ok = [&](V ok, V in_tile) { return in_tile == NULL ? ok : e.land(ok, in_tile); };

    std::vector<Copy> x_slots, y_slots;
    {
        std::vector<Slot> xs = copy_slots(XC);
        for(const Slot& sl : xs)
        {
            V wi = e.add(e.subi(q0, PAD), sl.col);
            V a = e.ge(wi, c0);
            a = e.land(a, e.lt(wi, p_W));
            V ok = e.land(sl.ok_c, a);
            ok = slot_ok(ok, sl.in_tile);
            V off = e.add(e.mul(wi, sx_w), sl.cc);
            x_slots.push_back({sl.lds_i, ok, sl.in_tile, off});
        }
        std::vector<Slot> ys = copy_slots(BQ);
        for(const Slot& sl : ys)
        {
            V wo = e.add(q0, sl.col);
            V ok = e.land(sl.ok_c, e.lt(wo, p_Wo));
            ok = slot_ok(ok, sl.in_tile);
            V off = e.add(e.mul(wo, sdy_w), sl.cc);
            y_slots.push_back({sl.lds_i, ok, sl.in_tile, off});
        }
    }
    const int PER_STEP = (int)(x_slots.size() + y_slots.size()); /* copies per row step */

    V zero8 = rocke_b_zero_vec(b, io_type, 8);

    auto copy_row
        = [&](V src, V stage, int dummy, const std::vector<Copy>& slots, V row_base, V row_ok) {
              for(const Copy& c : slots)
              {
                  V live = e.land(c.ok, row_ok);
                  V src_i = e.select(live, e.add(row_base, c.off), c0);
                  V lds_ix[2];
                  lds_ix[0] = c0;
                  V kd = e.ci(dummy);
                  lds_ix[1] = e.select(live, c.lds_i, kd);
                  rocke_b_global_load_async_to_lds(b, src, src_i, stage, lds_ix, 2, 16, 0, 0);
                  V pad = c.in_tile == NULL ? e.lnot(live) : e.land(c.in_tile, e.lnot(live));
                  e.if_(pad, [&]() { e.smem_store(stage, c0, c.lds_i, zero8, 8); });
              }
          };

    /* ---- fragment reads: ds_load_tr16_b128 of a 16 (column) x 16
     * (channel) block of [column][channel]; after the transpose lane l holds
     * channel l % 16 at columns (l // 16) * 16 + 0 .. 15. ---- */
    V l16 = e.modi(lane, 16);
    V lhalf = e.divi(lane, 16);
    V rd_base = e.muli(lhalf, 16);
    rd_base = e.add(rd_base, e.modi(lane, 8));
    rd_base = e.muli(rd_base, PITCH);
    {
        V ch = e.muli(wave, kTile);
        V cg = e.divi(l16, 8);
        cg = e.muli(cg, 8);
        ch = e.add(ch, cg);
        rd_base = e.add(rd_base, ch);
    }

    auto frag = [&](V stage, int col) {
        V addr = e.addi(rd_base, (int64_t)col * PITCH);
        V ix_lo[2] = {c0, addr};
        V lo = rocke_b_ds_read_tr16_b128(b, stage, ix_lo, 2, io_type);
        V ix_hi[2] = {c0, e.addi(addr, 8 * PITCH)};
        V hi = rocke_b_ds_read_tr16_b128(b, stage, ix_hi, 2, io_type);
        return rocke_b_vec_concat(b, lo, hi);
    };

    V zero_acc = rocke_b_zero_vec_f32(b, 8);
    V zero_frag = rocke_b_zero_vec(b, io_type, 16);
    const int NACC = KH * KW;
    /* The carried ring holds the dY fragments of the KH - 1 rows before the
     * current one, newest first: [row][q]. */
    const int n_ring = (KH - 1) * QS;

    auto pack = [&](const std::vector<V>& accs, const Ring& ring) {
        std::vector<V> out(accs);
        for(const auto& row : ring)
        {
            out.insert(out.end(), row.begin(), row.end());
        }
        return out;
    };
    auto unpack = [&](const std::vector<V>& vals, std::vector<V>* accs, Ring* ring) {
        accs->assign(vals.begin(), vals.begin() + NACC);
        ring->assign(KH - 1, std::vector<V>());
        for(int j = 0; j < KH - 1; ++j)
        {
            int o = NACC + j * QS;
            (*ring)[j].assign(vals.begin() + o, vals.begin() + o + QS);
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

            /* Copy step t: X row y0 - PRE + t (steps >= PRE) and dY row y + PAD. */
            auto issue = [&](int buf, V t) {
                V y = e.add(e.subi(y0, PRE), t);
                V x_ok = e.gei(t, PRE);
                x_ok = e.land(x_ok, e.lt(t, n_steps));
                V x_row = e.add(x_img, e.mul(y, sx_h));
                copy_row(x_ptr, x_stage[buf], X_DUMMY, x_slots, x_row, x_ok);
                V pr = e.addi(y, PAD);
                V y_ok = e.ge(pr, c0);
                y_ok = e.land(y_ok, e.lt(pr, p_Ho));
                y_ok = e.land(y_ok, e.lt(t, n_steps));
                V y_row = e.add(y_img, e.mul(pr, sdy_h));
                copy_row(dy_ptr, y_stage[buf], Y_DUMMY, y_slots, y_row, y_ok);
            };

            /* Publish step t, copy step t + D - 1, multiply step t. */
            auto step = [&](V t, int buf, std::vector<V>& accs_s, Ring& ring_s, bool do_compute) {
                rocke_b_s_wait_asynccnt(b, (D - 2) * PER_STEP);
                rocke_b_sync_lds_only(b);
                issue((buf + D - 1) % D, e.addi(t, D - 1));
                std::vector<V> newest;
                for(int q = 0; q < QS; ++q)
                {
                    newest.push_back(frag(y_stage[buf], kKStep * q));
                }
                Ring full;
                full.push_back(newest);
                full.insert(full.end(), ring_s.begin(), ring_s.end());
                if(do_compute)
                {
                    for(int s = 0; s < KW; ++s)
                    {
                        for(int q = 0; q < QS; ++q)
                        {
                            V xb = frag(x_stage[buf], kKStep * q + s);
                            for(int r = 0; r < KH; ++r)
                            {
                                int i = r * KW + s;
                                accs_s[i]
                                    = rocke_b_mma(b, wmma_op, full[r][q], xb, accs_s[i], NULL, 0);
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
            /* Main: one input row per step, unrolled by D so buffers are constant. */
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
            /* Drain the copy issued by the last step before the next unit
             * reuses the buffers. */
            rocke_b_s_wait_asynccnt(b, 0);
            rocke_b_sync_lds_only(b);
            return res;
        });
    if((int)unit_out.size() != NACC + n_ring)
    {
        return NULL;
    }
    std::vector<V> accs(unit_out.begin(), unit_out.begin() + NACC);

    /* ---- epilogue: each wave stores the diagonal blocks of its tile ----
     * Lane l, item e: dY channel o = 8 * (l // 16) + e, X channel c = l % 16.
     * cpg 4: lane l keeps items 4 * (g % 2) .. + 3 when g // 2 == l // 16,
     * g = c // 4; cpg 8: all 8 items when c // 8 == l // 16. */
    V part = e.add(e.mul(zslot, n_q_v), qtile);
    V gc = e.divi(l16, CPG);
    V c_in = e.modi(l16, CPG);
    V keep, hi = NULL, o_loc;
    if(CPG == 4)
    {
        keep = e.eq(e.divi(gc, 2), lhalf);
        hi = e.eq(e.modi(gc, 2), c1);
        o_loc = e.muli(gc, 4);
    }
    else
    {
        keep = e.eq(gc, lhalf);
        o_loc = e.muli(lhalf, 8);
    }
    V o_glb = e.add(c_blk0, e.muli(wave, kTile));
    o_glb = e.add(o_glb, o_loc);
    V cond = e.land(keep, e.lt(o_glb, p_K));
    e.if_(cond, [&]() {
        for(int r = 0; r < KH; ++r)
        {
            for(int s = 0; s < KW; ++s)
            {
                V acc = accs[r * KW + s];
                V k = e.ci((int64_t)(r * KW + s) * CPG);
                V tap = e.add(k, c_in);
                V off = e.muli(part, TAPS);
                off = e.add(off, tap);
                off = e.mul(off, p_K);
                off = e.add(off, o_glb);
                if(CPG == 4)
                {
                    V vals[4];
                    for(int el = 0; el < 4; ++el)
                    {
                        V a = rocke_b_vec_extract(b, acc, 4 + el);
                        V a0 = rocke_b_vec_extract(b, acc, el);
                        vals[el] = e.select(hi, a, a0);
                    }
                    rocke_b_global_store_vN(
                        b, P.D, off, rocke_b_vec_pack(b, vals, 4, rocke_f32()), 4, 16);
                }
                else
                {
                    rocke_b_global_store_vN(b, P.D, off, acc, 8, 16);
                }
            }
        }
    });
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_grouped_wgrad_wmma(
    rocke_ir_builder_t* b, const rocke_grouped_wgrad_wmma_spec_t* spec, const char* arch)
{
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    return build_grouped(b, spec, arch ? arch : "gfx1250");
}

extern "C" rocke_kernel_def_t* rocke_build_grouped_wgrad_wmma_new(
    rocke_ir_builder_t* b, const rocke_grouped_wgrad_wmma_spec_t* spec, const char* arch)
{
    return build_new(
        b,
        spec != NULL,
        "build_grouped_wgrad_wmma",
        [&](char* out, size_t cap) { return rocke_grouped_wgrad_wmma_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_grouped_wgrad_wmma(b, spec, arch); });
}

extern "C" rocke_status_t
    rocke_grouped_wgrad_wmma_lower_to_llvm(const rocke_grouped_wgrad_wmma_spec_t* spec,
                                           const char* arch,
                                           rocke_llvm_flavor_t flavor,
                                           char** out_ll,
                                           char* err,
                                           size_t err_cap)
{
    if(arch == NULL)
    {
        arch = "gfx1250";
    }
    return lower_to_llvm(
        spec != NULL,
        arch,
        flavor,
        out_ll,
        err,
        err_cap,
        "build_grouped_wgrad_wmma failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_grouped_wgrad_wmma_new(b, spec, arch); });
}
