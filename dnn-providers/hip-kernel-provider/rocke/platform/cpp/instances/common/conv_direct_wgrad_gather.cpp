// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_wgrad_gather.cpp -- C++ engine port of build_wgrad_gather
 * (library/kernels/common/conv_direct_wgrad_gather.py): the dense small-C
 * direct wgrad, one GEMM per filter row with N = (s, c).
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the contract in
 * rocke/instance_conv_direct_wgrad_internal.hpp.
 */
#include "rocke/instance_conv_direct_wgrad_gather.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/arch_target.h" /* rocke_set_unknown_arch_reason */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */
#include "rocke/instance_conv_direct_wgrad_internal.hpp"
#include "rocke/instance_conv_direct_wgrad_tiled.h" /* rocke_direct_wgrad_pick_swizzle */

using namespace rocke_dwg;

namespace
{

constexpr int kAtom = 16;
constexpr int kWo = 32;
constexpr int kVec = 8;

int block_k(const rocke_direct_wgrad_gather_spec_t* s)
{
    return kAtom * s->wave_k16 * s->waves_k;
}

int n_atoms(const rocke_direct_wgrad_gather_spec_t* s)
{
    return cdiv(s->problem.KW * s->problem.cpg, kAtom);
}

int unroll(const rocke_direct_wgrad_gather_spec_t* s)
{
    return s->prefetch + cdiv(s->problem.KH, s->problem.stride);
}

int rps(const rocke_direct_wgrad_gather_spec_t* s)
{
    return cdiv(s->rows_per_seg, unroll(s)) * unroll(s);
}

int x_phase(const rocke_direct_wgrad_gather_spec_t* s)
{
    return pymod(-s->problem.PAD * s->problem.cpg, kVec);
}

/* Elements of one staged X row: phase + the strip's columns. */
int x_elems(const rocke_direct_wgrad_gather_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return x_phase(s) + ((kWo - 1) * p->stride + p->KW) * p->cpg;
}

int n_k_tiles(const rocke_direct_wgrad_gather_spec_t* s)
{
    return cdiv(s->problem.kpg, block_k(s));
}

/* WgradGatherSpec.dma_instrs(): 16-byte DMA instructions per wave and row. */
int dma_instrs_x(const rocke_direct_wgrad_gather_spec_t* s)
{
    return cdiv(cdiv(cdiv(x_elems(s), kVec), kWave), s->waves_k);
}

int dma_instrs_dy(const rocke_direct_wgrad_gather_spec_t* s)
{
    return cdiv(cdiv(kWo * block_k(s) / kVec, kWave), s->waves_k);
}

/* Elements of one X ring slot: whole DMA instructions of every wave, +16
 * because a transpose read of the last column atom runs past it. */
int x_slot_elems(const rocke_direct_wgrad_gather_spec_t* s)
{
    return s->waves_k * kWave * dma_instrs_x(s) * kVec + 16;
}

int dy_slot_elems(const rocke_direct_wgrad_gather_spec_t* s)
{
    return s->waves_k * kWave * dma_instrs_dy(s) * kVec;
}

} // namespace

extern "C" rocke_direct_wgrad_gather_spec_t rocke_direct_wgrad_gather_spec_default(void)
{
    rocke_direct_wgrad_gather_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "direct_wgrad_gather";
    s.wave_k16 = 2;
    s.waves_k = 1;
    s.prefetch = 2;
    s.rows_per_seg = 16;
    s.target_blocks = 1024;
    return s;
}

extern "C" int rocke_direct_wgrad_gather_threads(const rocke_direct_wgrad_gather_spec_t* s)
{
    return s->waves_k * kWave;
}

extern "C" long rocke_direct_wgrad_gather_lds_bytes(const rocke_direct_wgrad_gather_spec_t* s)
{
    /* The X and dY rings as the builder allocates them. */
    long U = unroll(s);
    return ((long)s->problem.stride * U * x_slot_elems(s) + U * dy_slot_elems(s)) * 2;
}

extern "C" int rocke_direct_wgrad_gather_n_partitions(const rocke_direct_wgrad_gather_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    int n_units = p->N * cdiv(problem_wo(p), kWo) * cdiv(problem_ho(p), rps(s));
    int c = cdiv(s->target_blocks, n_k_tiles(s));
    return n_units < c ? n_units : c;
}

extern "C" int rocke_direct_wgrad_gather_fold_taps(const rocke_direct_wgrad_gather_spec_t* s)
{
    return s->problem.KH * s->problem.KW * s->problem.cpg;
}

extern "C" long rocke_direct_wgrad_gather_workspace_bytes(const rocke_direct_wgrad_gather_spec_t* s)
{
    return (long)rocke_direct_wgrad_gather_n_partitions(s) * rocke_direct_wgrad_gather_fold_taps(s)
           * s->problem.kpg * 4;
}

extern "C" void rocke_direct_wgrad_gather_launch_geometry(const rocke_direct_wgrad_gather_spec_t* s,
                                                          int grid[3],
                                                          int block[3])
{
    grid[0] = n_k_tiles(s);
    grid[1] = rocke_direct_wgrad_gather_n_partitions(s);
    grid[2] = 1;
    block[0] = rocke_direct_wgrad_gather_threads(s);
    block[1] = 1;
    block[2] = 1;
}

extern "C" rocke_status_t rocke_direct_wgrad_gather_kernel_name(
    const rocke_direct_wgrad_gather_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    char f_buf[64], wk_buf[32], pf_buf[24], rps_buf[24], tb_buf[24];
    snprintf(f_buf, sizeof f_buf, "f%dx%dp%ds%dc%d", p->KH, p->KW, p->PAD, p->stride, p->cpg);
    snprintf(wk_buf, sizeof wk_buf, "wk%dx%d", s->wave_k16, s->waves_k);
    snprintf(pf_buf, sizeof pf_buf, "pf%d", s->prefetch);
    snprintf(rps_buf, sizeof rps_buf, "rps%d", s->rows_per_seg);
    snprintf(tb_buf, sizeof tb_buf, "tb%d", s->target_blocks);
    const char* parts[5] = {f_buf, wk_buf, pf_buf, rps_buf, tb_buf};
    const char* flag_names[1] = {"bf16"};
    int on[1] = {problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(s->name, parts, 5, flag_names, on, 1, out, out_cap, NULL);
}

extern "C" bool rocke_direct_wgrad_gather_is_valid_spec(const rocke_direct_wgrad_gather_spec_t* s,
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
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(dilation_reason(p, msg, sizeof msg))
    {
        set_reason(reason, reason_cap, msg);
        return false;
    }
    const char* why = NULL;
    int acc = p->KH * s->wave_k16 * n_atoms(s) * 4;
    if(p->groups != 1)
    {
        why = "dense layers only";
    }
    else if(!problem_dtype_ok(p))
    {
        why = "fp16/bf16 only";
    }
    else if(!target->memory.has_ds_read_tr)
    {
        ROCKE_ERR_SNPRINTF(msg, sizeof msg, "needs ds_read_b64_tr_b16 (absent on %s)", arch);
        why = msg;
    }
    else if(target->wave_size != 64 || rocke_archtarget_async_lds_max_dwords(target) < 4)
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "needs wave64 + 16-byte buffer_load_lds (absent on %s)", arch);
        why = msg;
    }
    else if(p->kpg % kVec)
    {
        why = "K must be a multiple of 8";
    }
    else if(p->KW * p->cpg > 96)
    {
        why = "KW * C must be at most 96";
    }
    /* Whole 16-byte chunks from an aligned start at a build-time phase. */
    else if((p->W * p->cpg) % kVec || (kWo * p->stride * p->cpg) % kVec)
    {
        why = "W * C and 32 * stride * C must be multiples of 8";
    }
    else if(acc > 192)
    {
        ROCKE_ERR_SNPRINTF(msg, sizeof msg, "%d accumulator VGPRs (> 192)", acc);
        why = msg;
    }
    else if(s->wave_k16 < 1 || s->waves_k < 1 || rocke_direct_wgrad_gather_threads(s) > 1024)
    {
        why = "wave_k16 >= 1 and waves_k in 1..16";
    }
    else if(s->prefetch < 1)
    {
        why = "prefetch >= 1";
    }
    else if(rocke_direct_wgrad_gather_lds_bytes(s) > 160 * 1024)
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "LDS %ld B > 160 KiB", rocke_direct_wgrad_gather_lds_bytes(s));
        why = msg;
    }
    else if(s->rows_per_seg < 1)
    {
        why = "rows_per_seg >= 1";
    }
    else if(s->target_blocks < 1)
    {
        why = "target_blocks >= 1";
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

struct Chunk
{
    V a, b, ok; /* X: (element, -, ok); dY: (row, k element, ok) */
};

struct Off
{
    V off, ok;
};

rocke_kernel_def_t*
    build(rocke_ir_builder_t* b, const rocke_direct_wgrad_gather_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_direct_wgrad_gather_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string prefix = std::string("invalid WgradGatherSpec for ") + arch;
        fail_invalid(b, prefix.c_str(), reason);
        return NULL;
    }
    Em e{b};
    const rocke_direct_conv_problem_t& p = spec->problem;
    const int KH = p.KH, KW = p.KW, PAD = p.PAD, S = p.stride, C = p.cpg;
    const bool bf16 = problem_is_bf16(&p);
    const rocke_type_t* io_type = bf16 ? rocke_bf16() : rocke_f16();
    const int WK16 = spec->wave_k16;
    const int NA = n_atoms(spec);
    const int BK = block_k(spec);
    const int CHK = BK / kVec;
    const int PH = x_phase(spec);
    const int XE = x_elems(spec);
    const int XCH = cdiv(XE, kVec); /* 16-byte chunks of a staged X row */
    const int D = spec->prefetch;
    const int U = unroll(spec);
    const int NX = S * U, NY = U;
    const int RPS = rps(spec);
    const int PRE = KH - S > 0 ? KH - S : 0;
    const int TAPS = KH * KW * C;
    const int THREADS = rocke_direct_wgrad_gather_threads(spec);

    const int IX = dma_instrs_x(spec);
    const int IY = dma_instrs_dy(spec);
    const int X_ELEMS = x_slot_elems(spec);
    /* When a pixel row of the strip (S * C elements) and the phase keep every
     * lane's 8-byte read aligned, the X operand is a transpose read too. */
    const bool X_TR = (S * C) % 4 == 0 && PH % 4 == 0;
    const int Y_ELEMS = dy_slot_elems(spec);
    const int PER_STEP = S * IX + IY;

    std::vector<int> k_origins;
    for(int o = 0; o < BK; o += kAtom)
    {
        k_origins.push_back(o);
    }
    std::vector<int> swz_k;
    {
        int zero_shift = 0;
        int bits[6];
        int n = rocke_direct_wgrad_pick_swizzle(
            CHK, &zero_shift, 1, kWo, kWo, k_origins.data(), (int)k_origins.size(), bits);
        swz_k.assign(bits, bits + n);
    }

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
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
    V p_K = P.p_total_k;
    V tid = rocke_b_thread_id_x(b);
    V lane = e.modi(tid, kWave);
    V wave = rocke_b_readfirstlane(b, e.divi(tid, kWave));
    V k_tile = rocke_b_block_id_x(b);
    V part = rocke_b_block_id_y(b);
    V k_blk0 = e.muli(k_tile, BK);
    V n_q = e.cdivi(P.p_Wo, kWo);
    V n_hb = e.cdivi(P.p_Ho, RPS);
    V n_units = e.mul(e.mul(P.p_N, n_q), n_hb);
    V n_kt = e.cdivi(p_K, BK);
    V tb = e.ci(spec->target_blocks);
    V np = e.add(tb, e.sub(n_kt, c1));
    np = e.div(np, n_kt);
    V n_part = e.smin(n_units, np);

    V a_rsrc = rocke_b_buffer_rsrc(b, P.A, P.A_bytes); /* dY */
    V b_rsrc = rocke_b_buffer_rsrc(b, P.Bp, P.B_bytes); /* X */

    std::vector<V> x_bufs, y_bufs, x_addr, y_addr;
    for(int i = 0; i < NX; ++i)
    {
        std::string h = "xg" + std::to_string(i);
        x_bufs.push_back(e.smem_alloc_1d(io_type, X_ELEMS, h.c_str()));
    }
    for(int i = 0; i < NY; ++i)
    {
        std::string h = "yg" + std::to_string(i);
        y_bufs.push_back(e.smem_alloc_1d(io_type, Y_ELEMS, h.c_str()));
    }
    for(V s : x_bufs)
    {
        x_addr.push_back(rocke_b_smem_addr_of(b, s));
    }
    for(V s : y_bufs)
    {
        y_addr.push_back(rocke_b_smem_addr_of(b, s));
    }

    auto dma = [&](V rsrc, V addr, int n_instr, int i, V voff) {
        V o = e.muli(wave, n_instr);
        o = e.addi(o, i);
        o = e.muli(o, kWave * 16);
        V lds = rocke_b_smem_ptr_add(b, addr, e.zext64(o));
        rocke_b_async_buffer_load_lds(b, rsrc, lds, voff, c0, 4, 0);
    };
    auto dma_slot = [&](int n_instr, int i) {
        V slot = e.muli(wave, n_instr);
        slot = e.addi(slot, i);
        slot = e.muli(slot, kWave);
        return e.add(slot, lane);
    };

    /* X: chunk L of a row is elements [8 L, 8 L + 8) from the aligned start. */
    std::vector<Chunk> x_chunks;
    for(int i = 0; i < IX; ++i)
    {
        V slot = dma_slot(IX, i);
        Chunk ch;
        ch.a = e.muli(slot, kVec);
        ch.b = NULL;
        ch.ok = e.lti(slot, XCH);
        x_chunks.push_back(ch);
    }
    /* dY: slot -> (pixel row, swizzled k chunk). */
    std::vector<Chunk> y_chunks;
    for(int i = 0; i < IY; ++i)
    {
        V slot = dma_slot(IY, i);
        V row = e.divi(slot, CHK);
        V phys = e.modi(slot, CHK);
        V sw = e.swz_val(row, swz_k, c1);
        V log = (sw == NULL) ? phys : e.bxor(phys, sw);
        Chunk ch;
        ch.a = row;
        ch.b = e.muli(log, kVec);
        ch.ok = e.lti(row, kWo);
        y_chunks.push_back(ch);
    }

    /* ---- fragment addresses ---- */
    V lane_hi = e.divi(lane, 16);
    V lane_q = e.modi(e.divi(lane, 4), 4);
    V lane_c4 = e.modi(lane, 4);
    V lane_n = e.modi(lane, 16);
    V wave_k0 = e.muli(wave, WK16 * kAtom);

    auto dy_index = [&](int rd, V col0) {
        V row = e.add(e.muli(lane_hi, 8), lane_q);
        row = e.addi(row, rd * 4);
        V col = e.add(col0, e.muli(lane_c4, 4));
        V chunk = e.divi(col, kVec);
        V sw = e.swz_val(row, swz_k, c1);
        V phys = (sw == NULL) ? chunk : e.bxor(chunk, sw);
        V idx = e.muli(row, CHK);
        idx = e.add(idx, phys);
        idx = e.muli(idx, kVec);
        return e.add(idx, e.modi(col, kVec));
    };
    std::vector<std::vector<V>> dy_idx(WK16);
    for(int mi = 0; mi < WK16; ++mi)
    {
        for(int rd = 0; rd < 2; ++rd)
        {
            V col0 = e.addi(wave_k0, mi * kAtom);
            dy_idx[mi].push_back(dy_index(rd, col0));
        }
    }
    /* X gather base per atom: element PH + (S * px0 + s) * C + c, px0 = 8 * (lane / 16). */
    std::vector<V> x_base;
    for(int nj = 0; nj < NA; ++nj)
    {
        V n = e.addi(lane_n, nj * kAtom);
        V s_i = e.divi(n, C);
        s_i = e.smini(s_i, KW - 1); /* n >= KW*C: any in-range column */
        V c_i = e.modi(n, C);
        V kph = e.ci(PH);
        V t = e.add(e.muli(lane_hi, 8 * S), s_i);
        t = e.add(e.muli(t, C), c_i);
        x_base.push_back(e.add(kph, t));
    }
    std::vector<std::vector<V>> x_tr_idx(NA);
    for(int nj = 0; nj < NA; ++nj)
    {
        for(int rd = 0; rd < 2; ++rd)
        {
            V k0 = e.ci(PH + nj * kAtom);
            V t = e.add(e.muli(lane_hi, 8), lane_q);
            t = e.addi(t, rd * 4);
            t = e.muli(t, S * C);
            t = e.add(t, e.muli(lane_c4, 4));
            x_tr_idx[nj].push_back(e.add(k0, t));
        }
    }

    auto read_dy = [&](int buf) {
        std::vector<V> out;
        for(int mi = 0; mi < WK16; ++mi)
        {
            V lo = e.ds_read_tr(y_bufs[buf], c0, dy_idx[mi][0], io_type);
            V hi = e.ds_read_tr(y_bufs[buf], c0, dy_idx[mi][1], io_type);
            out.push_back(rocke_b_vec_concat(b, lo, hi));
        }
        return out;
    };
    auto read_x = [&](int buf, int nj) {
        if(X_TR)
        {
            V lo = e.ds_read_tr(x_bufs[buf], c0, x_tr_idx[nj][0], io_type);
            V hi = e.ds_read_tr(x_bufs[buf], c0, x_tr_idx[nj][1], io_type);
            return rocke_b_vec_concat(b, lo, hi);
        }
        V vals[8];
        for(int j = 0; j < 8; ++j)
        {
            V idx = e.addi(x_base[nj], j * S * C);
            vals[j] = rocke_b_vec_extract(b, e.smem_load(x_bufs[buf], c0, idx, io_type, 1), 0);
        }
        return rocke_b_vec_pack(b, vals, 8, io_type);
    };

    V zero_acc = rocke_b_zero_vec_f32(b, 4);
    V zero_elem = rocke_b_vec_extract(b, rocke_b_zero_vec(b, io_type, 2), 0);
    const int n_acc = KH * WK16 * NA;
    auto acc_index = [&](int r, int mi, int nj) { return (r * WK16 + mi) * NA + nj; };

    std::vector<V> accs = e.for_iter(
        part,
        n_units,
        n_part,
        iter_names("acc", n_acc),
        std::vector<V>(n_acc, zero_acc),
        "unit",
        [&](V unit, std::vector<V> carried) {
            V hb = e.mod(unit, n_hb);
            V rest = e.div(unit, n_hb);
            V q = e.mod(rest, n_q);
            V n_img = e.div(rest, n_q);
            V h0 = e.muli(hb, RPS);
            V q0 = e.muli(q, kWo);
            V col0 = e.subi(e.muli(q0, S), PAD); /* first X column of the strip */
            V x_row0 = e.subi(e.muli(h0, S), PAD);
            V k_rps = e.ci(RPS);
            V n_steps = e.smin(k_rps, e.sub(P.p_Ho, h0));
            /* Strip X start, 8-aligned: element col0 * C sits at PH. */
            V xs0 = e.mul(n_img, P.p_B_stride_n);
            V x_start = e.add(xs0, e.subi(e.muli(col0, C), PH));
            std::vector<Off> y_offs;
            for(const Chunk& ch : y_chunks)
            {
                V wo = e.add(q0, ch.a);
                V kk = e.add(k_blk0, ch.b);
                V a = e.lt(wo, P.p_Wo);
                a = e.land(a, e.lt(kk, p_K));
                V ok = e.land(ch.ok, a);
                V off = e.mul(n_img, P.p_A_stride_n);
                off = e.add(off, e.mul(wo, P.p_A_stride_wi));
                y_offs.push_back({e.add(off, kk), ok});
            }
            /* Padding columns staged from neighbouring rows: [lo, hi) element ranges
         * of the row to zero, for the edge strips. */
            V pad_lo = e.muli(e.sub(c0, col0), C);
            pad_lo = e.smax(pad_lo, c0); /* cols < 0 */
            V cols_in = e.sub(P.p_Wi, col0); /* cols >= W start here (relative) */
            V kph = e.ci(PH);
            V pad_hi0 = e.add(kph, e.muli(cols_in, C));
            V edge = e.lt(col0, c0);
            edge = e.lor(edge, e.lti(pad_hi0, XE));

            auto issue_x = [&](V xr, int slot) {
                V hi = e.add(x_row0, xr);
                V a = e.ge(hi, c0);
                a = e.land(a, e.lt(hi, P.p_Hi));
                V lim = e.addi(e.muli(n_steps, S), PRE);
                V ok_r = e.land(a, e.lt(xr, lim));
                V row_off = e.add(x_start, e.mul(hi, P.p_B_stride_hi));
                for(int i = 0; i < (int)x_chunks.size(); ++i)
                {
                    V okk = e.land(x_chunks[i].ok, ok_r);
                    V voff = e.mul(e.add(row_off, x_chunks[i].a), c2);
                    voff = e.select(okk, voff, oob);
                    dma(b_rsrc, x_addr[slot], IX, i, voff);
                }
            };
            auto issue_y = [&](V t, int slot) {
                V ho = e.add(h0, t);
                V ok_r = e.lt(t, n_steps);
                for(int i = 0; i < (int)y_offs.size(); ++i)
                {
                    V okk = e.land(y_offs[i].ok, ok_r);
                    V row = e.mul(ho, P.p_A_stride_hi);
                    V voff = e.mul(e.add(y_offs[i].off, row), c2);
                    voff = e.select(okk, voff, oob);
                    dma(a_rsrc, y_addr[slot], IY, i, voff);
                }
            };
            auto issue_step = [&](V t, int tm) {
                for(int j = 0; j < S; ++j)
                {
                    V xr = e.addi(e.muli(t, S), PRE + j);
                    issue_x(xr, (S * tm + PRE + j) % NX);
                }
                issue_y(t, tm % NY);
            };
            /* Zero the padding columns of freshly landed X rows (edge strips). */
            auto fix_padding = [&](const std::vector<int>& slots) {
                e.if_(edge, [&]() {
                    for(int slot : slots)
                    {
                        for(int k = 0; k < cdiv(XE, THREADS); ++k)
                        {
                            V el = e.addi(tid, k * THREADS);
                            V rel = e.subi(el, PH);
                            V in_row = e.lti(el, XE);
                            V lo = e.lt(rel, pad_lo);
                            V hi = e.ge(el, pad_hi0);
                            V is_pad = e.land(in_row, e.lor(lo, hi));
                            e.if_(is_pad,
                                  [&]() { e.smem_store(x_bufs[slot], c0, el, zero_elem, 1); });
                        }
                    }
                    rocke_b_s_waitcnt(b, -1, 0, -1);
                    rocke_b_s_barrier_bare(b);
                });
            };

            for(int j = 0; j < PRE; ++j)
            {
                issue_x(e.ci(j), j % NX);
            }
            for(int t = 0; t < D; ++t)
            {
                issue_step(e.ci(t), t);
            }

            V trips = e.cdivi(n_steps, U);
            trips = e.smax(trips, c1);
            std::vector<V> accs_out
                = e.for_iter(c0,
                             trips,
                             c1,
                             iter_names("a", carried.size()),
                             carried,
                             "rb",
                             [&](V rb, std::vector<V> la) {
                                 for(int tm = 0; tm < U; ++tm)
                                 {
                                     V t = e.addi(e.muli(rb, U), tm);
                                     rocke_b_s_waitcnt(b, (D - 1) * PER_STEP, 0, -1);
                                     rocke_b_s_barrier_bare(b);
                                     std::vector<int> new_slots;
                                     if(tm == 0)
                                     {
                                         /* Step 0 of a segment also consumes the PRE leading rows. */
                                         for(int j = 0; j < PRE; ++j)
                                         {
                                             new_slots.push_back(j % NX);
                                         }
                                     }
                                     for(int j = 0; j < S; ++j)
                                     {
                                         new_slots.push_back((S * tm + PRE + j) % NX);
                                     }
                                     fix_padding(new_slots);
                                     issue_step(e.addi(t, D), (tm + D) % U);
                                     std::vector<V> dyf = read_dy(tm % NY);
                                     for(int r = 0; r < KH; ++r)
                                     {
                                         int slot = (S * tm + r) % NX;
                                         for(int nj = 0; nj < NA; ++nj)
                                         {
                                             V xf = read_x(slot, nj);
                                             for(int mi = 0; mi < WK16; ++mi)
                                             {
                                                 int i = acc_index(r, mi, nj);
                                                 la[i] = e.mfma16x16x32(bf16, dyf[mi], xf, la[i]);
                                             }
                                         }
                                     }
                                 }
                                 return la;
                             });
            rocke_b_s_waitcnt(b, 0, 0, -1);
            rocke_b_s_barrier_bare(b);
            return accs_out;
        });
    if((int)accs.size() != n_acc)
    {
        return NULL;
    }

    /* ---- epilogue: partial slot ws[part][(r, s, c)][k] ----
     * acc lane layout: k = (lane / 16) * 4 + slot of atom mi, n = lane % 16 of atom nj. */
    V k_lane0 = e.add(k_blk0, wave_k0);
    k_lane0 = e.add(k_lane0, e.muli(lane_hi, 4));
    for(int r = 0; r < KH; ++r)
    {
        for(int nj = 0; nj < NA; ++nj)
        {
            V n = e.addi(lane_n, nj * kAtom);
            V n_ok = e.lti(n, KW * C);
            V kt = e.ci(r * KW * C);
            V tap = e.add(kt, n); /* (r, s, c) flattened */
            V row_base = e.add(e.muli(part, TAPS), tap);
            row_base = e.mul(row_base, p_K);
            for(int mi = 0; mi < WK16; ++mi)
            {
                for(int slot = 0; slot < 4; ++slot)
                {
                    V kk = e.addi(k_lane0, mi * kAtom + slot);
                    V ok = e.land(n_ok, e.lt(kk, p_K));
                    V val = rocke_b_vec_extract(b, accs[acc_index(r, mi, nj)], slot);
                    e.if_(ok, [&]() { rocke_b_global_store(b, P.D, e.add(row_base, kk), val, 4); });
                }
            }
        }
    }
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_direct_wgrad_gather(
    rocke_ir_builder_t* b, const rocke_direct_wgrad_gather_spec_t* spec, const char* arch)
{
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    return build(b, spec, arch ? arch : "gfx950");
}

extern "C" rocke_kernel_def_t* rocke_build_direct_wgrad_gather_new(
    rocke_ir_builder_t* b, const rocke_direct_wgrad_gather_spec_t* spec, const char* arch)
{
    return build_new(
        b,
        spec != NULL,
        "build_wgrad_gather",
        [&](char* out, size_t cap) {
            return rocke_direct_wgrad_gather_kernel_name(spec, out, cap);
        },
        [&]() { return rocke_build_direct_wgrad_gather(b, spec, arch); });
}

extern "C" rocke_status_t
    rocke_direct_wgrad_gather_lower_to_llvm(const rocke_direct_wgrad_gather_spec_t* spec,
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
        "build_wgrad_gather failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_direct_wgrad_gather_new(b, spec, arch); });
}
