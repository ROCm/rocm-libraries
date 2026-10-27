// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_wgrad_tiled.cpp -- C++ engine port of build_direct_wgrad_tiled
 * (library/kernels/common/conv_direct_wgrad_tiled.py): the stride-1 direct
 * wgrad with multi-atom wave tiles and an LDS-DMA row pipeline.
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the contract in
 * rocke/instance_conv_direct_wgrad_internal.hpp. The functions below follow
 * the Python closures one to one (seg_lane_offsets, issue_row, frag_index,
 * step, segment).
 */
#include "rocke/instance_conv_direct_wgrad_tiled.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "rocke/arch_target.h" /* rocke_set_unknown_arch_reason */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */
#include "rocke/instance_conv_direct_wgrad_internal.hpp"

using namespace rocke_dwg;

namespace
{

constexpr int kAtom = 16;
constexpr int kWo = 32;
constexpr int kVec = 8;
/* Bank model used to pick the swizzle: 64 four-byte banks, a 32-lane half of a
 * ds_read_b64 serviced per pass. */
constexpr int kBanks = 64;
constexpr int kHalf = 32;

int wave_k(const rocke_direct_wgrad_tiled_spec_t* s)
{
    return kAtom * s->wave_k16;
}

int wave_c(const rocke_direct_wgrad_tiled_spec_t* s)
{
    return kAtom * s->wave_c16;
}

int n_waves(const rocke_direct_wgrad_tiled_spec_t* s)
{
    return s->waves_k * s->waves_c;
}

int strip_w(const rocke_direct_wgrad_tiled_spec_t* s)
{
    return kWo / s->unfold_n;
}

/* DirectWgradTiledSpec.dma_instrs(): 1 KiB DMA instructions per wave and row. */
int dma_instrs_dy(const rocke_direct_wgrad_tiled_spec_t* s)
{
    return cdiv(cdiv(kWo * wave_k(s) * s->waves_k / kVec, kWave), n_waves(s));
}

int dma_instrs_x(const rocke_direct_wgrad_tiled_spec_t* s)
{
    int seg = strip_w(s) + s->problem.KW - 1;
    return cdiv(cdiv(s->unfold_n * seg * wave_c(s) * s->waves_c / kVec, kWave), n_waves(s));
}

/* ---- pick_swizzle ------------------------------------------------------ */

/* _frag_row: LDS row a ds_read_b64_tr_b16 lane reads for reduction block rd. */
int frag_row(int lane, int rd, int shift, int wu, int seg)
{
    int kk0 = (lane / 16) * 8;
    return (kk0 / wu) * seg + kk0 % wu + rd * 4 + (lane / 4) % 4 + shift;
}

int swz_of(const std::vector<int>& bits, int row)
{
    int v = 0;
    for(size_t j = 0; j < bits.size(); ++j)
    {
        v |= ((row >> bits[j]) & 1) << j;
    }
    return 2 * v;
}

/* _swizzle_ok: no two lanes of a 32-lane half hit one bank, for every read. */
bool swizzle_ok(const std::vector<int>& bits,
                int chunks,
                const int* shifts,
                int n_shifts,
                int wu,
                int seg,
                const int* origins,
                int n_origins)
{
    for(int si = 0; si < n_shifts; ++si)
    {
        for(int oi = 0; oi < n_origins; ++oi)
        {
            for(int rd = 0; rd < 2; ++rd)
            {
                for(int half = 0; half < 2; ++half)
                {
                    std::set<int> banks;
                    for(int lane = half * kHalf; lane < (half + 1) * kHalf; ++lane)
                    {
                        int row = frag_row(lane, rd, shifts[si], wu, seg);
                        int col = origins[oi] + (lane % 4) * 4;
                        int phys = (col / kVec) ^ swz_of(bits, row);
                        int byte = (row * chunks + phys) * 16 + (col % kVec) * 2;
                        for(int w = 0; w < 2; ++w)
                        {
                            banks.insert((byte / 4 + w) % kBanks);
                        }
                    }
                    if((int)banks.size() != 2 * kHalf)
                    {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

/* itertools.permutations(range(6), nbits) in its (lexicographic) order;
 * stops at the first selection `ok` accepts. */
template <class Ok>
bool first_permutation(int nbits, std::vector<int>& cur, std::vector<bool>& used, Ok&& ok)
{
    if((int)cur.size() == nbits)
    {
        return ok(cur);
    }
    for(int v = 0; v < 6; ++v)
    {
        if(used[v])
        {
            continue;
        }
        used[v] = true;
        cur.push_back(v);
        if(first_permutation(nbits, cur, used, ok))
        {
            return true;
        }
        cur.pop_back();
        used[v] = false;
    }
    return false;
}

std::vector<int> pick_swizzle(
    int chunks, const std::vector<int>& shifts, int wu, int seg, const std::vector<int>& origins)
{
    int out[6];
    int n = rocke_direct_wgrad_pick_swizzle(chunks,
                                            shifts.data(),
                                            (int)shifts.size(),
                                            wu,
                                            seg,
                                            origins.data(),
                                            (int)origins.size(),
                                            out);
    return std::vector<int>(out, out + n);
}

} // namespace

extern "C" int rocke_direct_wgrad_pick_swizzle(int chunks,
                                               const int* shifts,
                                               int n_shifts,
                                               int wu,
                                               int seg,
                                               const int* col_origins,
                                               int n_origins,
                                               int bits_out[6])
{
    auto ok = [&](const std::vector<int>& bits) {
        return swizzle_ok(bits, chunks, shifts, n_shifts, wu, seg, col_origins, n_origins);
    };
    std::vector<int> none;
    if(ok(none))
    {
        return 0;
    }
    int n_pairs = chunks / 2;
    int width = 0;
    if(n_pairs > 1)
    {
        /* int(math.log2(n_pairs)) */
        while((2 << width) <= n_pairs)
        {
            ++width;
        }
    }
    for(int nbits = 1; nbits <= width; ++nbits)
    {
        std::vector<int> cur;
        std::vector<bool> used(6, false);
        if(first_permutation(nbits, cur, used, ok))
        {
            for(int j = 0; j < nbits; ++j)
            {
                bits_out[j] = cur[j];
            }
            return nbits;
        }
    }
    return 0;
}

/* ===================================================================== *
 *  Spec defaults / derived geometry / kernel_name / is_valid
 * ===================================================================== */

extern "C" rocke_direct_wgrad_tiled_spec_t rocke_direct_wgrad_tiled_spec_default(void)
{
    rocke_direct_wgrad_tiled_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "direct_wgrad_tiled";
    s.wave_k16 = 2;
    s.wave_c16 = 2;
    s.waves_k = 4;
    s.waves_c = 2;
    s.rows_per_seg = 30;
    s.prefetch = 2;
    s.target_blocks = 256;
    s.row_unroll = 0;
    s.unfold_n = 1;
    s.q_groups = 1;
    s.pingpong = false;
    return s;
}

extern "C" int rocke_direct_wgrad_tiled_threads_per_block(const rocke_direct_wgrad_tiled_spec_t* s)
{
    return n_waves(s) * s->q_groups * kWave;
}

extern "C" int rocke_direct_wgrad_tiled_unroll(const rocke_direct_wgrad_tiled_spec_t* s)
{
    if(s->row_unroll)
    {
        return s->row_unroll;
    }
    return std::lcm(s->prefetch + 1, s->problem.KH);
}

extern "C" int rocke_direct_wgrad_tiled_acc_vgprs(const rocke_direct_wgrad_tiled_spec_t* s)
{
    return s->problem.KH * s->problem.KW * s->wave_k16 * s->wave_c16 * 4;
}

extern "C" long rocke_direct_wgrad_tiled_lds_bytes(const rocke_direct_wgrad_tiled_spec_t* s)
{
    /* The padded dY / X ring buffers, plus the q-group reduction buffers. */
    long slot
        = (long)s->q_groups * n_waves(s) * (dma_instrs_dy(s) + dma_instrs_x(s)) * kWave * kVec * 2;
    long red = 0;
    if(s->q_groups > 1)
    {
        red = 2L * n_waves(s) * s->wave_k16 * s->wave_c16 * kWave * 4 * 4;
    }
    return (s->prefetch + 1) * slot + red;
}

extern "C" int rocke_direct_wgrad_tiled_n_channel_tiles(const rocke_direct_wgrad_tiled_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    int block_k = wave_k(s) * s->waves_k;
    int block_c = wave_c(s) * s->waves_c;
    return p->groups * cdiv(p->kpg, block_k) * cdiv(p->cpg, block_c);
}

extern "C" int rocke_direct_wgrad_tiled_n_segments(const rocke_direct_wgrad_tiled_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return cdiv(p->N, s->unfold_n) * cdiv(problem_wo(p), strip_w(s)) * cdiv(p->H, s->rows_per_seg);
}

extern "C" int rocke_direct_wgrad_tiled_n_splits(const rocke_direct_wgrad_tiled_spec_t* s)
{
    int a = cdiv(rocke_direct_wgrad_tiled_n_segments(s), s->q_groups);
    int c = cdiv(s->target_blocks, rocke_direct_wgrad_tiled_n_channel_tiles(s));
    return a < c ? a : c;
}

extern "C" void rocke_direct_wgrad_tiled_launch_geometry(const rocke_direct_wgrad_tiled_spec_t* s,
                                                         int grid[3],
                                                         int block[3])
{
    grid[0] = rocke_direct_wgrad_tiled_n_channel_tiles(s) * rocke_direct_wgrad_tiled_n_splits(s);
    grid[1] = 1;
    grid[2] = 1;
    block[0] = rocke_direct_wgrad_tiled_threads_per_block(s);
    block[1] = 1;
    block[2] = 1;
}

extern "C" rocke_status_t rocke_direct_wgrad_tiled_kernel_name(
    const rocke_direct_wgrad_tiled_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    char f_buf[48], wt_buf[32], w_buf[32], rps_buf[24], pf_buf[24], tb_buf[24];
    char u_buf[24], un_buf[24], qg_buf[24];
    snprintf(f_buf, sizeof f_buf, "f%dx%dp%d", p->KH, p->KW, p->PAD);
    snprintf(wt_buf, sizeof wt_buf, "wt%dx%d", wave_k(s), wave_c(s));
    snprintf(w_buf, sizeof w_buf, "w%dx%d", s->waves_k, s->waves_c);
    snprintf(rps_buf, sizeof rps_buf, "rps%d", s->rows_per_seg);
    snprintf(pf_buf, sizeof pf_buf, "pf%d", s->prefetch);
    snprintf(tb_buf, sizeof tb_buf, "tb%d", s->target_blocks);
    snprintf(u_buf, sizeof u_buf, "u%d", s->row_unroll);
    snprintf(un_buf, sizeof un_buf, "un%d", s->unfold_n);
    snprintf(qg_buf, sizeof qg_buf, "qg%d", s->q_groups);
    const char* parts[6] = {f_buf, wt_buf, w_buf, rps_buf, pf_buf, tb_buf};
    const char* flag_names[5] = {"bf16", u_buf, un_buf, qg_buf, "pp"};
    int flag_on[5] = {problem_is_bf16(p) ? 1 : 0,
                      s->row_unroll ? 1 : 0,
                      s->unfold_n > 1 ? 1 : 0,
                      s->q_groups > 1 ? 1 : 0,
                      s->pingpong ? 1 : 0};
    return rocke_kernel_name_join(s->name, parts, 6, flag_names, flag_on, 5, out, out_cap, NULL);
}

extern "C" bool rocke_direct_wgrad_tiled_is_valid_spec(const rocke_direct_wgrad_tiled_spec_t* s,
                                                       const char* arch,
                                                       char* reason,
                                                       size_t reason_cap)
{
    char msg[ROCKE_ERR_MSG_CAP];
#define TILED_REJECT(...)                                  \
    do                                                     \
    {                                                      \
        ROCKE_ERR_SNPRINTF(msg, sizeof(msg), __VA_ARGS__); \
        set_reason(reason, reason_cap, msg);               \
        return false;                                      \
    } while(0)

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
    if(!problem_dtype_ok(p))
    {
        if(p->dtype == NULL)
        {
            TILED_REJECT("unsupported dtype None");
        }
        TILED_REJECT("unsupported dtype '%s'", p->dtype);
    }
    if(p->stride != 1)
    {
        TILED_REJECT("stride-1 algorithm (got stride=%d)", p->stride);
    }
    if(target->wave_size != 64 || !target->memory.has_ds_read_tr)
    {
        TILED_REJECT("needs wave64 + ds_read_b64_tr_b16 (absent on %s)", arch);
    }
    if(rocke_archtarget_async_lds_max_dwords(target) < 4)
    {
        TILED_REJECT("needs 16-byte buffer_load_lds (absent on %s)", arch);
    }
    const char* ab = problem_is_bf16(p) ? "bf16" : "f16";
    if(!rocke_mma_catalog_has_shape(
           rocke_archtarget_mma(target), "mma", ab, ab, "fp32", 16, 16, 32))
    {
        TILED_REJECT("missing mfma_f32_16x16x32_%s on %s", ab, arch);
    }
    auto in124 = [](int v) { return v == 1 || v == 2 || v == 4; };
    if(!in124(s->wave_k16) || !in124(s->wave_c16))
    {
        TILED_REJECT("wave_k16 / wave_c16 must be 1, 2 or 4");
    }
    if(s->waves_k < 1 || (s->waves_k & (s->waves_k - 1)))
    {
        TILED_REJECT("waves_k must be a power of two");
    }
    if(s->waves_c < 1 || (s->waves_c & (s->waves_c - 1)))
    {
        TILED_REJECT("waves_c must be a power of two");
    }
    if(!in124(s->unfold_n))
    {
        TILED_REJECT("unfold_n must be 1, 2 or 4");
    }
    if(s->q_groups != 1 && s->q_groups != 2)
    {
        TILED_REJECT("q_groups must be 1 or 2");
    }
    if(s->pingpong && s->q_groups != 2)
    {
        TILED_REJECT("pingpong needs q_groups == 2");
    }
    if(rocke_direct_wgrad_tiled_threads_per_block(s)
       > rocke_archtarget_max_threads_per_block(target))
    {
        TILED_REJECT("too many threads per block");
    }
    if(p->cpg % kVec || p->kpg % kVec)
    {
        TILED_REJECT("cpg and kpg must be multiples of 8 (16-byte chunks)");
    }
    if(p->kpg < wave_k(s) || p->cpg < wave_c(s))
    {
        TILED_REJECT("wave tile wider than the group's channels");
    }
    if(rocke_direct_wgrad_tiled_acc_vgprs(s) > 192)
    {
        TILED_REJECT("accumulators need %d VGPRs (> 192)", rocke_direct_wgrad_tiled_acc_vgprs(s));
    }
    if(s->row_unroll < 0)
    {
        TILED_REJECT("row_unroll must be >= 0");
    }
    if(s->target_blocks < 1)
    {
        TILED_REJECT("target_blocks must be >= 1");
    }
    /* (Python divides by prefetch + 1 here; a negative prefetch, which raises
     * there, falls through to the prefetch check.) */
    if(s->prefetch >= 0 && s->row_unroll % (s->prefetch + 1))
    {
        TILED_REJECT("row_unroll must be a multiple of prefetch + 1");
    }
    if(s->prefetch < 1)
    {
        TILED_REJECT("prefetch must be >= 1");
    }
    {
        int u = rocke_direct_wgrad_tiled_unroll(s);
        if(s->rows_per_seg <= 0 || u <= 0 || s->rows_per_seg % u)
        {
            TILED_REJECT("rows_per_seg must be a positive multiple of %d", u);
        }
    }
    if(p->KH > 8 || p->KW > 8)
    {
        TILED_REJECT("filter extent above 8");
    }
    if(rocke_direct_wgrad_tiled_lds_bytes(s) > 160 * 1024)
    {
        TILED_REJECT("LDS %ld B > 160 KiB", rocke_direct_wgrad_tiled_lds_bytes(s));
    }
#undef TILED_REJECT
    set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  Builder
 * ===================================================================== */
namespace
{

struct DmaLane
{
    V img, col, ch, okr;
};

struct LaneOff
{
    V off, ok;
};

struct Ctx
{
    Em e;
    const rocke_direct_wgrad_tiled_spec_t* spec;
    rocke_direct_conv_problem_t p;
    const rocke_type_t* io_type;
    bool bf16;

    int KH, KW, PAD, NW, WK16, WC16, BK, BC, CHK, CHC, UN, WU, SEG, STRIP, D, NB, HPB, U, T_PRO;
    int IK, IC, N_INSTR, QG, GDY_ELEMS, GX_ELEMS;
    std::vector<int> swz_k, swz_c;

    rocke_dconv_params_t params;
    V c0, c1, c_two, oob;
    V c_cpg, c_kpg;
    V lane, qg, wave;
    V k_blk0, c_glb0, c_blk0, group;
    V a_rsrc, b_rsrc;
    std::vector<V> dy_bufs, x_bufs, dy_addr, x_addr;
    V qg_dy_bytes = NULL, qg_x_bytes = NULL;
    std::vector<DmaLane> dy_lanes, x_lanes;
    std::vector<std::vector<V>> dy_idx; /* [mi][rd] */
    std::vector<std::vector<std::vector<V>>> x_idx; /* [s][ni][rd] */
    V zero_frag;
    int n_acc;
};

int acc_index(const Ctx& c, int r, int s, int mi, int ni)
{
    return ((r * c.KW + s) * c.WK16 + mi) * c.WC16 + ni;
}

std::vector<DmaLane>
    dma_lanes(Ctx& c, int n_instr, int chunks, int rows, int per_img, const std::vector<int>& bits)
{
    const Em& e = c.e;
    std::vector<DmaLane> out;
    for(int i = 0; i < n_instr; ++i)
    {
        V slot = e.muli(c.wave, n_instr);
        slot = e.addi(slot, i);
        slot = e.muli(slot, kWave);
        slot = e.add(slot, c.lane);
        V row = e.divi(slot, chunks);
        V phys = e.modi(slot, chunks);
        V sw = e.swz_val(row, bits, c.c1);
        V log = (sw == NULL) ? phys : e.bxor(phys, sw);
        DmaLane d;
        d.img = c.UN > 1 ? e.divi(row, per_img) : c.c0;
        d.col = c.UN > 1 ? e.modi(row, per_img) : row;
        d.ch = e.muli(log, kVec);
        d.okr = e.lti(row, rows);
        out.push_back(d);
    }
    return out;
}

/* seg_lane_offsets(n0, q0) */
void seg_lane_offsets(Ctx& c, V n0, V q0, std::vector<LaneOff>& dy, std::vector<LaneOff>& xs)
{
    const Em& e = c.e;
    const rocke_dconv_params_t& P = c.params;
    for(const DmaLane& d : c.dy_lanes)
    {
        V n_i = e.add(n0, d.img);
        V wo = e.add(q0, d.col);
        V a = e.lt(wo, P.p_Wo);
        V bq = e.lt(n_i, P.p_N);
        V ok = e.land(d.okr, e.land(a, bq));
        V m1 = e.mul(n_i, P.p_A_stride_n);
        V m2 = e.mul(wo, P.p_A_stride_wi);
        V s = e.add(m1, m2);
        V t = e.add(c.k_blk0, d.ch);
        dy.push_back({e.add(s, t), ok});
    }
    for(const DmaLane& d : c.x_lanes)
    {
        V n_i = e.add(n0, d.img);
        V wi = e.add(q0, e.subi(d.col, c.PAD));
        V a = e.ge(wi, c.c0);
        V bq = e.lt(wi, P.p_Wi);
        V ab = e.land(a, bq);
        V cn = e.lt(n_i, P.p_N);
        V ok = e.land(d.okr, e.land(ab, cn));
        V m1 = e.mul(n_i, P.p_B_stride_n);
        V m2 = e.mul(wi, P.p_B_stride_wi);
        V s = e.add(m1, m2);
        V t = e.add(c.c_glb0, d.ch);
        xs.push_back({e.add(s, t), ok});
    }
}

struct Seg
{
    V h0, h_end, t_end;
    std::vector<LaneOff> dy, xs;
};

/* issue_row(buf, t, h0, h_end, lane_offs, t_end) */
void issue_row(Ctx& c, const Seg& sg, int buf, V t)
{
    const Em& e = c.e;
    const rocke_dconv_params_t& P = c.params;
    V hi = e.add(sg.h0, e.subi(t, c.T_PRO));
    V ho = e.addi(hi, c.PAD);
    V live = e.lt(t, sg.t_end);
    V a = e.ge(ho, c.c0);
    V bq = e.lt(ho, P.p_Ho);
    V dy_ok = e.land(live, e.land(a, bq));
    V x0 = e.gei(t, c.T_PRO);
    V x1 = e.ge(hi, c.c0);
    V x2 = e.lt(hi, sg.h_end);
    V x12 = e.land(x1, x2);
    V x_ok = e.land(live, e.land(x0, x12));
    V dy_row = e.mul(ho, P.p_A_stride_hi);
    V x_row = e.mul(hi, P.p_B_stride_hi);
    for(int i = 0; i < (int)sg.dy.size(); ++i)
    {
        V okk = e.land(sg.dy[i].ok, dy_ok);
        V voff = e.mul(e.add(sg.dy[i].off, dy_row), c.c_two);
        voff = e.select(okk, voff, c.oob);
        V base
            = c.QG == 1 ? c.dy_addr[buf] : rocke_b_smem_ptr_add(e.b, c.dy_addr[buf], c.qg_dy_bytes);
        V o = e.muli(c.wave, c.IK);
        o = e.addi(o, i);
        o = e.muli(o, kWave * 16);
        o = e.zext64(o);
        V lds = rocke_b_smem_ptr_add(e.b, base, o);
        rocke_b_async_buffer_load_lds(e.b, c.a_rsrc, lds, voff, c.c0, 4, 0);
    }
    for(int i = 0; i < (int)sg.xs.size(); ++i)
    {
        V okk = e.land(sg.xs[i].ok, x_ok);
        V voff = e.mul(e.add(sg.xs[i].off, x_row), c.c_two);
        voff = e.select(okk, voff, c.oob);
        V base = c.QG == 1 ? c.x_addr[buf] : rocke_b_smem_ptr_add(e.b, c.x_addr[buf], c.qg_x_bytes);
        V o = e.muli(c.wave, c.IC);
        o = e.addi(o, i);
        o = e.muli(o, kWave * 16);
        o = e.zext64(o);
        V lds = rocke_b_smem_ptr_add(e.b, base, o);
        rocke_b_async_buffer_load_lds(e.b, c.b_rsrc, lds, voff, c.c0, 4, 0);
    }
}

V read_frag(Ctx& c, V smem, const std::vector<V>& idx)
{
    V lo = c.e.ds_read_tr(smem, c.c0, idx[0], c.io_type);
    V hi = c.e.ds_read_tr(smem, c.c0, idx[1], c.io_type);
    return rocke_b_vec_concat(c.e.b, lo, hi);
}

typedef std::vector<std::vector<V>> Ring; /* [KH][WK16] */

/* step(buf, t, ring, accs, compute, issue) */
void step(Ctx& c, const Seg& sg, int buf, V t, Ring& ring, std::vector<V>& accs, bool compute)
{
    const Em& e = c.e;
    rocke_b_s_waitcnt(e.b, (c.D - 1) * c.N_INSTR, 0, -1);
    rocke_b_s_barrier_bare(e.b);
    issue_row(c, sg, (buf + c.D) % c.NB, e.addi(t, c.D));
    std::vector<V> newest;
    for(int mi = 0; mi < c.WK16; ++mi)
    {
        newest.push_back(read_frag(c, c.dy_bufs[buf], c.dy_idx[mi]));
    }
    ring.erase(ring.begin());
    ring.push_back(newest);
    if(!compute)
    {
        return;
    }
    std::vector<std::vector<V>> xf(c.KW);
    for(int s = 0; s < c.KW; ++s)
    {
        for(int ni = 0; ni < c.WC16; ++ni)
        {
            xf[s].push_back(read_frag(c, c.x_bufs[buf], c.x_idx[s][ni]));
        }
    }
    if(c.spec->pingpong)
    {
        rocke_b_s_waitcnt(e.b, -1, 0, -1);
        rocke_b_s_barrier_bare(e.b);
        rocke_b_s_setprio(e.b, 1);
    }
    for(int r = 0; r < c.KH; ++r)
    {
        const std::vector<V>& dyf = ring[c.KH - 1 - r];
        for(int s = 0; s < c.KW; ++s)
        {
            for(int mi = 0; mi < c.WK16; ++mi)
            {
                for(int ni = 0; ni < c.WC16; ++ni)
                {
                    int i = acc_index(c, r, s, mi, ni);
                    accs[i] = e.mfma16x16x32(c.bf16, dyf[mi], xf[s][ni], accs[i]);
                }
            }
        }
    }
    if(c.spec->pingpong)
    {
        rocke_b_s_setprio(e.b, 0);
    }
}

/* segment(accs, n0, q, h0, h_end, valid=None) */
std::vector<V> segment(Ctx& c, std::vector<V> accs, V n0, V q, V h0, V h_end, V valid)
{
    const Em& e = c.e;
    Seg sg;
    sg.h0 = h0;
    sg.h_end = h_end;
    seg_lane_offsets(c, n0, e.muli(q, c.WU), sg.dy, sg.xs);
    sg.t_end = e.ci(c.T_PRO + c.HPB);
    if(valid != NULL)
    {
        sg.t_end = e.select(valid, sg.t_end, c.c0);
    }
    for(int t = 0; t < c.D; ++t)
    {
        issue_row(c, sg, t % c.NB, e.ci(t));
    }
    Ring ring(c.KH, std::vector<V>(c.WK16, c.zero_frag));
    for(int t = 0; t < c.T_PRO; ++t)
    {
        step(c, sg, t % c.NB, e.ci(t), ring, accs, false);
    }
    std::vector<std::string> names = iter_names("a", accs.size());
    std::vector<V> inits = accs;
    for(int j = 1; j < c.KH; ++j)
    {
        for(int mi = 0; mi < c.WK16; ++mi)
        {
            names.push_back("r" + std::to_string(j) + "_" + std::to_string(mi));
            inits.push_back(ring[j][mi]);
        }
    }
    V hi_bound = e.ci(c.HPB / c.U);
    std::vector<V> res
        = e.for_iter(c.c0, hi_bound, c.c1, names, inits, "rb", [&](V rb, std::vector<V> rc) {
              std::vector<V> accs_i(rc.begin(), rc.begin() + c.n_acc);
              Ring ring_i;
              ring_i.push_back(std::vector<V>(c.WK16, c.zero_frag));
              for(int j = 1; j < c.KH; ++j)
              {
                  ring_i.push_back(std::vector<V>(rc.begin() + c.n_acc + (j - 1) * c.WK16,
                                                  rc.begin() + c.n_acc + j * c.WK16));
              }
              V t_base = e.addi(e.muli(rb, c.U), c.T_PRO);
              for(int u = 0; u < c.U; ++u)
              {
                  step(c, sg, (c.T_PRO + u) % c.NB, e.addi(t_base, u), ring_i, accs_i, true);
              }
              std::vector<V> out = accs_i;
              for(int j = 1; j < c.KH; ++j)
              {
                  out.insert(out.end(), ring_i[j].begin(), ring_i[j].end());
              }
              return out;
          });
    std::vector<V> accs_out(res.begin(), res.begin() + std::min((size_t)c.n_acc, res.size()));
    rocke_b_s_waitcnt(e.b, 0, 0, -1);
    rocke_b_s_barrier_bare(e.b);
    return accs_out;
}

/* frag_index(shift, rd, col0, chunks, bits, seg, g_elems) */
struct FragLanes
{
    V lane_q, lane_c4, img_l, jl;
};

V frag_index(Ctx& c,
             const FragLanes& fl,
             int shift,
             int rd,
             V col0,
             int chunks,
             const std::vector<int>& bits,
             int seg,
             int g_elems)
{
    const Em& e = c.e;
    V row = e.muli(fl.img_l, seg);
    row = e.add(row, e.add(fl.jl, fl.lane_q));
    row = e.addi(row, shift + rd * 4);
    V col = e.add(col0, e.muli(fl.lane_c4, 4));
    V chunk = e.divi(col, kVec);
    V sw = e.swz_val(row, bits, c.c1);
    V phys = (sw == NULL) ? chunk : e.bxor(chunk, sw);
    V idx = e.muli(row, chunks);
    idx = e.add(idx, phys);
    idx = e.muli(idx, kVec);
    idx = e.add(idx, e.modi(col, kVec));
    if(c.QG > 1)
    {
        idx = e.add(idx, e.muli(c.qg, g_elems));
    }
    return idx;
}

rocke_kernel_def_t*
    build(rocke_ir_builder_t* b, const rocke_direct_wgrad_tiled_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_direct_wgrad_tiled_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string prefix = std::string("invalid DirectWgradTiledSpec for ") + arch;
        fail_invalid(b, prefix.c_str(), reason);
        return NULL;
    }
    Ctx c;
    c.e.b = b;
    const Em& e = c.e;
    c.spec = spec;
    c.p = spec->problem;
    const rocke_direct_conv_problem_t& p = c.p;
    c.bf16 = problem_is_bf16(&p);
    c.io_type = c.bf16 ? rocke_bf16() : rocke_f16();
    c.KH = p.KH;
    c.KW = p.KW;
    c.PAD = p.PAD;
    c.NW = n_waves(spec);
    c.WK16 = spec->wave_k16;
    c.WC16 = spec->wave_c16;
    c.BK = wave_k(spec) * spec->waves_k;
    c.BC = wave_c(spec) * spec->waves_c;
    c.CHK = c.BK / kVec;
    c.CHC = c.BC / kVec;
    c.UN = spec->unfold_n;
    c.WU = strip_w(spec);
    c.SEG = c.WU + c.KW - 1;
    c.STRIP = c.UN * c.SEG;
    c.D = spec->prefetch;
    c.NB = spec->prefetch + 1;
    c.HPB = spec->rows_per_seg;
    c.U = rocke_direct_wgrad_tiled_unroll(spec);
    c.T_PRO = c.KH - 1;
    c.IK = dma_instrs_dy(spec);
    c.IC = dma_instrs_x(spec);
    c.N_INSTR = c.IK + c.IC;
    c.QG = spec->q_groups;
    c.GDY_ELEMS = c.NW * c.IK * kWave * kVec;
    c.GX_ELEMS = c.NW * c.IC * kWave * kVec;
    int DY_ELEMS = c.QG * c.GDY_ELEMS;
    int X_ELEMS = c.QG * c.GX_ELEMS;
    c.n_acc = c.KH * c.KW * c.WK16 * c.WC16;

    std::vector<int> k_origins, c_origins;
    for(int wk = 0; wk < spec->waves_k; ++wk)
    {
        for(int mi = 0; mi < c.WK16; ++mi)
        {
            k_origins.push_back(wk * wave_k(spec) + mi * kAtom);
        }
    }
    for(int wc = 0; wc < spec->waves_c; ++wc)
    {
        for(int ni = 0; ni < c.WC16; ++ni)
        {
            c_origins.push_back(wc * wave_c(spec) + ni * kAtom);
        }
    }
    std::vector<int> shifts_c;
    for(int s = 0; s < c.KW; ++s)
    {
        shifts_c.push_back(s);
    }
    c.swz_k = pick_swizzle(c.CHK, std::vector<int>{0}, c.WU, c.WU, k_origins);
    c.swz_c = pick_swizzle(c.CHC, shifts_c, c.WU, c.SEG, c_origins);

    rocke_attr_set_int(b,
                       &b->kernel->attrs,
                       "max_workgroup_size",
                       rocke_direct_wgrad_tiled_threads_per_block(spec));
    /* Separate scopes per row buffer, or the waitcnt pass drains the DMA ring. */
    rocke_attr_set_bool(b, &b->kernel->attrs, "lds_alias_scopes", true);
    /* Two waves per SIMD: the register budget the 32x32 tile is sized for. */
    {
        const int64_t wpe[2] = {2, 8};
        rocke_attr_set_int_list(b, &b->kernel->attrs, "waves_per_eu", wpe, 2);
    }
    rocke_dconv_emit_params(b, &c.params, "wgrad", c.io_type);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    const rocke_dconv_params_t& P = c.params;

    c.c0 = e.ci(0);
    c.c1 = e.ci(1);
    c.c_two = e.ci(2);
    c.oob = e.ci(((int64_t)1 << 31) - 1);

    V p_N = P.p_N;
    V p_H = P.p_Hi;
    V p_Wo = P.p_Wo;
    V p_groups = P.p_groups;
    c.c_cpg = e.div(P.p_total_c, p_groups);
    c.c_kpg = e.div(P.p_total_k, p_groups);

    V tid = rocke_b_thread_id_x(b);
    c.lane = e.modi(tid, kWave);
    V wave_all = rocke_b_readfirstlane(b, e.divi(tid, kWave));
    if(c.QG > 1)
    {
        c.qg = e.divi(wave_all, c.NW);
        c.wave = e.modi(wave_all, c.NW);
    }
    else
    {
        c.qg = c.c0;
        c.wave = wave_all;
    }

    /* ---- grid decode ---- */
    V n_k_tiles = e.cdivi(c.c_kpg, c.BK);
    V n_c_tiles = e.cdivi(c.c_cpg, c.BC);
    V n_ch_tiles = e.mul(e.mul(p_groups, n_k_tiles), n_c_tiles);
    V n_q = e.cdivi(p_Wo, c.WU);
    V n_ng = e.cdivi(p_N, c.UN);
    V n_hb = e.cdivi(p_H, c.HPB);
    V total_segs = e.mul(e.mul(n_ng, n_q), n_hb);
    /* Same formula as DirectWgradTiledSpec.n_splits, which sizes the grid. */
    V sa = e.cdivi(total_segs, spec->q_groups);
    V tb = e.ci(spec->target_blocks);
    V sb = e.add(tb, e.sub(n_ch_tiles, c.c1));
    sb = e.div(sb, n_ch_tiles);
    V n_splits = e.smin(sa, sb);

    /* Renumber block ids so each XCD gets a contiguous run. */
    V lin = rocke_b_block_id_x(b);
    V n_blocks = e.mul(n_ch_tiles, n_splits);
    V xcd = e.modi(lin, 8);
    V pos = e.divi(lin, 8);
    V per = e.divi(n_blocks, 8);
    V rem = e.modi(n_blocks, 8);
    V xp = e.mul(xcd, per);
    V xr = e.smin(xcd, rem);
    lin = e.add(e.add(xp, xr), pos);
    V ch_tile = e.mod(lin, n_ch_tiles);
    V split = e.div(lin, n_ch_tiles);
    V c_tile = e.mod(ch_tile, n_c_tiles);
    V gk = e.div(ch_tile, n_c_tiles);
    V k_tile = e.mod(gk, n_k_tiles);
    c.group = e.div(gk, n_k_tiles);
    {
        V g = e.mul(c.group, c.c_kpg);
        V k = e.muli(k_tile, c.BK);
        c.k_blk0 = e.add(g, k);
    }
    c.c_blk0 = e.muli(c_tile, c.BC);
    c.c_glb0 = e.add(e.mul(c.group, c.c_cpg), c.c_blk0);

    V wave_k_id = e.modi(c.wave, spec->waves_k);
    V wave_c_id = e.divi(c.wave, spec->waves_k);

    c.a_rsrc = rocke_b_buffer_rsrc(b, P.A, P.A_bytes);
    c.b_rsrc = rocke_b_buffer_rsrc(b, P.Bp, P.B_bytes);

    /* ---- LDS: one allocation per ring buffer (one alias scope each) ---- */
    for(int i = 0; i < c.NB; ++i)
    {
        std::string h = "dy" + std::to_string(i);
        c.dy_bufs.push_back(e.smem_alloc_1d(c.io_type, DY_ELEMS, h.c_str()));
    }
    for(int i = 0; i < c.NB; ++i)
    {
        std::string h = "xs" + std::to_string(i);
        c.x_bufs.push_back(e.smem_alloc_1d(c.io_type, X_ELEMS, h.c_str()));
    }
    for(V s : c.dy_bufs)
    {
        c.dy_addr.push_back(rocke_b_smem_addr_of(b, s));
    }
    for(V s : c.x_bufs)
    {
        c.x_addr.push_back(rocke_b_smem_addr_of(b, s));
    }
    if(c.QG > 1)
    {
        c.qg_dy_bytes = e.zext64(e.muli(c.qg, c.GDY_ELEMS * 2));
        c.qg_x_bytes = e.zext64(e.muli(c.qg, c.GX_ELEMS * 2));
    }

    /* ---- DMA lane geometry ---- */
    c.dy_lanes = dma_lanes(c, c.IK, c.CHK, kWo, c.WU, c.swz_k);
    c.x_lanes = dma_lanes(c, c.IC, c.CHC, c.STRIP, c.SEG, c.swz_c);

    /* ---- fragment read addresses ---- */
    V lane_hi = e.divi(c.lane, 16);
    FragLanes fl;
    fl.lane_q = e.modi(e.divi(c.lane, 4), 4);
    fl.lane_c4 = e.modi(c.lane, 4);
    V kk0 = e.muli(lane_hi, 8);
    if(c.UN > 1)
    {
        fl.img_l = e.divi(kk0, c.WU);
        fl.jl = e.modi(kk0, c.WU);
    }
    else
    {
        fl.img_l = c.c0;
        fl.jl = kk0;
    }
    V k_wave0 = e.muli(wave_k_id, wave_k(spec));
    V c_wave0 = e.muli(wave_c_id, wave_c(spec));
    c.dy_idx.resize(c.WK16);
    for(int mi = 0; mi < c.WK16; ++mi)
    {
        for(int rd = 0; rd < 2; ++rd)
        {
            V col0 = e.addi(k_wave0, mi * kAtom);
            c.dy_idx[mi].push_back(
                frag_index(c, fl, 0, rd, col0, c.CHK, c.swz_k, c.WU, c.GDY_ELEMS));
        }
    }
    c.x_idx.assign(c.KW, std::vector<std::vector<V>>(c.WC16));
    for(int s = 0; s < c.KW; ++s)
    {
        for(int ni = 0; ni < c.WC16; ++ni)
        {
            for(int rd = 0; rd < 2; ++rd)
            {
                V col0 = e.addi(c_wave0, ni * kAtom);
                c.x_idx[s][ni].push_back(
                    frag_index(c, fl, s, rd, col0, c.CHC, c.swz_c, c.SEG, c.GX_ELEMS));
            }
        }
    }

    V zero_acc = rocke_b_zero_vec_f32(b, 4);
    c.zero_frag = rocke_b_zero_vec(b, c.io_type, kVec);

    /* ---- persistent segment loop ---- */
    if(spec->pingpong)
    {
        /* Group 1 starts one barrier late. */
        e.if_(e.eq(c.qg, c.c1), [&]() { rocke_b_s_barrier_bare(b); });
    }
    std::vector<std::string> acc_names = iter_names("acc", c.n_acc);
    std::vector<V> zero_accs(c.n_acc, zero_acc);
    std::vector<V> accs;
    if(c.QG == 1)
    {
        accs = e.for_iter(split,
                          total_segs,
                          n_splits,
                          acc_names,
                          zero_accs,
                          "seg",
                          [&](V seg, std::vector<V> carried) {
                              V hb = e.mod(seg, n_hb);
                              V rest = e.div(seg, n_hb);
                              V h0 = e.muli(hb, c.HPB);
                              V n0 = e.muli(e.div(rest, n_q), c.UN);
                              V q = e.mod(rest, n_q);
                              V h_end = e.smin(e.addi(h0, c.HPB), p_H);
                              return segment(c, carried, n0, q, h0, h_end, NULL);
                          });
    }
    else
    {
        /* Group g of block `split` takes virtual split split * QG + g. */
        V n_vsplits = e.muli(n_splits, c.QG);
        V vsplit0 = e.muli(split, c.QG);
        V s1 = e.sub(total_segs, vsplit0);
        V s2 = e.sub(n_vsplits, c.c1);
        V n_iter = e.div(e.add(s1, s2), n_vsplits);
        accs = e.for_iter(
            c.c0, n_iter, c.c1, acc_names, zero_accs, "segi", [&](V segi, std::vector<V> carried) {
                V a = e.add(vsplit0, c.qg);
                V seg = e.add(a, e.mul(segi, n_vsplits));
                V hb = e.mod(seg, n_hb);
                V rest = e.div(seg, n_hb);
                V h0 = e.muli(hb, c.HPB);
                V n0 = e.muli(e.div(rest, n_q), c.UN);
                V q = e.mod(rest, n_q);
                V h_end = e.smin(e.addi(h0, c.HPB), p_H);
                V valid = e.lt(seg, total_segs);
                return segment(c, carried, n0, q, h0, h_end, valid);
            });
    }
    if((int)accs.size() != c.n_acc || !rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    if(spec->pingpong)
    {
        e.if_(e.eq(c.qg, c.c0), [&]() { rocke_b_s_barrier_bare(b); });
    }

    /* ---- epilogue: atomic add into dW[total_k, KH, KW, cpg] ---- */
    V D_ptr = P.D;
    V rs_stride = c.c_cpg;
    V k_stride = e.muli(c.c_cpg, c.KH * c.KW);
    V k_lane0 = e.add(c.k_blk0, k_wave0);
    k_lane0 = e.add(k_lane0, e.muli(lane_hi, 4));
    V k_in_grp0 = e.sub(k_lane0, e.mul(c.group, c.c_kpg));
    V c_lane0 = e.add(c.c_blk0, c_wave0);
    c_lane0 = e.add(c_lane0, e.modi(c.lane, 16));
    if(c.QG > 1)
    {
        /* Tap t is summed and written by group t % 2; the other group hands its
         * accumulators over through LDS, double-buffered on the tap parity. */
        int n_atoms = c.WK16 * c.WC16;
        V red[2];
        for(int i = 0; i < 2; ++i)
        {
            int shape[2] = {1, c.NW * n_atoms * kWave * 4};
            std::string h = "red" + std::to_string(i);
            red[i] = rocke_b_smem_alloc_f32(b, shape, 2, h.c_str());
        }
        auto red_index = [&](int atom) {
            V v = e.muli(c.wave, n_atoms);
            v = e.addi(v, atom);
            v = e.muli(v, kWave);
            v = e.add(v, c.lane);
            return e.muli(v, 4);
        };
        for(int r = 0; r < c.KH; ++r)
        {
            for(int s = 0; s < c.KW; ++s)
            {
                int tap = r * c.KW + s;
                V buf = red[tap % 2];
                V is_recv = e.eqi(c.qg, tap % 2);
                e.if_(e.lnot(is_recv), [&]() {
                    for(int mi = 0; mi < c.WK16; ++mi)
                    {
                        for(int ni = 0; ni < c.WC16; ++ni)
                        {
                            V idx = red_index(mi * c.WC16 + ni);
                            e.smem_store_f32(buf, c.c0, idx, accs[acc_index(c, r, s, mi, ni)], 4);
                        }
                    }
                });
                rocke_b_s_waitcnt(b, -1, 0, -1);
                rocke_b_s_barrier_bare(b);
                e.if_(is_recv, [&]() {
                    for(int ni = 0; ni < c.WC16; ++ni)
                    {
                        V c_in = e.addi(c_lane0, ni * kAtom);
                        V c_ok = e.lt(c_in, c.c_cpg);
                        for(int mi = 0; mi < c.WK16; ++mi)
                        {
                            V ld = e.smem_load_f32(buf, c.c0, red_index(mi * c.WC16 + ni), 4);
                            V tot = rocke_b_fadd(b, accs[acc_index(c, r, s, mi, ni)], ld);
                            for(int slot = 0; slot < 4; ++slot)
                            {
                                V k_in = e.addi(k_in_grp0, mi * kAtom + slot);
                                V k_glb = e.addi(k_lane0, mi * kAtom + slot);
                                V ok = e.land(c_ok, e.lt(k_in, c.c_kpg));
                                V off = e.add(e.mul(k_glb, k_stride), c_in);
                                V tk = e.ci(tap);
                                off = e.add(off, e.mul(tk, rs_stride));
                                e.if_(ok, [&]() {
                                    V val = rocke_b_vec_extract(b, tot, slot);
                                    rocke_b_global_atomic_add(b, D_ptr, off, val, NULL);
                                });
                            }
                        }
                    }
                });
            }
        }
        return rocke_ir_builder_ok(b) ? b->kernel : NULL;
    }
    for(int ni = 0; ni < c.WC16; ++ni)
    {
        V c_in = e.addi(c_lane0, ni * kAtom);
        V c_ok = e.lt(c_in, c.c_cpg);
        for(int mi = 0; mi < c.WK16; ++mi)
        {
            for(int slot = 0; slot < 4; ++slot)
            {
                V k_in = e.addi(k_in_grp0, mi * kAtom + slot);
                V k_glb = e.addi(k_lane0, mi * kAtom + slot);
                V ok = e.land(c_ok, e.lt(k_in, c.c_kpg));
                V base = e.add(e.mul(k_glb, k_stride), c_in);
                for(int r = 0; r < c.KH; ++r)
                {
                    for(int s = 0; s < c.KW; ++s)
                    {
                        V val = rocke_b_vec_extract(b, accs[acc_index(c, r, s, mi, ni)], slot);
                        V tk = e.ci(r * c.KW + s);
                        V off = e.add(base, e.mul(tk, rs_stride));
                        e.if_(ok, [&]() { rocke_b_global_atomic_add(b, D_ptr, off, val, NULL); });
                    }
                }
            }
        }
    }
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_direct_wgrad_tiled(
    rocke_ir_builder_t* b, const rocke_direct_wgrad_tiled_spec_t* spec, const char* arch)
{
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    return build(b, spec, arch ? arch : "gfx950");
}

extern "C" rocke_kernel_def_t* rocke_build_direct_wgrad_tiled_new(
    rocke_ir_builder_t* b, const rocke_direct_wgrad_tiled_spec_t* spec, const char* arch)
{
    return build_new(
        b,
        spec != NULL,
        "build_direct_wgrad_tiled",
        [&](char* out, size_t cap) { return rocke_direct_wgrad_tiled_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_direct_wgrad_tiled(b, spec, arch); });
}

extern "C" rocke_status_t
    rocke_direct_wgrad_tiled_lower_to_llvm(const rocke_direct_wgrad_tiled_spec_t* spec,
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
        "build_direct_wgrad_tiled failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_direct_wgrad_tiled_new(b, spec, arch); });
}
