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
#include <functional>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "rocke/arch_target.h" /* rocke_set_unknown_arch_reason */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_pack_f32_to, rocke_b_io_ir_type */
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

/* DirectWgradTiledSpec.block_k / block_c: gpack spreads the block's waves
 * over whole groups, one group per wave. */
int block_k(const rocke_direct_wgrad_tiled_spec_t* s)
{
    return wave_k(s) * (s->gpack ? n_waves(s) : s->waves_k);
}

int block_c(const rocke_direct_wgrad_tiled_spec_t* s)
{
    return wave_c(s) * (s->gpack ? n_waves(s) : s->waves_c);
}

/* DirectWgradTiledSpec.pingpong_halves: ping-pong between the halves of the
 * block's waves (q_groups 1, or every group's halves with q_groups 4). */
bool pingpong_halves(const rocke_direct_wgrad_tiled_spec_t* s)
{
    return s->pingpong && (s->q_groups == 1 || s->q_groups == 4);
}

/* The ping-pong in halves keeps one more row buffer, unless pp_tight. */
int n_buffers(const rocke_direct_wgrad_tiled_spec_t* s)
{
    if(pingpong_halves(s) && !s->pp_tight)
    {
        return s->prefetch + 2;
    }
    return s->prefetch + 1;
}

int strip_w(const rocke_direct_wgrad_tiled_spec_t* s)
{
    return kWo / s->unfold_n;
}

/* DirectWgradTiledSpec.dma_instrs(): 1 KiB DMA instructions per wave and row. */
int dma_instrs_dy(const rocke_direct_wgrad_tiled_spec_t* s)
{
    return cdiv(cdiv(kWo * block_k(s) / kVec, kWave), n_waves(s));
}

int dma_instrs_x(const rocke_direct_wgrad_tiled_spec_t* s)
{
    int seg = strip_w(s) + s->problem.KW - 1;
    return cdiv(cdiv(s->unfold_n * seg * block_c(s) / kVec, kWave), n_waves(s));
}

/* red_slice_bytes(): one wave group's accumulators of one tap. */
long red_slice_bytes(const rocke_direct_wgrad_tiled_spec_t* s)
{
    return (long)n_waves(s) * s->wave_k16 * s->wave_c16 * kWave * 16;
}

/* red_slices(): two groups hand over the sender's slice, four every group's. */
int red_slices(const rocke_direct_wgrad_tiled_spec_t* s)
{
    return s->q_groups == 2 ? 1 : s->q_groups;
}

/* red_buffers(): the q-group reduction's LDS buffers, two (one barrier per
 * tap) where they fit, else one (two barriers per tap). */
int red_buffers(const rocke_direct_wgrad_tiled_spec_t* s)
{
    if(s->q_groups == 2)
    {
        return 2;
    }
    long rings = (long)n_buffers(s) * s->q_groups * n_waves(s)
                 * (dma_instrs_dy(s) + dma_instrs_x(s)) * kWave * kVec * 2;
    long one = red_slices(s) * red_slice_bytes(s);
    return rings + 2 * one <= 160L * 1024 ? 2 : 1;
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
    s.pp_tight = false;
    s.row_tail = false;
    s.one_split = false;
    s.dw16 = false;
    s.pk16 = false;
    s.gpack = false;
    s.slab = false;
    s.l64 = false;
    s.pp_sched = false;
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
    return std::lcm(n_buffers(s), s->problem.KH);
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
        red = red_buffers(s) * red_slices(s) * red_slice_bytes(s);
    }
    return n_buffers(s) * slot + red;
}

extern "C" int rocke_direct_wgrad_tiled_n_channel_tiles(const rocke_direct_wgrad_tiled_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(s->gpack)
    {
        return cdiv(p->groups, n_waves(s));
    }
    return p->groups * cdiv(p->kpg, block_k(s)) * cdiv(p->cpg, block_c(s));
}

extern "C" int rocke_direct_wgrad_tiled_n_segments(const rocke_direct_wgrad_tiled_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    int n_hb = s->rows_per_seg ? cdiv(p->H, s->rows_per_seg) : 1;
    return cdiv(p->N, s->unfold_n) * cdiv(problem_wo(p), strip_w(s)) * n_hb;
}

extern "C" int rocke_direct_wgrad_tiled_n_splits(const rocke_direct_wgrad_tiled_spec_t* s)
{
    if(s->one_split)
    {
        return 1;
    }
    int a = cdiv(rocke_direct_wgrad_tiled_n_segments(s), s->q_groups);
    int tiles = rocke_direct_wgrad_tiled_n_channel_tiles(s);
    int c = s->rows_per_seg ? cdiv(s->target_blocks, tiles) : s->target_blocks / tiles;
    if(c < 1)
    {
        c = 1;
    }
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
    char f_buf[64], wt_buf[32], w_buf[32], rps_buf[24], pf_buf[24], tb_buf[24];
    char u_buf[24], un_buf[24], qg_buf[24];
    const int pad_w = rocke_direct_conv_problem_pad_w(p);
    if(pad_w != p->PAD)
    {
        snprintf(f_buf, sizeof f_buf, "f%dx%dp%dx%d", p->KH, p->KW, p->PAD, pad_w);
    }
    else
    {
        snprintf(f_buf, sizeof f_buf, "f%dx%dp%d", p->KH, p->KW, p->PAD);
    }
    snprintf(wt_buf, sizeof wt_buf, "wt%dx%d", wave_k(s), wave_c(s));
    snprintf(w_buf, sizeof w_buf, "w%dx%d", s->waves_k, s->waves_c);
    if(s->rows_per_seg)
    {
        snprintf(rps_buf, sizeof rps_buf, "rps%d", s->rows_per_seg);
    }
    else
    {
        snprintf(rps_buf, sizeof rps_buf, "rpsH");
    }
    snprintf(pf_buf, sizeof pf_buf, "pf%d", s->prefetch);
    snprintf(tb_buf, sizeof tb_buf, "tb%d", s->target_blocks);
    snprintf(u_buf, sizeof u_buf, "u%d", s->row_unroll);
    snprintf(un_buf, sizeof un_buf, "un%d", s->unfold_n);
    snprintf(qg_buf, sizeof qg_buf, "qg%d", s->q_groups);
    const char* parts[6] = {f_buf, wt_buf, w_buf, rps_buf, pf_buf, tb_buf};
    const char* flag_names[14] = {"bf16",
                                  u_buf,
                                  un_buf,
                                  qg_buf,
                                  "pp",
                                  "ppt",
                                  "rt",
                                  "os",
                                  "d16",
                                  "pk",
                                  "gp",
                                  "sl",
                                  "l64",
                                  "pps"};
    int flag_on[14] = {problem_is_bf16(p) ? 1 : 0,
                       s->row_unroll ? 1 : 0,
                       s->unfold_n > 1 ? 1 : 0,
                       s->q_groups > 1 ? 1 : 0,
                       s->pingpong ? 1 : 0,
                       s->pp_tight ? 1 : 0,
                       s->row_tail ? 1 : 0,
                       s->one_split ? 1 : 0,
                       s->dw16 ? 1 : 0,
                       s->pk16 ? 1 : 0,
                       s->gpack ? 1 : 0,
                       s->slab ? 1 : 0,
                       s->l64 ? 1 : 0,
                       s->pp_sched ? 1 : 0};
    return rocke_kernel_name_join(s->name, parts, 6, flag_names, flag_on, 14, out, out_cap, NULL);
}

extern "C" bool rocke_direct_wgrad_tiled_is_valid_spec(const rocke_direct_wgrad_tiled_spec_t* s,
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
        arch = "gfx950";
    }
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(rocke_direct_dilation_reason(p, msg, sizeof msg))
    {
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(!problem_dtype_ok(p))
    {
        if(p->dtype == NULL)
        {
            ROCKE_SPEC_FAIL(reason, reason_cap, "unsupported dtype None");
        }
        ROCKE_SPEC_FAIL(reason, reason_cap, "unsupported dtype '%s'", p->dtype);
    }
    if(p->stride != 1)
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "stride-1 algorithm (got stride=%d)", p->stride);
    }
    if(target->wave_size != 64 || !target->memory.has_ds_read_tr)
    {
        ROCKE_SPEC_FAIL(
            reason, reason_cap, "needs wave64 + ds_read_b64_tr_b16 (absent on %s)", arch);
    }
    if(rocke_archtarget_async_lds_max_dwords(target) < 4)
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "needs 16-byte buffer_load_lds (absent on %s)", arch);
    }
    const char* ab = problem_is_bf16(p) ? "bf16" : "f16";
    if(!rocke_mma_catalog_has_shape(
           rocke_archtarget_mma(target), "mma", ab, ab, "fp32", 16, 16, 32))
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "missing mfma_f32_16x16x32_%s on %s", ab, arch);
    }
    auto in124 = [](int v) { return v == 1 || v == 2 || v == 4; };
    if(!in124(s->wave_k16) || !in124(s->wave_c16))
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "wave_k16 / wave_c16 must be 1, 2 or 4");
    }
    if(s->waves_k < 1 || (s->waves_k & (s->waves_k - 1)))
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "waves_k must be a power of two");
    }
    if(s->waves_c < 1 || (s->waves_c & (s->waves_c - 1)))
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "waves_c must be a power of two");
    }
    if(!in124(s->unfold_n) && s->unfold_n != 8 && s->unfold_n != 16 && s->unfold_n != 32)
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "unfold_n must be a power of two up to 32");
    }
    if(s->unfold_n > 4 && p->KW != 1)
    {
        /* A lane's 8 reduction elements must be 8 consecutive X tile rows:
         * with fewer than 8 columns per image that holds only without a
         * column halo (KW == 1). */
        ROCKE_SPEC_FAIL(reason, reason_cap, "unfold_n above 4 needs KW == 1");
    }
    if(s->q_groups != 1 && s->q_groups != 2 && s->q_groups != 4)
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "q_groups must be 1, 2 or 4");
    }
    if(s->pp_sched && !s->pingpong)
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "pp_sched needs the ping-pong");
    }
    if(s->pp_tight && !pingpong_halves(s))
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "pp_tight needs the ping-pong in halves");
    }
    if(pingpong_halves(s) && n_waves(s) % 2)
    {
        ROCKE_SPEC_FAIL(
            reason, reason_cap, "pingpong in halves needs an even wave count per group");
    }
    if(rocke_direct_wgrad_tiled_threads_per_block(s)
       > rocke_archtarget_max_threads_per_block(target))
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "too many threads per block");
    }
    if(p->cpg % kVec || p->kpg % 4)
    {
        /* dY's 16-byte chunks need only 8-byte alignment (kpg % 4): a chunk
         * that runs past kpg reads the next pixel's channels into dW rows the
         * epilogue does not store. */
        ROCKE_SPEC_FAIL(reason, reason_cap, "cpg must be a multiple of 8 and kpg of 4");
    }
    if(s->dw16 && !s->one_split)
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "dw16 stores dW whole: needs one_split");
    }
    if(s->pk16 && !(p->dtype != NULL && strcmp(p->dtype, "fp16") == 0))
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "pk16: bf16 pair atomics lose too much precision");
    }
    if(s->pk16 && s->one_split)
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "pk16 adds atomically: not with one_split");
    }
    if(s->gpack && (p->groups < 2 || p->kpg != wave_k(s) || p->cpg != wave_c(s)))
    {
        ROCKE_SPEC_FAIL(
            reason, reason_cap, "gpack needs groups > 1 with kpg == wave_k and cpg == wave_c");
    }
    if(p->kpg < wave_k(s) || p->cpg < wave_c(s))
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "wave tile wider than the group's channels");
    }
    if(rocke_direct_wgrad_tiled_acc_vgprs(s) > 192)
    {
        ROCKE_SPEC_FAIL(reason,
                        reason_cap,
                        "accumulators need %d VGPRs (> 192)",
                        rocke_direct_wgrad_tiled_acc_vgprs(s));
    }
    if(s->slab && (s->one_split || s->dw16 || s->pk16))
    {
        ROCKE_SPEC_FAIL(
            reason, reason_cap, "slab stores fp32 partials: not with one_split / dw16 / pk16");
    }
    if(s->row_tail && s->rows_per_seg)
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "row_tail needs the auto launch (rows_per_seg 0)");
    }
    if(s->l64)
    {
        /* unfold_n * 2 * max(Ho * Wo * total_k, H * W * total_c) in Python ints. */
        int64_t dy = (int64_t)problem_ho(p) * problem_wo(p) * ((int64_t)p->groups * p->kpg);
        int64_t x = (int64_t)p->H * p->W * ((int64_t)p->groups * p->cpg);
        if((int64_t)s->unfold_n * 2 * (dy > x ? dy : x) >= ((int64_t)1 << 31) - 1)
        {
            ROCKE_SPEC_FAIL(
                reason, reason_cap, "l64: a segment's images exceed the i32 buffer range");
        }
    }
    if(s->row_unroll < 0)
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "row_unroll must be >= 0");
    }
    if(s->target_blocks < 1)
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "target_blocks must be >= 1");
    }
    /* Before the n_buffers check: prefetch -1 (-2 with the single-group
     * ping-pong's extra buffer) makes n_buffers 0. */
    if(s->prefetch < 1)
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "prefetch must be >= 1");
    }
    if(s->row_unroll % n_buffers(s))
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "row_unroll must be a multiple of n_buffers");
    }
    {
        int u = rocke_direct_wgrad_tiled_unroll(s);
        if(s->rows_per_seg < 0 || u <= 0 || s->rows_per_seg % u)
        {
            ROCKE_SPEC_FAIL(
                reason, reason_cap, "rows_per_seg must be 0 or a positive multiple of %d", u);
        }
    }
    if(p->KH > 8 || p->KW > 8)
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "filter extent above 8");
    }
    if(rocke_direct_wgrad_tiled_lds_bytes(s) > 160 * 1024)
    {
        ROCKE_SPEC_FAIL(
            reason, reason_cap, "LDS %ld B > 160 KiB", rocke_direct_wgrad_tiled_lds_bytes(s));
    }
    rocke_spec_set_reason(reason, reason_cap, "ok");
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

    int KH, KW, PAD, PAD_W, NW, WK16, WC16, BK, BC, CHK, CHC, UN, WU, SEG, STRIP, D, NB, HPB, U,
        T_PRO;
    bool AUTO; /* rows_per_seg == 0: whole strips, one-round grid */
    bool GP, L64, PPS; /* gpack, l64, pp_sched */
    int IK, IC, N_INSTR, QG, GDY_ELEMS, GX_ELEMS;
    std::vector<int> swz_k, swz_c;

    rocke_dconv_params_t params;
    V c0, c1, c_two, oob;
    V c_cpg, c_kpg;
    V lane, qg, wave;
    V pp_grp = NULL;
    V k_blk0, c_glb0, c_blk0, group;
    V a_rsrc, b_rsrc;
    std::vector<V> dy_bufs, x_bufs, dy_addr, x_addr;
    V qg_dy_bytes = NULL, qg_x_bytes = NULL;
    std::vector<DmaLane> dy_lanes, x_lanes;
    std::vector<std::vector<V>> dy_idx; /* [mi][rd] */
    std::vector<std::vector<std::vector<V>>> x_idx; /* [s][ni][rd] */
    V zero_frag, zero_acc;
    V n_rb = NULL; /* AUTO: row-loop trip count, ceil(H / U) (H / U with row_tail) */
    V n_tail = NULL; /* row_tail: H - n_rb * U */
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

/* seg_rsrcs(n0): L64 dY / X descriptors of the segment's images
 * n0 .. n0 + UN - 1 (a 64-bit rebased base; images past N are masked by the
 * lanes). */
void seg_rsrcs(Ctx& c, V n0, V& a_rsrc, V& b_rsrc)
{
    const Em& e = c.e;
    const rocke_dconv_params_t& P = c.params;
    n0 = rocke_b_readfirstlane(e.b, n0);
    V out[2];
    V ptrs[2] = {P.A, P.Bp};
    V sns[2] = {P.p_A_stride_n, P.p_B_stride_n};
    for(int i = 0; i < 2; ++i)
    {
        V img_b = e.mul(sns[i], c.c_two);
        V n64 = e.zext64(n0);
        V i64 = e.zext64(img_b);
        V base = rocke_b_global_ptr_offset(e.b, ptrs[i], e.mul(n64, i64));
        V k = e.ci(c.UN);
        out[i] = rocke_b_buffer_rsrc(e.b, base, e.mul(k, img_b));
    }
    a_rsrc = out[0];
    b_rsrc = out[1];
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
        V m1 = e.mul(c.L64 ? d.img : n_i, P.p_A_stride_n);
        V m2 = e.mul(wo, P.p_A_stride_wi);
        V s = e.add(m1, m2);
        V t = e.add(c.k_blk0, d.ch);
        dy.push_back({e.add(s, t), ok});
    }
    for(const DmaLane& d : c.x_lanes)
    {
        V n_i = e.add(n0, d.img);
        V wi = e.add(q0, e.subi(d.col, c.PAD_W));
        V a = e.ge(wi, c.c0);
        V bq = e.lt(wi, P.p_Wi);
        V ab = e.land(a, bq);
        V cn = e.lt(n_i, P.p_N);
        V ok = e.land(d.okr, e.land(ab, cn));
        V m1 = e.mul(c.L64 ? d.img : n_i, P.p_B_stride_n);
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
    V a_rsrc, b_rsrc; /* the block's descriptors, or the segment's (L64) */
};

/* A segment's first image, strip and row range (seg_origin). */
struct SegOrigin
{
    V n0, q, h0, h_end;
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
    /* pp_sched forms every DMA address of the row before the first DMA;
     * otherwise each DMA issues as soon as its address is formed. */
    struct Load
    {
        V rsrc, lds, voff;
    };
    std::vector<Load> loads;
    auto dma = [&](V rsrc, V lds, V voff) {
        if(c.PPS)
        {
            loads.push_back({rsrc, lds, voff});
        }
        else
        {
            rocke_b_async_buffer_load_lds(e.b, rsrc, lds, voff, c.c0, 4, 0);
        }
    };
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
        dma(sg.a_rsrc, lds, voff);
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
        dma(sg.b_rsrc, lds, voff);
    }
    if(c.PPS)
    {
        /* Every address in its own register before the first DMA. */
        rocke_b_sched_barrier(e.b, 0);
        for(const Load& l : loads)
        {
            rocke_b_async_buffer_load_lds(e.b, l.rsrc, l.lds, l.voff, c.c0, 4, 0);
        }
    }
}

V read_frag(Ctx& c, V smem, const std::vector<V>& idx)
{
    V lo = c.e.ds_read_tr(smem, c.c0, idx[0], c.io_type);
    V hi = c.e.ds_read_tr(smem, c.c0, idx[1], c.io_type);
    return rocke_b_vec_concat(c.e.b, lo, hi);
}

typedef std::vector<std::vector<V>> Ring; /* [KH][WK16] */

/* wait_rows(lead) */
void wait_rows(Ctx& c, bool lead)
{
    const Em& e = c.e;
    if(pingpong_halves(c.spec))
    {
        if(lead)
        {
            rocke_b_s_waitcnt(e.b, -1, 0, -1);
        }
        else if(c.spec->pp_tight)
        {
            /* Without the extra row buffer the reads retire here. */
            rocke_b_s_waitcnt(e.b, (c.D - 1) * c.N_INSTR, 0, -1);
        }
        else
        {
            rocke_b_s_waitcnt(e.b, (c.D - 1) * c.N_INSTR, -1, -1);
        }
    }
    else if(lead)
    {
        rocke_b_s_waitcnt(e.b, (c.D - 1) * c.N_INSTR, 0, -1);
    }
    else
    {
        rocke_b_s_waitcnt(e.b, -1, 0, -1);
    }
}

/* mid_barrier() */
void mid_barrier(Ctx& c)
{
    wait_rows(c, false);
    rocke_b_s_barrier_bare(c.e.b);
    if(pingpong_halves(c.spec) || c.PPS)
    {
        rocke_b_sched_barrier(c.e.b, 0);
    }
}

/* step(buf, t, ring, accs, compute, issue) */
void step(Ctx& c, const Seg& sg, int buf, V t, Ring& ring, std::vector<V>& accs, bool compute)
{
    const Em& e = c.e;
    wait_rows(c, true);
    rocke_b_s_barrier_bare(e.b);
    if(c.PPS)
    {
        /* The row's setup stays after the barrier (not in the MFMA tail). */
        rocke_b_sched_barrier(e.b, 0);
    }
    bool late = pingpong_halves(c.spec) || c.PPS;
    if(!late)
    {
        issue_row(c, sg, (buf + c.D) % c.NB, e.addi(t, c.D));
    }
    std::vector<V> newest;
    for(int mi = 0; mi < c.WK16; ++mi)
    {
        newest.push_back(read_frag(c, c.dy_bufs[buf], c.dy_idx[mi]));
    }
    ring.erase(ring.begin());
    ring.push_back(newest);
    std::vector<std::vector<V>> xf(c.KW);
    if(compute)
    {
        for(int s = 0; s < c.KW; ++s)
        {
            for(int ni = 0; ni < c.WC16; ++ni)
            {
                xf[s].push_back(read_frag(c, c.x_bufs[buf], c.x_idx[s][ni]));
            }
        }
    }
    if(late)
    {
        rocke_b_sched_barrier(e.b, 0);
        issue_row(c, sg, (buf + c.D) % c.NB, e.addi(t, c.D));
    }
    if(!compute)
    {
        if(pingpong_halves(c.spec))
        {
            mid_barrier(c);
        }
        return;
    }
    if(c.spec->pingpong)
    {
        mid_barrier(c);
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
        if(pingpong_halves(c.spec) || c.PPS)
        {
            rocke_b_sched_barrier(e.b, 0);
        }
        rocke_b_s_setprio(e.b, 0);
    }
    else if(c.AUTO)
    {
        /* Keep each row's MFMAs in its row (runtime trip count). */
        rocke_b_sched_barrier(e.b, 0);
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
    if(c.L64)
    {
        seg_rsrcs(c, n0, sg.a_rsrc, sg.b_rsrc);
    }
    else
    {
        sg.a_rsrc = c.a_rsrc;
        sg.b_rsrc = c.b_rsrc;
    }
    /* A whole strip needs no tail cut (see the Python builder). */
    sg.t_end = e.ci(c.AUTO ? (1 << 30) : c.T_PRO + c.HPB);
    if(valid != NULL)
    {
        sg.t_end = e.select(valid, sg.t_end, c.c0);
    }
    for(int t = 0; t < c.D; ++t)
    {
        issue_row(c, sg, t % c.NB, e.ci(t));
    }
    if(pingpong_halves(c.spec))
    {
        rocke_b_s_waitcnt(e.b, (c.D - 1) * c.N_INSTR, -1, -1);
        e.if_(e.eq(c.pp_grp, c.c1), [&]() { rocke_b_s_barrier_bare(e.b); });
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
            names.push_back("r" + std::to_string(j) + "_m" + std::to_string(mi));
            inits.push_back(ring[j][mi]);
        }
    }
    V hi_bound = c.AUTO ? c.n_rb : e.ci(c.HPB / c.U);
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
    if(c.spec->row_tail && c.U > 1 && res.size() == inits.size())
    {
        Ring ring_t;
        ring_t.push_back(std::vector<V>(c.WK16, c.zero_frag));
        for(int j = 1; j < c.KH; ++j)
        {
            ring_t.push_back(std::vector<V>(res.begin() + c.n_acc + (j - 1) * c.WK16,
                                            res.begin() + c.n_acc + j * c.WK16));
        }
        V t_tail = e.addi(e.muli(c.n_rb, c.U), c.T_PRO);
        const rocke_type_t* acc_type = c.zero_acc->type;
        std::vector<const rocke_type_t*> acc_types(c.n_acc, acc_type);
        /* Tail rows u.. of the strip, each nested in the previous row's test
         * (one straight path with exits off it). */
        std::function<std::vector<V>(int, Ring, std::vector<V>)> tail
            = [&](int u, Ring ring_u, std::vector<V> accs_u) -> std::vector<V> {
            V k = e.ci(u);
            V cond = e.lt(k, c.n_tail);
            rocke_if_else_t ife = rocke_b_scf_if_else_results(e.b, cond, acc_types.data(), c.n_acc);
            if(ife.op == NULL || ife.then_region == NULL || ife.else_region == NULL)
            {
                return std::vector<V>();
            }
            rocke_b_region_enter(e.b, ife.then_region);
            {
                Ring ring_n = ring_u;
                std::vector<V> accs_n = accs_u;
                step(c, sg, (c.T_PRO + u) % c.NB, e.addi(t_tail, u), ring_n, accs_n, true);
                if(u + 1 < c.U - 1)
                {
                    accs_n = tail(u + 1, ring_n, accs_n);
                }
                rocke_b_scf_yield(e.b, accs_n.data(), (int)accs_n.size());
            }
            rocke_b_region_leave(e.b);
            rocke_b_region_enter(e.b, ife.else_region);
            rocke_b_scf_yield(e.b, accs_u.data(), (int)accs_u.size());
            rocke_b_region_leave(e.b);
            return std::vector<V>(ife.op->results, ife.op->results + ife.op->num_results);
        };
        accs_out = tail(0, ring_t, accs_out);
    }
    rocke_b_s_waitcnt(e.b, 0, 0, -1);
    rocke_b_s_barrier_bare(e.b);
    if(pingpong_halves(c.spec))
    {
        e.if_(e.eq(c.pp_grp, c.c0), [&]() { rocke_b_s_barrier_bare(e.b); });
    }
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
    c.PAD_W = rocke_direct_conv_problem_pad_w(&p);
    c.NW = n_waves(spec);
    c.WK16 = spec->wave_k16;
    c.WC16 = spec->wave_c16;
    c.BK = block_k(spec);
    c.BC = block_c(spec);
    c.GP = spec->gpack;
    c.L64 = spec->l64;
    c.PPS = spec->pp_sched;
    c.CHK = c.BK / kVec;
    c.CHC = c.BC / kVec;
    c.UN = spec->unfold_n;
    c.WU = strip_w(spec);
    c.SEG = c.WU + c.KW - 1;
    c.STRIP = c.UN * c.SEG;
    c.D = spec->prefetch;
    c.NB = n_buffers(spec);
    c.HPB = spec->rows_per_seg;
    c.AUTO = c.HPB == 0;
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
    for(int wk = 0; wk < (c.GP ? c.NW : spec->waves_k); ++wk)
    {
        for(int mi = 0; mi < c.WK16; ++mi)
        {
            k_origins.push_back(wk * wave_k(spec) + mi * kAtom);
        }
    }
    for(int wc = 0; wc < (c.GP ? c.NW : spec->waves_c); ++wc)
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
    rocke_dconv_emit_params_ex(b, &c.params, "wgrad", c.io_type, spec->dw16, spec->pk16);
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
    if(c.QG > 2)
    {
        /* Group fastest: wave w's group is w % QG, so the block's first half
         * (the ping waves) holds wave 0 of every group. */
        c.qg = e.modi(wave_all, c.QG);
        c.wave = e.divi(wave_all, c.QG);
    }
    else if(c.QG > 1)
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
    V n_k_tiles = NULL, n_c_tiles = NULL, n_ch_tiles;
    if(c.GP)
    {
        n_ch_tiles = e.cdivi(p_groups, c.NW);
    }
    else
    {
        n_k_tiles = e.cdivi(c.c_kpg, c.BK);
        n_c_tiles = e.cdivi(c.c_cpg, c.BC);
        n_ch_tiles = e.mul(e.mul(p_groups, n_k_tiles), n_c_tiles);
    }
    V n_q = e.cdivi(p_Wo, c.WU);
    V n_ng = e.cdivi(p_N, c.UN);
    V n_hb = NULL;
    V total_segs;
    if(c.AUTO)
    {
        /* One segment per strip: all H rows, the trip count rounded up (or
         * whole unrolled blocks plus a row tail). */
        if(spec->row_tail)
        {
            c.n_rb = e.divi(p_H, c.U);
            c.n_tail = e.sub(p_H, e.muli(c.n_rb, c.U));
        }
        else
        {
            c.n_rb = e.cdivi(p_H, c.U);
        }
        total_segs = e.mul(n_ng, n_q);
    }
    else
    {
        n_hb = e.cdivi(p_H, c.HPB);
        total_segs = e.mul(e.mul(n_ng, n_q), n_hb);
    }
    /* Same formula as DirectWgradTiledSpec.n_splits, which sizes the grid. */
    V sa = e.cdivi(total_segs, spec->q_groups);
    V tb = e.ci(spec->target_blocks);
    V sb;
    if(c.AUTO)
    {
        sb = e.smax(e.div(tb, n_ch_tiles), c.c1);
    }
    else
    {
        sb = e.add(tb, e.sub(n_ch_tiles, c.c1));
        sb = e.div(sb, n_ch_tiles);
    }
    V n_splits = spec->one_split ? c.c1 : e.smin(sa, sb);

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
    V wave_k_id, wave_c_id;
    if(c.GP)
    {
        /* Block = NW consecutive groups; wave w owns group gb * NW + w. */
        V g_blk0 = e.muli(ch_tile, c.NW);
        c.k_blk0 = e.mul(g_blk0, c.c_kpg); /* global k */
        c.c_glb0 = e.mul(g_blk0, c.c_cpg);
        c.c_blk0 = c.c0;
        c.group = e.add(g_blk0, c.wave);
        wave_k_id = c.wave;
        wave_c_id = c.wave;
    }
    else
    {
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

        wave_k_id = e.modi(c.wave, spec->waves_k);
        wave_c_id = e.divi(c.wave, spec->waves_k);
    }

    c.a_rsrc = NULL;
    c.b_rsrc = NULL;
    if(!c.L64)
    {
        c.a_rsrc = rocke_b_buffer_rsrc(b, P.A, P.A_bytes);
        c.b_rsrc = rocke_b_buffer_rsrc(b, P.Bp, P.B_bytes);
    }

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
    c.zero_acc = zero_acc;
    c.zero_frag = rocke_b_zero_vec(b, c.io_type, kVec);

    /* ---- persistent segment loop ---- */
    /* Segment seg -> first image, strip and row range (seg_origin in the
     * Python builder). Row block fastest, then strip, then image group; in
     * auto mode a segment is a whole strip. */
    auto seg_origin = [&](V seg) {
        V rest = seg, h0 = c.c0;
        if(!c.AUTO)
        {
            V hb = e.mod(seg, n_hb);
            rest = e.div(seg, n_hb);
            h0 = e.muli(hb, c.HPB);
        }
        V n0 = e.muli(e.div(rest, n_q), c.UN);
        V q = e.mod(rest, n_q);
        V h_end = c.AUTO ? p_H : e.smin(e.addi(h0, c.HPB), p_H);
        return SegOrigin{n0, q, h0, h_end};
    };
    /* Ping-pong group: the q group, or the half of the block's waves. */
    c.pp_grp = c.qg;
    if(pingpong_halves(spec))
    {
        c.pp_grp = e.divi(c.wave, c.NW / 2);
    }
    const bool pp_groups = spec->pingpong && !pingpong_halves(spec);
    if(pp_groups)
    {
        /* Group 1 starts one barrier late. */
        e.if_(e.eq(c.pp_grp, c.c1), [&]() { rocke_b_s_barrier_bare(b); });
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
                              SegOrigin o = seg_origin(seg);
                              return segment(c, carried, o.n0, o.q, o.h0, o.h_end, NULL);
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
                SegOrigin o = seg_origin(seg);
                V valid = e.lt(seg, total_segs);
                return segment(c, carried, o.n0, o.q, o.h0, o.h_end, valid);
            });
    }
    if((int)accs.size() != c.n_acc || !rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    if(pp_groups)
    {
        e.if_(e.eq(c.pp_grp, c.c0), [&]() { rocke_b_s_barrier_bare(b); });
    }

    /* ---- epilogue: atomic add into dW[total_k, KH, KW, cpg] ---- */
    V D_ptr = P.D;
    V rs_stride = c.c_cpg;
    V k_stride = e.muli(c.c_cpg, c.KH * c.KW);
    V ws_shift = NULL;
    if(spec->slab)
    {
        /* Slab `split` of the scratch starts dW's element count past slab 0. */
        ws_shift = e.mul(split, e.mul(P.p_total_k, k_stride));
    }
    /* _put(ptr, off, val) */
    auto put = [&](V off, V val) {
        if(spec->slab)
        {
            rocke_b_global_store(b, D_ptr, e.add(off, ws_shift), val, 4);
        }
        else if(spec->dw16)
        {
            rocke_b_global_store(b, D_ptr, off, rocke_b_cast_f32_to(b, val, c.io_type), 2);
        }
        else if(spec->one_split)
        {
            rocke_b_global_store(b, D_ptr, off, val, 4);
        }
        else
        {
            rocke_b_global_atomic_add(b, D_ptr, off, val, NULL);
        }
    };
    /* _emit(ptr, off, val, ok): put under ok; pk16: even lanes add (own, odd
     * neighbour's) as one packed pair. `val` builds the value: inside the
     * guard, or before the swap for pk16. */
    auto emit = [&](V off, const std::function<V()>& val, V ok) {
        if(!spec->pk16)
        {
            e.if_(ok, [&]() { put(off, val()); });
            return;
        }
        V v = val();
        V nb = rocke_b_warp_shuffle_xor_dpp(b, v, 1);
        V even = e.eq(e.modi(c.lane, 2), c.c0);
        e.if_(e.land(ok, even), [&]() {
            V pair_in[2] = {v, nb};
            V pair = rocke_b_pack_f32_to(b, pair_in, 2, c.bf16 ? "bf16" : "fp16");
            if(c.bf16)
            {
                rocke_b_global_atomic_add_pk_bf16(b, D_ptr, off, pair, NULL);
            }
            else
            {
                rocke_b_global_atomic_add_pk_f16(b, D_ptr, off, pair, NULL);
            }
        });
    };
    V k_lane0 = e.add(c.k_blk0, k_wave0);
    k_lane0 = e.add(k_lane0, e.muli(lane_hi, 4));
    V k_in_grp0 = e.sub(k_lane0, e.mul(c.group, c.c_kpg));
    V c_lane0, c_cpg_ok;
    if(c.GP)
    {
        /* In-group c is the lane's column alone; groups past the last are
         * masked through c (test the group). */
        c_lane0 = e.modi(c.lane, 16);
        c_cpg_ok = e.select(e.lt(c.group, p_groups), c.c_cpg, c.c0);
    }
    else
    {
        c_lane0 = e.add(c.c_blk0, c_wave0);
        c_lane0 = e.add(c_lane0, e.modi(c.lane, 16));
        c_cpg_ok = c.c_cpg;
    }
    if(c.QG == 4)
    {
        /* Every group hands a tap's accumulators over through LDS, and group
         * g sums its quarter of the tap's floats over the four slices, so all
         * waves issue the atomics. Double-buffered on the tap parity (one
         * barrier per tap) where it fits. */
        const int n_atoms = c.WK16 * c.WC16;
        const int per_g = n_atoms * 4 / c.QG; /* floats of a lane a group sums per tap */
        const int vw = per_g < 4 ? per_g : 4;
        const int slice_f = c.NW * n_atoms * kWave * 4;
        std::vector<V> red;
        for(int i = 0; i < red_buffers(spec); ++i)
        {
            int shape[2] = {1, c.QG * slice_f};
            std::string h = "red" + std::to_string(i);
            red.push_back(rocke_b_smem_alloc_f32(b, shape, 2, h.c_str()));
        }
        /* red_index(src, flt), flt = atom * 4 + slot of this lane */
        auto red_index = [&](V src, V flt) {
            V atom = e.divi(flt, 4);
            V a = e.muli(src, slice_f);
            V v = e.muli(c.wave, n_atoms);
            v = e.add(v, atom);
            v = e.muli(v, kWave);
            v = e.add(v, c.lane);
            v = e.muli(v, 4);
            v = e.add(v, e.modi(flt, 4));
            return e.add(a, v);
        };
        for(int r = 0; r < c.KH; ++r)
        {
            for(int s = 0; s < c.KW; ++s)
            {
                int tap = r * c.KW + s;
                V buf = red[tap % red.size()];
                for(int mi = 0; mi < c.WK16; ++mi)
                {
                    for(int ni = 0; ni < c.WC16; ++ni)
                    {
                        V flt = e.ci((mi * c.WC16 + ni) * 4);
                        V idx = red_index(c.qg, flt);
                        e.smem_store_f32(buf, c.c0, idx, accs[acc_index(c, r, s, mi, ni)], 4);
                    }
                }
                rocke_b_s_waitcnt(b, -1, 0, -1);
                rocke_b_s_barrier_bare(b);
                for(int q = 0; q < per_g / vw; ++q)
                {
                    V flt = e.addi(e.muli(c.qg, per_g), q * vw);
                    V atom = e.divi(flt, 4);
                    V mi_v = e.divi(atom, c.WC16);
                    V ni_v = e.modi(atom, c.WC16);
                    V tot = NULL;
                    for(int src = 0; src < c.QG; ++src)
                    {
                        V sv = e.ci(src);
                        V v = e.smem_load_f32(buf, c.c0, red_index(sv, flt), vw);
                        tot = tot == NULL ? v : rocke_b_fadd(b, tot, v);
                    }
                    V c_in = e.add(c_lane0, e.muli(ni_v, kAtom));
                    V c_ok = e.lt(c_in, c_cpg_ok);
                    for(int el = 0; el < vw; ++el)
                    {
                        V k_off = e.muli(mi_v, kAtom);
                        V k_sl = e.addi(e.modi(flt, 4), el);
                        k_off = e.add(k_off, k_sl);
                        V k_in = e.add(k_in_grp0, k_off);
                        V k_glb = e.add(k_lane0, k_off);
                        V ok = e.land(c_ok, e.lt(k_in, c.c_kpg));
                        V off = e.add(e.mul(k_glb, k_stride), c_in);
                        V tk = e.ci(tap);
                        off = e.add(off, e.mul(tk, rs_stride));
                        V val = rocke_b_vec_extract(b, tot, el);
                        emit(off, [&]() { return val; }, ok);
                    }
                }
                if(red.size() == 1)
                {
                    /* The reads retire before the next tap's stores. */
                    rocke_b_s_waitcnt(b, -1, 0, -1);
                    rocke_b_s_barrier_bare(b);
                }
            }
        }
        return rocke_ir_builder_ok(b) ? b->kernel : NULL;
    }
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
                        V c_ok = e.lt(c_in, c_cpg_ok);
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
                                emit(off, [&]() { return rocke_b_vec_extract(b, tot, slot); }, ok);
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
        V c_ok = e.lt(c_in, c_cpg_ok);
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
                        emit(off, [&]() { return val; }, ok);
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

/* ===================================================================== *
 *  Slab fold: dW[e] = sum_split ws[split][e] (the slab epilogue's reduce)
 * ===================================================================== */
namespace
{

constexpr int kFoldThreads = 256;
/* Slab loads each thread keeps in flight. */
constexpr int kFoldUnroll = 4;

int fold_quads(const rocke_direct_wgrad_slab_fold_spec_t* s)
{
    return kFoldThreads / s->slices;
}

bool fold_out_is(const rocke_direct_wgrad_slab_fold_spec_t* s, const char* dt)
{
    return s->out != NULL && strcmp(s->out, dt) == 0;
}

rocke_kernel_def_t* build_slab_fold(rocke_ir_builder_t* b,
                                    const rocke_direct_wgrad_slab_fold_spec_t* spec,
                                    const char* arch)
{
    (void)arch;
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_direct_wgrad_slab_fold_is_valid_spec(spec, reason, sizeof reason))
    {
        std::string prefix = std::string("invalid WgradSlabFoldSpec for ") + arch;
        fail_invalid(b, prefix.c_str(), reason);
        return NULL;
    }
    Em e{b};
    const int S = spec->slices;
    const int Q = fold_quads(spec);
    const int E = rocke_direct_wgrad_slab_fold_elems_per_block(spec);
    const int U = kFoldUnroll;
    const bool f32_out = fold_out_is(spec, "fp32");
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", kFoldThreads);

    rocke_param_opts_t o;
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.readonly = true;
    o.readonly_set = true;
    o.align = 16;
    o.align_set = true;
    V ws = rocke_b_param(b, "A", rocke_ptr_type(b, rocke_f32(), "global"), &o);
    const rocke_type_t* out_t = f32_out ? rocke_f32() : rocke_b_io_ir_type(b, spec->out);
    if(out_t == NULL || !rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.writeonly = true;
    o.writeonly_set = true;
    o.align = f32_out ? 4 : 2;
    o.align_set = true;
    V dw = rocke_b_param(b, "D", rocke_ptr_type(b, out_t, "global"), &o);
    V ws_bytes = rocke_b_param(b, "A_bytes", rocke_i32(), NULL);
    rocke_b_param(b, "D_bytes", rocke_i32(), NULL);
    V parts = rocke_b_param(b, "parts", rocke_i32(), NULL);
    V n_el = rocke_b_param(b, "n_elems", rocke_i32(), NULL);

    V rsrc = rocke_b_buffer_rsrc(b, ws, ws_bytes);
    V tid = rocke_b_thread_id_x(b);
    V q = e.modi(tid, Q);
    V slc = e.divi(tid, Q);
    V e0 = e.muli(rocke_b_block_id_x(b), E);
    /* Byte offset of this thread's quad in slab 0 and the slab stride. A slab
     * index past the last loads from offset -16 (past any buffer), which
     * reads 0: the scratch may be bigger than this call's slabs. */
    V q_bytes = e.muli(e.add(e0, e.muli(q, 4)), 4);
    V slab_bytes = e.muli(n_el, 4);
    V zero = rocke_b_const_f32(b, 0.0);
    V zeros[4] = {zero, zero, zero, zero};
    V zero4 = rocke_b_vec_pack(b, zeros, 4, rocke_f32());
    V n_it = e.cdivi(parts, S * U);
    V lo = e.ci(0);
    V st = e.ci(1);
    std::vector<V> res = e.for_iter(
        lo,
        n_it,
        st,
        iter_names("a", U),
        std::vector<V>(U, zero4),
        "it",
        [&](V it, std::vector<V> acc) {
            std::vector<V> vals;
            for(int j = 0; j < U; ++j)
            {
                V pidx = e.addi(e.muli(it, U), j);
                pidx = e.muli(pidx, S);
                pidx = e.add(slc, pidx);
                V cond = e.lt(pidx, parts);
                V voff = e.add(q_bytes, e.mul(pidx, slab_bytes));
                voff = e.select(cond, voff, e.ci(-16));
                V soff = e.ci(0);
                vals.push_back(rocke_b_buffer_load_vN(b, rsrc, voff, soff, rocke_f32(), 4));
            }
            for(int j = 0; j < U; ++j)
            {
                acc[j] = rocke_b_fadd(b, acc[j], vals[j]);
            }
            return acc;
        });
    if((int)res.size() != U || !rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    V tot = res[0];
    for(int j = 1; j < U; ++j)
    {
        tot = rocke_b_fadd(b, tot, res[j]);
    }
    if(S == 1)
    {
        V el = e.add(e0, e.muli(q, 4));
        e.if_(e.lt(el, n_el), [&]() {
            if(f32_out)
            {
                rocke_b_global_store_vN(b, dw, el, tot, 4, 0);
            }
            else
            {
                V v4[4];
                for(int i = 0; i < 4; ++i)
                {
                    v4[i] = rocke_b_vec_extract(b, tot, i);
                }
                rocke_b_global_store_vN(b, dw, el, rocke_b_pack_f32_to(b, v4, 4, spec->out), 4, 0);
            }
        });
        return rocke_ir_builder_ok(b) ? b->kernel : NULL;
    }
    int shape[2] = {1, S * E};
    V red = rocke_b_smem_alloc_f32(b, shape, 2, "slabfold");
    {
        V z = e.ci(0);
        V lo_i = e.muli(slc, E);
        V idx = e.add(lo_i, e.muli(q, 4));
        e.smem_store_f32(red, z, idx, tot, 4);
    }
    rocke_b_sync(b);
    for(int i = 0; i < cdiv(E, kFoldThreads); ++i)
    {
        V le = e.addi(tid, i * kFoldThreads);
        V el = e.add(e0, le);
        V ok = e.lt(el, n_el);
        if(E % kFoldThreads)
        {
            ok = e.land(ok, e.lti(le, E));
        }
        e.if_(ok, [&]() {
            V t = NULL;
            for(int sl = 0; sl < S; ++sl)
            {
                V z = e.ci(0);
                V v = rocke_b_vec_extract(b, e.smem_load_f32(red, z, e.addi(le, sl * E), 1), 0);
                t = t == NULL ? v : rocke_b_fadd(b, t, v);
            }
            if(f32_out)
            {
                rocke_b_global_store(b, dw, el, t, 4);
            }
            else
            {
                rocke_b_global_store(b, dw, el, rocke_b_cast_f32_to(b, t, out_t), 2);
            }
        });
    }
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_direct_wgrad_slab_fold_spec_t rocke_direct_wgrad_slab_fold_spec_default(void)
{
    rocke_direct_wgrad_slab_fold_spec_t s;
    memset(&s, 0, sizeof(s));
    s.out = "fp16";
    s.slices = 8;
    s.name = "direct_wgrad_slab_fold";
    return s;
}

extern "C" int
    rocke_direct_wgrad_slab_fold_elems_per_block(const rocke_direct_wgrad_slab_fold_spec_t* s)
{
    return 4 * fold_quads(s);
}

extern "C" void rocke_direct_wgrad_slab_fold_launch_geometry(
    const rocke_direct_wgrad_slab_fold_spec_t* s, int n_elems, int grid[3], int block[3])
{
    grid[0] = cdiv(n_elems, rocke_direct_wgrad_slab_fold_elems_per_block(s));
    grid[1] = 1;
    grid[2] = 1;
    block[0] = kFoldThreads;
    block[1] = 1;
    block[2] = 1;
}

extern "C" rocke_status_t rocke_direct_wgrad_slab_fold_kernel_name(
    const rocke_direct_wgrad_slab_fold_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0 || s->out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    char s_buf[24];
    snprintf(s_buf, sizeof s_buf, "s%d", s->slices);
    const char* parts[2] = {s_buf, s->out};
    return rocke_kernel_name_join(s->name, parts, 2, NULL, NULL, 0, out, out_cap, NULL);
}

extern "C" bool rocke_direct_wgrad_slab_fold_is_valid_spec(
    const rocke_direct_wgrad_slab_fold_spec_t* s, char* reason, size_t reason_cap)
{
    if(s == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(!fold_out_is(s, "fp32") && !fold_out_is(s, "fp16") && !fold_out_is(s, "bf16"))
    {
        ROCKE_SPEC_FAIL(
            reason, reason_cap, "unsupported out dtype '%s'", s->out != NULL ? s->out : "None");
    }
    if(s->slices < 1 || kFoldThreads % s->slices || s->slices > 64)
    {
        ROCKE_SPEC_FAIL(reason, reason_cap, "slices must divide 256 (at most 64)");
    }
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

extern "C" rocke_kernel_def_t* rocke_build_direct_wgrad_slab_fold(
    rocke_ir_builder_t* b, const rocke_direct_wgrad_slab_fold_spec_t* spec, const char* arch)
{
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    return build_slab_fold(b, spec, arch ? arch : "gfx950");
}

extern "C" rocke_kernel_def_t* rocke_build_direct_wgrad_slab_fold_new(
    rocke_ir_builder_t* b, const rocke_direct_wgrad_slab_fold_spec_t* spec, const char* arch)
{
    return build_new(
        b,
        spec != NULL,
        "build_wgrad_slab_fold",
        [&](char* out, size_t cap) {
            return rocke_direct_wgrad_slab_fold_kernel_name(spec, out, cap);
        },
        [&]() { return rocke_build_direct_wgrad_slab_fold(b, spec, arch); });
}

extern "C" rocke_status_t
    rocke_direct_wgrad_slab_fold_lower_to_llvm(const rocke_direct_wgrad_slab_fold_spec_t* spec,
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
    return lower_to_llvm(spec != NULL,
                         arch,
                         flavor,
                         out_ll,
                         err,
                         err_cap,
                         "build_wgrad_slab_fold failed",
                         [&](rocke_ir_builder_t* b) {
                             return rocke_build_direct_wgrad_slab_fold_new(b, spec, arch);
                         });
}
