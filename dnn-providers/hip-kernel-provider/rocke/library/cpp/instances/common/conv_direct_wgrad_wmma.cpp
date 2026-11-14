// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_wgrad_wmma.cpp -- C++ port of build_direct_wgrad_wmma
 * (library/kernels/common/conv_direct_wgrad_wmma.py): the wave32 WMMA dense
 * direct wgrad for gfx1250, streamed row by row. See the header and the
 * Python module for the algorithm.
 *
 * Every helper emits in PYTHON EVALUATION ORDER (arguments left-to-right,
 * innermost call first) so the SSA sequence -- and therefore the .ll bytes --
 * match exactly. Nested builder calls are hoisted into ordered locals: C++
 * leaves the evaluation order of call arguments unspecified.
 */
#include "rocke/instance_conv_direct_wgrad_wmma.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <array>
#include <string>
#include <vector>

#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, rocke_ceil_div */
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */
#include "rocke/instance_conv_direct_wgrad_internal.hpp" /* build_new, lower_to_llvm */
#include "rocke/ir_internal.h" /* rocke_i_set_err_msg */
#include "rocke/lower_llvm.h"

/* conv_direct_wgrad_wmma module constants. */
#define WGW_WAVE 32
#define WGW_TILE 16 /* WMMA M = N */
#define WGW_KSTEP 32 /* output columns per WMMA (WMMA K) */
#define WGW_PAD_CH 8 /* default LDS row padding (channels) */
#define WGW_DUMMY 8 /* elements past each buffer: the padding lanes' copy target */
#define WGW_VGPR_BUDGET 480 /* accumulators + fragments a wave may hold */
/* The kernel's LDS budget, not the arch capacity: the ceiling its knob grid
 * was tuned under. The arch's own capacity is checked as well. */
#define WGW_LDS_BUDGET (300 * 1024)

namespace
{

const char* wgw_dtype(const rocke_direct_wgrad_wmma_spec_t* spec)
{
    return rocke_direct_conv_problem_dtype(&spec->problem);
}

bool wgw_is_bf16(const rocke_direct_wgrad_wmma_spec_t* spec)
{
    return strcmp(wgw_dtype(spec), "bf16") == 0;
}

} // namespace

/* ===================================================================== *
 *  Spec defaults + derived geometry
 * ===================================================================== */

rocke_direct_wgrad_wmma_spec_t rocke_direct_wgrad_wmma_spec_default(void)
{
    rocke_direct_wgrad_wmma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "direct_wgrad_wmma";
    s.waves_k = 4;
    s.waves_c = 2;
    s.tiles_k = 2;
    s.tiles_c = 2;
    s.q_steps = 1;
    s.stages = 3;
    s.target_blocks = 256;
    s.prefetch = false;
    s.pad = WGW_PAD_CH;
    s.sched = false;
    s.xcd = 0;
    s.tdm = false;
    s.rows = 1;
    return s;
}

int rocke_direct_wgrad_wmma_threads(const rocke_direct_wgrad_wmma_spec_t* spec)
{
    return spec->waves_k * spec->waves_c * WGW_WAVE;
}

int rocke_direct_wgrad_wmma_block_k(const rocke_direct_wgrad_wmma_spec_t* spec)
{
    return WGW_TILE * spec->tiles_k * spec->waves_k;
}

int rocke_direct_wgrad_wmma_block_c(const rocke_direct_wgrad_wmma_spec_t* spec)
{
    return WGW_TILE * spec->tiles_c * spec->waves_c;
}

int rocke_direct_wgrad_wmma_block_q(const rocke_direct_wgrad_wmma_spec_t* spec)
{
    return WGW_KSTEP * spec->q_steps;
}

int rocke_direct_wgrad_wmma_x_cols(const rocke_direct_wgrad_wmma_spec_t* spec)
{
    return rocke_direct_wgrad_wmma_block_q(spec) + spec->problem.KW - 1;
}

int rocke_direct_wgrad_wmma_pitch_c(const rocke_direct_wgrad_wmma_spec_t* spec)
{
    return rocke_direct_wgrad_wmma_block_c(spec) + spec->pad;
}

int rocke_direct_wgrad_wmma_pitch_k(const rocke_direct_wgrad_wmma_spec_t* spec)
{
    return rocke_direct_wgrad_wmma_block_k(spec) + spec->pad;
}

long long rocke_direct_wgrad_wmma_n_k_tiles(const rocke_direct_wgrad_wmma_spec_t* spec)
{
    long long bk = rocke_direct_wgrad_wmma_block_k(spec);
    return bk != 0 ? rocke_ceil_div(rocke_direct_conv_problem_total_k(&spec->problem), bk) : 0;
}

long long rocke_direct_wgrad_wmma_n_c_tiles(const rocke_direct_wgrad_wmma_spec_t* spec)
{
    long long bc = rocke_direct_wgrad_wmma_block_c(spec);
    return bc != 0 ? rocke_ceil_div(rocke_direct_conv_problem_total_c(&spec->problem), bc) : 0;
}

long long rocke_direct_wgrad_wmma_total_steps(const rocke_direct_wgrad_wmma_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    long long bq = rocke_direct_wgrad_wmma_block_q(spec);
    if(bq == 0)
        return 0;
    return rocke_ceil_div(rocke_direct_conv_problem_Wo(p), bq) * p->N
           * ((long long)p->H + p->KH - 1);
}

long long rocke_direct_wgrad_wmma_n_split(const rocke_direct_wgrad_wmma_spec_t* spec)
{
    long long planes
        = rocke_direct_wgrad_wmma_n_k_tiles(spec) * rocke_direct_wgrad_wmma_n_c_tiles(spec);
    if(planes == 0)
        return 0;
    /* Nearest whole number of runs: the grid lands closest to target_blocks. */
    long long runs = rocke_floor_div(spec->target_blocks + rocke_floor_div(planes, 2), planes);
    long long total = rocke_direct_wgrad_wmma_total_steps(spec);
    long long n = total < runs ? total : runs;
    return n > 1 ? n : 1;
}

long long rocke_direct_wgrad_wmma_vgprs(const rocke_direct_wgrad_wmma_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    long long frags = (long long)spec->tiles_k * spec->q_steps * p->KH + spec->tiles_c;
    if(spec->prefetch) /* a whole step's fragments in flight */
        frags += (long long)spec->q_steps * (spec->tiles_k + (long long)p->KW * spec->tiles_c);
    return 8 * ((long long)p->KH * p->KW * spec->tiles_k * spec->tiles_c + frags);
}

long long rocke_direct_wgrad_wmma_lds_bytes(const rocke_direct_wgrad_wmma_spec_t* spec)
{
    long long x_rows
        = (long long)rocke_direct_wgrad_wmma_x_cols(spec) + (spec->tdm ? spec->problem.PAD : 0);
    long long per_stage = x_rows * rocke_direct_wgrad_wmma_pitch_c(spec)
                          + (long long)rocke_direct_wgrad_wmma_block_q(spec)
                                * rocke_direct_wgrad_wmma_pitch_k(spec);
    /* The builder's allocations: the dY row and, without TDM, the X row carry
     * a dummy slot past the buffer; TDM's X row has PAD slack rows. */
    long long dummies = spec->tdm ? WGW_DUMMY : 2 * WGW_DUMMY;
    return (long long)spec->stages * spec->rows * (per_stage + dummies) * 2;
}

void rocke_direct_wgrad_wmma_launch_geometry(const rocke_direct_wgrad_wmma_spec_t* spec,
                                             long long grid[3],
                                             int block[3])
{
    long long planes
        = rocke_direct_wgrad_wmma_n_c_tiles(spec) * rocke_direct_wgrad_wmma_n_k_tiles(spec);
    long long n_split = rocke_direct_wgrad_wmma_n_split(spec);
    if(spec->xcd)
    {
        grid[0] = planes * n_split;
        grid[1] = 1;
        grid[2] = 1;
    }
    else
    {
        grid[0] = planes;
        grid[1] = 1;
        grid[2] = n_split;
    }
    block[0] = rocke_direct_wgrad_wmma_threads(spec);
    block[1] = 1;
    block[2] = 1;
}

rocke_status_t rocke_direct_wgrad_wmma_kernel_name(const rocke_direct_wgrad_wmma_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap)
{
    char f[96], w[48], t[48], qs[32], st[32], tb[32], pad[32], xcd[32], g[32];
    const rocke_direct_conv_problem_t* p = &spec->problem;
    snprintf(f, sizeof f, "f%dx%dp%ds%d", p->KH, p->KW, p->PAD, p->stride);
    snprintf(w, sizeof w, "w%dx%d", spec->waves_k, spec->waves_c);
    snprintf(t, sizeof t, "t%dx%d", spec->tiles_k, spec->tiles_c);
    snprintf(qs, sizeof qs, "qs%d", spec->q_steps);
    snprintf(st, sizeof st, "st%d", spec->stages);
    snprintf(tb, sizeof tb, "tb%d", spec->target_blocks);
    if(spec->pad != WGW_PAD_CH)
        snprintf(pad, sizeof pad, "pad%d", spec->pad);
    else
        pad[0] = '\0';
    snprintf(xcd, sizeof xcd, "xcd%d", spec->xcd);
    snprintf(g, sizeof g, "g%d", spec->rows);
    const char* parts[] = {f, w, t, qs, st, tb, pad};
    const char* flag_names[] = {"bf16", "pf", "sched", xcd, "tdm", g};
    int flag_on[] = {wgw_is_bf16(spec) ? 1 : 0,
                     spec->prefetch ? 1 : 0,
                     spec->sched ? 1 : 0,
                     spec->xcd != 0 ? 1 : 0,
                     spec->tdm ? 1 : 0,
                     spec->rows > 1 ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "direct_wgrad_wmma",
                                  parts,
                                  sizeof(parts) / sizeof(parts[0]),
                                  flag_names,
                                  flag_on,
                                  sizeof(flag_names) / sizeof(flag_names[0]),
                                  out,
                                  out_cap,
                                  NULL);
}

/* ===================================================================== *
 *  is_valid_direct_wgrad_wmma_spec
 * ===================================================================== */

bool rocke_direct_wgrad_wmma_is_valid_spec(const rocke_direct_wgrad_wmma_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap)
{
    char msg[512];
    if(spec == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
        arch = "gfx1250";
    const rocke_direct_conv_problem_t* p = &spec->problem;
    if(rocke_direct_dilation_reason(p, reason, reason_cap))
        return false;
    const rocke_archtarget_t* target = rocke_dwg::wgrad_wmma_arch_target(
        arch, rocke_dwg::problem_is_bf16(p), reason, reason_cap);
    if(target == NULL)
        return false;
    if(p->groups != 1)
    {
        rocke_spec_set_reason(reason, reason_cap, "groups == 1 only");
        return false;
    }
    if(strcmp(wgw_dtype(spec), "fp16") != 0 && strcmp(wgw_dtype(spec), "bf16") != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "fp16/bf16 only");
        return false;
    }
    if(p->stride != 1)
    {
        rocke_spec_set_reason(reason, reason_cap, "stride 1 only");
        return false;
    }
    if(p->PAD > p->KH - 1 || p->PAD > p->KW - 1)
    {
        rocke_spec_set_reason(reason, reason_cap, "PAD <= K - 1");
        return false;
    }
    if(rocke_direct_conv_one_padding_reason(p, "", reason, reason_cap))
        return false; /* the kernel offsets columns by PAD */
    if(rocke_direct_conv_problem_total_c(p) % 8 != 0
       || rocke_direct_conv_problem_total_k(p) % 8 != 0)
    {
        rocke_spec_set_reason(
            reason, reason_cap, "C and K must be multiples of 8 (16-byte chunks)");
        return false;
    }
    int nw = spec->waves_k * spec->waves_c;
    if(nw != 2 && nw != 4 && nw != 8 && nw != 16)
    {
        rocke_spec_set_reason(reason, reason_cap, "2, 4, 8 or 16 waves");
        return false;
    }
    auto tile_ok = [](int t) { return t == 1 || t == 2 || t == 4; };
    if(!tile_ok(spec->tiles_k) || !tile_ok(spec->tiles_c))
    {
        rocke_spec_set_reason(reason, reason_cap, "tiles_k / tiles_c in (1, 2, 4)");
        return false;
    }
    if(spec->q_steps != 1 && spec->q_steps != 2)
    {
        rocke_spec_set_reason(reason, reason_cap, "q_steps in (1, 2)");
        return false;
    }
    if(spec->tdm && 2 * spec->rows > nw && (2 * spec->rows) % nw != 0)
    {
        rocke_spec_set_reason(
            reason, reason_cap, "tdm: 2 * rows transfers must spread evenly over the waves");
        return false;
    }
    if(spec->rows < 1)
    {
        rocke_spec_set_reason(reason, reason_cap, "rows >= 1");
        return false;
    }
    if(spec->rows > 1 && spec->prefetch && spec->stages < 3)
    {
        rocke_spec_set_reason(reason, reason_cap, "rows > 1 with prefetch needs stages >= 3");
        return false;
    }
    if(spec->pad < 0 || spec->pad % 8 != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "pad: a multiple of 8 channels (16-byte rows)");
        return false;
    }
    if(spec->stages < 2 || spec->stages > 8)
    {
        rocke_spec_set_reason(reason, reason_cap, "stages in 2..8");
        return false;
    }
    if(spec->target_blocks < 1)
    {
        rocke_spec_set_reason(reason, reason_cap, "target_blocks >= 1");
        return false;
    }
    if(rocke_direct_wgrad_wmma_vgprs(spec) > WGW_VGPR_BUDGET)
    {
        snprintf(msg,
                 sizeof msg,
                 "~%lld VGPRs of accumulators and fragments",
                 rocke_direct_wgrad_wmma_vgprs(spec));
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(rocke_dwg::wgrad_wmma_lds_over(rocke_direct_wgrad_wmma_lds_bytes(spec),
                                      WGW_LDS_BUDGET,
                                      target,
                                      arch,
                                      reason,
                                      reason_cap))
        return false;
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  Builder
 * ===================================================================== */
namespace
{

/* One copy slot of copy_slots(): (LDS index, live mask, in-tile mask or
 * NULL, column, absolute channel). */
struct wgw_slot
{
    rocke_value_t* lds_i;
    rocke_value_t* ok;
    rocke_value_t* in_tile;
    rocke_value_t* col;
    rocke_value_t* ch_abs;
};

/* The copy cursor (g, y, n, q0). */
struct wgw_cursor
{
    rocke_value_t* g;
    rocke_value_t* y;
    rocke_value_t* n;
    rocke_value_t* q0;
};

using wgw_vals = std::vector<rocke_value_t*>;

struct wgw_ctx
{
    rocke_ir_builder_t* b;
    const rocke_direct_wgrad_wmma_spec_t* spec;
    rocke_dconv_params_t params;
    const rocke_type_t* io_type;
    const char* wmma_op;

    int KH, KW, PAD, TK, TC, QS, BK, BC, BQ, XC, PC, PK, T, D, G, NS, PRE, NW;
    int X_DUMMY, Y_DUMMY, PER_STEP, TDM_PER_WAVE, N_DY;

    rocke_value_t *c0, *c1;
    rocke_value_t *tid, *lane, *wave, *wave_k, *wave_c;
    rocke_value_t *spu, *g0, *g1, *c_blk0, *k_blk0;
    rocke_value_t* zero8;
    rocke_value_t *c_ext, *k_ext;
    rocke_value_t *y_base, *x_base;

    std::vector<rocke_value_t*> x_stage, y_stage, x_lds, y_lds;
    std::vector<wgw_slot> x_slots, y_slots;
};

rocke_value_t* ci(wgw_ctx& c, int64_t v)
{
    return rocke_b_const_i32(c.b, v);
}

/* copy_slots(cols, chans, pitch, ch0, limit): slot = i * T + tid ->
 * (column, 8-channel chunk). The absolute channel (ch0 + ch) is added by the
 * caller after every slot is built, as the Python list comprehension does. */
std::vector<wgw_slot>
    copy_slots(wgw_ctx& c, int cols, int chans, int pitch, rocke_value_t* ch0, rocke_value_t* limit)
{
    rocke_ir_builder_t* b = c.b;
    std::vector<wgw_slot> out;
    std::vector<rocke_value_t*> chs;
    int ch8 = chans / 8;
    long long n = rocke_ceil_div((long long)cols * ch8, c.T);
    for(long long i = 0; i < n; ++i)
    {
        rocke_value_t* slot;
        {
            rocke_value_t* x = ci(c, i * c.T);
            slot = rocke_b_add(b, x, c.tid);
        }
        rocke_value_t* col;
        {
            rocke_value_t* x = ci(c, ch8);
            col = rocke_b_div(b, slot, x);
        }
        rocke_value_t* ch;
        {
            rocke_value_t* x = ci(c, ch8);
            rocke_value_t* m = rocke_b_mod(b, slot, x);
            rocke_value_t* y = ci(c, 8);
            ch = rocke_b_mul(b, m, y);
        }
        rocke_value_t* ok;
        {
            rocke_value_t* x = rocke_b_add(b, ch0, ch);
            ok = rocke_b_cmp_lt(b, x, limit);
        }
        rocke_value_t* in_tile = NULL;
        if((i + 1) * c.T > (long long)cols * ch8) /* the last pass has idle lanes */
        {
            rocke_value_t* x = ci(c, cols);
            in_tile = rocke_b_cmp_lt(b, col, x);
            ok = rocke_b_land(b, ok, in_tile);
        }
        rocke_value_t* lds_i;
        {
            rocke_value_t* x = ci(c, pitch);
            rocke_value_t* y = rocke_b_mul(b, col, x);
            lds_i = rocke_b_add(b, y, ch);
        }
        out.push_back(wgw_slot{lds_i, ok, in_tile, col, NULL});
        chs.push_back(ch);
    }
    for(size_t i = 0; i < out.size(); ++i)
        out[i].ch_abs = rocke_b_add(b, ch0, chs[i]);
    return out;
}

/* copy(src, stage, dummy, lds_i, live, in_tile, src_off): one 16-byte async
 * copy per lane; a padding lane copies element 0 into the dummy slot and
 * zero-fills its own slot. */
void copy(wgw_ctx& c,
          rocke_value_t* src,
          rocke_value_t* stage,
          int dummy,
          rocke_value_t* lds_i,
          rocke_value_t* live,
          rocke_value_t* in_tile,
          rocke_value_t* src_off)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* s_off = rocke_b_select(b, live, src_off, c.c0);
    rocke_value_t* idx[2];
    idx[0] = c.c0;
    {
        rocke_value_t* x = ci(c, dummy);
        idx[1] = rocke_b_select(b, live, lds_i, x);
    }
    rocke_b_global_load_async_to_lds(b, src, s_off, stage, idx, 2, 16, 0, 0);
    rocke_value_t* pad;
    if(in_tile == NULL)
        pad = rocke_b_lnot(b, live);
    else
    {
        rocke_value_t* x = rocke_b_lnot(b, live);
        pad = rocke_b_land(b, in_tile, x);
    }
    rocke_if_t gate = rocke_b_scf_if(b, pad);
    rocke_b_region_enter(b, gate.then_region);
    {
        rocke_value_t* sidx[2] = {c.c0, lds_i};
        rocke_b_smem_store_vN(b, stage, sidx, 2, c.zero8, 8);
    }
    rocke_b_region_leave(b);
}

/* cursor_at(g): steps before the list (g < 0) only mask. */
wgw_cursor cursor_at(wgw_ctx& c, rocke_value_t* g, rocke_value_t* p_N)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* gpos = rocke_b_smax(b, g, c.c0);
    rocke_value_t* unit = rocke_b_div(b, gpos, c.spu);
    rocke_value_t* y;
    {
        rocke_value_t* m = rocke_b_mod(b, gpos, c.spu);
        rocke_value_t* x = ci(c, c.PRE);
        rocke_value_t* s = rocke_b_sub(b, m, x);
        rocke_value_t* d = rocke_b_sub(b, gpos, g);
        y = rocke_b_sub(b, s, d);
    }
    rocke_value_t* n = rocke_b_mod(b, unit, p_N);
    rocke_value_t* q0;
    {
        rocke_value_t* x = rocke_b_div(b, unit, p_N);
        rocke_value_t* z = ci(c, c.BQ);
        q0 = rocke_b_mul(b, x, z);
    }
    return wgw_cursor{g, y, n, q0};
}

wgw_cursor advance(wgw_ctx& c, const wgw_cursor& cur)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    rocke_value_t* y1 = rocke_b_add(b, cur.y, c.c1);
    rocke_value_t* wrap = rocke_b_cmp_eq(b, y1, params->p_Hi);
    rocke_value_t* n1;
    {
        rocke_value_t* x = rocke_b_add(b, cur.n, c.c1);
        n1 = rocke_b_select(b, wrap, x, cur.n);
    }
    rocke_value_t* wrap_n = rocke_b_cmp_eq(b, n1, params->p_N);
    wgw_cursor out;
    out.g = rocke_b_add(b, cur.g, c.c1);
    {
        rocke_value_t* x = ci(c, -c.PRE);
        out.y = rocke_b_select(b, wrap, x, y1);
    }
    out.n = rocke_b_select(b, wrap_n, c.c0, n1);
    {
        rocke_value_t* x = ci(c, c.BQ);
        rocke_value_t* q = rocke_b_add(b, cur.q0, x);
        out.q0 = rocke_b_select(b, wrap_n, q, cur.q0);
    }
    return out;
}

/* One rank-2 TDM transfer, issued by wave `wave_sel`. */
bool tdm_load(wgw_ctx& c,
              rocke_value_t* gptr,
              rocke_value_t* lds_addr,
              rocke_value_t* dim0,
              rocke_value_t* dim1,
              rocke_value_t* row_stride,
              int tile0,
              int tile1)
{
    rocke_value_t* d[5];
    if(!rocke_b_tdm_descriptor_2d(
           c.b, gptr, lds_addr, 2, dim0, dim1, row_stride, tile0, tile1, 0, -1, 0, d))
        return false;
    rocke_b_tensor_load_to_lds(c.b, d[0], d[1], d[2], d[3], d[4], 0);
    return true;
}

/* tdm_copies: one TDM transfer per operand; on the left image edge the X row
 * lands `-w0` rows down and the rows above it are zeroed with LDS stores. */
bool tdm_copies(wgw_ctx& c,
                int buf,
                rocke_value_t* n_img,
                rocke_value_t* y,
                rocke_value_t* q0,
                rocke_value_t* x_ok,
                rocke_value_t* pr,
                rocke_value_t* y_ok)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    rocke_value_t* w0;
    {
        rocke_value_t* x = ci(c, c.PAD);
        w0 = rocke_b_sub(b, q0, x);
    }
    rocke_value_t* w_start = rocke_b_smax(b, w0, c.c0);
    rocke_value_t* shift = rocke_b_sub(b, w_start, w0);
    int g_row = buf % c.G;
    {
        rocke_value_t* x = ci(c, (2 * g_row) % c.NW);
        rocke_value_t* cond = rocke_b_cmp_eq(b, c.wave, x);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        rocke_value_t* x_cols;
        {
            rocke_value_t* s = rocke_b_sub(b, params->p_Wi, w_start);
            rocke_value_t* z = ci(c, c.XC);
            rocke_value_t* m = rocke_b_smin(b, s, z);
            x_cols = rocke_b_smax(b, m, c.c0);
        }
        rocke_value_t* off;
        {
            rocke_value_t* a = rocke_b_mul(b, n_img, params->p_B_stride_n);
            rocke_value_t* sel = rocke_b_select(b, x_ok, y, c.c0);
            rocke_value_t* m = rocke_b_mul(b, sel, params->p_B_stride_hi);
            rocke_value_t* s1 = rocke_b_add(b, a, m);
            rocke_value_t* w = rocke_b_mul(b, w_start, params->p_B_stride_wi);
            rocke_value_t* s2 = rocke_b_add(b, w, c.c_blk0);
            off = rocke_b_add(b, s1, s2);
        }
        rocke_value_t* gptr;
        {
            rocke_value_t* x2 = ci(c, 2);
            rocke_value_t* bytes = rocke_b_mul(b, off, x2);
            gptr = rocke_b_global_ptr_add(b, params->Bp, bytes);
        }
        rocke_value_t* lds_addr;
        {
            rocke_value_t* xpc = ci(c, c.PC);
            rocke_value_t* m = rocke_b_mul(b, shift, xpc);
            rocke_value_t* x2 = ci(c, 2);
            rocke_value_t* m2 = rocke_b_mul(b, m, x2);
            lds_addr = rocke_b_add(b, c.x_lds[buf], m2);
        }
        rocke_value_t* dim1 = rocke_b_select(b, x_ok, x_cols, c.c0);
        if(!tdm_load(c, gptr, lds_addr, c.c_ext, dim1, params->p_B_stride_wi, c.PC, c.XC))
            return false;
        rocke_b_region_leave(b);
    }
    {
        rocke_value_t* x = ci(c, (2 * g_row + 1) % c.NW);
        rocke_value_t* cond = rocke_b_cmp_eq(b, c.wave, x);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        rocke_value_t* y_cols;
        {
            rocke_value_t* s = rocke_b_sub(b, params->p_Wo, q0);
            rocke_value_t* z = ci(c, c.BQ);
            rocke_value_t* m = rocke_b_smin(b, s, z);
            y_cols = rocke_b_smax(b, m, c.c0);
        }
        rocke_value_t* off;
        {
            rocke_value_t* a = rocke_b_mul(b, n_img, params->p_A_stride_n);
            rocke_value_t* sel = rocke_b_select(b, y_ok, pr, c.c0);
            rocke_value_t* m = rocke_b_mul(b, sel, params->p_A_stride_hi);
            rocke_value_t* s1 = rocke_b_add(b, a, m);
            rocke_value_t* w = rocke_b_mul(b, q0, params->p_A_stride_wi);
            rocke_value_t* s2 = rocke_b_add(b, w, c.k_blk0);
            off = rocke_b_add(b, s1, s2);
        }
        rocke_value_t* gptr;
        {
            rocke_value_t* x2 = ci(c, 2);
            rocke_value_t* bytes = rocke_b_mul(b, off, x2);
            gptr = rocke_b_global_ptr_add(b, params->A, bytes);
        }
        rocke_value_t* dim1 = rocke_b_select(b, y_ok, y_cols, c.c0);
        if(!tdm_load(c, gptr, c.y_lds[buf], c.k_ext, dim1, params->p_A_stride_wi, c.PK, c.BQ))
            return false;
        rocke_b_region_leave(b);
    }
    if(c.PAD)
    {
        int chunks = c.PAD * c.PC / 8;
        rocke_value_t* cond = rocke_b_cmp_gt(b, shift, c.c0);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        for(int j = 0; j < chunks; j += c.T)
        {
            rocke_value_t* idx;
            {
                rocke_value_t* x = ci(c, j);
                idx = rocke_b_add(b, c.tid, x);
            }
            rocke_value_t* in_rows;
            {
                rocke_value_t* xpc = ci(c, c.PC);
                rocke_value_t* m = rocke_b_mul(b, shift, xpc);
                rocke_value_t* x8 = ci(c, 8);
                rocke_value_t* d = rocke_b_div(b, m, x8);
                in_rows = rocke_b_cmp_lt(b, idx, d);
            }
            rocke_if_t g2 = rocke_b_scf_if(b, in_rows);
            rocke_b_region_enter(b, g2.then_region);
            {
                rocke_value_t* sidx[2];
                sidx[0] = c.c0;
                rocke_value_t* x8 = ci(c, 8);
                sidx[1] = rocke_b_mul(b, idx, x8);
                rocke_b_smem_store_vN(b, c.x_stage[buf], sidx, 2, c.zero8, 8);
            }
            rocke_b_region_leave(b);
        }
        rocke_b_region_leave(b);
    }
    return true;
}

void masked_copies(wgw_ctx& c,
                   int buf,
                   rocke_value_t* q0,
                   rocke_value_t* x_ok,
                   rocke_value_t* x_row,
                   rocke_value_t* y_ok,
                   rocke_value_t* y_row)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    for(const wgw_slot& s : c.x_slots)
    {
        rocke_value_t* wi;
        {
            rocke_value_t* x = ci(c, c.PAD);
            rocke_value_t* d = rocke_b_sub(b, q0, x);
            wi = rocke_b_add(b, d, s.col);
        }
        rocke_value_t* live;
        {
            rocke_value_t* l1 = rocke_b_land(b, s.ok, x_ok);
            rocke_value_t* ge = rocke_b_cmp_ge(b, wi, c.c0);
            rocke_value_t* lt = rocke_b_cmp_lt(b, wi, params->p_Wi);
            rocke_value_t* l2 = rocke_b_land(b, ge, lt);
            live = rocke_b_land(b, l1, l2);
        }
        rocke_value_t* off;
        {
            rocke_value_t* m = rocke_b_mul(b, wi, params->p_B_stride_wi);
            rocke_value_t* a = rocke_b_add(b, m, s.ch_abs);
            off = rocke_b_add(b, x_row, a);
        }
        copy(c, params->Bp, c.x_stage[buf], c.X_DUMMY, s.lds_i, live, s.in_tile, off);
    }
    for(const wgw_slot& s : c.y_slots)
    {
        rocke_value_t* wo = rocke_b_add(b, q0, s.col);
        rocke_value_t* live;
        {
            rocke_value_t* l1 = rocke_b_land(b, s.ok, y_ok);
            rocke_value_t* lt = rocke_b_cmp_lt(b, wo, params->p_Wo);
            live = rocke_b_land(b, l1, lt);
        }
        rocke_value_t* off;
        {
            rocke_value_t* m = rocke_b_mul(b, wo, params->p_A_stride_wi);
            rocke_value_t* a = rocke_b_add(b, m, s.ch_abs);
            off = rocke_b_add(b, y_row, a);
        }
        copy(c, params->A, c.y_stage[buf], c.Y_DUMMY, s.lds_i, live, s.in_tile, off);
    }
}

/* issue(buf, cur): copy the cursor's step -- X row y (masked before the run)
 * and dY row y + PAD. */
bool issue(wgw_ctx& c, int buf, const wgw_cursor& cur)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    rocke_value_t* live_g;
    {
        rocke_value_t* lt = rocke_b_cmp_lt(b, cur.g, c.g1);
        rocke_value_t* ge = rocke_b_cmp_ge(b, cur.g, c.c0);
        live_g = rocke_b_land(b, lt, ge);
    }
    rocke_value_t* x_ok;
    {
        rocke_value_t* ge = rocke_b_cmp_ge(b, cur.g, c.g0);
        rocke_value_t* l1 = rocke_b_land(b, live_g, ge);
        rocke_value_t* y0 = rocke_b_cmp_ge(b, cur.y, c.c0);
        rocke_value_t* yh = rocke_b_cmp_lt(b, cur.y, params->p_Hi);
        rocke_value_t* l2 = rocke_b_land(b, y0, yh);
        x_ok = rocke_b_land(b, l1, l2);
    }
    rocke_value_t* x_row;
    {
        rocke_value_t* a = rocke_b_mul(b, cur.n, params->p_B_stride_n);
        rocke_value_t* m = rocke_b_mul(b, cur.y, params->p_B_stride_hi);
        x_row = rocke_b_add(b, a, m);
    }
    rocke_value_t* pr;
    {
        rocke_value_t* x = ci(c, c.PAD);
        pr = rocke_b_add(b, cur.y, x);
    }
    rocke_value_t* y_ok;
    {
        rocke_value_t* ge = rocke_b_cmp_ge(b, pr, c.c0);
        rocke_value_t* lt = rocke_b_cmp_lt(b, pr, params->p_Ho);
        rocke_value_t* l = rocke_b_land(b, ge, lt);
        y_ok = rocke_b_land(b, live_g, l);
    }
    rocke_value_t* y_row;
    {
        rocke_value_t* a = rocke_b_mul(b, cur.n, params->p_A_stride_n);
        rocke_value_t* m = rocke_b_mul(b, pr, params->p_A_stride_hi);
        y_row = rocke_b_add(b, a, m);
    }
    if(c.spec->tdm)
        return tdm_copies(c, buf, cur.n, cur.y, cur.q0, x_ok, pr, y_ok);
    masked_copies(c, buf, cur.q0, x_ok, x_row, y_ok, y_row);
    return true;
}

/* wait_copies: wait until at most `stages_in_flight` stages' copies are
 * outstanding. */
void wait_copies(wgw_ctx& c, int stages_in_flight)
{
    if(c.spec->tdm)
        rocke_b_s_wait_tensorcnt(c.b, stages_in_flight * c.TDM_PER_WAVE);
    else
        rocke_b_s_wait_asynccnt(c.b, stages_in_flight * c.G * c.PER_STEP);
}

/* frag(stage, base, pitch, col, ch): one 16x16 transposed fragment. */
rocke_value_t*
    frag(wgw_ctx& c, rocke_value_t* stage, rocke_value_t* base, int pitch, int col, int ch)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* addr;
    {
        rocke_value_t* x = ci(c, (int64_t)col * pitch + ch);
        addr = rocke_b_add(b, base, x);
    }
    rocke_value_t* lo;
    {
        rocke_value_t* idx[2] = {c.c0, addr};
        lo = rocke_b_ds_read_tr16_b128(b, stage, idx, 2, c.io_type);
    }
    rocke_value_t* hi;
    {
        rocke_value_t* x = ci(c, 8 * pitch);
        rocke_value_t* a2 = rocke_b_add(b, addr, x);
        rocke_value_t* idx[2] = {c.c0, a2};
        hi = rocke_b_ds_read_tr16_b128(b, stage, idx, 2, c.io_type);
    }
    return rocke_b_vec_concat(b, lo, hi);
}

wgw_vals read_frags(wgw_ctx& c, int buf)
{
    wgw_vals out;
    for(int tk = 0; tk < c.TK; ++tk)
        for(int q = 0; q < c.QS; ++q)
            out.push_back(frag(c, c.y_stage[buf], c.y_base, c.PK, WGW_KSTEP * q, WGW_TILE * tk));
    for(int q = 0; q < c.QS; ++q)
        for(int s = 0; s < c.KW; ++s)
            for(int tc = 0; tc < c.TC; ++tc)
                out.push_back(
                    frag(c, c.x_stage[buf], c.x_base, c.PC, WGW_KSTEP * q + s, WGW_TILE * tc));
    return out;
}

int acc_i(const wgw_ctx& c, int r, int s, int tk, int tc)
{
    return ((r * c.KW + s) * c.TK + tk) * c.TC + tc;
}

/* multiply(accs, ring, frags) -> the ring becomes full[:KH - 1]. */
void multiply(wgw_ctx& c, wgw_vals& accs, std::vector<wgw_vals>& ring, const wgw_vals& frags)
{
    wgw_vals newest(frags.begin(), frags.begin() + c.N_DY);
    wgw_vals xs(frags.begin() + c.N_DY, frags.end());
    std::vector<wgw_vals> full;
    full.push_back(newest);
    for(const wgw_vals& row : ring)
        full.push_back(row);
    for(int q = 0; q < c.QS; ++q)
        for(int tk = 0; tk < c.TK; ++tk)
            for(int r = 0; r < c.KH; ++r)
            {
                rocke_value_t* a = full[r][tk * c.QS + q];
                for(int s = 0; s < c.KW; ++s)
                    for(int tc = 0; tc < c.TC; ++tc)
                    {
                        int i = acc_i(c, r, s, tk, tc);
                        rocke_value_t* bb = xs[(q * c.KW + s) * c.TC + tc];
                        accs[i] = rocke_b_mma(c.b, c.wmma_op, a, bb, accs[i], NULL, 0);
                    }
            }
    full.resize(c.KH - 1);
    ring = full;
}

void mult(wgw_ctx& c, wgw_vals& accs, std::vector<wgw_vals>& ring, const wgw_vals& frags)
{
    if(c.spec->sched)
    {
        /* Keep the copies and the reads ahead of the WMMAs. */
        rocke_b_sched_barrier(c.b, 0);
        rocke_b_s_setprio(c.b, 1);
    }
    multiply(c, accs, ring, frags);
    if(c.spec->sched)
        rocke_b_s_setprio(c.b, 0);
}

bool issue_stage(wgw_ctx& c, int d, wgw_cursor& cur)
{
    for(int g = 0; g < c.G; ++g)
    {
        if(!issue(c, d * c.G + g, cur))
            return false;
        cur = advance(c, cur);
    }
    return true;
}

/* stage(cur, d, accs, ring, frags): stage j (d = j % D), G rows behind one
 * barrier. */
bool stage(wgw_ctx& c,
           wgw_cursor& cur,
           int d,
           wgw_vals& accs,
           std::vector<wgw_vals>& ring,
           wgw_vals& frags)
{
    rocke_ir_builder_t* b = c.b;
    int D = c.D, G = c.G;
    if(!c.spec->prefetch)
    {
        wait_copies(c, D - 2);
        rocke_b_sync_lds_only(b);
        if(!issue_stage(c, (d + D - 1) % D, cur))
            return false;
        for(int g = 0; g < G; ++g)
            mult(c, accs, ring, read_frags(c, d * G + g));
        frags.clear();
        return true;
    }
    if(G == 1)
    {
        wait_copies(c, D - 2);
        rocke_b_sync_lds_only(b);
        if(!issue_stage(c, d, cur))
            return false;
        wgw_vals nxt = read_frags(c, (d + 1) % D);
        mult(c, accs, ring, frags);
        frags = nxt;
        return true;
    }
    wait_copies(c, D - 3);
    rocke_b_sync_lds_only(b);
    if(!issue_stage(c, (d + D - 1) % D, cur))
        return false;
    for(int g = 0; g < G; ++g)
    {
        int nslot = (g + 1 < G) ? d * G + g + 1 : ((d + 1) % D) * G;
        wgw_vals nxt = read_frags(c, nslot);
        mult(c, accs, ring, frags);
        frags = nxt;
    }
    return true;
}

} // namespace

rocke_kernel_def_t* rocke_build_direct_wgrad_wmma(rocke_ir_builder_t* b,
                                                  const rocke_direct_wgrad_wmma_spec_t* spec,
                                                  const char* arch)
{
    char reason[512];
    wgw_ctx c;

    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx1250";
    if(!rocke_direct_wgrad_wmma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string msg = std::string("invalid DirectWgradWmmaSpec: ") + reason;
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
        return NULL;
    }
    const rocke_direct_conv_problem_t* p = &spec->problem;
    c.b = b;
    c.spec = spec;
    c.KH = p->KH;
    c.KW = p->KW;
    c.PAD = p->PAD;
    c.TK = spec->tiles_k;
    c.TC = spec->tiles_c;
    c.QS = spec->q_steps;
    c.BK = rocke_direct_wgrad_wmma_block_k(spec);
    c.BC = rocke_direct_wgrad_wmma_block_c(spec);
    c.BQ = rocke_direct_wgrad_wmma_block_q(spec);
    c.XC = rocke_direct_wgrad_wmma_x_cols(spec);
    c.PC = rocke_direct_wgrad_wmma_pitch_c(spec);
    c.PK = rocke_direct_wgrad_wmma_pitch_k(spec);
    c.T = rocke_direct_wgrad_wmma_threads(spec);
    c.D = spec->stages;
    c.G = spec->rows;
    c.NS = c.D * c.G; /* row slots: slot d * G + g holds row g of stage d */
    c.PRE = c.KH - 1; /* ring steps before a unit's first input row */
    c.NW = spec->waves_k * spec->waves_c;
    c.TDM_PER_WAVE = (2 * c.G) / c.NW > 1 ? (2 * c.G) / c.NW : 1;
    c.N_DY = c.TK * c.QS;
    c.io_type = rocke_b_io_ir_type(b, wgw_dtype(spec));
    if(c.io_type == NULL)
        return NULL;
    c.wmma_op
        = wgw_is_bf16(spec) ? "wmma_gfx1250_f32_16x16x32_bf16" : "wmma_gfx1250_f32_16x16x32_f16";

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", c.T);
    rocke_attr_set_bool(b, &b->kernel->attrs, "lds_alias_scopes", true);
    rocke_dconv_emit_params(b, &c.params, "wgrad", c.io_type);
    const rocke_dconv_params_t* params = &c.params;

    c.c0 = ci(c, 0);
    c.c1 = ci(c, 1);
    rocke_value_t* p_N = params->p_N;
    rocke_value_t* p_C = params->p_total_c;
    rocke_value_t* p_K = params->p_total_k;

    c.tid = rocke_b_thread_id_x(b);
    {
        rocke_value_t* x = ci(c, WGW_WAVE);
        c.lane = rocke_b_mod(b, c.tid, x);
        rocke_value_t* y = ci(c, WGW_WAVE);
        rocke_value_t* d = rocke_b_div(b, c.tid, y);
        c.wave = rocke_b_readfirstlane(b, d);
    }
    {
        rocke_value_t* x = ci(c, spec->waves_k);
        c.wave_k = rocke_b_mod(b, c.wave, x);
        rocke_value_t* y = ci(c, spec->waves_k);
        c.wave_c = rocke_b_div(b, c.wave, y);
    }

    /* ---- grid: tile = k_tile * n_c_tiles + c_tile, run of the step list. */
    auto ceil_tiles = [&](rocke_value_t* v, int blk) {
        rocke_value_t* x = ci(c, blk - 1);
        rocke_value_t* a = rocke_b_add(b, v, x);
        rocke_value_t* y = ci(c, blk);
        return rocke_b_div(b, a, y);
    };
    rocke_value_t* n_ct = ceil_tiles(p_C, c.BC);
    rocke_value_t* n_kt = ceil_tiles(p_K, c.BK);
    rocke_value_t* n_qc = ceil_tiles(params->p_Wo, c.BQ);
    {
        rocke_value_t* x = ci(c, c.PRE);
        c.spu = rocke_b_add(b, params->p_Hi, x); /* steps per unit */
    }
    rocke_value_t* total;
    {
        rocke_value_t* m = rocke_b_mul(b, n_qc, p_N);
        total = rocke_b_mul(b, m, c.spu);
    }
    /* Same formula as rocke_direct_wgrad_wmma_n_split, which sizes the grid. */
    rocke_value_t* planes = rocke_b_mul(b, n_ct, n_kt);
    rocke_value_t* runs;
    {
        rocke_value_t* x = ci(c, spec->target_blocks);
        rocke_value_t* two = ci(c, 2);
        rocke_value_t* h = rocke_b_div(b, planes, two);
        rocke_value_t* a = rocke_b_add(b, x, h);
        runs = rocke_b_div(b, a, planes);
    }
    rocke_value_t* n_split;
    {
        rocke_value_t* m = rocke_b_smin(b, total, runs);
        n_split = rocke_b_smax(b, m, c.c1);
    }
    rocke_value_t *tile, *run;
    if(spec->xcd)
    {
        /* 1-D grid: renumber the blocks so that each XCD gets a contiguous
         * range of logical blocks, i.e. every tile of a few runs. */
        rocke_value_t* hw = rocke_b_block_id_x(b);
        rocke_value_t* n_blocks = rocke_b_mul(b, planes, n_split);
        rocke_value_t* per_xcd;
        {
            rocke_value_t* x = ci(c, spec->xcd);
            per_xcd = rocke_b_div(b, n_blocks, x);
        }
        rocke_value_t* even;
        {
            rocke_value_t* x = ci(c, spec->xcd);
            rocke_value_t* m = rocke_b_mul(b, per_xcd, x);
            even = rocke_b_cmp_eq(b, m, n_blocks);
        }
        rocke_value_t* logical;
        {
            rocke_value_t* x = ci(c, spec->xcd);
            rocke_value_t* md = rocke_b_mod(b, hw, x);
            rocke_value_t* m = rocke_b_mul(b, md, per_xcd);
            rocke_value_t* y = ci(c, spec->xcd);
            rocke_value_t* dv = rocke_b_div(b, hw, y);
            rocke_value_t* a = rocke_b_add(b, m, dv);
            logical = rocke_b_select(b, even, a, hw);
        }
        tile = rocke_b_mod(b, logical, planes);
        run = rocke_b_div(b, logical, planes);
    }
    else
    {
        tile = rocke_b_block_id_x(b);
        run = rocke_b_block_id_z(b);
    }
    {
        rocke_value_t* m = rocke_b_mod(b, tile, n_ct);
        rocke_value_t* x = ci(c, c.BC);
        c.c_blk0 = rocke_b_mul(b, m, x);
    }
    {
        rocke_value_t* d = rocke_b_div(b, tile, n_ct);
        rocke_value_t* x = ci(c, c.BK);
        c.k_blk0 = rocke_b_mul(b, d, x);
    }
    rocke_value_t* per_run;
    {
        rocke_value_t* s = rocke_b_sub(b, n_split, c.c1);
        rocke_value_t* a = rocke_b_add(b, total, s);
        per_run = rocke_b_div(b, a, n_split);
    }
    c.g0 = rocke_b_mul(b, run, per_run);
    {
        rocke_value_t* a = rocke_b_add(b, c.g0, per_run);
        c.g1 = rocke_b_smin(b, a, total);
    }
    rocke_value_t* g_first;
    {
        rocke_value_t* x = ci(c, c.PRE);
        g_first = rocke_b_sub(b, c.g0, x); /* replay PRE ring steps before the run */
    }

    /* ---- LDS. */
    c.X_DUMMY = c.XC * c.PC;
    c.Y_DUMMY = c.BQ * c.PK;
    /* TDM lands the X row PAD rows down on the left image edge: slack rows. */
    int X_ELEMS = spec->tdm ? (c.XC + c.PAD) * c.PC : c.X_DUMMY + WGW_DUMMY;
    for(int i = 0; i < c.NS; ++i)
    {
        int shape[2] = {1, X_ELEMS};
        std::string nm = "xs" + std::to_string(i);
        c.x_stage.push_back(rocke_b_smem_alloc(b, c.io_type, shape, 2, nm.c_str()));
    }
    for(int i = 0; i < c.NS; ++i)
    {
        int shape[2] = {1, c.Y_DUMMY + WGW_DUMMY};
        std::string nm = "ys" + std::to_string(i);
        c.y_stage.push_back(rocke_b_smem_alloc(b, c.io_type, shape, 2, nm.c_str()));
    }

    /* ---- copy slots. */
    c.x_slots = copy_slots(c, c.XC, c.BC, c.PC, c.c_blk0, p_C);
    c.y_slots = copy_slots(c, c.BQ, c.BK, c.PK, c.k_blk0, p_K);
    c.PER_STEP = (int)(c.x_slots.size() + c.y_slots.size()); /* copies per row step */

    c.zero8 = rocke_b_zero_vec(b, c.io_type, 8);

    if(spec->tdm)
    {
        for(rocke_value_t* st : c.x_stage)
        {
            rocke_value_t* a = rocke_b_smem_addr_of(b, st);
            c.x_lds.push_back(rocke_b_trunc(b, a, rocke_i32()));
        }
        for(rocke_value_t* st : c.y_stage)
        {
            rocke_value_t* a = rocke_b_smem_addr_of(b, st);
            c.y_lds.push_back(rocke_b_trunc(b, a, rocke_i32()));
        }
        {
            rocke_value_t* s = rocke_b_sub(b, p_C, c.c_blk0);
            rocke_value_t* x = ci(c, c.BC);
            rocke_value_t* m = rocke_b_smin(b, s, x);
            c.c_ext = rocke_b_smax(b, m, c.c0);
        }
        {
            rocke_value_t* s = rocke_b_sub(b, p_K, c.k_blk0);
            rocke_value_t* x = ci(c, c.BK);
            rocke_value_t* m = rocke_b_smin(b, s, x);
            c.k_ext = rocke_b_smax(b, m, c.c0);
        }
    }

    /* ---- fragment reads: ds_load_tr16_b128 of a 16 (column) x 16 (channel)
     * block of [column][channel]. */
    rocke_value_t *l16, *lhalf;
    {
        rocke_value_t* x = ci(c, 16);
        l16 = rocke_b_mod(b, c.lane, x);
        rocke_value_t* y = ci(c, 16);
        lhalf = rocke_b_div(b, c.lane, y);
    }
    rocke_value_t* tr_row;
    {
        rocke_value_t* x = ci(c, 16);
        rocke_value_t* m = rocke_b_mul(b, lhalf, x);
        rocke_value_t* y = ci(c, 8);
        rocke_value_t* md = rocke_b_mod(b, c.lane, y);
        tr_row = rocke_b_add(b, m, md);
    }
    rocke_value_t* tr_ch;
    {
        rocke_value_t* x = ci(c, 8);
        rocke_value_t* d = rocke_b_div(b, l16, x);
        rocke_value_t* y = ci(c, 8);
        tr_ch = rocke_b_mul(b, d, y);
    }
    auto base = [&](int pitch, rocke_value_t* wv, int tiles) {
        rocke_value_t* x = ci(c, pitch);
        rocke_value_t* m = rocke_b_mul(b, tr_row, x);
        rocke_value_t* y = ci(c, tiles * WGW_TILE);
        rocke_value_t* m2 = rocke_b_mul(b, wv, y);
        rocke_value_t* a = rocke_b_add(b, m2, tr_ch);
        return rocke_b_add(b, m, a);
    };
    c.y_base = base(c.PK, c.wave_k, c.TK);
    c.x_base = base(c.PC, c.wave_c, c.TC);

    int n_acc = c.KH * c.KW * c.TK * c.TC;
    rocke_value_t* zero_acc = rocke_b_zero_vec_f32(b, 8);
    rocke_value_t* zero_frag = rocke_b_zero_vec(b, c.io_type, 16);

    wgw_cursor cur = cursor_at(c, g_first, p_N);
    int n_pre = (spec->prefetch && c.G == 1) ? c.D : c.D - 1;
    for(int d = 0; d < n_pre; ++d)
        if(!issue_stage(c, d, cur))
            return NULL;
    wgw_vals frags;
    if(spec->prefetch)
    {
        wait_copies(c, n_pre - 1);
        rocke_b_sync_lds_only(b);
        frags = read_frags(c, 0);
    }
    int n_frags = (int)frags.size();
    int n_ring = (c.KH - 1) * c.TK * c.QS;
    rocke_value_t* trips;
    {
        rocke_value_t* s = rocke_b_sub(b, c.g1, g_first);
        rocke_value_t* x = ci(c, c.NS - 1);
        rocke_value_t* a = rocke_b_add(b, s, x);
        rocke_value_t* y = ci(c, c.NS);
        trips = rocke_b_div(b, a, y);
    }
    /* init = accs + ring + frags + cursor. */
    wgw_vals init;
    for(int i = 0; i < n_acc; ++i)
        init.push_back(zero_acc);
    for(int i = 0; i < n_ring; ++i)
        init.push_back(zero_frag);
    for(rocke_value_t* v : frags)
        init.push_back(v);
    init.push_back(cur.g);
    init.push_back(cur.y);
    init.push_back(cur.n);
    init.push_back(cur.q0);
    int n_init = (int)init.size();
    std::vector<std::string> names(n_init);
    std::vector<rocke_iter_arg_t> iargs(n_init);
    for(int i = 0; i < n_init; ++i)
    {
        names[i] = "a" + std::to_string(i);
        iargs[i].name = names[i].c_str();
        iargs[i].init = init[i];
    }
    rocke_for_t loop
        = rocke_b_scf_for_iter(b, c.c0, trips, c.c1, iargs.data(), n_init, "tr", false, false);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    rocke_b_region_enter(b, loop.body);
    {
        rocke_value_t** cv = loop.iter_vars;
        wgw_vals accs(cv, cv + n_acc);
        std::vector<wgw_vals> ring;
        for(int j = 0; j < c.KH - 1; ++j)
            ring.emplace_back(cv + n_acc + j * c.N_DY, cv + n_acc + (j + 1) * c.N_DY);
        wgw_vals fr(cv + n_acc + n_ring, cv + n_acc + n_ring + n_frags);
        wgw_cursor lc{cv[n_init - 4], cv[n_init - 3], cv[n_init - 2], cv[n_init - 1]};
        for(int d = 0; d < c.D; ++d)
            if(!stage(c, lc, d, accs, ring, fr))
                return NULL;
        wgw_vals out(accs);
        for(const wgw_vals& row : ring)
            out.insert(out.end(), row.begin(), row.end());
        out.insert(out.end(), fr.begin(), fr.end());
        out.push_back(lc.g);
        out.push_back(lc.y);
        out.push_back(lc.n);
        out.push_back(lc.q0);
        rocke_b_scf_yield(b, out.data(), (int)out.size());
    }
    rocke_b_region_leave(b);
    wait_copies(c, 0);
    rocke_value_t* const* accs = loop.op->results;

    /* ---- epilogue: cascade atomics into the zeroed fp32 dW. Lane l, item e:
     * dY channel k = 8 * (l // 16) + e, X channel c = l % 16. */
    rocke_value_t* d_rsrc = rocke_b_buffer_rsrc(b, params->D, params->D_bytes);
    rocke_value_t* k_lane;
    {
        rocke_value_t* x = ci(c, c.TK * WGW_TILE);
        rocke_value_t* m = rocke_b_mul(b, c.wave_k, x);
        rocke_value_t* a = rocke_b_add(b, c.k_blk0, m);
        rocke_value_t* y = ci(c, 8);
        rocke_value_t* m2 = rocke_b_mul(b, lhalf, y);
        k_lane = rocke_b_add(b, a, m2);
    }
    rocke_value_t* c_lane;
    {
        rocke_value_t* x = ci(c, c.TC * WGW_TILE);
        rocke_value_t* m = rocke_b_mul(b, c.wave_c, x);
        rocke_value_t* a = rocke_b_add(b, c.c_blk0, m);
        c_lane = rocke_b_add(b, a, l16);
    }
    for(int tc = 0; tc < c.TC; ++tc)
    {
        rocke_value_t* cc;
        {
            rocke_value_t* x = ci(c, WGW_TILE * tc);
            cc = rocke_b_add(b, c_lane, x);
        }
        rocke_value_t* c_ok = rocke_b_cmp_lt(b, cc, p_C);
        for(int tk = 0; tk < c.TK; ++tk)
        {
            for(int e = 0; e < 8; ++e)
            {
                rocke_value_t* k;
                {
                    rocke_value_t* x = ci(c, WGW_TILE * tk + e);
                    k = rocke_b_add(b, k_lane, x);
                }
                rocke_value_t* ok;
                {
                    rocke_value_t* lt = rocke_b_cmp_lt(b, k, p_K);
                    ok = rocke_b_land(b, c_ok, lt);
                }
                rocke_value_t* k_row;
                {
                    rocke_value_t* x = ci(c, c.KH * c.KW);
                    k_row = rocke_b_mul(b, k, x);
                }
                rocke_if_t gate = rocke_b_scf_if(b, ok);
                rocke_b_region_enter(b, gate.then_region);
                for(int r = 0; r < c.KH; ++r)
                {
                    for(int s = 0; s < c.KW; ++s)
                    {
                        rocke_value_t* off;
                        {
                            rocke_value_t* x = ci(c, r * c.KW + s);
                            rocke_value_t* a = rocke_b_add(b, k_row, x);
                            rocke_value_t* m = rocke_b_mul(b, a, p_C);
                            off = rocke_b_add(b, m, cc);
                        }
                        rocke_value_t* val
                            = rocke_b_vec_extract(b, accs[acc_i(c, r, s, tk, tc)], e);
                        rocke_value_t* x4 = ci(c, 4);
                        rocke_value_t* voff = rocke_b_mul(b, off, x4);
                        rocke_b_buffer_atomic_fadd_f32(
                            b, d_rsrc, voff, c.c0, val, ROCKE_CPOL_ATOMIC_CASCADE_DEV);
                    }
                }
                rocke_b_region_leave(b);
            }
        }
    }
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_direct_wgrad_wmma_new(rocke_ir_builder_t* b,
                                                      const rocke_direct_wgrad_wmma_spec_t* spec,
                                                      const char* arch)
{
    return rocke_dwg::build_new(
        b,
        spec != NULL,
        "build_direct_wgrad_wmma",
        [&](char* out, size_t cap) { return rocke_direct_wgrad_wmma_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_direct_wgrad_wmma(b, spec, arch); });
}

rocke_status_t rocke_direct_wgrad_wmma_lower_to_llvm(const rocke_direct_wgrad_wmma_spec_t* spec,
                                                     const char* arch,
                                                     rocke_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap)
{
    if(arch == NULL)
        arch = "gfx1250";
    return rocke_dwg::lower_to_llvm(
        spec != NULL,
        arch,
        flavor,
        out_ll,
        err,
        err_cap,
        "build_direct_wgrad_wmma failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_direct_wgrad_wmma_new(b, spec, arch); });
}
