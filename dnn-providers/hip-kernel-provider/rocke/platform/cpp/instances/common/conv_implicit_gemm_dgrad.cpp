// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C++ port of the implicit-GEMM backward-data convolution spec, validation,
 * descriptors, and tilde decomposition helpers
 * (rocke/instances/common/conv_implicit_gemm_dgrad.py).
 *
 * GEMM orientation (dgrad):
 *   M     = N*Hi*Wi       (input spatial positions)
 *   N_dg  = C             (input channels per group)
 *   K_dg  = Y*X*K         (filter spatial x output channels, reduction)
 *
 * Operand roles:
 *   A = dY (NHWK, output gradient)
 *   B = W  (KYXC, weights)
 *   D = dX (NHWC, input gradient)
 */
#include "rocke/instance_conv_implicit_gemm_dgrad.h"

#include <cmath> /* ceil, log2 */
#include <cstdio> /* snprintf */
#include <cstdlib> /* malloc, free */
#include <cstring> /* strcmp, memset */

#include "rocke/arena.h" /* rocke_arena_strdup */
#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_t, rocke_mmaop_t */
#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom, rocke_c_warp_params, decode */
#include "rocke/helper_rocke.helpers.distribution.h" /* tile distribution */
#include "rocke/helper_rocke.helpers.epilogues.h" /* DirectEpilogue, CShuffleEpilogue, WarpGrid */
#include "rocke/helper_rocke.helpers.grid.h" /* chiplet_aware_super_tile */
#include "rocke/helper_rocke.helpers.loads.h" /* CoalescedTileLoader */
#include "rocke/helper_rocke.helpers.mfma_gemm_inner.h" /* decode_mfma_lanes */
#include "rocke/helper_rocke.helpers.schedule.h" /* SchedulePolicy */
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_implicit_gemm.h" /* rocke_conv_acc_epilogue_default */
#include "rocke/instance_conv_implicit_gemm_internal.h" /* rocke_conv_build_ctx_t, phase fns */
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */
#include "rocke/lower_llvm.h"

// ---------------------------------------------------------------------------
// Pure-arithmetic helpers
// ---------------------------------------------------------------------------

static int _ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}
static int _floor_div(int a, int b)
{
    return a / b;
}
static int _gcd(int a, int b)
{
    while(b)
    {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}
static int _max(int a, int b)
{
    return a > b ? a : b;
}
static int _min(int a, int b)
{
    return a < b ? a : b;
}

/* Map a dtype string ("fp16", "bf16", "fp32") to the corresponding IR scalar type. */
static const rocke_type_t* _dtype_to_ir(const char* dtype)
{
    if(dtype && strcmp(dtype, "bf16") == 0)
        return rocke_bf16();
    if(dtype && strcmp(dtype, "fp32") == 0)
        return rocke_f32();
    return rocke_f16(); /* fp16 and default */
}

// ---------------------------------------------------------------------------
// Spec default + property accessors  (Python DgradConvSpec @property)
// ---------------------------------------------------------------------------

rocke_dgrad_conv_spec_t rocke_dgrad_conv_spec_default(void)
{
    rocke_dgrad_conv_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = "conv_igemm_dgrad";
    s.dtype_a = "fp16";
    s.dtype_b = "fp16";
    s.dtype_d = "fp16";
    s.dtype_acc = "fp32";
    s.tile_m = 64;
    s.tile_n = 64;
    s.tile_k = 64;
    s.warp_m = 2;
    s.warp_n = 2;
    s.warp_tile_m = 32;
    s.warp_tile_n = 32;
    s.warp_tile_k = 16;
    s.wave_size = 64;
    s.pipeline = "mem";
    s.epilogue = "default";
    s.chiplet_wgm = 8;
    s.chiplet_num_xcds = 8;
    s.chiplet_chunk_size = 64;
    s.acc_epilogue = rocke_conv_acc_epilogue_default();
    s.split_k = 1;
    s.num_load_waves = 4;
    return s;
}

int rocke_dgrad_conv_spec_block_size(const rocke_dgrad_conv_spec_t* s)
{
    return s->warp_m * s->warp_n * s->wave_size;
}

int rocke_dgrad_conv_spec_launch_block_size(const rocke_dgrad_conv_spec_t* s)
{
    int bs = rocke_dgrad_conv_spec_block_size(s);
    if(s->pipeline && strcmp(s->pipeline, "wavelet") == 0)
        return bs + s->num_load_waves * s->wave_size;
    return bs;
}

int rocke_dgrad_conv_spec_k_atoms_per_tile_k(const rocke_dgrad_conv_spec_t* s)
{
    return s->warp_tile_k > 0 ? s->tile_k / s->warp_tile_k : 0;
}

int rocke_dgrad_conv_spec_mfmas_per_warp_m(const rocke_dgrad_conv_spec_t* s)
{
    int d = s->warp_m * s->warp_tile_m;
    return d > 0 ? s->tile_m / d : 0;
}

int rocke_dgrad_conv_spec_mfmas_per_warp_n(const rocke_dgrad_conv_spec_t* s)
{
    int d = s->warp_n * s->warp_tile_n;
    return d > 0 ? s->tile_n / d : 0;
}

int rocke_dgrad_conv_spec_dg_M(const rocke_dgrad_conv_spec_t* s)
{
    return s->problem.N * s->problem.Hi * s->problem.Wi;
}

int rocke_dgrad_conv_spec_dg_N(const rocke_dgrad_conv_spec_t* s)
{
    return s->problem.C;
}

int rocke_dgrad_conv_spec_dg_K(const rocke_dgrad_conv_spec_t* s)
{
    return s->problem.Y * s->problem.X * s->problem.K;
}

int rocke_dgrad_conv_spec_dg_K_padded(const rocke_dgrad_conv_spec_t* s)
{
    int sk = s->split_k > 1 ? s->split_k : 1;
    int stride = s->tile_k * sk;
    int k = rocke_dgrad_conv_spec_dg_K(s);
    return _ceil_div(k, stride) * stride;
}

bool rocke_dgrad_conv_spec_is_strided(const rocke_dgrad_conv_spec_t* s)
{
    return s->problem.sH != 1 || s->problem.sW != 1 || s->problem.dH != 1 || s->problem.dW != 1;
}

bool rocke_dgrad_conv_spec_needs_atomic(const rocke_dgrad_conv_spec_t* s)
{
    /* split_k>1: multiple CTAs accumulate into the same dX elements via atomic_add.
     * Tilde sub-GEMMs with split_k=1 write to disjoint (n,hi,wi) positions and
     * use direct buffer_store — no atomics needed. */
    return s->split_k > 1;
}

rocke_status_t
    rocke_dgrad_conv_spec_kernel_name(const rocke_dgrad_conv_spec_t* s, char* out, size_t out_cap)
{
    if(!out || out_cap < 16)
        return ROCKE_ERR_VALUE;
    char prob[128];
    rocke_conv_problem_short(&s->problem, prob, sizeof(prob), NULL);
    int n = snprintf(out,
                     out_cap,
                     "%s_%s_t%dx%dx%d_w%dx%d_a%dx%dx%d_%s_%s",
                     s->name ? s->name : "conv_igemm_dgrad",
                     prob,
                     s->tile_m,
                     s->tile_n,
                     s->tile_k,
                     s->warp_m,
                     s->warp_n,
                     s->warp_tile_m,
                     s->warp_tile_n,
                     s->warp_tile_k,
                     s->pipeline ? s->pipeline : "mem",
                     s->epilogue ? s->epilogue : "default");
    if(s->split_k > 1)
    {
        int pos = n;
        n += snprintf(out + pos, out_cap - pos, "_spk%d", s->split_k);
    }
    if(s->async_dma)
    {
        int pos = n;
        n += snprintf(out + pos, out_cap - pos, "_async");
    }
    if(n >= (int)out_cap)
        return ROCKE_ERR_VALUE;
    return ROCKE_OK;
}

// ---------------------------------------------------------------------------
// Arch-aware spec validation (Python is_valid_dgrad_spec)
// ---------------------------------------------------------------------------

bool rocke_dgrad_conv_is_valid_spec(const rocke_dgrad_conv_spec_t* s,
                                    const char* arch,
                                    char* reason,
                                    size_t reason_cap)
{
    if(!arch)
        arch = "gfx950";
    if(!reason || reason_cap < 4)
        return false;

    const rocke_conv_problem_t* p = &s->problem;

    if(p->is_3d)
    {
        snprintf(reason, reason_cap, "dgrad only supports 2-D convolution currently");
        return false;
    }

    if(s->tile_m % (s->warp_m * s->warp_tile_m))
    {
        snprintf(reason, reason_cap, "tile_m not divisible by warp_m * warp_tile_m");
        return false;
    }
    if(s->tile_n % (s->warp_n * s->warp_tile_n))
    {
        snprintf(reason, reason_cap, "tile_n not divisible by warp_n * warp_tile_n");
        return false;
    }
    if(s->tile_k % s->warp_tile_k)
    {
        snprintf(reason, reason_cap, "tile_k not divisible by warp_tile_k");
        return false;
    }

    /* Arch target lookup (Python: ArchTarget.from_gfx). */
    const rocke_arch_target_t* tgt = rocke_arch_target_from_gfx(arch);
    if(!tgt)
    {
        snprintf(reason, reason_cap, "unknown arch %s", arch);
        return false;
    }

    /* block_size vs hardware cap (Python: block_size > target.max_threads_per_block). */
    int bs = rocke_dgrad_conv_spec_block_size(s);
    int max_tpb = rocke_arch_max_threads_per_block(tgt);
    if(bs > max_tpb)
    {
        snprintf(reason, reason_cap, "block_size %d > %d (hardware cap) on %s", bs, max_tpb, arch);
        return false;
    }

    /* vector_size_c > 1 incompatible with default epilogue — except when
     * split_k > 1 (atomic) or stride > 1 (tilde non-atomic direct) ignores it. */
    {
        bool is_strided = rocke_dgrad_conv_spec_is_strided(s);
        if(s->vector_size_c > 1 && strcmp(s->epilogue, "default") == 0 && s->split_k <= 1
           && !is_strided)
        {
            snprintf(reason,
                     reason_cap,
                     "default epilogue is not supported with vector size c: %d",
                     s->vector_size_c);
            return false;
        }
    }

    /* wave_size must match arch (Python: spec.wave_size != target.wave_size). */
    if(s->wave_size != tgt->wave_size)
    {
        snprintf(reason,
                 reason_cap,
                 "spec wave_size %d != %s wave_size %d",
                 s->wave_size,
                 arch,
                 tgt->wave_size);
        return false;
    }

    const char* family = (tgt->wave_size == 32) ? "wmma" : "mma";

    /* split_k range and arch/dtype gating (Python: sk checks). */
    int sk = s->split_k;
    if(sk < -1 || sk == 0)
    {
        snprintf(reason, reason_cap, "split_k must be -1 (auto), 1, or >1 (got %d)", sk);
        return false;
    }
    if(sk > 1 && strcmp(family, "mma") != 0)
    {
        snprintf(
            reason, reason_cap, "split_k > 1 is CDNA-only (got family %s on %s)", family, arch);
        return false;
    }
    if(sk > 1)
    {
        /* Accept fp32, bf16, fp16 for atomic accumulation. */
        const char* dd = s->dtype_d;
        int ok_dtype
            = (strcmp(dd, "fp32") == 0 || strcmp(dd, "bf16") == 0 || strcmp(dd, "fp16") == 0);
        if(!ok_dtype)
        {
            snprintf(
                reason, reason_cap, "split_k > 1 requires dtype_d in fp32/bf16/fp16 (got %s)", dd);
            return false;
        }
        /* Even-C constraint for packed bf16/fp16 atomics. */
        int packed = (strcmp(dd, "bf16") == 0 || strcmp(dd, "fp16") == 0);
        if(packed && p->C % 2 != 0)
        {
            snprintf(reason,
                     reason_cap,
                     "split_k > 1 with dtype_d=%s requires even C (got C=%d)",
                     dd,
                     p->C);
            return false;
        }
    }

    /* MMA shape availability (Python: target.mma.has_shape). */
    if(!rocke_mma_catalog_has_shape(&tgt->mma,
                                    family,
                                    s->dtype_a,
                                    s->dtype_b,
                                    "fp32",
                                    s->warp_tile_m,
                                    s->warp_tile_n,
                                    s->warp_tile_k))
    {
        snprintf(reason,
                 reason_cap,
                 "unsupported %s warp_tile %dx%dx%d on %s",
                 s->dtype_a,
                 s->warp_tile_m,
                 s->warp_tile_n,
                 s->warp_tile_k,
                 arch);
        return false;
    }

    /* LDS budget (Python: target.fits_lds check). */
    int ab_dtype_bytes = (strcmp(s->dtype_a, "fp32") == 0) ? 4 : 2;
    /* Simplified LDS estimate (no cshuffle for dgrad, pipeline="mem" only in practice). */
    long a_lds = (long)s->tile_m * s->tile_k * ab_dtype_bytes;
    long b_lds = (long)s->tile_n * s->tile_k * ab_dtype_bytes;
    long total_lds = a_lds + b_lds;
    if(!rocke_arch_fits_lds(tgt, total_lds))
    {
        snprintf(reason,
                 reason_cap,
                 "LDS budget %ld bytes > %d cap on %s",
                 total_lds,
                 tgt->lds_capacity_bytes,
                 arch);
        return false;
    }

    /* wavelet-specific checks (Python: spec.pipeline == "wavelet" block). */
    bool is_wavelet = (s->pipeline && strcmp(s->pipeline, "wavelet") == 0);
    if(is_wavelet)
    {
        if(s->num_load_waves < 1)
        {
            snprintf(reason, reason_cap, "pipeline='wavelet' requires num_load_waves >= 1");
            return false;
        }
        if(strcmp(family, "wmma") != 0)
        {
            snprintf(reason,
                     reason_cap,
                     "pipeline='wavelet' is WMMA/gfx1250 only: on MFMA targets "
                     "the single-buffer LDS is overwritten each K iteration and load/math "
                     "waves execute sequentially rather than truly concurrently.");
            return false;
        }
        if(s->async_dma)
        {
            snprintf(reason,
                     reason_cap,
                     "pipeline='wavelet' is incompatible with async_dma=True: "
                     "the wavelet loaders are only constructed in the non-async branch "
                     "and a_wavelet_loader/b_wavelet_loader would be None at fetch time.");
            return false;
        }
        int mfmas_m = s->tile_m / (_max(s->warp_m * s->warp_tile_m, 1));
        int mfmas_n = s->tile_n / (_max(s->warp_n * s->warp_tile_n, 1));
        int dg_K = rocke_dgrad_conv_spec_dg_K(s);
        int k_iters = _ceil_div(dg_K, _max(s->tile_k, 1));
        int wmma_cost = k_iters * mfmas_m * mfmas_n;
        const int WMMA_COST_LIMIT = 4096;
        if(wmma_cost > WMMA_COST_LIMIT)
        {
            snprintf(reason,
                     reason_cap,
                     "pipeline='wavelet' unrolled WMMA count %d "
                     "(K_iters=%d x mfmas=%dx%d) exceeds compile-time limit %d; "
                     "reduce tile_k, tile_m, or tile_n",
                     wmma_cost,
                     k_iters,
                     mfmas_m,
                     mfmas_n,
                     WMMA_COST_LIMIT);
            return false;
        }
        int launch_bs = rocke_dgrad_conv_spec_launch_block_size(s);
        if(launch_bs > max_tpb)
        {
            snprintf(reason,
                     reason_cap,
                     "launch_block_size %d > %d (hardware cap) on %s",
                     launch_bs,
                     max_tpb,
                     arch);
            return false;
        }
    }

    /* WMMA-specific restrictions (Python: family == "wmma" block). */
    if(strcmp(family, "wmma") == 0)
    {
        /* gfx1250 supports 16x16x16 and 16x16x32; other WMMA supports only 16x16x16.
         * Both atoms are valid for both "mem" and "wavelet" pipelines. */
        bool atom_ok = (s->warp_tile_m == 16 && s->warp_tile_n == 16
                        && (s->warp_tile_k == 16 || s->warp_tile_k == 32));
        if(!atom_ok)
        {
            snprintf(reason,
                     reason_cap,
                     "WMMA dgrad supports 16x16x16 or 16x16x32 (got %dx%dx%d) on %s",
                     s->warp_tile_m,
                     s->warp_tile_n,
                     s->warp_tile_k,
                     arch);
            return false;
        }
        if(strcmp(s->pipeline, "mem") != 0 && strcmp(s->pipeline, "wavelet") != 0)
        {
            snprintf(reason,
                     reason_cap,
                     "WMMA dgrad supports only 'mem' or 'wavelet' pipeline (got %s) on %s",
                     s->pipeline,
                     arch);
            return false;
        }
        if(strcmp(s->epilogue, "default") != 0 && strcmp(s->epilogue, "cshuffle") != 0)
        {
            snprintf(reason,
                     reason_cap,
                     "WMMA dgrad supports 'default' and 'cshuffle' epilogues (got %s) on %s",
                     s->epilogue,
                     arch);
            return false;
        }
        bool split_k_bad = (s->split_k > 1 && !is_wavelet);
        if(s->async_dma || s->unroll_k || s->chiplet_swizzle || split_k_bad)
        {
            snprintf(reason,
                     reason_cap,
                     "WMMA dgrad does not support async_dma/unroll_k/chiplet_swizzle"
                     "/split_k>1 (non-wavelet) on %s",
                     arch);
            return false;
        }
    }

    snprintf(reason, reason_cap, "ok");
    return true;
}

// ---------------------------------------------------------------------------
// Tilde decomposition (Python lines 128-330)
// ---------------------------------------------------------------------------

rocke_tilde_decomposition_t rocke_compute_tilde(const rocke_conv_problem_t* p)
{
    rocke_tilde_decomposition_t t;
    memset(&t, 0, sizeof(t));
    t.gcd_h = _gcd(p->sH > 0 ? p->sH : 1, p->dH > 0 ? p->dH : 1);
    t.gcd_w = _gcd(p->sW > 0 ? p->sW : 1, p->dW > 0 ? p->dW : 1);
    int sH = p->sH > 0 ? p->sH : 1;
    int sW = p->sW > 0 ? p->sW : 1;
    int dH = p->dH > 0 ? p->dH : 1;
    int dW = p->dW > 0 ? p->dW : 1;
    t.y_tilde = sH / t.gcd_h;
    t.x_tilde = sW / t.gcd_w;
    t.y_dot = _ceil_div(p->Y, t.y_tilde);
    t.x_dot = _ceil_div(p->X, t.x_tilde);
    int Ho = rocke_conv_problem_ho(p);
    int Wo = rocke_conv_problem_wo(p);
    t.h_tilde = p->Y > 1 ? Ho + _ceil_div(dH * (p->Y - 1), sH) : Ho;
    t.w_tilde = p->X > 1 ? Wo + _ceil_div(dW * (p->X - 1), sW) : Wo;
    return t;
}

int rocke_enumerate_sub_gemms(const rocke_conv_problem_t* p,
                              const rocke_tilde_decomposition_t* tilde,
                              int tile_m,
                              int tile_n,
                              int tile_k,
                              int split_k,
                              rocke_sub_gemm_params_t* out,
                              int out_cap)
{
    int count = 0;
    int cumulative_tiles = 0;
    int sH = p->sH > 0 ? p->sH : 1;
    int sW = p->sW > 0 ? p->sW : 1;
    int dH = p->dH > 0 ? p->dH : 1;
    int dW = p->dW > 0 ? p->dW : 1;
    int cpg = p->C;
    int kpg = p->K;

    for(int i_yt = 0; i_yt < tilde->y_tilde; i_yt++)
    {
        for(int i_xt = 0; i_xt < tilde->x_tilde; i_xt++)
        {
            int y_dot_slice = _ceil_div(p->Y - i_yt, tilde->y_tilde);
            int x_dot_slice = _ceil_div(p->X - i_xt, tilde->x_tilde);
            if(y_dot_slice <= 0 || x_dot_slice <= 0)
                continue;

            int h_tsb = _floor_div(_max(0, p->pH - dH * (tilde->y_tilde - 1)), sH);
            int h_tse = _min(tilde->h_tilde, _ceil_div(p->pH + p->Hi - 1, sH) + 1);
            int w_tsb = _floor_div(_max(0, p->pW - dW * (tilde->x_tilde - 1)), sW);
            int w_tse = _min(tilde->w_tilde, _ceil_div(p->pW + p->Wi - 1, sW) + 1);

            int h_ts = h_tse - h_tsb;
            int w_ts = w_tse - w_tsb;
            if(h_ts <= 0 || w_ts <= 0)
                continue;

            int gemm_m = p->N * h_ts * w_ts;
            int gemm_n = cpg;
            int gemm_k = y_dot_slice * x_dot_slice * kpg;
            int num_m_tiles = _ceil_div(gemm_m, tile_m);
            int num_n_tiles = _ceil_div(gemm_n, tile_n);
            int num_tiles = num_m_tiles * num_n_tiles;
            int block_start = cumulative_tiles;
            int block_end = cumulative_tiles + num_tiles;
            cumulative_tiles = block_end;

            int sk = split_k > 1 ? split_k : 1;
            int stride_k = tile_k * sk;
            int gemm_k_padded = _ceil_div(gemm_k, stride_k) * stride_k;

            if(count < out_cap)
            {
                rocke_sub_gemm_params_t* sg = &out[count];
                memset(sg, 0, sizeof(*sg));
                sg->i_ytilde = i_yt;
                sg->i_xtilde = i_xt;
                sg->y_dot_slice = y_dot_slice;
                sg->x_dot_slice = x_dot_slice;
                sg->h_tilde_slice_begin = h_tsb;
                sg->h_tilde_slice = h_ts;
                sg->w_tilde_slice_begin = w_tsb;
                sg->w_tilde_slice = w_ts;
                sg->gemm_m = gemm_m;
                sg->gemm_n = gemm_n;
                sg->gemm_k = gemm_k;
                sg->block_start = block_start;
                sg->block_end = block_end;
                sg->a_embed_h_coeff = -(dH / tilde->gcd_h);
                sg->a_embed_w_coeff = -(dW / tilde->gcd_w);
                sg->b_y_stride = tilde->y_tilde;
                sg->b_y_offset = i_yt;
                sg->b_x_stride = tilde->x_tilde;
                sg->b_x_offset = i_xt;
                sg->d_h_stride = sH;
                sg->d_h_offset = dH * i_yt + sH * h_tsb - p->pH;
                sg->d_w_stride = sW;
                sg->d_w_offset = dW * i_xt + sW * w_tsb - p->pW;
                sg->gemm_k_padded = gemm_k_padded;
            }
            count++;
        }
    }
    return count;
}

int rocke_pack_sub_gemm_buffer(const rocke_sub_gemm_params_t* sgs,
                               int count,
                               int tile_m,
                               int tile_n,
                               int* out_buf,
                               int out_cap)
{
    int total = count * ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS;
    if(total > out_cap)
        return 0;
    for(int i = 0; i < count; i++)
    {
        const rocke_sub_gemm_params_t* sg = &sgs[i];
        int* rec = &out_buf[i * ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS];
        rec[0] = sg->block_start;
        rec[1] = _ceil_div(sg->gemm_m, tile_m);
        rec[2] = _ceil_div(sg->gemm_n, tile_n);
        rec[3] = sg->gemm_m;
        rec[4] = sg->gemm_k;
        rec[5] = sg->h_tilde_slice;
        rec[6] = sg->w_tilde_slice;
        rec[7] = sg->h_tilde_slice_begin;
        rec[8] = sg->w_tilde_slice_begin;
        rec[9] = sg->y_dot_slice;
        rec[10] = sg->x_dot_slice;
        rec[11] = sg->a_embed_h_coeff;
        rec[12] = sg->a_embed_w_coeff;
        rec[13] = sg->b_y_stride;
        rec[14] = sg->b_y_offset;
        rec[15] = sg->b_x_stride;
        rec[16] = sg->b_x_offset;
        rec[17] = sg->d_h_stride;
        rec[18] = sg->d_h_offset;
        rec[19] = sg->d_w_stride;
        rec[20] = sg->d_w_offset;
        rec[21] = sg->gemm_k_padded;
    }
    return total;
}

// ---------------------------------------------------------------------------
// Descriptor builders (Python lines 332-450)
// ---------------------------------------------------------------------------

struct rocke_tensor_descriptor* rocke_dgrad_make_dy_descriptor(rocke_ir_builder_t* b,
                                                               const rocke_conv_problem_t* p,
                                                               const char* dtype)
{
    (void)dtype; /* descriptor pins f16 internally */
    int Ho = rocke_conv_problem_ho(p);
    int Wo = rocke_conv_problem_wo(p);

    const int lengths[] = {p->N, Ho, Wo, p->K};
    const char* coord_names[] = {"n", "ho", "wo", "k_out"};
    rocke_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive(b, "dY_nhwk", lengths, 4, NULL, coord_names, 4);
    if(!desc)
        return NULL;

    /* unmerge('m' -> [n, hi, wi], dims=[N, Hi, Wi]) */
    const char* m_into[] = {"n", "hi", "wi"};
    const int m_dims[] = {p->N, p->Hi, p->Wi};
    rocke_transform_t* t_um = rocke_unmerge_magic(b, "m", m_into, 3, m_dims);

    /* embed([hi, y] -> ho, strides=[1, -1], offset=pH, lo=0, hi=Ho) */
    const char* hi_y[] = {"hi", "y"};
    const int hi_strides[] = {1, -1};
    rocke_transform_t* t_eh = rocke_embed_bounded(b, hi_y, 2, "ho", hi_strides, p->pH, 0, Ho);

    /* embed([wi, x] -> wo, strides=[1, -1], offset=pW, lo=0, hi=Wo) */
    const char* wi_x[] = {"wi", "x"};
    const int wi_strides[] = {1, -1};
    rocke_transform_t* t_ew = rocke_embed_bounded(b, wi_x, 2, "wo", wi_strides, p->pW, 0, Wo);

    /* unmerge('k_dg' -> [k_out, y, x], dims=[K, Y, X])  -- k_out outermost for split-K */
    const char* k_into[] = {"k_out", "y", "x"};
    const int k_dims[] = {p->K, p->Y, p->X};
    rocke_transform_t* t_uk = rocke_unmerge_magic(b, "k_dg", k_into, 3, k_dims);

    /* pad k_out, y, x */
    rocke_transform_t* t_pk = rocke_pad(b, "k_out", 0, p->K);
    rocke_transform_t* t_py = rocke_pad(b, "y", 0, p->Y);
    rocke_transform_t* t_px = rocke_pad(b, "x", 0, p->X);

    const rocke_transform_t* chain[] = {t_um, t_eh, t_ew, t_uk, t_pk, t_py, t_px};
    return rocke_tensor_descriptor_transform(b, desc, chain, 7);
}

struct rocke_tensor_descriptor* rocke_dgrad_make_w_descriptor(rocke_ir_builder_t* b,
                                                              const rocke_conv_problem_t* p,
                                                              const char* dtype)
{
    (void)dtype;
    const int lengths[] = {p->K, p->Y, p->X, p->C};
    const char* coord_names[] = {"k_out", "y", "x", "c"};
    rocke_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive(b, "W_kyxc", lengths, 4, NULL, coord_names, 4);
    if(!desc)
        return NULL;

    const char* k_into[] = {"k_out", "y", "x"};
    const int k_dims[] = {p->K, p->Y, p->X};
    rocke_transform_t* t_uk = rocke_unmerge_magic(b, "k_dg", k_into, 3, k_dims);
    rocke_transform_t* t_pk = rocke_pad(b, "k_out", 0, p->K);
    rocke_transform_t* t_py = rocke_pad(b, "y", 0, p->Y);
    rocke_transform_t* t_px = rocke_pad(b, "x", 0, p->X);

    const rocke_transform_t* chain[] = {t_uk, t_pk, t_py, t_px};
    return rocke_tensor_descriptor_transform(b, desc, chain, 4);
}

struct rocke_tensor_descriptor* rocke_dgrad_make_dx_descriptor(rocke_ir_builder_t* b,
                                                               const rocke_conv_problem_t* p,
                                                               const char* dtype)
{
    (void)dtype;
    const int lengths[] = {p->N, p->Hi, p->Wi, p->C};
    const char* coord_names[] = {"n", "hi", "wi", "c"};
    rocke_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive(b, "dX_nhwc", lengths, 4, NULL, coord_names, 4);
    if(!desc)
        return NULL;

    const char* m_into[] = {"n", "hi", "wi"};
    const int m_dims[] = {p->N, p->Hi, p->Wi};
    rocke_transform_t* t_um = rocke_unmerge_magic(b, "m", m_into, 3, m_dims);

    const rocke_transform_t* chain[] = {t_um};
    return rocke_tensor_descriptor_transform(b, desc, chain, 1);
}

// ===========================================================================
// MMA resolution (mirrors _dgrad_mma_family / _resolve_dgrad_op)
// ===========================================================================

static const char* _dgrad_mma_family(const char* arch)
{
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(target && target->wave_size == 32)
        return "wmma";
    return "mma";
}

static const rocke_mmaop_t*
    _resolve_dgrad_op(rocke_ir_builder_t* b, const rocke_dgrad_conv_spec_t* spec, const char* arch)
{
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(!target)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "dgrad: unknown arch '%s'", arch);
        return NULL;
    }
    const rocke_mmaop_t* op = rocke_archtarget_op_for_shape(target,
                                                            _dgrad_mma_family(arch),
                                                            spec->dtype_a,
                                                            spec->dtype_b,
                                                            "fp32",
                                                            spec->warp_tile_m,
                                                            spec->warp_tile_n,
                                                            spec->warp_tile_k);
    if(!op)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "no MMA atom for dgrad warp_tile (%d,%d,%d) on %s",
                        spec->warp_tile_m,
                        spec->warp_tile_n,
                        spec->warp_tile_k,
                        arch);
        return NULL;
    }
    return op;
}

// ===========================================================================
// Direct epilogue helpers
// ===========================================================================

struct dgrad_dx_addr_ctx
{
    rocke_tensor_descriptor_t* desc;
};

static rocke_value_t* _dgrad_dx_addr_fn(rocke_ir_builder_t* b,
                                        rocke_value_t* m_global,
                                        rocke_value_t* n_global,
                                        rocke_value_t** out_valid,
                                        void* user)
{
    struct dgrad_dx_addr_ctx* ctx = (struct dgrad_dx_addr_ctx*)user;
    const char* names[] = {"m", "c"};
    rocke_value_t* vals[] = {m_global, n_global};
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    rocke_transforms_descriptor_offset(b, ctx->desc, names, vals, 2, &off, &valid);
    if(out_valid)
        *out_valid = valid;
    return off;
}

static void _emit_dgrad_direct_epilogue(rocke_ir_builder_t* b,
                                        const rocke_dgrad_conv_spec_t* spec,
                                        rocke_value_t* const* accs,
                                        int num_accs,
                                        const rocke_warp_grid_t* grid,
                                        rocke_value_t* dx_rsrc)
{
    const rocke_conv_problem_t* p = &spec->problem;
    rocke_tensor_descriptor_t* dX_desc = rocke_dgrad_make_dx_descriptor(b, p, spec->dtype_d);

    struct dgrad_dx_addr_ctx addr_ctx;
    addr_ctx.desc = dX_desc;

    rocke_direct_epilogue_t epi;
    epi.atom
        = rocke_mfma_atom(spec->dtype_a, spec->warp_tile_m, spec->warp_tile_n, spec->warp_tile_k);
    epi.grid = *grid;
    epi.out_dtype = spec->dtype_d;

    rocke_value_t* bounds_m = rocke_b_const_i32(b, rocke_dgrad_conv_spec_dg_M(spec));
    rocke_value_t* bounds_n = rocke_b_const_i32(b, rocke_dgrad_conv_spec_dg_N(spec));

    rocke_direct_epilogue_store(
        b, &epi, accs, num_accs, _dgrad_dx_addr_fn, &addr_ctx, dx_rsrc, bounds_m, bounds_n, false);
}

// ===========================================================================
// MFMA cshuffle epilogue for stride=1 (Python _emit_dgrad_cshuffle_epilogue)
// ===========================================================================

static void _emit_dgrad_cshuffle_epilogue(rocke_ir_builder_t* b,
                                          const rocke_dgrad_conv_spec_t* spec,
                                          rocke_value_t* const* accs,
                                          int num_accs,
                                          const rocke_warp_grid_t* grid,
                                          rocke_value_t* dx_rsrc)
{
    const rocke_conv_problem_t* p = &spec->problem;
    rocke_tensor_descriptor_t* dX_desc = rocke_dgrad_make_dx_descriptor(b, p, spec->dtype_d);

    struct dgrad_dx_addr_ctx addr_ctx;
    addr_ctx.desc = dX_desc;

    // Deduce max_store_vec from C (last dim of dX NHWC) — mirrors _dgrad_store_vec().
    bool is_fp32_d = (spec->dtype_d && strcmp(spec->dtype_d, "fp32") == 0);
    int vec_candidates[] = {is_fp32_d ? 4 : 8, 4, 2, 1};
    int vec_c = 1;
    for(int v : vec_candidates)
        if(p->C % v == 0)
        {
            vec_c = v;
            break;
        }
    int max_store_vec = spec->has_vector_size_c ? spec->vector_size_c : vec_c;

    const rocke_mfma_atom_t* atom
        = rocke_mfma_atom(spec->dtype_a, spec->warp_tile_m, spec->warp_tile_n, spec->warp_tile_k);
    rocke_cshuffle_epilogue_t epi = rocke_cshuffle_epilogue_from_grid(atom, grid, max_store_vec);
    epi.out_dtype = spec->dtype_d;

    rocke_value_t* bounds_m = rocke_b_const_i32(b, rocke_dgrad_conv_spec_dg_M(spec));
    rocke_value_t* bounds_n = rocke_b_const_i32(b, rocke_dgrad_conv_spec_dg_N(spec));

    rocke_cshuffle_epilogue_store(
        b, &epi, accs, num_accs, _dgrad_dx_addr_fn, &addr_ctx, dx_rsrc, bounds_m, bounds_n);
}

// ===========================================================================
// WMMA direct epilogue (Python _emit_dgrad_direct_epilogue_wmma)
// ===========================================================================

static void _emit_dgrad_direct_epilogue_wmma(rocke_ir_builder_t* b,
                                             const rocke_dgrad_conv_spec_t* spec,
                                             const rocke_mmaop_t* op,
                                             rocke_value_t* const* accs,
                                             int num_accs,
                                             rocke_value_t* warp_m_idx,
                                             rocke_value_t* warp_n_idx,
                                             rocke_value_t* lane,
                                             rocke_value_t* block_m_off,
                                             rocke_value_t* block_n_off,
                                             rocke_value_t* dx_rsrc,
                                             rocke_value_t* c0)
{
    (void)num_accs;
    const rocke_conv_problem_t* p = &spec->problem;
    int mfmas_m = rocke_dgrad_conv_spec_mfmas_per_warp_m(spec);
    int mfmas_n = rocke_dgrad_conv_spec_mfmas_per_warp_n(spec);
    int dg_M = rocke_dgrad_conv_spec_dg_M(spec);
    int dg_N = rocke_dgrad_conv_spec_dg_N(spec);

    bool is_fp32_out = (spec->dtype_d && strcmp(spec->dtype_d, "fp32") == 0);
    bool is_bf16_out = (spec->dtype_d && strcmp(spec->dtype_d, "bf16") == 0);
    int elem_bytes = is_fp32_out ? 4 : 2;

    rocke_value_t* warp_m_off
        = rocke_b_mul(b, warp_m_idx, rocke_b_const_i32(b, mfmas_m * spec->warp_tile_m));
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, warp_n_idx, rocke_b_const_i32(b, mfmas_n * spec->warp_tile_n));
    rocke_value_t* c_M = rocke_b_const_i32(b, dg_M);
    rocke_value_t* c_N = rocke_b_const_i32(b, dg_N);

    rocke_tensor_descriptor_t* dX_desc = rocke_dgrad_make_dx_descriptor(b, p, spec->dtype_d);

    int flat = 0;
    for(int mi = 0; mi < mfmas_m; mi++)
    {
        for(int ni = 0; ni < mfmas_n; ni++)
        {
            rocke_value_t* acc = accs[flat++];
            rocke_value_t* _bwm = rocke_b_add(b, block_m_off, warp_m_off);
            rocke_value_t* atom_m_off
                = rocke_b_add(b, _bwm, rocke_b_const_i32(b, mi * spec->warp_tile_m));
            rocke_value_t* _bwn = rocke_b_add(b, block_n_off, warp_n_off);
            rocke_value_t* atom_n_off
                = rocke_b_add(b, _bwn, rocke_b_const_i32(b, ni * spec->warp_tile_n));

            const rocke_arch_layout_map_t* c_map = rocke_mmaop_c_layout(op, b);
            for(int i = 0; i < op->c_frag_len; i++)
            {
                rocke_value_t* row_off;
                rocke_value_t* col_off;
                rocke_arch_layout_map_coord(c_map, b, lane, i, &row_off, &col_off);

                rocke_value_t* m_val = rocke_b_add(b, atom_m_off, row_off);
                rocke_value_t* n_val = rocke_b_add(b, atom_n_off, col_off);
                rocke_value_t* m_ok = rocke_b_cmp_lt(b, m_val, c_M);
                rocke_value_t* n_ok = rocke_b_cmp_lt(b, n_val, c_N);
                rocke_value_t* ok = rocke_b_land(b, m_ok, n_ok);

                rocke_value_t* v_f32 = rocke_b_vec_extract(b, acc, i);
                const char* dx_names[] = {"m", "c"};
                rocke_value_t* dx_vals[] = {m_val, n_val};
                rocke_value_t* dx_off_elems = NULL;
                rocke_value_t* dx_valid = NULL;
                rocke_transforms_descriptor_offset(
                    b, dX_desc, dx_names, dx_vals, 2, &dx_off_elems, &dx_valid);
                rocke_value_t* dx_off_bytes
                    = rocke_b_mul(b, dx_off_elems, rocke_b_const_i32(b, elem_bytes));
                rocke_value_t* safe_off
                    = rocke_b_select(b, ok, dx_off_bytes, rocke_b_const_i32(b, 0x7FFFFFFF));

                if(is_fp32_out)
                    rocke_b_buffer_store_f32(b, dx_rsrc, safe_off, c0, v_f32);
                else if(is_bf16_out)
                    rocke_b_buffer_store_bf16(
                        b, dx_rsrc, safe_off, c0, rocke_b_trunc_f32_to_bf16(b, v_f32));
                else
                    rocke_b_buffer_store_f16(
                        b, dx_rsrc, safe_off, c0, rocke_b_trunc_f32_to_f16(b, v_f32));
            }
        }
    }
}

// ===========================================================================
// LDS layout for dgrad (mirrors spec.effective_lds_layout())
// ===========================================================================

static rocke_conv_lds_layout_t _dgrad_effective_lds_layout(const rocke_dgrad_conv_spec_t* spec)
{
    rocke_conv_lds_layout_t l;
    memset(&l, 0, sizeof(l));
    l.logical_cols = spec->tile_k;
    if(spec->has_lds_k_pad)
        l.k_pad = spec->lds_k_pad;
    else
        l.k_pad = (spec->tile_k >= 16) ? 8 : 0;
    if(spec->async_dma)
        l.k_pad = 0;
    l.row_stride = l.logical_cols + l.k_pad;
    l.swizzle = NULL;
    l.requires_packed_async = false;
    return l;
}

// ===========================================================================
// Tilde binary search (Python _emit_binary_search, lines 1324-1352)
// ===========================================================================

static rocke_value_t* _emit_binary_search(rocke_ir_builder_t* b,
                                          rocke_value_t* flat_block_id,
                                          rocke_value_t* sub_gemm_buf,
                                          int num_sub_gemms)
{
    rocke_value_t* lo = rocke_b_const_i32(b, 0);
    rocke_value_t* hi = rocke_b_const_i32(b, num_sub_gemms);
    rocke_value_t* c_record_stride = rocke_b_const_i32(b, ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS);

    int max_iters = (int)ceil(log2(_max(num_sub_gemms, 2))) + 1;
    for(int i = 0; i < max_iters; i++)
    {
        rocke_value_t* _mid_sum = rocke_b_add(b, lo, hi);
        rocke_value_t* mid = rocke_b_div(b, _mid_sum, rocke_b_const_i32(b, 2));
        rocke_value_t* mid_off = rocke_b_mul(b, mid, c_record_stride);
        rocke_value_t* mid_block_start = rocke_b_global_load_i32(b, sub_gemm_buf, mid_off, 4);
        rocke_value_t* take_lo = rocke_b_cmp_le(b, mid_block_start, flat_block_id);
        lo = rocke_b_select(b, take_lo, mid, lo);
        hi = rocke_b_select(b, take_lo, hi, mid);
    }
    return lo;
}

static rocke_value_t* _emit_load_record_field(rocke_ir_builder_t* b,
                                              rocke_value_t* sub_gemm_buf,
                                              rocke_value_t* sg_idx,
                                              int field_idx)
{
    rocke_value_t* base
        = rocke_b_mul(b, sg_idx, rocke_b_const_i32(b, ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS));
    rocke_value_t* offset = rocke_b_add(b, base, rocke_b_const_i32(b, field_idx));
    return rocke_b_global_load_i32(b, sub_gemm_buf, offset, 4);
}

// ===========================================================================
// Tilde atomic epilogue (Python _emit_dgrad_tilde_atomic_epilogue)
// ===========================================================================

static void _emit_dgrad_tilde_atomic_epilogue(rocke_ir_builder_t* b,
                                              const rocke_dgrad_conv_spec_t* spec,
                                              const rocke_mfma_atom_t* atom,
                                              rocke_value_t* const* accs,
                                              int num_accs,
                                              rocke_value_t* warp_m_idx,
                                              rocke_value_t* warp_n_idx,
                                              rocke_value_t* lane,
                                              rocke_value_t* block_m_off,
                                              rocke_value_t* block_n_off,
                                              rocke_value_t* dx_ptr,
                                              int c_per_lane,
                                              rocke_value_t* gemm_m,
                                              rocke_value_t* gemm_n,
                                              rocke_value_t* h_tilde_slice,
                                              rocke_value_t* w_tilde_slice,
                                              rocke_value_t* d_h_stride,
                                              rocke_value_t* d_h_offset,
                                              rocke_value_t* d_w_stride,
                                              rocke_value_t* d_w_offset,
                                              rocke_value_t* c_Hi,
                                              rocke_value_t* c_Wi,
                                              rocke_value_t* c_C)
{
    (void)num_accs;
    int mfmas_m = rocke_dgrad_conv_spec_mfmas_per_warp_m(spec);
    int mfmas_n = rocke_dgrad_conv_spec_mfmas_per_warp_n(spec);
    bool is_fp32 = (spec->dtype_d && strcmp(spec->dtype_d, "fp32") == 0);
    bool is_bf16 = (spec->dtype_d && strcmp(spec->dtype_d, "bf16") == 0);

    rocke_value_t* warp_m_off
        = rocke_b_mul(b, warp_m_idx, rocke_b_const_i32(b, mfmas_m * spec->warp_tile_m));
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, warp_n_idx, rocke_b_const_i32(b, mfmas_n * spec->warp_tile_n));
    rocke_value_t* block_warp_m_off = rocke_b_add(b, block_m_off, warp_m_off);
    rocke_value_t* block_warp_n_off = rocke_b_add(b, block_n_off, warp_n_off);

    int kc_m0, kc_mlane, kc_m1, kc_nlane;
    rocke_c_warp_params(atom, &kc_m0, &kc_mlane, &kc_m1, &kc_nlane);

    rocke_tile_distribution_encoding_t* enc = rocke_make_c_warp_dstr_encoding(b, atom);
    const rocke_tile_distribution_t* dist = rocke_make_static_tile_distribution(b, enc);

    rocke_value_t* c_nlane = rocke_b_const_i32(b, kc_nlane);
    rocke_value_t* n_in_atom = rocke_b_mod(b, lane, c_nlane);
    rocke_value_t* m_blk = rocke_b_div(b, lane, c_nlane);

    rocke_value_t* rows[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* cols[ROCKE_CONV_MAX_ACCS];
    for(int i = 0; i < c_per_lane; i++)
    {
        rocke_value_t* ys[2] = {rocke_b_const_i32(b, i / kc_m1), rocke_b_const_i32(b, i % kc_m1)};
        rocke_value_t* p_lane_arr[2] = {m_blk, n_in_atom};
        rocke_value_t* const* ps_arr[1] = {p_lane_arr};
        int ps_counts[1] = {2};
        rocke_value_t* x_out[2];
        rocke_tile_distribution_calculate_x(b, dist, ys, 2, ps_arr, ps_counts, 1, x_out, 2);
        rows[i] = x_out[0];
        cols[i] = x_out[1];
    }

    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* hw_tilde = rocke_b_mul(b, h_tilde_slice, w_tilde_slice);

    int flat = 0;
    for(int mi = 0; mi < mfmas_m; mi++)
    {
        rocke_value_t* atom_m_base
            = rocke_b_add(b, block_warp_m_off, rocke_b_const_i32(b, mi * spec->warp_tile_m));
        for(int ni = 0; ni < mfmas_n; ni++)
        {
            rocke_value_t* acc = accs[flat++];
            rocke_value_t* atom_n_base
                = rocke_b_add(b, block_warp_n_off, rocke_b_const_i32(b, ni * spec->warp_tile_n));

            for(int i = 0; i < c_per_lane; i++)
            {
                rocke_value_t* c_m = rocke_b_add(b, atom_m_base, rows[i]);
                rocke_value_t* c_n = rocke_b_add(b, atom_n_base, cols[i]);

                // Decompose c_m -> (n, htl, wtl)
                rocke_value_t* n_val = rocke_b_div(b, c_m, hw_tilde);
                rocke_value_t* m_rem = rocke_b_mod(b, c_m, hw_tilde);
                rocke_value_t* htl = rocke_b_div(b, m_rem, w_tilde_slice);
                rocke_value_t* wtl = rocke_b_mod(b, m_rem, w_tilde_slice);

                // hi = htl * d_h_stride + d_h_offset
                rocke_value_t* hi = rocke_b_add(b, rocke_b_mul(b, htl, d_h_stride), d_h_offset);
                rocke_value_t* wi = rocke_b_add(b, rocke_b_mul(b, wtl, d_w_stride), d_w_offset);

                // Bounds: c_m < gemm_m, c_n < gemm_n, 0<=hi<Hi, 0<=wi<Wi
                rocke_value_t* m_ok = rocke_b_cmp_lt(b, c_m, gemm_m);
                rocke_value_t* n_ok = rocke_b_cmp_lt(b, c_n, gemm_n);
                rocke_value_t* hi_ge = rocke_b_cmp_ge(b, hi, c0);
                rocke_value_t* hi_lt = rocke_b_cmp_lt(b, hi, c_Hi);
                rocke_value_t* hi_ok = rocke_b_land(b, hi_ge, hi_lt);
                rocke_value_t* wi_ge = rocke_b_cmp_ge(b, wi, c0);
                rocke_value_t* wi_lt = rocke_b_cmp_lt(b, wi, c_Wi);
                rocke_value_t* wi_ok = rocke_b_land(b, wi_ge, wi_lt);
                rocke_value_t* _mn_ok = rocke_b_land(b, m_ok, n_ok);
                rocke_value_t* _hw_ok = rocke_b_land(b, hi_ok, wi_ok);
                rocke_value_t* ok = rocke_b_land(b, _mn_ok, _hw_ok);

                // NHWC offset: ((n * Hi + hi) * Wi + wi) * C + c
                rocke_value_t* _nhwc0 = rocke_b_mul(b, n_val, c_Hi);
                rocke_value_t* _nhwc1 = rocke_b_add(b, _nhwc0, hi);
                rocke_value_t* _nhwc2 = rocke_b_mul(b, _nhwc1, c_Wi);
                rocke_value_t* _nhwc3 = rocke_b_add(b, _nhwc2, wi);
                rocke_value_t* _nhwc4 = rocke_b_mul(b, _nhwc3, c_C);
                rocke_value_t* dx_offset = rocke_b_add(b, _nhwc4, c_n);

                rocke_value_t* val_f32 = rocke_b_vec_extract(b, acc, i);
                rocke_if_t if_ok = rocke_b_scf_if(b, ok);
                rocke_b_region_enter(b, if_ok.then_region);
                {
                    if(is_fp32)
                    {
                        rocke_b_global_atomic_add(b, dx_ptr, dx_offset, val_f32, NULL);
                    }
                    else
                    {
                        rocke_value_t* val_cvt = is_bf16 ? rocke_b_trunc_f32_to_bf16(b, val_f32)
                                                         : rocke_b_trunc_f32_to_f16(b, val_f32);
                        rocke_value_t* zero_f32 = rocke_b_const_f32(b, 0.0f);
                        rocke_value_t* zero_cvt = is_bf16 ? rocke_b_trunc_f32_to_bf16(b, zero_f32)
                                                          : rocke_b_trunc_f32_to_f16(b, zero_f32);
                        rocke_value_t* c_n_is_odd = rocke_b_mod(b, c_n, rocke_b_const_i32(b, 2));
                        rocke_value_t* is_odd
                            = rocke_b_cmp_ne(b, c_n_is_odd, rocke_b_const_i32(b, 0));
                        rocke_value_t* c_n_even = rocke_b_sub(b, c_n, c_n_is_odd);
                        rocke_value_t* _oe0 = rocke_b_mul(b, n_val, c_Hi);
                        rocke_value_t* _oe1 = rocke_b_add(b, _oe0, hi);
                        rocke_value_t* _oe2 = rocke_b_mul(b, _oe1, c_Wi);
                        rocke_value_t* _oe3 = rocke_b_add(b, _oe2, wi);
                        rocke_value_t* _oe4 = rocke_b_mul(b, _oe3, c_C);
                        rocke_value_t* off_even = rocke_b_add(b, _oe4, c_n_even);
                        rocke_value_t* v_even = rocke_b_select(b, is_odd, zero_cvt, val_cvt);
                        rocke_value_t* v_odd = rocke_b_select(b, is_odd, val_cvt, zero_cvt);
                        rocke_value_t* comps[2] = {v_even, v_odd};
                        rocke_value_t* vec = rocke_b_vec_pack(b, comps, 2, val_cvt->type);
                        if(is_bf16)
                            rocke_b_global_atomic_add_pk_bf16(b, dx_ptr, off_even, vec, NULL);
                        else
                            rocke_b_global_atomic_add_pk_f16(b, dx_ptr, off_even, vec, NULL);
                    }
                }
                rocke_b_region_leave(b);
            }
        }
    }
}

// ===========================================================================
// Tilde dgrad kernel builder (Python _build_tilde_dgrad, lines 1366-1757)
// ===========================================================================

// ===========================================================================
// Tilde descriptor closure context types + callbacks
// ===========================================================================

struct tilde_dy_ctx_t
{
    rocke_value_t* block_m_off;
    rocke_value_t* k_off;
    rocke_value_t* rec_x_dot_slice; // xdot_slice for yx_rem decomposition
    rocke_value_t* hw_tilde;
    rocke_value_t* rec_w_tilde_slice;
    rocke_value_t* rec_h_tilde_slice_begin;
    rocke_value_t* rec_w_tilde_slice_begin;
    rocke_value_t* rec_a_embed_h_coeff;
    rocke_value_t* rec_a_embed_w_coeff;
    rocke_value_t* c_Ho;
    rocke_value_t* c_Wo;
    rocke_value_t* c_K; // K_conv — innermost divisor in k_dg decomposition
    rocke_value_t* c0;
};

static rocke_value_t* _tilde_dy_descriptor(rocke_ir_builder_t* b_,
                                           rocke_value_t* row,
                                           rocke_value_t* col,
                                           rocke_value_t** out_valid,
                                           void* user)
{
    tilde_dy_ctx_t* ctx = (tilde_dy_ctx_t*)user;
    rocke_value_t* m_sub = rocke_b_add(b_, ctx->block_m_off, row);
    rocke_value_t* k_sub = rocke_b_add(b_, ctx->k_off, col);

    // k_out innermost (CK-compatible): k_sub = ydot*xdot_slice*K + xdot*K + k_out
    // Consecutive k_sub → consecutive k_out → contiguous in dY (NHWK, last dim K).
    rocke_value_t* k_out = rocke_b_mod(b_, k_sub, ctx->c_K);
    rocke_value_t* yx_rem = rocke_b_div(b_, k_sub, ctx->c_K);
    rocke_value_t* ydot = rocke_b_div(b_, yx_rem, ctx->rec_x_dot_slice);
    rocke_value_t* xdot = rocke_b_mod(b_, yx_rem, ctx->rec_x_dot_slice);

    rocke_value_t* n_val = rocke_b_div(b_, m_sub, ctx->hw_tilde);
    rocke_value_t* m_rem = rocke_b_mod(b_, m_sub, ctx->hw_tilde);
    rocke_value_t* htl = rocke_b_div(b_, m_rem, ctx->rec_w_tilde_slice);
    rocke_value_t* wtl = rocke_b_mod(b_, m_rem, ctx->rec_w_tilde_slice);

    rocke_value_t* ho_base = rocke_b_add(b_, htl, ctx->rec_h_tilde_slice_begin);
    rocke_value_t* ho_mul = rocke_b_mul(b_, ydot, ctx->rec_a_embed_h_coeff);
    rocke_value_t* ho = rocke_b_add(b_, ho_base, ho_mul);
    rocke_value_t* wo_base = rocke_b_add(b_, wtl, ctx->rec_w_tilde_slice_begin);
    rocke_value_t* wo_mul = rocke_b_mul(b_, xdot, ctx->rec_a_embed_w_coeff);
    rocke_value_t* wo = rocke_b_add(b_, wo_base, wo_mul);

    rocke_value_t* ho_ge = rocke_b_cmp_ge(b_, ho, ctx->c0);
    rocke_value_t* ho_lt = rocke_b_cmp_lt(b_, ho, ctx->c_Ho);
    rocke_value_t* ho_ok = rocke_b_land(b_, ho_ge, ho_lt);
    rocke_value_t* wo_ge = rocke_b_cmp_ge(b_, wo, ctx->c0);
    rocke_value_t* wo_lt = rocke_b_cmp_lt(b_, wo, ctx->c_Wo);
    rocke_value_t* wo_ok = rocke_b_land(b_, wo_ge, wo_lt);
    rocke_value_t* k_ok = rocke_b_cmp_lt(b_, k_out, ctx->c_K);
    rocke_value_t* valid = rocke_b_land(b_, rocke_b_land(b_, ho_ok, wo_ok), k_ok);

    rocke_value_t* n_ho = rocke_b_add(b_, rocke_b_mul(b_, n_val, ctx->c_Ho), ho);
    rocke_value_t* wo_k = rocke_b_mul(b_, ctx->c_Wo, ctx->c_K);
    rocke_value_t* hi_part = rocke_b_mul(b_, n_ho, wo_k);
    rocke_value_t* wo_part = rocke_b_add(b_, rocke_b_mul(b_, wo, ctx->c_K), k_out);
    rocke_value_t* offset = rocke_b_add(b_, hi_part, wo_part);
    rocke_value_t* safe_offset = rocke_b_select(b_, valid, offset, rocke_b_const_i32(b_, 0));

    if(out_valid)
        *out_valid = valid;
    return safe_offset;
}

struct tilde_w_ctx_t
{
    rocke_value_t* block_n_off;
    rocke_value_t* k_off;
    rocke_value_t* rec_x_dot_slice;
    rocke_value_t* rec_b_y_stride;
    rocke_value_t* rec_b_y_offset;
    rocke_value_t* rec_b_x_stride;
    rocke_value_t* rec_b_x_offset;
    rocke_value_t* c_Y;
    rocke_value_t* c_X;
    rocke_value_t* c_K;
    rocke_value_t* c_C;
    rocke_value_t* c0;
};

static rocke_value_t* _tilde_w_descriptor(rocke_ir_builder_t* b_,
                                          rocke_value_t* row,
                                          rocke_value_t* col,
                                          rocke_value_t** out_valid,
                                          void* user)
{
    tilde_w_ctx_t* ctx = (tilde_w_ctx_t*)user;
    rocke_value_t* c_val = rocke_b_add(b_, ctx->block_n_off, row);
    rocke_value_t* k_sub = rocke_b_add(b_, ctx->k_off, col);

    // Same k_out-innermost decomposition as _tilde_dy_descriptor (must match).
    // c (row axis) is stride-1 in KYXC; vectorised loads along c use vector_axis_row=true.
    rocke_value_t* k_out = rocke_b_mod(b_, k_sub, ctx->c_K);
    rocke_value_t* yx_rem = rocke_b_div(b_, k_sub, ctx->c_K);
    rocke_value_t* ydot = rocke_b_div(b_, yx_rem, ctx->rec_x_dot_slice);
    rocke_value_t* xdot = rocke_b_mod(b_, yx_rem, ctx->rec_x_dot_slice);

    rocke_value_t* y_mul = rocke_b_mul(b_, ydot, ctx->rec_b_y_stride);
    rocke_value_t* y = rocke_b_add(b_, y_mul, ctx->rec_b_y_offset);
    rocke_value_t* x_mul = rocke_b_mul(b_, xdot, ctx->rec_b_x_stride);
    rocke_value_t* x = rocke_b_add(b_, x_mul, ctx->rec_b_x_offset);

    rocke_value_t* y_lt = rocke_b_cmp_lt(b_, y, ctx->c_Y);
    rocke_value_t* x_lt = rocke_b_cmp_lt(b_, x, ctx->c_X);
    rocke_value_t* yx_ok = rocke_b_land(b_, y_lt, x_lt);
    rocke_value_t* k_ok = rocke_b_cmp_lt(b_, k_out, ctx->c_K);
    rocke_value_t* valid = rocke_b_land(b_, yx_ok, k_ok);

    rocke_value_t* k_y = rocke_b_add(b_, rocke_b_mul(b_, k_out, ctx->c_Y), y);
    rocke_value_t* k_y_x = rocke_b_add(b_, rocke_b_mul(b_, k_y, ctx->c_X), x);
    rocke_value_t* offset = rocke_b_add(b_, rocke_b_mul(b_, k_y_x, ctx->c_C), c_val);
    rocke_value_t* safe_offset = rocke_b_select(b_, valid, offset, rocke_b_const_i32(b_, 0));

    if(out_valid)
        *out_valid = valid;
    return safe_offset;
}

// ===========================================================================
// WMMA tilde direct epilogue (Python _emit_dgrad_tilde_direct_epilogue_wmma)
// ===========================================================================

static void _emit_dgrad_tilde_direct_epilogue_wmma(rocke_ir_builder_t* b,
                                                   const rocke_dgrad_conv_spec_t* spec,
                                                   const rocke_mmaop_t* op,
                                                   rocke_value_t* const* accs,
                                                   int num_accs,
                                                   rocke_value_t* warp_m_idx,
                                                   rocke_value_t* warp_n_idx,
                                                   rocke_value_t* lane,
                                                   rocke_value_t* block_m_off,
                                                   rocke_value_t* block_n_off,
                                                   rocke_value_t* dx_rsrc,
                                                   rocke_value_t* c0,
                                                   rocke_value_t* bounds_m,
                                                   rocke_value_t* bounds_n,
                                                   rocke_value_t* hw_tilde,
                                                   rocke_value_t* w_tilde_slice,
                                                   rocke_value_t* d_h_stride,
                                                   rocke_value_t* d_h_offset,
                                                   rocke_value_t* d_w_stride,
                                                   rocke_value_t* d_w_offset,
                                                   rocke_value_t* c_Hi,
                                                   rocke_value_t* c_Wi,
                                                   rocke_value_t* c_C)
{
    (void)num_accs;
    int mfmas_m = rocke_dgrad_conv_spec_mfmas_per_warp_m(spec);
    int mfmas_n = rocke_dgrad_conv_spec_mfmas_per_warp_n(spec);
    bool is_fp32 = (spec->dtype_d && strcmp(spec->dtype_d, "fp32") == 0);
    bool is_bf16 = (spec->dtype_d && strcmp(spec->dtype_d, "bf16") == 0);
    int elem_bytes = is_fp32 ? 4 : 2;

    rocke_value_t* warp_m_off
        = rocke_b_mul(b, warp_m_idx, rocke_b_const_i32(b, mfmas_m * spec->warp_tile_m));
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, warp_n_idx, rocke_b_const_i32(b, mfmas_n * spec->warp_tile_n));

    const rocke_arch_layout_map_t* c_map = rocke_mmaop_c_layout(op, b);

    int flat = 0;
    for(int mi = 0; mi < mfmas_m; mi++)
    {
        rocke_value_t* atom_m_off = rocke_b_add(b,
                                                rocke_b_add(b, block_m_off, warp_m_off),
                                                rocke_b_const_i32(b, mi * spec->warp_tile_m));
        for(int ni = 0; ni < mfmas_n; ni++)
        {
            rocke_value_t* acc = accs[flat++];
            rocke_value_t* atom_n_off = rocke_b_add(b,
                                                    rocke_b_add(b, block_n_off, warp_n_off),
                                                    rocke_b_const_i32(b, ni * spec->warp_tile_n));

            for(int i = 0; i < op->c_frag_len; i++)
            {
                rocke_value_t* row_off;
                rocke_value_t* col_off;
                rocke_arch_layout_map_coord(c_map, b, lane, i, &row_off, &col_off);

                rocke_value_t* m_val = rocke_b_add(b, atom_m_off, row_off);
                rocke_value_t* n_val = rocke_b_add(b, atom_n_off, col_off);

                // Tilde M decomposition → (n_batch, htl, wtl) → (hi, wi)
                rocke_value_t* n_batch = rocke_b_div(b, m_val, hw_tilde);
                rocke_value_t* m_rem = rocke_b_mod(b, m_val, hw_tilde);
                rocke_value_t* htl = rocke_b_div(b, m_rem, w_tilde_slice);
                rocke_value_t* wtl = rocke_b_mod(b, m_rem, w_tilde_slice);
                rocke_value_t* hi = rocke_b_add(b, rocke_b_mul(b, htl, d_h_stride), d_h_offset);
                rocke_value_t* wi = rocke_b_add(b, rocke_b_mul(b, wtl, d_w_stride), d_w_offset);

                rocke_value_t* m_ok = rocke_b_cmp_lt(b, m_val, bounds_m);
                rocke_value_t* n_ok = rocke_b_cmp_lt(b, n_val, bounds_n);
                rocke_value_t* hi_ok
                    = rocke_b_land(b, rocke_b_cmp_ge(b, hi, c0), rocke_b_cmp_lt(b, hi, c_Hi));
                rocke_value_t* wi_ok
                    = rocke_b_land(b, rocke_b_cmp_ge(b, wi, c0), rocke_b_cmp_lt(b, wi, c_Wi));
                rocke_value_t* ok
                    = rocke_b_land(b, rocke_b_land(b, m_ok, n_ok), rocke_b_land(b, hi_ok, wi_ok));

                // NHWC offset: ((n_batch*Hi + hi)*Wi + wi)*C + c
                rocke_value_t* _o0 = rocke_b_mul(b, n_batch, c_Hi);
                rocke_value_t* _o1 = rocke_b_add(b, _o0, hi);
                rocke_value_t* _o2 = rocke_b_mul(b, _o1, c_Wi);
                rocke_value_t* _o3 = rocke_b_add(b, _o2, wi);
                rocke_value_t* _o4 = rocke_b_mul(b, _o3, c_C);
                rocke_value_t* off_elems = rocke_b_add(b, _o4, n_val);
                rocke_value_t* off_bytes
                    = rocke_b_mul(b, off_elems, rocke_b_const_i32(b, elem_bytes));
                rocke_value_t* safe_off
                    = rocke_b_select(b, ok, off_bytes, rocke_b_const_i32(b, 0x7FFFFFFF));

                rocke_value_t* v_f32 = rocke_b_vec_extract(b, acc, i);
                if(is_fp32)
                    rocke_b_buffer_store_f32(b, dx_rsrc, safe_off, c0, v_f32);
                else if(is_bf16)
                    rocke_b_buffer_store_bf16(
                        b, dx_rsrc, safe_off, c0, rocke_b_trunc_f32_to_bf16(b, v_f32));
                else
                    rocke_b_buffer_store_f16(
                        b, dx_rsrc, safe_off, c0, rocke_b_trunc_f32_to_f16(b, v_f32));
            }
        }
    }
}

// ===========================================================================
// ===========================================================================
// Tilde non-atomic epilogues (split_k==1 strided convolutions)
//
// For split_k==1 the tilde decomposition guarantees each sub-GEMM writes to
// a disjoint subset of dX elements, so plain buffer_store is safe.
//
// Shared addr_fn (tilde_dx_ctx_t / _tilde_dx_addr_fn):
//   m_global  -> (n_val, htl, wtl) via hw_tilde / w_tilde_slice
//   hi        = htl * d_h_stride + d_h_offset
//   wi        = wtl * d_w_stride + d_w_offset
//   NHWC off  = ((n_val * Hi + hi) * Wi + wi) * C + n_global
//
// Two functions mirror the non-tilde (stride=1) pattern:
//   _emit_dgrad_tilde_direct_epilogue   -- scalar store, no LDS staging
//   _emit_dgrad_tilde_cshuffle_epilogue -- LDS-staged, vector_size_c-wide stores
// ===========================================================================

struct tilde_dx_ctx_t
{
    rocke_value_t* hw_tilde;
    rocke_value_t* w_tilde_slice;
    rocke_value_t* d_h_stride;
    rocke_value_t* d_h_offset;
    rocke_value_t* d_w_stride;
    rocke_value_t* d_w_offset;
    rocke_value_t* c_Hi;
    rocke_value_t* c_Wi;
    rocke_value_t* c_C;
    // c0 is NOT stored here: it is created locally inside _tilde_dx_addr_fn,
    // matching Python's closure which does c0 = b.const_i32(0) inside the fn.
};

static tilde_dx_ctx_t _make_tilde_dx_ctx(rocke_value_t* hw_tilde,
                                         rocke_value_t* w_tilde_slice,
                                         rocke_value_t* d_h_stride,
                                         rocke_value_t* d_h_offset,
                                         rocke_value_t* d_w_stride,
                                         rocke_value_t* d_w_offset,
                                         rocke_value_t* c_Hi,
                                         rocke_value_t* c_Wi,
                                         rocke_value_t* c_C)
{
    tilde_dx_ctx_t ctx;
    ctx.hw_tilde = hw_tilde;
    ctx.w_tilde_slice = w_tilde_slice;
    ctx.d_h_stride = d_h_stride;
    ctx.d_h_offset = d_h_offset;
    ctx.d_w_stride = d_w_stride;
    ctx.d_w_offset = d_w_offset;
    ctx.c_Hi = c_Hi;
    ctx.c_Wi = c_Wi;
    ctx.c_C = c_C;
    /* c0 not stored — created locally inside _tilde_dx_addr_fn */
    return ctx;
}

// Epilogue addr_fn: (m_global, n_global) -> (element offset, hw_valid).
// bounds_m / bounds_n (m < gemm_m, n < C) are handled by the epilogue caller.
static rocke_value_t* _tilde_dx_addr_fn(rocke_ir_builder_t* b,
                                        rocke_value_t* m_global,
                                        rocke_value_t* n_global,
                                        rocke_value_t** out_valid,
                                        void* user)
{
    tilde_dx_ctx_t* ctx = (tilde_dx_ctx_t*)user;

    // c0 created locally to match Python closure order (b.const_i32(0) first in fn).
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);

    // Decompose m_global -> (n_val, htl, wtl)
    rocke_value_t* n_val = rocke_b_div(b, m_global, ctx->hw_tilde);
    rocke_value_t* m_rem = rocke_b_mod(b, m_global, ctx->hw_tilde);
    rocke_value_t* htl = rocke_b_div(b, m_rem, ctx->w_tilde_slice);
    rocke_value_t* wtl = rocke_b_mod(b, m_rem, ctx->w_tilde_slice);

    rocke_value_t* hi = rocke_b_add(b, rocke_b_mul(b, htl, ctx->d_h_stride), ctx->d_h_offset);
    rocke_value_t* wi = rocke_b_add(b, rocke_b_mul(b, wtl, ctx->d_w_stride), ctx->d_w_offset);

    // hi/wi validity — explicit sequencing (sge before slt) to match Python left-to-right.
    rocke_value_t* hi_ge = rocke_b_cmp_ge(b, hi, c0);
    rocke_value_t* hi_lt = rocke_b_cmp_lt(b, hi, ctx->c_Hi);
    rocke_value_t* hi_ok = rocke_b_land(b, hi_ge, hi_lt);
    rocke_value_t* wi_ge = rocke_b_cmp_ge(b, wi, c0);
    rocke_value_t* wi_lt = rocke_b_cmp_lt(b, wi, ctx->c_Wi);
    rocke_value_t* wi_ok = rocke_b_land(b, wi_ge, wi_lt);
    rocke_value_t* hw_ok = rocke_b_land(b, hi_ok, wi_ok);

    if(out_valid)
        *out_valid = hw_ok;

    // NHWC element offset: ((n_val * Hi + hi) * Wi + wi) * C + n_global
    rocke_value_t* _o0 = rocke_b_mul(b, n_val, ctx->c_Hi);
    rocke_value_t* _o1 = rocke_b_add(b, _o0, hi);
    rocke_value_t* _o2 = rocke_b_mul(b, _o1, ctx->c_Wi);
    rocke_value_t* _o3 = rocke_b_add(b, _o2, wi);
    rocke_value_t* _o4 = rocke_b_mul(b, _o3, ctx->c_C);
    rocke_value_t* offset = rocke_b_add(b, _o4, n_global);

    // Return raw element offset (no sentinel select here).
    // DirectEpilogue/CShuffleEpilogue apply the sentinel via the out_valid flag,
    // matching stride-1 (_dgrad_dx_addr_fn) which also returns raw offset + valid.
    return offset;
}

// Scalar (per-element) store — mirrors _emit_dgrad_direct_epilogue for the tilde case.
static void _emit_dgrad_tilde_direct_epilogue(rocke_ir_builder_t* b,
                                              const rocke_dgrad_conv_spec_t* spec,
                                              const rocke_mfma_atom_t* atom,
                                              const rocke_warp_grid_t* grid,
                                              rocke_value_t* const* accs,
                                              int num_accs,
                                              rocke_value_t* dx_rsrc,
                                              rocke_value_t* bounds_m,
                                              rocke_value_t* bounds_n,
                                              rocke_value_t* hw_tilde,
                                              rocke_value_t* w_tilde_slice,
                                              rocke_value_t* d_h_stride,
                                              rocke_value_t* d_h_offset,
                                              rocke_value_t* d_w_stride,
                                              rocke_value_t* d_w_offset,
                                              rocke_value_t* c_Hi,
                                              rocke_value_t* c_Wi,
                                              rocke_value_t* c_C)
{
    tilde_dx_ctx_t ctx = _make_tilde_dx_ctx(
        hw_tilde, w_tilde_slice, d_h_stride, d_h_offset, d_w_stride, d_w_offset, c_Hi, c_Wi, c_C);
    rocke_direct_epilogue_t epi;
    epi.atom = atom;
    epi.grid = *grid;
    epi.out_dtype = spec->dtype_d;
    rocke_direct_epilogue_store(
        b, &epi, accs, num_accs, _tilde_dx_addr_fn, &ctx, dx_rsrc, bounds_m, bounds_n, false);
}

// LDS-staged wide store — mirrors _emit_dgrad_cshuffle_epilogue for the tilde case.
// The N dimension (channel index) is contiguous in dX for fixed m, so the
// cshuffle epilogue can issue vector_size_c-wide buffer stores along N.
static void _emit_dgrad_tilde_cshuffle_epilogue(rocke_ir_builder_t* b,
                                                const rocke_dgrad_conv_spec_t* spec,
                                                const rocke_mfma_atom_t* atom,
                                                const rocke_warp_grid_t* grid,
                                                rocke_value_t* const* accs,
                                                int num_accs,
                                                rocke_value_t* dx_rsrc,
                                                rocke_value_t* bounds_m,
                                                rocke_value_t* bounds_n,
                                                rocke_value_t* hw_tilde,
                                                rocke_value_t* w_tilde_slice,
                                                rocke_value_t* d_h_stride,
                                                rocke_value_t* d_h_offset,
                                                rocke_value_t* d_w_stride,
                                                rocke_value_t* d_w_offset,
                                                rocke_value_t* c_Hi,
                                                rocke_value_t* c_Wi,
                                                rocke_value_t* c_C)
{
    tilde_dx_ctx_t ctx = _make_tilde_dx_ctx(
        hw_tilde, w_tilde_slice, d_h_stride, d_h_offset, d_w_stride, d_w_offset, c_Hi, c_Wi, c_C);
    int max_store_vec
        = (spec->has_vector_size_c && spec->vector_size_c > 1) ? spec->vector_size_c : 8;
    rocke_cshuffle_epilogue_t epi = rocke_cshuffle_epilogue_from_grid(atom, grid, max_store_vec);
    epi.out_dtype = spec->dtype_d;
    rocke_cshuffle_epilogue_store(
        b, &epi, accs, num_accs, _tilde_dx_addr_fn, &ctx, dx_rsrc, bounds_m, bounds_n);
}

// ===========================================================================
// Tilde dgrad kernel builder (Python _build_tilde_dgrad, lines 1366-1757)
// ===========================================================================

static rocke_kernel_def_t*
    _build_tilde_dgrad(rocke_ir_builder_t* b, const rocke_dgrad_conv_spec_t* spec, const char* arch)
{
    const rocke_conv_problem_t* p = &spec->problem;
    rocke_tilde_decomposition_t tilde = rocke_compute_tilde(p);

    rocke_sub_gemm_params_t sub_gemms[128];
    int num_sub_gemms = rocke_enumerate_sub_gemms(
        p, &tilde, spec->tile_m, spec->tile_n, spec->tile_k, spec->split_k, sub_gemms, 128);
    if(num_sub_gemms <= 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "dgrad tilde: no non-empty sub-GEMMs");
        return NULL;
    }

    int block_m = spec->tile_m;
    int block_n = spec->tile_n;
    int block_k = spec->tile_k;

    // ---- waves_per_eu ----
    if(spec->has_waves_per_eu && b->kernel)
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);

    // ---- params ----
    const rocke_type_t* ab_ir = _dtype_to_ir(spec->dtype_a);
    const rocke_type_t* d_ir = _dtype_to_ir(spec->dtype_d);
    const rocke_type_t* ab_global = rocke_ptr_type(b, ab_ir, "global");
    const rocke_type_t* d_global = rocke_ptr_type(b, d_ir, "global");
    rocke_param_opts_t ro_opts;
    memset(&ro_opts, 0, sizeof(ro_opts));
    ro_opts.noalias = true;
    ro_opts.noalias_set = true;
    ro_opts.readonly = true;
    ro_opts.readonly_set = true;
    ro_opts.align = 16;
    ro_opts.align_set = true;

    rocke_value_t* dY = rocke_b_param(b, "dY", ab_global, &ro_opts);
    rocke_value_t* W = rocke_b_param(b, "W", ab_global, &ro_opts);

    // split_k>1 uses atomic_add (multiple blocks accumulate into same dX elements).
    // Tilde sub-GEMMs with split_k=1 use direct buffer_store (disjoint writes), so writeonly.
    bool uses_atomic_store = (spec->split_k > 1);
    rocke_param_opts_t d_opts;
    memset(&d_opts, 0, sizeof(d_opts));
    d_opts.noalias = true;
    d_opts.noalias_set = true;
    if(!uses_atomic_store)
    {
        d_opts.writeonly = true;
        d_opts.writeonly_set = true;
    }
    d_opts.align = 16;
    d_opts.align_set = true;
    rocke_value_t* dX = rocke_b_param(b, "dX", d_global, &d_opts);

    rocke_value_t* dY_bytes = rocke_b_param(b, "dY_bytes", rocke_i32(), NULL);
    rocke_value_t* W_bytes = rocke_b_param(b, "W_bytes", rocke_i32(), NULL);
    rocke_value_t* dX_bytes = rocke_b_param(b, "dX_bytes", rocke_i32(), NULL);

    // sub_gemm_buf and num_sub_gemms params
    rocke_param_opts_t buf_opts;
    memset(&buf_opts, 0, sizeof(buf_opts));
    buf_opts.noalias = true;
    buf_opts.noalias_set = true;
    buf_opts.readonly = true;
    buf_opts.readonly_set = true;
    buf_opts.align = 4;
    buf_opts.align_set = true;
    const rocke_type_t* i32_global = rocke_ptr_type(b, rocke_i32(), "global");
    rocke_value_t* sub_gemm_buf = rocke_b_param(b, "sub_gemm_buf", i32_global, &buf_opts);
    rocke_value_t* num_sub_gemms_param = rocke_b_param(b, "num_sub_gemms", rocke_i32(), NULL);
    (void)num_sub_gemms_param;

    // ---- resolve op + atom ----
    const rocke_mmaop_t* op = _resolve_dgrad_op(b, spec, arch);
    if(!op)
        return NULL;
    bool is_wmma = (op->family && strcmp(op->family, "wmma") == 0);
    const rocke_mfma_atom_t* atom
        = is_wmma ? NULL
                  : rocke_mfma_atom(
                        spec->dtype_a, spec->warp_tile_m, spec->warp_tile_n, spec->warp_tile_k);
    int a_per_lane = op->a_frag_len;
    int b_per_lane = op->b_frag_len;
    int c_per_lane = op->c_frag_len;

    // ---- 1D grid: block_id_x covers all sub-GEMMs' tiles ----
    rocke_value_t* flat_block_id = rocke_b_block_id_x(b);

    // ---- binary search ----
    rocke_value_t* sg_idx = _emit_binary_search(b, flat_block_id, sub_gemm_buf, num_sub_gemms);

    // ---- load all record fields ----
    rocke_value_t* rec_block_start = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 0);
    (void)_emit_load_record_field(b, sub_gemm_buf, sg_idx, 1); /* rec_num_m_tiles: unused */
    rocke_value_t* rec_num_n_tiles = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 2);
    rocke_value_t* rec_gemm_m = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 3);
    rocke_value_t* rec_gemm_k = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 4);
    rocke_value_t* rec_h_tilde_slice = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 5);
    rocke_value_t* rec_w_tilde_slice = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 6);
    rocke_value_t* rec_h_tilde_slice_begin = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 7);
    rocke_value_t* rec_w_tilde_slice_begin = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 8);
    (void)_emit_load_record_field(
        b, sub_gemm_buf, sg_idx, 9); /* rec_y_dot_slice: unused (k_out innermost) */
    rocke_value_t* rec_x_dot_slice = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 10);
    rocke_value_t* rec_a_embed_h_coeff = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 11);
    rocke_value_t* rec_a_embed_w_coeff = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 12);
    rocke_value_t* rec_b_y_stride = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 13);
    rocke_value_t* rec_b_y_offset = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 14);
    rocke_value_t* rec_b_x_stride = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 15);
    rocke_value_t* rec_b_x_offset = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 16);
    rocke_value_t* rec_d_h_stride = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 17);
    rocke_value_t* rec_d_h_offset = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 18);
    rocke_value_t* rec_d_w_stride = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 19);
    rocke_value_t* rec_d_w_offset = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 20);

    // ---- compute local tile indices ----
    rocke_value_t* local_flat = rocke_b_sub(b, flat_block_id, rec_block_start);
    rocke_value_t* local_m_tile = rocke_b_div(b, local_flat, rec_num_n_tiles);
    rocke_value_t* local_n_tile = rocke_b_mod(b, local_flat, rec_num_n_tiles);

    // ---- WarpGrid ----
    rocke_warp_grid_t grid;
    memset(&grid, 0, sizeof(grid));
    grid.tile_m = block_m;
    grid.tile_n = block_n;
    grid.tile_k = block_k;
    grid.warp_m = spec->warp_m;
    grid.warp_n = spec->warp_n;
    grid.warp_k = 1;
    grid.warp_tile_m = spec->warp_tile_m;
    grid.warp_tile_n = spec->warp_tile_n;
    grid.warp_tile_k = spec->warp_tile_k;
    grid.wave_size = spec->wave_size;

    int block_size = rocke_dgrad_conv_spec_block_size(spec);
    bool is_wavelet = (spec->pipeline && strcmp(spec->pipeline, "wavelet") == 0);
    int launch_block_size = rocke_dgrad_conv_spec_launch_block_size(spec);
    if(b->kernel)
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", launch_block_size);

    rocke_value_t* wave = rocke_b_const_i32(b, spec->wave_size);
    rocke_value_t* c_warps_n = rocke_b_const_i32(b, spec->warp_n);
    rocke_value_t* c_warps_n_warp_m = rocke_b_const_i32(b, spec->warp_n * spec->warp_m);
    rocke_value_t* c_tile_m = rocke_b_const_i32(b, block_m);
    rocke_value_t* c_tile_n = rocke_b_const_i32(b, block_n);
    rocke_value_t* c_tile_k = rocke_b_const_i32(b, block_k);
    (void)c_warps_n_warp_m;
    (void)c_tile_k;

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* lane = rocke_b_mod(b, tid, wave);
    rocke_value_t* warp_id = rocke_b_div(b, tid, wave);
    rocke_value_t* warp_m_idx = rocke_b_div(b, warp_id, c_warps_n);
    rocke_value_t* warp_n_idx = rocke_b_mod(b, warp_id, c_warps_n);

    grid.tid = tid;
    grid.lane = lane;
    grid.warp_id = warp_id;
    grid.warp_m_idx = warp_m_idx;
    grid.warp_n_idx = warp_n_idx;
    grid.warp_k_idx = rocke_b_const_i32(b, 0);
    grid.block_m_off = rocke_b_mul(b, rocke_b_block_id_y(b), c_tile_m);
    grid.block_n_off = rocke_b_mul(b, rocke_b_block_id_x(b), c_tile_n);
    grid.block_k_off = rocke_b_const_i32(b, 0);

    // Override block_m/n_off with local tile offsets
    rocke_value_t* block_m_off_v = rocke_b_mul(b, local_m_tile, rocke_b_const_i32(b, block_m));
    rocke_value_t* block_n_off_v = rocke_b_mul(b, local_n_tile, rocke_b_const_i32(b, block_n));
    grid.block_m_off = block_m_off_v;
    grid.block_n_off = block_n_off_v;

    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c_block_k = rocke_b_const_i32(b, block_k);

    // ---- split-K bounds ----
    bool is_split_k = spec->split_k > 1;
    rocke_value_t* k_lo;
    rocke_value_t* k_hi;
    if(is_split_k)
    {
        rocke_value_t* rec_gemm_k_padded = _emit_load_record_field(b, sub_gemm_buf, sg_idx, 21);
        rocke_value_t* c_split_k = rocke_b_const_i32(b, spec->split_k);
        rocke_value_t* k_slice = rocke_b_div(b, rec_gemm_k_padded, c_split_k);
        k_lo = rocke_b_mul(b, rocke_b_block_id_z(b), k_slice);
        k_hi = rocke_b_add(b, k_lo, k_slice);
    }
    else
    {
        k_lo = c0;
        k_hi = rec_gemm_k;
    }

    // ---- compile-time problem constants ----
    int Ho = rocke_conv_problem_ho(p);
    int Wo = rocke_conv_problem_wo(p);
    rocke_value_t* c_Ho = rocke_b_const_i32(b, Ho);
    rocke_value_t* c_Wo = rocke_b_const_i32(b, Wo);
    rocke_value_t* c_K = rocke_b_const_i32(b, p->K);
    rocke_value_t* c_Hi = rocke_b_const_i32(b, p->Hi);
    rocke_value_t* c_Wi = rocke_b_const_i32(b, p->Wi);
    rocke_value_t* c_C = rocke_b_const_i32(b, p->C);
    rocke_value_t* c_Y = rocke_b_const_i32(b, p->Y);
    rocke_value_t* c_X = rocke_b_const_i32(b, p->X);
    rocke_value_t* c_dg_N = rocke_b_const_i32(b, p->C);

    // k_out-innermost: k_sub = ydot*xdot_slice*K + xdot*K + k_out
    // (c_K is the innermost divisor; ydot_times_xdot no longer needed in descriptors)
    rocke_value_t* hw_tilde = rocke_b_mul(b, rec_h_tilde_slice, rec_w_tilde_slice);

    // ---- LDS ----
    rocke_conv_lds_layout_t lds_layout = _dgrad_effective_lds_layout(spec);
    int a_shape[2] = {block_m, lds_layout.row_stride};
    int b_shape_arr[2] = {block_n, lds_layout.row_stride};
    rocke_value_t* A_smem = rocke_b_smem_alloc(b, ab_ir, a_shape, 2, "A_smem");
    rocke_value_t* B_smem = rocke_b_smem_alloc(b, ab_ir, b_shape_arr, 2, "B_smem");

    // ---- MFMA tile counts ----
    int mfmas_m = rocke_dgrad_conv_spec_mfmas_per_warp_m(spec);
    int mfmas_n = rocke_dgrad_conv_spec_mfmas_per_warp_n(spec);
    int k_atoms = rocke_dgrad_conv_spec_k_atoms_per_tile_k(spec);
    int num_accs = mfmas_m * mfmas_n;

    // ---- accumulators ----
    rocke_value_t* acc_init = rocke_b_zero_vec_f32(b, c_per_lane);
    rocke_iter_arg_t iter_args[ROCKE_CONV_MAX_ACCS];
    char acc_name_bufs[ROCKE_CONV_MAX_ACCS][32];
    for(int i = 0, idx = 0; i < mfmas_m; i++)
    {
        for(int j = 0; j < mfmas_n; j++, idx++)
        {
            snprintf(acc_name_bufs[idx], sizeof(acc_name_bufs[0]), "acc_m%d_n%d", i, j);
            iter_args[idx].name = acc_name_bufs[idx];
            iter_args[idx].init = acc_init;
        }
    }

    // ---- buffer resources ----
    rocke_value_t* dy_rsrc = rocke_b_buffer_rsrc(b, dY, dY_bytes);
    (void)rocke_b_const_i32(b, 0); /* dy soffset */
    rocke_value_t* w_rsrc = rocke_b_buffer_rsrc(b, W, W_bytes);
    (void)rocke_b_const_i32(b, 0); /* w soffset */
    rocke_value_t* dx_rsrc = rocke_b_buffer_rsrc(b, dX, dX_bytes);
    (void)rocke_b_const_i32(b, 0); /* dx soffset */

    // k_off_capture is managed via the tilde descriptor context structs

    int threads = block_size;
    // load_vec_a: k_out innermost -> consecutive k_sub -> consecutive k_out
    // -> contiguous in dY (NHWK, last dim K).  Condition: K % load_vec_a == 0.
    // For split_k > 1 the slice boundary may not be K-aligned; use 1 there.
    int load_vec_a = 1;
    if(spec->split_k <= 1)
    {
        bool is_fp32_a = (spec->dtype_a && strcmp(spec->dtype_a, "fp32") == 0);
        int max_from_K = 1;
        int kand[] = {is_fp32_a ? 4 : 8, 4, 2, 1};
        for(int v : kand)
            if(p->K % v == 0)
            {
                max_from_K = v;
                break;
            }
        int safe_vec = max_from_K;
        rocke_coalesced_tile_loader_choose_vec(block_m, block_k, block_size, max_from_K, &safe_vec);
        load_vec_a = spec->has_vector_size_a ? spec->vector_size_a : safe_vec;
    }
    // load_vec_b: B (W, KYXC) — the GEMM row axis is N_dg = c (input channels), which
    // is the stride-1 axis of KYXC.  Vectorise along the free (row) axis and transpose
    // into the row-major (N, K) LDS tile on store (vector_axis_row=true),
    // exactly as wgrad does for its B (X, NHWC) operand.  Condition: C % load_vec_b == 0.
    int load_vec_b = 1;
    bool axis_b_row = false;
    {
        bool is_fp32_b = (spec->dtype_b && strcmp(spec->dtype_b, "fp32") == 0);
        int cand[] = {is_fp32_b ? 4 : 8, 4, 2, 1};
        int max_from_C = 1;
        for(int v : cand)
            if(p->C % v == 0)
            {
                max_from_C = v;
                break;
            }
        int chosen = 1;
        rocke_status_t st = rocke_coalesced_tile_loader_choose_vec_axis(
            block_n, block_k, threads, max_from_C, true, &chosen);
        if(st != ROCKE_OK)
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "dgrad tilde: no usable free-axis load_vec for B tile geometry");
            return NULL;
        }
        if(spec->has_vector_size_b)
        {
            load_vec_b = spec->vector_size_b;
            axis_b_row = (load_vec_b > 1);
        }
        else if(chosen > 1)
        {
            load_vec_b = chosen;
            axis_b_row = true;
        }
    }

    rocke_coalesced_tile_loader_t a_sync_loader;
    a_sync_loader.tile_rows = block_m;
    a_sync_loader.tile_cols = block_k;
    a_sync_loader.block_size = threads;
    a_sync_loader.load_vec = load_vec_a;
    a_sync_loader.use_buffer_rsrc = true;
    a_sync_loader.oob_sentinel = 2147483647;
    a_sync_loader.vector_axis_row = false;
    a_sync_loader.has_inner_dim = false;
    a_sync_loader.inner_dim = 0;

    rocke_coalesced_tile_loader_t b_sync_loader;
    b_sync_loader.tile_rows = block_n;
    b_sync_loader.tile_cols = block_k;
    b_sync_loader.block_size = threads;
    b_sync_loader.load_vec = load_vec_b;
    b_sync_loader.use_buffer_rsrc = true;
    b_sync_loader.oob_sentinel = 2147483647;
    b_sync_loader.vector_axis_row = axis_b_row;
    b_sync_loader.has_inner_dim = false;
    b_sync_loader.inner_dim = 0;

    // ---- wavelet loaders (pipeline="wavelet" only) ----
    rocke_coalesced_tile_loader_t a_wavelet_loader;
    rocke_coalesced_tile_loader_t b_wavelet_loader;
    rocke_value_t* wavelet_is_math = NULL;
    rocke_value_t* wavelet_load_tid = NULL;
    int wavelet_epi_barriers = 0;
    int wavelet_K_iters = 0;
    if(is_wavelet)
    {
        int load_threads = spec->num_load_waves * spec->wave_size;
        rocke_status_t sa = rocke_coalesced_tile_loader_from_tile(
            block_m, block_k, load_threads, load_vec_a, true, &a_wavelet_loader);
        rocke_status_t sb = rocke_coalesced_tile_loader_from_tile(
            block_n, block_k, load_threads, load_vec_b, true, &b_wavelet_loader);
        if(sa != ROCKE_OK || sb != ROCKE_OK)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "dgrad: wavelet tile loader from_tile failed");
            return NULL;
        }
        int n_math_warps = spec->warp_m * spec->warp_n;
        rocke_value_t* c_nmath = rocke_b_const_i32(b, n_math_warps);
        rocke_value_t* warp_id_s = rocke_b_readfirstlane(b, warp_id);
        wavelet_is_math = rocke_b_cmp_lt(b, warp_id_s, c_nmath);
        wavelet_load_tid = rocke_b_sub(b, tid, rocke_b_const_i32(b, block_size));

        /* epi_barriers mirrors compute_wavelet_epi_barriers(spec.epilogue, no_alias).
         * no_alias=true for wavelet (A/B live across both branches). */
        bool no_alias = true;
        const int war_barriers = 2;
        bool use_cshuffle = (spec->epilogue && strcmp(spec->epilogue, "cshuffle") == 0);
        wavelet_epi_barriers = use_cshuffle ? (no_alias ? 0 : war_barriers) + 1 : 0;

        /* K_iters uses dg_K_padded (worst-case across sub-GEMMs) so it is a
         * compile-time constant that wavelet can unroll. */
        int dg_K_padded = rocke_dgrad_conv_spec_dg_K_padded(spec);
        int slice_k = (spec->split_k <= 1) ? dg_K_padded : (dg_K_padded / spec->split_k);
        wavelet_K_iters = _ceil_div(slice_k, block_k);
    }

    tilde_dy_ctx_t dy_tctx;
    dy_tctx.block_m_off = block_m_off_v;
    dy_tctx.k_off = NULL;
    dy_tctx.rec_x_dot_slice = rec_x_dot_slice;
    dy_tctx.hw_tilde = hw_tilde;
    dy_tctx.rec_w_tilde_slice = rec_w_tilde_slice;
    dy_tctx.rec_h_tilde_slice_begin = rec_h_tilde_slice_begin;
    dy_tctx.rec_w_tilde_slice_begin = rec_w_tilde_slice_begin;
    dy_tctx.rec_a_embed_h_coeff = rec_a_embed_h_coeff;
    dy_tctx.rec_a_embed_w_coeff = rec_a_embed_w_coeff;
    dy_tctx.c_Ho = c_Ho;
    dy_tctx.c_Wo = c_Wo;
    dy_tctx.c_K = c_K;
    dy_tctx.c0 = c0;

    tilde_w_ctx_t w_tctx;
    w_tctx.block_n_off = block_n_off_v;
    w_tctx.k_off = NULL;
    w_tctx.rec_x_dot_slice = rec_x_dot_slice;
    w_tctx.rec_b_y_stride = rec_b_y_stride;
    w_tctx.rec_b_y_offset = rec_b_y_offset;
    w_tctx.rec_b_x_stride = rec_b_x_stride;
    w_tctx.rec_b_x_offset = rec_b_x_offset;
    w_tctx.c_Y = c_Y;
    w_tctx.c_X = c_X;
    w_tctx.c_K = c_K;
    w_tctx.c_C = c_C;
    w_tctx.c0 = c0;

    // ---- schedule ----
    rocke_schedule_policy_t schedule = rocke_schedule_policy_for_pipeline(b, spec->pipeline);
    rocke_schedule_policy_emit_prologue(&schedule, b);

    // ---- helper lambda-equivalent: emit WMMA phase from LDS into accs ----
    // Used by both the wavelet and standard K-loop paths.
    auto emit_wmma_phase = [&](rocke_value_t* A_src,
                               rocke_value_t* B_src,
                               rocke_value_t* const* in_accs,
                               rocke_value_t** out_accs) {
        const rocke_arch_layout_map_t* a_map = rocke_mmaop_a_layout(op, b);
        const rocke_arch_layout_map_t* b_map = rocke_mmaop_b_layout(op, b);
        rocke_value_t* a_row_in_atom = NULL;
        rocke_value_t* a_k_in_atom = NULL;
        rocke_value_t* b_k_in_atom = NULL;
        rocke_value_t* b_col_in_atom = NULL;
        rocke_arch_layout_map_coord(a_map, b, lane, 0, &a_row_in_atom, &a_k_in_atom);
        rocke_arch_layout_map_coord(b_map, b, lane, 0, &b_k_in_atom, &b_col_in_atom);
        rocke_value_t* warp_m_off = rocke_warp_grid_warp_m_off(b, &grid);
        rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, &grid);
        rocke_value_t* a_rows[ROCKE_CONV_MAX_ACCS];
        rocke_value_t* b_wma_cols[ROCKE_CONV_MAX_ACCS];
        for(int i = 0; i < num_accs; i++)
            out_accs[i] = in_accs[i];
        for(int kk = 0; kk < k_atoms; kk++)
        {
            rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * spec->warp_tile_k);
            for(int mi = 0; mi < mfmas_m; mi++)
            {
                rocke_value_t* atom_row
                    = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * spec->warp_tile_m));
                a_rows[mi] = rocke_conv_emit_frag_smem_load(
                    b, A_src, a_row_in_atom, a_k_in_atom, atom_row, k_tile_base, a_per_lane);
            }
            for(int ni = 0; ni < mfmas_n; ni++)
            {
                rocke_value_t* atom_row
                    = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * spec->warp_tile_n));
                b_wma_cols[ni] = rocke_conv_emit_frag_smem_load(
                    b, B_src, b_col_in_atom, b_k_in_atom, atom_row, k_tile_base, b_per_lane);
            }
            int flat2 = 0;
            for(int mi = 0; mi < mfmas_m; mi++)
                for(int ni = 0; ni < mfmas_n; ni++)
                {
                    out_accs[flat2] = rocke_b_mma(
                        b, op->op_id, a_rows[mi], b_wma_cols[ni], out_accs[flat2], NULL, 0);
                    flat2++;
                }
        }
    };

    // ---- helper: dispatch dgrad epilogue ----
    auto dispatch_dgrad_epilogue = [&](rocke_value_t* const* epi_accs_, int n_epi) {
        bool is_split_k_atomic_ = (spec->split_k > 1);
        bool is_strided_ = rocke_dgrad_conv_spec_is_strided(spec);
        if(!is_split_k_atomic_ && !is_strided_)
        {
            if(is_wmma)
            {
                bool use_cshuffle_ = (spec->epilogue && strcmp(spec->epilogue, "cshuffle") == 0);
                if(use_cshuffle_)
                    _emit_dgrad_direct_epilogue(b, spec, epi_accs_, n_epi, &grid, dx_rsrc);
                else
                    _emit_dgrad_direct_epilogue_wmma(b,
                                                     spec,
                                                     op,
                                                     epi_accs_,
                                                     n_epi,
                                                     warp_m_idx,
                                                     warp_n_idx,
                                                     lane,
                                                     block_m_off_v,
                                                     block_n_off_v,
                                                     dx_rsrc,
                                                     c0);
            }
            else
            {
                bool use_cshuffle_ = (spec->epilogue && strcmp(spec->epilogue, "cshuffle") == 0);
                if(use_cshuffle_)
                    _emit_dgrad_cshuffle_epilogue(b, spec, epi_accs_, n_epi, &grid, dx_rsrc);
                else
                    _emit_dgrad_direct_epilogue(b, spec, epi_accs_, n_epi, &grid, dx_rsrc);
            }
        }
        else if(!is_split_k_atomic_ && is_wmma)
        {
            _emit_dgrad_tilde_direct_epilogue_wmma(b,
                                                   spec,
                                                   op,
                                                   epi_accs_,
                                                   n_epi,
                                                   warp_m_idx,
                                                   warp_n_idx,
                                                   lane,
                                                   block_m_off_v,
                                                   block_n_off_v,
                                                   dx_rsrc,
                                                   c0,
                                                   rec_gemm_m,
                                                   c_dg_N,
                                                   hw_tilde,
                                                   rec_w_tilde_slice,
                                                   rec_d_h_stride,
                                                   rec_d_h_offset,
                                                   rec_d_w_stride,
                                                   rec_d_w_offset,
                                                   c_Hi,
                                                   c_Wi,
                                                   c_C);
        }
        else if(!is_split_k_atomic_ && !is_wmma && atom)
        {
            bool use_cshuffle_ = (spec->epilogue && strcmp(spec->epilogue, "cshuffle") == 0);
            if(use_cshuffle_)
                _emit_dgrad_tilde_cshuffle_epilogue(b,
                                                    spec,
                                                    atom,
                                                    &grid,
                                                    epi_accs_,
                                                    n_epi,
                                                    dx_rsrc,
                                                    rec_gemm_m,
                                                    c_dg_N,
                                                    hw_tilde,
                                                    rec_w_tilde_slice,
                                                    rec_d_h_stride,
                                                    rec_d_h_offset,
                                                    rec_d_w_stride,
                                                    rec_d_w_offset,
                                                    c_Hi,
                                                    c_Wi,
                                                    c_C);
            else
                _emit_dgrad_tilde_direct_epilogue(b,
                                                  spec,
                                                  atom,
                                                  &grid,
                                                  epi_accs_,
                                                  n_epi,
                                                  dx_rsrc,
                                                  rec_gemm_m,
                                                  c_dg_N,
                                                  hw_tilde,
                                                  rec_w_tilde_slice,
                                                  rec_d_h_stride,
                                                  rec_d_h_offset,
                                                  rec_d_w_stride,
                                                  rec_d_w_offset,
                                                  c_Hi,
                                                  c_Wi,
                                                  c_C);
        }
        else
        {
            _emit_dgrad_tilde_atomic_epilogue(b,
                                              spec,
                                              atom,
                                              epi_accs_,
                                              n_epi,
                                              warp_m_idx,
                                              warp_n_idx,
                                              lane,
                                              block_m_off_v,
                                              block_n_off_v,
                                              dX,
                                              c_per_lane,
                                              rec_gemm_m,
                                              c_dg_N,
                                              rec_h_tilde_slice,
                                              rec_w_tilde_slice,
                                              rec_d_h_stride,
                                              rec_d_h_offset,
                                              rec_d_w_stride,
                                              rec_d_w_offset,
                                              c_Hi,
                                              c_Wi,
                                              c_C);
        }
    };

    // ---- wavelet K-loop (pipeline="wavelet", WMMA/gfx1250 only) ----
    if(is_wavelet)
    {
        /* WMMA path: scf_if_else with a shared join block (gfx1250).
         * Barrier protocol mirrors rocke_conv_emit_kloop_wavelet WMMA branch. */
        rocke_ctl_staged_t a_staged;
        rocke_ctl_staged_t b_staged;
        rocke_if_else_t ife = rocke_b_scf_if_else(b, wavelet_is_math);

        // ---- MATH WAVE branch ----
        rocke_value_t* current_accs[ROCKE_CONV_MAX_ACCS];
        rocke_value_t* new_accs_wv[ROCKE_CONV_MAX_ACCS];
        for(int i = 0; i < num_accs; i++)
            current_accs[i] = iter_args[i].init;

        rocke_b_region_enter(b, ife.then_region);
        {
            rocke_b_sync(b); /* barrier_0 */
            for(int it = 0; it < wavelet_K_iters - 1; it++)
            {
                dy_tctx.k_off = rocke_b_const_i32(b, it * block_k);
                w_tctx.k_off = rocke_b_const_i32(b, it * block_k);
                emit_wmma_phase(A_smem, B_smem, current_accs, new_accs_wv);
                for(int i = 0; i < num_accs; i++)
                    current_accs[i] = new_accs_wv[i];
                rocke_b_sync(b); /* barrier_A */
                rocke_b_sync(b); /* barrier_B */
            }
            /* tail MFMA -- no barriers */
            dy_tctx.k_off = rocke_b_const_i32(b, (wavelet_K_iters - 1) * block_k);
            w_tctx.k_off = rocke_b_const_i32(b, (wavelet_K_iters - 1) * block_k);
            emit_wmma_phase(A_smem, B_smem, current_accs, new_accs_wv);
            for(int i = 0; i < num_accs; i++)
                current_accs[i] = new_accs_wv[i];

            /* epilogue (inside math branch, no iter-var yield) */
            rocke_value_t* epi_accs[ROCKE_CONV_MAX_ACCS];
            rocke_conv_apply_accumulator_epilogue(
                b, &spec->acc_epilogue, current_accs, num_accs, epi_accs);
            dispatch_dgrad_epilogue(epi_accs, num_accs);
        }
        rocke_b_region_leave(b);

        // ---- LOAD WAVE branch ----
        rocke_b_region_enter(b, ife.else_region);
        {
            /* fetch tile 0 -> regs, store -> LDS, barrier_0 */
            dy_tctx.k_off = c0;
            w_tctx.k_off = c0;
            rocke_coalesced_tile_loader_load_global(b,
                                                    &a_wavelet_loader,
                                                    wavelet_load_tid,
                                                    _tilde_dy_descriptor,
                                                    &dy_tctx,
                                                    dy_rsrc,
                                                    NULL,
                                                    &a_staged);
            rocke_coalesced_tile_loader_load_global(b,
                                                    &b_wavelet_loader,
                                                    wavelet_load_tid,
                                                    _tilde_w_descriptor,
                                                    &w_tctx,
                                                    w_rsrc,
                                                    NULL,
                                                    &b_staged);
            rocke_b_s_waitcnt(b, 0, -1, -1); /* vmcnt=0 */
            rocke_coalesced_tile_loader_store_lds(b, &a_wavelet_loader, A_smem, &a_staged);
            rocke_coalesced_tile_loader_store_lds(b, &b_wavelet_loader, B_smem, &b_staged);
            rocke_b_s_waitcnt(b, -1, 0, -1); /* lgkmcnt=0 */
            rocke_b_sync(b); /* barrier_0 */

            for(int it = 0; it < wavelet_K_iters - 1; it++)
            {
                dy_tctx.k_off = rocke_b_const_i32(b, (it + 1) * block_k);
                w_tctx.k_off = rocke_b_const_i32(b, (it + 1) * block_k);
                rocke_coalesced_tile_loader_load_global(b,
                                                        &a_wavelet_loader,
                                                        wavelet_load_tid,
                                                        _tilde_dy_descriptor,
                                                        &dy_tctx,
                                                        dy_rsrc,
                                                        NULL,
                                                        &a_staged);
                rocke_coalesced_tile_loader_load_global(b,
                                                        &b_wavelet_loader,
                                                        wavelet_load_tid,
                                                        _tilde_w_descriptor,
                                                        &w_tctx,
                                                        w_rsrc,
                                                        NULL,
                                                        &b_staged);
                rocke_b_sync(b); /* barrier_A */
                rocke_b_s_waitcnt(b, 0, -1, -1); /* vmcnt=0 */
                rocke_coalesced_tile_loader_store_lds(b, &a_wavelet_loader, A_smem, &a_staged);
                rocke_coalesced_tile_loader_store_lds(b, &b_wavelet_loader, B_smem, &b_staged);
                rocke_b_s_waitcnt(b, -1, 0, -1); /* lgkmcnt=0 */
                rocke_b_sync(b); /* barrier_B */
            }
            /* epilogue stub: epi_barriers bare barriers matching math branch */
            for(int i = 0; i < wavelet_epi_barriers; i++)
                rocke_b_sync(b);
        }
        rocke_b_region_leave(b);

        return b->kernel;
    }

    // ---- K loop (simple scf.for_iter) ----
    rocke_for_t for_op
        = rocke_b_scf_for_iter(b, k_lo, k_hi, c_block_k, iter_args, num_accs, "k0", false, true);

    rocke_value_t* k0 = for_op.iv;
    rocke_value_t* iter_vars[ROCKE_CONV_MAX_ACCS];
    for(int i = 0; i < for_op.num_iter_vars; i++)
        iter_vars[i] = for_op.iter_vars[i];

    rocke_b_region_enter(b, for_op.body);
    {
        // Set k_off_capture for descriptor closures
        dy_tctx.k_off = k0;
        w_tctx.k_off = k0;

        rocke_coalesced_tile_loader_load(
            b, &a_sync_loader, tid, A_smem, _tilde_dy_descriptor, &dy_tctx, dy_rsrc, NULL);
        rocke_coalesced_tile_loader_load(
            b, &b_sync_loader, tid, B_smem, _tilde_w_descriptor, &w_tctx, w_rsrc, NULL);
        rocke_b_sync(b);

        // MFMA phase
        rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];
        if(!is_wmma && atom)
        {
            rocke_lane_decode_t decoded = rocke_decode_mfma_lanes(b, atom, lane);
            rocke_value_t* m_in_atom = decoded.m_in_atom;
            rocke_value_t* n_in_atom = decoded.n_in_atom;
            rocke_value_t* k_blk = decoded.k_blk;
            rocke_value_t* warp_m_off = rocke_warp_grid_warp_m_off(b, &grid);
            rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, &grid);

            for(int i = 0; i < num_accs; i++)
                new_accs[i] = iter_vars[i];

            for(int kk = 0; kk < k_atoms; kk++)
            {
                rocke_value_t* _cb_mul = rocke_b_mul(b, k_blk, rocke_b_const_i32(b, a_per_lane));
                rocke_value_t* col_base
                    = rocke_b_add(b, _cb_mul, rocke_b_const_i32(b, kk * spec->warp_tile_k));

                rocke_value_t* a_rows[ROCKE_CONV_MAX_ACCS];
                for(int mi = 0; mi < mfmas_m; mi++)
                {
                    rocke_value_t* a_row = rocke_b_add(
                        b,
                        warp_m_off,
                        rocke_b_add(b, rocke_b_const_i32(b, mi * spec->warp_tile_m), m_in_atom));
                    a_rows[mi] = rocke_conv_emit_smem_load(b, A_smem, a_row, col_base, a_per_lane);
                }

                rocke_value_t* b_cols[ROCKE_CONV_MAX_ACCS];
                for(int ni = 0; ni < mfmas_n; ni++)
                {
                    rocke_value_t* b_row = rocke_b_add(
                        b,
                        warp_n_off,
                        rocke_b_add(b, rocke_b_const_i32(b, ni * spec->warp_tile_n), n_in_atom));
                    b_cols[ni] = rocke_conv_emit_smem_load(b, B_smem, b_row, col_base, b_per_lane);
                }

                int flat = 0;
                for(int mi = 0; mi < mfmas_m; mi++)
                {
                    for(int ni = 0; ni < mfmas_n; ni++)
                    {
                        new_accs[flat]
                            = rocke_conv_emit_mfma(b, atom, a_rows[mi], b_cols[ni], new_accs[flat]);
                        flat++;
                    }
                }

                rocke_schedule_policy_emit_after_mfma_step(
                    &schedule, b, mfmas_m + mfmas_n, mfmas_m * mfmas_n);
            }
        }
        else if(is_wmma)
        {
            emit_wmma_phase(A_smem, B_smem, iter_vars, new_accs);
        }
        else
        {
            for(int i = 0; i < num_accs; i++)
                new_accs[i] = iter_vars[i];
        }

        rocke_b_sync(b);
        rocke_b_scf_yield(b, new_accs, num_accs);
    }
    rocke_b_region_leave(b);

    // ---- final_accs ----
    rocke_value_t* final_accs[ROCKE_CONV_MAX_ACCS];
    for(int i = 0; i < for_op.op->num_results; i++)
        final_accs[i] = for_op.op->results[i];
    int num_final = for_op.op->num_results;

    // ---- accumulator epilogue ----
    rocke_value_t* epi_accs[ROCKE_CONV_MAX_ACCS];
    rocke_conv_apply_accumulator_epilogue(b, &spec->acc_epilogue, final_accs, num_final, epi_accs);

    // ---- epilogue dispatch ----
    dispatch_dgrad_epilogue(epi_accs, num_final);

    return b->kernel;
}

// ===========================================================================
// Public build entry: rocke_build_implicit_gemm_conv_dgrad
// (Python build_implicit_gemm_conv_dgrad, lines 833-1314)
// ===========================================================================

rocke_kernel_def_t* rocke_build_implicit_gemm_conv_dgrad(rocke_ir_builder_t* b,
                                                         const rocke_dgrad_conv_spec_t* spec,
                                                         const char* arch)
{
    char reason[256];

    if(!b || !spec)
        return NULL;
    if(!arch)
        arch = "gfx950";

    // Name the kernel
    if(b->kernel)
    {
        char name[256];
        if(rocke_dgrad_conv_spec_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
            return NULL;
        b->kernel->name = rocke_arena_strdup(&b->arena, name);
    }

    // Validate
    if(!rocke_dgrad_conv_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid dgrad spec for %s: %s", arch, reason);
        return NULL;
    }

    return _build_tilde_dgrad(b, spec, arch);
}

// ===========================================================================
// Convenience: init builder then build
// ===========================================================================

rocke_kernel_def_t* rocke_build_implicit_gemm_conv_dgrad_new(rocke_ir_builder_t* b,
                                                             const rocke_dgrad_conv_spec_t* spec,
                                                             const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(!b || !spec)
            return NULL;
        if(rocke_dgrad_conv_spec_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_implicit_gemm_conv_dgrad(b, spec, arch);
    });
}

// ===========================================================================
// Convenience: build + lower to LLVM .ll
// ===========================================================================

static void _dgrad_set_err(char* err, size_t err_cap, const char* msg)
{
    if(!err || err_cap == 0)
        return;
    if(!msg)
        msg = "";
    size_t n = strlen(msg);
    if(n >= err_cap)
        n = err_cap - 1;
    memcpy(err, msg, n);
    err[n] = '\0';
}

rocke_status_t rocke_dgrad_conv_implicit_gemm_lower_to_llvm(const rocke_dgrad_conv_spec_t* spec,
                                                            const char* arch,
                                                            rocke_llvm_flavor_t flavor,
                                                            char** out_ll,
                                                            char* err,
                                                            size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll)
        *out_ll = NULL;
    if(!spec || !out_ll)
    {
        _dgrad_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(!arch)
        arch = "gfx950";

    kernel = rocke_build_implicit_gemm_conv_dgrad_new(&b, spec, arch);
    if(!kernel)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        _dgrad_set_err(err, err_cap, (m && m[0]) ? m : "build dgrad failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
