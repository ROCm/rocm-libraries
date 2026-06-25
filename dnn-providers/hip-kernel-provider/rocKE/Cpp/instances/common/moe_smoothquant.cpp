// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_smoothquant.c -- C99 port of
 * rocke/instances/common/moe_smoothquant.py.
 *
 * Byte-identical builder-call sequence vs the Python build_moe_smoothquant: same
 * op order / attrs. The same higher-level distribution / tensor-view helpers the
 * sibling SmoothQuant port leans on (load_tile / store_tile /
 * make_static_distributed_tensor, make_naive_tensor_view_packed / make_lds_view,
 * the F32-view load_vec_as_f32) are wired through the same local inline shims
 * here. The local `rocke_moe_sq_*` helpers mirror the smoothquant ones verbatim
 * (renamed to avoid linker collision) to honour the "instance files only"
 * optimisation scope -- the Python keeps _tree_fmax / make_row_x_distribution
 * as instance-local duplicates for exactly the same reason.
 *
 * MoE-specific additions vs SmoothQuant:
 *   * (i_topk, i_token) decode from out_row via div/mod (runtime) or, when
 *     spec.tokens_set, a compile-time const divisor (the AMDGPU backend folds
 *     it to a reciprocal-mul pair).
 *   * TopkIds[i_token*topk + i_topk] gathers the per-token expert id, pinned to
 *     SGPR via to_sgpr_u32.
 *   * SmScale is a flat (experts*N,) view; sm_row_base = i_expert*N is hoisted
 *     once (also SGPR-pinned) and added per chunk.
 *   * QY / X tile windows are origin'd at out_row / i_token respectively, and
 *     YScale is stored at out_row.
 */
#include "rocke/instance_moe_smoothquant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.distribution.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.quant.h"
#include "rocke/helper_rocke.helpers.reduction.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.sweep.h"
#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */
#include "rocke/lower_llvm.h"

/* --------------------------------------------------------------- locals */

#define ROCKE_MOE_SQ_NAME_CAP 256
#define ROCKE_MOE_SQ_MAX_VEC 8

/* QUANT_MAX_ABS table (helpers/quant.py). i8=127, fp8e4m3=448, bf8e5m2=57344. */
static double rocke_moe_sq_quant_max_abs(const char* out_dtype)
{
    if(out_dtype == NULL)
    {
        return 0.0;
    }
    if(strcmp(out_dtype, "i8") == 0 || strcmp(out_dtype, "int8") == 0)
    {
        return 127.0;
    }
    if(strcmp(out_dtype, "fp8e4m3") == 0 || strcmp(out_dtype, "fp8") == 0
       || strcmp(out_dtype, "fp8_e4m3") == 0)
    {
        return 448.0;
    }
    if(strcmp(out_dtype, "bf8e5m2") == 0 || strcmp(out_dtype, "bf8") == 0
       || strcmp(out_dtype, "fp8_e5m2") == 0)
    {
        return 57344.0;
    }
    return 0.0;
}

/* Canonicalised out-dtype tag for quantize dispatch ("i8"/"fp8e4m3"/"bf8e5m2"). */
static const char* rocke_moe_sq_canon_out(const char* out_dtype)
{
    if(out_dtype == NULL)
    {
        return NULL;
    }
    if(strcmp(out_dtype, "i8") == 0 || strcmp(out_dtype, "int8") == 0)
    {
        return "i8";
    }
    if(strcmp(out_dtype, "fp8e4m3") == 0 || strcmp(out_dtype, "fp8") == 0
       || strcmp(out_dtype, "fp8_e4m3") == 0)
    {
        return "fp8e4m3";
    }
    if(strcmp(out_dtype, "bf8e5m2") == 0 || strcmp(out_dtype, "bf8") == 0
       || strcmp(out_dtype, "fp8_e5m2") == 0)
    {
        return "bf8e5m2";
    }
    return NULL;
}

/* quantize_scalar_f32(b, x_f32, inv_scale, qdtype) -- helpers/quant.py.
 *   scaled  = b.fmul(x_f32, inv_scale)
 *   clamped = b.clamp_f32(scaled, -qmax, +qmax)
 *   result  = cvt_f32_to_<qdtype>(clamped)
 */
static rocke_value_t* rocke_moe_sq_quantize_scalar_f32(rocke_ir_builder_t* b,
                                                       rocke_value_t* x_f32,
                                                       rocke_value_t* inv_scale,
                                                       const char* qdtype)
{
    const char* canon;
    double qmax;
    rocke_value_t* c_pos;
    rocke_value_t* c_neg;
    rocke_value_t* scaled;
    rocke_value_t* clamped;

    if(b != NULL && b->status != ROCKE_OK)
    {
        return NULL;
    }
    canon = rocke_moe_sq_canon_out(qdtype);
    if(canon == NULL)
    {
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "quantize_scalar_f32: unsupported qdtype");
    }
    qmax = rocke_moe_sq_quant_max_abs(canon);
    c_pos = rocke_b_const_f32(b, qmax);
    c_neg = rocke_b_const_f32(b, -qmax);
    scaled = rocke_b_fmul(b, x_f32, inv_scale);
    clamped = rocke_b_clamp_f32(b, scaled, c_neg, c_pos);
    if(strcmp(canon, "i8") == 0)
    {
        return rocke_b_cvt_f32_to_i8_sat(b, clamped);
    }
    if(strcmp(canon, "fp8e4m3") == 0)
    {
        return rocke_b_cvt_f32_to_fp8(b, clamped);
    }
    /* bf8e5m2 */
    return rocke_b_cvt_f32_to_bf8(b, clamped);
}

/* _tree_fmax(b, values) -- balanced pairwise fmax tree (moe_smoothquant.py). */
static rocke_value_t* rocke_moe_sq_tree_fmax(rocke_ir_builder_t* b,
                                             rocke_value_t** values,
                                             int n,
                                             rocke_value_t** scratch)
{
    rocke_value_t** cur;
    int cur_n;

    if(b != NULL && b->status != ROCKE_OK)
    {
        return NULL;
    }
    if(n < 1)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "_tree_fmax: empty");
    }
    cur = scratch;
    for(int i = 0; i < n; ++i)
    {
        cur[i] = values[i];
    }
    cur_n = n;
    while(cur_n > 1)
    {
        int w = 0;
        int i;
        for(i = 0; i + 1 < cur_n; i += 2)
        {
            cur[w++] = rocke_b_fmax(b, cur[i], cur[i + 1]);
        }
        if(cur_n % 2)
        {
            cur[w++] = cur[cur_n - 1];
        }
        cur_n = w;
    }
    return cur[0];
}

/* make_row_x_distribution(block_size, vec, elems_per_thread) --
 * moe_smoothquant.py. Identical encoding to the SmoothQuant sibling:
 *   Hs            = ((1,), (chunks, block_size, vec))
 *   Ps2RHs_major  = ((2,),)
 *   Ps2RHs_minor  = ((1,),)
 *   Ys2RHs_major  = (1, 2, 2)
 *   Ys2RHs_minor  = (0, 0, 2)
 */
static rocke_tile_distribution_t* rocke_moe_sq_make_row_x_distribution(rocke_ir_builder_t* b,
                                                                       int block_size,
                                                                       int vec,
                                                                       int elems_per_thread)
{
    int chunks = elems_per_thread / vec;
    rocke_tile_distribution_encoding_t* enc;

    int h0_levels[1];
    int h1_levels[3];
    rocke_h_row_t Hs[2];

    int p0_major[1];
    int p0_minor[1];
    rocke_p_seq_t Ps[1];

    int Ys_major[3];
    int Ys_minor[3];

    if(b != NULL && b->status != ROCKE_OK)
    {
        return NULL;
    }

    h0_levels[0] = 1;
    h1_levels[0] = chunks;
    h1_levels[1] = block_size;
    h1_levels[2] = vec;
    Hs[0].levels = h0_levels;
    Hs[0].count = 1;
    Hs[1].levels = h1_levels;
    Hs[1].count = 3;

    p0_major[0] = 2;
    p0_minor[0] = 1;
    Ps[0].major = p0_major;
    Ps[0].minor = p0_minor;
    Ps[0].count = 1;

    Ys_major[0] = 1;
    Ys_major[1] = 2;
    Ys_major[2] = 2;
    Ys_minor[0] = 0;
    Ys_minor[1] = 0;
    Ys_minor[2] = 2;

    enc = rocke_make_tile_distribution_encoding(b,
                                                /* Rs    */ NULL,
                                                /* num_R */ 0,
                                                /* Hs    */ Hs,
                                                /* num_X */ 2,
                                                /* Ps    */ Ps,
                                                /* num_P */ 1,
                                                /* Ys_major */ Ys_major,
                                                /* Ys_minor */ Ys_minor,
                                                /* num_Y */ 3);
    if(enc == NULL)
    {
        return NULL;
    }
    return rocke_make_static_tile_distribution(b, enc);
}

/* --------------------------------------------------------------- load/store tile
 *
 * Faithful inline port of distribution.py load_tile / store_tile + the
 * LoadStoreTraits picker, specialised to the row distribution. Mirrors the
 * smoothquant sibling shims verbatim (renamed); see that file for the rationale. */

/* Per-Y-dim length from the encoding (Hs[major-1][minor]; major==0 => Rs). */
static int rocke_moe_sq_y_length(const rocke_tile_distribution_encoding_t* enc, int y)
{
    int major = enc->Ys_major[y];
    int minor = enc->Ys_minor[y];
    if(major == 0)
    {
        return enc->Rs[minor];
    }
    return enc->Hs[major - 1].levels[minor];
}

/* _y_x_stride: stride of Y dim y in its target X dim (1 for R-mapped). */
static int rocke_moe_sq_y_x_stride(const rocke_tile_distribution_encoding_t* enc, int y)
{
    int major = enc->Ys_major[y];
    int minor = enc->Ys_minor[y];
    const rocke_h_row_t* h;
    int stride = 1;
    int level;
    if(major == 0)
    {
        return 1;
    }
    h = &enc->Hs[major - 1];
    for(level = minor + 1; level < h->count; ++level)
    {
        stride *= h->levels[level];
    }
    return stride;
}

/* make_load_store_traits picker (max_vec=8, min_vec=1). */
static int rocke_moe_sq_pick_traits(const rocke_tile_distribution_encoding_t* enc,
                                    int* vector_dim_y)
{
    int num_Y = enc->num_Y;
    int best_y = -1, best_len = -1;
    int y, spv, full_len;
    for(y = 0; y < num_Y; ++y)
    {
        if(rocke_moe_sq_y_x_stride(enc, y) == 1)
        {
            int len = rocke_moe_sq_y_length(enc, y);
            if(len >= best_len) /* >= so ties pick the higher Y index */
            {
                best_len = len;
                best_y = y;
            }
        }
    }
    if(best_y < 0)
    {
        *vector_dim_y = num_Y - 1;
        return 1;
    }
    full_len = best_len;
    spv = full_len < 8 ? full_len : 8;
    while(spv > 1 && (full_len % spv != 0 || (spv & (spv - 1)) != 0))
    {
        spv /= 2;
    }
    if(spv < 1)
    {
        spv = 1;
    }
    *vector_dim_y = best_y;
    return spv;
}

/* StaticDistributedTensor row-major linear index for a Y tuple (y_to_linear). */
static int rocke_moe_sq_y_to_linear(const rocke_tile_distribution_encoding_t* enc, const int* y)
{
    int off = 0;
    int i;
    for(i = 0; i < enc->num_Y; ++i)
    {
        off = off * rocke_moe_sq_y_length(enc, i) + y[i];
    }
    return off;
}

/* _pack_quant_local: quantise + pack scaled-f32 scalars into <n x q_ty>. */
static rocke_value_t* rocke_moe_sq_pack_quant_local(rocke_ir_builder_t* b,
                                                    rocke_value_t* const* scaled,
                                                    int n,
                                                    const rocke_type_t* q_ty,
                                                    const char* qname)
{
    int off, i;
    if((strcmp(qname, "fp8e4m3") == 0 || strcmp(qname, "bf8e5m2") == 0) && (n % 4) == 0)
    {
        rocke_value_t* out = NULL;
        for(off = 0; off < n; off += 4)
        {
            rocke_value_t* quad = rocke_b_vec_pack(b, &scaled[off], 4, rocke_f32());
            rocke_value_t* chunk = (strcmp(qname, "fp8e4m3") == 0)
                                       ? rocke_b_cvt_pk_fp8_f32x4(b, quad)
                                       : rocke_b_cvt_pk_bf8_f32x4(b, quad);
            out = (out == NULL) ? chunk : rocke_b_vec_concat(b, out, chunk);
        }
        return out;
    }
    {
        rocke_value_t* qs[ROCKE_MOE_SQ_MAX_VEC];
        for(i = 0; i < n; ++i)
        {
            if(strcmp(qname, "i8") == 0)
            {
                rocke_value_t* c_neg = rocke_b_const_f32(b, -127.0);
                rocke_value_t* c_pos = rocke_b_const_f32(b, 127.0);
                rocke_value_t* cl = rocke_b_clamp_f32(b, scaled[i], c_neg, c_pos);
                qs[i] = rocke_b_cvt_f32_to_i8_sat(b, cl);
            }
            else if(strcmp(qname, "fp8e4m3") == 0)
            {
                qs[i] = rocke_b_cvt_f32_to_fp8(b, scaled[i]);
            }
            else /* bf8e5m2 */
            {
                qs[i] = rocke_b_cvt_f32_to_bf8(b, scaled[i]);
            }
        }
        return rocke_b_vec_pack(b, qs, n, q_ty);
    }
}

/* SmScale F32-view load_vec_as_f32 (tensor_view.py): for an f32 view the per-lane
 * cast is a no-op; just vec_load + vec_extract. n in {2,4,8}; n==1 -> scalar. */
static void rocke_moe_sq_view_load_vec_as_f32(rocke_ir_builder_t* b,
                                              const rocke_tensor_view_t* v,
                                              rocke_value_t* const* indices,
                                              int num_indices,
                                              int n,
                                              rocke_value_t** out)
{
    if(b != NULL && b->status != ROCKE_OK)
    {
        return;
    }
    if(n == 1)
    {
        out[0] = rocke_tensor_view_load_scalar(b, v, indices, num_indices);
        return;
    }
    {
        rocke_value_t* vec = rocke_tensor_view_load_vec(b, v, indices, num_indices, n);
        for(int i = 0; i < n; ++i)
        {
            out[i] = rocke_b_vec_extract(b, vec, i);
        }
    }
}

/* --------------------------------------------------------------- spec */

void rocke_moe_smoothquant_spec_init(rocke_moe_smoothquant_spec_t* spec,
                                     int n_per_block,
                                     int topk,
                                     int experts)
{
    if(spec == NULL)
    {
        return;
    }
    spec->n_per_block = n_per_block;
    spec->topk = topk;
    spec->experts = experts;
    spec->dtype = "f16";
    spec->out_dtype = "i8";
    spec->block_size = 256;
    spec->vec = 4;
    spec->save_yscale = true;
    spec->wave_size = 64;
    spec->name = "rocke_moe_smoothquant";
    spec->tokens_set = false;
    spec->tokens = 0;
}

int rocke_moe_smoothquant_elems_per_thread(const rocke_moe_smoothquant_spec_t* spec)
{
    if(spec == NULL || spec->block_size == 0)
    {
        return 0;
    }
    return spec->n_per_block / spec->block_size;
}

rocke_status_t rocke_moe_smoothquant_kernel_name(const rocke_moe_smoothquant_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap)
{
    char nbuf[32];
    char ebuf[32];
    char kbuf[32];
    char bbuf[32];
    char vbuf[32];
    const char* parts[7];
    const char* flag_names[1];
    int flag_on[1];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* kernel_name_join(name, dtype, out_dtype, "N{n}", "E{experts}", "K{topk}",
     *                  "b{bs}", "v{vec}", flags={"ys": save_yscale}) */
    snprintf(nbuf, sizeof(nbuf), "N%d", spec->n_per_block);
    snprintf(ebuf, sizeof(ebuf), "E%d", spec->experts);
    snprintf(kbuf, sizeof(kbuf), "K%d", spec->topk);
    snprintf(bbuf, sizeof(bbuf), "b%d", spec->block_size);
    snprintf(vbuf, sizeof(vbuf), "v%d", spec->vec);
    parts[0] = spec->dtype;
    parts[1] = spec->out_dtype;
    parts[2] = nbuf;
    parts[3] = ebuf;
    parts[4] = kbuf;
    parts[5] = bbuf;
    parts[6] = vbuf;
    flag_names[0] = "ys";
    flag_on[0] = spec->save_yscale ? 1 : 0;
    return rocke_kernel_name_join(spec->name, parts, 7, flag_names, flag_on, 1, out, out_cap, NULL);
}

/* --------------------------------------------------------------- valid */

bool rocke_moe_smoothquant_is_valid_spec(const rocke_moe_smoothquant_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap)
{
    const rocke_arch_target_t* target;
    const char* canon;
    long bytes_lds;

    if(reason != NULL && reason_cap > 0)
    {
        reason[0] = '\0';
    }
    if(spec == NULL)
    {
        return false;
    }
    if(arch == NULL)
    {
        arch = ROCKE_MOE_SMOOTHQUANT_DEFAULT_ARCH;
    }

    /* target = ArchTarget.from_gfx(arch); KeyError -> (False, msg). */
    target = rocke_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "unknown arch %s", arch);
        }
        return false;
    }

    /* out_dtype not in ("i8","fp8e4m3","bf8e5m2"). */
    canon = rocke_moe_sq_canon_out(spec->out_dtype);
    if(canon == NULL
       || (strcmp(canon, "i8") != 0 && strcmp(canon, "fp8e4m3") != 0
           && strcmp(canon, "bf8e5m2") != 0))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "unsupported out_dtype %s",
                     spec->out_dtype ? spec->out_dtype : "(null)");
        }
        return false;
    }

    /* fp8/bf8 need the CDNA-only v_cvt_pk_{fp8,bf8}_f32 op. */
    if((strcmp(canon, "fp8e4m3") == 0 || strcmp(canon, "bf8e5m2") == 0)
       && (target->family == NULL || strcmp(target->family, "cdna") != 0))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "out_dtype %s needs the CDNA-only v_cvt_pk_{fp8,bf8}_f32 "
                     "conversion; %s (family %s) has no fp8/bf8 pack op -- use "
                     "out_dtype='i8'",
                     spec->out_dtype,
                     arch,
                     target->family ? target->family : "(null)");
        }
        return false;
    }

    /* topk < 1 / experts < 1. */
    if(spec->topk < 1)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "topk must be >= 1 (got %d)", spec->topk);
        }
        return false;
    }
    if(spec->experts < 1)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "experts must be >= 1 (got %d)", spec->experts);
        }
        return false;
    }

    /* validate_io(IOSpecRule(dtype, block_size, vec, n_per_block, max=64)). */
    {
        rocke_arena_t tmp;
        rocke_io_spec_rule_t rule;
        const char* why = NULL;
        int ok;

        rocke_io_spec_rule_init(&rule, spec->dtype, spec->block_size, spec->vec);
        rule.n_per_block_set = 1;
        rule.n_per_block = spec->n_per_block;
        rule.max_elems_per_thread_set = 1;
        rule.max_elems_per_thread = 64;

        if(rocke_arena_init(&tmp, 0) != 0)
        {
            return false;
        }
        ok = rocke_validate_io(&tmp, &rule, &why);
        if(!ok)
        {
            if(reason != NULL && reason_cap > 0)
            {
                snprintf(reason, reason_cap, "%s", why ? why : "invalid io");
            }
            rocke_arena_destroy(&tmp);
            return false;
        }
        rocke_arena_destroy(&tmp);
    }

    /* block_size > max_threads_per_block. */
    if(spec->block_size > target->limits.max_threads_per_block)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "block_size %d > max_threads_per_block %d on %s",
                     spec->block_size,
                     target->limits.max_threads_per_block,
                     arch);
        }
        return false;
    }

    /* One f32 LDS reduction buffer of block_size words. */
    bytes_lds = (long)spec->block_size * 4;
    if(!rocke_arch_fits_lds(target, bytes_lds))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "LDS budget %ld > %d cap on %s",
                     bytes_lds,
                     target->lds_capacity_bytes,
                     arch);
        }
        return false;
    }

    return true;
}

/* --------------------------------------------------------------- build */

/* make_naive_tensor_view_packed for a packed row-major (1, N) view: strides are
 * (N, 1); build directly via rocke_make_global_view (packed when strides==NULL). */
static rocke_status_t rocke_moe_sq_make_packed_view(rocke_tensor_view_t* out,
                                                    rocke_value_t* base,
                                                    int n,
                                                    const rocke_type_t* dtype)
{
    int shape[2];
    shape[0] = 1;
    shape[1] = n;
    return rocke_make_global_view(out, base, shape, 2, dtype, NULL);
}

rocke_kernel_def_t* rocke_build_moe_smoothquant(rocke_ir_builder_t* b,
                                                const rocke_moe_smoothquant_spec_t* spec,
                                                const char* arch)
{
    char reason[256];
    const rocke_type_t* io_ty;
    const rocke_type_t* q_ty;
    double qmax;
    int BS, VEC, N, EPT, topk;
    const char* out_canon;

    rocke_value_t* X;
    rocke_value_t* SmScale;
    rocke_value_t* TopkIds;
    rocke_value_t* QY;
    rocke_value_t* YScale = NULL;
    rocke_value_t* tokens;
    rocke_value_t* eps;
    rocke_value_t* tid;
    rocke_value_t* out_row;

    rocke_value_t* i_topk;
    rocke_value_t* i_token;
    rocke_value_t* c_topk;
    rocke_value_t* topkids_idx;
    rocke_value_t* i_expert;
    rocke_value_t* sm_row_base;

    rocke_tensor_view_t x_view;
    rocke_tensor_view_t qy_view;
    rocke_tensor_view_t sm_view;
    rocke_tile_window_t x_tile;
    rocke_tile_window_t qy_tile;
    rocke_value_t* lds;

    rocke_tile_distribution_t* x_dist;
    rocke_value_t** cached = NULL; /* per-thread f32 X cache (k*VEC + i)   */
    int num_cached;

    rocke_value_t* s_amax;
    rocke_value_t* total_amax;
    rocke_value_t* safe_amax;
    rocke_value_t* yscale;
    rocke_value_t* inv_yscale;

    int chunks_p1, chunks, k, i;
    rocke_value_t* c_vec;

    if(b == NULL)
    {
        return NULL;
    }
    if(b->status != ROCKE_OK)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = ROCKE_MOE_SMOOTHQUANT_DEFAULT_ARCH;
    }

    if(!rocke_moe_smoothquant_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "invalid moe_smoothquant spec: %s", reason);
    }

    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    q_ty = rocke_b_quant_ir_type(b, spec->out_dtype);
    if(io_ty == NULL || q_ty == NULL)
    {
        return NULL;
    }
    out_canon = rocke_moe_sq_canon_out(spec->out_dtype);
    qmax = rocke_moe_sq_quant_max_abs(out_canon);

    BS = spec->block_size;
    VEC = spec->vec;
    N = spec->n_per_block;
    EPT = rocke_moe_smoothquant_elems_per_thread(spec);
    topk = spec->topk;

    /* b.kernel.attrs["max_workgroup_size"] = BS */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    /* --- params --- */
    {
        rocke_param_opts_t opts;

        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        X = rocke_b_param(b, "X", rocke_ptr_type(b, io_ty, "global"), &opts);

        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        SmScale = rocke_b_param(b, "SmScale", rocke_ptr_type(b, rocke_f32(), "global"), &opts);

        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 4;
        opts.align_set = true;
        TopkIds = rocke_b_param(b, "TopkIds", rocke_ptr_type(b, rocke_i32(), "global"), &opts);

        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        QY = rocke_b_param(b, "QY", rocke_ptr_type(b, q_ty, "global"), &opts);

        if(spec->save_yscale)
        {
            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.writeonly = true;
            opts.writeonly_set = true;
            opts.align = 4;
            opts.align_set = true;
            YScale = rocke_b_param(b, "YScale", rocke_ptr_type(b, rocke_f32(), "global"), &opts);
        }

        tokens = rocke_b_param(b, "tokens", rocke_i32(), NULL);
        (void)rocke_b_param(b, "N", rocke_i32(), NULL);
        eps = rocke_b_param(b, "eps", rocke_f32(), NULL);
    }

    tid = rocke_b_thread_id_x(b);
    out_row = rocke_b_block_id_x(b);

    /* Decode (i_topk, i_token) from out_row = i_topk * tokens + i_token.
     * spec.tokens_set => const divisor (folds to reciprocal-mul); else runtime. */
    if(spec->tokens_set)
    {
        rocke_value_t* c_tok_const = rocke_b_const_i32(b, spec->tokens);
        i_topk = rocke_b_div(b, out_row, c_tok_const);
        i_token = rocke_b_mod(b, out_row, c_tok_const);
    }
    else
    {
        i_topk = rocke_b_div(b, out_row, tokens);
        i_token = rocke_b_mod(b, out_row, tokens);
    }

    /* i_expert = to_sgpr_u32(global_load_i32(TopkIds, i_token*topk + i_topk)).
     * Python evaluates the mul before the add; sequence explicitly. */
    c_topk = rocke_b_const_i32(b, topk);
    {
        rocke_value_t* tk_mul = rocke_b_mul(b, i_token, c_topk);
        topkids_idx = rocke_b_add(b, tk_mul, i_topk);
    }
    i_expert = rocke_b_to_sgpr_u32(b, rocke_b_global_load_i32(b, TopkIds, topkids_idx, 4));

    /* --- views & tile windows --- */
    if(rocke_moe_sq_make_packed_view(&x_view, X, N, io_ty) != ROCKE_OK)
    {
        return NULL;
    }
    if(rocke_moe_sq_make_packed_view(&qy_view, QY, N, q_ty) != ROCKE_OK)
    {
        return NULL;
    }
    {
        /* SmScale flat (experts * N,) view. */
        int sm_shape[1];
        sm_shape[0] = spec->experts * N;
        if(rocke_make_global_view(&sm_view, SmScale, sm_shape, 1, rocke_f32(), NULL) != ROCKE_OK)
        {
            return NULL;
        }
    }
    {
        int lengths[2];
        rocke_value_t* x_origin[2];
        rocke_value_t* qy_origin[2];
        lengths[0] = 1;
        lengths[1] = N;
        /* x_tile origin row = i_token; qy_tile origin row = out_row. */
        x_origin[0] = i_token;
        x_origin[1] = rocke_b_const_i32(b, 0);
        if(rocke_make_tile_window(&x_tile, &x_view, lengths, x_origin, 2) != ROCKE_OK)
        {
            return NULL;
        }
        qy_origin[0] = out_row;
        qy_origin[1] = rocke_b_const_i32(b, 0);
        if(rocke_make_tile_window(&qy_tile, &qy_view, lengths, qy_origin, 2) != ROCKE_OK)
        {
            return NULL;
        }
    }

    /* sm_row_base = to_sgpr_u32(i_expert * N) -- hoist the per-expert row base. */
    sm_row_base = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, i_expert, rocke_b_const_i32(b, N)));

    /* LDS scratch for the block-wide amax reduction (block_size f32 words). */
    {
        int lds_shape[1];
        lds_shape[0] = BS;
        lds = rocke_b_smem_alloc(b, rocke_f32(), lds_shape, 1, "lds_amax");
        if(lds == NULL)
        {
            return NULL;
        }
    }

    /* --- pass 1: load X through the row distribution, fold the amax --- */
    x_dist = rocke_moe_sq_make_row_x_distribution(b, BS, VEC, EPT);
    if(x_dist == NULL)
    {
        return NULL;
    }

    /* x_dt = load_tile(b, x_tile, distribution=x_dist, ps=[[tid]]) */
    {
        const rocke_tile_distribution_encoding_t* enc = x_dist->encoding;
        int vec_dim_y, spv, num_acc, acc;
        rocke_static_distributed_tensor_t* x_dt
            = rocke_make_static_distributed_tensor(b, x_dist, io_ty);
        if(x_dt == NULL)
        {
            return NULL;
        }
        num_cached = x_dt->num_storage;
        cached = x_dt->storage;
        spv = rocke_moe_sq_pick_traits(enc, &vec_dim_y);
        num_acc = num_cached / (spv > 0 ? spv : 1);

        for(acc = 0; acc < num_acc; ++acc)
        {
            int y_base[3];
            rocke_value_t* ys[3];
            rocke_value_t* p0[1];
            rocke_value_t* const* ps[1];
            int ps_counts[1];
            rocke_value_t* x_coords[2];
            rocke_value_t* xs[ROCKE_MOE_SQ_MAX_VEC];
            int yy, j;
            y_base[0] = 0;
            y_base[1] = acc;
            y_base[2] = 0;
            for(yy = 0; yy < enc->num_Y; ++yy)
            {
                ys[yy] = rocke_b_const_i32(b, y_base[yy]);
            }
            p0[0] = tid;
            ps[0] = p0;
            ps_counts[0] = 1;
            if(!rocke_tile_distribution_calculate_x(
                   b, x_dist, ys, enc->num_Y, ps, ps_counts, 1, x_coords, 2))
            {
                return NULL;
            }
            rocke_tile_window_load_vec_as_f32(b, &x_tile, x_coords, 2, spv, xs);
            for(j = 0; j < spv; ++j)
            {
                int y_full[3];
                y_full[0] = y_base[0];
                y_full[1] = y_base[1];
                y_full[2] = y_base[2];
                y_full[vec_dim_y] = j;
                cached[rocke_moe_sq_y_to_linear(enc, y_full)] = xs[j];
            }
        }
    }

    chunks_p1 = EPT / VEC;

    /* s_amax = b.const_f32(0.0); per chunk fold the |y|=fmax(y,-y) tree.
     * sm_off = sm_row_base + n_off (per-expert gather).
     * Python emits s_amax (f32 0.0) BEFORE the pass-1 VEC const (c_vec_p1),
     * so build s_amax first to match the constant-emission order. */
    s_amax = rocke_b_const_f32(b, 0.0);
    c_vec = rocke_b_const_i32(b, VEC);
    for(k = 0; k < chunks_p1; ++k)
    {
        /* n_off = b.add(b.mul(const(k*BS), VEC), b.mul(tid, VEC)). */
        rocke_value_t* km = rocke_b_mul(b, rocke_b_const_i32(b, k * BS), c_vec);
        rocke_value_t* tm = rocke_b_mul(b, tid, c_vec);
        rocke_value_t* n_off = rocke_b_add(b, km, tm);
        rocke_value_t* sm_off = rocke_b_add(b, sm_row_base, n_off);
        rocke_value_t* sm_idx[1];
        rocke_value_t* sm_scalars[ROCKE_MOE_SQ_MAX_VEC];
        rocke_value_t* abs_ys[ROCKE_MOE_SQ_MAX_VEC];
        rocke_value_t* tf_scratch[ROCKE_MOE_SQ_MAX_VEC];
        rocke_value_t* chunk_amax;

        sm_idx[0] = sm_off;
        rocke_moe_sq_view_load_vec_as_f32(b, &sm_view, sm_idx, 1, VEC, sm_scalars);
        for(i = 0; i < VEC; ++i)
        {
            rocke_value_t* y = rocke_b_fmul(b, cached[k * VEC + i], sm_scalars[i]);
            abs_ys[i] = rocke_b_fmax(b, y, rocke_b_fneg(b, y));
        }
        chunk_amax = rocke_moe_sq_tree_fmax(b, abs_ys, VEC, tf_scratch);
        s_amax = rocke_b_fmax(b, s_amax, chunk_amax);
    }

    /* total_amax = block_lds_reduce(b, s_amax, lds, tid, block_size=BS, "max") */
    total_amax = rocke_block_lds_reduce(b, s_amax, lds, tid, BS, ROCKE_REDUCE_MAX);
    if(total_amax == NULL)
    {
        return NULL;
    }

    /* yscale = max(amax, eps) / quant_max; inv_yscale = 1 / yscale. */
    safe_amax = rocke_b_fmax(b, total_amax, eps);
    yscale = rocke_b_fmul(b, safe_amax, rocke_b_const_f32(b, 1.0 / qmax));
    inv_yscale = rocke_b_rcp(b, yscale);

    if(spec->save_yscale)
    {
        /* with b.scf_if(b.cmp_eq(tid, 0)): b.global_store(YScale, out_row, yscale, 4) */
        rocke_if_t gate = rocke_b_scf_if(b, rocke_b_cmp_eq(b, tid, rocke_b_const_i32(b, 0)));
        rocke_b_region_enter(b, gate.then_region);
        rocke_b_global_store(b, YScale, out_row, yscale, 4);
        rocke_b_region_leave(b);
    }

    /* --- pass 2: re-load SmScale, fuse multiply + quantise + store --- */
    chunks = EPT / VEC;
    c_vec = rocke_b_const_i32(b, VEC);

    if(VEC == 4 || VEC == 8)
    {
        /* use_packed_store path. */
        rocke_tile_distribution_t* qy_dist = rocke_moe_sq_make_row_x_distribution(b, BS, VEC, EPT);
        rocke_static_distributed_tensor_t* qy_dt;
        const rocke_tile_distribution_encoding_t* enc;
        int vec_dim_y, spv, num_acc, acc;
        const char* qname = out_canon;
        if(qy_dist == NULL)
        {
            return NULL;
        }
        qy_dt = rocke_make_static_distributed_tensor(b, qy_dist, q_ty);
        if(qy_dt == NULL)
        {
            return NULL;
        }
        enc = qy_dist->encoding;
        for(k = 0; k < chunks; ++k)
        {
            rocke_value_t* km = rocke_b_mul(b, rocke_b_const_i32(b, k * BS), c_vec);
            rocke_value_t* tm = rocke_b_mul(b, tid, c_vec);
            rocke_value_t* n_off = rocke_b_add(b, km, tm);
            rocke_value_t* sm_off = rocke_b_add(b, sm_row_base, n_off);
            rocke_value_t* sm_idx[1];
            rocke_value_t* sm_scalars[ROCKE_MOE_SQ_MAX_VEC];
            sm_idx[0] = sm_off;
            rocke_moe_sq_view_load_vec_as_f32(b, &sm_view, sm_idx, 1, VEC, sm_scalars);
            for(i = 0; i < VEC; ++i)
            {
                rocke_value_t* x_f32 = cached[k * VEC + i];
                rocke_value_t* y_f32 = rocke_b_fmul(b, x_f32, sm_scalars[i]);
                int y_full[3];
                y_full[0] = 0;
                y_full[1] = k;
                y_full[2] = i;
                /* qy_dt.set([0, k, i], y_f32 * inv_yscale) */
                qy_dt->storage[rocke_moe_sq_y_to_linear(enc, y_full)]
                    = rocke_b_fmul(b, y_f32, inv_yscale);
            }
        }

        /* store_tile(b, qy_tile, qy_dt, ps=[[tid]]): quant dtype path. */
        spv = rocke_moe_sq_pick_traits(enc, &vec_dim_y);
        num_acc = qy_dt->num_storage / (spv > 0 ? spv : 1);
        for(acc = 0; acc < num_acc; ++acc)
        {
            int y_base[3];
            rocke_value_t* ys[3];
            rocke_value_t* p0[1];
            rocke_value_t* const* ps[1];
            int ps_counts[1];
            rocke_value_t* x_coords[2];
            rocke_value_t* scalars[ROCKE_MOE_SQ_MAX_VEC];
            rocke_value_t* packed;
            int yy, j;
            y_base[0] = 0;
            y_base[1] = acc;
            y_base[2] = 0;
            for(yy = 0; yy < enc->num_Y; ++yy)
            {
                ys[yy] = rocke_b_const_i32(b, y_base[yy]);
            }
            p0[0] = tid;
            ps[0] = p0;
            ps_counts[0] = 1;
            if(!rocke_tile_distribution_calculate_x(
                   b, qy_dist, ys, enc->num_Y, ps, ps_counts, 1, x_coords, 2))
            {
                return NULL;
            }
            for(j = 0; j < spv; ++j)
            {
                int y_full[3];
                y_full[0] = y_base[0];
                y_full[1] = y_base[1];
                y_full[2] = y_base[2];
                y_full[vec_dim_y] = j;
                scalars[j] = qy_dt->storage[rocke_moe_sq_y_to_linear(enc, y_full)];
            }
            packed = rocke_moe_sq_pack_quant_local(b, scalars, spv, q_ty, qname);
            rocke_tile_window_store_vec(b, &qy_tile, x_coords, 2, packed, spv);
        }
    }
    else
    {
        /* VEC == 2: per-element scalar quant + store fallback. */
        for(k = 0; k < chunks; ++k)
        {
            rocke_value_t* km = rocke_b_mul(b, rocke_b_const_i32(b, k * BS), c_vec);
            rocke_value_t* tm = rocke_b_mul(b, tid, c_vec);
            rocke_value_t* n_off = rocke_b_add(b, km, tm);
            rocke_value_t* sm_off = rocke_b_add(b, sm_row_base, n_off);
            rocke_value_t* sm_idx[1];
            rocke_value_t* sm_scalars[ROCKE_MOE_SQ_MAX_VEC];
            sm_idx[0] = sm_off;
            rocke_moe_sq_view_load_vec_as_f32(b, &sm_view, sm_idx, 1, VEC, sm_scalars);
            for(i = 0; i < VEC; ++i)
            {
                rocke_value_t* x_f32 = cached[k * VEC + i];
                rocke_value_t* y_f32 = rocke_b_fmul(b, x_f32, sm_scalars[i]);
                rocke_value_t* q
                    = rocke_moe_sq_quantize_scalar_f32(b, y_f32, inv_yscale, spec->out_dtype);
                rocke_value_t* col = rocke_b_add(b, n_off, rocke_b_const_i32(b, i));
                rocke_value_t* local_idx[2];
                local_idx[0] = rocke_b_const_i32(b, 0);
                local_idx[1] = col;
                /* qy_tile.store_scalar(b, 0, col, value=q) */
                rocke_tile_window_store_scalar(b, &qy_tile, local_idx, 2, q, 0);
            }
        }
    }

    if(b->status != ROCKE_OK)
    {
        return NULL;
    }
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_moe_smoothquant_new(rocke_ir_builder_t* b,
                                                    const rocke_moe_smoothquant_spec_t* spec,
                                                    const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[ROCKE_MOE_SQ_NAME_CAP];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_moe_smoothquant_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_moe_smoothquant(b, spec, arch);
    });
}

/* --------------------------------------------------------------- grid */

rocke_status_t
    rocke_moe_smoothquant_grid(int tokens, const rocke_moe_smoothquant_spec_t* spec, int out[3])
{
    int totals[1];
    int tiles[1];
    if(out == NULL || spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* ceil_div_grid((tokens * topk, 1)) -- one CTA per output row. */
    totals[0] = tokens * spec->topk;
    tiles[0] = 1;
    return rocke_ceil_div_grid(totals, tiles, 1, out);
}

/* --------------------------------------------------------------- lower */

rocke_status_t rocke_moe_smoothquant_lower_to_llvm(const rocke_moe_smoothquant_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        if(err != NULL && err_cap > 0)
        {
            snprintf(err, err_cap, "lower_to_llvm: null spec/out");
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = ROCKE_MOE_SMOOTHQUANT_DEFAULT_ARCH;
    }

    kernel = rocke_build_moe_smoothquant_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            snprintf(err, err_cap, "%s", m ? m : "build_moe_smoothquant failed");
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
