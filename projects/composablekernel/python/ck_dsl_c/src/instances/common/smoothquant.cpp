// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_smoothquant.c -- C99 port of ck_dsl/instances/common/smoothquant.py.
 *
 * Byte-identical builder-call sequence vs the Python build_smoothquant: same op
 * order / attrs. Higher-level helpers this instance leans on that are not yet in
 * the C helper set (distribution.load_tile / store_tile /
 * make_static_distributed_tensor; tensor_view.make_naive_tensor_view_packed /
 * make_lds_view; the F32-view load_vec_as_f32) are wired through local shims; the
 * heaviest distribution paths are STUBBED with TODO(port) markers that the
 * verify+fix loop resolves once those helpers land. The pass arithmetic (the
 * _tree_fmax amax fold, the block_lds_reduce, the yscale recipe, and the
 * quantize_scalar_f32 chain) is ported faithfully.
 */
#include "ckc/instance_smoothquant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/helper_ck_dsl.core.arch.h"
#include "ckc/helper_ck_dsl.helpers.distribution.h"
#include "ckc/helper_ck_dsl.helpers.io.h"
#include "ckc/helper_ck_dsl.helpers.quant.h"
#include "ckc/helper_ck_dsl.helpers.reduction.h"
#include "ckc/helper_ck_dsl.helpers.spec.h"
#include "ckc/helper_ck_dsl.helpers.sweep.h"
#include "ckc/helper_ck_dsl.helpers.tensor_view.h"
#include "ckc/ir.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err, ckc_i_live */
#include "ckc/lower_llvm.h"
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */

/* --------------------------------------------------------------- locals */

#define CKC_SQ_NAME_CAP 256
#define CKC_SQ_MAX_VEC 8

/* QUANT_MAX_ABS table (helpers/quant.py). i8=127, fp8e4m3=448, bf8e5m2=57344. */
static double ckc_sq_quant_max_abs(const char* out_dtype)
{
    if(out_dtype == NULL)
    {
        return 0.0;
    }
    if(strcmp(out_dtype, "i8") == 0 || strcmp(out_dtype, "int8") == 0)
    {
        return 127.0;
    }
    if(strcmp(out_dtype, "fp8e4m3") == 0 || strcmp(out_dtype, "fp8") == 0 ||
       strcmp(out_dtype, "fp8_e4m3") == 0)
    {
        return 448.0;
    }
    if(strcmp(out_dtype, "bf8e5m2") == 0 || strcmp(out_dtype, "bf8") == 0 ||
       strcmp(out_dtype, "fp8_e5m2") == 0)
    {
        return 57344.0;
    }
    return 0.0;
}

/* Canonicalised out-dtype tag for quantize dispatch ("i8"/"fp8e4m3"/"bf8e5m2"). */
static const char* ckc_sq_canon_out(const char* out_dtype)
{
    if(out_dtype == NULL)
    {
        return NULL;
    }
    if(strcmp(out_dtype, "i8") == 0 || strcmp(out_dtype, "int8") == 0)
    {
        return "i8";
    }
    if(strcmp(out_dtype, "fp8e4m3") == 0 || strcmp(out_dtype, "fp8") == 0 ||
       strcmp(out_dtype, "fp8_e4m3") == 0)
    {
        return "fp8e4m3";
    }
    if(strcmp(out_dtype, "bf8e5m2") == 0 || strcmp(out_dtype, "bf8") == 0 ||
       strcmp(out_dtype, "fp8_e5m2") == 0)
    {
        return "bf8e5m2";
    }
    return NULL;
}

/* quantize_scalar_f32(b, x_f32, inv_scale, qdtype) -- helpers/quant.py.
 *
 *   scaled  = b.fmul(x_f32, inv_scale)
 *   clamped = b.clamp_f32(scaled, -qmax, +qmax)
 *   result  = cvt_f32_to_<qdtype>(clamped)
 *
 * The default skip_clamp_on_pack=False path is used by smoothquant's VEC==2
 * scalar fallback, so the explicit clamp is always emitted. */
static ckc_value_t* ckc_sq_quantize_scalar_f32(ckc_ir_builder_t* b,
                                               ckc_value_t* x_f32,
                                               ckc_value_t* inv_scale,
                                               const char* qdtype)
{
    const char* canon;
    double qmax;
    ckc_value_t* c_pos;
    ckc_value_t* c_neg;
    ckc_value_t* scaled;
    ckc_value_t* clamped;

    if(b != NULL && b->status != CKC_OK)
    {
        return NULL;
    }
    canon = ckc_sq_canon_out(qdtype);
    if(canon == NULL)
    {
        return (ckc_value_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "quantize_scalar_f32: unsupported qdtype");
    }
    qmax    = ckc_sq_quant_max_abs(canon);
    c_pos   = ckc_b_const_f32(b, qmax);
    c_neg   = ckc_b_const_f32(b, -qmax);
    scaled  = ckc_b_fmul(b, x_f32, inv_scale);
    clamped = ckc_b_clamp_f32(b, scaled, c_neg, c_pos);
    if(strcmp(canon, "i8") == 0)
    {
        return ckc_b_cvt_f32_to_i8_sat(b, clamped);
    }
    if(strcmp(canon, "fp8e4m3") == 0)
    {
        return ckc_b_cvt_f32_to_fp8(b, clamped);
    }
    /* bf8e5m2 */
    return ckc_b_cvt_f32_to_bf8(b, clamped);
}

/* _tree_fmax(b, values) -- balanced pairwise fmax tree (smoothquant.py). The
 * critical path is ceil(log2(n)) fmax ops; total op count matches a left-fold.
 * Mirrors the Python loop exactly (pair cur[i]/cur[i+1] left-to-right, carry the
 * odd tail forward). Returns NULL for n<1. `scratch` is caller-provided working
 * storage of length >= n (two ping-pong halves fit in n). */
static ckc_value_t*
ckc_sq_tree_fmax(ckc_ir_builder_t* b, ckc_value_t** values, int n, ckc_value_t** scratch)
{
    ckc_value_t** cur;
    int cur_n;

    if(b != NULL && b->status != CKC_OK)
    {
        return NULL;
    }
    if(n < 1)
    {
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "_tree_fmax: empty");
    }
    /* cur = list(values) */
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
            cur[w++] = ckc_b_fmax(b, cur[i], cur[i + 1]);
        }
        if(cur_n % 2)
        {
            cur[w++] = cur[cur_n - 1];
        }
        cur_n = w;
    }
    return cur[0];
}

/* make_row_x_distribution(block_size, vec, elems_per_thread) -- smoothquant.py.
 * Builds the per-row activation X tile distribution via the canonical encoding:
 *
 *   Hs            = ((1,), (chunks, block_size, vec))
 *   Ps2RHs_major  = ((2,),)
 *   Ps2RHs_minor  = ((1,),)
 *   Ys2RHs_major  = (1, 2, 2)
 *   Ys2RHs_minor  = (0, 0, 2)
 *
 * Emits NO IR (pure host-side analysis). Returns NULL on a validation failure
 * (sticky error set on b). */
static ckc_tile_distribution_t*
ckc_sq_make_row_x_distribution(ckc_ir_builder_t* b, int block_size, int vec, int elems_per_thread)
{
    int chunks = elems_per_thread / vec;
    ckc_tile_distribution_encoding_t* enc;

    int h0_levels[1];
    int h1_levels[3];
    ckc_h_row_t Hs[2];

    int p0_major[1];
    int p0_minor[1];
    ckc_p_seq_t Ps[1];

    int Ys_major[3];
    int Ys_minor[3];

    if(b != NULL && b->status != CKC_OK)
    {
        return NULL;
    }

    /* Hs[0] = (1,), Hs[1] = (chunks, block_size, vec). */
    h0_levels[0] = 1;
    h1_levels[0] = chunks;
    h1_levels[1] = block_size;
    h1_levels[2] = vec;
    Hs[0].levels = h0_levels;
    Hs[0].count  = 1;
    Hs[1].levels = h1_levels;
    Hs[1].count  = 3;

    /* P0 -> X1 level 1 (the block_size / ThreadPerWarp dim). */
    p0_major[0] = 2;
    p0_minor[0] = 1;
    Ps[0].major = p0_major;
    Ps[0].minor = p0_minor;
    Ps[0].count = 1;

    /* Y0->X0 level0; Y1->X1 level0 (Repeat); Y2->X1 level2 (Vector). */
    Ys_major[0] = 1;
    Ys_major[1] = 2;
    Ys_major[2] = 2;
    Ys_minor[0] = 0;
    Ys_minor[1] = 0;
    Ys_minor[2] = 2;

    enc = ckc_make_tile_distribution_encoding(b,
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
    return ckc_make_static_tile_distribution(b, enc);
}

/* --------------------------------------------------------------- load/store tile
 *
 * Faithful inline port of distribution.py load_tile / store_tile + the
 * LoadStoreTraits picker (make_load_store_traits) and StaticDistributedTensor
 * storage layout, specialised to the SmoothQuant row distribution. Reproduces
 * the exact SSA op order (calculate_x index reconstruction + load_vec_as_f32 /
 * _pack_quant_local + store_vec) so the emitted .ll is byte-identical to the
 * Python emitter. The C primitives used (ckc_tile_distribution_calculate_x,
 * ckc_make_static_distributed_tensor, ckc_b_vec_pack, the cvt ops) already
 * exist in the helper layer; only the load_tile / store_tile orchestration was
 * missing, so it lives here under the "instance files only" scope. */

/* Per-Y-dim length from the encoding (Hs[major-1][minor]; major==0 => Rs). */
static int ckc_sq_y_length(const ckc_tile_distribution_encoding_t* enc, int y)
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
static int ckc_sq_y_x_stride(const ckc_tile_distribution_encoding_t* enc, int y)
{
    int major = enc->Ys_major[y];
    int minor = enc->Ys_minor[y];
    const ckc_h_row_t* h;
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

/* make_load_store_traits picker (max_vec=8, min_vec=1): choose vector_dim_y +
 * scalar_per_vector. Returns scalar_per_vector; writes vector_dim_y. */
static int ckc_sq_pick_traits(const ckc_tile_distribution_encoding_t* enc, int* vector_dim_y)
{
    int num_Y  = enc->num_Y;
    int best_y = -1, best_len = -1;
    int y, spv, full_len;
    /* Largest length among stride-1 candidates; ties -> highest Y index. */
    for(y = 0; y < num_Y; ++y)
    {
        if(ckc_sq_y_x_stride(enc, y) == 1)
        {
            int len = ckc_sq_y_length(enc, y);
            if(len >= best_len) /* >= so ties pick the higher Y index */
            {
                best_len = len;
                best_y   = y;
            }
        }
    }
    if(best_y < 0)
    {
        *vector_dim_y = num_Y - 1;
        return 1;
    }
    full_len = best_len;
    spv      = full_len < 8 ? full_len : 8;
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
static int ckc_sq_y_to_linear(const ckc_tile_distribution_encoding_t* enc, const int* y)
{
    int off = 0;
    int i;
    for(i = 0; i < enc->num_Y; ++i)
    {
        off = off * ckc_sq_y_length(enc, i) + y[i];
    }
    return off;
}

/* _pack_quant_local: quantise + pack scaled-f32 scalars into <n x q_ty>. */
static ckc_value_t* ckc_sq_pack_quant_local(ckc_ir_builder_t* b,
                                            ckc_value_t* const* scaled,
                                            int n,
                                            const ckc_type_t* q_ty,
                                            const char* qname)
{
    int off, i;
    if((strcmp(qname, "fp8e4m3") == 0 || strcmp(qname, "bf8e5m2") == 0) && (n % 4) == 0)
    {
        ckc_value_t* out = NULL;
        for(off = 0; off < n; off += 4)
        {
            ckc_value_t* quad  = ckc_b_vec_pack(b, &scaled[off], 4, ckc_f32());
            ckc_value_t* chunk = (strcmp(qname, "fp8e4m3") == 0) ? ckc_b_cvt_pk_fp8_f32x4(b, quad)
                                                                 : ckc_b_cvt_pk_bf8_f32x4(b, quad);
            out                = (out == NULL) ? chunk : ckc_b_vec_concat(b, out, chunk);
        }
        return out;
    }
    {
        ckc_value_t* qs[CKC_SQ_MAX_VEC];
        for(i = 0; i < n; ++i)
        {
            if(strcmp(qname, "i8") == 0)
            {
                ckc_value_t* c_neg = ckc_b_const_f32(b, -127.0);
                ckc_value_t* c_pos = ckc_b_const_f32(b, 127.0);
                ckc_value_t* cl    = ckc_b_clamp_f32(b, scaled[i], c_neg, c_pos);
                qs[i]              = ckc_b_cvt_f32_to_i8_sat(b, cl);
            }
            else if(strcmp(qname, "fp8e4m3") == 0)
            {
                qs[i] = ckc_b_cvt_f32_to_fp8(b, scaled[i]);
            }
            else /* bf8e5m2 */
            {
                qs[i] = ckc_b_cvt_f32_to_bf8(b, scaled[i]);
            }
        }
        return ckc_b_vec_pack(b, qs, n, q_ty);
    }
}

/* SmScale F32-view load_vec_as_f32 (tensor_view.py): for a f32 view the per-lane
 * cast is a no-op; just vec_load + vec_extract. n in {2,4,8}; n==1 routes
 * through a scalar load. Writes the n scalars to out[0..n). */
static void ckc_sq_view_load_vec_as_f32(ckc_ir_builder_t* b,
                                        const ckc_tensor_view_t* v,
                                        ckc_value_t* const* indices,
                                        int num_indices,
                                        int n,
                                        ckc_value_t** out)
{
    if(b != NULL && b->status != CKC_OK)
    {
        return;
    }
    if(n == 1)
    {
        out[0] = ckc_tensor_view_load_scalar(b, v, indices, num_indices);
        return;
    }
    {
        ckc_value_t* vec = ckc_tensor_view_load_vec(b, v, indices, num_indices, n);
        for(int i = 0; i < n; ++i)
        {
            out[i] = ckc_b_vec_extract(b, vec, i);
        }
    }
}

/* --------------------------------------------------------------- spec */

void ckc_smoothquant_spec_init(ckc_smoothquant_spec_t* spec, int n_per_block)
{
    if(spec == NULL)
    {
        return;
    }
    spec->n_per_block = n_per_block;
    spec->dtype       = "f16";
    spec->out_dtype   = "i8";
    spec->block_size  = 256;
    spec->vec         = 4;
    spec->save_yscale = true;
    spec->wave_size   = 64;
    spec->name        = "ck_dsl_smoothquant";
}

int ckc_smoothquant_elems_per_thread(const ckc_smoothquant_spec_t* spec)
{
    if(spec == NULL || spec->block_size == 0)
    {
        return 0;
    }
    return spec->n_per_block / spec->block_size;
}

ckc_status_t
ckc_smoothquant_kernel_name(const ckc_smoothquant_spec_t* spec, char* out, size_t out_cap)
{
    char nbuf[32];
    char bbuf[32];
    char vbuf[32];
    const char* parts[5];
    const char* flag_names[1];
    int flag_on[1];

    if(spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    /* kernel_name_join(name, dtype, out_dtype, "N{n}", "b{bs}", "v{vec}",
     *                  flags={"ys": save_yscale}) */
    snprintf(nbuf, sizeof(nbuf), "N%d", spec->n_per_block);
    snprintf(bbuf, sizeof(bbuf), "b%d", spec->block_size);
    snprintf(vbuf, sizeof(vbuf), "v%d", spec->vec);
    parts[0]      = spec->dtype;
    parts[1]      = spec->out_dtype;
    parts[2]      = nbuf;
    parts[3]      = bbuf;
    parts[4]      = vbuf;
    flag_names[0] = "ys";
    flag_on[0]    = spec->save_yscale ? 1 : 0;
    return ckc_kernel_name_join(spec->name, parts, 5, flag_names, flag_on, 1, out, out_cap, NULL);
}

/* --------------------------------------------------------------- valid */

bool ckc_smoothquant_is_valid_spec(const ckc_smoothquant_spec_t* spec,
                                   const char* arch,
                                   char* reason,
                                   size_t reason_cap)
{
    const ckc_arch_target_t* target;
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
        arch = CKC_SMOOTHQUANT_DEFAULT_ARCH;
    }

    /* target = ArchTarget.from_gfx(arch); KeyError -> (False, msg). */
    target = ckc_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "unknown arch %s", arch);
        }
        return false;
    }

    /* out_dtype not in ("i8","fp8e4m3","bf8e5m2"). */
    canon = ckc_sq_canon_out(spec->out_dtype);
    if(canon == NULL ||
       (strcmp(canon, "i8") != 0 && strcmp(canon, "fp8e4m3") != 0 && strcmp(canon, "bf8e5m2") != 0))
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
    if((strcmp(canon, "fp8e4m3") == 0 || strcmp(canon, "bf8e5m2") == 0) &&
       (target->family == NULL || strcmp(target->family, "cdna") != 0))
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

    /* validate_io(IOSpecRule(dtype, block_size, vec, n_per_block, max=64)). */
    {
        ckc_arena_t tmp;
        ckc_io_spec_rule_t rule;
        const char* why = NULL;
        int ok;

        ckc_io_spec_rule_init(&rule, spec->dtype, spec->block_size, spec->vec);
        rule.n_per_block_set          = 1;
        rule.n_per_block              = spec->n_per_block;
        rule.max_elems_per_thread_set = 1;
        rule.max_elems_per_thread     = 64;

        if(ckc_arena_init(&tmp, 0) != 0)
        {
            return false;
        }
        ok = ckc_validate_io(&tmp, &rule, &why);
        if(!ok)
        {
            if(reason != NULL && reason_cap > 0)
            {
                snprintf(reason, reason_cap, "%s", why ? why : "invalid io");
            }
            ckc_arena_destroy(&tmp);
            return false;
        }
        ckc_arena_destroy(&tmp);
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
    if(!ckc_arch_fits_lds(target, bytes_lds))
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

/* STUB(port): make_naive_tensor_view_packed is not yet in the C helper set.
 * For a packed row-major (1, N) view the descriptor strides are (N, 1); build it
 * directly via ckc_make_global_view (which produces packed row-major strides
 * when strides==NULL). This matches the Python make_naive_tensor_view_packed for
 * the rank-2 packed case smoothquant uses. */
static ckc_status_t
ckc_sq_make_packed_view(ckc_tensor_view_t* out, ckc_value_t* base, int n, const ckc_type_t* dtype)
{
    int shape[2];
    shape[0] = 1;
    shape[1] = n;
    return ckc_make_global_view(out, base, shape, 2, dtype, NULL);
}

ckc_kernel_def_t*
ckc_build_smoothquant(ckc_ir_builder_t* b, const ckc_smoothquant_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char reason[256];
        const ckc_type_t* io_ty;
        const ckc_type_t* q_ty;
        double qmax;
        int BS, VEC, N, EPT;
        const char* out_canon;

        ckc_value_t* X;
        ckc_value_t* SmScale;
        ckc_value_t* QY;
        ckc_value_t* YScale = NULL;
        ckc_value_t* eps;
        ckc_value_t* tid;
        ckc_value_t* row;

        ckc_tensor_view_t x_view;
        ckc_tensor_view_t qy_view;
        ckc_tensor_view_t sm_view;
        ckc_tile_window_t x_tile;
        ckc_tile_window_t qy_tile;
        ckc_value_t* lds;

        ckc_tile_distribution_t* x_dist;
        ckc_value_t** cached = NULL; /* per-thread f32 X cache (k*VEC + i)   */
        int num_cached;

        ckc_value_t* s_amax;
        ckc_value_t* total_amax;
        ckc_value_t* safe_amax;
        ckc_value_t* yscale;
        ckc_value_t* inv_yscale;

        int chunks_p1, chunks, k, i;
        ckc_value_t* c_vec;

        if(b == NULL)
        {
            return NULL;
        }
        if(b->status != CKC_OK)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = CKC_SMOOTHQUANT_DEFAULT_ARCH;
        }

        if(!ckc_smoothquant_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            return (ckc_kernel_def_t*)ckc_i_set_err(
                b, CKC_ERR_VALUE, "invalid smoothquant spec: %s", reason);
        }

        io_ty = ckc_b_io_ir_type(b, spec->dtype);
        q_ty  = ckc_b_quant_ir_type(b, spec->out_dtype);
        if(io_ty == NULL || q_ty == NULL)
        {
            return NULL;
        }
        out_canon = ckc_sq_canon_out(spec->out_dtype);
        qmax      = ckc_sq_quant_max_abs(out_canon);

        BS  = spec->block_size;
        VEC = spec->vec;
        N   = spec->n_per_block;
        EPT = ckc_smoothquant_elems_per_thread(spec);

        /* b.kernel.attrs["max_workgroup_size"] = BS */
        ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

        /* --- params --- */
        {
            ckc_param_opts_t opts;

            memset(&opts, 0, sizeof(opts));
            opts.noalias      = true;
            opts.noalias_set  = true;
            opts.readonly     = true;
            opts.readonly_set = true;
            opts.align        = 16;
            opts.align_set    = true;
            X                 = ckc_b_param(b, "X", ckc_ptr_type(b, io_ty, "global"), &opts);

            memset(&opts, 0, sizeof(opts));
            opts.noalias      = true;
            opts.noalias_set  = true;
            opts.readonly     = true;
            opts.readonly_set = true;
            opts.align        = 16;
            opts.align_set    = true;
            SmScale = ckc_b_param(b, "SmScale", ckc_ptr_type(b, ckc_f32(), "global"), &opts);

            memset(&opts, 0, sizeof(opts));
            opts.noalias       = true;
            opts.noalias_set   = true;
            opts.writeonly     = true;
            opts.writeonly_set = true;
            opts.align         = 16;
            opts.align_set     = true;
            QY                 = ckc_b_param(b, "QY", ckc_ptr_type(b, q_ty, "global"), &opts);

            if(spec->save_yscale)
            {
                memset(&opts, 0, sizeof(opts));
                opts.noalias       = true;
                opts.noalias_set   = true;
                opts.writeonly     = true;
                opts.writeonly_set = true;
                opts.align         = 4;
                opts.align_set     = true;
                YScale = ckc_b_param(b, "YScale", ckc_ptr_type(b, ckc_f32(), "global"), &opts);
            }

            (void)ckc_b_param(b, "M", ckc_i32(), NULL);
            (void)ckc_b_param(b, "N", ckc_i32(), NULL);
            eps = ckc_b_param(b, "eps", ckc_f32(), NULL);
        }

        tid = ckc_b_thread_id_x(b);
        row = ckc_b_block_id_x(b);

        /* --- views & tile windows --- */
        if(ckc_sq_make_packed_view(&x_view, X, N, io_ty) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_sq_make_packed_view(&qy_view, QY, N, q_ty) != CKC_OK)
        {
            return NULL;
        }
        {
            int sm_shape[1];
            sm_shape[0] = N;
            if(ckc_make_global_view(&sm_view, SmScale, sm_shape, 1, ckc_f32(), NULL) != CKC_OK)
            {
                return NULL;
            }
        }
        {
            int lengths[2];
            ckc_value_t* x_origin[2];
            ckc_value_t* qy_origin[2];
            lengths[0]  = 1;
            lengths[1]  = N;
            x_origin[0] = row;
            x_origin[1] = ckc_b_const_i32(b, 0);
            if(ckc_make_tile_window(&x_tile, &x_view, lengths, x_origin, 2) != CKC_OK)
            {
                return NULL;
            }
            qy_origin[0] = row;
            qy_origin[1] = ckc_b_const_i32(b, 0);
            if(ckc_make_tile_window(&qy_tile, &qy_view, lengths, qy_origin, 2) != CKC_OK)
            {
                return NULL;
            }
        }

        /* LDS scratch for the block-wide amax reduction (block_size f32 words).
         *
         * STUB(port): make_lds_view(...).base is not yet ported; the smoothquant
         * use only needs a block_size f32 LDS token, which ckc_b_smem_alloc
         * provides directly. */
        {
            int lds_shape[1];
            lds_shape[0] = BS;
            lds          = ckc_b_smem_alloc(b, ckc_f32(), lds_shape, 1, "lds_amax");
            if(lds == NULL)
            {
                return NULL;
            }
        }

        /* --- pass 1: load X through the row distribution, fold the amax --- */
        x_dist = ckc_sq_make_row_x_distribution(b, BS, VEC, EPT);
        if(x_dist == NULL)
        {
            return NULL;
        }

        /* x_dt = load_tile(b, x_tile, distribution=x_dist, ps=[[tid]])
         *
         * Faithful load_tile: traits pick vector_dim_y=2 / scalar_per_vector=VEC;
         * iterate_accesses yields y_base=(0,k,0) row-major (the snake fold is a
         * no-op since the only non-vector outer axis with len>1 is Y1, whose
         * parity gate sums the always-zero Y0). For each access calculate_x emits
         * the n_off reconstruction, then load_vec_as_f32 promotes the f16/bf16
         * lanes to f32. The scalars land in the StaticDistributedTensor storage at
         * y_to_linear((0,k,j)) = k*VEC + j -- the same cache layout pass 1/2 use. */
        {
            const ckc_tile_distribution_encoding_t* enc = x_dist->encoding;
            int vec_dim_y, spv, num_acc, acc;
            ckc_static_distributed_tensor_t* x_dt =
                ckc_make_static_distributed_tensor(b, x_dist, io_ty);
            if(x_dt == NULL)
            {
                return NULL;
            }
            num_cached = x_dt->num_storage;
            cached     = x_dt->storage;
            spv        = ckc_sq_pick_traits(enc, &vec_dim_y);
            num_acc    = num_cached / (spv > 0 ? spv : 1);

            for(acc = 0; acc < num_acc; ++acc)
            {
                int y_base[3];
                ckc_value_t* ys[3];
                ckc_value_t* p0[1];
                ckc_value_t* const* ps[1];
                int ps_counts[1];
                ckc_value_t* x_coords[2];
                ckc_value_t* xs[CKC_SQ_MAX_VEC];
                int yy, j;
                /* non-vector axis is Y1 (chunk index k = acc); Y0 (M) is len 1. */
                y_base[0] = 0;
                y_base[1] = acc;
                y_base[2] = 0;
                for(yy = 0; yy < enc->num_Y; ++yy)
                {
                    ys[yy] = ckc_b_const_i32(b, y_base[yy]);
                }
                p0[0]        = tid;
                ps[0]        = p0;
                ps_counts[0] = 1;
                if(!ckc_tile_distribution_calculate_x(
                       b, x_dist, ys, enc->num_Y, ps, ps_counts, 1, x_coords, 2))
                {
                    return NULL;
                }
                ckc_tile_window_load_vec_as_f32(b, &x_tile, x_coords, 2, spv, xs);
                for(j = 0; j < spv; ++j)
                {
                    int y_full[3];
                    y_full[0]                               = y_base[0];
                    y_full[1]                               = y_base[1];
                    y_full[2]                               = y_base[2];
                    y_full[vec_dim_y]                       = j;
                    cached[ckc_sq_y_to_linear(enc, y_full)] = xs[j];
                }
            }
        }

        chunks_p1 = EPT / VEC;

        /* s_amax = b.const_f32(0.0); per chunk fold the |y|=fmax(y,-y) tree.
         * Python emits s_amax (f32 0.0) BEFORE the pass-1 VEC const (c_vec_p1),
         * so build s_amax first to match the constant-emission order. */
        s_amax = ckc_b_const_f32(b, 0.0);
        c_vec  = ckc_b_const_i32(b, VEC);
        for(k = 0; k < chunks_p1; ++k)
        {
            /* n_off = b.add(b.mul(const(k*BS), VEC), b.mul(tid, VEC)).
             * Sequence the two muls explicitly: Python evaluates the const-mul
             * before the tid-mul, and C arg-eval order is unspecified. */
            ckc_value_t* km    = ckc_b_mul(b, ckc_b_const_i32(b, k * BS), c_vec);
            ckc_value_t* tm    = ckc_b_mul(b, tid, c_vec);
            ckc_value_t* n_off = ckc_b_add(b, km, tm);
            ckc_value_t* sm_idx[1];
            ckc_value_t* sm_scalars[CKC_SQ_MAX_VEC];
            ckc_value_t* abs_ys[CKC_SQ_MAX_VEC];
            ckc_value_t* tf_scratch[CKC_SQ_MAX_VEC];
            ckc_value_t* chunk_amax;

            sm_idx[0] = n_off;
            ckc_sq_view_load_vec_as_f32(b, &sm_view, sm_idx, 1, VEC, sm_scalars);
            for(i = 0; i < VEC; ++i)
            {
                ckc_value_t* y = ckc_b_fmul(b, cached[k * VEC + i], sm_scalars[i]);
                abs_ys[i]      = ckc_b_fmax(b, y, ckc_b_fneg(b, y));
            }
            chunk_amax = ckc_sq_tree_fmax(b, abs_ys, VEC, tf_scratch);
            s_amax     = ckc_b_fmax(b, s_amax, chunk_amax);
        }

        /* total_amax = block_lds_reduce(b, s_amax, lds, tid, block_size=BS, "max") */
        total_amax = ckc_block_lds_reduce(b, s_amax, lds, tid, BS, CKC_REDUCE_MAX);
        if(total_amax == NULL)
        {
            return NULL;
        }

        /* yscale = max(amax, eps) / quant_max; inv_yscale = 1 / yscale. */
        safe_amax  = ckc_b_fmax(b, total_amax, eps);
        yscale     = ckc_b_fmul(b, safe_amax, ckc_b_const_f32(b, 1.0 / qmax));
        inv_yscale = ckc_b_rcp(b, yscale);

        if(spec->save_yscale)
        {
            /* with b.scf_if(b.cmp_eq(tid, 0)): b.global_store(YScale, row, yscale, 4) */
            ckc_if_t gate = ckc_b_scf_if(b, ckc_b_cmp_eq(b, tid, ckc_b_const_i32(b, 0)));
            ckc_b_region_enter(b, gate.then_region);
            ckc_b_global_store(b, YScale, row, yscale, 4);
            ckc_b_region_leave(b);
        }

        /* --- pass 2: re-load SmScale, fuse multiply + quantise + store --- */
        chunks = EPT / VEC;
        c_vec  = ckc_b_const_i32(b, VEC);

        if(VEC == 4 || VEC == 8)
        {
            /* use_packed_store path: build a QY distribution identical to X, fill a
             * StaticDistributedTensor of dtype q_ty with the already-scaled f32
             * chunk values (the SmScale reload uses the same n_off addressing as
             * pass 1, NOT the distribution), then store_tile applies the
             * saturating cvt + vec_pack + one coalesced global_store_vN per
             * access. Faithful inline port of distribution.py store_tile. */
            ckc_tile_distribution_t* qy_dist = ckc_sq_make_row_x_distribution(b, BS, VEC, EPT);
            ckc_static_distributed_tensor_t* qy_dt;
            const ckc_tile_distribution_encoding_t* enc;
            int vec_dim_y, spv, num_acc, acc;
            const char* qname = out_canon;
            if(qy_dist == NULL)
            {
                return NULL;
            }
            qy_dt = ckc_make_static_distributed_tensor(b, qy_dist, q_ty);
            if(qy_dt == NULL)
            {
                return NULL;
            }
            enc = qy_dist->encoding;
            for(k = 0; k < chunks; ++k)
            {
                /* n_off: const-mul before tid-mul (Python order). */
                ckc_value_t* km    = ckc_b_mul(b, ckc_b_const_i32(b, k * BS), c_vec);
                ckc_value_t* tm    = ckc_b_mul(b, tid, c_vec);
                ckc_value_t* n_off = ckc_b_add(b, km, tm);
                ckc_value_t* sm_idx[1];
                ckc_value_t* sm_scalars[CKC_SQ_MAX_VEC];
                sm_idx[0] = n_off;
                ckc_sq_view_load_vec_as_f32(b, &sm_view, sm_idx, 1, VEC, sm_scalars);
                for(i = 0; i < VEC; ++i)
                {
                    ckc_value_t* x_f32 = cached[k * VEC + i];
                    ckc_value_t* y_f32 = ckc_b_fmul(b, x_f32, sm_scalars[i]);
                    int y_full[3];
                    y_full[0] = 0;
                    y_full[1] = k;
                    y_full[2] = i;
                    /* qy_dt.set([0, k, i], y_f32 * inv_yscale) */
                    qy_dt->storage[ckc_sq_y_to_linear(enc, y_full)] =
                        ckc_b_fmul(b, y_f32, inv_yscale);
                }
            }

            /* store_tile(b, qy_tile, qy_dt, ps=[[tid]]): quant dtype path. */
            spv     = ckc_sq_pick_traits(enc, &vec_dim_y);
            num_acc = qy_dt->num_storage / (spv > 0 ? spv : 1);
            for(acc = 0; acc < num_acc; ++acc)
            {
                int y_base[3];
                ckc_value_t* ys[3];
                ckc_value_t* p0[1];
                ckc_value_t* const* ps[1];
                int ps_counts[1];
                ckc_value_t* x_coords[2];
                ckc_value_t* scalars[CKC_SQ_MAX_VEC];
                ckc_value_t* packed;
                int yy, j;
                y_base[0] = 0;
                y_base[1] = acc;
                y_base[2] = 0;
                for(yy = 0; yy < enc->num_Y; ++yy)
                {
                    ys[yy] = ckc_b_const_i32(b, y_base[yy]);
                }
                p0[0]        = tid;
                ps[0]        = p0;
                ps_counts[0] = 1;
                if(!ckc_tile_distribution_calculate_x(
                       b, qy_dist, ys, enc->num_Y, ps, ps_counts, 1, x_coords, 2))
                {
                    return NULL;
                }
                for(j = 0; j < spv; ++j)
                {
                    int y_full[3];
                    y_full[0]         = y_base[0];
                    y_full[1]         = y_base[1];
                    y_full[2]         = y_base[2];
                    y_full[vec_dim_y] = j;
                    scalars[j]        = qy_dt->storage[ckc_sq_y_to_linear(enc, y_full)];
                }
                packed = ckc_sq_pack_quant_local(b, scalars, spv, q_ty, qname);
                ckc_tile_window_store_vec(b, &qy_tile, x_coords, 2, packed, spv);
            }
        }
        else
        {
            /* VEC == 2: per-element scalar quant + store fallback. */
            for(k = 0; k < chunks; ++k)
            {
                /* n_off: const-mul before tid-mul (Python order). */
                ckc_value_t* km    = ckc_b_mul(b, ckc_b_const_i32(b, k * BS), c_vec);
                ckc_value_t* tm    = ckc_b_mul(b, tid, c_vec);
                ckc_value_t* n_off = ckc_b_add(b, km, tm);
                ckc_value_t* sm_idx[1];
                ckc_value_t* sm_scalars[CKC_SQ_MAX_VEC];
                sm_idx[0] = n_off;
                ckc_sq_view_load_vec_as_f32(b, &sm_view, sm_idx, 1, VEC, sm_scalars);
                for(i = 0; i < VEC; ++i)
                {
                    ckc_value_t* x_f32 = cached[k * VEC + i];
                    ckc_value_t* y_f32 = ckc_b_fmul(b, x_f32, sm_scalars[i]);
                    ckc_value_t* q =
                        ckc_sq_quantize_scalar_f32(b, y_f32, inv_yscale, spec->out_dtype);
                    ckc_value_t* col = ckc_b_add(b, n_off, ckc_b_const_i32(b, i));
                    ckc_value_t* local_idx[2];
                    local_idx[0] = ckc_b_const_i32(b, 0);
                    local_idx[1] = col;
                    /* qy_tile.store_scalar(b, 0, col, value=q) */
                    ckc_tile_window_store_scalar(b, &qy_tile, local_idx, 2, q, 0);
                }
            }
        }

        if(b->status != CKC_OK)
        {
            return NULL;
        }
        return b->kernel;
    });
}

ckc_kernel_def_t*
ckc_build_smoothquant_new(ckc_ir_builder_t* b, const ckc_smoothquant_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[CKC_SQ_NAME_CAP];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_smoothquant_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_smoothquant(b, spec, arch);
    });
}

/* --------------------------------------------------------------- grid */

ckc_status_t ckc_smoothquant_grid(int m, const ckc_smoothquant_spec_t* spec, int out[3])
{
    int totals[1];
    int tiles[1];
    (void)spec;
    if(out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    /* ceil_div_grid((m, 1)) -- one CTA per row. */
    totals[0] = m;
    tiles[0]  = 1;
    return ckc_ceil_div_grid(totals, tiles, 1, out);
}

/* --------------------------------------------------------------- lower */

ckc_status_t ckc_smoothquant_lower_to_llvm(const ckc_smoothquant_spec_t* spec,
                                           const char* arch,
                                           ckc_llvm_flavor_t flavor,
                                           char** out_ll,
                                           char* err,
                                           size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

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
        return CKC_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = CKC_SMOOTHQUANT_DEFAULT_ARCH;
    }

    kernel = ckc_build_smoothquant_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(&b);
            snprintf(err, err_cap, "%s", m ? m : "build_smoothquant failed");
        }
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
