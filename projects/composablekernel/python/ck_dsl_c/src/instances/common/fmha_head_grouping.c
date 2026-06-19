/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_fmha_head_grouping.c -- C99 port of
 * ck_dsl/instances/common/fmha_head_grouping.py.
 *
 * Byte-identical builder-call sequence vs the Python build():
 *   kb.block_size(64)
 *   _declare_params(kb)            (Q, K, V, O tensors; scale_log2/seqlen_q/k
 *                                   scalars; q/k/v/o strides)
 *   kb.decode_grid(has_batch_axis=True)
 *   to_sgpr_u32 of head/kv_head/batch idx
 *   k_token_offset / v_token_offset / q_tile_local / q_tile_base SGPR math
 *   mfma_attention_fwd_inner_body(...)
 *   b.ret()
 */
#include "ckc/instance_fmha_head_grouping.h"

#include <stdio.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err */
#include "ckc/helper_ck_dsl.helpers.spec.h"            /* ckc_kernel_name_join          */
#include "ckc/helper_ck_dsl.helpers.mfma_attention.h"  /* ckc_mfma_attention_fwd_inner_body */
#include "ckc/helper_ck_dsl.instances.common.fmha_arch.h" /* ckc_validate_fmha_mfma_atom */

/* Python: FmhaFwdHeadGroupingSpec.name default. */
static const char* const HG_DEFAULT_NAME = "ck_dsl_fmha_fwd_head_grouping";

/* Python module-level constants (MFMA_ATTN_BLOCK_M / _K = 16). Reuse the macros
 * exported by the mfma_attention helper port. */
#ifndef CKC_MFMA_ATTN_BLOCK_M
#define CKC_MFMA_ATTN_BLOCK_M 16
#endif
#ifndef CKC_MFMA_ATTN_BLOCK_K
#define CKC_MFMA_ATTN_BLOCK_K 16
#endif

/* Map the FMHA-common mask-mode enum to the attention-helper mask-mode enum.
 * Both enums share NONE=0/CAUSAL=1/SLIDING_WINDOW=2; alibi/custom have no
 * attention-helper analogue and fall through to NONE (the inner body never
 * special-cases them on this MFMA path -- they are gated upstream). */
static ckc_attn_mask_mode_t hg_attn_mask_mode(ckc_fmha_mask_mode_t m)
{
    switch (m)
    {
        case CKC_FMHA_MASK_CAUSAL:
            return CKC_ATTN_MASK_CAUSAL;
        case CKC_FMHA_MASK_SLIDING_WINDOW:
            return CKC_ATTN_MASK_SLIDING_WINDOW;
        case CKC_FMHA_MASK_NONE:
        case CKC_FMHA_MASK_ALIBI:
        case CKC_FMHA_MASK_CUSTOM:
        default:
            return CKC_ATTN_MASK_NONE;
    }
}

/* ====================================================================== *
 * FmhaFwdHeadGroupingSpec value helpers
 * ====================================================================== */

ckc_fmha_head_grouping_spec_t ckc_fmha_head_grouping_spec_make(
    ckc_fmha_common_spec_t common,
    int seqlen_q,
    int seqlen_k)
{
    ckc_fmha_head_grouping_spec_t spec;
    spec.common = common;
    spec.seqlen_q = seqlen_q;
    spec.seqlen_k = seqlen_k;
    spec.name = HG_DEFAULT_NAME;
    return spec;
}

/* FmhaFwdHeadGroupingSpec.kernel_name():
 *   kernel_name_join(name,
 *       f"H{head_size}", f"HQ{num_query_heads}", f"HK{num_kv_heads}",
 *       dtype, f"Q{seqlen_q}", f"K{seqlen_k}", mask_mode)
 */
ckc_status_t ckc_fmha_head_grouping_kernel_name(
    const ckc_fmha_head_grouping_spec_t* spec,
    char* out,
    size_t out_cap)
{
    char h[32], hq[32], hk[32], q[32], k[32];
    const char* parts[7];
    const ckc_fmha_shape_t* s;
    const char* mask_name;
    const char* prefix;

    if (spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    s = &spec->common.shape;

    /* f-strings -> snprintf. */
    snprintf(h, sizeof(h), "H%d", s->head_size);
    snprintf(hq, sizeof(hq), "HQ%d", s->num_query_heads);
    snprintf(hk, sizeof(hk), "HK%d", s->num_kv_heads);
    snprintf(q, sizeof(q), "Q%d", spec->seqlen_q);
    snprintf(k, sizeof(k), "K%d", spec->seqlen_k);

    mask_name = ckc_fmha_mask_mode_name(spec->common.mask_mode);
    if (mask_name == NULL)
    {
        mask_name = "none";
    }

    parts[0] = h;
    parts[1] = hq;
    parts[2] = hk;
    parts[3] = spec->common.dtype;
    parts[4] = q;
    parts[5] = k;
    parts[6] = mask_name;

    prefix = (spec->name != NULL) ? spec->name : HG_DEFAULT_NAME;

    return ckc_kernel_name_join(prefix, parts, 7,
                                NULL, NULL, 0,
                                out, out_cap, NULL);
}

bool ckc_fmha_head_grouping_is_mqa(const ckc_fmha_head_grouping_spec_t* spec)
{
    if (spec == NULL)
    {
        return false;
    }
    return spec->common.shape.num_kv_heads == 1;
}

bool ckc_fmha_head_grouping_is_gqa(const ckc_fmha_head_grouping_spec_t* spec)
{
    const ckc_fmha_shape_t* s;
    if (spec == NULL)
    {
        return false;
    }
    s = &spec->common.shape;
    return s->num_query_heads > s->num_kv_heads && s->num_kv_heads > 1;
}

const char* ckc_fmha_head_grouping_grouping_label(
    const ckc_fmha_head_grouping_spec_t* spec)
{
    if (ckc_fmha_head_grouping_is_mqa(spec))
    {
        return "mqa";
    }
    if (ckc_fmha_head_grouping_is_gqa(spec))
    {
        return "gqa";
    }
    return "mha";
}

/* ====================================================================== *
 * is_valid_spec
 * ====================================================================== */

/* Write `msg` into the caller reason buffer (NUL-terminated, truncated). */
static void hg_reason(char* reason, size_t reason_cap, const char* msg)
{
    size_t n;
    if (reason == NULL || reason_cap == 0 || msg == NULL)
    {
        return;
    }
    n = strlen(msg);
    if (n >= reason_cap)
    {
        n = reason_cap - 1;
    }
    memcpy(reason, msg, n);
    reason[n] = '\0';
}

bool ckc_fmha_head_grouping_is_valid_spec(
    ckc_arena_t* arena,
    const ckc_fmha_head_grouping_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap)
{
    const ckc_fmha_shape_t* s;
    const char* common_reason = NULL;
    char buf[256];

    if (spec == NULL)
    {
        hg_reason(reason, reason_cap, "null spec");
        return false;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = validate_common_spec(spec.common) */
    if (!ckc_fmha_validate_common_spec(arena, &spec->common, &common_reason))
    {
        hg_reason(reason, reason_cap,
                  (common_reason != NULL) ? common_reason : "invalid common spec");
        return false;
    }

    /* ok, why = validate_fmha_mfma_atom(spec.common.dtype, arch) */
    if (!ckc_validate_fmha_mfma_atom(spec->common.dtype, arch, buf, sizeof(buf)))
    {
        hg_reason(reason, reason_cap, buf);
        return false;
    }

    /* if spec.seqlen_q <= 0 or spec.seqlen_k <= 0 */
    if (spec->seqlen_q <= 0 || spec->seqlen_k <= 0)
    {
        snprintf(buf, sizeof(buf),
                 "seqlen_q / seqlen_k must be > 0 (got %d, %d)",
                 spec->seqlen_q, spec->seqlen_k);
        hg_reason(reason, reason_cap, buf);
        return false;
    }

    /* if spec.seqlen_q % MFMA_ATTN_BLOCK_M != 0 */
    if (spec->seqlen_q % CKC_MFMA_ATTN_BLOCK_M != 0)
    {
        snprintf(buf, sizeof(buf),
                 "MFMA head_grouping needs seqlen_q (%d) to be a "
                 "multiple of BLOCK_M (%d)",
                 spec->seqlen_q, CKC_MFMA_ATTN_BLOCK_M);
        hg_reason(reason, reason_cap, buf);
        return false;
    }

    /* if spec.seqlen_k % MFMA_ATTN_BLOCK_K != 0 */
    if (spec->seqlen_k % CKC_MFMA_ATTN_BLOCK_K != 0)
    {
        snprintf(buf, sizeof(buf),
                 "MFMA head_grouping needs seqlen_k (%d) to be a "
                 "multiple of BLOCK_K (%d)",
                 spec->seqlen_k, CKC_MFMA_ATTN_BLOCK_K);
        hg_reason(reason, reason_cap, buf);
        return false;
    }

    /* if spec.common.shape.head_size % 16 != 0 */
    s = &spec->common.shape;
    if (s->head_size % 16 != 0)
    {
        snprintf(buf, sizeof(buf),
                 "MFMA head_grouping needs head_size %% 16 == 0 (got %d)",
                 s->head_size);
        hg_reason(reason, reason_cap, buf);
        return false;
    }

    /* if s.num_query_heads <= s.num_kv_heads */
    if (s->num_query_heads <= s->num_kv_heads)
    {
        snprintf(buf, sizeof(buf),
                 "head_grouping spec requires HQ > HK; got HQ=%d, HK=%d",
                 s->num_query_heads, s->num_kv_heads);
        hg_reason(reason, reason_cap, buf);
        return false;
    }

    hg_reason(reason, reason_cap, "ok");
    return true;
}

/* ====================================================================== *
 * _declare_params -- the head_grouping kernel ABI.
 * ====================================================================== */
static void hg_declare_params(ckc_fmha_kernel_builder_t* kb)
{
    static const char* const stride_names[4] = {"q", "k", "v", "o"};

    /* kb.add_tensor("Q", readonly=True) etc. (dtype NULL => common.dtype). */
    ckc_fmha_kernel_builder_add_tensor(kb, "Q", NULL, /*readonly*/ true,
                                       /*writeonly*/ false, /*align*/ 16);
    ckc_fmha_kernel_builder_add_tensor(kb, "K", NULL, true, false, 16);
    ckc_fmha_kernel_builder_add_tensor(kb, "V", NULL, true, false, 16);
    ckc_fmha_kernel_builder_add_tensor(kb, "O", NULL, /*readonly*/ false,
                                       /*writeonly*/ true, 16);

    /* kb.add_scalar("scale_log2", "f32"); seqlen_q/k as i32. */
    ckc_fmha_kernel_builder_add_scalar(kb, "scale_log2", "f32");
    ckc_fmha_kernel_builder_add_scalar(kb, "seqlen_q", "i32");
    ckc_fmha_kernel_builder_add_scalar(kb, "seqlen_k", "i32");

    /* kb.add_strides("q", "k", "v", "o"). */
    ckc_fmha_kernel_builder_add_strides(kb, stride_names, 4);
}

/* ====================================================================== *
 * build_fmha_fwd_head_grouping -- THE DRIVER.
 * ====================================================================== */
ckc_kernel_def_t* ckc_build_fmha_fwd_head_grouping(
    ckc_fmha_kernel_builder_t* kb,
    const ckc_fmha_head_grouping_spec_t* spec,
    const char* arch)
{
    char name[256];
    ckc_arena_t reason_arena;
    bool ok;
    const ckc_fmha_common_spec_t* s;
    ckc_ir_builder_t* b;

    ckc_value_t* seqlen_q;
    ckc_value_t* seqlen_k;
    ckc_value_t* q_tile_idx;
    ckc_value_t* head_idx;
    ckc_value_t* kv_head_idx;
    ckc_value_t* batch_idx;
    ckc_value_t* k_token_offset;
    ckc_value_t* v_token_offset;
    ckc_value_t* q_tile_local;
    ckc_value_t* q_tile_base;
    ckc_value_t* causal_ctx;
    ckc_mfma_attn_params_t p;

    if (kb == NULL || spec == NULL)
    {
        return NULL;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
    if (ckc_arena_init(&reason_arena, 0) != 0)
    {
        return NULL;
    }
    ok = ckc_fmha_head_grouping_is_valid_spec(&reason_arena, spec, arch, NULL, 0);
    ckc_arena_destroy(&reason_arena);
    if (!ok)
    {
        /* Mirror the Python `raise ValueError(...)`: surface as a sticky error
         * once the builder exists. Init the builder so the caller can read the
         * error / free it. */
        if (ckc_fmha_head_grouping_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if (ckc_fmha_kernel_builder_init(kb, name, &spec->common) != CKC_OK)
        {
            return NULL;
        }
        b = ckc_fmha_kernel_builder_builder(kb);
        ckc_i_set_err(b, CKC_ERR_VALUE, "invalid head_grouping spec");
        return NULL;
    }

    s = &spec->common;

    /* kb = FmhaKernelBuilder(spec.kernel_name(), s) */
    if (ckc_fmha_head_grouping_kernel_name(spec, name, sizeof(name)) != CKC_OK)
    {
        return NULL;
    }
    if (ckc_fmha_kernel_builder_init(kb, name, s) != CKC_OK)
    {
        return NULL;
    }

    /* kb.block_size(64) -- one wave64 warp per CTA. */
    ckc_fmha_kernel_builder_block_size(kb, 64);

    /* _declare_params(kb) */
    hg_declare_params(kb);

    /* kb.decode_grid(has_batch_axis=True) */
    ckc_fmha_kernel_builder_decode_grid(kb, /*num_queries_per_kv*/ -1,
                                        /*has_batch_axis*/ true,
                                        NULL, NULL, NULL);

    /* b = kb.builder */
    b = ckc_fmha_kernel_builder_builder(kb);

    seqlen_q = ckc_fmha_kernel_builder_scalar(kb, "seqlen_q");
    seqlen_k = ckc_fmha_kernel_builder_scalar(kb, "seqlen_k");

    /* q_tile_idx = kb.q_token (same axis as block_id_x). */
    q_tile_idx = kb->q_token;

    /* head_idx = b.to_sgpr_u32(kb.head_idx) etc. */
    head_idx = ckc_b_to_sgpr_u32(b, kb->head_idx);
    kv_head_idx = ckc_b_to_sgpr_u32(b, kb->kv_head_idx);
    batch_idx = ckc_b_to_sgpr_u32(b, kb->batch_idx);

    /* k_token_offset = to_sgpr_u32(batch_idx * seqlen_k * stride_token("k")) */
    k_token_offset = ckc_b_to_sgpr_u32(
        b,
        ckc_b_mul(b,
                  ckc_b_mul(b, batch_idx, seqlen_k),
                  ckc_fmha_kernel_builder_stride_token(kb, "k")));

    /* v_token_offset = to_sgpr_u32(batch_idx * seqlen_k * stride_token("v")) */
    v_token_offset = ckc_b_to_sgpr_u32(
        b,
        ckc_b_mul(b,
                  ckc_b_mul(b, batch_idx, seqlen_k),
                  ckc_fmha_kernel_builder_stride_token(kb, "v")));

    /* q_tile_local = to_sgpr_u32(q_tile_idx * const_i32(BLOCK_M)) */
    q_tile_local = ckc_b_to_sgpr_u32(
        b, ckc_b_mul(b, q_tile_idx, ckc_b_const_i32(b, CKC_MFMA_ATTN_BLOCK_M)));

    /* q_tile_base = to_sgpr_u32(q_tile_local + batch_idx * seqlen_q) */
    q_tile_base = ckc_b_to_sgpr_u32(
        b, ckc_b_add(b, q_tile_local, ckc_b_mul(b, batch_idx, seqlen_q)));

    /* causal_ctx = const_i32(0) if mask in {causal, sliding_window} else None */
    causal_ctx = NULL;
    if (s->mask_mode == CKC_FMHA_MASK_CAUSAL ||
        s->mask_mode == CKC_FMHA_MASK_SLIDING_WINDOW)
    {
        causal_ctx = ckc_b_const_i32(b, 0);
    }

    /* mfma_attention_fwd_inner_body(b, ...) */
    memset(&p, 0, sizeof(p));
    p.Q = ckc_fmha_kernel_builder_tensor(kb, "Q");
    p.K = ckc_fmha_kernel_builder_tensor(kb, "K");
    p.V = ckc_fmha_kernel_builder_tensor(kb, "V");
    p.O = ckc_fmha_kernel_builder_tensor(kb, "O");
    p.head_size = s->shape.head_size;
    p.seqlen_k = seqlen_k;
    p.q_tile_base = q_tile_base;
    /* q_pos_base = q_tile_local (within-batch Q position for the mask). */
    p.q_pos_base = q_tile_local;
    p.head_idx = head_idx;
    p.kv_head_idx = kv_head_idx;
    p.stride_q_token = ckc_fmha_kernel_builder_stride_token(kb, "q");
    p.stride_q_head = ckc_fmha_kernel_builder_stride_head(kb, "q");
    p.stride_k_token = ckc_fmha_kernel_builder_stride_token(kb, "k");
    p.stride_k_head = ckc_fmha_kernel_builder_stride_head(kb, "k");
    p.stride_v_token = ckc_fmha_kernel_builder_stride_token(kb, "v");
    p.stride_v_head = ckc_fmha_kernel_builder_stride_head(kb, "v");
    p.stride_o_token = ckc_fmha_kernel_builder_stride_token(kb, "o");
    p.stride_o_head = ckc_fmha_kernel_builder_stride_head(kb, "o");
    p.scale_log2 = ckc_fmha_kernel_builder_scalar(kb, "scale_log2");
    p.dtype = s->dtype;
    p.mask_mode = hg_attn_mask_mode(s->mask_mode);
    p.sliding_window = s->sliding_window;
    p.causal_ctx_offset = causal_ctx;
    p.k_token_offset_elems = k_token_offset;
    p.v_token_offset_elems = v_token_offset;
    p.arch = arch;

    ckc_mfma_attention_fwd_inner_body(b, &p);

    /* b.ret() */
    ckc_b_ret(b);

    /* return kb.kernel */
    return ckc_fmha_kernel_builder_kernel(kb);
}

/* ====================================================================== *
 * fmha_fwd_head_grouping_grid
 * ====================================================================== */
void ckc_fmha_head_grouping_grid(const ckc_fmha_head_grouping_spec_t* spec,
                                 int batch,
                                 int* gx,
                                 int* gy,
                                 int* gz)
{
    if (spec == NULL)
    {
        return;
    }
    if (gx != NULL)
    {
        *gx = spec->seqlen_q / CKC_MFMA_ATTN_BLOCK_M;
    }
    if (gy != NULL)
    {
        *gy = spec->common.shape.num_query_heads;
    }
    if (gz != NULL)
    {
        *gz = batch;
    }
}

/* ====================================================================== *
 * lower-to-.ll convenience.
 * ====================================================================== */
ckc_status_t ckc_fmha_head_grouping_lower_to_llvm(
    const ckc_fmha_head_grouping_spec_t* spec,
    const char* arch,
    ckc_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap)
{
    ckc_fmha_kernel_builder_t kb;
    ckc_kernel_def_t* kernel;
    ckc_ir_builder_t* b;
    ckc_status_t st;

    if (out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if (spec == NULL || out_ll == NULL)
    {
        if (err != NULL && err_cap > 0)
        {
            hg_reason(err, err_cap, "lower_to_llvm: null spec/out");
        }
        return CKC_ERR_VALUE;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    /* The driver inits kb's embedded IR builder via the kernel name. */
    kernel = ckc_build_fmha_fwd_head_grouping(&kb, spec, arch);
    if (kernel == NULL)
    {
        b = ckc_fmha_kernel_builder_builder(&kb);
        st = ckc_ir_builder_status(b);
        if (err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(b);
            hg_reason(err, err_cap,
                      (m != NULL) ? m : "build_fmha_fwd_head_grouping failed");
        }
        ckc_fmha_kernel_builder_free(&kb);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_fmha_kernel_builder_free(&kb);
    return st;
}
