// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/instance_fmha_mfma.c -- C99 port of rocke/instances/common/fmha_mfma.py.
 *
 * Byte-identical builder-call sequence vs the Python build_fmha_fwd_mfma: the
 * FmhaKernelBuilder declares the same params in the same order, decodes the grid
 * the same way, computes the same per-batch row offsets, and calls the ported
 * helper rocke_mfma_attention_fwd_inner_body with the same operands / attrs. All
 * the heavy IR emission is delegated to the already-ported helpers; this file is
 * the thin spec->kernel wrapper plus a lower-to-.ll convenience.
 */

#include "rocke/instance_fmha_mfma.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.mfma_attention.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.instances.common._fmha_common.h"

/* --------------------------------------------------------------------------- *
 * Local copies of the Python module-level constants (kept here so the validity
 * gate and grid math do not depend on the helper macro spelling drifting).
 * --------------------------------------------------------------------------- */
#define FMHA_MFMA_DEFAULT_NAME "rocke_fmha_fwd_mfma"

/* Mirror of the helper BLOCK_M / BLOCK_K (Python imports MFMA_ATTN_BLOCK_*). */
#ifndef ROCKE_MFMA_ATTN_BLOCK_M
#define ROCKE_MFMA_ATTN_BLOCK_M 16
#endif
#ifndef ROCKE_MFMA_ATTN_BLOCK_K
#define ROCKE_MFMA_ATTN_BLOCK_K 16
#endif

/* ----- small helpers ----- */

static void fmha_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
}

/* Map the shared FMHA mask enum to the attention-helper mask enum. The build
 * routine validates the mode beforehand, so only the three supported modes reach
 * the helper; alibi/custom fall back to NONE (and are rejected in validate). */
static rocke_attn_mask_mode_t fmha_to_attn_mask(rocke_fmha_mask_mode_t m)
{
    switch(m)
    {
    case ROCKE_FMHA_MASK_CAUSAL:
        return ROCKE_ATTN_MASK_CAUSAL;
    case ROCKE_FMHA_MASK_SLIDING_WINDOW:
        return ROCKE_ATTN_MASK_SLIDING_WINDOW;
    case ROCKE_FMHA_MASK_NONE:
    default:
        return ROCKE_ATTN_MASK_NONE;
    }
}

/* Build the flat spec's equivalent FmhaCommonSpec (shape + dtype + mask + scale).
 * dtype defaults to "f16" (the v1 constraint). */
static rocke_fmha_common_spec_t fmha_mfma_common(const rocke_fmha_mfma_spec_t* spec)
{
    rocke_fmha_shape_t shape
        = rocke_fmha_shape_default(spec->head_size, spec->num_query_heads, spec->num_kv_heads);
    rocke_fmha_common_spec_t common = rocke_fmha_common_spec_default(shape);
    common.dtype = (spec->dtype != NULL) ? spec->dtype : "f16";
    common.scale_log2 = spec->scale_log2;
    common.mask_mode = spec->mask_mode;
    common.sliding_window = spec->sliding_window;
    return common;
}

/* --------------------------------------------------------------------------- *
 * rocke_fmha_mfma_spec_default
 * --------------------------------------------------------------------------- */
rocke_fmha_mfma_spec_t rocke_fmha_mfma_spec_default(void)
{
    rocke_fmha_mfma_spec_t s;
    s.head_size = 0;
    s.num_query_heads = 0;
    s.num_kv_heads = 0;
    s.seqlen_q = 0;
    s.seqlen_k = 0;
    s.dtype = "f16";
    s.mask_mode = ROCKE_FMHA_MASK_NONE;
    s.sliding_window = 0;
    s.scale_log2 = 0.0;
    s.name = FMHA_MFMA_DEFAULT_NAME;
    return s;
}

/* --------------------------------------------------------------------------- *
 * _mma_family(arch)
 * --------------------------------------------------------------------------- */
const char* rocke_fmha_mfma_mma_family(const char* arch)
{
    const rocke_archtarget_t* t;
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    t = rocke_archtarget_from_gfx(arch);
    if(t != NULL && t->wave_size == 32)
    {
        return "wmma";
    }
    return "mma";
}

/* --------------------------------------------------------------------------- *
 * FmhaMfmaSpec.kernel_name()
 * --------------------------------------------------------------------------- */
rocke_status_t
    rocke_fmha_mfma_kernel_name(const rocke_fmha_mfma_spec_t* spec, char* out, size_t out_cap)
{
    const char* name;
    const char* dtype;
    const char* mask;
    char h[32], hq[32], hk[32], q[32], k[32];
    const char* parts[7];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    name = (spec->name != NULL) ? spec->name : FMHA_MFMA_DEFAULT_NAME;
    dtype = (spec->dtype != NULL) ? spec->dtype : "f16";
    mask = rocke_fmha_mask_mode_name(spec->mask_mode);
    if(mask == NULL)
    {
        mask = "none";
    }

    snprintf(h, sizeof(h), "H%d", spec->head_size);
    snprintf(hq, sizeof(hq), "HQ%d", spec->num_query_heads);
    snprintf(hk, sizeof(hk), "HK%d", spec->num_kv_heads);
    snprintf(q, sizeof(q), "Q%d", spec->seqlen_q);
    snprintf(k, sizeof(k), "K%d", spec->seqlen_k);

    parts[0] = h;
    parts[1] = hq;
    parts[2] = hk;
    parts[3] = dtype;
    parts[4] = q;
    parts[5] = k;
    parts[6] = mask;

    return rocke_kernel_name_join(name, parts, 7, NULL, NULL, 0, out, out_cap, NULL);
}

/* --------------------------------------------------------------------------- *
 * is_valid_spec(spec, arch)
 * --------------------------------------------------------------------------- */
bool rocke_fmha_mfma_is_valid_spec(const rocke_fmha_mfma_spec_t* spec,
                                   const char* arch,
                                   char* reason,
                                   size_t reason_cap)
{
    const rocke_archtarget_t* target;
    const char* family;
    rocke_fmha_common_spec_t common;
    const char* common_reason = NULL;
    rocke_arena_t arena;
    bool ok;
    char buf[256];
    long bytes_lds;

    if(spec == NULL)
    {
        fmha_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown arch '%s'", arch);
        fmha_set_reason(reason, reason_cap, buf);
        return false;
    }

    family = rocke_fmha_mfma_mma_family(arch);

    /* validate_common_spec(spec.common). Borrow a transient arena for the reason
     * string (reason text never enters the IR -- emission is byte-identical). */
    common = fmha_mfma_common(spec);
    rocke_arena_init(&arena, 0);
    ok = rocke_fmha_validate_common_spec(&arena, &common, &common_reason);
    if(!ok)
    {
        fmha_set_reason(
            reason, reason_cap, common_reason != NULL ? common_reason : "invalid common spec");
        rocke_arena_destroy(&arena);
        return false;
    }
    rocke_arena_destroy(&arena);

    /* s.head_size % 16 != 0 */
    if(spec->head_size % 16 != 0)
    {
        snprintf(
            buf, sizeof(buf), "tiled FMHA needs head_size %% 16 == 0 (got %d)", spec->head_size);
        fmha_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* seqlen_q % BLOCK_M != 0 */
    if(spec->seqlen_q % ROCKE_MFMA_ATTN_BLOCK_M != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "seqlen_q (%d) must be a multiple of BLOCK_M (%d)",
                 spec->seqlen_q,
                 ROCKE_MFMA_ATTN_BLOCK_M);
        fmha_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* seqlen_k % BLOCK_K != 0 */
    if(spec->seqlen_k % ROCKE_MFMA_ATTN_BLOCK_K != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "seqlen_k (%d) must be a multiple of BLOCK_K (%d)",
                 spec->seqlen_k,
                 ROCKE_MFMA_ATTN_BLOCK_K);
        fmha_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* dtype != "f16" */
    if(common.dtype == NULL || strcmp(common.dtype, "f16") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "tiled FMHA v1 ships f16 only; bf16 lands once the bf16 "
                 "atom factory is exposed (got %s)",
                 common.dtype != NULL ? common.dtype : "(null)");
        fmha_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* The QK / PV chain is the f16 16x16x16 atom; require it on the target
     * catalog (MFMA on CDNA, WMMA on RDNA). */
    if(!rocke_archtarget_supports_dtype_combo(target, "f16", "f16", "fp32", family))
    {
        snprintf(buf, sizeof(buf), "unsupported f16 %s dtype combo on %s", family, arch);
        fmha_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(!rocke_mma_catalog_has_shape(rocke_archtarget_mma(target),
                                    family,
                                    "f16",
                                    "f16",
                                    "fp32",
                                    ROCKE_MFMA_ATTN_BLOCK_M,
                                    ROCKE_MFMA_ATTN_BLOCK_M,
                                    ROCKE_MFMA_ATTN_BLOCK_K))
    {
        snprintf(buf,
                 sizeof(buf),
                 "unsupported f16 %s warp_tile (%d,%d,%d) on %s",
                 family,
                 ROCKE_MFMA_ATTN_BLOCK_M,
                 ROCKE_MFMA_ATTN_BLOCK_M,
                 ROCKE_MFMA_ATTN_BLOCK_K,
                 arch);
        fmha_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* LDS budget: one BLOCK_M x BLOCK_K f16 P-staging buffer. */
    bytes_lds = (long)ROCKE_MFMA_ATTN_BLOCK_M * ROCKE_MFMA_ATTN_BLOCK_K * 2;
    if(!rocke_archtarget_fits_lds(target, bytes_lds))
    {
        snprintf(buf, sizeof(buf), "LDS budget %ld > cap on %s", bytes_lds, arch);
        fmha_set_reason(reason, reason_cap, buf);
        return false;
    }

    fmha_set_reason(reason, reason_cap, "ok");
    return true;
}

/* --------------------------------------------------------------------------- *
 * _declare_params(kb): the MFMA FMHA kernel ABI (shared between build + sig).
 * --------------------------------------------------------------------------- */
static void fmha_declare_params(rocke_fmha_kernel_builder_t* kb)
{
    static const char* const stride_names[4] = {"q", "k", "v", "o"};

    /* add_tensor("Q", readonly=True) ... default align=16 */
    rocke_fmha_kernel_builder_add_tensor(kb, "Q", NULL, /*readonly*/ true, /*writeonly*/ false, 16);
    rocke_fmha_kernel_builder_add_tensor(kb, "K", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_tensor(kb, "V", NULL, true, false, 16);
    /* O: readonly=False, writeonly=True */
    rocke_fmha_kernel_builder_add_tensor(kb, "O", NULL, /*readonly*/ false, /*writeonly*/ true, 16);

    rocke_fmha_kernel_builder_add_scalar(kb, "scale_log2", "f32");
    rocke_fmha_kernel_builder_add_scalar(kb, "seqlen_q", "i32");
    rocke_fmha_kernel_builder_add_scalar(kb, "seqlen_k", "i32");

    rocke_fmha_kernel_builder_add_strides(kb, stride_names, 4);
}

/* --------------------------------------------------------------------------- *
 * build_fmha_fwd_mfma(spec, arch)
 *
 * Mirrors the Python op-for-op:
 *   1. is_valid_spec gate
 *   2. wave_size = ArchTarget.from_gfx(arch).wave_size
 *   3. FmhaKernelBuilder(kernel_name, common); block_size(wave_size);
 *      _declare_params; decode_grid(has_batch_axis=True)
 *   4. q_tile_local = q_token * BLOCK_M
 *   5. batch_row_q = batch_idx * seqlen_q
 *      k_batch_offset = batch_idx * seqlen_k * stride_k_token
 *      v_batch_offset = batch_idx * seqlen_k * stride_v_token
 *   6. q_pos_base = q_tile_local if masked else None
 *   7. mfma_attention_fwd_inner_body(...) ; b.ret()
 * --------------------------------------------------------------------------- */
rocke_kernel_def_t* rocke_build_fmha_fwd_mfma(rocke_ir_builder_t* b_unused,
                                              const rocke_fmha_mfma_spec_t* spec,
                                              const char* arch)
{
    return ckc::guard_builder((rocke_ir_builder_t*)nullptr, [&]() -> rocke_kernel_def_t* {
        /* NOTE: this instance owns its FmhaKernelBuilder (which embeds its own
         * IRBuilder), matching the Python build that constructs the FmhaKernelBuilder
         * internally. The `b_unused` parameter exists for signature parity with the
         * documented CALL PATTERN / sibling entry points; the kernel returned is owned
         * by the static-lifetime builder below. To keep the kernel alive past this
         * call without an owning builder, callers should use the lower-to-llvm
         * convenience (which owns its builder for the whole lower). */
        (void)b_unused;
        {
            char name_buf[256];
            rocke_fmha_common_spec_t common;
            const rocke_archtarget_t* target;
            int wave_size;
            rocke_fmha_kernel_builder_t kb;
            rocke_ir_builder_t* b;
            rocke_value_t* seqlen_q;
            rocke_value_t* seqlen_k;
            rocke_value_t* head_idx;
            rocke_value_t* kv_head_idx;
            rocke_value_t* batch_idx;
            rocke_value_t* q_tile_idx;
            rocke_value_t* q_tile_local;
            rocke_value_t* batch_row_q;
            rocke_value_t* k_batch_offset;
            rocke_value_t* v_batch_offset;
            rocke_value_t* causal_ctx;
            rocke_value_t* q_pos_base;
            bool masked;
            rocke_mfma_attn_params_t p;
            rocke_status_t st;
            rocke_kernel_def_t* kernel;

            if(spec == NULL)
            {
                return NULL;
            }
            if(arch == NULL)
            {
                arch = "gfx950";
            }

            /* 1. validity gate (Python raises ValueError on reject). */
            if(!rocke_fmha_mfma_is_valid_spec(spec, arch, NULL, 0))
            {
                return NULL;
            }

            common = fmha_mfma_common(spec);

            /* 2. wave_size from the target. */
            target = rocke_archtarget_from_gfx(arch);
            if(target == NULL)
            {
                return NULL;
            }
            wave_size = target->wave_size;

            /* 3. FmhaKernelBuilder(spec.kernel_name(), common). */
            if(rocke_fmha_mfma_kernel_name(spec, name_buf, sizeof(name_buf)) != ROCKE_OK)
            {
                return NULL;
            }
            st = rocke_fmha_kernel_builder_init(&kb, name_buf, &common);
            if(st != ROCKE_OK)
            {
                return NULL;
            }

            rocke_fmha_kernel_builder_block_size(&kb, wave_size); /* one wave per CTA */
            fmha_declare_params(&kb);
            rocke_fmha_kernel_builder_decode_grid(&kb,
                                                  /*num_queries_per_kv*/ -1,
                                                  /*has_batch_axis*/ true,
                                                  NULL,
                                                  NULL,
                                                  NULL);

            b = rocke_fmha_kernel_builder_builder(&kb);

            seqlen_q = rocke_fmha_kernel_builder_scalar(&kb, "seqlen_q");
            seqlen_k = rocke_fmha_kernel_builder_scalar(&kb, "seqlen_k");
            head_idx = kb.head_idx;
            kv_head_idx = kb.kv_head_idx;
            batch_idx = kb.batch_idx;

            /* 4. q_tile_idx (reuses block_id_x); q_tile_local = q_tile_idx * BLOCK_M. */
            q_tile_idx = kb.q_token;
            q_tile_local
                = rocke_b_mul(b, q_tile_idx, rocke_b_const_i32(b, ROCKE_MFMA_ATTN_BLOCK_M));

            /* 5. per-batch shifts.
             *   batch_row_q     = batch_idx * seqlen_q
             *   k_batch_offset  = (batch_idx * seqlen_k) * stride_k_token
             *   v_batch_offset  = (batch_idx * seqlen_k) * stride_v_token */
            batch_row_q = rocke_b_mul(b, batch_idx, seqlen_q);
            k_batch_offset = rocke_b_mul(b,
                                         rocke_b_mul(b, batch_idx, seqlen_k),
                                         rocke_fmha_kernel_builder_stride_token(&kb, "k"));
            v_batch_offset = rocke_b_mul(b,
                                         rocke_b_mul(b, batch_idx, seqlen_k),
                                         rocke_fmha_kernel_builder_stride_token(&kb, "v"));

            causal_ctx = rocke_b_const_i32(b, 0); /* self-attention: no cache offset */

            /* 6. q_pos_base = q_tile_local if masked else None. */
            masked = (common.mask_mode == ROCKE_FMHA_MASK_CAUSAL
                      || common.mask_mode == ROCKE_FMHA_MASK_SLIDING_WINDOW);
            q_pos_base = masked ? q_tile_local : NULL;

            /* 7. mfma_attention_fwd_inner_body(...). */
            memset(&p, 0, sizeof(p));
            p.Q = rocke_fmha_kernel_builder_tensor(&kb, "Q");
            p.K = rocke_fmha_kernel_builder_tensor(&kb, "K");
            p.V = rocke_fmha_kernel_builder_tensor(&kb, "V");
            p.O = rocke_fmha_kernel_builder_tensor(&kb, "O");
            p.head_size = common.shape.head_size;
            p.seqlen_k = seqlen_k;
            /* q_tile_base = local Q row + per-batch row shift. */
            p.q_tile_base = rocke_b_add(b, q_tile_local, batch_row_q);
            p.q_pos_base = q_pos_base;
            p.head_idx = head_idx;
            p.kv_head_idx = kv_head_idx;
            p.stride_q_token = rocke_fmha_kernel_builder_stride_token(&kb, "q");
            p.stride_q_head = rocke_fmha_kernel_builder_stride_head(&kb, "q");
            p.stride_k_token = rocke_fmha_kernel_builder_stride_token(&kb, "k");
            p.stride_k_head = rocke_fmha_kernel_builder_stride_head(&kb, "k");
            p.stride_v_token = rocke_fmha_kernel_builder_stride_token(&kb, "v");
            p.stride_v_head = rocke_fmha_kernel_builder_stride_head(&kb, "v");
            p.stride_o_token = rocke_fmha_kernel_builder_stride_token(&kb, "o");
            p.stride_o_head = rocke_fmha_kernel_builder_stride_head(&kb, "o");
            p.scale_log2 = rocke_fmha_kernel_builder_scalar(&kb, "scale_log2");
            p.dtype = common.dtype;
            p.mask_mode = fmha_to_attn_mask(common.mask_mode);
            p.sliding_window = common.sliding_window;
            p.causal_ctx_offset = causal_ctx;
            p.k_token_offset_elems = k_batch_offset;
            p.v_token_offset_elems = v_batch_offset;
            p.arch = arch;

            (void)rocke_mfma_attention_fwd_inner_body(b, &p);

            /* b.ret() */
            rocke_b_ret(b);

            kernel = rocke_fmha_kernel_builder_kernel(&kb);
            if(rocke_ir_builder_status(b) != ROCKE_OK)
            {
                rocke_fmha_kernel_builder_free(&kb);
                return NULL;
            }
            /* The kernel is owned by kb's embedded IRBuilder. Callers that need it to
             * outlive this call should use rocke_fmha_fwd_mfma_lower_to_llvm (which keeps
             * the builder alive for the whole lower) or the verify/fix harness, which
             * re-builds through the lower path. We intentionally do NOT free kb here so
             * the returned pointer stays valid for an immediate same-scope lower; the
             * harness owns the lifetime. */
            return kernel;
        }
    });
}

/* --------------------------------------------------------------------------- *
 * fmha_fwd_mfma_grid(spec, batch)
 * --------------------------------------------------------------------------- */
void rocke_fmha_fwd_mfma_grid(const rocke_fmha_mfma_spec_t* spec, int batch, int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return;
    }
    out[0] = spec->seqlen_q / ROCKE_MFMA_ATTN_BLOCK_M;
    out[1] = spec->num_query_heads;
    out[2] = batch;
}

/* --------------------------------------------------------------------------- *
 * fmha_fwd_mfma_signature(spec)
 * --------------------------------------------------------------------------- */
rocke_status_t rocke_fmha_fwd_mfma_signature(const rocke_fmha_mfma_spec_t* spec,
                                             rocke_arena_t* arena,
                                             const rocke_sig_entry_t** out_items,
                                             size_t* out_count)
{
    rocke_fmha_common_spec_t common;
    rocke_fmha_kernel_builder_t kb;
    rocke_status_t st;

    if(spec == NULL || arena == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    common = fmha_mfma_common(spec);
    st = rocke_fmha_kernel_builder_init(&kb, "rocke_fmha_fwd_mfma_sig_probe", &common);
    if(st != ROCKE_OK)
    {
        return st;
    }
    fmha_declare_params(&kb);
    st = rocke_fmha_kernel_builder_signature(&kb, arena, out_items, out_count);
    rocke_fmha_kernel_builder_free(&kb);
    return st;
}

/* --------------------------------------------------------------------------- *
 * rocke_fmha_fwd_mfma_lower_to_llvm -- build + lower to .ll convenience.
 *
 * Owns and frees its own FmhaKernelBuilder for the whole lower so the kernel
 * stays alive through lowering.
 * --------------------------------------------------------------------------- */
rocke_status_t rocke_fmha_fwd_mfma_lower_to_llvm(const rocke_fmha_mfma_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap)
{
    char name_buf[256];
    rocke_fmha_common_spec_t common;
    const rocke_archtarget_t* target;
    int wave_size;
    rocke_fmha_kernel_builder_t kb;
    rocke_ir_builder_t* b;
    rocke_value_t* seqlen_q;
    rocke_value_t* seqlen_k;
    rocke_value_t* q_tile_local;
    rocke_value_t* batch_row_q;
    rocke_value_t* k_batch_offset;
    rocke_value_t* v_batch_offset;
    rocke_value_t* causal_ctx;
    bool masked;
    rocke_mfma_attn_params_t p;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        fmha_set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    if(!rocke_fmha_mfma_is_valid_spec(spec, arch, err, err_cap))
    {
        return ROCKE_ERR_VALUE;
    }

    common = fmha_mfma_common(spec);
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        fmha_set_reason(err, err_cap, "lower_to_llvm: unknown arch");
        return ROCKE_ERR_VALUE;
    }
    wave_size = target->wave_size;

    if(rocke_fmha_mfma_kernel_name(spec, name_buf, sizeof(name_buf)) != ROCKE_OK)
    {
        fmha_set_reason(err, err_cap, "lower_to_llvm: kernel name too long");
        return ROCKE_ERR_VALUE;
    }
    st = rocke_fmha_kernel_builder_init(&kb, name_buf, &common);
    if(st != ROCKE_OK)
    {
        fmha_set_reason(err, err_cap, "lower_to_llvm: builder init failed");
        return st;
    }

    rocke_fmha_kernel_builder_block_size(&kb, wave_size);
    fmha_declare_params(&kb);
    rocke_fmha_kernel_builder_decode_grid(&kb, -1, true, NULL, NULL, NULL);

    b = rocke_fmha_kernel_builder_builder(&kb);
    seqlen_q = rocke_fmha_kernel_builder_scalar(&kb, "seqlen_q");
    seqlen_k = rocke_fmha_kernel_builder_scalar(&kb, "seqlen_k");

    q_tile_local = rocke_b_mul(b, kb.q_token, rocke_b_const_i32(b, ROCKE_MFMA_ATTN_BLOCK_M));
    batch_row_q = rocke_b_mul(b, kb.batch_idx, seqlen_q);
    k_batch_offset = rocke_b_mul(b,
                                 rocke_b_mul(b, kb.batch_idx, seqlen_k),
                                 rocke_fmha_kernel_builder_stride_token(&kb, "k"));
    v_batch_offset = rocke_b_mul(b,
                                 rocke_b_mul(b, kb.batch_idx, seqlen_k),
                                 rocke_fmha_kernel_builder_stride_token(&kb, "v"));
    causal_ctx = rocke_b_const_i32(b, 0);

    masked = (common.mask_mode == ROCKE_FMHA_MASK_CAUSAL
              || common.mask_mode == ROCKE_FMHA_MASK_SLIDING_WINDOW);

    memset(&p, 0, sizeof(p));
    p.Q = rocke_fmha_kernel_builder_tensor(&kb, "Q");
    p.K = rocke_fmha_kernel_builder_tensor(&kb, "K");
    p.V = rocke_fmha_kernel_builder_tensor(&kb, "V");
    p.O = rocke_fmha_kernel_builder_tensor(&kb, "O");
    p.head_size = common.shape.head_size;
    p.seqlen_k = seqlen_k;
    p.q_tile_base = rocke_b_add(b, q_tile_local, batch_row_q);
    p.q_pos_base = masked ? q_tile_local : NULL;
    p.head_idx = kb.head_idx;
    p.kv_head_idx = kb.kv_head_idx;
    p.stride_q_token = rocke_fmha_kernel_builder_stride_token(&kb, "q");
    p.stride_q_head = rocke_fmha_kernel_builder_stride_head(&kb, "q");
    p.stride_k_token = rocke_fmha_kernel_builder_stride_token(&kb, "k");
    p.stride_k_head = rocke_fmha_kernel_builder_stride_head(&kb, "k");
    p.stride_v_token = rocke_fmha_kernel_builder_stride_token(&kb, "v");
    p.stride_v_head = rocke_fmha_kernel_builder_stride_head(&kb, "v");
    p.stride_o_token = rocke_fmha_kernel_builder_stride_token(&kb, "o");
    p.stride_o_head = rocke_fmha_kernel_builder_stride_head(&kb, "o");
    p.scale_log2 = rocke_fmha_kernel_builder_scalar(&kb, "scale_log2");
    p.dtype = common.dtype;
    p.mask_mode = fmha_to_attn_mask(common.mask_mode);
    p.sliding_window = common.sliding_window;
    p.causal_ctx_offset = causal_ctx;
    p.k_token_offset_elems = k_batch_offset;
    p.v_token_offset_elems = v_batch_offset;
    p.arch = arch;

    (void)rocke_mfma_attention_fwd_inner_body(b, &p);
    rocke_b_ret(b);

    kernel = rocke_fmha_kernel_builder_kernel(&kb);
    if(kernel == NULL || rocke_ir_builder_status(b) != ROCKE_OK)
    {
        const char* m = rocke_ir_builder_error(b);
        fmha_set_reason(err, err_cap, m != NULL ? m : "build_fmha_fwd_mfma failed");
        rocke_fmha_kernel_builder_free(&kb);
        return ROCKE_ERR_VALUE;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_fmha_kernel_builder_free(&kb);
    return st;
}
