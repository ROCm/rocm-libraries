// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * ckc/instance_sparse_attention_public_entry_glue.c -- PUBLIC ENTRY / GLUE part
 * of the chunked C99 port of ck_dsl/instances/common/sparse_attention.py.
 *
 * SCOPE OF THIS TU.
 *   The thin public drivers that wire the phase functions (implemented by sibling
 *   .c TUs, declared in ckc/instance_sparse_attention_internal.h) together in the
 *   exact Python order:
 *
 *     ckc_build_jenga_sparse_attention   (mirrors build_jenga_sparse_attention,
 *                                         lines 448-539): stack-allocate a
 *                                         ckc_jenga_sparse_ctx_t, then call
 *                                         ckc_jenga_prologue -> ckc_jenga_stage_mask
 *                                         -> ckc_jenga_emit_body.
 *     ckc_build_vsa_sparse_attention     (mirrors build_vsa_sparse_attention,
 *                                         lines 547-641): the VSA chain
 *                                         ckc_vsa_prologue -> ckc_vsa_stage_bitmap
 *                                         -> ckc_vsa_emit_body.
 *     ckc_{jenga,vsa}_sparse_attention_grid       (lines 644-657).
 *     ckc_{jenga,vsa}_sparse_attention_signature  (lines 660-669): the throwaway
 *                                         "jenga_sig_probe" / "vsa_sig_probe"
 *                                         FmhaKernelBuilder + declare params +
 *                                         return signature().
 *     ckc_{jenga,vsa}_sparse_attention_lower_to_llvm : the build -> lower-to-.ll
 *                                         convenience wrappers (own the builder for
 *                                         the whole lower).
 *
 *   The actual IR emission lives in the phase functions; this TU emits no IR
 *   directly. The build / lower drivers each own a ctx whose embedded
 *   FmhaKernelBuilder owns the IRBuilder and every IR node, exactly like the
 *   Python build that constructs the FmhaKernelBuilder internally.
 */

#include "ckc/instance_sparse_attention.h"
#include "ckc/instance_sparse_attention_internal.h"

#include <string.h>

#include "ckc/arena.h"
#include "ckc/lower_llvm.h"
#include "ckc/helper_ck_dsl.helpers.mfma_attention.h" /* CKC_MFMA_ATTN_BLOCK_M */
#include "ckc/helper_ck_dsl.instances.common._fmha_common.h"
#include "ckc/helper_ck_dsl.helpers.spec.h" /* ckc_spec_set_reason */
#include "ckc/error_boundary.hpp"           /* ckc::guard_builder boundary shim */

/* ----- small helper: best-effort copy of a reason / diagnostic string. ----- */
static void sparse_set_err(char* err, size_t err_cap, const char* msg)
{
    ckc_spec_set_reason(err, err_cap, msg);
}

/* --------------------------------------------------------------------------- *
 * build_jenga_sparse_attention(spec, arch=...) -- lines 448-539.
 *
 *   ok, why = is_valid_jenga_spec(spec, arch)   (inside the prologue)
 *   kb = FmhaKernelBuilder(spec.kernel_name(), s); kb.block_size(64);
 *   _declare_jenga_params(kb); kb.decode_grid(); ... q_block decode  (prologue)
 *   tid = ...; mask_lds = _stage_jenga_mask_to_lds(...); b.sync();   (stage_mask)
 *   tiles_per_block_k = ...; mfma_attention_fwd_inner_body(...); b.ret()
 *                                                                  (emit_body)
 *   return kb.kernel
 *
 * The ctx is stack-allocated here (its embedded kb owns the IRBuilder + all IR
 * nodes); we intentionally do NOT free the ctx's builder on the success path so
 * the returned KernelDef stays valid for an immediate same-scope lower (matching
 * the sibling instance entries). Callers needing the kernel to outlive this call
 * should use ckc_jenga_sparse_attention_lower_to_llvm (which keeps the builder
 * alive through the whole lower).
 * --------------------------------------------------------------------------- */
ckc_kernel_def_t* ckc_build_jenga_sparse_attention(ckc_ir_builder_t* b_unused,
                                                   const ckc_jenga_sparse_spec_t* spec,
                                                   const char* arch)
{
    return ckc::guard_builder((ckc_ir_builder_t*)nullptr, [&]() -> ckc_kernel_def_t* {
        ckc_jenga_sparse_ctx_t ctx;
        ckc_kernel_def_t* kernel;

        (void)b_unused; /* reserved for signature parity; the driver owns its builder */

        if(spec == NULL)
        {
            return NULL;
        }

        memset(&ctx, 0, sizeof(ctx));
        ctx.spec = spec;
        ctx.arch = (arch != NULL) ? arch : "gfx950";
        ctx.s    = spec->common;

        /* Prologue: validity gate + FmhaKernelBuilder init + params + grid + the
         * q_tile_base / q_block_idx / mask_row_base decode. Returns false (with the
         * builder / sticky error set, if any) on a rejected spec. */
        if(!ckc_jenga_prologue(&ctx))
        {
            ckc_fmha_kernel_builder_free(&ctx.kb);
            return NULL;
        }

        /* LDS staging: tid + stage_jenga_mask_to_lds + sync + tiles_per_block_k. */
        ckc_jenga_stage_mask(&ctx);

        /* Inner body + b.ret(); returns kb.kernel (NULL on any builder error). */
        kernel = ckc_jenga_emit_body(&ctx);
        if(kernel == NULL || ckc_ir_builder_status(ctx.b) != CKC_OK)
        {
            ckc_fmha_kernel_builder_free(&ctx.kb);
            return NULL;
        }
        return kernel;
    });
}

/* --------------------------------------------------------------------------- *
 * build_vsa_sparse_attention(spec, arch=...) -- lines 547-641.
 *
 *   (prologue) is_valid_vsa_spec gate; FmhaKernelBuilder init; block_size(64);
 *              _declare_vsa_params; decode_grid; q_tile_base / q_block_idx /
 *              lut_row_base decode.
 *   (stage_bitmap) tid; bitmap_lds = _stage_vsa_bitmap_to_lds(...); b.sync();
 *                  tiles_per_block_k.
 *   (emit_body) mfma_attention_fwd_inner_body(...); b.ret(); return kb.kernel.
 * --------------------------------------------------------------------------- */
ckc_kernel_def_t* ckc_build_vsa_sparse_attention(ckc_ir_builder_t* b_unused,
                                                 const ckc_vsa_sparse_spec_t* spec,
                                                 const char* arch)
{
    return ckc::guard_builder((ckc_ir_builder_t*)nullptr, [&]() -> ckc_kernel_def_t* {
        ckc_vsa_sparse_ctx_t ctx;
        ckc_kernel_def_t* kernel;

        (void)b_unused;

        if(spec == NULL)
        {
            return NULL;
        }

        memset(&ctx, 0, sizeof(ctx));
        ctx.spec = spec;
        ctx.arch = (arch != NULL) ? arch : "gfx950";
        ctx.s    = spec->common;

        if(!ckc_vsa_prologue(&ctx))
        {
            ckc_fmha_kernel_builder_free(&ctx.kb);
            return NULL;
        }

        ckc_vsa_stage_bitmap(&ctx);

        kernel = ckc_vsa_emit_body(&ctx);
        if(kernel == NULL || ckc_ir_builder_status(ctx.b) != CKC_OK)
        {
            ckc_fmha_kernel_builder_free(&ctx.kb);
            return NULL;
        }
        return kernel;
    });
}

/* --------------------------------------------------------------------------- *
 * jenga_sparse_attention_grid(spec) -- lines 644-649.
 *   return (seqlen_q // MFMA_ATTN_BLOCK_M, num_query_heads, 1)
 * --------------------------------------------------------------------------- */
void ckc_jenga_sparse_attention_grid(const ckc_jenga_sparse_spec_t* spec, int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return;
    }
    out[0] = spec->seqlen_q / CKC_MFMA_ATTN_BLOCK_M;
    out[1] = spec->common.shape.num_query_heads;
    out[2] = 1;
}

/* --------------------------------------------------------------------------- *
 * vsa_sparse_attention_grid(spec) -- lines 652-657.
 *   return (seqlen_q // MFMA_ATTN_BLOCK_M, num_query_heads, 1)
 * --------------------------------------------------------------------------- */
void ckc_vsa_sparse_attention_grid(const ckc_vsa_sparse_spec_t* spec, int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return;
    }
    out[0] = spec->seqlen_q / CKC_MFMA_ATTN_BLOCK_M;
    out[1] = spec->common.shape.num_query_heads;
    out[2] = 1;
}

/* --------------------------------------------------------------------------- *
 * jenga_sparse_attention_signature(spec) -- lines 660-663.
 *   kb = FmhaKernelBuilder("jenga_sig_probe", spec.common)
 *   _declare_jenga_params(kb)
 *   return kb.signature()
 *
 * The Python builds a throwaway probe builder (no block_size / decode_grid /
 * body); only the declared param order matters for the ABI. The C declare phase
 * (ckc_jenga_declare_params) takes the shared ctx, so we populate a minimal ctx
 * carrying just the spec / common / kb and the spelled-out probe name.
 * --------------------------------------------------------------------------- */
ckc_status_t ckc_jenga_sparse_attention_signature(const ckc_jenga_sparse_spec_t* spec,
                                                  ckc_arena_t* arena,
                                                  const ckc_sig_entry_t** out_items,
                                                  size_t* out_count)
{
    ckc_jenga_sparse_ctx_t ctx;
    ckc_status_t st;

    if(spec == NULL || arena == NULL || out_items == NULL || out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.spec = spec;
    ctx.s    = spec->common;

    st = ckc_fmha_kernel_builder_init(&ctx.kb, "jenga_sig_probe", &ctx.s);
    if(st != CKC_OK)
    {
        return st;
    }
    ctx.b = ckc_fmha_kernel_builder_builder(&ctx.kb);

    ckc_jenga_declare_params(&ctx);

    st = ckc_fmha_kernel_builder_signature(&ctx.kb, arena, out_items, out_count);
    ckc_fmha_kernel_builder_free(&ctx.kb);
    return st;
}

/* --------------------------------------------------------------------------- *
 * vsa_sparse_attention_signature(spec) -- lines 666-669.
 *   kb = FmhaKernelBuilder("vsa_sig_probe", spec.common)
 *   _declare_vsa_params(kb)
 *   return kb.signature()
 * --------------------------------------------------------------------------- */
ckc_status_t ckc_vsa_sparse_attention_signature(const ckc_vsa_sparse_spec_t* spec,
                                                ckc_arena_t* arena,
                                                const ckc_sig_entry_t** out_items,
                                                size_t* out_count)
{
    ckc_vsa_sparse_ctx_t ctx;
    ckc_status_t st;

    if(spec == NULL || arena == NULL || out_items == NULL || out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.spec = spec;
    ctx.s    = spec->common;

    st = ckc_fmha_kernel_builder_init(&ctx.kb, "vsa_sig_probe", &ctx.s);
    if(st != CKC_OK)
    {
        return st;
    }
    ctx.b = ckc_fmha_kernel_builder_builder(&ctx.kb);

    ckc_vsa_declare_params(&ctx);

    st = ckc_fmha_kernel_builder_signature(&ctx.kb, arena, out_items, out_count);
    ckc_fmha_kernel_builder_free(&ctx.kb);
    return st;
}

/* --------------------------------------------------------------------------- *
 * ckc_jenga_sparse_attention_lower_to_llvm -- build + lower to .ll convenience.
 *
 * Owns and frees its own ctx (FmhaKernelBuilder + IRBuilder) for the whole lower
 * so the kernel stays alive through lowering, then bulk-frees on the way out.
 * --------------------------------------------------------------------------- */
ckc_status_t ckc_jenga_sparse_attention_lower_to_llvm(const ckc_jenga_sparse_spec_t* spec,
                                                      const char* arch,
                                                      ckc_llvm_flavor_t flavor,
                                                      char** out_ll,
                                                      char* err,
                                                      size_t err_cap)
{
    ckc_jenga_sparse_ctx_t ctx;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        sparse_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.spec = spec;
    ctx.arch = (arch != NULL) ? arch : "gfx950";
    ctx.s    = spec->common;

    if(!ckc_jenga_prologue(&ctx))
    {
        const char* m = (ctx.b != NULL) ? ckc_ir_builder_error(ctx.b) : NULL;
        sparse_set_err(err, err_cap, m != NULL ? m : "invalid jenga_sparse spec");
        ckc_fmha_kernel_builder_free(&ctx.kb);
        return CKC_ERR_VALUE;
    }

    ckc_jenga_stage_mask(&ctx);

    kernel = ckc_jenga_emit_body(&ctx);
    if(kernel == NULL || ckc_ir_builder_status(ctx.b) != CKC_OK)
    {
        const char* m = ckc_ir_builder_error(ctx.b);
        sparse_set_err(err, err_cap, m != NULL ? m : "build_jenga_sparse_attention failed");
        ckc_fmha_kernel_builder_free(&ctx.kb);
        return CKC_ERR_VALUE;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, ctx.arch, out_ll, err, err_cap);
    ckc_fmha_kernel_builder_free(&ctx.kb);
    return st;
}

/* --------------------------------------------------------------------------- *
 * ckc_vsa_sparse_attention_lower_to_llvm -- build + lower to .ll convenience.
 * --------------------------------------------------------------------------- */
ckc_status_t ckc_vsa_sparse_attention_lower_to_llvm(const ckc_vsa_sparse_spec_t* spec,
                                                    const char* arch,
                                                    ckc_llvm_flavor_t flavor,
                                                    char** out_ll,
                                                    char* err,
                                                    size_t err_cap)
{
    ckc_vsa_sparse_ctx_t ctx;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        sparse_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.spec = spec;
    ctx.arch = (arch != NULL) ? arch : "gfx950";
    ctx.s    = spec->common;

    if(!ckc_vsa_prologue(&ctx))
    {
        const char* m = (ctx.b != NULL) ? ckc_ir_builder_error(ctx.b) : NULL;
        sparse_set_err(err, err_cap, m != NULL ? m : "invalid vsa_sparse spec");
        ckc_fmha_kernel_builder_free(&ctx.kb);
        return CKC_ERR_VALUE;
    }

    ckc_vsa_stage_bitmap(&ctx);

    kernel = ckc_vsa_emit_body(&ctx);
    if(kernel == NULL || ckc_ir_builder_status(ctx.b) != CKC_OK)
    {
        const char* m = ckc_ir_builder_error(ctx.b);
        sparse_set_err(err, err_cap, m != NULL ? m : "build_vsa_sparse_attention failed");
        ckc_fmha_kernel_builder_free(&ctx.kb);
        return CKC_ERR_VALUE;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, ctx.arch, out_ll, err, err_cap);
    ckc_fmha_kernel_builder_free(&ctx.kb);
    return st;
}
