// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/instance_sparse_attention_public_entry_glue.c -- PUBLIC ENTRY / GLUE part
 * of the chunked C99 port of rocke/instances/common/sparse_attention.py.
 *
 * SCOPE OF THIS TU.
 *   The thin public drivers that wire the phase functions (implemented by sibling
 *   .c TUs, declared in rocke/instance_sparse_attention_internal.h) together in the
 *   exact Python order:
 *
 *     rocke_build_jenga_sparse_attention   (mirrors build_jenga_sparse_attention,
 *                                         lines 448-539): stack-allocate a
 *                                         rocke_jenga_sparse_ctx_t, then call
 *                                         rocke_jenga_prologue -> rocke_jenga_stage_mask
 *                                         -> rocke_jenga_emit_body.
 *     rocke_build_vsa_sparse_attention     (mirrors build_vsa_sparse_attention,
 *                                         lines 547-641): the VSA chain
 *                                         rocke_vsa_prologue -> rocke_vsa_stage_bitmap
 *                                         -> rocke_vsa_emit_body.
 *     rocke_{jenga,vsa}_sparse_attention_grid       (lines 644-657).
 *     rocke_{jenga,vsa}_sparse_attention_signature  (lines 660-669): the throwaway
 *                                         "jenga_sig_probe" / "vsa_sig_probe"
 *                                         FmhaKernelBuilder + declare params +
 *                                         return signature().
 *     rocke_{jenga,vsa}_sparse_attention_lower_to_llvm : the build -> lower-to-.ll
 *                                         convenience wrappers (own the builder for
 *                                         the whole lower).
 *
 *   The actual IR emission lives in the phase functions; this TU emits no IR
 *   directly. The build / lower drivers each own a ctx whose embedded
 *   FmhaKernelBuilder owns the IRBuilder and every IR node, exactly like the
 *   Python build that constructs the FmhaKernelBuilder internally.
 */

#include "rocke/instance_sparse_attention.h"
#include "rocke/instance_sparse_attention_internal.h"

#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.mfma_attention.h" /* ROCKE_MFMA_ATTN_BLOCK_M */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_spec_set_reason */
#include "rocke/helper_rocke.instances.common._fmha_common.h"
#include "rocke/lower_llvm.h"

/* ----- small helper: best-effort copy of a reason / diagnostic string. ----- */
static void sparse_set_err(char* err, size_t err_cap, const char* msg)
{
    rocke_spec_set_reason(err, err_cap, msg);
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
 * should use rocke_jenga_sparse_attention_lower_to_llvm (which keeps the builder
 * alive through the whole lower).
 * --------------------------------------------------------------------------- */
rocke_kernel_def_t* rocke_build_jenga_sparse_attention(rocke_ir_builder_t* b_unused,
                                                       const rocke_jenga_sparse_spec_t* spec,
                                                       const char* arch)
{
    return ckc::guard_builder((rocke_ir_builder_t*)nullptr, [&]() -> rocke_kernel_def_t* {
        rocke_jenga_sparse_ctx_t ctx;
        rocke_kernel_def_t* kernel;

        (void)b_unused; /* reserved for signature parity; the driver owns its builder */

        if(spec == NULL)
        {
            return NULL;
        }

        memset(&ctx, 0, sizeof(ctx));
        ctx.spec = spec;
        ctx.arch = (arch != NULL) ? arch : "gfx950";
        ctx.s = spec->common;

        /* Prologue: validity gate + FmhaKernelBuilder init + params + grid + the
         * q_tile_base / q_block_idx / mask_row_base decode. Returns false (with the
         * builder / sticky error set, if any) on a rejected spec. */
        if(!rocke_jenga_prologue(&ctx))
        {
            rocke_fmha_kernel_builder_free(&ctx.kb);
            return NULL;
        }

        /* LDS staging: tid + stage_jenga_mask_to_lds + sync + tiles_per_block_k. */
        rocke_jenga_stage_mask(&ctx);

        /* Inner body + b.ret(); returns kb.kernel (NULL on any builder error). */
        kernel = rocke_jenga_emit_body(&ctx);
        if(kernel == NULL || rocke_ir_builder_status(ctx.b) != ROCKE_OK)
        {
            rocke_fmha_kernel_builder_free(&ctx.kb);
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
rocke_kernel_def_t* rocke_build_vsa_sparse_attention(rocke_ir_builder_t* b_unused,
                                                     const rocke_vsa_sparse_spec_t* spec,
                                                     const char* arch)
{
    return ckc::guard_builder((rocke_ir_builder_t*)nullptr, [&]() -> rocke_kernel_def_t* {
        rocke_vsa_sparse_ctx_t ctx;
        rocke_kernel_def_t* kernel;

        (void)b_unused;

        if(spec == NULL)
        {
            return NULL;
        }

        memset(&ctx, 0, sizeof(ctx));
        ctx.spec = spec;
        ctx.arch = (arch != NULL) ? arch : "gfx950";
        ctx.s = spec->common;

        if(!rocke_vsa_prologue(&ctx))
        {
            rocke_fmha_kernel_builder_free(&ctx.kb);
            return NULL;
        }

        rocke_vsa_stage_bitmap(&ctx);

        kernel = rocke_vsa_emit_body(&ctx);
        if(kernel == NULL || rocke_ir_builder_status(ctx.b) != ROCKE_OK)
        {
            rocke_fmha_kernel_builder_free(&ctx.kb);
            return NULL;
        }
        return kernel;
    });
}

/* --------------------------------------------------------------------------- *
 * jenga_sparse_attention_grid(spec) -- lines 644-649.
 *   return (seqlen_q // MFMA_ATTN_BLOCK_M, num_query_heads, 1)
 * --------------------------------------------------------------------------- */
void rocke_jenga_sparse_attention_grid(const rocke_jenga_sparse_spec_t* spec, int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return;
    }
    out[0] = spec->seqlen_q / ROCKE_MFMA_ATTN_BLOCK_M;
    out[1] = spec->common.shape.num_query_heads;
    out[2] = 1;
}

/* --------------------------------------------------------------------------- *
 * vsa_sparse_attention_grid(spec) -- lines 652-657.
 *   return (seqlen_q // MFMA_ATTN_BLOCK_M, num_query_heads, 1)
 * --------------------------------------------------------------------------- */
void rocke_vsa_sparse_attention_grid(const rocke_vsa_sparse_spec_t* spec, int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return;
    }
    out[0] = spec->seqlen_q / ROCKE_MFMA_ATTN_BLOCK_M;
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
 * (rocke_jenga_declare_params) takes the shared ctx, so we populate a minimal ctx
 * carrying just the spec / common / kb and the spelled-out probe name.
 * --------------------------------------------------------------------------- */
rocke_status_t rocke_jenga_sparse_attention_signature(const rocke_jenga_sparse_spec_t* spec,
                                                      rocke_arena_t* arena,
                                                      const rocke_sig_entry_t** out_items,
                                                      size_t* out_count)
{
    rocke_jenga_sparse_ctx_t ctx;
    rocke_status_t st;

    if(spec == NULL || arena == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.spec = spec;
    ctx.s = spec->common;

    st = rocke_fmha_kernel_builder_init(&ctx.kb, "jenga_sig_probe", &ctx.s);
    if(st != ROCKE_OK)
    {
        return st;
    }
    ctx.b = rocke_fmha_kernel_builder_builder(&ctx.kb);

    rocke_jenga_declare_params(&ctx);

    st = rocke_fmha_kernel_builder_signature(&ctx.kb, arena, out_items, out_count);
    rocke_fmha_kernel_builder_free(&ctx.kb);
    return st;
}

/* --------------------------------------------------------------------------- *
 * vsa_sparse_attention_signature(spec) -- lines 666-669.
 *   kb = FmhaKernelBuilder("vsa_sig_probe", spec.common)
 *   _declare_vsa_params(kb)
 *   return kb.signature()
 * --------------------------------------------------------------------------- */
rocke_status_t rocke_vsa_sparse_attention_signature(const rocke_vsa_sparse_spec_t* spec,
                                                    rocke_arena_t* arena,
                                                    const rocke_sig_entry_t** out_items,
                                                    size_t* out_count)
{
    rocke_vsa_sparse_ctx_t ctx;
    rocke_status_t st;

    if(spec == NULL || arena == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.spec = spec;
    ctx.s = spec->common;

    st = rocke_fmha_kernel_builder_init(&ctx.kb, "vsa_sig_probe", &ctx.s);
    if(st != ROCKE_OK)
    {
        return st;
    }
    ctx.b = rocke_fmha_kernel_builder_builder(&ctx.kb);

    rocke_vsa_declare_params(&ctx);

    st = rocke_fmha_kernel_builder_signature(&ctx.kb, arena, out_items, out_count);
    rocke_fmha_kernel_builder_free(&ctx.kb);
    return st;
}

/* --------------------------------------------------------------------------- *
 * rocke_jenga_sparse_attention_lower_to_llvm -- build + lower to .ll convenience.
 *
 * Owns and frees its own ctx (FmhaKernelBuilder + IRBuilder) for the whole lower
 * so the kernel stays alive through lowering, then bulk-frees on the way out.
 * --------------------------------------------------------------------------- */
rocke_status_t rocke_jenga_sparse_attention_lower_to_llvm(const rocke_jenga_sparse_spec_t* spec,
                                                          const char* arch,
                                                          rocke_llvm_flavor_t flavor,
                                                          char** out_ll,
                                                          char* err,
                                                          size_t err_cap)
{
    rocke_jenga_sparse_ctx_t ctx;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        sparse_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.spec = spec;
    ctx.arch = (arch != NULL) ? arch : "gfx950";
    ctx.s = spec->common;

    if(!rocke_jenga_prologue(&ctx))
    {
        const char* m = (ctx.b != NULL) ? rocke_ir_builder_error(ctx.b) : NULL;
        sparse_set_err(err, err_cap, m != NULL ? m : "invalid jenga_sparse spec");
        rocke_fmha_kernel_builder_free(&ctx.kb);
        return ROCKE_ERR_VALUE;
    }

    rocke_jenga_stage_mask(&ctx);

    kernel = rocke_jenga_emit_body(&ctx);
    if(kernel == NULL || rocke_ir_builder_status(ctx.b) != ROCKE_OK)
    {
        const char* m = rocke_ir_builder_error(ctx.b);
        sparse_set_err(err, err_cap, m != NULL ? m : "build_jenga_sparse_attention failed");
        rocke_fmha_kernel_builder_free(&ctx.kb);
        return ROCKE_ERR_VALUE;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, ctx.arch, out_ll, err, err_cap);
    rocke_fmha_kernel_builder_free(&ctx.kb);
    return st;
}

/* --------------------------------------------------------------------------- *
 * rocke_vsa_sparse_attention_lower_to_llvm -- build + lower to .ll convenience.
 * --------------------------------------------------------------------------- */
rocke_status_t rocke_vsa_sparse_attention_lower_to_llvm(const rocke_vsa_sparse_spec_t* spec,
                                                        const char* arch,
                                                        rocke_llvm_flavor_t flavor,
                                                        char** out_ll,
                                                        char* err,
                                                        size_t err_cap)
{
    rocke_vsa_sparse_ctx_t ctx;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        sparse_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.spec = spec;
    ctx.arch = (arch != NULL) ? arch : "gfx950";
    ctx.s = spec->common;

    if(!rocke_vsa_prologue(&ctx))
    {
        const char* m = (ctx.b != NULL) ? rocke_ir_builder_error(ctx.b) : NULL;
        sparse_set_err(err, err_cap, m != NULL ? m : "invalid vsa_sparse spec");
        rocke_fmha_kernel_builder_free(&ctx.kb);
        return ROCKE_ERR_VALUE;
    }

    rocke_vsa_stage_bitmap(&ctx);

    kernel = rocke_vsa_emit_body(&ctx);
    if(kernel == NULL || rocke_ir_builder_status(ctx.b) != ROCKE_OK)
    {
        const char* m = rocke_ir_builder_error(ctx.b);
        sparse_set_err(err, err_cap, m != NULL ? m : "build_vsa_sparse_attention failed");
        rocke_fmha_kernel_builder_free(&ctx.kb);
        return ROCKE_ERR_VALUE;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, ctx.arch, out_ll, err, err_cap);
    rocke_fmha_kernel_builder_free(&ctx.kb);
    return st;
}
