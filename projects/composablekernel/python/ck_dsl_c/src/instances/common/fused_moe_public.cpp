// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_fused_moe_fused_moe_public.c -- PUBLIC ENTRY / GLUE for the C99 port
 * of the five MoE-specific kernel builders in
 * ck_dsl/instances/common/fused_moe.py:
 *   build_moe_gather, build_moe_silu_mul, build_moe_silu_mul_packed,
 *   build_moe_static_scatter_gather, build_moe_topk_weighted_reduce.
 *
 * SCOPE (this TU only).
 *   The five build entries -- each mirrors one Python `build_moe_*`: it runs the
 *   shared prologue (which performs the is_valid_spec gate, seeds the kernel
 *   max_workgroup_size, derives the geometry consts, and declares the params in
 *   ABI order) and then the body phase functions in Python source order. For
 *   silu_mul / silu_mul_packed the body dispatches scalar-vs-tile on VEC==1
 *   exactly as the Python `if VEC == 1:` branch does. Each returns b->kernel.
 *
 *   The five `*_new` convenience wrappers init the supplied builder with
 *   spec.kernel_name(phase) (mirroring `b = IRBuilder(spec.kernel_name(...))`)
 *   then delegate to the matching build entry.
 *
 *   The five `*_lower_to_llvm` convenience functions own a local IRBuilder:
 *   init -> build -> lower-to-text -> free, returning a malloc'd .ll string.
 *
 * Peers (prologue + body phase functions, plus the shared module-level helpers)
 * live in sibling TUs and are reached through the internal header. This TU does
 * NOT itself emit IR -- it is pure glue.
 */
#include "ckc/instance_fused_moe.h"
#include "ckc/instance_fused_moe_internal.h"

#include <string.h>
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */

/* arch NULL-normalisation mirrors the Python `_resolve_launch_arch` fallback at
 * IR level: the MoE streaming kernels emit no MFMA, so "gfx950" is the
 * byte-identical baseline. The prologues consume ctx->arch only to stash it. */
#define CKC_MOE_DEFAULT_ARCH "gfx950"

/* Kernel-name buffer: kernel_name_join of the prefix + 9 shape/dtype parts. The
 * longest phase tag is "static_scatter_gather"; 256 is generous headroom. */
#define CKC_MOE_KERNEL_NAME_CAP 256

/* ===================================================================== *
 *  BUILD ENTRIES
 *
 *  Each is the C analog of one Python `build_moe_*` minus the
 *  `b = IRBuilder(spec.kernel_name(...))` line (the caller / `*_new` wrapper
 *  owns builder init). The prologue runs the is_valid_spec gate + param decls;
 *  on a rejected spec it returns false with the builder's sticky error set, and
 *  the entry returns NULL -- the C stand-in for the Python `raise ValueError`.
 * ===================================================================== */

ckc_kernel_def_t*
ckc_build_moe_gather(ckc_ir_builder_t* b, const ckc_fused_moe_spec_t* spec, const char* arch)
{
    ckc_moe_stream_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    memset(&ctx, 0, sizeof(ctx));
    ctx.b    = b;
    ctx.spec = spec;
    ctx.arch = (arch != NULL) ? arch : CKC_MOE_DEFAULT_ARCH;
    ctx.kind = CKC_MOE_STREAM_GATHER;

    /* Python lines 368-412: gate + derive consts + declare params + thread/grid
     * decode + row bases. */
    if(!ckc_moe_gather_prologue(&ctx))
    {
        return NULL;
    }

    /* Python lines 416-439: scf_if(valid_token){ chunk loop }. */
    return ckc_moe_gather_body(&ctx);
}

ckc_kernel_def_t*
ckc_build_moe_silu_mul(ckc_ir_builder_t* b, const ckc_fused_moe_spec_t* spec, const char* arch)
{
    ckc_moe_stream_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    memset(&ctx, 0, sizeof(ctx));
    ctx.b    = b;
    ctx.spec = spec;
    ctx.arch = (arch != NULL) ? arch : CKC_MOE_DEFAULT_ARCH;
    ctx.kind = CKC_MOE_STREAM_SILU_MUL;

    /* Python lines 491-523. */
    if(!ckc_moe_silu_mul_prologue(&ctx))
    {
        return NULL;
    }

    /* Python lines 525-573: `if VEC == 1:` scalar fallback, else tile path. */
    if(ctx.VEC == 1)
    {
        return ckc_moe_silu_mul_body_scalar(&ctx);
    }
    return ckc_moe_silu_mul_body_tile(&ctx);
}

ckc_kernel_def_t* ckc_build_moe_silu_mul_packed(ckc_ir_builder_t* b,
                                                const ckc_fused_moe_spec_t* spec,
                                                const char* arch)
{
    ckc_moe_stream_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    memset(&ctx, 0, sizeof(ctx));
    ctx.b    = b;
    ctx.spec = spec;
    ctx.arch = (arch != NULL) ? arch : CKC_MOE_DEFAULT_ARCH;
    ctx.kind = CKC_MOE_STREAM_SILU_MUL_PACKED;

    /* Python lines 612-648. */
    if(!ckc_moe_silu_mul_packed_prologue(&ctx))
    {
        return NULL;
    }

    /* Python lines 657-708: `if VEC == 1:` scalar fallback, else tile path. */
    if(ctx.VEC == 1)
    {
        return ckc_moe_silu_mul_packed_body_scalar(&ctx);
    }
    return ckc_moe_silu_mul_packed_body_tile(&ctx);
}

ckc_kernel_def_t* ckc_build_moe_static_scatter_gather(ckc_ir_builder_t* b,
                                                      const ckc_fused_moe_spec_t* spec,
                                                      const char* arch)
{
    ckc_moe_ssg_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    memset(&ctx, 0, sizeof(ctx));
    ctx.b    = b;
    ctx.spec = spec;
    ctx.arch = (arch != NULL) ? arch : CKC_MOE_DEFAULT_ARCH;

    /* Python lines 759-803: gate + derive consts + 12 params + smem_alloc +
     * num_pairs + in_bounds. */
    if(!ckc_moe_ssg_prologue(&ctx))
    {
        return NULL;
    }

    /* Python lines 805-819: slot-claim nest (opens scf_if scopes that stay open
     * for the copy phase). */
    ckc_moe_ssg_claim(&ctx);

    /* Python lines 820-847: sync + broadcast + interleaved copy; closes the
     * scopes and returns b->kernel. */
    return ckc_moe_ssg_copy(&ctx);
}

ckc_kernel_def_t* ckc_build_moe_topk_weighted_reduce(ckc_ir_builder_t* b,
                                                     const ckc_fused_moe_spec_t* spec,
                                                     const char* arch)
{
    ckc_moe_stream_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    memset(&ctx, 0, sizeof(ctx));
    ctx.b    = b;
    ctx.spec = spec;
    ctx.arch = (arch != NULL) ? arch : CKC_MOE_DEFAULT_ARCH;
    ctx.kind = CKC_MOE_STREAM_REDUCE;

    /* Python lines 911-958, 979-980. */
    if(!ckc_moe_reduce_prologue(&ctx))
    {
        return NULL;
    }

    /* Python lines 981-995: scf_if(valid_token){ block-partitioned chunk loop }. */
    return ckc_moe_reduce_body(&ctx);
}

/* ===================================================================== *
 *  *_new CONVENIENCE WRAPPERS
 *
 *  Each is `b = IRBuilder(spec.kernel_name(phase)); return build(...)`: init the
 *  caller-owned builder with the phase kernel name, then delegate. On a name /
 *  init failure return NULL without building. The caller owns `b` and frees it.
 * ===================================================================== */

/* Shared: derive spec.kernel_name(phase) and init `b` with it. Returns CKC_OK on
 * success; on failure the builder is left in whatever state init produced and the
 * caller returns NULL. */
static ckc_status_t
ckc_moe_init_named(ckc_ir_builder_t* b, const ckc_fused_moe_spec_t* spec, const char* phase)
{
    char name[CKC_MOE_KERNEL_NAME_CAP];
    ckc_status_t st;

    st = ckc_fused_moe_spec_kernel_name(spec, phase, name, sizeof(name));
    if(st != CKC_OK)
    {
        return st;
    }
    return ckc_ir_builder_init(b, name);
}

ckc_kernel_def_t*
ckc_build_moe_gather_new(ckc_ir_builder_t* b, const ckc_fused_moe_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_moe_init_named(b, spec, "gather") != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_moe_gather(b, spec, arch);
    });
}

ckc_kernel_def_t*
ckc_build_moe_silu_mul_new(ckc_ir_builder_t* b, const ckc_fused_moe_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_moe_init_named(b, spec, "silu_mul") != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_moe_silu_mul(b, spec, arch);
    });
}

ckc_kernel_def_t* ckc_build_moe_silu_mul_packed_new(ckc_ir_builder_t* b,
                                                    const ckc_fused_moe_spec_t* spec,
                                                    const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_moe_init_named(b, spec, "silu_mul_packed") != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_moe_silu_mul_packed(b, spec, arch);
    });
}

ckc_kernel_def_t* ckc_build_moe_static_scatter_gather_new(ckc_ir_builder_t* b,
                                                          const ckc_fused_moe_spec_t* spec,
                                                          const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_moe_init_named(b, spec, "static_scatter_gather") != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_moe_static_scatter_gather(b, spec, arch);
    });
}

ckc_kernel_def_t* ckc_build_moe_topk_weighted_reduce_new(ckc_ir_builder_t* b,
                                                         const ckc_fused_moe_spec_t* spec,
                                                         const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        /* Python build_moe_topk_weighted_reduce uses the "reduce" phase tag. */
        if(ckc_moe_init_named(b, spec, "reduce") != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_moe_topk_weighted_reduce(b, spec, arch);
    });
}

/* ===================================================================== *
 *  *_lower_to_llvm CONVENIENCE
 *
 *  build -> lower-to-text. Each owns a local IRBuilder for the whole call: init
 *  via the matching `*_new` wrapper (which seeds the kernel name), build, lower
 *  the resulting kernel to LLVM .ll text, then free the builder. On CKC_OK
 *  *out_ll receives a malloc'd NUL-terminated string the caller frees with
 *  free(); on failure it is left NULL.
 * ===================================================================== */

/* Build function pointer for the shared lower driver. */
typedef ckc_kernel_def_t* (*ckc_moe_build_new_fn)(ckc_ir_builder_t* b,
                                                  const ckc_fused_moe_spec_t* spec,
                                                  const char* arch);

/* Shared lower driver: owns + frees the IRBuilder; builds via `build_new`, then
 * lowers b->kernel to .ll. Mirrors the Python convenience flow exactly. */
static ckc_status_t ckc_moe_lower_to_llvm_impl(ckc_moe_build_new_fn build_new,
                                               const ckc_fused_moe_spec_t* spec,
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
    if(build_new == NULL || spec == NULL || out_ll == NULL)
    {
        return CKC_ERR_VALUE;
    }

    kernel = build_new(&b, spec, arch);
    if(kernel == NULL)
    {
        /* Surface the builder's sticky diagnostic (init or build failure). */
        st = ckc_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* msg = ckc_ir_builder_error(&b);
            if(msg != NULL)
            {
                size_t n = strlen(msg);
                if(n >= err_cap)
                {
                    n = err_cap - 1;
                }
                memcpy(err, msg, n);
                err[n] = '\0';
            }
            else
            {
                err[0] = '\0';
            }
        }
        ckc_ir_builder_free(&b);
        return (st != CKC_OK) ? st : CKC_ERR_VALUE;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);

    ckc_ir_builder_free(&b);
    return st;
}

ckc_status_t ckc_moe_gather_lower_to_llvm(const ckc_fused_moe_spec_t* spec,
                                          const char* arch,
                                          ckc_llvm_flavor_t flavor,
                                          char** out_ll,
                                          char* err,
                                          size_t err_cap)
{
    return ckc_moe_lower_to_llvm_impl(
        ckc_build_moe_gather_new, spec, arch, flavor, out_ll, err, err_cap);
}

ckc_status_t ckc_moe_silu_mul_lower_to_llvm(const ckc_fused_moe_spec_t* spec,
                                            const char* arch,
                                            ckc_llvm_flavor_t flavor,
                                            char** out_ll,
                                            char* err,
                                            size_t err_cap)
{
    return ckc_moe_lower_to_llvm_impl(
        ckc_build_moe_silu_mul_new, spec, arch, flavor, out_ll, err, err_cap);
}

ckc_status_t ckc_moe_silu_mul_packed_lower_to_llvm(const ckc_fused_moe_spec_t* spec,
                                                   const char* arch,
                                                   ckc_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap)
{
    return ckc_moe_lower_to_llvm_impl(
        ckc_build_moe_silu_mul_packed_new, spec, arch, flavor, out_ll, err, err_cap);
}

ckc_status_t ckc_moe_static_scatter_gather_lower_to_llvm(const ckc_fused_moe_spec_t* spec,
                                                         const char* arch,
                                                         ckc_llvm_flavor_t flavor,
                                                         char** out_ll,
                                                         char* err,
                                                         size_t err_cap)
{
    return ckc_moe_lower_to_llvm_impl(
        ckc_build_moe_static_scatter_gather_new, spec, arch, flavor, out_ll, err, err_cap);
}

ckc_status_t ckc_moe_topk_weighted_reduce_lower_to_llvm(const ckc_fused_moe_spec_t* spec,
                                                        const char* arch,
                                                        ckc_llvm_flavor_t flavor,
                                                        char** out_ll,
                                                        char* err,
                                                        size_t err_cap)
{
    return ckc_moe_lower_to_llvm_impl(
        ckc_build_moe_topk_weighted_reduce_new, spec, arch, flavor, out_ll, err, err_cap);
}
