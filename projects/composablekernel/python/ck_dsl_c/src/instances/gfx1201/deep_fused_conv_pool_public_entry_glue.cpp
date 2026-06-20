// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1201_deep_fused_conv_pool_public_entry_glue.c -- the PUBLIC entry
 * + glue bucket of the chunked C99 port of the gfx1201 (RDNA4, wave32, WMMA
 * 16x16x16) arch shim over build_deep_fused_conv_pool
 *   (ck_dsl/instances/gfx1201/deep_fused_conv_pool.py -- the 72-LOC thin shim
 *    that pins the WMMA geometry and re-exports the common builder).
 *
 * SCOPE OF THIS PART-FILE (the one public/glue bucket):
 *   - ckc_build_gfx1201_deep_fused_conv_pool / _new   (the gfx1201 public build
 *     driver: normalise arch -> "gfx1201", resolve conv_spec + WMMA op, init the
 *     ctx, drive the wrapped build_implicit_gemm_conv with the closure phase fns
 *     + ctx -- byte-faithful to the common build_deep_fused_conv_pool driver).
 *   - ckc_gfx1201_deep_fused_conv_pool_lower_to_llvm   (build -> lower .ll).
 *   - the spec re-exports (thin forwards over &spec->base to the common
 *     ckc_make_deep_fused_conv_pool_spec / ckc_deep_fused_conv_pool_* with the
 *     gfx1201 geometry/name stamped):
 *       ckc_gfx1201_deep_fused_conv_pool_make_spec / _spec_default
 *       ckc_gfx1201_deep_fused_conv_pool_is_valid_spec / _signature / _grid /
 *       _kernel_name.
 *   - ckc_gfx1201_dfcp_build_ctx_init  (the Python prologue: arch normalise,
 *     common_spec=&spec->base, conv_spec + WMMA op via _resolve_conv_op,
 *     defer/deferred_epi, use_wmma_register_maxpool, A-load routing flags).
 *
 * Mirrors the Python gfx1201 shim re-exports + the common
 * build_deep_fused_conv_pool driver. The numeric closure bodies are the peer
 * gfx1201 phase functions declared in the internal header; this bucket only
 * stamps geometry, populates the ctx, and drives the wrapped conv builder.
 *
 * As with the common build-entry bucket, the peer build_implicit_gemm_conv /
 * _resolve_conv_op / spec.conv_spec() ports live behind a header whose
 * ckc_conv_acc_epilogue_t typedef collides with the deep-fused helper's
 * same-named slice typedef, so the peer surface is forward-declared below as
 * opaque externs (matching the byte-faithful Python call pattern); the
 * verify+fix loop links them once the peer ports are wired.
 */
#include "ckc/instance_gfx1201_deep_fused_conv_pool.h"
#include "ckc/instance_gfx1201_deep_fused_conv_pool_internal.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/ir_internal.h"      /* ckc_i_set_err */
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "ckc/lower_llvm.h"

/* ===================================================================== *
 * PEER PORT FORWARD DECLARATIONS (opaque; see file banner)
 *
 * These mirror the Python symbols the wrapped common driver calls:
 *   spec.conv_spec()              -> ckc_deep_fused_conv_pool_spec_conv_spec(...)
 *   _resolve_conv_op(spec, arch)  -> ckc_conv_resolve_op(...)
 *   build_implicit_gemm_conv(...) -> ckc_build_implicit_gemm_conv(...)
 *
 * The peer build entry takes a ckc_conv_build_overrides_t with the six optional
 * callbacks; we declare a structurally-identical local mirror of that struct so
 * this TU type-checks without including the colliding conv public header. The
 * real port's struct layout matches one-for-one (the documented stable callback
 * ABI in ckc/instance_conv_implicit_gemm.h). The gfx1201 spec embeds the common
 * spec as `base`, so the peer is fed &spec->base verbatim.
 * ===================================================================== */

/* spec.conv_spec(): build the wrapped ImplicitGemmConvSpec from the common spec
 * view. Returns NULL + sets b's sticky error on failure. */
/* C++ build: cross-TU C-ABI peers; forward decls must be extern "C". */
#ifdef __cplusplus
extern "C" {
#endif
extern const ckc_implicit_gemm_conv_spec_t*
ckc_deep_fused_conv_pool_spec_conv_spec(ckc_ir_builder_t* b,
                                        const ckc_deep_fused_conv_pool_spec_t* spec);

/* _resolve_conv_op(conv_spec, arch) -> MmaOp. For arch "gfx1201" this resolves
 * the wave32 WMMA 16x16x16 op. Sets b's error + returns NULL on the no-atom
 * ValueError path. */
extern const ckc_mma_op_t* ckc_conv_resolve_op(ckc_ir_builder_t* b,
                                               const ckc_implicit_gemm_conv_spec_t* conv_spec,
                                               const char* arch);
#ifdef __cplusplus
}
#endif

/* The conv builder callback ABI (mirror of ckc_conv_build_overrides_t from
 * ckc/instance_conv_implicit_gemm.h). Layout is the stable override surface. */
typedef struct ckc_gfx1201_dfcp_conv_overrides
{
    void* (*extra_params)(ckc_ir_builder_t* b, void* user);
    ckc_value_t* (*m_index_fn)(ckc_ir_builder_t* b,
                               ckc_value_t* row,
                               const ckc_warp_grid_t* grid,
                               void* user);
    void (*a_mhw_index_fn)(ckc_ir_builder_t* b,
                           ckc_value_t* row,
                           const ckc_warp_grid_t* grid,
                           ckc_value_t** out_n,
                           ckc_value_t** out_ho,
                           ckc_value_t** out_wo,
                           void* user);
    void* (*input_cache_setup)(ckc_ir_builder_t* b,
                               const ckc_implicit_gemm_conv_spec_t* spec,
                               const ckc_warp_grid_t* grid,
                               ckc_value_t* a_rsrc,
                               void* user);
    void (*a_load_override)(ckc_ir_builder_t* b,
                            const ckc_implicit_gemm_conv_spec_t* spec,
                            ckc_value_t* k_off,
                            ckc_value_t* A_dst,
                            const ckc_warp_grid_t* grid,
                            void* input_cache_context,
                            void* user);
    ckc_value_t* (*a_operand_override)(ckc_ir_builder_t* b,
                                       const ckc_implicit_gemm_conv_spec_t* spec,
                                       ckc_value_t* a_row,
                                       ckc_value_t* k_off,
                                       ckc_value_t* col_base,
                                       int a_per_lane,
                                       const ckc_warp_grid_t* grid,
                                       void* input_cache_context,
                                       void* user);
    void (*epilogue_override)(ckc_ir_builder_t* b,
                              const ckc_implicit_gemm_conv_spec_t* spec,
                              ckc_value_t* const* accs,
                              int num_accs,
                              const ckc_warp_grid_t* grid,
                              ckc_value_t* d_rsrc,
                              void* extra_context,
                              void* user);
    void* user;
} ckc_gfx1201_dfcp_conv_overrides_t;

/* build_implicit_gemm_conv(conv_spec, arch=..., **overrides). The canonical
 * prototype comes from ckc/instance_conv_implicit_gemm.h (pulled in transitively
 * through the peer common driver TU); it takes the canonical
 * ckc_conv_build_overrides_t*, layout-identical to the local mirror above. The
 * call site casts the staged mirror to the canonical type. */

/* ===================================================================== *
 * ckc_gfx1201_dfcp_build_ctx_init -- the Python build_deep_fused_conv_pool
 * prologue, driven by the gfx1201 (WMMA) op.
 *
 * Mirrors:
 *   ok, why = is_valid_spec(spec, arch); if not ok: raise ...   (driver gate)
 *   conv_spec = spec.conv_spec()
 *   op = _resolve_conv_op(conv_spec, "gfx1201")
 *   + the override-routing flag derivation done at the build call tail.
 *
 * The validity gate + conv_spec/op resolution are performed by the public driver
 * (it owns the conv_spec/op storage); this init takes them as args and stages the
 * build-time-constant ctx fields, per the internal-header contract:
 * normalise arch -> "gfx1201", common_spec = &spec->base, derive defer /
 * deferred_epi, the WMMA register-residency decision, and the A-load routing
 * flags.
 * ===================================================================== */
ckc_status_t ckc_gfx1201_dfcp_build_ctx_init(ckc_gfx1201_dfcp_build_ctx_t* ctx,
                                             ckc_ir_builder_t* b,
                                             const ckc_gfx1201_deep_fused_conv_pool_spec_t* spec,
                                             const char* arch,
                                             const ckc_implicit_gemm_conv_spec_t* conv_spec,
                                             const ckc_mma_op_t* op)
{
    const ckc_deep_fused_conv_pool_spec_t* cs;

    if(ctx == NULL || b == NULL || spec == NULL)
    {
        return CKC_ERR_VALUE;
    }

    memset(ctx, 0, sizeof(*ctx));

    /* (A) build-time constants ------------------------------------------ */
    ctx->b    = b;
    ctx->spec = spec;
    /* common spec view of the gfx1201 spec (== &spec->base); the closures forward
     * this to the family-agnostic common emit helpers. */
    ctx->common_spec = &spec->base;
    cs               = ctx->common_spec;
    /* arch NULL => "gfx1201" for this shim. */
    ctx->arch      = (arch != NULL) ? arch : CKC_GFX1201_DEEP_FUSED_CONV_POOL_ARCH;
    ctx->conv_spec = conv_spec;
    ctx->op        = op;

    /* defer = _epilogue_is_pool_deferrable(spec.conv1_epilogue)
     * deferred_epi = spec.conv1_epilogue if defer else None
     * (computed once here so each maxpool phase reads the single decision). */
    ctx->defer        = ckc_dfcp_epilogue_is_pool_deferrable(&cs->conv1_epilogue);
    ctx->deferred_epi = ctx->defer ? &cs->conv1_epilogue : NULL;

    /* WMMA register-residency decision (gfx1201 maxpool routing). The
     * per-callback grid is not known at prologue time; the epilogue phase stages
     * the grid and re-evaluates the predicate with it. Staged here as false; the
     * predicate is geometry-gated and re-derived in the epilogue dispatch over
     * the live grid + op (the MFMA-32x32 intra-lane fast path is gated off for
     * WMMA warp_tile 16x16). */
    ctx->use_wmma_register_maxpool = false;

    /* A-load routing (the build_implicit_gemm_conv override selection tail):
     *   use_input_cache  = cache_input_footprint or direct_conv0_from_input_cache
     *   use_specialized  = (not use_input_cache) and
     *                      _can_use_specialized_conv0_a_loader(spec)
     *   use_operand_ovr  = direct_conv0_from_input_cache */
    ctx->use_input_cache = cs->cache_input_footprint || cs->direct_conv0_from_input_cache;
    ctx->use_specialized =
        (!ctx->use_input_cache) && ckc_dfcp_can_use_specialized_conv0_a_loader(cs);
    ctx->use_operand_ovr = cs->direct_conv0_from_input_cache;

    return CKC_OK;
}

/* ===================================================================== *
 * Conv-builder callback trampolines.
 *
 * The conv builder calls the override hooks with its own ABI
 * (ckc_gfx1201_dfcp_conv_overrides_t) + an opaque `user` cookie. We thread the
 * shared ckc_gfx1201_dfcp_build_ctx_t through `user` and forward to the gfx1201
 * ctx phases (declared in the internal header). The conv builder's opaque
 * extra_context / input_cache_context Values are the Values our setup_* phases
 * return; we round-trip them as void*.
 * ===================================================================== */

static void* ckc_gfx1201_dfcp_tramp_extra_params(ckc_ir_builder_t* b, void* user)
{
    ckc_gfx1201_dfcp_build_ctx_t* ctx = (ckc_gfx1201_dfcp_build_ctx_t*)user;
    (void)b;
    return (void*)ckc_gfx1201_dfcp_extra_params(ctx);
}

static ckc_value_t* ckc_gfx1201_dfcp_tramp_m_index_fn(ckc_ir_builder_t* b,
                                                      ckc_value_t* row,
                                                      const ckc_warp_grid_t* grid,
                                                      void* user)
{
    ckc_gfx1201_dfcp_build_ctx_t* ctx = (ckc_gfx1201_dfcp_build_ctx_t*)user;
    (void)b;
    return ckc_gfx1201_dfcp_m_index_fn(ctx, row, grid);
}

static void ckc_gfx1201_dfcp_tramp_a_mhw_index_fn(ckc_ir_builder_t* b,
                                                  ckc_value_t* row,
                                                  const ckc_warp_grid_t* grid,
                                                  ckc_value_t** out_n,
                                                  ckc_value_t** out_ho,
                                                  ckc_value_t** out_wo,
                                                  void* user)
{
    ckc_gfx1201_dfcp_build_ctx_t* ctx = (ckc_gfx1201_dfcp_build_ctx_t*)user;
    (void)b;
    ckc_gfx1201_dfcp_a_mhw_index_fn(ctx, row, grid, out_n, out_ho, out_wo);
}

static void* ckc_gfx1201_dfcp_tramp_input_cache_setup(ckc_ir_builder_t* b,
                                                      const ckc_implicit_gemm_conv_spec_t* spec,
                                                      const ckc_warp_grid_t* grid,
                                                      ckc_value_t* a_rsrc,
                                                      void* user)
{
    ckc_gfx1201_dfcp_build_ctx_t* ctx = (ckc_gfx1201_dfcp_build_ctx_t*)user;
    (void)b;
    /* select setup_input_cache vs setup_specialized_a_loader by the same gate the
     * driver wired (Python's input_cache_setup= ternary). */
    if(ctx->use_input_cache)
    {
        return (void*)ckc_gfx1201_dfcp_setup_input_cache(ctx, spec, grid, a_rsrc);
    }
    return (void*)ckc_gfx1201_dfcp_setup_specialized_a_loader(ctx, spec, grid, a_rsrc);
}

static void ckc_gfx1201_dfcp_tramp_a_load_override(ckc_ir_builder_t* b,
                                                   const ckc_implicit_gemm_conv_spec_t* spec,
                                                   ckc_value_t* k_off,
                                                   ckc_value_t* a_dst,
                                                   const ckc_warp_grid_t* grid,
                                                   void* input_cache_context,
                                                   void* user)
{
    ckc_gfx1201_dfcp_build_ctx_t* ctx = (ckc_gfx1201_dfcp_build_ctx_t*)user;
    ckc_value_t* cache                = (ckc_value_t*)input_cache_context;
    (void)b;
    if(ctx->use_input_cache)
    {
        ckc_gfx1201_dfcp_load_a_tile_from_cache(ctx, spec, k_off, a_dst, grid, cache);
    }
    else
    {
        ckc_gfx1201_dfcp_load_a_tile_specialized(ctx, spec, k_off, a_dst, grid, cache);
    }
}

static ckc_value_t*
ckc_gfx1201_dfcp_tramp_a_operand_override(ckc_ir_builder_t* b,
                                          const ckc_implicit_gemm_conv_spec_t* spec,
                                          ckc_value_t* a_row,
                                          ckc_value_t* k_off,
                                          ckc_value_t* col_base,
                                          int a_per_lane,
                                          const ckc_warp_grid_t* grid,
                                          void* input_cache_context,
                                          void* user)
{
    ckc_gfx1201_dfcp_build_ctx_t* ctx = (ckc_gfx1201_dfcp_build_ctx_t*)user;
    ckc_value_t* cache                = (ckc_value_t*)input_cache_context;
    (void)b;
    return ckc_gfx1201_dfcp_load_a_operand_from_cache(
        ctx, spec, a_row, k_off, col_base, a_per_lane, grid, cache);
}

static void ckc_gfx1201_dfcp_tramp_epilogue_override(ckc_ir_builder_t* b,
                                                     const ckc_implicit_gemm_conv_spec_t* spec,
                                                     ckc_value_t* const* accs,
                                                     int num_accs,
                                                     const ckc_warp_grid_t* grid,
                                                     ckc_value_t* d_rsrc,
                                                     void* extra_context,
                                                     void* user)
{
    ckc_gfx1201_dfcp_build_ctx_t* ctx = (ckc_gfx1201_dfcp_build_ctx_t*)user;
    ckc_value_t* w1_rsrc              = (ckc_value_t*)extra_context;
    (void)b;
    ckc_gfx1201_dfcp_epilogue_override(
        ctx, spec, accs, (num_accs < 0) ? 0u : (size_t)num_accs, grid, d_rsrc, w1_rsrc);
}

/* ===================================================================== *
 * ckc_build_gfx1201_deep_fused_conv_pool -- the gfx1201 public build driver.
 *
 * Mirrors the re-exported common build_deep_fused_conv_pool over the gfx1201
 * spec (arch pinned to "gfx1201"):
 *   ok, why = is_valid_spec(spec, "gfx1201"); if not ok: raise ValueError(...)
 *   conv_spec = spec.conv_spec()
 *   op = _resolve_conv_op(conv_spec, "gfx1201")     # WMMA 16x16x16
 *   <define the nine nested closures>
 *   return build_implicit_gemm_conv(conv_spec, arch="gfx1201", **overrides)
 *
 * Thin arch shim: forwards &spec->base + the normalised arch to the wrapped
 * conv driver; the WMMA emitter path is selected inside the shared builder via
 * the MmaOp resolver.
 * ===================================================================== */
ckc_kernel_def_t*
ckc_build_gfx1201_deep_fused_conv_pool(ckc_ir_builder_t* b_unused,
                                       const ckc_gfx1201_deep_fused_conv_pool_spec_t* spec,
                                       const char* arch)
{
    ckc_ir_builder_t* b = b_unused; /* the surface this routine emits into */
    const ckc_deep_fused_conv_pool_spec_t* cs;
    char reason[256];
    const ckc_implicit_gemm_conv_spec_t* conv_spec;
    const ckc_mma_op_t* op;
    ckc_gfx1201_dfcp_build_ctx_t ctx;
    ckc_gfx1201_dfcp_conv_overrides_t ov;
    ckc_status_t st;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = CKC_GFX1201_DEEP_FUSED_CONV_POOL_ARCH; /* arch default "gfx1201" */
    }
    cs = &spec->base; /* common spec view the wrapped driver consumes */

    /* 1. is_valid_spec gate -> ValueError on reject. */
    reason[0] = '\0';
    if(!ckc_deep_fused_conv_pool_is_valid_spec(cs, arch, reason, sizeof(reason)))
    {
        ckc_i_set_err(
            b, CKC_ERR_VALUE, "invalid deep fused conv/pool spec for %s: %s", arch, reason);
        return NULL;
    }

    /* 2. conv_spec = spec.conv_spec() */
    conv_spec = ckc_deep_fused_conv_pool_spec_conv_spec(b, cs);
    if(conv_spec == NULL)
    {
        return NULL; /* error routed through b */
    }

    /* 3. op = _resolve_conv_op(conv_spec, "gfx1201")  (WMMA 16x16x16) */
    op = ckc_conv_resolve_op(b, conv_spec, arch);
    if(op == NULL)
    {
        return NULL; /* error routed through b */
    }

    /* 4. populate the shared closure ctx (build-time constants + routing). */
    st = ckc_gfx1201_dfcp_build_ctx_init(&ctx, b, spec, arch, conv_spec, op);
    if(st != CKC_OK)
    {
        ckc_i_set_err(b, st, "deep fused conv/pool: ctx init failed");
        return NULL;
    }

    /* 5. wire the override callbacks per the Python build call tail.
     *
     *   extra_params      = extra_params                            (always)
     *   m_index_fn        = m_index_fn                              (always)
     *   a_mhw_index_fn    = a_mhw_index_fn                          (always)
     *   input_cache_setup = setup_input_cache  if use_input_cache
     *                       else setup_specialized_a_loader if use_specialized
     *                       else None
     *   a_load_override   = load_a_tile_from_cache if use_input_cache
     *                       else load_a_tile_specialized if use_specialized
     *                       else None
     *   a_operand_override= load_a_operand_from_cache if use_operand_ovr
     *                       else None
     *   epilogue_override = epilogue_override                       (always)
     *
     * The trampolines select the cache vs specialized phase internally from the
     * ctx flags; install them only when the corresponding Python hook is not
     * None so the conv builder's None-vs-callable behaviour is byte-faithful. */
    memset(&ov, 0, sizeof(ov));
    ov.user           = &ctx;
    ov.extra_params   = ckc_gfx1201_dfcp_tramp_extra_params;
    ov.m_index_fn     = ckc_gfx1201_dfcp_tramp_m_index_fn;
    ov.a_mhw_index_fn = ckc_gfx1201_dfcp_tramp_a_mhw_index_fn;

    if(ctx.use_input_cache || ctx.use_specialized)
    {
        ov.input_cache_setup = ckc_gfx1201_dfcp_tramp_input_cache_setup;
        ov.a_load_override   = ckc_gfx1201_dfcp_tramp_a_load_override;
    }
    if(ctx.use_operand_ovr)
    {
        ov.a_operand_override = ckc_gfx1201_dfcp_tramp_a_operand_override;
    }
    ov.epilogue_override = ckc_gfx1201_dfcp_tramp_epilogue_override;

    /* 6. drive the wrapped conv0 builder; return its kernel. The staged `ov`
     * mirror is layout-identical to the canonical ckc_conv_build_overrides_t. */
    return ckc_build_implicit_gemm_conv(b, conv_spec, arch, (const ckc_conv_build_overrides_t*)&ov);
}

/* Convenience: init `b` with the gfx1201 kernel name, then build. */
ckc_kernel_def_t* ckc_build_gfx1201_deep_fused_conv_pool_new(
    ckc_ir_builder_t* b, const ckc_gfx1201_deep_fused_conv_pool_spec_t* spec, const char* arch)
{
    /* Catch validity/build raises at this boundary so an invalid spec is
     * reported as a clean NULL+error (matching the sibling instances) rather
     * than escaping to std::terminate. */
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_gfx1201_deep_fused_conv_pool_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_gfx1201_deep_fused_conv_pool(b, spec, arch);
    });
}

/* ===================================================================== *
 * ckc_gfx1201_deep_fused_conv_pool_lower_to_llvm -- build + lower to .ll.
 * Owns and frees its own IRBuilder (mirrors the sibling instance ports).
 * `arch` NULL => "gfx1201".
 * ===================================================================== */
ckc_status_t
ckc_gfx1201_deep_fused_conv_pool_lower_to_llvm(const ckc_gfx1201_deep_fused_conv_pool_spec_t* spec,
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
        arch = CKC_GFX1201_DEEP_FUSED_CONV_POOL_ARCH;
    }

    kernel = ckc_build_gfx1201_deep_fused_conv_pool_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(&b);
            if(m == NULL)
            {
                m = "build_gfx1201_deep_fused_conv_pool failed";
            }
            snprintf(err, err_cap, "%s", m);
        }
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}

/* ===================================================================== *
 * Spec re-exports -- the gfx1201-named spec value-type surface
 * (ckc_gfx1201_deep_fused_conv_pool_make_spec / _spec_default /
 *  _is_valid_spec / _signature / _grid / _kernel_name) is DEFINED in the
 * dedicated value-types part-file
 *   instance_gfx1201_deep_fused_conv_pool_value_types_and_spec.c
 * and declared in the public header. This glue translation unit only USES
 * those entries (e.g. _kernel_name above); it no longer redefines them, to
 * avoid duplicate symbols at link time.
 * ===================================================================== */
