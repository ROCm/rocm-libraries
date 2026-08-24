// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx950_deep_fused_conv_pool_gfx950_dfcp_public_entry_glue.c -- the
 * PUBLIC entry + glue bucket of the chunked C99 port of the gfx950 (CDNA, wave64,
 * MFMA 32x32x16) arch shim over build_deep_fused_conv_pool
 *   (rocke/instances/gfx950/deep_fused_conv_pool.py -- the 67-LOC thin shim that
 *    pins the MFMA geometry and re-exports the common builder).
 *
 * SCOPE OF THIS PART-FILE (the one public/glue bucket):
 *   - the spec value-type re-exports (thin forwards over &spec->base to the common
 *     rocke_make_deep_fused_conv_pool_spec / rocke_deep_fused_conv_pool_* with the
 *     gfx950 geometry/name stamped):
 *       rocke_gfx950_deep_fused_conv_pool_make_spec / _spec_default
 *       rocke_gfx950_deep_fused_conv_pool_is_valid_spec / _signature / _grid /
 *       _kernel_name.
 *   - rocke_gfx950_dfcp_build_ctx_init  (the Python prologue: arch normalise,
 *     common_spec=&spec->base, conv_spec + MFMA op via _resolve_conv_op,
 *     defer/deferred_epi, use_intra_lane_maxpool, A-load routing flags).
 *   - rocke_build_gfx950_deep_fused_conv_pool / _new   (the gfx950 public build
 *     driver: normalise arch -> "gfx950", resolve conv_spec + MFMA op, init the
 *     ctx, drive the wrapped build_implicit_gemm_conv with the nine closure phase
 *     trampolines + ctx -- byte-faithful to the common build_deep_fused_conv_pool
 *     driver).
 *   - rocke_gfx950_deep_fused_conv_pool_lower_to_llvm   (build -> lower .ll).
 *
 * Mirrors the Python gfx950 shim re-exports + the common build_deep_fused_conv_pool
 * driver. The numeric closure bodies are the peer gfx950 phase functions declared
 * in the internal header; this bucket only stamps geometry, populates the ctx, and
 * drives the wrapped conv builder.
 *
 * As with the common build-entry bucket, the peer build_implicit_gemm_conv /
 * _resolve_conv_op / spec.conv_spec() ports live behind a header whose
 * rocke_conv_acc_epilogue_t typedef collides with the deep-fused helper's same-named
 * slice typedef, so the peer surface is forward-declared below as opaque externs
 * (matching the byte-faithful Python call pattern), with a structurally-identical
 * local mirror of the conv override struct; the verify+fix loop links them once
 * the peer ports are wired.
 */
#include "rocke/instance_gfx950_deep_fused_conv_pool.h"
#include "rocke/instance_gfx950_deep_fused_conv_pool_internal.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */
#include "rocke/lower_llvm.h"

/* ------------------------------------------------------------------ *
 * The gfx950-pinned geometry / name the shim stamps (Python lines 38-66).
 * These mirror the public-header macros (kept local-named for clarity at the
 * stamping sites); they are the CDNA MFMA defaults (wave_size=64, warp_tile
 * 32x32x16) -- which are ALSO the common defaults -- plus the gfx950 kernel name.
 * ------------------------------------------------------------------ */
#define GFX950_DFCP_NAME ROCKE_GFX950_DEEP_FUSED_CONV_POOL_NAME
#define GFX950_DFCP_WAVE_SIZE ROCKE_GFX950_DEEP_FUSED_CONV_POOL_WAVE_SIZE
#define GFX950_DFCP_WARP_TILE_M ROCKE_GFX950_DEEP_FUSED_CONV_POOL_WARP_TILE_M
#define GFX950_DFCP_WARP_TILE_N ROCKE_GFX950_DEEP_FUSED_CONV_POOL_WARP_TILE_N
#define GFX950_DFCP_WARP_TILE_K ROCKE_GFX950_DEEP_FUSED_CONV_POOL_WARP_TILE_K
#define GFX950_DFCP_ARCH ROCKE_GFX950_DEEP_FUSED_CONV_POOL_ARCH

/* ===================================================================== *
 * PEER PORT FORWARD DECLARATIONS (opaque; see file banner)
 *
 * These mirror the Python symbols the wrapped common driver calls:
 *   spec.conv_spec()              -> rocke_deep_fused_conv_pool_spec_conv_spec(...)
 *   _resolve_conv_op(spec, arch)  -> rocke_conv_resolve_op(...)
 *   build_implicit_gemm_conv(...) -> rocke_build_implicit_gemm_conv(...)
 *
 * The peer build entry takes a rocke_conv_build_overrides_t with the six optional
 * callbacks; we declare a structurally-identical local mirror of that struct so
 * this TU type-checks without including the colliding conv public header. The real
 * port's struct layout matches one-for-one (the documented stable callback ABI in
 * rocke/instance_conv_implicit_gemm.h). The gfx950 spec embeds the common spec as
 * `base`, so the peer is fed &spec->base verbatim.
 * ===================================================================== */

/* spec.conv_spec(): build the wrapped ImplicitGemmConvSpec from the common spec
 * view. Returns NULL + sets b's sticky error on failure. */
/* C++ build: cross-TU C-ABI peers; forward decls must be extern "C". */
#ifdef __cplusplus
extern "C" {
#endif
extern const rocke_implicit_gemm_conv_spec_t*
    rocke_deep_fused_conv_pool_spec_conv_spec(rocke_ir_builder_t* b,
                                              const rocke_deep_fused_conv_pool_spec_t* spec);

/* _resolve_conv_op(conv_spec, arch) -> MmaOp. For arch "gfx950" this resolves the
 * wave64 MFMA 32x32x16 op. Sets b's error + returns NULL on the no-atom
 * ValueError path. */
extern const rocke_mma_op_t* rocke_conv_resolve_op(rocke_ir_builder_t* b,
                                                   const rocke_implicit_gemm_conv_spec_t* conv_spec,
                                                   const char* arch);
#ifdef __cplusplus
}
#endif

/* The conv builder callback ABI (mirror of rocke_conv_build_overrides_t from
 * rocke/instance_conv_implicit_gemm.h). Layout is the stable override surface. */
typedef struct rocke_gfx950_dfcp_conv_overrides
{
    void* (*extra_params)(rocke_ir_builder_t* b, void* user);
    rocke_value_t* (*m_index_fn)(rocke_ir_builder_t* b,
                                 rocke_value_t* row,
                                 const rocke_warp_grid_t* grid,
                                 void* user);
    void (*a_mhw_index_fn)(rocke_ir_builder_t* b,
                           rocke_value_t* row,
                           const rocke_warp_grid_t* grid,
                           rocke_value_t** out_n,
                           rocke_value_t** out_ho,
                           rocke_value_t** out_wo,
                           void* user);
    void* (*input_cache_setup)(rocke_ir_builder_t* b,
                               const rocke_implicit_gemm_conv_spec_t* spec,
                               const rocke_warp_grid_t* grid,
                               rocke_value_t* a_rsrc,
                               void* user);
    void (*a_load_override)(rocke_ir_builder_t* b,
                            const rocke_implicit_gemm_conv_spec_t* spec,
                            rocke_value_t* k_off,
                            rocke_value_t* A_dst,
                            const rocke_warp_grid_t* grid,
                            void* input_cache_context,
                            void* user);
    rocke_value_t* (*a_operand_override)(rocke_ir_builder_t* b,
                                         const rocke_implicit_gemm_conv_spec_t* spec,
                                         rocke_value_t* a_row,
                                         rocke_value_t* k_off,
                                         rocke_value_t* col_base,
                                         int a_per_lane,
                                         const rocke_warp_grid_t* grid,
                                         void* input_cache_context,
                                         void* user);
    void (*epilogue_override)(rocke_ir_builder_t* b,
                              const rocke_implicit_gemm_conv_spec_t* spec,
                              rocke_value_t* const* accs,
                              int num_accs,
                              const rocke_warp_grid_t* grid,
                              rocke_value_t* d_rsrc,
                              void* extra_context,
                              void* user);
    void* user;
} rocke_gfx950_dfcp_conv_overrides_t;

/* build_implicit_gemm_conv(conv_spec, arch=..., **overrides). The canonical
 * prototype comes from rocke/instance_conv_implicit_gemm.h (pulled in transitively
 * through the peer common driver TU); it takes the canonical
 * rocke_conv_build_overrides_t*, layout-identical to the local mirror above. The
 * call site casts the staged mirror to the canonical type. */

/* ------------------------------------------------------------------ *
 * Gfx950DeepFusedConvPoolSpec dataclass defaults
 * ------------------------------------------------------------------ *
 *
 * Python (gfx950/deep_fused_conv_pool.py lines 41-52):
 *   @dataclass(frozen=True)
 *   class Gfx950DeepFusedConvPoolSpec(DeepFusedConvPoolSpec):
 *       name: str = _GFX950_NAME
 *
 * i.e. the common DeepFusedConvPoolSpec defaults with only `name` overridden (the
 * wave64 32x32x16 geometry is already the common default). The C mirror takes the
 * common default spec and stamps the gfx950 name onto the embedded `base` (it also
 * restamps the geometry, which is a no-op since it equals the common default, to
 * make the gfx950 pinning explicit); the caller fills base.problem. */
rocke_gfx950_deep_fused_conv_pool_spec_t rocke_gfx950_deep_fused_conv_pool_spec_default(void)
{
    rocke_gfx950_deep_fused_conv_pool_spec_t s;

    /* The common DeepFusedConvPoolSpec dataclass defaults. */
    s.base = rocke_deep_fused_conv_pool_spec_default();

    /* The gfx950 dataclass field override (Python line 52): name only. The
     * geometry restamps below are no-ops vs the common default but make the gfx950
     * pinning explicit / robust to future common-default drift. */
    s.base.name = GFX950_DFCP_NAME;
    s.base.wave_size = GFX950_DFCP_WAVE_SIZE;
    s.base.warp_tile_m = GFX950_DFCP_WARP_TILE_M;
    s.base.warp_tile_n = GFX950_DFCP_WARP_TILE_N;
    s.base.warp_tile_k = GFX950_DFCP_WARP_TILE_K;

    return s;
}

/* ------------------------------------------------------------------ *
 * make_deep_fused_conv_pool_spec(**kwargs)
 * ------------------------------------------------------------------ *
 *
 * Python (gfx950/deep_fused_conv_pool.py lines 55-66):
 *   def make_deep_fused_conv_pool_spec(**kwargs):
 *       base = _make_common_spec(
 *           name=_GFX950_NAME, wave_size=64, warp_tile_m=32, warp_tile_n=32,
 *           **kwargs,
 *       )
 *       return Gfx950DeepFusedConvPoolSpec(
 *           **{f.name: getattr(base, f.name)
 *              for f in fields(DeepFusedConvPoolSpec)}
 *       )
 *
 * So: build the COMMON spec via the common factory with the gfx950 name + wave64 +
 * warp_tile_m/n=32 pinned (warp_tile_k keeps the common factory default 16,
 * matching the MFMA k), then re-wrap by copying every common spec field into the
 * gfx950 spec. The C gfx950 spec embeds the common spec verbatim as `base`, so the
 * field-copy mirror is exactly the assignment `s.base = base` -- byte-identical to
 * the Python dataclass field comprehension.
 *
 * tile_m is auto-derived inside the common factory exactly as in Python; this shim
 * does not touch it. The gfx950-pinned name / wave_size / warp_tile_* are stamped
 * here and ignore caller geometry, matching the Python that hard-codes them. */
rocke_gfx950_deep_fused_conv_pool_spec_t
    rocke_gfx950_deep_fused_conv_pool_make_spec(int n,
                                                int h,
                                                int w,
                                                int c,
                                                int k0,
                                                int k1,
                                                int r,
                                                int s,
                                                int pool_tile_h,
                                                int pool_tile_w,
                                                int tile_n,
                                                int tile_k,
                                                int conv1_tile_k,
                                                int warp_m,
                                                int warp_n,
                                                const char* pipeline,
                                                bool unroll_k,
                                                bool async_dma,
                                                bool cache_input_footprint,
                                                bool direct_conv0_from_input_cache)
{
    rocke_gfx950_deep_fused_conv_pool_spec_t out;
    rocke_deep_fused_conv_pool_spec_t base;

    /* base = _make_common_spec(name=_GFX950_NAME, wave_size=64,
     *                          warp_tile_m=32, warp_tile_n=32, **kwargs)
     *
     * The gfx950-pinned geometry (name / wave_size=64 / warp_tile_m=32 /
     * warp_tile_n=32) is passed to the common factory; the MFMA warp_tile_k is the
     * common factory default (16). All other args are the caller's kwargs forwarded
     * verbatim. */
    base = rocke_make_deep_fused_conv_pool_spec(n,
                                                h,
                                                w,
                                                c,
                                                k0,
                                                k1,
                                                r,
                                                s,
                                                pool_tile_h,
                                                pool_tile_w,
                                                tile_n,
                                                tile_k,
                                                conv1_tile_k,
                                                warp_m,
                                                warp_n,
                                                /* warp_tile_m */ GFX950_DFCP_WARP_TILE_M,
                                                /* warp_tile_n */ GFX950_DFCP_WARP_TILE_N,
                                                /* warp_tile_k */ GFX950_DFCP_WARP_TILE_K,
                                                /* wave_size   */ GFX950_DFCP_WAVE_SIZE,
                                                /* name        */ GFX950_DFCP_NAME,
                                                pipeline,
                                                unroll_k,
                                                async_dma,
                                                cache_input_footprint,
                                                direct_conv0_from_input_cache);

    /* return Gfx950DeepFusedConvPoolSpec(**{f.name: getattr(base, f.name)
     *          for f in fields(DeepFusedConvPoolSpec)})
     *
     * The gfx950 spec embeds the common spec as `base`, so copying every common
     * field == assigning the whole common spec value (the same field set the Python
     * comprehension iterates). */
    memset(&out, 0, sizeof(out));
    out.base = base;

    return out;
}

/* ------------------------------------------------------------------ *
 * Re-exported spec value-type surface (gfx950-named entries over &spec->base)
 * ------------------------------------------------------------------ *
 *
 * Python __all__ re-exports the common is_valid_spec / signature / grid (and the
 * kernel_name is the common spec's, always producing the gfx950 name because the
 * field-copy mirror pins spec.name = _GFX950_NAME). The gfx950 spec is
 * layout-compatible with the common spec via the embedded `base`, so each entry
 * forwards &spec->base to the corresponding common helper. `arch` NULL is
 * normalised to "gfx950" for this shim before forwarding. */

/* is_valid_spec re-export. */
bool rocke_gfx950_deep_fused_conv_pool_is_valid_spec(
    const rocke_gfx950_deep_fused_conv_pool_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap)
{
    if(spec == NULL)
    {
        if(reason != NULL && reason_cap > 0)
        {
            reason[0] = '\0';
        }
        return false;
    }
    if(arch == NULL)
    {
        arch = GFX950_DFCP_ARCH;
    }
    return rocke_deep_fused_conv_pool_is_valid_spec(&spec->base, arch, reason, reason_cap);
}

/* deep_fused_conv_pool_signature re-export (forwards over &spec->base). The gfx950
 * signature is identical to the common one -- same A/B/Y/W1 ptrs + *_bytes scalars
 * -- since the MFMA shim changes only name, not the kernel ABI. */
rocke_status_t rocke_gfx950_deep_fused_conv_pool_signature(
    rocke_arena_t* arena,
    const rocke_gfx950_deep_fused_conv_pool_spec_t* spec,
    const rocke_sig_entry_t** out_items,
    size_t* out_count)
{
    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    return rocke_deep_fused_conv_pool_signature(arena, &spec->base, out_items, out_count);
}

/* deep_fused_conv_pool_grid re-export (forwards over &spec->base). The gfx950 grid
 * is identical to the common one ((1, pool_ho//pool_tile_h,
 * pool_wo//pool_tile_w)). */
rocke_status_t
    rocke_gfx950_deep_fused_conv_pool_grid(const rocke_gfx950_deep_fused_conv_pool_spec_t* spec,
                                           int out[3])
{
    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    return rocke_deep_fused_conv_pool_grid(&spec->base, out);
}

/* kernel_name re-export (always the gfx950 name; forwards over &spec->base). The
 * common kernel_name() reads spec.name, which the field-copy mirror pins to the
 * gfx950 name, so this reproduces the Python re-export verbatim. */
rocke_status_t rocke_gfx950_deep_fused_conv_pool_kernel_name(
    const rocke_gfx950_deep_fused_conv_pool_spec_t* spec, char* out, size_t out_cap)
{
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    return rocke_deep_fused_conv_pool_spec_kernel_name(&spec->base, out, out_cap);
}

/* rocke_gfx950_dfcp_build_ctx_init is defined in the sibling part-file
 * instance_gfx950_deep_fused_conv_pool_gfx950_dfcp_ctx_and_loaders.c (the
 * canonical owner of the ctx init + conv0 A-load closures, per the internal
 * header contract). The public driver below reaches it through the declaration
 * in rocke/instance_gfx950_deep_fused_conv_pool_internal.h. */

/* ===================================================================== *
 * Conv-builder callback trampolines.
 *
 * The conv builder calls the override hooks with its own ABI
 * (rocke_gfx950_dfcp_conv_overrides_t) + an opaque `user` cookie. We thread the
 * shared rocke_gfx950_dfcp_build_ctx_t through `user` and forward to the gfx950 ctx
 * phases (declared in the internal header). The conv builder's opaque
 * extra_context / input_cache_context Values are the Values our setup_* phases
 * return; we round-trip them as void*.
 * ===================================================================== */

static void* rocke_gfx950_dfcp_tramp_extra_params(rocke_ir_builder_t* b, void* user)
{
    rocke_gfx950_dfcp_build_ctx_t* ctx = (rocke_gfx950_dfcp_build_ctx_t*)user;
    (void)b;
    return (void*)rocke_gfx950_dfcp_extra_params(ctx);
}

static rocke_value_t* rocke_gfx950_dfcp_tramp_m_index_fn(rocke_ir_builder_t* b,
                                                         rocke_value_t* row,
                                                         const rocke_warp_grid_t* grid,
                                                         void* user)
{
    rocke_gfx950_dfcp_build_ctx_t* ctx = (rocke_gfx950_dfcp_build_ctx_t*)user;
    (void)b;
    return rocke_gfx950_dfcp_m_index_fn(ctx, row, grid);
}

static void rocke_gfx950_dfcp_tramp_a_mhw_index_fn(rocke_ir_builder_t* b,
                                                   rocke_value_t* row,
                                                   const rocke_warp_grid_t* grid,
                                                   rocke_value_t** out_n,
                                                   rocke_value_t** out_ho,
                                                   rocke_value_t** out_wo,
                                                   void* user)
{
    rocke_gfx950_dfcp_build_ctx_t* ctx = (rocke_gfx950_dfcp_build_ctx_t*)user;
    (void)b;
    rocke_gfx950_dfcp_a_mhw_index_fn(ctx, row, grid, out_n, out_ho, out_wo);
}

static void* rocke_gfx950_dfcp_tramp_input_cache_setup(rocke_ir_builder_t* b,
                                                       const rocke_implicit_gemm_conv_spec_t* spec,
                                                       const rocke_warp_grid_t* grid,
                                                       rocke_value_t* a_rsrc,
                                                       void* user)
{
    rocke_gfx950_dfcp_build_ctx_t* ctx = (rocke_gfx950_dfcp_build_ctx_t*)user;
    (void)b;
    /* select setup_input_cache vs setup_specialized_a_loader by the same gate the
     * driver wired (Python's input_cache_setup= ternary). */
    if(ctx->use_input_cache)
    {
        return (void*)rocke_gfx950_dfcp_setup_input_cache(ctx, spec, grid, a_rsrc);
    }
    return (void*)rocke_gfx950_dfcp_setup_specialized_a_loader(ctx, spec, grid, a_rsrc);
}

static void rocke_gfx950_dfcp_tramp_a_load_override(rocke_ir_builder_t* b,
                                                    const rocke_implicit_gemm_conv_spec_t* spec,
                                                    rocke_value_t* k_off,
                                                    rocke_value_t* a_dst,
                                                    const rocke_warp_grid_t* grid,
                                                    void* input_cache_context,
                                                    void* user)
{
    rocke_gfx950_dfcp_build_ctx_t* ctx = (rocke_gfx950_dfcp_build_ctx_t*)user;
    rocke_value_t* cache = (rocke_value_t*)input_cache_context;
    (void)b;
    if(ctx->use_input_cache)
    {
        rocke_gfx950_dfcp_load_a_tile_from_cache(ctx, spec, k_off, a_dst, grid, cache);
    }
    else
    {
        rocke_gfx950_dfcp_load_a_tile_specialized(ctx, spec, k_off, a_dst, grid, cache);
    }
}

static rocke_value_t*
    rocke_gfx950_dfcp_tramp_a_operand_override(rocke_ir_builder_t* b,
                                               const rocke_implicit_gemm_conv_spec_t* spec,
                                               rocke_value_t* a_row,
                                               rocke_value_t* k_off,
                                               rocke_value_t* col_base,
                                               int a_per_lane,
                                               const rocke_warp_grid_t* grid,
                                               void* input_cache_context,
                                               void* user)
{
    rocke_gfx950_dfcp_build_ctx_t* ctx = (rocke_gfx950_dfcp_build_ctx_t*)user;
    rocke_value_t* cache = (rocke_value_t*)input_cache_context;
    (void)b;
    return rocke_gfx950_dfcp_load_a_operand_from_cache(
        ctx, spec, a_row, k_off, col_base, a_per_lane, grid, cache);
}

static void rocke_gfx950_dfcp_tramp_epilogue_override(rocke_ir_builder_t* b,
                                                      const rocke_implicit_gemm_conv_spec_t* spec,
                                                      rocke_value_t* const* accs,
                                                      int num_accs,
                                                      const rocke_warp_grid_t* grid,
                                                      rocke_value_t* d_rsrc,
                                                      void* extra_context,
                                                      void* user)
{
    rocke_gfx950_dfcp_build_ctx_t* ctx = (rocke_gfx950_dfcp_build_ctx_t*)user;
    rocke_value_t* w1_rsrc = (rocke_value_t*)extra_context;
    (void)b;
    rocke_gfx950_dfcp_epilogue_override(
        ctx, spec, accs, (num_accs < 0) ? 0u : (size_t)num_accs, grid, d_rsrc, w1_rsrc);
}

/* ===================================================================== *
 * rocke_build_gfx950_deep_fused_conv_pool -- the gfx950 public build driver.
 *
 * Mirrors the re-exported common build_deep_fused_conv_pool over the gfx950 spec
 * (arch pinned to "gfx950"):
 *   ok, why = is_valid_spec(spec, "gfx950"); if not ok: raise ValueError(...)
 *   conv_spec = spec.conv_spec()
 *   op = _resolve_conv_op(conv_spec, "gfx950")     # MFMA 32x32x16
 *   <define the nine nested closures>
 *   return build_implicit_gemm_conv(conv_spec, arch="gfx950", **overrides)
 *
 * Thin arch shim: forwards &spec->base + the normalised arch to the wrapped conv
 * driver; the MFMA emitter path is selected inside the shared builder via the
 * MmaOp resolver.
 * ===================================================================== */
rocke_kernel_def_t*
    rocke_build_gfx950_deep_fused_conv_pool(rocke_ir_builder_t* b_unused,
                                            const rocke_gfx950_deep_fused_conv_pool_spec_t* spec,
                                            const char* arch)
{
    rocke_ir_builder_t* b = b_unused; /* the surface this routine emits into */
    const rocke_deep_fused_conv_pool_spec_t* cs;
    char reason[256];
    const rocke_implicit_gemm_conv_spec_t* conv_spec;
    const rocke_mma_op_t* op;
    rocke_gfx950_dfcp_build_ctx_t ctx;
    rocke_gfx950_dfcp_conv_overrides_t ov;
    rocke_status_t st;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = GFX950_DFCP_ARCH; /* arch default "gfx950" */
    }
    cs = &spec->base; /* common spec view the wrapped driver consumes */

    /* 1. is_valid_spec gate -> ValueError on reject. */
    reason[0] = '\0';
    if(!rocke_deep_fused_conv_pool_is_valid_spec(cs, arch, reason, sizeof(reason)))
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "invalid deep fused conv/pool spec for %s: %s", arch, reason);
        return NULL;
    }

    /* 2. conv_spec = spec.conv_spec() */
    conv_spec = rocke_deep_fused_conv_pool_spec_conv_spec(b, cs);
    if(conv_spec == NULL)
    {
        return NULL; /* error routed through b */
    }

    /* 3. op = _resolve_conv_op(conv_spec, "gfx950")  (MFMA 32x32x16) */
    op = rocke_conv_resolve_op(b, conv_spec, arch);
    if(op == NULL)
    {
        return NULL; /* error routed through b */
    }

    /* 4. populate the shared closure ctx (build-time constants + routing). */
    st = rocke_gfx950_dfcp_build_ctx_init(&ctx, b, spec, arch, conv_spec, op);
    if(st != ROCKE_OK)
    {
        rocke_i_set_err(b, st, "deep fused conv/pool: ctx init failed");
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
     * ctx flags; install them only when the corresponding Python hook is not None
     * so the conv builder's None-vs-callable behaviour is byte-faithful. */
    memset(&ov, 0, sizeof(ov));
    ov.user = &ctx;
    ov.extra_params = rocke_gfx950_dfcp_tramp_extra_params;
    ov.m_index_fn = rocke_gfx950_dfcp_tramp_m_index_fn;
    ov.a_mhw_index_fn = rocke_gfx950_dfcp_tramp_a_mhw_index_fn;

    if(ctx.use_input_cache || ctx.use_specialized)
    {
        ov.input_cache_setup = rocke_gfx950_dfcp_tramp_input_cache_setup;
        ov.a_load_override = rocke_gfx950_dfcp_tramp_a_load_override;
    }
    if(ctx.use_operand_ovr)
    {
        ov.a_operand_override = rocke_gfx950_dfcp_tramp_a_operand_override;
    }
    ov.epilogue_override = rocke_gfx950_dfcp_tramp_epilogue_override;

    /* 6. drive the wrapped conv0 builder; return its kernel. The staged `ov`
     * mirror is layout-identical to the canonical rocke_conv_build_overrides_t. */
    return rocke_build_implicit_gemm_conv(
        b, conv_spec, arch, (const rocke_conv_build_overrides_t*)&ov);
}

/* Convenience: init `b` with the gfx950 kernel name, then build. */
rocke_kernel_def_t* rocke_build_gfx950_deep_fused_conv_pool_new(
    rocke_ir_builder_t* b, const rocke_gfx950_deep_fused_conv_pool_spec_t* spec, const char* arch)
{
    /* Catch validity/build raises at this boundary so an invalid spec is
     * reported as a clean NULL+error (matching the sibling instances) rather
     * than escaping to std::terminate. */
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_gfx950_deep_fused_conv_pool_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_gfx950_deep_fused_conv_pool(b, spec, arch);
    });
}

/* ===================================================================== *
 * rocke_gfx950_deep_fused_conv_pool_lower_to_llvm -- build + lower to .ll.
 * Owns and frees its own IRBuilder (mirrors the sibling instance ports).
 * `arch` NULL => "gfx950".
 * ===================================================================== */
rocke_status_t rocke_gfx950_deep_fused_conv_pool_lower_to_llvm(
    const rocke_gfx950_deep_fused_conv_pool_spec_t* spec,
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
        arch = GFX950_DFCP_ARCH;
    }

    kernel = rocke_build_gfx950_deep_fused_conv_pool_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            if(m == NULL)
            {
                m = "build_gfx950_deep_fused_conv_pool failed";
            }
            snprintf(err, err_cap, "%s", m);
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
