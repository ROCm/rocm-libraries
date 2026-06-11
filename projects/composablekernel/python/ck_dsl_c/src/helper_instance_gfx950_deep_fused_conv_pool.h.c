/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_instance_gfx950_deep_fused_conv_pool.h.c -- C99 port of the gfx950
 * (CDNA, wave64, MFMA 32x32x16) arch shim from
 *   ck_dsl/instances/gfx950/deep_fused_conv_pool.py  (66-LOC thin shim).
 *
 * SCOPE. The whole gfx950 shim surface, since (unlike gfx1201) the gfx950 Python
 * module is a PURE thin re-export: its build_deep_fused_conv_pool IS the common
 * builder re-exported verbatim, so there are no gfx950-specific closure phase
 * bodies to port. Concretely this file implements:
 *
 *   - make_deep_fused_conv_pool_spec(**kwargs)  (Python lines 55-66): build the
 *     common spec via _make_common_spec(name=_GFX950_NAME, wave_size=64,
 *     warp_tile_m=32, warp_tile_n=32, **kwargs), then re-wrap as the gfx950
 *     spec by copying EVERY common-spec field (the Python
 *     `{f.name: getattr(base, f.name) for f in fields(DeepFusedConvPoolSpec)}`
 *     comprehension). tile_m auto-derive happens inside the common factory,
 *     byte-identical to Python.
 *       -> ckc_gfx950_deep_fused_conv_pool_make_spec
 *
 *   - Gfx950DeepFusedConvPoolSpec dataclass defaults (Python lines 41-52): the
 *     common default spec with the gfx950 name stamped (the wave64 32x32x16
 *     geometry is already the common default, so name is the only override).
 *       -> ckc_gfx950_deep_fused_conv_pool_spec_default
 *
 *   - the re-exported spec value-type surface (Python __all__ re-exports of the
 *     common is_valid_spec / signature / grid / kernel_name over the gfx950
 *     spec): thin wrappers that forward &spec->base to the common helpers, with
 *     `arch` NULL-normalised to "gfx950" for this shim.
 *       -> ckc_gfx950_deep_fused_conv_pool_is_valid_spec
 *       -> ckc_gfx950_deep_fused_conv_pool_signature
 *       -> ckc_gfx950_deep_fused_conv_pool_grid
 *       -> ckc_gfx950_deep_fused_conv_pool_kernel_name
 *
 *   - the re-exported build / build_new / lower-to-.ll entries. The gfx950 build
 *     forwards &spec->base + normalised arch straight to the common driver
 *     ckc_build_deep_fused_conv_pool (the MFMA emitter path is selected inside
 *     the shared builder via the MmaOp resolver when arch == "gfx950"); there is
 *     NO gfx950 closure body (the long emitter bits live in the common port).
 *       -> ckc_build_gfx950_deep_fused_conv_pool
 *       -> ckc_build_gfx950_deep_fused_conv_pool_new
 *       -> ckc_gfx950_deep_fused_conv_pool_lower_to_llvm
 *
 * Every numeric decision (tile_m auto-derive, validity chain, signature/grid
 * shapes) and the whole kernel body is delegated to the already-ported
 * family-agnostic common helpers, so the gfx950 surface stays byte-identical to
 * the common one with only the name pinning + spec re-wrap added -- exactly
 * mirroring the Python shim.
 */
#include "ckc/helper_instance_gfx950_deep_fused_conv_pool.h.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err */

#include <stdio.h>  /* snprintf */
#include <string.h> /* memset */

/* ------------------------------------------------------------------ *
 * The gfx950-pinned geometry / name the shim stamps (Python lines 38-66).
 * These mirror the public-header macros (kept local-named for clarity at the
 * stamping sites); they are the CDNA MFMA defaults (wave_size=64, warp_tile
 * 32x32x16) -- which are ALSO the common defaults -- plus the gfx950 kernel name.
 * ------------------------------------------------------------------ */
#define GFX950_DFCP_NAME CKC_GFX950_DEEP_FUSED_CONV_POOL_NAME
#define GFX950_DFCP_WAVE_SIZE CKC_GFX950_DEEP_FUSED_CONV_POOL_WAVE_SIZE
#define GFX950_DFCP_WARP_TILE_M CKC_GFX950_DEEP_FUSED_CONV_POOL_WARP_TILE_M
#define GFX950_DFCP_WARP_TILE_N CKC_GFX950_DEEP_FUSED_CONV_POOL_WARP_TILE_N
#define GFX950_DFCP_WARP_TILE_K CKC_GFX950_DEEP_FUSED_CONV_POOL_WARP_TILE_K
#define GFX950_DFCP_ARCH CKC_GFX950_DEEP_FUSED_CONV_POOL_ARCH

/* ------------------------------------------------------------------ *
 * Gfx950DeepFusedConvPoolSpec dataclass defaults
 * ------------------------------------------------------------------ *
 *
 * Python (gfx950/deep_fused_conv_pool.py lines 41-52):
 *   @dataclass(frozen=True)
 *   class Gfx950DeepFusedConvPoolSpec(DeepFusedConvPoolSpec):
 *       name: str = _GFX950_NAME
 *
 * i.e. the common DeepFusedConvPoolSpec defaults with only `name` overridden
 * (the wave64 32x32x16 geometry is already the common default). The C mirror
 * takes the common default spec and stamps the gfx950 name onto the embedded
 * `base` (it also restamps the geometry, which is a no-op since it equals the
 * common default, to make the gfx950 pinning explicit); the caller fills
 * base.problem. */
ckc_gfx950_deep_fused_conv_pool_spec_t ckc_gfx950_deep_fused_conv_pool_spec_default(void)
{
    ckc_gfx950_deep_fused_conv_pool_spec_t s;

    /* The common DeepFusedConvPoolSpec dataclass defaults. */
    s.base = ckc_deep_fused_conv_pool_spec_default();

    /* The gfx950 dataclass field override (Python line 52): name only. The
     * geometry restamps below are no-ops vs the common default but make the
     * gfx950 pinning explicit / robust to future common-default drift. */
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
 * So: build the COMMON spec via the common factory with the gfx950 name +
 * wave64 + warp_tile_m/n=32 pinned (warp_tile_k keeps the common factory
 * default 16, matching the MFMA k), then re-wrap by copying every common spec
 * field into the gfx950 spec. The C gfx950 spec embeds the common spec verbatim
 * as `base`, so the field-copy mirror is exactly the assignment `s.base = base`
 * -- byte-identical to the Python dataclass field comprehension.
 *
 * tile_m is auto-derived inside the common factory exactly as in Python; this
 * shim does not touch it. The gfx950-pinned name / wave_size / warp_tile_* are
 * stamped here and ignore caller geometry, matching the Python that hard-codes
 * them. */
ckc_gfx950_deep_fused_conv_pool_spec_t
ckc_gfx950_deep_fused_conv_pool_make_spec(int n,
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
    ckc_gfx950_deep_fused_conv_pool_spec_t out;
    ckc_deep_fused_conv_pool_spec_t base;

    /* base = _make_common_spec(name=_GFX950_NAME, wave_size=64,
     *                          warp_tile_m=32, warp_tile_n=32, **kwargs)
     *
     * The gfx950-pinned geometry (name / wave_size=64 / warp_tile_m=32 /
     * warp_tile_n=32) is passed to the common factory; the MFMA warp_tile_k is
     * the common factory default (16). All other args are the caller's kwargs
     * forwarded verbatim. */
    base = ckc_make_deep_fused_conv_pool_spec(
        n, h, w, c, k0, k1, r, s,
        pool_tile_h, pool_tile_w,
        tile_n, tile_k, conv1_tile_k,
        warp_m, warp_n,
        /* warp_tile_m */ GFX950_DFCP_WARP_TILE_M,
        /* warp_tile_n */ GFX950_DFCP_WARP_TILE_N,
        /* warp_tile_k */ GFX950_DFCP_WARP_TILE_K,
        /* wave_size   */ GFX950_DFCP_WAVE_SIZE,
        /* name        */ GFX950_DFCP_NAME,
        pipeline, unroll_k, async_dma,
        cache_input_footprint, direct_conv0_from_input_cache);

    /* return Gfx950DeepFusedConvPoolSpec(**{f.name: getattr(base, f.name)
     *          for f in fields(DeepFusedConvPoolSpec)})
     *
     * The gfx950 spec embeds the common spec as `base`, so copying every common
     * field == assigning the whole common spec value (the same field set the
     * Python comprehension iterates). */
    memset(&out, 0, sizeof(out));
    out.base = base;

    return out;
}

/* ------------------------------------------------------------------ *
 * Re-exported spec value-type surface (gfx950-named entries over &spec->base)
 * ------------------------------------------------------------------ *
 *
 * Python __all__ re-exports the common is_valid_spec / signature / grid (and
 * the kernel_name is the common spec's, always producing the gfx950 name because
 * the field-copy mirror pins spec.name = _GFX950_NAME). The gfx950 spec is
 * layout-compatible with the common spec via the embedded `base`, so each entry
 * forwards &spec->base to the corresponding common helper. `arch` NULL is
 * normalised to "gfx950" for this shim before forwarding. */

/* is_valid_spec re-export. */
bool ckc_gfx950_deep_fused_conv_pool_is_valid_spec(
    const ckc_gfx950_deep_fused_conv_pool_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap)
{
    if (spec == NULL)
    {
        if (reason != NULL && reason_cap > 0)
        {
            reason[0] = '\0';
        }
        return false;
    }
    if (arch == NULL)
    {
        arch = GFX950_DFCP_ARCH;
    }
    return ckc_deep_fused_conv_pool_is_valid_spec(&spec->base, arch, reason,
                                                  reason_cap);
}

/* deep_fused_conv_pool_signature re-export (forwards over &spec->base). The
 * gfx950 signature is identical to the common one -- same A/B/Y/W1 ptrs +
 * *_bytes scalars -- since the MFMA shim changes only name, not the kernel
 * ABI. */
ckc_status_t
ckc_gfx950_deep_fused_conv_pool_signature(ckc_arena_t* arena,
                                          const ckc_gfx950_deep_fused_conv_pool_spec_t* spec,
                                          const ckc_sig_entry_t** out_items,
                                          size_t* out_count)
{
    if (spec == NULL)
    {
        return CKC_ERR_VALUE;
    }
    return ckc_deep_fused_conv_pool_signature(arena, &spec->base, out_items,
                                              out_count);
}

/* deep_fused_conv_pool_grid re-export (forwards over &spec->base). The gfx950
 * grid is identical to the common one ((1, pool_ho//pool_tile_h,
 * pool_wo//pool_tile_w)). */
ckc_status_t
ckc_gfx950_deep_fused_conv_pool_grid(const ckc_gfx950_deep_fused_conv_pool_spec_t* spec,
                                     int out[3])
{
    if (spec == NULL)
    {
        return CKC_ERR_VALUE;
    }
    return ckc_deep_fused_conv_pool_grid(&spec->base, out);
}

/* kernel_name re-export (always the gfx950 name; forwards over &spec->base).
 * The common kernel_name() reads spec.name, which the field-copy mirror pins to
 * the gfx950 name, so this reproduces the Python re-export verbatim. */
ckc_status_t
ckc_gfx950_deep_fused_conv_pool_kernel_name(const ckc_gfx950_deep_fused_conv_pool_spec_t* spec,
                                            char* out,
                                            size_t out_cap)
{
    if (spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    return ckc_deep_fused_conv_pool_spec_kernel_name(&spec->base, out, out_cap);
}

/* ===================================================================== *
 * ckc_build_gfx950_deep_fused_conv_pool -- the gfx950 public build driver.
 *
 * Mirrors the re-exported common build_deep_fused_conv_pool over the gfx950 spec
 * (arch pinned to "gfx950"). Because the gfx950 Python re-exports the COMMON
 * build verbatim (no gfx950 closure body), this is a one-line forward of
 * &spec->base + the normalised arch to ckc_build_deep_fused_conv_pool; the MFMA
 * emitter path is selected inside the shared builder via the MmaOp resolver.
 * ===================================================================== */
ckc_kernel_def_t*
ckc_build_gfx950_deep_fused_conv_pool(ckc_ir_builder_t* b_unused,
                                      const ckc_gfx950_deep_fused_conv_pool_spec_t* spec,
                                      const char* arch)
{
    ckc_ir_builder_t* b = b_unused; /* the surface this routine emits into */

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (arch == NULL)
    {
        arch = GFX950_DFCP_ARCH; /* arch default "gfx950" */
    }
    /* Forward &spec->base (the common spec view) to the shared driver; all
     * validity / conv-spec / MmaOp-resolve / closure wiring lives there. */
    return ckc_build_deep_fused_conv_pool(b, &spec->base, arch);
}

/* Convenience: init `b` with the gfx950 kernel name, then build. */
ckc_kernel_def_t*
ckc_build_gfx950_deep_fused_conv_pool_new(ckc_ir_builder_t* b,
                                          const ckc_gfx950_deep_fused_conv_pool_spec_t* spec,
                                          const char* arch)
{
    char name[256];

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (ckc_gfx950_deep_fused_conv_pool_kernel_name(spec, name, sizeof(name)) !=
        CKC_OK)
    {
        return NULL;
    }
    if (ckc_ir_builder_init(b, name) != CKC_OK)
    {
        return NULL;
    }
    return ckc_build_gfx950_deep_fused_conv_pool(b, spec, arch);
}

/* ===================================================================== *
 * ckc_gfx950_deep_fused_conv_pool_lower_to_llvm -- build + lower to .ll.
 * Owns and frees its own IRBuilder (mirrors the sibling instance ports).
 * `arch` NULL => "gfx950".
 * ===================================================================== */
ckc_status_t ckc_gfx950_deep_fused_conv_pool_lower_to_llvm(
    const ckc_gfx950_deep_fused_conv_pool_spec_t* spec,
    const char* arch,
    ckc_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if (out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if (spec == NULL || out_ll == NULL)
    {
        if (err != NULL && err_cap > 0)
        {
            snprintf(err, err_cap, "lower_to_llvm: null spec/out");
        }
        return CKC_ERR_VALUE;
    }
    if (arch == NULL)
    {
        arch = GFX950_DFCP_ARCH;
    }

    kernel = ckc_build_gfx950_deep_fused_conv_pool_new(&b, spec, arch);
    if (kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        if (err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(&b);
            if (m == NULL)
            {
                m = "build_gfx950_deep_fused_conv_pool failed";
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
