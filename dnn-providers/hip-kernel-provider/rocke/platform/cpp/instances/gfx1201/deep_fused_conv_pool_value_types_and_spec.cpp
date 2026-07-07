// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1201_deep_fused_conv_pool_value_types_and_spec.c -- chunked C99
 * port of the gfx1201 (RDNA4, wave32, WMMA 16x16x16) arch shim's spec
 * value-type surface from
 *   rocke/instances/gfx1201/deep_fused_conv_pool.py  (72-LOC thin shim).
 *
 * SCOPE (this part-file): the gfx1201 spec surface that is NOT a pure forward
 * -- the gfx1201-pinned defaults / name construction shared with the public
 * bucket, the Gfx1201DeepFusedConvPoolSpec <-> common spec field-copy mirror,
 * and the gfx1201-specific signature / grid adjustments. Concretely:
 *
 *   - make_deep_fused_conv_pool_spec(**kwargs)  (Python lines 61-72): build the
 *     common spec via _make_common_spec(name=_GFX1201_NAME, wave_size=32,
 *     warp_tile_m=16, warp_tile_n=16, **kwargs), then re-wrap as the gfx1201
 *     spec by copying EVERY common-spec field (the Python
 *     `{f.name: getattr(base, f.name) for f in fields(DeepFusedConvPoolSpec)}`
 *     comprehension). tile_m auto-derive happens inside the common factory,
 *     byte-identical to Python.
 *       -> rocke_gfx1201_deep_fused_conv_pool_make_spec
 *
 *   - Gfx1201DeepFusedConvPoolSpec dataclass defaults (Python lines 45-58): the
 *     common default spec with the gfx1201 geometry/name stamped
 *     (name=_GFX1201_NAME, wave_size=32, warp_tile 16x16x16).
 *       -> rocke_gfx1201_deep_fused_conv_pool_spec_default
 *
 *   - the re-exported spec value-type surface (Python __all__ re-exports of the
 *     common is_valid_spec / signature / grid / kernel_name over the gfx1201
 *     spec): thin wrappers that forward &spec->base to the common helpers, with
 *     `arch` NULL-normalised to "gfx1201" for this shim. These are the gfx1201
 *     spec-named entry points the public bucket exposes; the field-copy mirror
 *     above guarantees &spec->base is the exact common spec the helpers expect.
 *       -> rocke_gfx1201_deep_fused_conv_pool_is_valid_spec
 *       -> rocke_gfx1201_deep_fused_conv_pool_signature
 *       -> rocke_gfx1201_deep_fused_conv_pool_grid
 *       -> rocke_gfx1201_deep_fused_conv_pool_kernel_name
 *
 * NOT in scope (peer part-files): the build / build_new / lower-to-.ll entries
 * (rocke_build_gfx1201_deep_fused_conv_pool*, *_lower_to_llvm), the build-context
 * init (rocke_gfx1201_dfcp_build_ctx_init), and the closure phase functions
 * (rocke_gfx1201_dfcp_*). Those live in the gfx1201 build/closure part-files and
 * are reached only through the internal header.
 *
 * This is PURE COMPUTE over value types: no IR is emitted here. Every numeric
 * decision (tile_m auto-derive, the validity chain, the signature/grid shapes)
 * is delegated to the already-ported family-agnostic common helpers so the
 * gfx1201 surface stays byte-identical to the common one with only the geometry
 * pinning + spec re-wrap added -- exactly mirroring the Python shim.
 */
#include "rocke/instance_gfx1201_deep_fused_conv_pool_internal.h"

#include <string.h> /* memset (defensive zero before stamping) */

/* ------------------------------------------------------------------ *
 * The gfx1201-pinned geometry / name the shim stamps (Python lines 42-58).
 * These mirror the public-header macros (kept local-named for clarity at the
 * stamping sites); they are the RDNA4 WMMA defaults
 * (wave_size=32, warp_tile 16x16x16) plus the gfx1201 kernel name.
 * ------------------------------------------------------------------ */
#define GFX1201_DFCP_NAME ROCKE_GFX1201_DEEP_FUSED_CONV_POOL_NAME
#define GFX1201_DFCP_WAVE_SIZE ROCKE_GFX1201_DEEP_FUSED_CONV_POOL_WAVE_SIZE
#define GFX1201_DFCP_WARP_TILE_M ROCKE_GFX1201_DEEP_FUSED_CONV_POOL_WARP_TILE_M
#define GFX1201_DFCP_WARP_TILE_N ROCKE_GFX1201_DEEP_FUSED_CONV_POOL_WARP_TILE_N
#define GFX1201_DFCP_WARP_TILE_K ROCKE_GFX1201_DEEP_FUSED_CONV_POOL_WARP_TILE_K
#define GFX1201_DFCP_ARCH ROCKE_GFX1201_DEEP_FUSED_CONV_POOL_ARCH

/* ------------------------------------------------------------------ *
 * Gfx1201DeepFusedConvPoolSpec dataclass defaults
 * ------------------------------------------------------------------ *
 *
 * Python (gfx1201/deep_fused_conv_pool.py lines 45-58):
 *   @dataclass(frozen=True)
 *   class Gfx1201DeepFusedConvPoolSpec(DeepFusedConvPoolSpec):
 *       name: str = _GFX1201_NAME
 *       wave_size: int = 32
 *       warp_tile_m: int = 16
 *       warp_tile_n: int = 16
 *       warp_tile_k: int = 16
 *
 * i.e. the common DeepFusedConvPoolSpec defaults with these five fields
 * overridden. The C mirror takes the common default spec and stamps the gfx1201
 * geometry/name onto the embedded `base`; the caller fills base.problem. */
rocke_gfx1201_deep_fused_conv_pool_spec_t rocke_gfx1201_deep_fused_conv_pool_spec_default(void)
{
    rocke_gfx1201_deep_fused_conv_pool_spec_t s;

    /* The common DeepFusedConvPoolSpec dataclass defaults. */
    s.base = rocke_deep_fused_conv_pool_spec_default();

    /* The gfx1201 dataclass field overrides (Python lines 54-58). */
    s.base.name = GFX1201_DFCP_NAME;
    s.base.wave_size = GFX1201_DFCP_WAVE_SIZE;
    s.base.warp_tile_m = GFX1201_DFCP_WARP_TILE_M;
    s.base.warp_tile_n = GFX1201_DFCP_WARP_TILE_N;
    s.base.warp_tile_k = GFX1201_DFCP_WARP_TILE_K;

    return s;
}

/* ------------------------------------------------------------------ *
 * make_deep_fused_conv_pool_spec(**kwargs)
 * ------------------------------------------------------------------ *
 *
 * Python (gfx1201/deep_fused_conv_pool.py lines 61-72):
 *   def make_deep_fused_conv_pool_spec(**kwargs):
 *       base = _make_common_spec(
 *           name=_GFX1201_NAME, wave_size=32, warp_tile_m=16, warp_tile_n=16,
 *           **kwargs,
 *       )
 *       return Gfx1201DeepFusedConvPoolSpec(
 *           **{f.name: getattr(base, f.name)
 *              for f in fields(DeepFusedConvPoolSpec)}
 *       )
 *
 * So: build the COMMON spec via the common factory with the gfx1201 name +
 * wave32 + warp_tile_m/n=16 pinned (warp_tile_k keeps the common factory
 * default; the Python passes only m/n and the common factory's warp_tile_k
 * default is 16, matching the WMMA k), then re-wrap by copying every common
 * spec field into the gfx1201 spec. The C gfx1201 spec embeds the common spec
 * verbatim as `base`, so the field-copy mirror is exactly the assignment
 * `s.base = base` -- byte-identical to the Python dataclass field comprehension
 * (which copies the same field set into the subclass).
 *
 * tile_m is auto-derived inside the common factory exactly as in Python
 * (tile_m = (pool_tile_h*pool_stride_h)*(pool_tile_w*pool_stride_w)); this shim
 * does not touch it.
 *
 * The kwargs the Python forwards map to the common factory's keyword-only
 * surface; this C signature exposes the subset the public bucket's gfx1201
 * make_spec declares (n,h,w,c,k0,k1,r,s, pool_tile_h/w, tile_n,tile_k,
 * conv1_tile_k, warp_m/n, pipeline, unroll_k, async_dma,
 * cache_input_footprint, direct_conv0_from_input_cache). The gfx1201-pinned
 * name / wave_size / warp_tile_* are stamped here and ignore caller geometry,
 * matching the Python that hard-codes them. */
rocke_gfx1201_deep_fused_conv_pool_spec_t
    rocke_gfx1201_deep_fused_conv_pool_make_spec(int n,
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
    rocke_gfx1201_deep_fused_conv_pool_spec_t out;
    rocke_deep_fused_conv_pool_spec_t base;

    /* base = _make_common_spec(name=_GFX1201_NAME, wave_size=32,
     *                          warp_tile_m=16, warp_tile_n=16, **kwargs)
     *
     * The gfx1201-pinned geometry (name / wave_size=32 / warp_tile_m=16 /
     * warp_tile_n=16) is passed positionally to the common factory; the
     * WMMA warp_tile_k is the common factory default (16). All other args are
     * the caller's kwargs forwarded verbatim. */
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
                                                /* warp_tile_m */ GFX1201_DFCP_WARP_TILE_M,
                                                /* warp_tile_n */ GFX1201_DFCP_WARP_TILE_N,
                                                /* warp_tile_k */ GFX1201_DFCP_WARP_TILE_K,
                                                /* wave_size   */ GFX1201_DFCP_WAVE_SIZE,
                                                /* name        */ GFX1201_DFCP_NAME,
                                                pipeline,
                                                unroll_k,
                                                async_dma,
                                                cache_input_footprint,
                                                direct_conv0_from_input_cache);

    /* return Gfx1201DeepFusedConvPoolSpec(**{f.name: getattr(base, f.name)
     *          for f in fields(DeepFusedConvPoolSpec)})
     *
     * The gfx1201 spec embeds the common spec as `base`, so copying every
     * common field == assigning the whole common spec value. (The Python
     * comprehension iterates only the COMMON DeepFusedConvPoolSpec fields, so
     * the subclass's own pinned defaults are re-supplied from `base` -- which
     * already carries them because _make_common_spec stamped them. The C
     * struct-copy is the exact same field set.) */
    memset(&out, 0, sizeof(out));
    out.base = base;

    return out;
}

/* ------------------------------------------------------------------ *
 * Re-exported spec value-type surface (gfx1201-named entries over &spec->base)
 * ------------------------------------------------------------------ *
 *
 * Python __all__ re-exports the common is_valid_spec / signature / grid (and
 * the kernel_name is the common spec's, always producing the gfx1201 name
 * because the field-copy mirror pins spec.name = _GFX1201_NAME). The gfx1201
 * spec is layout-compatible with the common spec via the embedded `base`, so
 * each entry forwards &spec->base to the corresponding common helper. `arch`
 * NULL is normalised to "gfx1201" for this shim before forwarding. */

/* is_valid_spec re-export. */
bool rocke_gfx1201_deep_fused_conv_pool_is_valid_spec(
    const rocke_gfx1201_deep_fused_conv_pool_spec_t* spec,
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
        arch = GFX1201_DFCP_ARCH;
    }
    return rocke_deep_fused_conv_pool_is_valid_spec(&spec->base, arch, reason, reason_cap);
}

/* deep_fused_conv_pool_signature re-export (forwards over &spec->base). The
 * gfx1201 signature is identical to the common one -- same A/B/Y/W1 ptrs +
 * *_bytes scalars -- since the WMMA shim changes only geometry, not the kernel
 * ABI. */
rocke_status_t rocke_gfx1201_deep_fused_conv_pool_signature(
    rocke_arena_t* arena,
    const rocke_gfx1201_deep_fused_conv_pool_spec_t* spec,
    const rocke_sig_entry_t** out_items,
    size_t* out_count)
{
    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    return rocke_deep_fused_conv_pool_signature(arena, &spec->base, out_items, out_count);
}

/* deep_fused_conv_pool_grid re-export (forwards over &spec->base). The gfx1201
 * grid is identical to the common one ((1, pool_ho//pool_tile_h,
 * pool_wo//pool_tile_w)); the wave32 geometry does not alter the per-CTA tile
 * decomposition. */
rocke_status_t
    rocke_gfx1201_deep_fused_conv_pool_grid(const rocke_gfx1201_deep_fused_conv_pool_spec_t* spec,
                                            int out[3])
{
    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    return rocke_deep_fused_conv_pool_grid(&spec->base, out);
}

/* kernel_name re-export (always the gfx1201 name; forwards over &spec->base).
 * The common kernel_name() reads spec.name, which the field-copy mirror pins to
 * the gfx1201 name, so this reproduces the Python re-export verbatim. */
rocke_status_t rocke_gfx1201_deep_fused_conv_pool_kernel_name(
    const rocke_gfx1201_deep_fused_conv_pool_spec_t* spec, char* out, size_t out_cap)
{
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    return rocke_deep_fused_conv_pool_spec_kernel_name(&spec->base, out, out_cap);
}
