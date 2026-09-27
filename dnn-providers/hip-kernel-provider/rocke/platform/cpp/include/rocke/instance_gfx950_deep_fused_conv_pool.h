/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx950_deep_fused_conv_pool.h -- PUBLIC C99 API for the gfx950
 * (CDNA, wave64, MFMA 32x32x16) arch shim over the family-agnostic deep-fused
 * conv0 -> conv1 -> maxpool kernel, ported from
 *   rocke/instances/gfx950/deep_fused_conv_pool.py  (67-LOC thin shim).
 *
 * WHAT THIS SHIM IS.
 *   The numeric kernel body is authored ONCE in the family-agnostic builder
 *   (rocke/instances/common/deep_fused_conv_pool.py -> the C port rooted at
 *   rocke/instance_deep_fused_conv_pool.h) and driven by the resolved MmaOp, so
 *   the same code emits the wave64 MFMA 32x32x16 path (gfx950) and the wave32
 *   WMMA 16x16x16 path (gfx1201) with no per-family branching. The gfx950
 *   Python module is a 67-LOC wrapper that:
 *     - pins the CDNA MFMA geometry (wave_size=64, warp_tile 32x32x16) -- which
 *       is ALSO the common default, so the gfx950 shim differs from common in
 *       name only,
 *     - sets the gfx950 kernel name ("rocke_gfx950_deep_fused_conv_pool"),
 *     - re-exports the common spec / signature / grid / builder under gfx950
 *       public names so existing harnesses / sweeps / manifests keep working
 *       byte-for-byte.
 *
 *   This header is the C mirror of that shim. It introduces a gfx950-pinned spec
 *   type (rocke_gfx950_deep_fused_conv_pool_spec_t) whose factory stamps the
 *   wave64 MFMA defaults + gfx950 name, and the gfx950-named build / lower
 *   entries. All gfx950 symbols carry the gfx950_deep_fused_conv_pool prefix to
 *   avoid clashes with the same-named common/ port (which uses the bare
 *   rocke_deep_fused_conv_pool_* / rocke_dfcp_* prefixes).
 *
 *   Python (gfx950/deep_fused_conv_pool.py)         C99 (this header)
 *   ---------------------------------------------   --------------------------------------
 *   Gfx950DeepFusedConvPoolSpec (frozen)            rocke_gfx950_deep_fused_conv_pool_spec_t
 *   make_deep_fused_conv_pool_spec(...)             rocke_gfx950_deep_fused_conv_pool_make_spec(...)
 *   (default-constructed gfx950 spec) rocke_gfx950_deep_fused_conv_pool_spec_default(...)
 *   is_valid_spec (re-export) rocke_gfx950_deep_fused_conv_pool_is_valid_spec(...)
 *   deep_fused_conv_pool_signature (re-export)      rocke_gfx950_deep_fused_conv_pool_signature(...)
 *   deep_fused_conv_pool_grid (re-export)           rocke_gfx950_deep_fused_conv_pool_grid(...)
 *   (kernel_name re-export) rocke_gfx950_deep_fused_conv_pool_kernel_name(...)
 *   build_deep_fused_conv_pool (re-export)          rocke_build_gfx950_deep_fused_conv_pool(...)
 *   (+ convenience: init+build)                     rocke_build_gfx950_deep_fused_conv_pool_new(...)
 *   (+ convenience: build -> lower .ll) rocke_gfx950_deep_fused_conv_pool_lower_to_llvm(...)
 *
 * SPEC TYPE NOTE (struct layout = common spec).
 *   rocke_gfx950_deep_fused_conv_pool_spec_t embeds the common
 *   rocke_deep_fused_conv_pool_spec_t verbatim as its first (and only) member so a
 *   pointer to it is layout-compatible with the common spec: the build / lower
 *   entries forward `&spec->base` to the shared common builder, exactly as the
 *   Python `Gfx950DeepFusedConvPoolSpec` is a subclass of the common spec the
 *   re-exported `build_deep_fused_conv_pool` consumes.
 *
 *   The arch string "gfx950" selects the MFMA emitter path inside the shared
 *   builder via the MmaOp resolver (_resolve_conv_op), so the gfx950 entries are
 *   thin name+default wrappers that stamp the geometry then drive the common
 *   conv0 builder. The closure phases the driver wires are realised over a shared
 *   build-context in rocke/instance_gfx950_deep_fused_conv_pool_internal.h
 *   (rocke_gfx950_dfcp_*); they are byte-faithful to the common closures driven by
 *   the MFMA-resolved op. KEY GFX950 DIFFERENCE FROM GFX1201: the MFMA-32x32
 *   intra-lane register-resident maxpool fast path is ACTIVE (gated on by
 *   rocke_dfcp_maxpool_is_intra_lane); the WMMA register-resident path and the WMMA
 *   operand layout are INACTIVE.
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool +
 * reason; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_GFX950_DEEP_FUSED_CONV_POOL_H
#define ROCKE_INSTANCE_GFX950_DEEP_FUSED_CONV_POOL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/arena.h" /* rocke_arena_t */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/helper_rocke.instances.common.deep_fused_conv_pool.h" /* make_spec/default/is_valid/sig/grid/kernel_name */
#include "rocke/instance_deep_fused_conv_pool.h" /* common spec/problem/build/lower it wraps */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The gfx950 kernel name (Python `_GFX950_NAME`). */
#define ROCKE_GFX950_DEEP_FUSED_CONV_POOL_NAME "rocke_gfx950_deep_fused_conv_pool"

/* The pinned CDNA MFMA geometry the gfx950 shim stamps (Python factory:
 * wave_size=64, warp_tile_m=32, warp_tile_n=32; warp_tile_k = common default 16).
 * These are ALSO the common DeepFusedConvPoolSpec defaults, so the gfx950 shim
 * differs from the common spec in `name` only. */
#define ROCKE_GFX950_DEEP_FUSED_CONV_POOL_WAVE_SIZE 64
#define ROCKE_GFX950_DEEP_FUSED_CONV_POOL_WARP_TILE_M 32
#define ROCKE_GFX950_DEEP_FUSED_CONV_POOL_WARP_TILE_N 32
#define ROCKE_GFX950_DEEP_FUSED_CONV_POOL_WARP_TILE_K 16

/* The arch string this shim targets; the common builder routes to MFMA codegen
 * when handed this. */
#define ROCKE_GFX950_DEEP_FUSED_CONV_POOL_ARCH "gfx950"

/* ------------------------------------------------------------------ *
 * Gfx950DeepFusedConvPoolSpec
 * ------------------------------------------------------------------ *
 *
 * The common spec pinned to the MFMA geometry + gfx950 kernel name. Identical
 * fields to the common rocke_deep_fused_conv_pool_spec_t (embedded as `base`); the
 * gfx950 default that differs from common is the kernel name (the wave64
 * 32x32x16 geometry is already the common default). The struct is a single-member
 * wrapper so `&spec->base` is the common spec the shared builder consumes
 * (mirroring the Python subclass relationship). */
typedef struct rocke_gfx950_deep_fused_conv_pool_spec
{
    rocke_deep_fused_conv_pool_spec_t base;
} rocke_gfx950_deep_fused_conv_pool_spec_t;

/* make_deep_fused_conv_pool_spec(**kwargs) -> Gfx950DeepFusedConvPoolSpec.
 *
 * Mirrors the Python factory: builds the common spec via
 * rocke_make_deep_fused_conv_pool_spec(...) with name pinned to the gfx950 name,
 * wave_size=64, warp_tile_m=warp_tile_n=32 (and the MFMA warp_tile_k=16), then
 * wraps it as the gfx950 spec. The argument list mirrors the common factory's
 * keyword-only surface; pass the common defaults for any field the caller does
 * not override (the gfx950-pinned wave_size / warp_tile_* are stamped here and
 * ignore caller values, matching the Python that hard-codes them).
 *
 * tile_m is auto-derived inside the common factory exactly as in Python. */
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
                                                bool direct_conv0_from_input_cache);

/* Default-constructed gfx950 spec: the common default spec with the gfx950
 * name stamped (the geometry already matches common). The caller fills
 * `base.problem`. */
rocke_gfx950_deep_fused_conv_pool_spec_t rocke_gfx950_deep_fused_conv_pool_spec_default(void);

/* ------------------------------------------------------------------ *
 * Re-exported spec surface (thin forwards to the common helpers over &spec->base)
 * ------------------------------------------------------------------ */

/* is_valid_spec re-export. `arch` NULL => "gfx950" for this shim. Forwards to
 * rocke_deep_fused_conv_pool_is_valid_spec(&spec->base, arch, ...). */
bool rocke_gfx950_deep_fused_conv_pool_is_valid_spec(
    const rocke_gfx950_deep_fused_conv_pool_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap);

/* deep_fused_conv_pool_signature re-export (forwards over &spec->base). */
rocke_status_t rocke_gfx950_deep_fused_conv_pool_signature(
    rocke_arena_t* arena,
    const rocke_gfx950_deep_fused_conv_pool_spec_t* spec,
    const rocke_sig_entry_t** out_items,
    size_t* out_count);

/* deep_fused_conv_pool_grid re-export (forwards over &spec->base). */
rocke_status_t
    rocke_gfx950_deep_fused_conv_pool_grid(const rocke_gfx950_deep_fused_conv_pool_spec_t* spec,
                                           int out[3]);

/* kernel_name re-export (always the gfx950 name; forwards over &spec->base). */
rocke_status_t rocke_gfx950_deep_fused_conv_pool_kernel_name(
    const rocke_gfx950_deep_fused_conv_pool_spec_t* spec, char* out, size_t out_cap);

/* ------------------------------------------------------------------ *
 * Public build entry -- build_deep_fused_conv_pool(spec, arch="gfx950")
 * ------------------------------------------------------------------ *
 *
 * Builds the one-CTA conv0 -> conv1 -> maxpool fused kernel for the gfx950 MFMA
 * target into the supplied builder `b` and returns the kernel (b->kernel) on
 * success or NULL with b's sticky error set. `arch` NULL => "gfx950".
 *
 * This is a thin arch shim: it stamps the gfx950 geometry/name onto the spec,
 * resolves the MFMA MmaOp (wave64, 32x32x16) via _resolve_conv_op, populates the
 * shared build-context (rocke_gfx950_dfcp_build_ctx_t), then drives the wrapped
 * family-agnostic conv0 builder rocke_build_implicit_gemm_conv(...) with the nine
 * closure phases (extra_params, m_index_fn, a_mhw_index_fn, setup_input_cache,
 * setup_specialized_a_loader, load_a_tile_from_cache, load_a_tile_specialized,
 * load_a_operand_from_cache, epilogue_override) in byte-faithful Python order.
 * The MFMA-32x32 intra-lane maxpool fast path is geometry-gated ON for gfx950;
 * the WMMA register-resident path is inactive. The caller owns `b` and frees it
 * with rocke_ir_builder_free(). `b_unused` is named to flag that, like the Python,
 * the caller-supplied builder is the surface emitted into via the wrapped conv
 * driver; it is NOT re-initialised here. */
rocke_kernel_def_t*
    rocke_build_gfx950_deep_fused_conv_pool(rocke_ir_builder_t* b_unused,
                                            const rocke_gfx950_deep_fused_conv_pool_spec_t* spec,
                                            const char* arch);

/* Convenience: init `b` with the gfx950 kernel name, then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL with
 * b's sticky error set. Mirrors the *_new convenience the other instance ports
 * expose. */
rocke_kernel_def_t* rocke_build_gfx950_deep_fused_conv_pool_new(
    rocke_ir_builder_t* b, const rocke_gfx950_deep_fused_conv_pool_spec_t* spec, const char* arch);

/* ------------------------------------------------------------------ *
 * Convenience: build -> lower to LLVM .ll text
 * ------------------------------------------------------------------ *
 *
 * Given a gfx950 spec, init a builder with the gfx950 kernel name, build, and
 * lower to LLVM .ll text. `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a
 * malloc'd NUL-terminated string the caller frees with free(); on failure it is
 * left NULL and (if err != NULL, capacity err_cap) a diagnostic is written.
 * Internally owns and frees its IRBuilder. Forwards to the common lower over
 * &spec->base. Call with arch="gfx950" to target CDNA MFMA (wave64, 32x32x16). */
rocke_status_t rocke_gfx950_deep_fused_conv_pool_lower_to_llvm(
    const rocke_gfx950_deep_fused_conv_pool_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX950_DEEP_FUSED_CONV_POOL_H */
