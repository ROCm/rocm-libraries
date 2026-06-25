/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_gfx1201_deep_fused_conv_pool.h -- PUBLIC C99 API for the gfx1201
 * (RDNA4, wave32, WMMA 16x16x16) arch shim over the family-agnostic deep-fused
 * conv0 -> conv1 -> maxpool kernel, ported from
 *   ck_dsl/instances/gfx1201/deep_fused_conv_pool.py  (72-LOC thin shim).
 *
 * WHAT THIS SHIM IS.
 *   The numeric kernel body is authored ONCE in the family-agnostic builder
 *   (ck_dsl/instances/common/deep_fused_conv_pool.py -> the C port rooted at
 *   ckc/instance_deep_fused_conv_pool.h) and driven by the resolved MmaOp, so
 *   the same code emits the wave64 MFMA 32x32x16 path (gfx950) and the wave32
 *   WMMA 16x16x16 path (gfx1201) with no per-family branching. The gfx1201
 *   Python module is a 72-LOC wrapper that:
 *     - pins the RDNA4 WMMA geometry (wave_size=32, warp_tile 16x16x16),
 *     - sets the gfx1201 kernel name ("ck_dsl_gfx1201_deep_fused_conv_pool"),
 *     - re-exports the common spec / signature / grid / builder under gfx1201
 *       public names.
 *
 *   This header is the C mirror of that shim. It introduces a gfx1201-pinned
 *   spec type (ckc_gfx1201_deep_fused_conv_pool_spec_t) whose factory stamps the
 *   wave32 WMMA defaults + gfx1201 name, and the gfx1201-named build / lower
 *   entries. All gfx1201 symbols carry the gfx1201_deep_fused_conv_pool prefix
 *   to avoid clashes with the same-named common/ port (which uses the bare
 *   ckc_deep_fused_conv_pool_* / ckc_dfcp_* prefixes).
 *
 *   Python (gfx1201/deep_fused_conv_pool.py)        C99 (this header)
 *   ---------------------------------------------   --------------------------------------
 *   Gfx1201DeepFusedConvPoolSpec (frozen)           ckc_gfx1201_deep_fused_conv_pool_spec_t
 *   make_deep_fused_conv_pool_spec(...)             ckc_gfx1201_deep_fused_conv_pool_make_spec(...)
 *   is_valid_spec (re-export) ckc_gfx1201_deep_fused_conv_pool_is_valid_spec(...)
 *   deep_fused_conv_pool_signature (re-export)      ckc_gfx1201_deep_fused_conv_pool_signature(...)
 *   deep_fused_conv_pool_grid (re-export)           ckc_gfx1201_deep_fused_conv_pool_grid(...)
 *   build_deep_fused_conv_pool (re-export)          ckc_build_gfx1201_deep_fused_conv_pool(...)
 *   (+ convenience: build -> lower .ll) ckc_gfx1201_deep_fused_conv_pool_lower_to_llvm(...)
 *
 * SPEC TYPE NOTE (struct layout = common spec).
 *   ckc_gfx1201_deep_fused_conv_pool_spec_t embeds the common
 *   ckc_deep_fused_conv_pool_spec_t verbatim as its first (and only) member so a
 *   pointer to it is layout-compatible with the common spec: the build / lower
 *   entries forward `&spec->base` to the shared common builder, exactly as the
 *   Python `Gfx1201DeepFusedConvPoolSpec` is a subclass of the common spec the
 *   re-exported `build_deep_fused_conv_pool` consumes. No arch-specific C shim
 *   body is needed: the arch string "gfx1201" selects the WMMA emitter path
 *   inside the shared builder via the MmaOp resolver (_resolve_conv_op), so the
 *   gfx1201 entries are thin name+default wrappers that stamp the geometry then
 *   call the common driver.
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (ckc_b_*); the validity gate returns a bool +
 * reason; the convenience lower returns a ckc_status_t.
 */
#ifndef CKC_INSTANCE_GFX1201_DEEP_FUSED_CONV_POOL_H
#define CKC_INSTANCE_GFX1201_DEEP_FUSED_CONV_POOL_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/arena.h" /* ckc_arena_t */
#include "ckc/helper_ck_dsl.helpers.spec.h" /* ckc_sig_entry_t */
#include "ckc/instance_deep_fused_conv_pool.h" /* common spec/problem/build/lower it wraps */
#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The gfx1201 kernel name (Python `_GFX1201_NAME`). */
#define CKC_GFX1201_DEEP_FUSED_CONV_POOL_NAME "ck_dsl_gfx1201_deep_fused_conv_pool"

/* The pinned RDNA4 WMMA geometry the gfx1201 shim stamps (Python class
 * defaults: wave_size=32, warp_tile 16x16x16). */
#define CKC_GFX1201_DEEP_FUSED_CONV_POOL_WAVE_SIZE 32
#define CKC_GFX1201_DEEP_FUSED_CONV_POOL_WARP_TILE_M 16
#define CKC_GFX1201_DEEP_FUSED_CONV_POOL_WARP_TILE_N 16
#define CKC_GFX1201_DEEP_FUSED_CONV_POOL_WARP_TILE_K 16

/* The arch string this shim targets; the common builder routes to WMMA codegen
 * when handed this. */
#define CKC_GFX1201_DEEP_FUSED_CONV_POOL_ARCH "gfx1201"

/* ------------------------------------------------------------------ *
 * Gfx1201DeepFusedConvPoolSpec
 * ------------------------------------------------------------------ *
 *
 * The common spec pinned to the WMMA geometry. Identical fields to the common
 * ckc_deep_fused_conv_pool_spec_t (embedded as `base`); the gfx1201 defaults are
 * wave_size=32, warp_tile 16x16x16, and the gfx1201 kernel name. The struct is
 * a single-member wrapper so `&spec->base` is the common spec the shared builder
 * consumes (mirroring the Python subclass relationship). */
typedef struct ckc_gfx1201_deep_fused_conv_pool_spec
{
    ckc_deep_fused_conv_pool_spec_t base;
} ckc_gfx1201_deep_fused_conv_pool_spec_t;

/* make_deep_fused_conv_pool_spec(**kwargs) -> Gfx1201DeepFusedConvPoolSpec.
 *
 * Mirrors the Python factory: builds the common spec via
 * ckc_make_deep_fused_conv_pool_spec(...) with name pinned to the gfx1201 name,
 * wave_size=32, warp_tile_m=warp_tile_n=16 (and the WMMA warp_tile_k=16), then
 * wraps it as the gfx1201 spec. The argument list mirrors the common factory's
 * keyword-only surface; pass the common defaults for any field the caller does
 * not override (the gfx1201-pinned wave_size / warp_tile_* are stamped here and
 * ignore caller values, matching the Python that hard-codes them).
 *
 * tile_m is auto-derived inside the common factory exactly as in Python. */
ckc_gfx1201_deep_fused_conv_pool_spec_t
    ckc_gfx1201_deep_fused_conv_pool_make_spec(int n,
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

/* Default-constructed gfx1201 spec: the common default spec with the gfx1201
 * geometry/name stamped. The caller fills `base.problem`. */
ckc_gfx1201_deep_fused_conv_pool_spec_t ckc_gfx1201_deep_fused_conv_pool_spec_default(void);

/* ------------------------------------------------------------------ *
 * Re-exported spec surface (thin forwards to the common helpers over &spec->base)
 * ------------------------------------------------------------------ */

/* is_valid_spec re-export. `arch` NULL => "gfx1201" for this shim. Forwards to
 * ckc_deep_fused_conv_pool_is_valid_spec(&spec->base, arch, ...). */
bool ckc_gfx1201_deep_fused_conv_pool_is_valid_spec(
    const ckc_gfx1201_deep_fused_conv_pool_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap);

/* deep_fused_conv_pool_signature re-export (forwards over &spec->base). */
ckc_status_t
    ckc_gfx1201_deep_fused_conv_pool_signature(ckc_arena_t* arena,
                                               const ckc_gfx1201_deep_fused_conv_pool_spec_t* spec,
                                               const ckc_sig_entry_t** out_items,
                                               size_t* out_count);

/* deep_fused_conv_pool_grid re-export (forwards over &spec->base). */
ckc_status_t
    ckc_gfx1201_deep_fused_conv_pool_grid(const ckc_gfx1201_deep_fused_conv_pool_spec_t* spec,
                                          int out[3]);

/* kernel_name re-export (always the gfx1201 name; forwards over &spec->base). */
ckc_status_t ckc_gfx1201_deep_fused_conv_pool_kernel_name(
    const ckc_gfx1201_deep_fused_conv_pool_spec_t* spec, char* out, size_t out_cap);

/* ------------------------------------------------------------------ *
 * Public build entry -- build_deep_fused_conv_pool(spec, arch="gfx1201")
 * ------------------------------------------------------------------ *
 *
 * Builds the one-CTA conv0 -> conv1 -> maxpool fused kernel for the gfx1201
 * WMMA target into the supplied builder `b` and returns the kernel (b->kernel)
 * on success or NULL with b's sticky error set. `arch` NULL => "gfx1201".
 *
 * This is a thin arch shim: it forwards &spec->base and the normalised arch to
 * the family-agnostic common driver ckc_build_deep_fused_conv_pool(...), which
 * resolves the WMMA MmaOp (wave32, 16x16x16) via _resolve_conv_op when arch ==
 * "gfx1201" and emits the WMMA codegen path (the MFMA-32x32 intra-lane maxpool
 * fast path is geometry-gated off; WMMA takes either its own register-resident
 * fast path or the layout-agnostic cshuffle-LDS gather + maxpool). The caller
 * owns `b` and frees it with ckc_ir_builder_free(). `b_unused` is named to flag
 * that, like the Python, the caller-supplied builder is the surface this routine
 * emits into via the wrapped conv driver; it is NOT re-initialised here. */
ckc_kernel_def_t*
    ckc_build_gfx1201_deep_fused_conv_pool(ckc_ir_builder_t* b_unused,
                                           const ckc_gfx1201_deep_fused_conv_pool_spec_t* spec,
                                           const char* arch);

/* Convenience: init `b` with the gfx1201 kernel name, then build. The caller
 * owns `b` and frees it with ckc_ir_builder_free(). Returns the kernel or NULL
 * with b's sticky error set. Mirrors the *_new convenience the other instance
 * ports expose. */
ckc_kernel_def_t* ckc_build_gfx1201_deep_fused_conv_pool_new(
    ckc_ir_builder_t* b, const ckc_gfx1201_deep_fused_conv_pool_spec_t* spec, const char* arch);

/* ------------------------------------------------------------------ *
 * Convenience: build -> lower to LLVM .ll text
 * ------------------------------------------------------------------ *
 *
 * Given a gfx1201 spec, init a builder with the gfx1201 kernel name, build, and
 * lower to LLVM .ll text. `arch` NULL => "gfx1201". On CKC_OK *out_ll receives a
 * malloc'd NUL-terminated string the caller frees with free(); on failure it is
 * left NULL and (if err != NULL, capacity err_cap) a diagnostic is written.
 * Internally owns and frees its IRBuilder. Forwards to the common lower over
 * &spec->base. Call with arch="gfx1201" to target RDNA4 WMMA (wave32, 16x16x16). */
ckc_status_t ckc_gfx1201_deep_fused_conv_pool_lower_to_llvm(
    const ckc_gfx1201_deep_fused_conv_pool_spec_t* spec,
    const char* arch,
    ckc_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_GFX1201_DEEP_FUSED_CONV_POOL_H */
