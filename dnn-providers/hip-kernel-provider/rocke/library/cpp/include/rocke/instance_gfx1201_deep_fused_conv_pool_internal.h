/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1201_deep_fused_conv_pool_internal.h -- PRIVATE shared state +
 * phase / closure-function contract for the C99 port of the gfx1201 (RDNA4,
 * wave32, WMMA 16x16x16) arch shim over build_deep_fused_conv_pool
 *   (rocke/instances/gfx1201/deep_fused_conv_pool.py  ->
 *    re-exported common rocke/instances/common/deep_fused_conv_pool.py
 *    build_deep_fused_conv_pool, Python lines 1212-1401).
 *
 * WHY THIS HEADER EXISTS.
 *   The gfx1201 module is a thin shim that pins WMMA geometry and re-exports the
 *   common builder. The common build_deep_fused_conv_pool() is a driver whose
 *   body is a STACK OF NESTED CLOSURES handed to the wrapped conv0 builder
 *   build_implicit_gemm_conv():
 *
 *     extra_params, m_index_fn, a_mhw_index_fn,
 *     setup_input_cache, setup_specialized_a_loader,
 *     load_a_tile_from_cache, load_a_tile_specialized, load_a_operand_from_cache,
 *     epilogue_override
 *
 *   Every closure captures the SAME enclosing-function locals: the resolved
 *   `spec` (here the gfx1201-pinned spec), the derived `conv_spec`
 *   (spec.conv_spec()), the resolved MMA `op` (_resolve_conv_op(conv_spec,
 *   "gfx1201") -> the WMMA 16x16x16 op), the `arch` string ("gfx1201"), and --
 *   inside epilogue_override -- the W1 buffer resource, the two staged LDS bases
 *   (c_smem / w1_smem), the conv1 accumulator vector, and the deferred-epilogue
 *   flag. build_implicit_gemm_conv calls these closures back with the builder
 *   `b`, a per-callback `conv_spec_`, the per-CTA WarpGrid `grid`, and the
 *   conv-managed resources (a_rsrc / y_rsrc / w1_rsrc / k_off / a_dst / row /
 *   col_base / accs).
 *
 *   In C there is no closure capture. The faithful port turns each Python
 *   closure into a free function taking a POINTER to one shared context struct,
 *   rocke_gfx1201_dfcp_build_ctx_t, which holds EXACTLY the variables the closures
 *   share. The gfx1201 public driver
 *   (rocke_build_gfx1201_deep_fused_conv_pool in
 *    rocke/instance_gfx1201_deep_fused_conv_pool.h) stamps the WMMA geometry/name
 *   onto the spec, populates the ctx in the order the Python prologue computes
 *   its locals (is_valid -> conv_spec -> op -> ctx), then hands the closure phase
 *   functions + the ctx (as the callback `user`) to the wrapped conv builder.
 *
 * RELATION TO THE COMMON INTERNAL HEADER.
 *   The common family-agnostic port has its own private surface
 *   (rocke/instance_deep_fused_conv_pool_internal.h, rocke_dfcp_build_ctx_t +
 *   rocke_dfcp_* phases). This gfx1201 surface is a SEPARATE, parallel set of
 *   symbols carrying the gfx1201_deep_fused_conv_pool / rocke_gfx1201_dfcp_ prefix
 *   so the gfx1201 shim translation units never clash with the common ones when
 *   both land in one program. The gfx1201 phase bodies are byte-faithful to the
 *   common closures driven by the WMMA-resolved op; they reuse the ported common
 *   emit/pure helpers (rocke_dfcp_* from
 *   rocke/helper_rocke.instances.common.deep_fused_conv_pool.h) and the shared
 *   helper closures (rocke_mmaop_{a,b,c}_layout / rocke_mmaop_coord from
 *   rocke/helper_rocke.core.arch.h, rocke_warp_grid_t + rocke_mmaop_{m,n}_* from
 *   rocke/helper_rocke.helpers.epilogues.h, rocke_coalesced_tile_loader_* from
 *   rocke/helper_rocke.helpers.loads.h, rocke_make_lds_view / rocke_tensor_view_t
 *   from rocke/helper_rocke.helpers.tensor_view.h, and the full rocke_mma_op_t
 *   struct from rocke/arch_target.h via the helper alias).
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every gfx1201 body-implementing agent
 *   binds to. It is DESIGNED TO BE COMPLETE: every local/closured variable the
 *   Python body shares across phases is a field here. A body agent implementing a
 *   phase .c file MUST be able to read/write only ctx fields and call the
 *   prototypes below WITHOUT editing this header. If a phase genuinely needs a
 *   value not present, that is a design bug in this header to fix once.
 *
 *   Naming: ctx fields mirror the Python local names 1:1; phase functions mirror
 *   the Python closure names with a `rocke_gfx1201_dfcp_` prefix.
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. It is included only by the
 * instance_gfx1201_deep_fused_conv_pool_*.c translation units. Public callers use
 * rocke/instance_gfx1201_deep_fused_conv_pool.h.
 */
#ifndef ROCKE_INSTANCE_GFX1201_DEEP_FUSED_CONV_POOL_INTERNAL_H
#define ROCKE_INSTANCE_GFX1201_DEEP_FUSED_CONV_POOL_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common.deep_fused_conv_pool.h"
#include "rocke/instance_gfx1201_deep_fused_conv_pool.h" /* public gfx1201 spec + build entry */
#include "rocke/ir.h"
/* spec/problem/epilogue value types + opaque rocke_warp_grid_t / rocke_mma_op_t /
 * rocke_implicit_gemm_conv_spec_t + the 22 emit/pure phase helper prototypes
 * (rocke_dfcp_*) the gfx1201 closures reuse. */

#ifdef __cplusplus
extern "C" {
#endif

/* The per-warp conv1 accumulator count is mfmas_per_warp_m * mfmas_per_warp_n.
 * For the gfx1201 WMMA deep-fusion tile space (one CTA owns all channels,
 * tile_n<=32, warp_tile 16x16) this is small (2 for the WMMA register-resident
 * path, a small handful otherwise). 64 is generous headroom and matches the
 * common internal-header convention. */
#define ROCKE_GFX1201_DFCP_MAX_ACCS 64

/* ===================================================================== *
 *  rocke_gfx1201_dfcp_build_ctx_t
 *
 *  The single shared state object for the gfx1201 shim. Holds every
 *  enclosing-function local that the build_deep_fused_conv_pool closures
 *  capture (with `op` resolved to the WMMA 16x16x16 op, `arch` == "gfx1201"),
 *  PLUS the per-callback scratch the conv builder threads back in (grid +
 *  conv-managed resources) staged here for the duration of one callback so a
 *  phase reads only the ctx. Grouped by lifetime: (A) build-time constants set
 *  once by the driver before the conv build, (B) per-callback scratch the conv
 *  builder hands in.
 * ===================================================================== */
typedef struct rocke_gfx1201_dfcp_build_ctx
{
    /* =============================================================== *
     * (A) BUILD-TIME CONSTANTS -- set once by
     *     rocke_build_gfx1201_deep_fused_conv_pool before calling
     *     build_implicit_gemm_conv; read by every closure.
     * =============================================================== */

    /* ---- inputs / resolved environment (driver args + prologue) ---- */
    rocke_ir_builder_t* b; /* the IRBuilder `b`        */
    const rocke_gfx1201_deep_fused_conv_pool_spec_t* spec; /* the gfx1201-pinned spec  */

    /* The common spec view of the gfx1201 spec (== &spec->base); the closures
     * forward this to the family-agnostic common emit helpers (rocke_dfcp_*),
     * which only know the common spec type. */
    const rocke_deep_fused_conv_pool_spec_t* common_spec;

    /* `arch` -- NULL-normalised to "gfx1201" for this shim. */
    const char* arch;

    /* conv_spec = spec.conv_spec(); the ImplicitGemmConvSpec wrapped by this
     * instance. Owned by the driver (value lives in driver scratch storage);
     * the ctx holds a pointer the closures forward as conv_spec / conv_spec_. */
    const rocke_implicit_gemm_conv_spec_t* conv_spec;

    /* op = _resolve_conv_op(conv_spec, "gfx1201"); the resolved WMMA MMA op
     * (wave32, m=n=k=16) that drives the numeric core
     * (op->m/n, op->{a,b,c}_frag_len, op->{a,b,c}_layout()). */
    const rocke_mma_op_t* op;

    /* ---- pool-deferral flag (epilogue_override local `defer`) ----
     * defer = _epilogue_is_pool_deferrable(spec.conv1_epilogue). The conv1
     * epilogue is deferred past the maxpool when set; `deferred_epi` is then the
     * conv1 epilogue, else NULL. Both are derivable from the spec, but staged
     * here so each maxpool phase reads the single decision the override computed
     * once. */
    bool defer;
    const rocke_conv_acc_epilogue_t* deferred_epi; /* &common_spec->conv1_epilogue or NULL */

    /* ---- WMMA register-residency decision (gfx1201 maxpool routing) ----
     * The gfx1201 WMMA path picks between the register-resident WMMA maxpool
     * fast path and the generic cshuffle-LDS gather maxpool. Staged so the
     * epilogue dispatch never re-derives the geometry predicate.
     *   use_wmma_register_maxpool =
     *       _maxpool_is_intra_lane_wmma(common_spec, grid, op)
     * (The MFMA-32x32 intra-lane fast path is geometry-gated off for WMMA
     *  warp_tile 16x16, so it is never selected here.) */
    bool use_wmma_register_maxpool;

    /* ---- conv0 A-load routing decision (driver prologue) ----
     * Mirrors the build_implicit_gemm_conv(...) override selection at the tail
     * of the Python driver. Each flag picks which closure the conv builder
     * invokes; staged so the phase dispatch never re-derives the spec predicate
     * chain.
     *   use_input_cache  = cache_input_footprint || direct_conv0_from_input_cache
     *   use_specialized  = (!use_input_cache) &&
     *                      _can_use_specialized_conv0_a_loader(common_spec)
     *   use_operand_ovr  = direct_conv0_from_input_cache */
    bool use_input_cache;
    bool use_specialized;
    bool use_operand_ovr;

    /* =============================================================== *
     * (B) PER-CALLBACK SCRATCH -- the conv builder threads these in when it
     *     calls a closure back. A phase function stages the args it was handed
     *     here (or the driver's callback trampoline does) so the body reads only
     *     the ctx. Each is valid only for the duration of the current callback.
     * =============================================================== */

    /* The per-callback conv_spec_ the builder passes (== conv_spec for this
     * instance; tracked separately to stay byte-faithful to the Python arg). */
    const rocke_implicit_gemm_conv_spec_t* conv_spec_cb;

    /* The per-CTA WarpGrid the builder passes to every callback. Holds the SSA
     * thread/lane/warp decode + tile geometry the emit helpers read (wave32
     * geometry for gfx1201). */
    const rocke_warp_grid_t* grid;

    /* ---- conv-managed resources (Values) threaded into callbacks ---- */
    rocke_value_t* a_rsrc; /* conv0 A buffer resource (loaders / cache setup) */
    rocke_value_t* y_rsrc; /* final pooled-output buffer resource (epilogue)  */
    rocke_value_t* w1_rsrc; /* W1 buffer resource -- RETURNED by extra_params,
                           * captured by epilogue_override (make_buffer_resource
                           * over the W1 / W1_bytes params). Persists across the
                           * whole conv build, so it is build-time once set. */

    /* ---- loader-callback inputs (per A-load callback) ---- */
    rocke_value_t* k_off; /* K-loop tile base offset (a_load / operand)      */
    rocke_value_t* a_dst; /* LDS A-tile destination (a_load_override)        */
    rocke_value_t* row; /* operand-load tile-local row (a_operand_override) */
    rocke_value_t* col_base; /* operand-load column base (a_operand_override)   */
    int frag_len; /* operand-load fragment width (a_operand_override) */

    /* ---- input-footprint cache (Value) ----
     * Returned by setup_input_cache (_setup_input_footprint_cache) and handed
     * back to load_a_tile_from_cache / load_a_operand_from_cache as `cache`.
     * For the specialized path setup_specialized_a_loader returns a_rsrc instead;
     * both are surfaced here as the opaque `cache` the loader closures consume. */
    rocke_value_t* input_cache;

    /* ---- epilogue_override staged locals (Values) ----
     * The two disjoint LDS producers the override stages before the merged
     * barrier, plus the conv1 accumulator vector the maxpool phases reduce. */
    rocke_value_t* c_smem; /* _stage_accumulators_to_cshuffle_lds(conv0, sync=False) */
    rocke_value_t* w1_smem; /* _load_conv1_weights_to_lds(sync=False)               */
    rocke_value_t* conv1_smem; /* _stage_accumulators_to_cshuffle_lds(conv1) -- generic
                              * cshuffle maxpool path only (else NULL).             */

    /* The conv0 accumulators the builder hands epilogue_override (`accs`) and the
     * conv1 accumulators _emit_conv1_1x1 returns. Inline-sized to the bounded
     * per-warp tile count. */
    rocke_value_t* conv0_accs[ROCKE_GFX1201_DFCP_MAX_ACCS];
    size_t num_conv0_accs;
    rocke_value_t* conv1_accs[ROCKE_GFX1201_DFCP_MAX_ACCS];
    size_t num_conv1_accs;
} rocke_gfx1201_dfcp_build_ctx_t;

/* Zero-initialise the ctx and populate the (A) build-time constants from the
 * gfx1201 spec + arch: normalises arch to "gfx1201", sets common_spec =
 * &spec->base, computes conv_spec / op (op is the WMMA 16x16x16 op via the peer
 * conv ports + _resolve_conv_op), derives `defer` / `deferred_epi`, resolves the
 * WMMA register-residency decision and the A-load routing flags. The caller owns
 * the conv_spec / op storage the ctx points at. Returns ROCKE_OK or a status
 * mirroring the Python prologue's ValueError paths (message in b->err). */
rocke_status_t
    rocke_gfx1201_dfcp_build_ctx_init(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                      rocke_ir_builder_t* b,
                                      const rocke_gfx1201_deep_fused_conv_pool_spec_t* spec,
                                      const char* arch,
                                      const rocke_implicit_gemm_conv_spec_t* conv_spec,
                                      const rocke_mma_op_t* op);

/* ===================================================================== *
 *  CLOSURE PHASE FUNCTIONS -- one per Python nested closure in
 *  build_deep_fused_conv_pool, driven by the gfx1201 (WMMA) op. Each reads/
 *  writes only ctx (and the builder it carries) and emits IR in the
 *  byte-identical Python order. The conv builder invokes these via thin
 *  trampolines (matching its callback signatures) that stage the per-callback
 *  args (grid / resources) onto the ctx, then call here.
 * ===================================================================== */

/* extra_params(b) -> W1 buffer resource.
 * Declares W1 (PtrType(F16,"global"), noalias, readonly, align=16) and W1_bytes
 * (I32) params, builds make_buffer_resource(W1, W1_bytes).rsrc, stores it in
 * ctx->w1_rsrc, and returns it. */
rocke_value_t* rocke_gfx1201_dfcp_extra_params(rocke_gfx1201_dfcp_build_ctx_t* ctx);

/* m_index_fn(b, row, grid) -> global (ho, wo) flattened M index.
 * tile-local row -> (local_h, local_w) (shift/mask when conv_tile_w is pow2),
 * then global_h*Wo + global_w. */
rocke_value_t* rocke_gfx1201_dfcp_m_index_fn(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                             rocke_value_t* row,
                                             const rocke_warp_grid_t* grid);

/* a_mhw_index_fn(b, row, grid) -> (n=0, global_h, global_w) coords.
 * Same decode as m_index_fn but returned as separate coords (N==1 => n const 0).
 * Writes the three coords into out_n / out_h / out_w. */
void rocke_gfx1201_dfcp_a_mhw_index_fn(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                       rocke_value_t* row,
                                       const rocke_warp_grid_t* grid,
                                       rocke_value_t** out_n,
                                       rocke_value_t** out_h,
                                       rocke_value_t** out_w);

/* setup_input_cache(b, conv_spec_, grid, a_rsrc) -> cache.
 * Stages grid/a_rsrc onto ctx, delegates to
 * rocke_dfcp_setup_input_footprint_cache (common helper) over ctx->common_spec,
 * stores + returns ctx->input_cache. */
rocke_value_t*
    rocke_gfx1201_dfcp_setup_input_cache(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                         const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                         const rocke_warp_grid_t* grid,
                                         rocke_value_t* a_rsrc);

/* setup_specialized_a_loader(b, conv_spec_, grid, a_rsrc) -> a_rsrc.
 * Identity passthrough (the specialized loader reads global memory directly);
 * stages a_rsrc onto ctx->input_cache and returns it. */
rocke_value_t*
    rocke_gfx1201_dfcp_setup_specialized_a_loader(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                                  const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                                  const rocke_warp_grid_t* grid,
                                                  rocke_value_t* a_rsrc);

/* load_a_tile_from_cache(b, conv_spec_, k_off, a_dst, grid, cache).
 * Early-return (no-op) when spec.direct_conv0_from_input_cache; else delegates
 * to rocke_dfcp_load_conv0_a_tile_from_input_cache (common helper). */
void rocke_gfx1201_dfcp_load_a_tile_from_cache(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                               const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                               rocke_value_t* k_off,
                                               rocke_value_t* a_dst,
                                               const rocke_warp_grid_t* grid,
                                               rocke_value_t* cache);

/* load_a_tile_specialized(b, conv_spec_, k_off, a_dst, grid, a_rsrc).
 * Delegates to rocke_dfcp_load_conv0_a_tile_specialized (common helper). */
void rocke_gfx1201_dfcp_load_a_tile_specialized(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                                const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                                rocke_value_t* k_off,
                                                rocke_value_t* a_dst,
                                                const rocke_warp_grid_t* grid,
                                                rocke_value_t* a_rsrc);

/* load_a_operand_from_cache(b, conv_spec_, row, k_off, col_base, frag_len, grid,
 *                           cache) -> packed operand fragment.
 * Delegates to rocke_dfcp_load_conv0_a_operand_from_input_cache (common helper). */
rocke_value_t*
    rocke_gfx1201_dfcp_load_a_operand_from_cache(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                                 const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                                 rocke_value_t* row,
                                                 rocke_value_t* k_off,
                                                 rocke_value_t* col_base,
                                                 int frag_len,
                                                 const rocke_warp_grid_t* grid,
                                                 rocke_value_t* cache);

/* epilogue_override(b, conv_spec_, accs, grid, y_rsrc, w1_rsrc).
 * The fused write-back: stage conv0 accs + W1 to disjoint LDS (sync=False),
 * single merged barrier, emit conv1 1x1 GEMM (deferred epilogue when
 * ctx->defer), then route to the WMMA register-resident maxpool
 * (rocke_dfcp_emit_wmma_maxpool_from_registers when
 *  ctx->use_wmma_register_maxpool) OR the generic cshuffle-LDS maxpool (stage
 * conv1 accs + rocke_dfcp_emit_inline_maxpool_from_cshuffle). Stages accs/grid/
 * resources onto ctx, then walks the phases in Python order. */
void rocke_gfx1201_dfcp_epilogue_override(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                          const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                          rocke_value_t* const* accs,
                                          size_t num_accs,
                                          const rocke_warp_grid_t* grid,
                                          rocke_value_t* y_rsrc,
                                          rocke_value_t* w1_rsrc);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1201_DEEP_FUSED_CONV_POOL_INTERNAL_H */
