/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_deep_fused_conv_pool_internal.h -- PRIVATE shared state + phase /
 * closure-function contract for the C99 port of build_deep_fused_conv_pool
 * (rocke/instances/common/deep_fused_conv_pool.py, lines 1212-1401).
 *
 * WHY THIS HEADER EXISTS.
 *   build_deep_fused_conv_pool() is a driver whose body is a STACK OF NESTED
 *   CLOSURES handed to the wrapped conv0 builder build_implicit_gemm_conv():
 *
 *     extra_params, m_index_fn, a_mhw_index_fn,
 *     setup_input_cache, setup_specialized_a_loader,
 *     load_a_tile_from_cache, load_a_tile_specialized, load_a_operand_from_cache,
 *     epilogue_override
 *
 *   Every closure captures the SAME enclosing-function locals: the resolved
 *   `spec`, the derived `conv_spec` (spec.conv_spec()), the resolved MMA `op`
 *   (_resolve_conv_op(conv_spec, arch)), the `arch` string, and -- inside
 *   epilogue_override -- the W1 buffer resource, the two staged LDS bases
 *   (c_smem / w1_smem), the conv1 accumulator vector, and the deferred-epilogue
 *   flag. build_implicit_gemm_conv calls these closures back with the builder
 *   `b`, a per-callback `conv_spec_`, the per-CTA WarpGrid `grid`, and the
 *   conv-managed resources (a_rsrc / y_rsrc / w1_rsrc / k_off / a_dst / row /
 *   col_base / accs).
 *
 *   In C there is no closure capture. The faithful port turns each Python
 *   closure into a free function taking a POINTER to one shared context struct,
 *   rocke_dfcp_build_ctx_t, which holds EXACTLY the variables the closures share.
 *   The public driver (rocke_build_deep_fused_conv_pool in
 *   rocke/instance_deep_fused_conv_pool.h) populates the ctx in the order the
 *   Python prologue computes its locals (is_valid -> conv_spec -> op -> ctx),
 *   then hands the closure phase functions + the ctx (as the callback `user`)
 *   to the wrapped conv builder.
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every body-implementing agent binds
 *   to. It is DESIGNED TO BE COMPLETE: every local/closured variable the Python
 *   body shares across phases is a field here. A body agent implementing a phase
 *   .c file MUST be able to read/write only ctx fields and call the prototypes
 *   below WITHOUT editing this header. If a phase genuinely needs a value not
 *   present, that is a design bug in this header to fix once, deliberately.
 *
 *   Naming: ctx fields mirror the Python local names 1:1 (Python `conv_spec`
 *   -> ctx->conv_spec; the W1 resource the extra_params closure returns ->
 *   ctx->w1_rsrc; the epilogue_override staged bases -> ctx->c_smem / w1_smem).
 *   Phase functions mirror the Python closure names with a `rocke_dfcp_` prefix
 *   (the module-level emit/pure helpers keep their own `rocke_dfcp_*` names from
 *   the helper header and are NOT re-declared here).
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. It is included only by the
 * instance_deep_fused_conv_pool_*.c translation units. Public callers use
 * rocke/instance_deep_fused_conv_pool.h.
 */
#ifndef ROCKE_INSTANCE_DEEP_FUSED_CONV_POOL_INTERNAL_H
#define ROCKE_INSTANCE_DEEP_FUSED_CONV_POOL_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common.deep_fused_conv_pool.h"
#include "rocke/instance_deep_fused_conv_pool.h" /* public spec + build entry */
#include "rocke/ir.h"
/* spec/problem/epilogue value types + opaque rocke_warp_grid_t / rocke_mma_op_t /
 * rocke_implicit_gemm_conv_spec_t + the 22 emit/pure phase helper prototypes */

#ifdef __cplusplus
extern "C" {
#endif

/* The per-warp conv1 accumulator count is mfmas_per_warp_m * mfmas_per_warp_n.
 * For the covered deep-fusion tile space (one CTA owns all channels, tile_n<=32,
 * warp_tile 16/32) this is tiny (1 for the MFMA-32x32 intra-lane path, 2 for the
 * WMMA path, a small handful otherwise). 64 is generous headroom and matches the
 * gemm internal-header convention. */
#define ROCKE_DFCP_MAX_ACCS 64

/* ===================================================================== *
 *  rocke_dfcp_build_ctx_t
 *
 *  The single shared state object. Holds every enclosing-function local that
 *  the build_deep_fused_conv_pool closures capture, PLUS the per-callback
 *  scratch the conv builder threads back in (grid + conv-managed resources)
 *  staged here for the duration of one callback so a phase reads only the ctx.
 *  Grouped by lifetime: (A) build-time constants set once by the driver before
 *  the conv build, (B) per-callback scratch the conv builder hands in.
 * ===================================================================== */
typedef struct rocke_dfcp_build_ctx
{
    /* =============================================================== *
     * (A) BUILD-TIME CONSTANTS -- set once by rocke_build_deep_fused_conv_pool
     *     before calling build_implicit_gemm_conv; read by every closure.
     * =============================================================== */

    /* ---- inputs / resolved environment (driver args + prologue) ---- */
    rocke_ir_builder_t* b; /* the IRBuilder `b`            */
    const rocke_deep_fused_conv_pool_spec_t* spec; /* the DeepFusedConvPoolSpec    */
    const char* arch; /* `arch` (NULL-normalised "gfx950") */

    /* conv_spec = spec.conv_spec(); the ImplicitGemmConvSpec wrapped by this
     * instance. Owned by the driver (value lives in driver scratch storage);
     * the ctx holds a pointer the closures forward as conv_spec / conv_spec_. */
    const rocke_implicit_gemm_conv_spec_t* conv_spec; /* spec.conv_spec()             */

    /* op = _resolve_conv_op(conv_spec, arch); the resolved MMA op that drives the
     * numeric core (op->m/n, op->{a,b,c}_frag_len, op->{a,b,c}_layout()). */
    const rocke_mma_op_t* op; /* _resolve_conv_op(...)        */

    /* ---- pool-deferral flag (epilogue_override local `defer`) ----
     * defer = _epilogue_is_pool_deferrable(spec.conv1_epilogue). The conv1
     * epilogue is deferred past the maxpool when set; `deferred_epi` is then the
     * conv1 epilogue, else NULL (Python `deferred_epi = ... if defer else None`).
     * Both are derivable from spec, but staged here so each maxpool phase reads
     * the single decision the override computed once. */
    bool defer; /* _epilogue_is_pool_deferrable */
    const rocke_conv_acc_epilogue_t* deferred_epi; /* &spec->conv1_epilogue or NULL */

    /* ---- conv0 A-load routing decision (driver prologue) ----
     * Mirrors the build_implicit_gemm_conv(...) override selection at the tail of
     * the Python driver. Each flag picks which closure the conv builder invokes;
     * staged so the phase dispatch never re-derives the spec predicate chain.
     *   use_input_cache  = cache_input_footprint || direct_conv0_from_input_cache
     *   use_specialized  = (!use_input_cache) &&
     *                      _can_use_specialized_conv0_a_loader(spec)
     *   use_operand_ovr  = direct_conv0_from_input_cache
     * (When use_input_cache is true the conv builder calls setup_input_cache +
     *  load_a_tile_from_cache; when use_specialized is true it calls
     *  setup_specialized_a_loader + load_a_tile_specialized; else neither.) */
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
     * thread/lane/warp decode + tile geometry the emit helpers read. */
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
    rocke_value_t* conv0_accs[ROCKE_DFCP_MAX_ACCS];
    size_t num_conv0_accs;
    rocke_value_t* conv1_accs[ROCKE_DFCP_MAX_ACCS];
    size_t num_conv1_accs;
} rocke_dfcp_build_ctx_t;

/* Zero-initialise the ctx and populate the (A) build-time constants from spec +
 * arch: normalises arch, computes conv_spec / op (via the peer conv ports),
 * derives `defer` / `deferred_epi`, and resolves the A-load routing flags. The
 * caller owns the conv_spec / op storage the ctx points at. Returns ROCKE_OK or a
 * status mirroring the Python prologue's ValueError paths (message in b->err). */
rocke_status_t rocke_dfcp_build_ctx_init(rocke_dfcp_build_ctx_t* ctx,
                                         rocke_ir_builder_t* b,
                                         const rocke_deep_fused_conv_pool_spec_t* spec,
                                         const char* arch,
                                         const rocke_implicit_gemm_conv_spec_t* conv_spec,
                                         const rocke_mma_op_t* op);

/* ===================================================================== *
 *  CLOSURE PHASE FUNCTIONS -- one per Python nested closure in
 *  build_deep_fused_conv_pool. Each reads/writes only ctx (and the builder it
 *  carries) and emits IR in the byte-identical Python order. The conv builder
 *  invokes these via thin trampolines (matching its callback signatures) that
 *  stage the per-callback args (grid / resources) onto the ctx, then call here.
 * ===================================================================== */

/* extra_params(b) -> W1 buffer resource.
 * Declares W1 (PtrType(F16,"global"), noalias, readonly, align=16) and W1_bytes
 * (I32) params, builds make_buffer_resource(W1, W1_bytes).rsrc, stores it in
 * ctx->w1_rsrc, and returns it. */
rocke_value_t* rocke_dfcp_extra_params(rocke_dfcp_build_ctx_t* ctx);

/* m_index_fn(b, row, grid) -> global (ho, wo) flattened M index.
 * tile-local row -> (local_h, local_w) (shift/mask when conv_tile_w is pow2),
 * then global_h*Wo + global_w. */
rocke_value_t* rocke_dfcp_m_index_fn(rocke_dfcp_build_ctx_t* ctx,
                                     rocke_value_t* row,
                                     const rocke_warp_grid_t* grid);

/* a_mhw_index_fn(b, row, grid) -> (n=0, global_h, global_w) coords.
 * Same decode as m_index_fn but returned as separate coords (N==1 => n const 0).
 * Writes the three coords into out_n / out_h / out_w. */
void rocke_dfcp_a_mhw_index_fn(rocke_dfcp_build_ctx_t* ctx,
                               rocke_value_t* row,
                               const rocke_warp_grid_t* grid,
                               rocke_value_t** out_n,
                               rocke_value_t** out_h,
                               rocke_value_t** out_w);

/* setup_input_cache(b, conv_spec_, grid, a_rsrc) -> cache.
 * Stages grid/a_rsrc onto ctx, delegates to
 * rocke_dfcp_setup_input_footprint_cache, stores + returns ctx->input_cache. */
rocke_value_t* rocke_dfcp_setup_input_cache(rocke_dfcp_build_ctx_t* ctx,
                                            const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                            const rocke_warp_grid_t* grid,
                                            rocke_value_t* a_rsrc);

/* setup_specialized_a_loader(b, conv_spec_, grid, a_rsrc) -> a_rsrc.
 * Identity passthrough (the specialized loader reads global memory directly);
 * stages a_rsrc onto ctx->input_cache and returns it. */
rocke_value_t*
    rocke_dfcp_setup_specialized_a_loader(rocke_dfcp_build_ctx_t* ctx,
                                          const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                          const rocke_warp_grid_t* grid,
                                          rocke_value_t* a_rsrc);

/* load_a_tile_from_cache(b, conv_spec_, k_off, a_dst, grid, cache).
 * Early-return (no-op) when spec.direct_conv0_from_input_cache; else delegates
 * to rocke_dfcp_load_conv0_a_tile_from_input_cache. */
void rocke_dfcp_load_a_tile_from_cache(rocke_dfcp_build_ctx_t* ctx,
                                       const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                       rocke_value_t* k_off,
                                       rocke_value_t* a_dst,
                                       const rocke_warp_grid_t* grid,
                                       rocke_value_t* cache);

/* load_a_tile_specialized(b, conv_spec_, k_off, a_dst, grid, a_rsrc).
 * Delegates to rocke_dfcp_load_conv0_a_tile_specialized. */
void rocke_dfcp_load_a_tile_specialized(rocke_dfcp_build_ctx_t* ctx,
                                        const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                        rocke_value_t* k_off,
                                        rocke_value_t* a_dst,
                                        const rocke_warp_grid_t* grid,
                                        rocke_value_t* a_rsrc);

/* load_a_operand_from_cache(b, conv_spec_, row, k_off, col_base, frag_len, grid,
 *                           cache) -> packed operand fragment.
 * Delegates to rocke_dfcp_load_conv0_a_operand_from_input_cache. */
rocke_value_t*
    rocke_dfcp_load_a_operand_from_cache(rocke_dfcp_build_ctx_t* ctx,
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
 * ctx->defer), then route to the intra-lane MFMA / WMMA register-resident
 * maxpool or the generic cshuffle-LDS maxpool. Stages accs/grid/resources onto
 * ctx, then walks the phases in Python order. */
void rocke_dfcp_epilogue_override(rocke_dfcp_build_ctx_t* ctx,
                                  const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                  rocke_value_t* const* accs,
                                  size_t num_accs,
                                  const rocke_warp_grid_t* grid,
                                  rocke_value_t* y_rsrc,
                                  rocke_value_t* w1_rsrc);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_DEEP_FUSED_CONV_POOL_INTERNAL_H */
