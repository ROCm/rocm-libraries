// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_deep_fused_conv_pool_build_entry_and_closures.c -- the PUBLIC build
 * entry + the nine closure-glue phase functions of the chunked C99 port of
 * ck_dsl/instances/common/deep_fused_conv_pool.py (build_deep_fused_conv_pool,
 * Python lines 1212-1401).
 *
 * SCOPE OF THIS PART-FILE (one bucket of the chunked port):
 *   - ckc_build_deep_fused_conv_pool / _new  (driver: is_valid gate, conv_spec /
 *     op resolve, ctx populate, drive the wrapped build_implicit_gemm_conv).
 *   - ckc_dfcp_build_ctx_init (ctx prologue).
 *   - the nine closure phases over the ctx:
 *       extra_params, m_index_fn, a_mhw_index_fn,
 *       setup_input_cache, setup_specialized_a_loader,
 *       load_a_tile_from_cache, load_a_tile_specialized,
 *       load_a_operand_from_cache, epilogue_override
 *     plus the conv-builder callback trampolines that adapt the conv-builder
 *     callback ABI (ckc_conv_build_overrides_t) onto these ctx phases.
 *   - ckc_deep_fused_conv_pool_lower_to_llvm convenience.
 *
 * This bucket is the ONLY part that binds to the peer build_implicit_gemm_conv /
 * _resolve_conv_op / spec.conv_spec() ports. Those peer C symbols live behind
 * ckc/instance_conv_implicit_gemm.h, whose ckc_conv_acc_epilogue_t typedef
 * collides with the deep-fused helper's same-named slice typedef (different
 * field order) -- the two public headers cannot coexist in one TU. So the peer
 * surface this driver needs is forward-declared below as opaque externs (matching
 * the byte-faithful Python call pattern); the verify+fix loop links them once the
 * peer ports are wired. Everything else binds to the internal header + helper
 * header + ir.h exactly as instructed.
 *
 * Byte-identical builder-call sequence: the driver walks the Python prologue
 * (is_valid -> conv_spec -> op -> ctx) then hands the phase trampolines + the ctx
 * (as the callback `user`) to the wrapped conv builder; epilogue_override walks
 * stage -> W1-load -> barrier -> conv1 -> maxpool-routing in Python order.
 */
#include "ckc/instance_deep_fused_conv_pool.h"
#include "ckc/instance_deep_fused_conv_pool_internal.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "ckc/ir.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err */
#include "ckc/lower_llvm.h"

/* ===================================================================== *
 * PEER PORT FORWARD DECLARATIONS (opaque; see file banner)
 *
 * These mirror the Python symbols this driver wraps:
 *   spec.conv_spec()              -> ckc_deep_fused_conv_pool_spec_conv_spec(...)
 *   _resolve_conv_op(spec, arch)  -> ckc_conv_resolve_op(...)
 *   build_implicit_gemm_conv(...) -> ckc_build_implicit_gemm_conv(...)
 *
 * The peer build entry takes a ckc_conv_build_overrides_t with the six optional
 * callbacks; we declare a structurally-identical local mirror of that struct +
 * the WarpGrid fwd so this TU type-checks without including the colliding
 * conv public header. The real port's struct layout matches one-for-one (it is
 * the documented stable callback ABI in ckc/instance_conv_implicit_gemm.h).
 * ===================================================================== */

/* spec.conv_spec(): build the wrapped ImplicitGemmConvSpec. The peer owns the
 * returned storage's lifetime (it is value-stable for the build). Returns NULL +
 * sets b's sticky error on failure. */
/* C++ build: cross-TU C-ABI peers; forward decls must be extern "C" so the
 * references are not mangled. No effect in C. */
#ifdef __cplusplus
extern "C" {
#endif
extern const ckc_implicit_gemm_conv_spec_t*
    ckc_deep_fused_conv_pool_spec_conv_spec(ckc_ir_builder_t* b,
                                            const ckc_deep_fused_conv_pool_spec_t* spec);

/* _resolve_conv_op(conv_spec, arch) -> MmaOp. Sets b's error + returns NULL on
 * the no-atom ValueError path. (Peer arch-port type is opaque to this TU.) */
extern const ckc_mma_op_t* ckc_conv_resolve_op(ckc_ir_builder_t* b,
                                               const ckc_implicit_gemm_conv_spec_t* conv_spec,
                                               const char* arch);
#ifdef __cplusplus
}
#endif

/* The conv builder callback ABI (mirror of ckc_conv_build_overrides_t from
 * ckc/instance_conv_implicit_gemm.h). Layout is the stable override surface. */
typedef struct ckc_dfcp_conv_overrides
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
} ckc_dfcp_conv_overrides_t;

/* build_implicit_gemm_conv(conv_spec, arch=..., **overrides). Emits the conv0
 * driver into b and returns the kernel (b->kernel) or NULL + b error.
 *
 * INTEGRATION: the canonical prototype now comes from
 * ckc/instance_conv_implicit_gemm.h (pulled in transitively). It takes the
 * canonical ckc_conv_build_overrides_t*, whose layout is identical to the local
 * ckc_dfcp_conv_overrides_t mirror above; the call site casts the staged mirror
 * to the canonical type. The previous local `extern` redeclaration with the
 * mirror type conflicted with the canonical prototype and is removed. */

/* ===================================================================== *
 * ckc_dfcp_build_ctx_init -- the Python build_deep_fused_conv_pool prologue.
 *
 * Mirrors:
 *   ok, why = is_valid_spec(spec, arch); if not ok: raise ...   (driver gate)
 *   conv_spec = spec.conv_spec()
 *   op = _resolve_conv_op(conv_spec, arch)
 *   + the override-routing flag derivation done at the build call tail.
 *
 * NOTE: the validity gate + conv_spec/op resolution are performed by the public
 * driver (it owns the conv_spec/op storage); this init takes them as args and
 * stages the build-time-constant ctx fields, exactly per the internal-header
 * contract.
 * ===================================================================== */
ckc_status_t ckc_dfcp_build_ctx_init(ckc_dfcp_build_ctx_t* ctx,
                                     ckc_ir_builder_t* b,
                                     const ckc_deep_fused_conv_pool_spec_t* spec,
                                     const char* arch,
                                     const ckc_implicit_gemm_conv_spec_t* conv_spec,
                                     const ckc_mma_op_t* op)
{
    if(ctx == NULL || b == NULL || spec == NULL)
    {
        return CKC_ERR_VALUE;
    }

    memset(ctx, 0, sizeof(*ctx));

    /* (A) build-time constants ------------------------------------------ */
    ctx->b = b;
    ctx->spec = spec;
    ctx->arch = (arch != NULL) ? arch : "gfx950"; /* arch NULL => "gfx950" */
    ctx->conv_spec = conv_spec;
    ctx->op = op;

    /* defer = _epilogue_is_pool_deferrable(spec.conv1_epilogue)
     * deferred_epi = spec.conv1_epilogue if defer else None
     * (computed once in the override; staged here so each maxpool phase reads
     * the single decision -- matches the internal-header contract). */
    ctx->defer = ckc_dfcp_epilogue_is_pool_deferrable(&spec->conv1_epilogue);
    ctx->deferred_epi = ctx->defer ? &spec->conv1_epilogue : NULL;

    /* A-load routing (the build_implicit_gemm_conv override selection tail):
     *   input_cache_setup = setup_input_cache
     *       if (cache_input_footprint or direct_conv0_from_input_cache)
     *       else setup_specialized_a_loader
     *            if _can_use_specialized_conv0_a_loader(spec)
     *       else None
     *   a_load_override   = load_a_tile_from_cache  (same gate)
     *                       else load_a_tile_specialized (specialized gate)
     *                       else None
     *   a_operand_override = load_a_operand_from_cache
     *                        if direct_conv0_from_input_cache else None */
    ctx->use_input_cache = spec->cache_input_footprint || spec->direct_conv0_from_input_cache;
    ctx->use_specialized
        = (!ctx->use_input_cache) && ckc_dfcp_can_use_specialized_conv0_a_loader(spec);
    ctx->use_operand_ovr = spec->direct_conv0_from_input_cache;

    return CKC_OK;
}

/* ===================================================================== *
 * CLOSURE PHASE: extra_params(b) -> W1 buffer resource.
 *
 *   W1       = b.param("W1", PtrType(F16,"global"), noalias=True,
 *                      readonly=True, align=16)
 *   W1_bytes = b.param("W1_bytes", I32)
 *   return make_buffer_resource(b, W1, num_bytes=W1_bytes).rsrc
 * ===================================================================== */
ckc_value_t* ckc_dfcp_extra_params(ckc_dfcp_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b;
    const ckc_type_t* f16;
    const ckc_type_t* ptr_f16_global;
    ckc_param_opts_t opts;
    ckc_value_t* w1;
    ckc_value_t* w1_bytes;
    ckc_value_t* rsrc;

    if(ctx == NULL || ctx->b == NULL)
    {
        return NULL;
    }
    b = ctx->b;

    /* PtrType(F16, "global") */
    f16 = ckc_f16();
    ptr_f16_global = ckc_ptr_type(b, f16, "global");

    /* W1 = b.param("W1", ptr, noalias=True, readonly=True, align=16) */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    w1 = ckc_b_param(b, "W1", ptr_f16_global, &opts);

    /* W1_bytes = b.param("W1_bytes", I32) */
    w1_bytes = ckc_b_param(b, "W1_bytes", ckc_i32(), NULL);

    /* make_buffer_resource(b, W1, num_bytes=W1_bytes).rsrc
     *
     * Python's make_buffer_resource (helpers/tensor_view.py) builds the rsrc AND
     * a pre-bound zero soffset (soffset = b.const_i32(0)) before returning the
     * BufferResource; extra_params keeps only .rsrc. That discarded const_i32(0)
     * still consumes one build-time SSA counter (it is DCE'd before printing), so
     * a port that omits it shifts every later numbered SSA name by -1 (e.g.
     * @A_smem22 vs the reference @A_smem23). Emit the throwaway const so the
     * whole IR stays byte-identical to the Python reference. */
    rsrc = ckc_b_buffer_rsrc(b, w1, w1_bytes);
    (void)ckc_b_const_i32(b, 0);

    ctx->w1_rsrc = rsrc;
    return rsrc;
}

/* ---- shared (ho, wo) decode used by m_index_fn / a_mhw_index_fn ----------
 * tile-local row -> (global_h, global_w). Strength-reduces div/mod by
 * conv_tile_w when power-of-2 (matches both Python closures verbatim). */
static void ckc_dfcp_decode_row_to_hw(ckc_ir_builder_t* b,
                                      const ckc_deep_fused_conv_pool_spec_t* spec,
                                      ckc_value_t* row,
                                      ckc_value_t** out_h,
                                      ckc_value_t** out_w)
{
    const ckc_fused_conv_pool_problem_t* p = &spec->problem;
    int conv_tile_w = spec->pool_tile_w * p->pool_stride_w;
    ckc_value_t* local_h;
    ckc_value_t* local_w;
    ckc_value_t* global_h;
    ckc_value_t* global_w;

    if(conv_tile_w > 0 && (conv_tile_w & (conv_tile_w - 1)) == 0)
    {
        /* shift = (conv_tile_w - 1).bit_length() */
        int shift = 0;
        int v = conv_tile_w - 1;
        while(v > 0)
        {
            ++shift;
            v >>= 1;
        }
        local_h = ckc_b_lshr(b, row, ckc_b_const_i32(b, shift));
        local_w = ckc_b_land(b, row, ckc_b_const_i32(b, conv_tile_w - 1));
    }
    else
    {
        ckc_value_t* c_conv_tile_w = ckc_b_const_i32(b, conv_tile_w);
        local_h = ckc_b_div(b, row, c_conv_tile_w);
        local_w = ckc_b_mod(b, row, c_conv_tile_w);
    }

    /* global_h = block_id_y()*(pool_tile_h*pool_stride_h) + local_h
     *
     * Python (b.mul(b.block_id_y(), b.const_i32(...))) evaluates its arguments
     * left-to-right, so block_id_y() takes its SSA counter slot BEFORE the const.
     * C argument evaluation order is unspecified (GCC here is right-to-left),
     * which would swap the two slots and shift every later numbered SSA name.
     * Hoist block_id_y/_z into temps first to pin the Python source-order. */
    {
        ckc_value_t* bid_y = ckc_b_block_id_y(b);
        global_h = ckc_b_add(
            b,
            ckc_b_mul(b, bid_y, ckc_b_const_i32(b, spec->pool_tile_h * p->pool_stride_h)),
            local_h);
    }
    /* global_w = block_id_z()*(pool_tile_w*pool_stride_w) + local_w */
    {
        ckc_value_t* bid_z = ckc_b_block_id_z(b);
        global_w = ckc_b_add(
            b,
            ckc_b_mul(b, bid_z, ckc_b_const_i32(b, spec->pool_tile_w * p->pool_stride_w)),
            local_w);
    }

    *out_h = global_h;
    *out_w = global_w;
}

/* ===================================================================== *
 * CLOSURE PHASE: m_index_fn(b, row, grid) -> global (ho, wo) flattened M index.
 *   return global_h * Wo + global_w
 * ===================================================================== */
ckc_value_t*
    ckc_dfcp_m_index_fn(ckc_dfcp_build_ctx_t* ctx, ckc_value_t* row, const ckc_warp_grid_t* grid)
{
    ckc_ir_builder_t* b;
    const ckc_conv_problem_t* c;
    ckc_value_t* global_h;
    ckc_value_t* global_w;

    (void)grid; /* Python `_grid` is unused in m_index_fn */
    if(ctx == NULL || ctx->b == NULL)
    {
        return NULL;
    }
    b = ctx->b;
    c = &ctx->spec->problem.conv;

    ckc_dfcp_decode_row_to_hw(b, ctx->spec, row, &global_h, &global_w);

    /* b.add(b.mul(global_h, b.const_i32(c.Wo)), global_w) */
    return ckc_b_add(
        b, ckc_b_mul(b, global_h, ckc_b_const_i32(b, ckc_conv_problem_wo(c))), global_w);
}

/* ===================================================================== *
 * CLOSURE PHASE: a_mhw_index_fn(b, row, grid) -> (n=0, global_h, global_w).
 *   Same (ho, wo) decode as m_index_fn, returned as separate coords. N==1 so
 *   n is constant 0.
 * ===================================================================== */
void ckc_dfcp_a_mhw_index_fn(ckc_dfcp_build_ctx_t* ctx,
                             ckc_value_t* row,
                             const ckc_warp_grid_t* grid,
                             ckc_value_t** out_n,
                             ckc_value_t** out_h,
                             ckc_value_t** out_w)
{
    ckc_ir_builder_t* b;
    ckc_value_t* global_h;
    ckc_value_t* global_w;

    (void)grid;
    if(ctx == NULL || ctx->b == NULL)
    {
        return;
    }
    b = ctx->b;

    ckc_dfcp_decode_row_to_hw(b, ctx->spec, row, &global_h, &global_w);

    if(out_n != NULL)
    {
        *out_n = ckc_b_const_i32(b, 0);
    }
    if(out_h != NULL)
    {
        *out_h = global_h;
    }
    if(out_w != NULL)
    {
        *out_w = global_w;
    }
}

/* ===================================================================== *
 * CLOSURE PHASE: setup_input_cache(b, conv_spec_, grid, a_rsrc) -> cache.
 *   return _setup_input_footprint_cache(b, spec, a_rsrc, grid)
 * ===================================================================== */
ckc_value_t* ckc_dfcp_setup_input_cache(ckc_dfcp_build_ctx_t* ctx,
                                        const ckc_implicit_gemm_conv_spec_t* conv_spec_,
                                        const ckc_warp_grid_t* grid,
                                        ckc_value_t* a_rsrc)
{
    ckc_value_t* cache;

    (void)conv_spec_;
    if(ctx == NULL || ctx->b == NULL)
    {
        return NULL;
    }
    /* stage per-callback scratch on the ctx so the body reads only the ctx */
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->a_rsrc = a_rsrc;

    cache = ckc_dfcp_setup_input_footprint_cache(ctx->b, ctx->spec, a_rsrc, grid);
    ctx->input_cache = cache;
    return cache;
}

/* ===================================================================== *
 * CLOSURE PHASE: setup_specialized_a_loader(b, conv_spec_, grid, a_rsrc).
 *   return a_rsrc   (identity passthrough)
 * ===================================================================== */
ckc_value_t* ckc_dfcp_setup_specialized_a_loader(ckc_dfcp_build_ctx_t* ctx,
                                                 const ckc_implicit_gemm_conv_spec_t* conv_spec_,
                                                 const ckc_warp_grid_t* grid,
                                                 ckc_value_t* a_rsrc)
{
    (void)conv_spec_;
    if(ctx == NULL)
    {
        return NULL;
    }
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->a_rsrc = a_rsrc;
    ctx->input_cache = a_rsrc; /* the specialized loader reads global directly */
    return a_rsrc;
}

/* ===================================================================== *
 * CLOSURE PHASE: load_a_tile_from_cache(b, conv_spec_, k_off, a_dst, grid, cache)
 *   if spec.direct_conv0_from_input_cache: return        (no-op)
 *   _load_conv0_a_tile_from_input_cache(b, spec, conv_spec_, k_off, a_dst, grid,
 *                                       cache)
 * ===================================================================== */
void ckc_dfcp_load_a_tile_from_cache(ckc_dfcp_build_ctx_t* ctx,
                                     const ckc_implicit_gemm_conv_spec_t* conv_spec_,
                                     ckc_value_t* k_off,
                                     ckc_value_t* a_dst,
                                     const ckc_warp_grid_t* grid,
                                     ckc_value_t* cache)
{
    if(ctx == NULL || ctx->b == NULL)
    {
        return;
    }
    /* Early-return when direct_conv0_from_input_cache (load is fused into the
     * operand override instead). */
    if(ctx->spec->direct_conv0_from_input_cache)
    {
        return;
    }
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->k_off = k_off;
    ctx->a_dst = a_dst;
    ctx->input_cache = cache;

    ckc_dfcp_load_conv0_a_tile_from_input_cache(
        ctx->b, ctx->spec, conv_spec_, k_off, a_dst, grid, cache);
}

/* ===================================================================== *
 * CLOSURE PHASE: load_a_tile_specialized(b, conv_spec_, k_off, a_dst, grid,
 *                                        a_rsrc)
 *   _load_conv0_a_tile_specialized(b, spec, conv_spec_, k_off, a_dst, grid,
 *                                  a_rsrc)
 * ===================================================================== */
void ckc_dfcp_load_a_tile_specialized(ckc_dfcp_build_ctx_t* ctx,
                                      const ckc_implicit_gemm_conv_spec_t* conv_spec_,
                                      ckc_value_t* k_off,
                                      ckc_value_t* a_dst,
                                      const ckc_warp_grid_t* grid,
                                      ckc_value_t* a_rsrc)
{
    if(ctx == NULL || ctx->b == NULL)
    {
        return;
    }
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->k_off = k_off;
    ctx->a_dst = a_dst;
    ctx->a_rsrc = a_rsrc;

    ckc_dfcp_load_conv0_a_tile_specialized(
        ctx->b, ctx->spec, conv_spec_, k_off, a_dst, grid, a_rsrc);
}

/* ===================================================================== *
 * CLOSURE PHASE: load_a_operand_from_cache(b, conv_spec_, row, k_off, col_base,
 *                                          frag_len, grid, cache) -> Value
 *   return _load_conv0_a_operand_from_input_cache(b, spec, row, k_off, col_base,
 *                                                 frag_len, cache)
 * ===================================================================== */
ckc_value_t* ckc_dfcp_load_a_operand_from_cache(ckc_dfcp_build_ctx_t* ctx,
                                                const ckc_implicit_gemm_conv_spec_t* conv_spec_,
                                                ckc_value_t* row,
                                                ckc_value_t* k_off,
                                                ckc_value_t* col_base,
                                                int frag_len,
                                                const ckc_warp_grid_t* grid,
                                                ckc_value_t* cache)
{
    (void)conv_spec_;
    if(ctx == NULL || ctx->b == NULL)
    {
        return NULL;
    }
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->row = row;
    ctx->k_off = k_off;
    ctx->col_base = col_base;
    ctx->frag_len = frag_len;
    ctx->input_cache = cache;

    return ckc_dfcp_load_conv0_a_operand_from_input_cache(
        ctx->b, ctx->spec, row, k_off, col_base, frag_len, cache);
}

/* ===================================================================== *
 * CLOSURE PHASE: epilogue_override(b, conv_spec_, accs, grid, y_rsrc, w1_rsrc)
 *
 * The fused write-back. Walks (Python order):
 *   c_smem = _stage_accumulators_to_cshuffle_lds(b, op, accs, grid, sync=False)
 *   w1_smem = _load_conv1_weights_to_lds(b, spec, w1_rsrc, grid, sync=False)
 *   b.sync()                                   # single merged barrier
 *   defer = _epilogue_is_pool_deferrable(spec.conv1_epilogue)
 *   conv1_accs = _emit_conv1_1x1(..., defer_epilogue=defer)
 *   deferred_epi = spec.conv1_epilogue if defer else None
 *   if _maxpool_is_intra_lane(spec, grid):
 *       _emit_inline_maxpool_from_registers(..., epilogue=deferred_epi)
 *   elif _maxpool_is_intra_lane_wmma(spec, grid, op):
 *       _emit_wmma_maxpool_from_registers(..., op, epilogue=deferred_epi)
 *   else:
 *       conv1_smem = _stage_accumulators_to_cshuffle_lds(b, op, conv1_accs, grid)
 *       _emit_inline_maxpool_from_cshuffle(..., epilogue=deferred_epi)
 * ===================================================================== */
void ckc_dfcp_epilogue_override(ckc_dfcp_build_ctx_t* ctx,
                                const ckc_implicit_gemm_conv_spec_t* conv_spec_,
                                ckc_value_t* const* accs,
                                size_t num_accs,
                                const ckc_warp_grid_t* grid,
                                ckc_value_t* y_rsrc,
                                ckc_value_t* w1_rsrc)
{
    ckc_ir_builder_t* b;
    const ckc_deep_fused_conv_pool_spec_t* spec;
    const ckc_mma_op_t* op;
    bool defer;
    const ckc_conv_acc_epilogue_t* deferred_epi;
    ckc_status_t st;
    size_t i;

    if(ctx == NULL || ctx->b == NULL)
    {
        return;
    }
    b = ctx->b;
    spec = ctx->spec;
    op = ctx->op;

    /* stage per-callback scratch onto the ctx */
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->y_rsrc = y_rsrc;
    ctx->w1_rsrc = w1_rsrc;
    ctx->num_conv0_accs = 0;
    for(i = 0; i < num_accs && i < (size_t)CKC_DFCP_MAX_ACCS; ++i)
    {
        ctx->conv0_accs[i] = accs[i];
    }
    ctx->num_conv0_accs
        = (num_accs < (size_t)CKC_DFCP_MAX_ACCS) ? num_accs : (size_t)CKC_DFCP_MAX_ACCS;

    /* Barrier-merge: stage conv0 accs + W1 to disjoint LDS without per-producer
     * barriers, then a single block-wide barrier gates the conv1 consumer. */
    ctx->c_smem
        = ckc_dfcp_stage_accumulators_to_cshuffle_lds(b, op, accs, num_accs, grid, /*sync=*/false);
    ctx->w1_smem = ckc_dfcp_load_conv1_weights_to_lds(b,
                                                      spec,
                                                      w1_rsrc,
                                                      grid,
                                                      /*sync=*/false);
    ckc_b_sync(b);

    /* VALU opt: monotonic conv1 epilogue commutes with maxpool -> defer it. */
    defer = ckc_dfcp_epilogue_is_pool_deferrable(&spec->conv1_epilogue);
    ctx->defer = defer;

    st = ckc_dfcp_emit_conv1_1x1(b,
                                 spec,
                                 conv_spec_,
                                 op,
                                 ctx->c_smem,
                                 ctx->w1_smem,
                                 grid,
                                 /*defer_epilogue=*/defer,
                                 ctx->conv1_accs,
                                 (size_t)CKC_DFCP_MAX_ACCS,
                                 &ctx->num_conv1_accs);
    if(st != CKC_OK)
    {
        /* error already routed through b by the emit helper */
        return;
    }

    deferred_epi = defer ? &spec->conv1_epilogue : NULL;
    ctx->deferred_epi = deferred_epi;

    if(ckc_dfcp_maxpool_is_intra_lane(spec, grid))
    {
        /* Handoff eliminated: each lane's vec<16> conv1 acc already holds its 4
         * pool windows (intra-lane). Reduce straight to global output. */
        ckc_dfcp_emit_inline_maxpool_from_registers(
            b, spec, ctx->conv1_accs, ctx->num_conv1_accs, y_rsrc, grid, deferred_epi);
    }
    else if(ckc_dfcp_maxpool_is_intra_lane_wmma(spec, grid, op))
    {
        /* RDNA4 analogue: 2x2 corners live in the same lane across the two
         * adjacent m-tile accs -> skip the cshuffle LDS handoff. */
        ckc_dfcp_emit_wmma_maxpool_from_registers(
            b, spec, ctx->conv1_accs, ctx->num_conv1_accs, y_rsrc, grid, op, deferred_epi);
    }
    else
    {
        ckc_value_t* conv1_smem = ckc_dfcp_stage_accumulators_to_cshuffle_lds(
            b, op, ctx->conv1_accs, ctx->num_conv1_accs, grid, /*sync=*/true);
        ctx->conv1_smem = conv1_smem;
        ckc_dfcp_emit_inline_maxpool_from_cshuffle(b, spec, conv1_smem, y_rsrc, grid, deferred_epi);
    }
}

/* ===================================================================== *
 * Conv-builder callback trampolines.
 *
 * The conv builder calls the override hooks with its own ABI
 * (ckc_dfcp_conv_overrides_t) + an opaque `user` cookie. We thread the shared
 * ckc_dfcp_build_ctx_t through `user` and forward to the ctx phases above. The
 * conv builder's opaque extra_context / input_cache_context Values are the
 * Values our setup_* phases return; we round-trip them as void*.
 * ===================================================================== */

static void* ckc_dfcp_tramp_extra_params(ckc_ir_builder_t* b, void* user)
{
    ckc_dfcp_build_ctx_t* ctx = (ckc_dfcp_build_ctx_t*)user;
    (void)b;
    return (void*)ckc_dfcp_extra_params(ctx);
}

static ckc_value_t* ckc_dfcp_tramp_m_index_fn(ckc_ir_builder_t* b,
                                              ckc_value_t* row,
                                              const ckc_warp_grid_t* grid,
                                              void* user)
{
    ckc_dfcp_build_ctx_t* ctx = (ckc_dfcp_build_ctx_t*)user;
    (void)b;
    return ckc_dfcp_m_index_fn(ctx, row, grid);
}

static void ckc_dfcp_tramp_a_mhw_index_fn(ckc_ir_builder_t* b,
                                          ckc_value_t* row,
                                          const ckc_warp_grid_t* grid,
                                          ckc_value_t** out_n,
                                          ckc_value_t** out_ho,
                                          ckc_value_t** out_wo,
                                          void* user)
{
    ckc_dfcp_build_ctx_t* ctx = (ckc_dfcp_build_ctx_t*)user;
    (void)b;
    ckc_dfcp_a_mhw_index_fn(ctx, row, grid, out_n, out_ho, out_wo);
}

static void* ckc_dfcp_tramp_input_cache_setup(ckc_ir_builder_t* b,
                                              const ckc_implicit_gemm_conv_spec_t* spec,
                                              const ckc_warp_grid_t* grid,
                                              ckc_value_t* a_rsrc,
                                              void* user)
{
    ckc_dfcp_build_ctx_t* ctx = (ckc_dfcp_build_ctx_t*)user;
    (void)b;
    /* select setup_input_cache vs setup_specialized_a_loader by the same gate
     * the driver wired (Python's input_cache_setup= ternary). */
    if(ctx->use_input_cache)
    {
        return (void*)ckc_dfcp_setup_input_cache(ctx, spec, grid, a_rsrc);
    }
    return (void*)ckc_dfcp_setup_specialized_a_loader(ctx, spec, grid, a_rsrc);
}

static void ckc_dfcp_tramp_a_load_override(ckc_ir_builder_t* b,
                                           const ckc_implicit_gemm_conv_spec_t* spec,
                                           ckc_value_t* k_off,
                                           ckc_value_t* a_dst,
                                           const ckc_warp_grid_t* grid,
                                           void* input_cache_context,
                                           void* user)
{
    ckc_dfcp_build_ctx_t* ctx = (ckc_dfcp_build_ctx_t*)user;
    ckc_value_t* cache = (ckc_value_t*)input_cache_context;
    (void)b;
    if(ctx->use_input_cache)
    {
        ckc_dfcp_load_a_tile_from_cache(ctx, spec, k_off, a_dst, grid, cache);
    }
    else
    {
        ckc_dfcp_load_a_tile_specialized(ctx, spec, k_off, a_dst, grid, cache);
    }
}

static ckc_value_t* ckc_dfcp_tramp_a_operand_override(ckc_ir_builder_t* b,
                                                      const ckc_implicit_gemm_conv_spec_t* spec,
                                                      ckc_value_t* a_row,
                                                      ckc_value_t* k_off,
                                                      ckc_value_t* col_base,
                                                      int a_per_lane,
                                                      const ckc_warp_grid_t* grid,
                                                      void* input_cache_context,
                                                      void* user)
{
    ckc_dfcp_build_ctx_t* ctx = (ckc_dfcp_build_ctx_t*)user;
    ckc_value_t* cache = (ckc_value_t*)input_cache_context;
    (void)b;
    return ckc_dfcp_load_a_operand_from_cache(
        ctx, spec, a_row, k_off, col_base, a_per_lane, grid, cache);
}

static void ckc_dfcp_tramp_epilogue_override(ckc_ir_builder_t* b,
                                             const ckc_implicit_gemm_conv_spec_t* spec,
                                             ckc_value_t* const* accs,
                                             int num_accs,
                                             const ckc_warp_grid_t* grid,
                                             ckc_value_t* d_rsrc,
                                             void* extra_context,
                                             void* user)
{
    ckc_dfcp_build_ctx_t* ctx = (ckc_dfcp_build_ctx_t*)user;
    ckc_value_t* w1_rsrc = (ckc_value_t*)extra_context;
    (void)b;
    ckc_dfcp_epilogue_override(
        ctx, spec, accs, (num_accs < 0) ? 0u : (size_t)num_accs, grid, d_rsrc, w1_rsrc);
}

/* ===================================================================== *
 * ckc_build_deep_fused_conv_pool -- the public driver.
 *
 * build_deep_fused_conv_pool(spec, arch="gfx950"):
 *   ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...)
 *   conv_spec = spec.conv_spec()
 *   op = _resolve_conv_op(conv_spec, arch)
 *   <define the nine nested closures>
 *   return build_implicit_gemm_conv(conv_spec, arch=arch,
 *       extra_params=..., m_index_fn=..., a_mhw_index_fn=...,
 *       input_cache_setup=(...), a_load_override=(...),
 *       a_operand_override=(...), epilogue_override=...)
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_deep_fused_conv_pool(ckc_ir_builder_t* b_unused,
                                                 const ckc_deep_fused_conv_pool_spec_t* spec,
                                                 const char* arch)
{
    ckc_ir_builder_t* b = b_unused; /* the surface this routine emits into */
    char reason[256];
    const ckc_implicit_gemm_conv_spec_t* conv_spec;
    const ckc_mma_op_t* op;
    ckc_dfcp_build_ctx_t ctx;
    ckc_dfcp_conv_overrides_t ov;
    ckc_status_t st;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950"; /* arch default */
    }

    /* 1. is_valid_spec gate -> ValueError on reject. */
    reason[0] = '\0';
    if(!ckc_deep_fused_conv_pool_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        ckc_i_set_err(
            b, CKC_ERR_VALUE, "invalid deep fused conv/pool spec for %s: %s", arch, reason);
        return NULL;
    }

    /* 2. conv_spec = spec.conv_spec() */
    conv_spec = ckc_deep_fused_conv_pool_spec_conv_spec(b, spec);
    if(conv_spec == NULL)
    {
        return NULL; /* error routed through b */
    }

    /* 3. op = _resolve_conv_op(conv_spec, arch) */
    op = ckc_conv_resolve_op(b, conv_spec, arch);
    if(op == NULL)
    {
        return NULL; /* error routed through b */
    }

    /* 4. populate the shared closure ctx (build-time constants + routing). */
    st = ckc_dfcp_build_ctx_init(&ctx, b, spec, arch, conv_spec, op);
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
    ov.user = &ctx;
    ov.extra_params = ckc_dfcp_tramp_extra_params;
    ov.m_index_fn = ckc_dfcp_tramp_m_index_fn;
    ov.a_mhw_index_fn = ckc_dfcp_tramp_a_mhw_index_fn;

    if(ctx.use_input_cache || ctx.use_specialized)
    {
        ov.input_cache_setup = ckc_dfcp_tramp_input_cache_setup;
        ov.a_load_override = ckc_dfcp_tramp_a_load_override;
    }
    if(ctx.use_operand_ovr)
    {
        ov.a_operand_override = ckc_dfcp_tramp_a_operand_override;
    }
    ov.epilogue_override = ckc_dfcp_tramp_epilogue_override;

    /* 6. drive the wrapped conv0 builder; return its kernel. The staged `ov`
     * mirror is layout-identical to the canonical ckc_conv_build_overrides_t. */
    return ckc_build_implicit_gemm_conv(b, conv_spec, arch, (const ckc_conv_build_overrides_t*)&ov);
}

/* Convenience: init `b` with spec.kernel_name(), then build. */
ckc_kernel_def_t* ckc_build_deep_fused_conv_pool_new(ckc_ir_builder_t* b,
                                                     const ckc_deep_fused_conv_pool_spec_t* spec,
                                                     const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_deep_fused_conv_pool_spec_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_deep_fused_conv_pool(b, spec, arch);
    });
}

/* ===================================================================== *
 * ckc_deep_fused_conv_pool_lower_to_llvm -- build + lower to .ll convenience.
 * Owns and frees its own IRBuilder (mirrors the sibling instance ports).
 * ===================================================================== */
ckc_status_t ckc_deep_fused_conv_pool_lower_to_llvm(const ckc_deep_fused_conv_pool_spec_t* spec,
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
        arch = "gfx950";
    }

    kernel = ckc_build_deep_fused_conv_pool_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(&b);
            if(m == NULL)
            {
                m = "build_deep_fused_conv_pool failed";
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
