/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_deep_fused_conv_pool.h -- PUBLIC C99 API for the family-agnostic
 * deep-fused conv0 -> conv1 -> maxpool kernel instance builder ported from
 * rocke/instances/common/deep_fused_conv_pool.py (1494 LOC).
 *
 * One CTA owns one pooled-output tile and fuses the whole stack:
 *
 *   implicit-GEMM conv0 -> accumulator epilogue -> LDS C-shuffle
 *   -> 1x1 conv1 -> LDS C-shuffle -> maxpool -> Y (NHWK)
 *
 * The numeric core is authored once and driven by the resolved MmaOp, so the
 * same body emits the wave64 MFMA 32x32x16 path (gfx950) and the wave32 WMMA
 * 16x16x16 path (gfx1201) with no per-family branching.
 *
 *   Python (deep_fused_conv_pool.py)        C99 (this header)
 *   -------------------------------------   --------------------------------------
 *   FusedConvPoolProblem (frozen)           rocke_fused_conv_pool_problem_t  (helper hdr)
 *   DeepFusedConvPoolSpec (frozen)          rocke_deep_fused_conv_pool_spec_t (helper hdr)
 *   make_deep_fused_conv_pool_spec(...)     rocke_make_deep_fused_conv_pool_spec(...) (helper hdr)
 *   is_valid_spec(spec, arch)               rocke_deep_fused_conv_pool_is_valid_spec(...) (helper
 * hdr) deep_fused_conv_pool_signature(spec)    rocke_deep_fused_conv_pool_signature(...) (helper hdr)
 *   deep_fused_conv_pool_grid(spec)         rocke_deep_fused_conv_pool_grid(...) (helper hdr)
 *   build_deep_fused_conv_pool(spec, arch)  rocke_build_deep_fused_conv_pool(...)   <-- THIS HEADER
 *   (+ convenience: build -> lower .ll)     rocke_deep_fused_conv_pool_lower_to_llvm(...)
 *
 * The value types (FusedConvPoolProblem / DeepFusedConvPoolSpec / the
 * ConvAccumulatorEpilogue slice), the spec factory + property accessors, the
 * validity gate, the signature, the grid, and the 22 builder-emit / pure phase
 * helpers all live in
 *   rocke/helper_rocke.instances.common.deep_fused_conv_pool.h
 * and are reused here. This header adds ONLY the public build entry that wires
 * the phases in Python `build_deep_fused_conv_pool` order, plus a lower-to-.ll
 * convenience that mirrors the other instance ports.
 *
 * PEER PORTS THE BUILD ENTRY DEPENDS ON (not yet public C symbols):
 *   - build_implicit_gemm_conv / _resolve_conv_op (conv_implicit_gemm.py): the
 *     conv0 driver this instance wraps. Until those land as public C symbols the
 *     build entry forward-declares them via the opaque rocke_implicit_gemm_conv_spec_t
 *     + rocke_mma_op_t handles (see the internal header) and routes the override
 *     callbacks through the build-context; the call site is documented below.
 *   - WarpGrid (geometry.py), LdsLayout (layouts.py), CoalescedTileLoader
 *     (loads.py), distribution.py producers: consumed only inside the emit
 *     helpers (helper header), not the public surface.
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (rocke_b_*); the validity gate (helper header)
 * returns a bool + reason; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_DEEP_FUSED_CONV_POOL_H
#define ROCKE_INSTANCE_DEEP_FUSED_CONV_POOL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/helper_rocke.instances.common.deep_fused_conv_pool.h" /* spec/problem/epilogue + emit helpers */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * Public build entry -- build_deep_fused_conv_pool(spec, arch)
 * ------------------------------------------------------------------ *
 *
 * Builds the one-CTA conv0 -> conv1 -> maxpool fused kernel into the supplied
 * builder `b` and returns the kernel (b->kernel) on success or NULL with b's
 * sticky error set. `arch` NULL => "gfx950". The first argument is named
 * `b_unused` to flag that, like the Python, the caller-supplied builder is the
 * surface this routine emits into via the wrapped conv driver; it is NOT
 * re-initialised here (the caller owns its lifetime and frees it with
 * rocke_ir_builder_free()).
 *
 * CALL PATTERN (byte-faithful to Python build_deep_fused_conv_pool):
 *   1. rocke_deep_fused_conv_pool_is_valid_spec(spec, arch, ...) gate
 *      -> NULL + ROCKE_ERR_VALUE (ValueError "invalid deep fused conv/pool spec")
 *         when invalid.
 *   2. Resolve conv_spec from spec.conv_spec()  (ctx->conv_spec).
 *   3. Resolve op via _resolve_conv_op(conv_spec, arch)  (ctx->op).
 *   4. Register the W1/W1_bytes extra params (the extra_params closure ->
 *      rocke_dfcp_extra_params phase) producing the W1 buffer resource.
 *   5. Build implicit_gemm_conv with an epilogue_override that:
 *        - stages conv0 accs to cshuffle LDS (sync=False)
 *          via rocke_dfcp_stage_accumulators_to_cshuffle_lds,
 *        - loads W1 weights to LDS (sync=False)
 *          via rocke_dfcp_load_conv1_weights_to_lds,
 *        - emits a single block-wide barrier (rocke_b_sync),
 *        - computes deferral via rocke_dfcp_epilogue_is_pool_deferrable,
 *        - emits conv1 1x1 GEMM via rocke_dfcp_emit_conv1_1x1
 *          (with deferred epilogue when pool-deferrable),
 *        - routes to the register-resident maxpool
 *          (rocke_dfcp_emit_inline_maxpool_from_registers for the MFMA-32x32 fast
 *           path gated by rocke_dfcp_maxpool_is_intra_lane, or
 *           rocke_dfcp_emit_wmma_maxpool_from_registers for the WMMA fast path
 *           gated by rocke_dfcp_maxpool_is_intra_lane_wmma) OR the generic
 *          cshuffle-LDS maxpool (stage conv1 accs +
 *          rocke_dfcp_emit_inline_maxpool_from_cshuffle).
 *      and the conv0 A-load overrides selected from the spec flags
 *      (cache_input_footprint / direct_conv0_from_input_cache /
 *       rocke_dfcp_can_use_specialized_conv0_a_loader): input_cache_setup,
 *      a_load_override, a_operand_override + the m_index_fn / a_mhw_index_fn
 *      coord callbacks.
 *   6. Return the kernel from the wrapped build_implicit_gemm_conv call.
 *
 * The override/coord callbacks above are the Python nested closures; the C port
 * realises them as free phase functions over the shared build-context declared
 * in rocke/instance_deep_fused_conv_pool_internal.h. The closures' captured locals
 * (spec, conv_spec, op, the W1 resource, the deferral flag) are exactly the ctx
 * fields, and this entry populates the ctx then drives the wrapped conv builder.
 */
rocke_kernel_def_t* rocke_build_deep_fused_conv_pool(rocke_ir_builder_t* b_unused,
                                                     const rocke_deep_fused_conv_pool_spec_t* spec,
                                                     const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL with
 * b's sticky error set. Mirrors the *_new convenience the other instance ports
 * expose. */
rocke_kernel_def_t* rocke_build_deep_fused_conv_pool_new(
    rocke_ir_builder_t* b, const rocke_deep_fused_conv_pool_spec_t* spec, const char* arch);

/* ------------------------------------------------------------------ *
 * Convenience: build -> lower to LLVM .ll text
 * ------------------------------------------------------------------ *
 *
 * Given a spec, init a builder with spec.kernel_name(), build, and lower to LLVM
 * .ll text. `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd
 * NUL-terminated string the caller frees with free(); on failure it is left NULL
 * and (if err != NULL, capacity err_cap) a diagnostic is written. Internally
 * owns and frees its IRBuilder. */
rocke_status_t
    rocke_deep_fused_conv_pool_lower_to_llvm(const rocke_deep_fused_conv_pool_spec_t* spec,
                                             const char* arch,
                                             rocke_llvm_flavor_t flavor,
                                             char** out_ll,
                                             char* err,
                                             size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_DEEP_FUSED_CONV_POOL_H */
