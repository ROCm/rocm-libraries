/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx950_attention_tiled_2d_fastkv_regp.h -- canonical public
 * surface for the C99 port of
 * rocke/instances/gfx950/attention_tiled_2d_fastkv_regp.py: the experimental
 * gfx950 "fast paged-KV descriptor + register-P" 2D-attention wrapper that
 * forces the production tiled R4 builder down the register-P residency path
 * (removing the otherwise-unused P_lds slab for the transposed 32x32 R4
 * dataflow).
 *
 * Ported symbols (task list -> module instance_gfx950_attention_tiled_2d_fastkv_regp):
 *
 *   Python (attention_tiled_2d_fastkv_regp.py)        C99
 *   -----------------------------------------------
 * ------------------------------------------------ class _FastKvRegisterPProxy
 * rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_t make_fastkv_register_p_spec(...)
 * rocke_gfx950_make_fastkv_register_p_spec supports_fastkv_register_p_2d(...)
 * rocke_gfx950_supports_fastkv_register_p_2d build_unified_attention_2d_fastkv_register_p(...)
 * rocke_build_unified_attention_2d_fastkv_register_p
 *
 *   (+ convenience: build -> lower .ll) rocke_gfx950_attention_tiled_2d_fastkv_regp_lower_to_llvm
 *
 * The four task symbols are defined in the byte-identical-call helper translation
 * unit (helper_instance_gfx950_attention_tiled_2d_fastkv_regp.{h,c}); this
 * canonical header RE-EXPORTS them by include (a single authoritative definition,
 * no duplicate symbols at link time) and adds the build->lower convenience entry,
 * mirroring the gfx942 tiled-2D instance header.
 *
 * The build entry's signature, validation order and builder-delegate are exactly
 * as the task specifies:
 *
 *   rocke_build_unified_attention_2d_fastkv_register_p(
 *       rocke_ir_builder_t* b,
 *       const rocke_attention_tiled_2d_spec_t* spec,
 *       const char* arch) -> rocke_kernel_def_t*
 *
 *   It validates spec.use_fast_paged_kv_desc, then
 *   (spec.use_mfma_32x32 && spec.use_transposed_qk_32x32), then
 *   !spec.kv_storage_dtype, wraps spec in a proxy that overrides
 *   use_register_pv = true, and calls
 *   rocke_build_unified_attention_2d_tiled_scalar(b, &proxy_spec, arch).
 *   gfx950 arch only; ``arch`` NULL == "gfx950".
 *
 * Error model mirrors the rest of the C port: pure helpers return a sentinel;
 * the builder routes the first failure through the sticky-error IRBuilder; the
 * convenience lower returns a rocke_status_t. Every IR node is arena-owned.
 */
#ifndef ROCKE_INSTANCE_GFX950_ATTENTION_TILED_2D_FASTKV_REGP_H
#define ROCKE_INSTANCE_GFX950_ATTENTION_TILED_2D_FASTKV_REGP_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h" /* rocke_ir_builder_t, rocke_kernel_def_t, rocke_status_t */
#include "rocke/lower_llvm.h" /* rocke_llvm_flavor_t                                 */

/* Re-export the four task symbols (proxy type + make_spec + supports + build).
 * Single authoritative definition; this header does not redeclare them. */
#include "rocke/helper_instance_gfx950_attention_tiled_2d_fastkv_regp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 * build -> lower convenience.
 * ============================================================ *
 *
 * Given a fastKV register-P spec, init an internally-owned IRBuilder via the
 * spec's kernel_name(), build the kernel through
 * rocke_build_unified_attention_2d_fastkv_register_p, then lower to LLVM .ll text.
 *
 * ``arch`` NULL == "gfx950" (the experiment is gfx950-only, threaded straight
 * through to the tiled builder so a non-gfx950 request fails with the same clean
 * structured error). On ROCKE_OK *out_ll receives a malloc'd NUL-terminated string
 * the caller frees with free(); on failure it is left NULL and (if err != NULL,
 * capacity err_cap) a diagnostic is written. Internally owns and frees its
 * IRBuilder. */
rocke_status_t rocke_gfx950_attention_tiled_2d_fastkv_regp_lower_to_llvm(
    const rocke_attention_tiled_2d_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX950_ATTENTION_TILED_2D_FASTKV_REGP_H */
