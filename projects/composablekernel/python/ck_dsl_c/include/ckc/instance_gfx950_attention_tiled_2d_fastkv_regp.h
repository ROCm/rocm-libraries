/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_gfx950_attention_tiled_2d_fastkv_regp.h -- canonical public
 * surface for the C99 port of
 * ck_dsl/instances/gfx950/attention_tiled_2d_fastkv_regp.py: the experimental
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
 * ckc_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_t make_fastkv_register_p_spec(...)
 * ckc_gfx950_make_fastkv_register_p_spec supports_fastkv_register_p_2d(...)
 * ckc_gfx950_supports_fastkv_register_p_2d build_unified_attention_2d_fastkv_register_p(...)
 * ckc_build_unified_attention_2d_fastkv_register_p
 *
 *   (+ convenience: build -> lower .ll) ckc_gfx950_attention_tiled_2d_fastkv_regp_lower_to_llvm
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
 *   ckc_build_unified_attention_2d_fastkv_register_p(
 *       ckc_ir_builder_t* b,
 *       const ckc_attention_tiled_2d_spec_t* spec,
 *       const char* arch) -> ckc_kernel_def_t*
 *
 *   It validates spec.use_fast_paged_kv_desc, then
 *   (spec.use_mfma_32x32 && spec.use_transposed_qk_32x32), then
 *   !spec.kv_storage_dtype, wraps spec in a proxy that overrides
 *   use_register_pv = true, and calls
 *   ckc_build_unified_attention_2d_tiled_scalar(b, &proxy_spec, arch).
 *   gfx950 arch only; ``arch`` NULL == "gfx950".
 *
 * Error model mirrors the rest of the C port: pure helpers return a sentinel;
 * the builder routes the first failure through the sticky-error IRBuilder; the
 * convenience lower returns a ckc_status_t. Every IR node is arena-owned.
 */
#ifndef CKC_INSTANCE_GFX950_ATTENTION_TILED_2D_FASTKV_REGP_H
#define CKC_INSTANCE_GFX950_ATTENTION_TILED_2D_FASTKV_REGP_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"         /* ckc_ir_builder_t, ckc_kernel_def_t, ckc_status_t */
#include "ckc/lower_llvm.h" /* ckc_llvm_flavor_t                                 */

/* Re-export the four task symbols (proxy type + make_spec + supports + build).
 * Single authoritative definition; this header does not redeclare them. */
#include "ckc/helper_instance_gfx950_attention_tiled_2d_fastkv_regp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 * build -> lower convenience.
 * ============================================================ *
 *
 * Given a fastKV register-P spec, init an internally-owned IRBuilder via the
 * spec's kernel_name(), build the kernel through
 * ckc_build_unified_attention_2d_fastkv_register_p, then lower to LLVM .ll text.
 *
 * ``arch`` NULL == "gfx950" (the experiment is gfx950-only, threaded straight
 * through to the tiled builder so a non-gfx950 request fails with the same clean
 * structured error). On CKC_OK *out_ll receives a malloc'd NUL-terminated string
 * the caller frees with free(); on failure it is left NULL and (if err != NULL,
 * capacity err_cap) a diagnostic is written. Internally owns and frees its
 * IRBuilder. */
ckc_status_t
ckc_gfx950_attention_tiled_2d_fastkv_regp_lower_to_llvm(const ckc_attention_tiled_2d_spec_t* spec,
                                                        const char* arch,
                                                        ckc_llvm_flavor_t flavor,
                                                        char** out_ll,
                                                        char* err,
                                                        size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_GFX950_ATTENTION_TILED_2D_FASTKV_REGP_H */
