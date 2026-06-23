/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_fmha_bwd.h -- C99 port of the FMHA backward-pass kernel instance
 * builder ck_dsl/instances/common/fmha_bwd.py.
 *
 * Computes the Flash-Attention backward gradients dQ / dK / dV with a
 * warp-distributed scalar body (one wave64 warp per CTA; lane t owns head-dim
 * slot t*EPT+k). The recompute-from-(M,L) path issues NO MFMA atoms and
 * accumulates the three gradients through f32 global atomics, so the emitted IR
 * is arch-portable; `arch` only feeds is_valid_spec's per-WG thread / wave-size
 * validation against ckc_arch_target.
 *
 *   Python (fmha_bwd.py)                   C99 (this header)
 *   ------------------------------------   --------------------------------------
 *   @dataclass(frozen=True) FmhaBwdSpec    ckc_fmha_bwd_spec_t
 *     .kernel_name()                       ckc_fmha_bwd_kernel_name(...)
 *   is_valid_spec(spec, arch)              ckc_fmha_bwd_is_valid_spec(...)
 *   build_fmha_bwd(spec, arch)             ckc_build_fmha_bwd(...)
 *   fmha_bwd_grid(spec)                    ckc_fmha_bwd_grid(...)
 *   fmha_bwd_signature(spec)               ckc_fmha_bwd_signature(...)
 *   (+ convenience: build -> lower .ll)    ckc_fmha_bwd_lower_to_llvm(...)
 *
 * The build-entry mirrors the gemm_universal.c pattern: the FmhaKernelBuilder
 * owns its own ckc_ir_builder_t, so ckc_build_fmha_bwd takes a caller-supplied
 * (uninitialised) ckc_fmha_kernel_builder_t whose lifetime the caller controls
 * (free with ckc_fmha_kernel_builder_free). On any invalid spec the build
 * returns NULL with the builder's sticky error set (read with
 * ckc_ir_builder_error on kb->b).
 *
 * Error model mirrors the rest of the C port: build routes through the
 * sticky-error IRBuilder; the validity gate returns a bool + a reason string;
 * the convenience lower returns a ckc_status_t.
 */
#ifndef CKC_INSTANCE_FMHA_BWD_H
#define CKC_INSTANCE_FMHA_BWD_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/helper_ck_dsl.instances.common._fmha_common.h"
#include "ckc/helper_ck_dsl.helpers.spec.h" /* ckc_sig_entry_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * FmhaBwdSpec
 * ------------------------------------------------------------------ *
 *
 * @dataclass(frozen=True)
 * class FmhaBwdSpec:
 *     common: FmhaCommonSpec
 *     seqlen_q: int
 *     seqlen_k: int
 *     name: str = "ck_dsl_fmha_bwd"
 *     use_mfma_body: bool = False
 *     output_grad_dtype: str = "f32"
 *
 * `name` / `output_grad_dtype` are referenced as-is (not copied); keep them
 * alive for the spec's use. The default `name` "ck_dsl_fmha_bwd" and
 * output_grad_dtype "f32" are filled by ckc_fmha_bwd_spec_default. */
typedef struct ckc_fmha_bwd_spec
{
    ckc_fmha_common_spec_t common;
    int seqlen_q;
    int seqlen_k;
    const char* name;              /* default "ck_dsl_fmha_bwd" */
    bool use_mfma_body;            /* default false             */
    const char* output_grad_dtype; /* default "f32"            */
} ckc_fmha_bwd_spec_t;

/* FmhaBwdSpec(common, seqlen_q, seqlen_k, name="ck_dsl_fmha_bwd",
 * use_mfma_body=False, output_grad_dtype="f32"): construct from the required
 * fields with the dataclass defaults for name / use_mfma_body /
 * output_grad_dtype. */
ckc_fmha_bwd_spec_t
ckc_fmha_bwd_spec_default(ckc_fmha_common_spec_t common, int seqlen_q, int seqlen_k);

/* FmhaBwdSpec.kernel_name(): join name + H{head_size} + HQ{nq} + HK{nk} +
 * dtype + Q{seqlen_q} + K{seqlen_k} + mask_mode via kernel_name_join. Writes
 * NUL-terminated into `out` (capacity out_cap). Returns CKC_OK or CKC_ERR_VALUE
 * (buffer too small). */
ckc_status_t ckc_fmha_bwd_kernel_name(const ckc_fmha_bwd_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message; returns false. On accept returns true and writes "ok". */
bool ckc_fmha_bwd_is_valid_spec(const ckc_fmha_bwd_spec_t* spec,
                                const char* arch,
                                char* reason,
                                size_t reason_cap);

/* build_fmha_bwd(spec, arch). Builds the IR into the supplied (uninitialised)
 * FmhaKernelBuilder `kb`, exactly as the Python build does, and returns the
 * kernel (kb->b.kernel) on success or NULL with kb->b's sticky error set.
 * `arch` NULL => "gfx950". The caller owns `kb` and frees it with
 * ckc_fmha_kernel_builder_free. On an invalid spec returns NULL (the reason is
 * recorded on kb->b's sticky error). */
ckc_kernel_def_t* ckc_build_fmha_bwd(ckc_fmha_kernel_builder_t* kb,
                                     const ckc_fmha_bwd_spec_t* spec,
                                     const char* arch);

/* fmha_bwd_grid(spec) -> (seqlen_q, num_query_heads, 1). */
void ckc_fmha_bwd_grid(const ckc_fmha_bwd_spec_t* spec, int* gx, int* gy, int* gz);

/* fmha_bwd_signature(spec): declare the ABI on a throwaway FmhaKernelBuilder
 * and emit the SignatureBuilder shape. The caller supplies an uninitialised
 * `kb` (owns + frees it) and an `arena` backing the signature storage. On
 * CKC_OK *out_items / *out_count hold the arena-owned array. */
ckc_status_t ckc_fmha_bwd_signature(ckc_fmha_kernel_builder_t* kb,
                                    const ckc_fmha_bwd_spec_t* spec,
                                    ckc_arena_t* arena,
                                    const ckc_sig_entry_t** out_items,
                                    size_t* out_count);

/* Convenience: given a spec, build the FMHA backward kernel and lower to LLVM
 * .ll text. `arch` NULL => "gfx950". On CKC_OK *out_ll receives a malloc'd
 * NUL-terminated string the caller frees with free(); on failure it is left
 * NULL and (if err!=NULL, capacity err_cap) a diagnostic is written. Internally
 * owns and frees its FmhaKernelBuilder. */
ckc_status_t ckc_fmha_bwd_lower_to_llvm(const ckc_fmha_bwd_spec_t* spec,
                                        const char* arch,
                                        ckc_llvm_flavor_t flavor,
                                        char** out_ll,
                                        char* err,
                                        size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_FMHA_BWD_H */
