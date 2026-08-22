/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_fmha_varlen.h -- C99 port of the variable-length FMHA forward
 * kernel instance builder rocke/instances/common/fmha_varlen.py (CK Tile
 * 01_fmha varlen parity).
 *
 *   Python (fmha_varlen.py)               C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   @dataclass FmhaFwdVarlenSpec          rocke_fmha_fwd_varlen_spec_t
 *     .kernel_name()                      rocke_fmha_fwd_varlen_kernel_name(...)
 *   is_valid_spec(spec, arch)             rocke_fmha_fwd_varlen_is_valid_spec(...)
 *   build_fmha_fwd_varlen(spec, arch)     rocke_build_fmha_fwd_varlen(...)
 *   fmha_fwd_varlen_grid(spec, total_q)   rocke_fmha_fwd_varlen_grid(...)
 *   (+ convenience: build -> lower .ll)   rocke_fmha_fwd_varlen_lower_to_llvm(...)
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python FmhaFwdVarlenSpec is a frozen value
 * type wrapping an FmhaCommonSpec (ported in
 * rocke/helper_rocke.instances.common._fmha_common.h) plus the three varlen
 * scalars (max_seqlen_q, max_seqlen_k, batch) and the kernel-name prefix. The
 * caller fills a rocke_fmha_fwd_varlen_spec_t; rocke_fmha_fwd_varlen_spec_default()
 * returns a struct with the name prefix at the Python dataclass default.
 *
 * BUILDER-CALL FIDELITY. rocke_build_fmha_fwd_varlen reproduces the Python
 * build_fmha_fwd_varlen op-for-op: FmhaKernelBuilder init, block_size(64),
 * the verbatim _declare_params() ABI, decode_grid(), the Python-unrolled
 * cu_seqlens_q scan (batch compares + cmovs), the per-sequence K base/offset
 * lookups, then mfma_attention_fwd_inner_body() and b.ret().
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the FmhaKernelBuilder's embedded sticky-error IRBuilder; the validity gate
 * returns a bool + a reason string; the convenience lower returns a status.
 */
#ifndef ROCKE_INSTANCE_FMHA_VARLEN_H
#define ROCKE_INSTANCE_FMHA_VARLEN_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common._fmha_common.h" /* rocke_fmha_common_spec_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------- FmhaFwdVarlenSpec *
 *
 * @dataclass(frozen=True)
 * class FmhaFwdVarlenSpec:
 *     common: FmhaCommonSpec
 *     max_seqlen_q: int
 *     max_seqlen_k: int
 *     batch: int
 *     name: str = "rocke_fmha_fwd_varlen"
 *
 * `name` is referenced as-is (not copied); keep it alive for the spec's use.
 * NULL is treated as the dataclass default "rocke_fmha_fwd_varlen". */
typedef struct rocke_fmha_fwd_varlen_spec
{
    rocke_fmha_common_spec_t common;
    int max_seqlen_q;
    int max_seqlen_k;
    int batch;
    const char* name; /* default "rocke_fmha_fwd_varlen" */
} rocke_fmha_fwd_varlen_spec_t;

/* FmhaFwdVarlenSpec(common, max_seqlen_q, max_seqlen_k, batch): construct with
 * the dataclass default name. */
rocke_fmha_fwd_varlen_spec_t rocke_fmha_fwd_varlen_spec_default(rocke_fmha_common_spec_t common,
                                                                int max_seqlen_q,
                                                                int max_seqlen_k,
                                                                int batch);

/* FmhaFwdVarlenSpec.kernel_name():
 *   kernel_name_join(name, "H{head_size}", "HQ{nq}", "HK{nkv}", dtype,
 *                    "Q{max_seqlen_q}", "K{max_seqlen_k}", "B{batch}",
 *                    mask_mode)
 * Written NUL-terminated into `out` (capacity out_cap). Returns ROCKE_OK or
 * ROCKE_ERR_VALUE (buffer too small). */
rocke_status_t rocke_fmha_fwd_varlen_kernel_name(const rocke_fmha_fwd_varlen_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message and returns false; on accept returns true and writes "ok".
 *
 * Checks (Python order):
 *   1. validate_common_spec(spec.common)
 *   2. validate_fmha_mfma_atom(spec.common.dtype, arch)
 *   3. batch > 0
 *   4. max_seqlen_q > 0 and max_seqlen_k > 0
 *   5. max_seqlen_q % MFMA_ATTN_BLOCK_M == 0
 *   6. max_seqlen_k % MFMA_ATTN_BLOCK_K == 0
 *   7. head_size % 16 == 0 */
bool rocke_fmha_fwd_varlen_is_valid_spec(const rocke_fmha_fwd_varlen_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap);

/* build_fmha_fwd_varlen(spec, arch="gfx950"). Builds the varlen FMHA-fwd
 * (MFMA-tiled body) IR and returns the kernel, or NULL with an error written
 * (if err != NULL, capacity err_cap). `arch` NULL => "gfx950".
 *
 * The kernel is owned by `*out_kb` (a caller-provided FmhaKernelBuilder whose
 * embedded IR builder owns the nodes); the caller frees it with
 * rocke_fmha_kernel_builder_free(out_kb). This mirrors the Python returning
 * `kb.kernel` while `kb` retains ownership. */
rocke_kernel_def_t* rocke_build_fmha_fwd_varlen(rocke_fmha_kernel_builder_t* out_kb,
                                                const rocke_fmha_fwd_varlen_spec_t* spec,
                                                const char* arch,
                                                char* err,
                                                size_t err_cap);

/* fmha_fwd_varlen_grid(spec, total_q) -> (total_q // BLOCK_M, num_query_heads,
 * 1). Writes the three components into out_grid[0..2]. */
void rocke_fmha_fwd_varlen_grid(const rocke_fmha_fwd_varlen_spec_t* spec,
                                int total_q,
                                int out_grid[3]);

/* Convenience: given a spec, build and lower to LLVM .ll text. `arch` NULL =>
 * "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated string the
 * caller frees with free(); on failure it is left NULL and (if err != NULL,
 * capacity err_cap) a diagnostic is written. Internally owns and frees its
 * FmhaKernelBuilder. */
rocke_status_t rocke_fmha_fwd_varlen_lower_to_llvm(const rocke_fmha_fwd_varlen_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_FMHA_VARLEN_H */
