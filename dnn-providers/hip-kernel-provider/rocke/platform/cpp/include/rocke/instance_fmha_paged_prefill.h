/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_fmha_paged_prefill.h -- C99 port of the paged-KV prefill FMHA
 * forward kernel instance builder
 * rocke/instances/common/fmha_paged_prefill.py.
 *
 * Paged-KV prefill FMHA differs from the contiguous varlen forward only in the
 * K / V layout: K / V live in a paged cache (num_blocks, block_size, HK, D) and
 * the per-sequence block_table[seq_idx, :] indirects each logical K-position to
 * its physical block. The kernel performs the FULL block-table indirection (not
 * just block_table[0]) so sequences spanning multiple non-contiguous physical
 * blocks work correctly.
 *
 *   Python (fmha_paged_prefill.py)             C99 (this header)
 *   ----------------------------------------   ----------------------------------
 *   FmhaFwdPagedPrefillSpec (frozen dataclass) rocke_fmha_fwd_paged_prefill_spec_t
 *     .kernel_name()                           rocke_fmha_fwd_paged_prefill_kernel_name
 *   is_valid_spec(spec, arch)                  rocke_fmha_fwd_paged_prefill_is_valid_spec
 *   build_fmha_fwd_paged_prefill(spec, arch)   rocke_build_fmha_fwd_paged_prefill
 *   fmha_fwd_paged_prefill_grid(spec,total_q)  rocke_fmha_fwd_paged_prefill_grid
 *   fmha_fwd_paged_prefill_signature(spec)     rocke_fmha_fwd_paged_prefill_signature
 *   (+ convenience: build -> lower .ll)        rocke_fmha_fwd_paged_prefill_lower_to_llvm
 *
 * The build entry mirrors the Python builder-call sequence op-for-op: it
 * constructs the FmhaKernelBuilder ABI, decodes the grid, emits the per-q_token
 * ceil(log2(batch+1))-iter binary search over cu_seqlens_q, then dispatches to
 * either the MFMA-tiled body (use_mfma_body=True) or the warp-distributed body,
 * threading the paged-row callbacks that hoist the kv-head term and emit the
 * block_table indirection (page_block_size validated power-of-two => lshr+land).
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder; the validity gate returns a bool + reason; the
 * convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_FMHA_PAGED_PREFILL_H
#define ROCKE_INSTANCE_FMHA_PAGED_PREFILL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/helper_rocke.instances.common._fmha_common.h" /* common spec + builder */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * FmhaFwdPagedPrefillSpec
 * ------------------------------------------------------------------ *
 *
 * @dataclass(frozen=True)
 * class FmhaFwdPagedPrefillSpec:
 *     common: FmhaCommonSpec
 *     page_block_size: int
 *     max_blocks_per_seq: int
 *     batch: int
 *     name: str = "rocke_fmha_fwd_paged_prefill"
 *     use_mfma_body: bool = False
 *
 * `name` is referenced as-is (not copied); keep it alive for the spec's use.
 * NULL `name` is treated as the dataclass default. */
typedef struct rocke_fmha_fwd_paged_prefill_spec
{
    rocke_fmha_common_spec_t common;
    int page_block_size;
    int max_blocks_per_seq;
    int batch;
    const char* name; /* default "rocke_fmha_fwd_paged_prefill" */
    bool use_mfma_body; /* default false */
} rocke_fmha_fwd_paged_prefill_spec_t;

/* FmhaFwdPagedPrefillSpec(common, page_block_size, max_blocks_per_seq, batch):
 * construct with the dataclass defaults (name / use_mfma_body). */
rocke_fmha_fwd_paged_prefill_spec_t rocke_fmha_fwd_paged_prefill_spec_default(
    rocke_fmha_common_spec_t common, int page_block_size, int max_blocks_per_seq, int batch);

/* FmhaFwdPagedPrefillSpec.kernel_name():
 *   kernel_name_join(name, f"H{head_size}", f"HQ{num_query_heads}",
 *       f"HK{num_kv_heads}", dtype, f"PG{page_block_size}", f"B{batch}",
 *       mask_mode)
 *
 * Writes the NUL-terminated name into `out` (capacity out_cap). Returns ROCKE_OK,
 * or ROCKE_ERR_VALUE on a too-small buffer / NULL args. */
rocke_status_t rocke_fmha_fwd_paged_prefill_kernel_name(
    const rocke_fmha_fwd_paged_prefill_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason).
 *
 * Python (fmha_paged_prefill.py:is_valid_spec):
 *   - validate_common_spec(spec.common)
 *   - if use_mfma_body: validate_fmha_mfma_atom(dtype, arch)
 *   - batch > 0
 *   - page_block_size > 0 and a power of two
 *   - max_blocks_per_seq > 0
 *
 * `arch` NULL => "gfx950". On reject, `reason` (if non-NULL, capacity
 * reason_cap) receives the message and false is returned; on accept returns
 * true and writes "ok". */
bool rocke_fmha_fwd_paged_prefill_is_valid_spec(const rocke_fmha_fwd_paged_prefill_spec_t* spec,
                                                const char* arch,
                                                char* reason,
                                                size_t reason_cap);

/* build_fmha_fwd_paged_prefill(spec, arch).
 *
 * Initialises `kb` with spec.kernel_name() and the common spec, declares the
 * paged-prefill ABI, decodes the grid, emits the binary search + paged
 * indirection + body dispatch, and emits ret. Returns the kernel (kb->b.kernel)
 * on success or NULL with the builder sticky error set. `arch` NULL => "gfx950".
 *
 * The caller owns `kb` and frees it with rocke_fmha_kernel_builder_free(). */
rocke_kernel_def_t*
    rocke_build_fmha_fwd_paged_prefill(rocke_fmha_kernel_builder_t* kb,
                                       const rocke_fmha_fwd_paged_prefill_spec_t* spec,
                                       const char* arch);

/* fmha_fwd_paged_prefill_grid(spec, total_q) -> (total_q, num_query_heads, 1).
 * Writes out[0..2]. Returns ROCKE_OK, or ROCKE_ERR_VALUE on NULL args. */
rocke_status_t rocke_fmha_fwd_paged_prefill_grid(const rocke_fmha_fwd_paged_prefill_spec_t* spec,
                                                 int total_q,
                                                 int out[3]);

/* fmha_fwd_paged_prefill_signature(spec): build a probe FmhaKernelBuilder,
 * declare the ABI, and return its signature. On ROCKE_OK *out_items / *out_count
 * hold the arena-owned array; `arena` backs the SignatureBuilder storage. */
rocke_status_t
    rocke_fmha_fwd_paged_prefill_signature(const rocke_fmha_fwd_paged_prefill_spec_t* spec,
                                           rocke_arena_t* arena,
                                           const rocke_sig_entry_t** out_items,
                                           size_t* out_count);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its FmhaKernelBuilder. */
rocke_status_t
    rocke_fmha_fwd_paged_prefill_lower_to_llvm(const rocke_fmha_fwd_paged_prefill_spec_t* spec,
                                               const char* arch,
                                               rocke_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_FMHA_PAGED_PREFILL_H */
