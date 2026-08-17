/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_sparse_attention.h -- C99 port of the two sparse-attention forward
 * kernel instance builders in rocke/instances/common/sparse_attention.py
 * (CK Tile ``50_sparse_attn`` parity).
 *
 * Two MFMA-tiled sparse-attention configurations share the
 * mfma_attention_fwd_inner_body QK->softmax->PV chain and gate each K-tile's
 * softmax update through an LDS-staged mask bitmap via extra_mask_predicate:
 *
 *   - Jenga block-sparse (build_jenga_sparse_attention): the caller pre-builds an
 *     i8 MaskBitmap[q_block, k_block] (1 = attend, 0 = skip). Each K-tile's
 *     contribution is gated by the bitmap byte for its enclosing sparsity K-block.
 *     The per-Q-block mask row is cooperatively staged to LDS once per CTA so the
 *     predicate body is a single ds_read_u8.
 *   - VSA / variable-size attention (build_vsa_sparse_attention): each q_block has
 *     a LUT BlockLut[q_block, slot] of length BlockCount[q_block] of the K-blocks
 *     it attends to. The LUT is scattered into an LDS i8 bitmap once per CTA
 *     (bitmap[lut_val] = 1, idempotent), collapsing the per-K-tile O(max_blocks)
 *     global LUT scan to one LDS byte read.
 *
 *   Python (sparse_attention.py)            C99 (this header)
 *   -------------------------------------   ------------------------------------
 *   _magic_div(b, dividend, divisor)        (private; see *_internal.h)
 *   @dataclass JengaSparseSpec              rocke_jenga_sparse_spec_t
 *     .num_q_blocks / .num_k_blocks         rocke_jenga_sparse_spec_num_{q,k}_blocks
 *     .kernel_name()                        rocke_jenga_sparse_kernel_name(...)
 *   @dataclass VsaSparseSpec                rocke_vsa_sparse_spec_t
 *     .num_q_blocks / .num_k_blocks         rocke_vsa_sparse_spec_num_{q,k}_blocks
 *     .kernel_name()                        rocke_vsa_sparse_kernel_name(...)
 *   is_valid_jenga_spec(spec, arch)         rocke_is_valid_jenga_spec(...)
 *   is_valid_vsa_spec(spec, arch)           rocke_is_valid_vsa_spec(...)
 *   build_jenga_sparse_attention(spec,arch) rocke_build_jenga_sparse_attention(...)
 *   build_vsa_sparse_attention(spec, arch)  rocke_build_vsa_sparse_attention(...)
 *   jenga_sparse_attention_grid(spec)       rocke_jenga_sparse_attention_grid(...)
 *   vsa_sparse_attention_grid(spec)         rocke_vsa_sparse_attention_grid(...)
 *   jenga_sparse_attention_signature(spec)  rocke_jenga_sparse_attention_signature(...)
 *   vsa_sparse_attention_signature(spec)    rocke_vsa_sparse_attention_signature(...)
 *   (+ convenience: build -> lower .ll)     rocke_{jenga,vsa}_sparse_attention_lower_to_llvm
 *
 * SPEC AS A FLAT C STRUCT. Each Python spec composes an FmhaCommonSpec (which in
 * turn composes an FmhaShape). The C entries take flat spec structs embedding a
 * rocke_fmha_common_spec_t by value (Python `common: FmhaCommonSpec`); the build
 * routines reconstitute the equivalent FmhaKernelBuilder state internally so the
 * helper-driven IR emission is byte-identical to the Python path.
 *
 * The shared LDS-bitmap primitives (_const_i8, _cooperative_iter,
 * _stage_jenga_mask_to_lds, _stage_vsa_bitmap_to_lds, _lds_bitmap_predicate) are
 * ported as a sibling helper -- see
 * rocke/helper_rocke.instances.common.sparse_attention.h.
 *
 * REUSED PORTED HELPERS (no new helper port required for this instance):
 *   - rocke/helper_rocke.helpers.mfma_attention.h : MFMA_ATTN_BLOCK_M/K,
 *     mfma_attention_fwd_inner_body (the QK/softmax/PV body + the
 *     extra_mask_predicate callback hook).
 *   - rocke/helper_rocke.instances.common._fmha_common.h : FmhaCommonSpec,
 *     FmhaKernelBuilder, validate_common_spec.
 *   - rocke/helper_rocke.instances.common.fmha_arch.h : validate_fmha_mfma_atom.
 *   - rocke/helper_rocke.helpers.spec.h : kernel_name_join, rocke_sig_entry_t.
 *   - rocke/helper_rocke.helpers.transforms.h : calculate_magic_numbers,
 *     do_magic_division (the sparsity-block index decode).
 *
 * Error model mirrors the rest of the C port: build/lower route errors through the
 * sticky-error IRBuilder (rocke_b_*); the validity gates return a bool + a reason
 * string; the convenience lower returns a rocke_status_t.
 *
 * Internal build-context + phase-function contract live in
 * rocke/instance_sparse_attention_internal.h (included only by the .c TUs).
 */
#ifndef ROCKE_INSTANCE_SPARSE_ATTENTION_H
#define ROCKE_INSTANCE_SPARSE_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t        */
#include "rocke/helper_rocke.instances.common._fmha_common.h" /* rocke_fmha_common_spec_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

struct rocke_arena; /* fwd (rocke/arena.h) */

/* _BLOCK_SIZE = 64 -- one wave64 per CTA (matches the mfma_attention helper).
 * Re-exported from the sibling helper port for the build-routine block_size set. */
#ifndef ROCKE_SPARSE_ATTN_BLOCK_SIZE
#define ROCKE_SPARSE_ATTN_BLOCK_SIZE 64
#endif

/* ===================================================================== *
 *  JengaSparseSpec
 *
 *  @dataclass(frozen=True)
 *  class JengaSparseSpec:
 *      common: FmhaCommonSpec
 *      seqlen_q: int
 *      seqlen_k: int
 *      block_q: int = 1
 *      block_k: int = 64
 *      name: str = "rocke_jenga_sparse_attn"
 * ===================================================================== */
typedef struct rocke_jenga_sparse_spec
{
    rocke_fmha_common_spec_t common;
    int seqlen_q;
    int seqlen_k;
    int block_q; /* default 1                          */
    int block_k; /* default 64                         */
    const char* name; /* NULL => "rocke_jenga_sparse_attn" */
} rocke_jenga_sparse_spec_t;

/* JengaSparseSpec(common, seqlen_q, seqlen_k, block_q=1, block_k=64,
 * name="rocke_jenga_sparse_attn"): take `common` + the required seqlens and the
 * dataclass defaults for block_q/block_k/name. */
rocke_jenga_sparse_spec_t
    rocke_jenga_sparse_spec_default(rocke_fmha_common_spec_t common, int seqlen_q, int seqlen_k);

/* JengaSparseSpec.num_q_blocks property: ceil(seqlen_q / block_q). */
int rocke_jenga_sparse_spec_num_q_blocks(const rocke_jenga_sparse_spec_t* spec);
/* JengaSparseSpec.num_k_blocks property: ceil(seqlen_k / block_k). */
int rocke_jenga_sparse_spec_num_k_blocks(const rocke_jenga_sparse_spec_t* spec);

/* JengaSparseSpec.kernel_name(): kernel_name_join(name, "H{hd}", "HQ{hq}",
 * "HK{hk}", dtype, "Q{sq}", "K{sk}", "BQ{bq}", "BK{bk}"). Writes NUL-terminated
 * into out (capacity out_cap). Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too
 * small). `name` NULL => "rocke_jenga_sparse_attn". */
rocke_status_t rocke_jenga_sparse_kernel_name(const rocke_jenga_sparse_spec_t* spec,
                                              char* out,
                                              size_t out_cap);

/* ===================================================================== *
 *  VsaSparseSpec
 *
 *  @dataclass(frozen=True)
 *  class VsaSparseSpec:
 *      common: FmhaCommonSpec
 *      seqlen_q: int
 *      seqlen_k: int
 *      block_q: int = 1
 *      block_k: int = 64
 *      max_blocks_per_q: int = 32
 *      name: str = "rocke_vsa_sparse_attn"
 *      use_wave_ballot_scatter: bool = True
 * ===================================================================== */
typedef struct rocke_vsa_sparse_spec
{
    rocke_fmha_common_spec_t common;
    int seqlen_q;
    int seqlen_k;
    int block_q; /* default 1                        */
    int block_k; /* default 64                       */
    int max_blocks_per_q; /* default 32                       */
    const char* name; /* NULL => "rocke_vsa_sparse_attn" */
    bool use_wave_ballot_scatter; /* default true                     */
} rocke_vsa_sparse_spec_t;

/* VsaSparseSpec(common, seqlen_q, seqlen_k, block_q=1, block_k=64,
 * max_blocks_per_q=32, name="rocke_vsa_sparse_attn",
 * use_wave_ballot_scatter=True): take `common` + the required seqlens and the
 * dataclass defaults for the remaining fields. */
rocke_vsa_sparse_spec_t
    rocke_vsa_sparse_spec_default(rocke_fmha_common_spec_t common, int seqlen_q, int seqlen_k);

/* VsaSparseSpec.num_q_blocks property: ceil(seqlen_q / block_q). */
int rocke_vsa_sparse_spec_num_q_blocks(const rocke_vsa_sparse_spec_t* spec);
/* VsaSparseSpec.num_k_blocks property: ceil(seqlen_k / block_k). */
int rocke_vsa_sparse_spec_num_k_blocks(const rocke_vsa_sparse_spec_t* spec);

/* VsaSparseSpec.kernel_name(): kernel_name_join(name, "H{hd}", "HQ{hq}",
 * "HK{hk}", dtype, "Q{sq}", "K{sk}", "BQ{bq}", "BK{bk}", "MB{mb}"). Writes
 * NUL-terminated into out (capacity out_cap). Returns ROCKE_OK or ROCKE_ERR_VALUE
 * (buffer too small). `name` NULL => "rocke_vsa_sparse_attn". */
rocke_status_t
    rocke_vsa_sparse_kernel_name(const rocke_vsa_sparse_spec_t* spec, char* out, size_t out_cap);

/* ===================================================================== *
 *  Validity gates.
 * ===================================================================== */

/* is_valid_jenga_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950".
 * Chains validate_common_spec -> validate_fmha_mfma_atom -> the seqlen/block/
 * head_size divisibility checks (block_k a multiple of MFMA BLOCK_K, etc.). On
 * reject `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message and the function returns false; on accept returns true and writes
 * "ok". */
bool rocke_is_valid_jenga_spec(const rocke_jenga_sparse_spec_t* spec,
                               const char* arch,
                               char* reason,
                               size_t reason_cap);

/* is_valid_vsa_spec(spec, arch) -> (ok, reason). As above, plus the
 * max_blocks_per_q > 0 check. `arch` NULL => "gfx950". */
bool rocke_is_valid_vsa_spec(const rocke_vsa_sparse_spec_t* spec,
                             const char* arch,
                             char* reason,
                             size_t reason_cap);

/* ===================================================================== *
 *  Build entries (mirror the Python `build_*` functions).
 * ===================================================================== */

/* build_jenga_sparse_attention(spec, arch). Validates, then builds the Jenga
 * block-sparse forward IR: stage Mask[q_block,:] to LDS, then run
 * mfma_attention_fwd_inner_body with the LDS-bitmap extra_mask_predicate. `arch`
 * NULL => "gfx950". On an invalid spec or any IR-emission error returns NULL; if
 * `b` is non-NULL its sticky error carries the diagnostic.
 *
 * CALL PATTERN: rocke_build_jenga_sparse_attention(NULL, &spec, "gfx950") returns
 * the KernelDef. `b_unused` is accepted for signature parity with the other
 * instance entries; this builder owns an internal FmhaKernelBuilder regardless of
 * `b_unused`, which is reserved (pass NULL). */
rocke_kernel_def_t* rocke_build_jenga_sparse_attention(rocke_ir_builder_t* b_unused,
                                                       const rocke_jenga_sparse_spec_t* spec,
                                                       const char* arch);

/* build_vsa_sparse_attention(spec, arch). Validates, then builds the VSA
 * forward IR: scatter BlockLut into an LDS bitmap, then run
 * mfma_attention_fwd_inner_body with the LDS-bitmap extra_mask_predicate. `arch`
 * NULL => "gfx950". On an invalid spec or any IR-emission error returns NULL.
 *
 * CALL PATTERN: rocke_build_vsa_sparse_attention(NULL, &spec, "gfx950") returns the
 * KernelDef. `b_unused` is reserved (pass NULL). */
rocke_kernel_def_t* rocke_build_vsa_sparse_attention(rocke_ir_builder_t* b_unused,
                                                     const rocke_vsa_sparse_spec_t* spec,
                                                     const char* arch);

/* ===================================================================== *
 *  Grid + signature.
 * ===================================================================== */

/* jenga_sparse_attention_grid(spec) -> (seqlen_q/BLOCK_M, num_query_heads, 1).
 * Writes the three axes to out[0..2]; `out` must hold 3 ints. */
void rocke_jenga_sparse_attention_grid(const rocke_jenga_sparse_spec_t* spec, int out[3]);

/* vsa_sparse_attention_grid(spec) -> (seqlen_q/BLOCK_M, num_query_heads, 1).
 * Writes the three axes to out[0..2]; `out` must hold 3 ints. */
void rocke_vsa_sparse_attention_grid(const rocke_vsa_sparse_spec_t* spec, int out[3]);

/* jenga_sparse_attention_signature(spec): the kernel ABI signature (Q/K/V/O
 * ptrs, the i8 mask ptr, scale_log2/seqlen_q/seqlen_k scalars, q/k/v/o stride
 * pairs) probed through a throwaway "jenga_sig_probe" FmhaKernelBuilder. On
 * ROCKE_OK *out_items / *out_count hold the arena-owned array; `arena` backs the
 * storage. On failure the out-params are untouched and the status is returned. */
rocke_status_t rocke_jenga_sparse_attention_signature(const rocke_jenga_sparse_spec_t* spec,
                                                      rocke_arena_t* arena,
                                                      const rocke_sig_entry_t** out_items,
                                                      size_t* out_count);

/* vsa_sparse_attention_signature(spec): the kernel ABI signature (Q/K/V/O ptrs,
 * the block_lut + block_count i32 ptrs, scale_log2/seqlen_q/seqlen_k scalars,
 * q/k/v/o stride pairs) probed through a throwaway "vsa_sig_probe"
 * FmhaKernelBuilder. On ROCKE_OK *out_items / *out_count hold the arena-owned
 * array; `arena` backs the storage. */
rocke_status_t rocke_vsa_sparse_attention_signature(const rocke_vsa_sparse_spec_t* spec,
                                                    rocke_arena_t* arena,
                                                    const rocke_sig_entry_t** out_items,
                                                    size_t* out_count);

/* ===================================================================== *
 *  Convenience: build + lower to LLVM .ll text.
 * ===================================================================== */

/* Build the Jenga kernel and lower it to LLVM .ll text. `arch` NULL => "gfx950".
 * On ROCKE_OK *out_ll receives a malloc'd NUL-terminated string the caller frees
 * with free(); on failure it is left NULL and (if err!=NULL, capacity err_cap) a
 * diagnostic is written. Owns and frees its IRBuilder. */
rocke_status_t rocke_jenga_sparse_attention_lower_to_llvm(const rocke_jenga_sparse_spec_t* spec,
                                                          const char* arch,
                                                          rocke_llvm_flavor_t flavor,
                                                          char** out_ll,
                                                          char* err,
                                                          size_t err_cap);

/* Build the VSA kernel and lower it to LLVM .ll text. Contract as above. */
rocke_status_t rocke_vsa_sparse_attention_lower_to_llvm(const rocke_vsa_sparse_spec_t* spec,
                                                        const char* arch,
                                                        rocke_llvm_flavor_t flavor,
                                                        char** out_ll,
                                                        char* err,
                                                        size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_SPARSE_ATTENTION_H */
