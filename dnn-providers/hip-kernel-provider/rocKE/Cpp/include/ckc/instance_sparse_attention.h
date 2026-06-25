/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_sparse_attention.h -- C99 port of the two sparse-attention forward
 * kernel instance builders in ck_dsl/instances/common/sparse_attention.py
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
 *   @dataclass JengaSparseSpec              ckc_jenga_sparse_spec_t
 *     .num_q_blocks / .num_k_blocks         ckc_jenga_sparse_spec_num_{q,k}_blocks
 *     .kernel_name()                        ckc_jenga_sparse_kernel_name(...)
 *   @dataclass VsaSparseSpec                ckc_vsa_sparse_spec_t
 *     .num_q_blocks / .num_k_blocks         ckc_vsa_sparse_spec_num_{q,k}_blocks
 *     .kernel_name()                        ckc_vsa_sparse_kernel_name(...)
 *   is_valid_jenga_spec(spec, arch)         ckc_is_valid_jenga_spec(...)
 *   is_valid_vsa_spec(spec, arch)           ckc_is_valid_vsa_spec(...)
 *   build_jenga_sparse_attention(spec,arch) ckc_build_jenga_sparse_attention(...)
 *   build_vsa_sparse_attention(spec, arch)  ckc_build_vsa_sparse_attention(...)
 *   jenga_sparse_attention_grid(spec)       ckc_jenga_sparse_attention_grid(...)
 *   vsa_sparse_attention_grid(spec)         ckc_vsa_sparse_attention_grid(...)
 *   jenga_sparse_attention_signature(spec)  ckc_jenga_sparse_attention_signature(...)
 *   vsa_sparse_attention_signature(spec)    ckc_vsa_sparse_attention_signature(...)
 *   (+ convenience: build -> lower .ll)     ckc_{jenga,vsa}_sparse_attention_lower_to_llvm
 *
 * SPEC AS A FLAT C STRUCT. Each Python spec composes an FmhaCommonSpec (which in
 * turn composes an FmhaShape). The C entries take flat spec structs embedding a
 * ckc_fmha_common_spec_t by value (Python `common: FmhaCommonSpec`); the build
 * routines reconstitute the equivalent FmhaKernelBuilder state internally so the
 * helper-driven IR emission is byte-identical to the Python path.
 *
 * The shared LDS-bitmap primitives (_const_i8, _cooperative_iter,
 * _stage_jenga_mask_to_lds, _stage_vsa_bitmap_to_lds, _lds_bitmap_predicate) are
 * ported as a sibling helper -- see
 * ckc/helper_ck_dsl.instances.common.sparse_attention.h.
 *
 * REUSED PORTED HELPERS (no new helper port required for this instance):
 *   - ckc/helper_ck_dsl.helpers.mfma_attention.h : MFMA_ATTN_BLOCK_M/K,
 *     mfma_attention_fwd_inner_body (the QK/softmax/PV body + the
 *     extra_mask_predicate callback hook).
 *   - ckc/helper_ck_dsl.instances.common._fmha_common.h : FmhaCommonSpec,
 *     FmhaKernelBuilder, validate_common_spec.
 *   - ckc/helper_ck_dsl.instances.common.fmha_arch.h : validate_fmha_mfma_atom.
 *   - ckc/helper_ck_dsl.helpers.spec.h : kernel_name_join, ckc_sig_entry_t.
 *   - ckc/helper_ck_dsl.helpers.transforms.h : calculate_magic_numbers,
 *     do_magic_division (the sparsity-block index decode).
 *
 * Error model mirrors the rest of the C port: build/lower route errors through the
 * sticky-error IRBuilder (ckc_b_*); the validity gates return a bool + a reason
 * string; the convenience lower returns a ckc_status_t.
 *
 * Internal build-context + phase-function contract live in
 * ckc/instance_sparse_attention_internal.h (included only by the .c TUs).
 */
#ifndef CKC_INSTANCE_SPARSE_ATTENTION_H
#define CKC_INSTANCE_SPARSE_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/helper_ck_dsl.helpers.spec.h" /* ckc_sig_entry_t        */
#include "ckc/helper_ck_dsl.instances.common._fmha_common.h" /* ckc_fmha_common_spec_t */
#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ckc_arena; /* fwd (ckc/arena.h) */

/* _BLOCK_SIZE = 64 -- one wave64 per CTA (matches the mfma_attention helper).
 * Re-exported from the sibling helper port for the build-routine block_size set. */
#ifndef CKC_SPARSE_ATTN_BLOCK_SIZE
#define CKC_SPARSE_ATTN_BLOCK_SIZE 64
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
 *      name: str = "ck_dsl_jenga_sparse_attn"
 * ===================================================================== */
typedef struct ckc_jenga_sparse_spec
{
    ckc_fmha_common_spec_t common;
    int seqlen_q;
    int seqlen_k;
    int block_q; /* default 1                          */
    int block_k; /* default 64                         */
    const char* name; /* NULL => "ck_dsl_jenga_sparse_attn" */
} ckc_jenga_sparse_spec_t;

/* JengaSparseSpec(common, seqlen_q, seqlen_k, block_q=1, block_k=64,
 * name="ck_dsl_jenga_sparse_attn"): take `common` + the required seqlens and the
 * dataclass defaults for block_q/block_k/name. */
ckc_jenga_sparse_spec_t
    ckc_jenga_sparse_spec_default(ckc_fmha_common_spec_t common, int seqlen_q, int seqlen_k);

/* JengaSparseSpec.num_q_blocks property: ceil(seqlen_q / block_q). */
int ckc_jenga_sparse_spec_num_q_blocks(const ckc_jenga_sparse_spec_t* spec);
/* JengaSparseSpec.num_k_blocks property: ceil(seqlen_k / block_k). */
int ckc_jenga_sparse_spec_num_k_blocks(const ckc_jenga_sparse_spec_t* spec);

/* JengaSparseSpec.kernel_name(): kernel_name_join(name, "H{hd}", "HQ{hq}",
 * "HK{hk}", dtype, "Q{sq}", "K{sk}", "BQ{bq}", "BK{bk}"). Writes NUL-terminated
 * into out (capacity out_cap). Returns CKC_OK or CKC_ERR_VALUE (buffer too
 * small). `name` NULL => "ck_dsl_jenga_sparse_attn". */
ckc_status_t
    ckc_jenga_sparse_kernel_name(const ckc_jenga_sparse_spec_t* spec, char* out, size_t out_cap);

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
 *      name: str = "ck_dsl_vsa_sparse_attn"
 *      use_wave_ballot_scatter: bool = True
 * ===================================================================== */
typedef struct ckc_vsa_sparse_spec
{
    ckc_fmha_common_spec_t common;
    int seqlen_q;
    int seqlen_k;
    int block_q; /* default 1                        */
    int block_k; /* default 64                       */
    int max_blocks_per_q; /* default 32                       */
    const char* name; /* NULL => "ck_dsl_vsa_sparse_attn" */
    bool use_wave_ballot_scatter; /* default true                     */
} ckc_vsa_sparse_spec_t;

/* VsaSparseSpec(common, seqlen_q, seqlen_k, block_q=1, block_k=64,
 * max_blocks_per_q=32, name="ck_dsl_vsa_sparse_attn",
 * use_wave_ballot_scatter=True): take `common` + the required seqlens and the
 * dataclass defaults for the remaining fields. */
ckc_vsa_sparse_spec_t
    ckc_vsa_sparse_spec_default(ckc_fmha_common_spec_t common, int seqlen_q, int seqlen_k);

/* VsaSparseSpec.num_q_blocks property: ceil(seqlen_q / block_q). */
int ckc_vsa_sparse_spec_num_q_blocks(const ckc_vsa_sparse_spec_t* spec);
/* VsaSparseSpec.num_k_blocks property: ceil(seqlen_k / block_k). */
int ckc_vsa_sparse_spec_num_k_blocks(const ckc_vsa_sparse_spec_t* spec);

/* VsaSparseSpec.kernel_name(): kernel_name_join(name, "H{hd}", "HQ{hq}",
 * "HK{hk}", dtype, "Q{sq}", "K{sk}", "BQ{bq}", "BK{bk}", "MB{mb}"). Writes
 * NUL-terminated into out (capacity out_cap). Returns CKC_OK or CKC_ERR_VALUE
 * (buffer too small). `name` NULL => "ck_dsl_vsa_sparse_attn". */
ckc_status_t
    ckc_vsa_sparse_kernel_name(const ckc_vsa_sparse_spec_t* spec, char* out, size_t out_cap);

/* ===================================================================== *
 *  Validity gates.
 * ===================================================================== */

/* is_valid_jenga_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950".
 * Chains validate_common_spec -> validate_fmha_mfma_atom -> the seqlen/block/
 * head_size divisibility checks (block_k a multiple of MFMA BLOCK_K, etc.). On
 * reject `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message and the function returns false; on accept returns true and writes
 * "ok". */
bool ckc_is_valid_jenga_spec(const ckc_jenga_sparse_spec_t* spec,
                             const char* arch,
                             char* reason,
                             size_t reason_cap);

/* is_valid_vsa_spec(spec, arch) -> (ok, reason). As above, plus the
 * max_blocks_per_q > 0 check. `arch` NULL => "gfx950". */
bool ckc_is_valid_vsa_spec(const ckc_vsa_sparse_spec_t* spec,
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
 * CALL PATTERN: ckc_build_jenga_sparse_attention(NULL, &spec, "gfx950") returns
 * the KernelDef. `b_unused` is accepted for signature parity with the other
 * instance entries; this builder owns an internal FmhaKernelBuilder regardless of
 * `b_unused`, which is reserved (pass NULL). */
ckc_kernel_def_t* ckc_build_jenga_sparse_attention(ckc_ir_builder_t* b_unused,
                                                   const ckc_jenga_sparse_spec_t* spec,
                                                   const char* arch);

/* build_vsa_sparse_attention(spec, arch). Validates, then builds the VSA
 * forward IR: scatter BlockLut into an LDS bitmap, then run
 * mfma_attention_fwd_inner_body with the LDS-bitmap extra_mask_predicate. `arch`
 * NULL => "gfx950". On an invalid spec or any IR-emission error returns NULL.
 *
 * CALL PATTERN: ckc_build_vsa_sparse_attention(NULL, &spec, "gfx950") returns the
 * KernelDef. `b_unused` is reserved (pass NULL). */
ckc_kernel_def_t* ckc_build_vsa_sparse_attention(ckc_ir_builder_t* b_unused,
                                                 const ckc_vsa_sparse_spec_t* spec,
                                                 const char* arch);

/* ===================================================================== *
 *  Grid + signature.
 * ===================================================================== */

/* jenga_sparse_attention_grid(spec) -> (seqlen_q/BLOCK_M, num_query_heads, 1).
 * Writes the three axes to out[0..2]; `out` must hold 3 ints. */
void ckc_jenga_sparse_attention_grid(const ckc_jenga_sparse_spec_t* spec, int out[3]);

/* vsa_sparse_attention_grid(spec) -> (seqlen_q/BLOCK_M, num_query_heads, 1).
 * Writes the three axes to out[0..2]; `out` must hold 3 ints. */
void ckc_vsa_sparse_attention_grid(const ckc_vsa_sparse_spec_t* spec, int out[3]);

/* jenga_sparse_attention_signature(spec): the kernel ABI signature (Q/K/V/O
 * ptrs, the i8 mask ptr, scale_log2/seqlen_q/seqlen_k scalars, q/k/v/o stride
 * pairs) probed through a throwaway "jenga_sig_probe" FmhaKernelBuilder. On
 * CKC_OK *out_items / *out_count hold the arena-owned array; `arena` backs the
 * storage. On failure the out-params are untouched and the status is returned. */
ckc_status_t ckc_jenga_sparse_attention_signature(const ckc_jenga_sparse_spec_t* spec,
                                                  ckc_arena_t* arena,
                                                  const ckc_sig_entry_t** out_items,
                                                  size_t* out_count);

/* vsa_sparse_attention_signature(spec): the kernel ABI signature (Q/K/V/O ptrs,
 * the block_lut + block_count i32 ptrs, scale_log2/seqlen_q/seqlen_k scalars,
 * q/k/v/o stride pairs) probed through a throwaway "vsa_sig_probe"
 * FmhaKernelBuilder. On CKC_OK *out_items / *out_count hold the arena-owned
 * array; `arena` backs the storage. */
ckc_status_t ckc_vsa_sparse_attention_signature(const ckc_vsa_sparse_spec_t* spec,
                                                ckc_arena_t* arena,
                                                const ckc_sig_entry_t** out_items,
                                                size_t* out_count);

/* ===================================================================== *
 *  Convenience: build + lower to LLVM .ll text.
 * ===================================================================== */

/* Build the Jenga kernel and lower it to LLVM .ll text. `arch` NULL => "gfx950".
 * On CKC_OK *out_ll receives a malloc'd NUL-terminated string the caller frees
 * with free(); on failure it is left NULL and (if err!=NULL, capacity err_cap) a
 * diagnostic is written. Owns and frees its IRBuilder. */
ckc_status_t ckc_jenga_sparse_attention_lower_to_llvm(const ckc_jenga_sparse_spec_t* spec,
                                                      const char* arch,
                                                      ckc_llvm_flavor_t flavor,
                                                      char** out_ll,
                                                      char* err,
                                                      size_t err_cap);

/* Build the VSA kernel and lower it to LLVM .ll text. Contract as above. */
ckc_status_t ckc_vsa_sparse_attention_lower_to_llvm(const ckc_vsa_sparse_spec_t* spec,
                                                    const char* arch,
                                                    ckc_llvm_flavor_t flavor,
                                                    char** out_ll,
                                                    char* err,
                                                    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_SPARSE_ATTENTION_H */
