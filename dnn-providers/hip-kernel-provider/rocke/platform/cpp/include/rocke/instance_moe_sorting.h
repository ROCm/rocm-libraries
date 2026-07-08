/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_moe_sorting.h -- C99 port of the MoE-sorting kernel instance
 * builders in rocke/instances/common/moe_sorting.py (CK Tile 13_moe_sorting
 * parity).
 *
 * Given the topk-softmax router output (topk_ids, topk_weights) of shape
 * (tokens, topk), the pipeline rearranges per-token routing so every expert
 * receives its assigned tokens in a contiguous block -- the prerequisite for the
 * per-expert batched GEMMs in a fused MoE forward. Three split kernels plus an
 * optional single-CTA fusion:
 *
 *   1. moe_sort_histogram : Hist[expert_id] += 1 for every (t,k) pair, via a
 *                           per-block LDS histogram merged to global with one
 *                           atomic per expert bin.
 *   2. moe_sort_scan      : exclusive prefix sum over Hist -> Offsets, copying
 *                           the counts to Counts. Single-block kernel; two paths
 *                           (wave64 Kogge-Stone ds_bpermute when E<=wave_size,
 *                           else LDS Hillis-Steele block_exclusive_scan_i32).
 *   3. moe_sort_scatter   : per (t,k) pair claim the next slot in expert_id's
 *                           bucket (atomic_add(Counter)), then write
 *                           token_id / topk_idx / weight at Offsets[eid]+local.
 *   (+) moe_sort_persistent : all three phases fused in one CTA, LDS-only inter-
 *                           phase state (AITER moe_sorting_opus parity).
 *
 *   Python (moe_sorting.py)                C99 (this header)
 *   ------------------------------------   -------------------------------------
 *   @dataclass(frozen) MoeSortingSpec       rocke_moe_sorting_spec_t
 *     .total_pairs (@property)              rocke_moe_sorting_spec_total_pairs
 *     .kernel_name(phase)                   rocke_moe_sorting_spec_kernel_name
 *   is_valid_spec(spec, arch)               rocke_moe_sorting_is_valid_spec
 *   build_moe_sort_histogram(spec, arch)    rocke_build_moe_sort_histogram
 *   build_moe_sort_scan(spec, arch)         rocke_build_moe_sort_scan
 *   build_moe_sort_scatter(spec, arch)      rocke_build_moe_sort_scatter
 *   build_moe_sort_persistent(spec, arch)   rocke_build_moe_sort_persistent
 *   moe_sort_*_grid(spec)                    rocke_moe_sort_*_grid
 *   moe_sort_*_signature(spec)               rocke_moe_sort_*_signature
 *   moe_sorting_workspace_bytes(spec)        rocke_moe_sorting_workspace_bytes
 *   (+ convenience: build -> lower .ll)      rocke_build_moe_sort_*_lower_to_llvm
 *
 * SPEC AS AN EXPLICIT C STRUCT. The frozen Python dataclass becomes a value
 * struct; rocke_moe_sorting_spec_default() installs the dataclass defaults
 * (block_size 256, name "rocke_moe_sorting") so a caller overrides only the
 * fields it cares about. tokens / topk / experts are required (zeroed by the
 * default helper; the caller fills them).
 *
 * REUSED PORTED HELPERS (one NEW helper port consumed by the build TUs):
 *   - rocke/helper_rocke.helpers.scan.h       : rocke_lds_zero_i32 +
 *     rocke_block_exclusive_scan_i32 (NEWLY ported for this instance; the inline
 *     _wave_kogge_stone_scan_i32 stays a static helper inside the scan TU).
 *   - rocke/helper_rocke.helpers.spec.h       : kernel_name_join, ceil_div_grid,
 *     SignatureBuilder, rocke_sig_entry_t.
 *   - rocke/helper_rocke.helpers.transforms.h : unmerge_magic + CoordVar (the
 *     pair-index decode).
 *   - rocke/helper_rocke.core.arch.h          : ArchTarget.from_gfx + .wave_size.
 *
 * Error model mirrors the rest of the C port: build/lower route errors through
 * the sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a rocke_status_t.
 *
 * Internal build-context + phase-function contract live in
 * rocke/instance_moe_sorting_internal.h (included only by the .c TUs).
 */
#ifndef ROCKE_INSTANCE_MOE_SORTING_H
#define ROCKE_INSTANCE_MOE_SORTING_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

struct rocke_sig_entry; /* fwd (rocke/helper_rocke.helpers.spec.h) */
struct rocke_arena; /* fwd (rocke/arena.h)                      */

/* ===================================================================== *
 *  MoeSortingSpec
 *
 *  @dataclass(frozen=True)
 *  class MoeSortingSpec:
 *      tokens: int
 *      topk: int
 *      experts: int
 *      block_size: int = 256
 *      name: str = "rocke_moe_sorting"
 *
 *  tokens / topk / experts are compile-time constants so the per-kernel IR can
 *  statically size the histogram and the grid; the runtime args are the buffer
 *  pointers + the shapes (for ABI compatibility with the CK Tile reference).
 * ===================================================================== */
typedef struct rocke_moe_sorting_spec
{
    int tokens;
    int topk;
    int experts;
    int block_size; /* default 256 */
    const char* name; /* default "rocke_moe_sorting" */
} rocke_moe_sorting_spec_t;

/* MoeSortingSpec with dataclass defaults (block_size 256,
 * name "rocke_moe_sorting") and the three required dims zeroed. Caller fills
 * tokens, topk, experts. */
rocke_moe_sorting_spec_t rocke_moe_sorting_spec_default(void);

/* @property total_pairs -> tokens * topk. */
int rocke_moe_sorting_spec_total_pairs(const rocke_moe_sorting_spec_t* spec);

/* kernel_name(phase):
 *   kernel_name_join(name, phase, f"T{tokens}", f"K{topk}", f"E{experts}",
 *                    f"b{block_size}")
 * `phase` is one of "hist" / "scan" / "scatter" / "persistent". Writes NUL-
 * terminated into out (capacity out_cap). Returns ROCKE_OK or ROCKE_ERR_VALUE. */
rocke_status_t rocke_moe_sorting_spec_kernel_name(const rocke_moe_sorting_spec_t* spec,
                                                  const char* phase,
                                                  char* out,
                                                  size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950".
 * Checks (in Python order): ArchTarget.from_gfx(arch) resolves; tokens / topk /
 * experts > 0; experts <= 1024 (LDS scan cap); block_size in {64,128,256,512,
 * 1024}; experts <= block_size (single-block scan maps one expert per lane). On
 * reject writes the reason (if non-NULL, cap reason_cap) and returns false; on
 * accept writes "ok" and returns true. */
bool rocke_moe_sorting_is_valid_spec(const rocke_moe_sorting_spec_t* spec,
                                     const char* arch,
                                     char* reason,
                                     size_t reason_cap);

/* ===================================================================== *
 *  BUILD ENTRIES
 *
 *  Each builds the IR into the supplied (already rocke_ir_builder_init'd with
 *  spec.kernel_name(<phase>)) builder `b`, exactly as the Python build does
 *  (is_valid_spec gate, then the phase body), and returns the kernel (b->kernel)
 *  on success or NULL with b's sticky error set. `arch` NULL => "gfx950". Does
 *  NOT re-init the builder.
 * ===================================================================== */

/* build_moe_sort_histogram(spec, arch). Per-block LDS histogram merged to
 * global Hist. Wave-size-agnostic (LDS + global atomics only). */
rocke_kernel_def_t* rocke_build_moe_sort_histogram(rocke_ir_builder_t* b,
                                                   const rocke_moe_sorting_spec_t* spec,
                                                   const char* arch);

/* build_moe_sort_scan(spec, arch). Exclusive prefix sum over Hist -> Offsets,
 * copies counts to Counts. Selects the wave Kogge-Stone path (E<=wave_size) or
 * the LDS Hillis-Steele fallback based on ArchTarget(arch).wave_size. */
rocke_kernel_def_t* rocke_build_moe_sort_scan(rocke_ir_builder_t* b,
                                              const rocke_moe_sorting_spec_t* spec,
                                              const char* arch);

/* build_moe_sort_scatter(spec, arch). Scatter each (t,k) into expert_id's
 * bucket. Wave-size-agnostic. */
rocke_kernel_def_t* rocke_build_moe_sort_scatter(rocke_ir_builder_t* b,
                                                 const rocke_moe_sorting_spec_t* spec,
                                                 const char* arch);

/* build_moe_sort_persistent(spec, arch). Single-CTA histogram + scan + scatter
 * with LDS-only inter-phase state. Uses block_exclusive_scan_i32 (LDS), so it is
 * wave-size-agnostic. */
rocke_kernel_def_t* rocke_build_moe_sort_persistent(rocke_ir_builder_t* b,
                                                    const rocke_moe_sorting_spec_t* spec,
                                                    const char* arch);

/* Convenience: init `b` with spec.kernel_name(<phase>), then the matching build.
 * Caller owns `b` and frees it with rocke_ir_builder_free(). Returns kernel/NULL. */
rocke_kernel_def_t* rocke_build_moe_sort_histogram_new(rocke_ir_builder_t* b,
                                                       const rocke_moe_sorting_spec_t* spec,
                                                       const char* arch);
rocke_kernel_def_t* rocke_build_moe_sort_scan_new(rocke_ir_builder_t* b,
                                                  const rocke_moe_sorting_spec_t* spec,
                                                  const char* arch);
rocke_kernel_def_t* rocke_build_moe_sort_scatter_new(rocke_ir_builder_t* b,
                                                     const rocke_moe_sorting_spec_t* spec,
                                                     const char* arch);
rocke_kernel_def_t* rocke_build_moe_sort_persistent_new(rocke_ir_builder_t* b,
                                                        const rocke_moe_sorting_spec_t* spec,
                                                        const char* arch);

/* ===================================================================== *
 *  GRID HELPERS  --  moe_sort_<phase>_grid(spec). Writes (x,y,z) into out[3].
 *    histogram / scatter : ceil_div_grid((total_pairs, block_size)).
 *    scan / persistent   : (1, 1, 1).
 *  Returns ROCKE_OK or ROCKE_ERR_VALUE (NULL args / non-positive block_size).
 * ===================================================================== */
rocke_status_t rocke_moe_sort_histogram_grid(const rocke_moe_sorting_spec_t* spec, int out[3]);
rocke_status_t rocke_moe_sort_scan_grid(const rocke_moe_sorting_spec_t* spec, int out[3]);
rocke_status_t rocke_moe_sort_scatter_grid(const rocke_moe_sorting_spec_t* spec, int out[3]);
rocke_status_t rocke_moe_sort_persistent_grid(const rocke_moe_sorting_spec_t* spec, int out[3]);

/* ===================================================================== *
 *  SIGNATURE (manifest)  --  moe_sort_<phase>_signature(spec).
 *    histogram  : ptr TopkIds:i32, ptr Hist:i32, scalar num_pairs:i32,
 *                 scalar num_experts:i32.                              (4)
 *    scan       : ptr Hist:i32, ptr Offsets:i32, ptr Counts:i32,
 *                 scalar num_experts:i32.                              (4)
 *    scatter    : ptr TopkIds:i32, ptr TopkWeights:f32, ptr Offsets:i32,
 *                 ptr Counter:i32, ptr SortedTokenIds:i32,
 *                 ptr SortedTopkIds:i32, ptr SortedWeights:f32,
 *                 scalar tokens:i32, scalar topk:i32, scalar num_experts:i32. (10)
 *    persistent : same 10 entries as scatter (superset ABI).          (10)
 *
 *  Writes the manifest entries into out[] (capacity out_cap) and sets
 *  *out_count. Strings live in `arena`. Returns ROCKE_OK or ROCKE_ERR_VALUE (NULL
 *  args / out_cap too small).
 * ===================================================================== */
rocke_status_t rocke_moe_sort_histogram_signature(struct rocke_arena* arena,
                                                  const rocke_moe_sorting_spec_t* spec,
                                                  struct rocke_sig_entry* out,
                                                  size_t out_cap,
                                                  size_t* out_count);
rocke_status_t rocke_moe_sort_scan_signature(struct rocke_arena* arena,
                                             const rocke_moe_sorting_spec_t* spec,
                                             struct rocke_sig_entry* out,
                                             size_t out_cap,
                                             size_t* out_count);
rocke_status_t rocke_moe_sort_scatter_signature(struct rocke_arena* arena,
                                                const rocke_moe_sorting_spec_t* spec,
                                                struct rocke_sig_entry* out,
                                                size_t out_cap,
                                                size_t* out_count);
rocke_status_t rocke_moe_sort_persistent_signature(struct rocke_arena* arena,
                                                   const rocke_moe_sorting_spec_t* spec,
                                                   struct rocke_sig_entry* out,
                                                   size_t out_cap,
                                                   size_t* out_count);

/* moe_sorting_workspace_bytes(spec) -> 4 * experts. The workspace holds the
 * histogram (experts i32) reused as the per-expert next-free counter. */
int rocke_moe_sorting_workspace_bytes(const rocke_moe_sorting_spec_t* spec);

/* ===================================================================== *
 *  CONVENIENCE: build -> lower to LLVM .ll text.
 *  `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 *  string the caller frees with free(); on failure it is left NULL and (if
 *  err!=NULL, cap err_cap) a diagnostic is written. Each owns and frees its
 *  IRBuilder.
 * ===================================================================== */
rocke_status_t rocke_build_moe_sort_histogram_lower_to_llvm(const rocke_moe_sorting_spec_t* spec,
                                                            const char* arch,
                                                            rocke_llvm_flavor_t flavor,
                                                            char** out_ll,
                                                            char* err,
                                                            size_t err_cap);
rocke_status_t rocke_build_moe_sort_scan_lower_to_llvm(const rocke_moe_sorting_spec_t* spec,
                                                       const char* arch,
                                                       rocke_llvm_flavor_t flavor,
                                                       char** out_ll,
                                                       char* err,
                                                       size_t err_cap);
rocke_status_t rocke_build_moe_sort_scatter_lower_to_llvm(const rocke_moe_sorting_spec_t* spec,
                                                          const char* arch,
                                                          rocke_llvm_flavor_t flavor,
                                                          char** out_ll,
                                                          char* err,
                                                          size_t err_cap);
rocke_status_t rocke_build_moe_sort_persistent_lower_to_llvm(const rocke_moe_sorting_spec_t* spec,
                                                             const char* arch,
                                                             rocke_llvm_flavor_t flavor,
                                                             char** out_ll,
                                                             char* err,
                                                             size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_MOE_SORTING_H */
