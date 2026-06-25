/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_moe_sorting_internal.h -- PRIVATE shared state + phase-function
 * contract for the C99 port of the four MoE-sorting builders in
 * ck_dsl/instances/common/moe_sorting.py:
 *   build_moe_sort_histogram, build_moe_sort_scan, build_moe_sort_scatter,
 *   build_moe_sort_persistent.
 *
 * WHY THIS HEADER EXISTS.
 *   Unlike conv_direct_grouped, the moe_sorting builders use no Python `def`
 *   closures -- but each builder body is a flat sequence of stages that share one
 *   set of enclosing-function locals (the IRBuilder, the param Values, the
 *   geometry constants, and the SSA constants `c_one`/`c_zero`/`c_E`/...). The
 *   builders ALSO share three module-level helper functions that take the builder
 *   + a handful of Values:
 *     _decode_pair_token_topk(b, pair_idx, topk) -> (t_idx, k_idx)
 *     _decode_expert_load(b, TopkIds, pair_idx, num_experts) -> (eid, valid_e)
 *     _wave_kogge_stone_scan_i32(b, val, length=, lane_id=) -> inclusive
 *
 *   In C there is no closure capture. The faithful port turns each builder body
 *   into a set of free phase functions taking a POINTER to one shared context
 *   struct that holds EXACTLY the variables the stages share. Because the four
 *   builders share so much (params, constants, the pair decode, the validity
 *   gate), a SINGLE ctx (ckc_moe_sort_ctx_t) carries the union of every shared
 *   local across all four kernels; each builder's driver populates the subset it
 *   needs in Python order, then calls its phase functions in Python order. The
 *   three module-level helpers become free functions over (ctx) (or over the bare
 *   builder, for _wave_kogge_stone_scan_i32 which captures nothing).
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every body-implementing .c TU binds to.
 *   It is DESIGNED TO BE COMPLETE: every local the four Python bodies share
 *   across stages is a field here. A body agent implementing a phase MUST be able
 *   to read/write only ctx fields and call the prototypes below WITHOUT editing
 *   this header. If a phase genuinely needs a value not present, that is a design
 *   bug to fix here once, deliberately.
 *
 *   Naming: ctx fields mirror the Python local names 1:1 (Python `pair_idx` ->
 *   `ctx->pair_idx`; Python `c_E` -> `ctx->c_E`; Python `lds_hist` ->
 *   `ctx->lds_hist`). Phase functions mirror the Python builder stages with a
 *   `ckc_moe_sort_<phase>_` prefix; the module helpers keep their leading-
 *   underscore intent as `ckc_moe_sort_decode_*` / `ckc_moe_sort_wave_kogge_*`.
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. Included only by the
 * instance_moe_sorting*.c translation units. Public callers use
 * ckc/instance_moe_sorting.h.
 */
#ifndef CKC_INSTANCE_MOE_SORTING_INTERNAL_H
#define CKC_INSTANCE_MOE_SORTING_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/instance_moe_sorting.h"
#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Upper bound on pairs_per_thread = ceil(total_pairs / block_size), the
 * persistent kernel's unrolled pair-loop trip count. total_pairs = tokens*topk;
 * for the covered decode/persistent regime (small T*K, block_size >= 64) this is
 * small. 64 is generous headroom; the populate routine sets n_pairs_per_thread to
 * the exact value and never indexes beyond it. */
#define CKC_MOE_SORT_MAX_PAIRS_PER_THREAD 64

/* ===================================================================== *
 *  ckc_moe_sort_ctx_t  --  shared state for all four moe_sorting builders.
 *
 *  Field groups follow the Python builder prologues. A given builder's driver
 *  fills only the subset its body reads; unused fields stay NULL/0. Each comment
 *  notes which builder(s) populate the field (H=histogram, C=scan, S=scatter,
 *  P=persistent).
 * ===================================================================== */
typedef struct ckc_moe_sort_ctx
{
    /* ---- inputs / resolved environment (all builders) -- */
    ckc_ir_builder_t* b; /* the IRBuilder `b`                 */
    const ckc_moe_sorting_spec_t* spec; /* the MoeSortingSpec                */
    const char* arch; /* NULL-normalised "gfx950"          */

    /* ---- geometry scalars (Python all-caps / int locals) -- */
    int BS; /* spec->block_size                  (H,C,S,P)            */
    int E; /* spec->experts                     (H,C,S,P)            */
    int NP; /* spec->total_pairs = tokens*topk   (P; H/S use param)   */
    int topk; /* spec->topk (compile-time, for the pair decode) (S,P)  */
    int wave_size; /* ArchTarget(arch).wave_size        (C: path select)     */
    int n_pairs_per_thread; /* ceil(NP/BS), the persistent pair-loop trips (P)*/

    /* ---- kernel params (Values). Each builder declares its own subset in
     *      ABI order; a field is non-NULL only when that builder declared it. */
    ckc_value_t* TopkIds; /* ptr<i32,global>   (H,S,P)                */
    ckc_value_t* TopkWeights; /* ptr<f32,global>   (S,P)                  */
    ckc_value_t* Hist; /* ptr<i32,global>   (H: out; C: in)        */
    ckc_value_t* Offsets; /* ptr<i32,global>   (C: out; S: in; P: out)*/
    ckc_value_t* Counts; /* ptr<i32,global>   (C,P: out)             */
    ckc_value_t* Counter; /* ptr<i32,global>   (S)                    */
    ckc_value_t* SortedTokenIds; /* ptr<i32,global>   (S,P)                  */
    ckc_value_t* SortedTopkIds; /* ptr<i32,global>   (S,P)                  */
    ckc_value_t* SortedWeights; /* ptr<f32,global>   (S,P)                  */
    ckc_value_t* num_pairs; /* scalar i32        (H)                    */
    ckc_value_t* tokens; /* scalar i32        (S,P; ABI)             */
    ckc_value_t* topk_param; /* scalar i32        (S,P; ABI; b.param "topk")*/
    ckc_value_t* num_experts; /* scalar i32        (H,C,S,P)              */

    /* ---- common SSA constants -- */
    ckc_value_t* c_zero; /* const_i32(0)            (P; H/C/S build inline)  */
    ckc_value_t* c_one; /* const_i32(1)            (P)                      */
    ckc_value_t* c_E; /* const_i32(E)            (H,C,S,P)                */
    ckc_value_t* c_BS; /* const_i32(BS)           (P; H/S build inline)    */
    ckc_value_t* c_NP; /* const_i32(NP)           (P)                      */

    /* ---- thread / grid decode (SSA) -- */
    ckc_value_t* tid; /* thread_id_x()         (H,C,S,P)               */
    ckc_value_t* bid; /* block_id_x()          (H,S)                   */
    ckc_value_t* pair_idx; /* bid*BS + tid          (H,S; P recomputes per i)*/
    ckc_value_t* in_bounds; /* cmp_lt(tid, c_E) (C) / cmp_lt(pair_idx, ...) (H,S)*/

    /* ---- LDS buffers -- */
    ckc_value_t* lds_hist; /* smem_alloc i32[E] "lds_hist"  (H,P)         */
    ckc_value_t* lds_scan; /* smem_alloc i32[E] "lds_scan"  (C fallback)  */
    ckc_value_t* lds_counter; /* smem_alloc i32[E] "lds_counter" (P)         */

    /* ---- scatter pair decode (shared scratch for the S/P inner body) -- */
    ckc_value_t* t_idx; /* decoded token id from pair_idx  (S,P)          */
    ckc_value_t* k_idx; /* decoded topk slot from pair_idx (S,P)          */
    ckc_value_t* eid; /* TopkIds[pair_idx]               (H,S,P)        */
    ckc_value_t* valid_e; /* (eid>=0) && (eid<num_experts)   (H,S,P)        */
} ckc_moe_sort_ctx_t;

/* ===================================================================== *
 *  SHARED MODULE HELPERS (Python module-level functions).
 *  Each emits IR in byte-identical Python order. These are the C analogues of
 *  the three leading-underscore helpers + the spec validity gate path.
 * ===================================================================== */

/* _decode_pair_token_topk(b, pair_idx, topk) -> (t_idx, k_idx).
 * Splits a flat pair_idx = t*topk + k via unmerge_magic("pair",("t_idx","k_idx"),
 * dims=(1,topk)) applied with CoordVar("pair", pair_idx). On success writes the
 * two outputs; on builder error leaves them NULL. */
void ckc_moe_sort_decode_pair_token_topk(ckc_ir_builder_t* b,
                                         ckc_value_t* pair_idx,
                                         int topk,
                                         ckc_value_t** out_t_idx,
                                         ckc_value_t** out_k_idx);

/* _decode_expert_load(b, TopkIds, pair_idx, num_experts) -> (eid, valid_e).
 * eid = global_load_i32(TopkIds, pair_idx);
 * valid_e = land(cmp_ge(eid, const_i32(0)), cmp_lt(eid, num_experts)).
 * Emits in that exact order (load -> const(0) -> ge -> lt -> land). */
void ckc_moe_sort_decode_expert_load(ckc_ir_builder_t* b,
                                     ckc_value_t* TopkIds,
                                     ckc_value_t* pair_idx,
                                     ckc_value_t* num_experts,
                                     ckc_value_t** out_eid,
                                     ckc_value_t** out_valid_e);

/* _wave_kogge_stone_scan_i32(b, val, length=, lane_id=) -> inclusive.
 * log2(length) ds_bpermute cross-lane shuffles (no LDS, no barriers); for each
 * stride in (1,2,...,length//2): src_lane = (lane>=stride)?lane-stride:0;
 * addr = src_lane<<2; neighbour = ds_bpermute(addr, cur);
 * cur = (lane>=stride)?cur+neighbour:cur. Captures nothing but the builder. */
ckc_value_t* ckc_moe_sort_wave_kogge_stone_scan_i32(ckc_ir_builder_t* b,
                                                    ckc_value_t* val,
                                                    int length,
                                                    ckc_value_t* lane_id);

/* is_valid_spec(spec, arch) -> bool (+ optional reason). Shared by all four
 * builders' opening gate; also drives wave_size resolution for the scan. On
 * accept returns true and (if out_wave_size non-NULL) writes the resolved
 * ArchTarget wave size. */
bool ckc_moe_sort_is_valid_spec_impl(const ckc_moe_sorting_spec_t* spec,
                                     const char* arch,
                                     char* reason,
                                     size_t reason_cap,
                                     int* out_wave_size);

/* ===================================================================== *
 *  HISTOGRAM PHASE FUNCTIONS  (build_moe_sort_histogram, lines 194-272).
 * ===================================================================== */

/* Prologue (lines 224-243): is_valid_spec gate, set max_workgroup_size, derive
 * BS/E, declare params (TopkIds, Hist, num_pairs, num_experts), decode tid/bid,
 * compute pair_idx = bid*BS + tid. Fills the corresponding ctx fields. Returns
 * false (builder error set) on a rejected spec. */
bool ckc_moe_sort_hist_prologue(ckc_moe_sort_ctx_t* ctx);

/* Stage 1 (lines 245-258): smem_alloc lds_hist[E], lds_zero_i32, then the
 * in_bounds-guarded _decode_expert_load + valid_e-guarded lds_atomic_add, sync. */
void ckc_moe_sort_hist_block_histogram(ckc_moe_sort_ctx_t* ctx);

/* Stage 2 + return (lines 260-272): lanes [0,E) read their LDS bin and, when
 * cnt>0, global_atomic_add into Hist. Returns the kernel (ctx->b->kernel). */
ckc_kernel_def_t* ckc_moe_sort_hist_merge_to_global(ckc_moe_sort_ctx_t* ctx);

/* ===================================================================== *
 *  SCAN PHASE FUNCTIONS  (build_moe_sort_scan, lines 325-439).
 * ===================================================================== */

/* Prologue (lines 363-384): is_valid_spec gate, resolve wave_size, set
 * max_workgroup_size, derive BS/E, declare params (Hist, Offsets, Counts,
 * num_experts), decode tid, build c_E + in_bounds. Returns false on reject. */
bool ckc_moe_sort_scan_prologue(ckc_moe_sort_ctx_t* ctx);

/* Wave path (lines 386-418, taken when E <= wave_size): per-lane Hist load +
 * Counts mirror, inclusive Kogge-Stone via the wave helper, ds_bpermute right-
 * shift to exclusive, store to Offsets. Returns the kernel (ctx->b->kernel). */
ckc_kernel_def_t* ckc_moe_sort_scan_wave_path(ckc_moe_sort_ctx_t* ctx);

/* LDS fallback path (lines 419-439, taken when E > wave_size): smem_alloc
 * lds_scan[E], copy Hist->LDS (and ->Counts), block_exclusive_scan_i32, copy
 * LDS->Offsets. Returns the kernel (ctx->b->kernel). */
ckc_kernel_def_t* ckc_moe_sort_scan_lds_path(ckc_moe_sort_ctx_t* ctx);

/* ===================================================================== *
 *  SCATTER PHASE FUNCTIONS  (build_moe_sort_scatter, lines 447-540).
 * ===================================================================== */

/* Prologue (lines 481-517): is_valid_spec gate, set max_workgroup_size, declare
 * the 10-entry ABI params, decode tid/bid, compute pair_idx, then
 * _decode_pair_token_topk to fill t_idx/k_idx, build num_pairs = tokens*topk and
 * in_bounds. Returns false on reject. */
bool ckc_moe_sort_scatter_prologue(ckc_moe_sort_ctx_t* ctx);

/* Scatter body + return (lines 527-540): under in_bounds, _decode_expert_load,
 * then under valid_e: atomic_add(Counter), Offsets[eid]+local_off, load weight,
 * write SortedTokenIds/SortedTopkIds/SortedWeights. Returns the kernel. */
ckc_kernel_def_t* ckc_moe_sort_scatter_body(ckc_moe_sort_ctx_t* ctx);

/* ===================================================================== *
 *  PERSISTENT PHASE FUNCTIONS  (build_moe_sort_persistent, lines 613-741).
 * ===================================================================== */

/* Prologue (lines 639-682): is_valid_spec gate, set max_workgroup_size, derive
 * BS/E/NP/n_pairs_per_thread, declare the 10-entry ABI params, decode tid, build
 * c_one/c_zero/c_E/c_BS/c_NP. Returns false on reject. */
bool ckc_moe_sort_persistent_prologue(ckc_moe_sort_ctx_t* ctx);

/* Phase 1 (lines 684-704): smem_alloc lds_hist[E], lds_zero_i32, the unrolled
 * pairs_per_thread histogram loop (per-i pair_idx, in_bounds, expert decode,
 * valid_e-guarded lds_atomic_add), sync, then write Counts to global before the
 * scan overwrites lds_hist. */
void ckc_moe_sort_persistent_histogram(ckc_moe_sort_ctx_t* ctx);

/* Phase 2 (lines 706-713): block_exclusive_scan_i32 in place over lds_hist, then
 * write Offsets to global. */
void ckc_moe_sort_persistent_scan(ckc_moe_sort_ctx_t* ctx);

/* Phase 3 + return (lines 715-741): smem_alloc lds_counter[E], lds_zero_i32, the
 * unrolled pairs_per_thread scatter loop (expert decode, lds_atomic_add counter,
 * base from lds_hist, pair decode, weight load, the three Sorted* stores).
 * Returns the kernel (ctx->b->kernel). */
ckc_kernel_def_t* ckc_moe_sort_persistent_scatter(ckc_moe_sort_ctx_t* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_MOE_SORTING_INTERNAL_H */
