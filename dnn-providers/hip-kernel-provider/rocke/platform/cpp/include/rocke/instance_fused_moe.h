/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_fused_moe.h -- C99 port of the five MoE-specific kernel instance
 * builders in rocke/instances/common/fused_moe.py (1355 LOC).
 *
 * The fused-MoE forward is a six-stage pipeline; the Python module ships the
 * three MoE-specific kernels with no plain-GEMM analogue (gather / SwiGLU
 * activation / atomic topk-weighted reduce), plus two fast-path variants
 * (packed-G1U1 SwiGLU + a fused static scatter+gather). This header is the
 * public surface for all FIVE IR builders; the per-expert GEMMs and the
 * sort/quant passes stay with their own instances. The host-side
 * FusedMoeLauncher class (Python lines 1084-1356) is OUT OF SCOPE for the IR
 * port -- it lives in the runtime layer, not in ckc.
 *
 *   Python (fused_moe.py)                  C99 (this header)
 *   ------------------------------------   -------------------------------------
 *   @dataclass(frozen) FusedMoeSpec         rocke_fused_moe_spec_t
 *     .total_pairs (@property)               rocke_fused_moe_spec_total_pairs
 *     .elems_per_thread_hidden (@property)   rocke_fused_moe_spec_elems_per_thread_hidden
 *     .elems_per_thread_inter  (@property)   rocke_fused_moe_spec_elems_per_thread_inter
 *     .kernel_name(phase)                    rocke_fused_moe_spec_kernel_name
 *   is_valid_spec(spec) -> (ok, why)         rocke_fused_moe_is_valid_spec
 *   build_moe_gather(spec)                   rocke_build_moe_gather
 *   build_moe_silu_mul(spec)                 rocke_build_moe_silu_mul
 *   build_moe_silu_mul_packed(spec)          rocke_build_moe_silu_mul_packed
 *   build_moe_static_scatter_gather(spec)    rocke_build_moe_static_scatter_gather
 *   build_moe_topk_weighted_reduce(spec)     rocke_build_moe_topk_weighted_reduce
 *   *_grid(spec) -> (gx,1,1)                 rocke_moe_*_grid
 *   *_signature(spec)                        rocke_moe_*_signature
 *   moe_fused_workspace_bytes(spec)          rocke_moe_fused_workspace_bytes
 *   (+ convenience build -> lower .ll)       rocke_*_lower_to_llvm
 *
 * SPEC AS AN EXPLICIT C STRUCT. The frozen FusedMoeSpec becomes a value struct;
 * rocke_fused_moe_spec_default() installs the Python dataclass defaults
 * (dtype "f16", block_size 256, vec 4, name "rocke_fused_moe",
 * bf16_accumulator false) so a caller fills only the required shapes
 * (tokens/experts/topk/hidden/intermediate).
 *
 * USAGE (mirrors the Python build flow exactly):
 *   1. rocke_fused_moe_spec_t spec = rocke_fused_moe_spec_default();
 *      spec.tokens = ...; spec.experts = ...; ... (override fields)
 *   2. rocke_ir_builder_t b;  (caller owns; the *_new convenience inits it)
 *   3. rocke_kernel_def_t* k = rocke_build_moe_gather(&b, &spec, "gfx950");
 *      The builder validates via rocke_fused_moe_is_valid_spec(), extracts the
 *      compile-time consts (H=hidden, BS=block_size, EPT=hidden/BS,
 *      VEC=_effective_vec(vec,BS,H), dtype), seeds the IRBuilder kernel name
 *      via spec.kernel_name(phase) + max_workgroup_size=BS, declares the params
 *      in ABI order, emits the streaming body, and returns b->kernel.
 *   4. rocke_<phase>_lower_to_llvm(&spec, arch, flavor, &ll, err, cap) for the
 *      build -> lower-to-text convenience.
 *
 * BYTE-IDENTICAL .LL VERIFICATION. The C port emits the IR in the exact Python
 * op order so a lowered .ll diffs clean against the Python-generated .ll (kernel
 * names, param order + noalias/readonly/writeonly attrs, max_workgroup_size,
 * loop trip counts chunks=EPT/VEC, interleaved-vs-block-partitioned address
 * arithmetic, atomic ordering). See the internal header for the full invariant
 * list and the phase-by-phase emission contract.
 *
 * REUSED PORTED HELPERS (no new helper port required beyond gather_scatter):
 *   - rocke/helper_rocke.helpers.gather_scatter.h : load_sorted_token_id,
 *     load_sorted_topk_weight (the two indirect moe-sort loads).
 *   - rocke/helper_rocke.helpers.io.h             : io_ir_type,
 *     load_scalar_as_f32, store_scalar_from_f32 (scalar VEC==1 fallback).
 *   - rocke/helper_rocke.helpers.spec.h           : kernel_name_join,
 *     rocke_sig_entry_t (SignatureBuilder manifests).
 *   - rocke/helper_rocke.helpers.distribution.h   : make_static_tile_distribution,
 *     make_static_distributed_tensor (the silu_mul tile path).
 *   - rocke/helper_rocke.helpers.tensor_view.h    : make_global_view,
 *     make_tile_window (the silu_mul tile path).
 *
 * Error model mirrors the rest of the C port: build/lower route errors through
 * the sticky-error IRBuilder; the validity gate returns a bool + reason string;
 * the convenience lowers return a rocke_status_t.
 *
 * Internal build-context + phase-function contract live in
 * rocke/instance_fused_moe_internal.h (included only by the .c TUs).
 */
#ifndef ROCKE_INSTANCE_FUSED_MOE_H
#define ROCKE_INSTANCE_FUSED_MOE_H

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
 *  FusedMoeSpec
 *
 *  @dataclass(frozen=True)
 *  class FusedMoeSpec:
 *      tokens, experts, topk, hidden, intermediate    # required
 *      dtype: "f16"|"fp16"|"bf16" = "f16"
 *      block_size: int = 256       # {64,128,256,512,1024}
 *      vec: int = 4                # {2,4,8}
 *      name: str = "rocke_fused_moe"
 *      bf16_accumulator: bool = False
 *
 *  The DType Literal becomes a plain C string ("f16"/"fp16"/"bf16"); the
 *  is_valid_spec gate enforces the membership check the Python Literal implies.
 * ===================================================================== */
typedef struct rocke_fused_moe_spec
{
    int tokens;
    int experts;
    int topk;
    int hidden;
    int intermediate;
    const char* dtype; /* default "f16" */
    int block_size; /* default 256 */
    int vec; /* default 4 */
    const char* name; /* default "rocke_fused_moe" */
    bool bf16_accumulator; /* default false; topk_reduce v2bf16 atomic path */
} rocke_fused_moe_spec_t;

/* FusedMoeSpec with the dataclass defaults (dtype "f16", block_size 256, vec 4,
 * name "rocke_fused_moe", bf16_accumulator false). The five shape fields are
 * zeroed; the caller fills tokens/experts/topk/hidden/intermediate. */
rocke_fused_moe_spec_t rocke_fused_moe_spec_default(void);

/* @property total_pairs -> tokens * topk (one bucket per (token, k_topk)). */
int rocke_fused_moe_spec_total_pairs(const rocke_fused_moe_spec_t* spec);
/* @property elems_per_thread_hidden -> hidden // block_size. */
int rocke_fused_moe_spec_elems_per_thread_hidden(const rocke_fused_moe_spec_t* spec);
/* @property elems_per_thread_inter -> intermediate // block_size. */
int rocke_fused_moe_spec_elems_per_thread_inter(const rocke_fused_moe_spec_t* spec);

/* kernel_name(phase):
 *   kernel_name_join(name, phase, f"T{tokens}", f"E{experts}", f"K{topk}",
 *                    f"H{hidden}", f"I{intermediate}", dtype,
 *                    f"b{block_size}", f"v{vec}")
 * Writes NUL-terminated into out (capacity out_cap). Returns ROCKE_OK or
 * ROCKE_ERR_VALUE (NULL args / out too small). `phase` is one of "gather",
 * "silu_mul", "silu_mul_packed", "static_scatter_gather", "reduce". */
rocke_status_t rocke_fused_moe_spec_kernel_name(const rocke_fused_moe_spec_t* spec,
                                                const char* phase,
                                                char* out,
                                                size_t out_cap);

/* is_valid_spec(spec) -> (ok, why). Mirrors the Python gate exactly (in order):
 *   tokens/experts/topk > 0; hidden/intermediate > 0; topk <= experts;
 *   block_size in {64,128,256,512,1024}; vec in {2,4,8}; dtype in
 *   {f16,fp16,bf16}; hidden % vec == 0; intermediate % vec == 0;
 *   hidden % block_size == 0; intermediate % block_size == 0.
 * On reject writes the Python-matching message (if reason non-NULL, cap
 * reason_cap) and returns false; on accept writes "ok" and returns true.
 * NOTE: the Python is_valid_spec takes no arch (the MoE streaming kernels emit
 * no MFMA, identical on gfx942/gfx950); arch is not a validity input. */
bool rocke_fused_moe_is_valid_spec(const rocke_fused_moe_spec_t* spec,
                                   char* reason,
                                   size_t reason_cap);

/* ===================================================================== *
 *  BUILD ENTRIES  (all share the contract: build into the supplied, already
 *  rocke_ir_builder_init'd builder `b`, exactly as the matching Python build
 *  does -- raise-on-invalid -> derive consts -> declare params -> emit body --
 *  and return b->kernel on success or NULL with b's sticky error set. `arch`
 *  NULL => "gfx950" but is otherwise unused at IR level. Do NOT re-init `b`.)
 *
 *  Grid for EVERY phase is (spec.total_pairs, 1, 1): one CTA per bucket row.
 * ===================================================================== */

/* build_moe_gather: Stage (2) gather. Per bucket b:
 *   GroupedInput[b, h] = X[SortedTokenIds[b], h]   (sentinel token_id<0 -> skip)
 * Signature: (X:ptr<dtype>, SortedTokenIds:ptr<i32>, GroupedInput:ptr<dtype>,
 *             tokens:i32, hidden:i32). Interleaved-chunk vec copy over hidden;
 * VEC==1 scalar fallback. */
rocke_kernel_def_t* rocke_build_moe_gather(rocke_ir_builder_t* b,
                                           const rocke_fused_moe_spec_t* spec,
                                           const char* arch);

/* build_moe_silu_mul: Stage (4) SwiGLU. Per bucket b, col i over intermediate:
 *   Hidden[b, i] = silu(GateOut[b, i]) * UpOut[b, i]   (all f32; sigmoid via
 *   rcp(1 + exp2(-x*log2e))). Signature: (GateOut:ptr<dtype>, UpOut:ptr<dtype>,
 *   Hidden:ptr<dtype>, total_pairs:i32, intermediate:i32). VEC==1 scalar
 *   fallback; else CK-Tile distribution tile path. */
rocke_kernel_def_t* rocke_build_moe_silu_mul(rocke_ir_builder_t* b,
                                             const rocke_fused_moe_spec_t* spec,
                                             const char* arch);

/* build_moe_silu_mul_packed: Stage (4') SwiGLU from a packed G1U1 buffer:
 *   Hidden[b, i] = silu(GateUp[b, i]) * GateUp[b, I + i],  GateUp shape (M,2*I).
 *   Signature: (GateUp:ptr<dtype>, Hidden:ptr<dtype>, total_pairs:i32,
 *   intermediate:i32). Same numerics as build_moe_silu_mul. */
rocke_kernel_def_t* rocke_build_moe_silu_mul_packed(rocke_ir_builder_t* b,
                                                    const rocke_fused_moe_spec_t* spec,
                                                    const char* arch);

/* build_moe_static_scatter_gather: fused sort-scatter + gather with a static
 * slot layout (slot = eid*slot_size + atomic_add(Counter[eid],1)). Lead lane
 * claims the slot + writes SortedTokenIds/SortedWeights; the wave broadcasts
 * the slot via LDS and cooperatively copies X[t,:] -> GroupedInput[slot,:].
 * Signature: (TopkIds:ptr<i32>, TopkWeights:ptr<f32>, Counter:ptr<i32>,
 *   X:ptr<dtype>, SortedTokenIds:ptr<i32>, SortedWeights:ptr<f32>,
 *   GroupedInput:ptr<dtype>, tokens:i32, topk:i32, num_experts:i32,
 *   hidden:i32, slot_size:i32). */
rocke_kernel_def_t* rocke_build_moe_static_scatter_gather(rocke_ir_builder_t* b,
                                                          const rocke_fused_moe_spec_t* spec,
                                                          const char* arch);

/* build_moe_topk_weighted_reduce: Stage (6) atomic accumulate. Per bucket b:
 *   atomic_add(Y[SortedTokenIds[b], h], SortedWeights[b] * DownOut[b, h])
 *   over hidden, BLOCK-PARTITIONED (each lane owns EPT contiguous cols) for
 *   atomic-contention efficiency; monotonic ordering; f32 accumulator (Y is f32;
 *   bf16_accumulator path -> v2bf16 atomic is a deliberate numerical change).
 *   Signature: (DownOut:ptr<dtype>, SortedTokenIds:ptr<i32>,
 *   SortedWeights:ptr<f32>, Y:ptr<f32>, total_pairs:i32, hidden:i32,
 *   tokens:i32). */
rocke_kernel_def_t* rocke_build_moe_topk_weighted_reduce(rocke_ir_builder_t* b,
                                                         const rocke_fused_moe_spec_t* spec,
                                                         const char* arch);

/* ----- convenience: init `b` with spec.kernel_name(phase), then build. Caller
 * owns `b` and frees it with rocke_ir_builder_free(). Returns kernel/NULL. ----- */
rocke_kernel_def_t* rocke_build_moe_gather_new(rocke_ir_builder_t* b,
                                               const rocke_fused_moe_spec_t* spec,
                                               const char* arch);
rocke_kernel_def_t* rocke_build_moe_silu_mul_new(rocke_ir_builder_t* b,
                                                 const rocke_fused_moe_spec_t* spec,
                                                 const char* arch);
rocke_kernel_def_t* rocke_build_moe_silu_mul_packed_new(rocke_ir_builder_t* b,
                                                        const rocke_fused_moe_spec_t* spec,
                                                        const char* arch);
rocke_kernel_def_t* rocke_build_moe_static_scatter_gather_new(rocke_ir_builder_t* b,
                                                              const rocke_fused_moe_spec_t* spec,
                                                              const char* arch);
rocke_kernel_def_t* rocke_build_moe_topk_weighted_reduce_new(rocke_ir_builder_t* b,
                                                             const rocke_fused_moe_spec_t* spec,
                                                             const char* arch);

/* ===================================================================== *
 *  GRIDS  --  every phase returns (total_pairs, 1, 1). Written into out[3].
 * ===================================================================== */
void rocke_moe_gather_grid(const rocke_fused_moe_spec_t* spec, int out[3]);
void rocke_moe_silu_mul_grid(const rocke_fused_moe_spec_t* spec, int out[3]);
void rocke_moe_silu_mul_packed_grid(const rocke_fused_moe_spec_t* spec, int out[3]);
void rocke_moe_static_scatter_gather_grid(const rocke_fused_moe_spec_t* spec, int out[3]);
void rocke_moe_topk_weighted_reduce_grid(const rocke_fused_moe_spec_t* spec, int out[3]);

/* ===================================================================== *
 *  SIGNATURES (manifests)  --  one per phase, matching the SignatureBuilder
 *  chains. Each writes the entries into out[] (capacity out_cap), sets
 *  *out_count, and parks strings in `arena`. Returns ROCKE_OK or ROCKE_ERR_VALUE
 *  (NULL args / out_cap too small). Entry counts: gather 5, silu_mul 5,
 *  silu_mul_packed 4, static_scatter_gather 12, topk_reduce 7.
 * ===================================================================== */
rocke_status_t rocke_moe_gather_signature(struct rocke_arena* arena,
                                          const rocke_fused_moe_spec_t* spec,
                                          struct rocke_sig_entry* out,
                                          size_t out_cap,
                                          size_t* out_count);
rocke_status_t rocke_moe_silu_mul_signature(struct rocke_arena* arena,
                                            const rocke_fused_moe_spec_t* spec,
                                            struct rocke_sig_entry* out,
                                            size_t out_cap,
                                            size_t* out_count);
rocke_status_t rocke_moe_silu_mul_packed_signature(struct rocke_arena* arena,
                                                   const rocke_fused_moe_spec_t* spec,
                                                   struct rocke_sig_entry* out,
                                                   size_t out_cap,
                                                   size_t* out_count);
rocke_status_t rocke_moe_static_scatter_gather_signature(struct rocke_arena* arena,
                                                         const rocke_fused_moe_spec_t* spec,
                                                         struct rocke_sig_entry* out,
                                                         size_t out_cap,
                                                         size_t* out_count);
rocke_status_t rocke_moe_topk_weighted_reduce_signature(struct rocke_arena* arena,
                                                        const rocke_fused_moe_spec_t* spec,
                                                        struct rocke_sig_entry* out,
                                                        size_t out_cap,
                                                        size_t* out_count);

/* ===================================================================== *
 *  WORKSPACE
 *
 *  moe_fused_workspace_bytes(spec):
 *    elem_bytes=2; grouped + gate + up + hidden_buf + down where grouped/down =
 *    total_pairs*hidden*2 and gate/up/hidden_buf = total_pairs*intermediate*2.
 *    Returned as int64 (large shapes overflow int32). Excludes the moe-sort
 *    workspace + the f32 Y accumulator (caller-cleared).
 * ===================================================================== */
long long rocke_moe_fused_workspace_bytes(const rocke_fused_moe_spec_t* spec);

/* ===================================================================== *
 *  CONVENIENCE: build -> lower to LLVM .ll text. `arch` NULL => "gfx950". On
 *  ROCKE_OK *out_ll receives a malloc'd NUL-terminated string the caller frees
 *  with free(); on failure it is left NULL and (if err!=NULL, cap err_cap) a
 *  diagnostic is written. Each owns and frees its IRBuilder.
 * ===================================================================== */
rocke_status_t rocke_moe_gather_lower_to_llvm(const rocke_fused_moe_spec_t* spec,
                                              const char* arch,
                                              rocke_llvm_flavor_t flavor,
                                              char** out_ll,
                                              char* err,
                                              size_t err_cap);
rocke_status_t rocke_moe_silu_mul_lower_to_llvm(const rocke_fused_moe_spec_t* spec,
                                                const char* arch,
                                                rocke_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap);
rocke_status_t rocke_moe_silu_mul_packed_lower_to_llvm(const rocke_fused_moe_spec_t* spec,
                                                       const char* arch,
                                                       rocke_llvm_flavor_t flavor,
                                                       char** out_ll,
                                                       char* err,
                                                       size_t err_cap);
rocke_status_t rocke_moe_static_scatter_gather_lower_to_llvm(const rocke_fused_moe_spec_t* spec,
                                                             const char* arch,
                                                             rocke_llvm_flavor_t flavor,
                                                             char** out_ll,
                                                             char* err,
                                                             size_t err_cap);
rocke_status_t rocke_moe_topk_weighted_reduce_lower_to_llvm(const rocke_fused_moe_spec_t* spec,
                                                            const char* arch,
                                                            rocke_llvm_flavor_t flavor,
                                                            char** out_ll,
                                                            char* err,
                                                            size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_FUSED_MOE_H */
