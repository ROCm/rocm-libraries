/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_fused_moe_e2e.h -- PUBLIC API for the C99 port of the end-to-end
 * fused-MoE forward orchestrator
 * rocke/instances/common/fused_moe_e2e.py (2678 LOC).
 *
 *   Python (fused_moe_e2e.py)               C99 (this header)
 *   -------------------------------------   ------------------------------------
 *   @dataclass FusedMoeForwardSpec          rocke_fmoe_forward_spec_t
 *     (+ tile factories / dtype helpers)      (reused: helper_rocke.helpers.
 *                                              fused_moe_e2e_spec.h)
 *   class FusedMoeForward                    rocke_fmoe_forward_t
 *   FusedMoeForward.__init__(spec)           rocke_fmoe_forward_init(...)
 *   FusedMoeForward.forward(...)             (KernelDef.forward_fn -- see below)
 *   build entry (no direct Python peer)      rocke_build_fused_moe_forward(...)
 *   (+ convenience: build -> lower .ll)      rocke_fused_moe_forward_lower_to_llvm(...)
 *
 * WHAT THIS MODULE IS.
 *   fused_moe_e2e.py defines NO NEW KERNELS. It is a pure host-side
 *   ORCHESTRATOR that composes five pre-existing, already-ported kernel
 *   families into one forward pipeline:
 *
 *     router (topk_softmax) -> sort (moe_sorting: hist+scan+scatter)
 *       -> gather (fused_moe) -> gate+up GEMM -> silu_mul -> down GEMM
 *       -> topk_reduce
 *
 *   The orchestrator owns: a tuning-driven arch-aware MFMA tile policy
 *   (gfx950 wide atoms vs gfx942 narrow atoms), lazy launcher caches for each
 *   sub-kernel + trait combination, a workspace pool, a static-vs-dynamic
 *   dispatch heuristic (T*K*E <= 512), per-expert padded / de-padded grouped
 *   GEMM dispatch, and host weight packing / preshuffle.
 *
 * BUILD ENTRY CONTRACT.
 *   rocke_build_fused_moe_forward(spec, arch) returns a rocke_kernel_def_t* that
 *   ENCAPSULATES the full orchestrator. Mirroring the task's contract:
 *
 *       kernel_def = rocke_build_fused_moe_forward(&spec, "gfx950");
 *       forward    = kernel_def->forward_fn;   // see rocke_fmoe_forward_fn_t
 *       status     = forward(routing_logits, X, W_gate, W_up, W_down, Y, stream);
 *
 *   The returned KernelDef wraps the router->sort->gather->gate+up->silu_mul->
 *   down->topk_reduce chain with static/dynamic dispatch, launcher caches, and
 *   workspace pooling -- exactly what FusedMoeForward holds in Python.
 *
 * PORT SCOPE (faithful vs bounded).
 *   The value-producing surface (spec, tile policy, dtype helpers, workspace
 *   table, host preshuffle shape, per-expert problem arithmetic, the
 *   static/dynamic sizing decision) is ported byte-faithfully -- those live in
 *   the two helper headers this includes plus the internal header. The actual
 *   HIP launch chain, torch buffer zeroing, and D->H copies are a runtime
 *   concern with no IR-builder analogue; in this codegen-only library they are
 *   marked TODO(port) and the forward_fn returns ROCKE_ERR_NOTIMPL until a HIP
 *   runtime backend is wired. The IR that DOES get emitted (and is .ll-lowerable
 *   for golden-digest verification) is the five sub-kernel builders selected by
 *   the spec's tile + trait policy.
 *
 * The ~1900-line orchestrator body (FusedMoeForward + the two forward paths) is
 * a stack of closures over one shared build context; that private state + the
 * per-phase function contract live in the sibling PRIVATE header
 * rocke/instance_fused_moe_e2e_internal.h. Public callers only ever touch THIS
 * header.
 *
 * Error model mirrors the rest of the C port: builders route failures through
 * the sticky-error IRBuilder (rocke_b_* / NULL return); pure helpers return a
 * sentinel; the convenience lower returns a rocke_status_t; the runtime forward
 * entry returns a rocke_status_t (ROCKE_ERR_NOTIMPL for the stubbed launch path).
 */
#ifndef ROCKE_INSTANCE_FUSED_MOE_E2E_H
#define ROCKE_INSTANCE_FUSED_MOE_E2E_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/ir.h" /* rocke_status_t, rocke_kernel_def_t, rocke_ir_builder_t */
#include "rocke/lower_llvm.h" /* rocke_llvm_flavor_t                                */

/* The spec dataclass + tile policy + dtype helpers are already ported. Reuse
 * them verbatim: rocke_fmoe_forward_spec_t, rocke_fmoe_forward_spec_default(), the
 * seven tile factories, rocke_fmoe_resolve_launch_arch(),
 * rocke_fmoe_gemm_dtype_to_universal(), rocke_fmoe_ensure_2byte_dtype(), and the
 * spec.to_*_spec() converters. */
#include "rocke/helper_rocke.helpers.fused_moe_e2e_spec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 * forward_fn ABI   (FusedMoeForward.forward signature)
 * ============================================================ *
 *
 * The callable hung off the returned KernelDef. Arguments mirror the Python
 * keyword-only forward(routing_logits, X, W_gate, W_up, W_down, Y, stream):
 *
 *   routing_logits : (T, E) f32 device ptr
 *   X              : (T, H) act-dtype device ptr
 *   W_gate         : (E, I, H) act-dtype, row-major
 *   W_up           : (E, I, H) act-dtype, row-major
 *   W_down         : (E, H, I) act-dtype, row-major
 *   Y              : (T, H) act-dtype output, populated in place
 *   stream         : HIP stream handle (0 == default)
 *
 * The opaque `self` is the orchestrator instance the KernelDef carries (a
 * rocke_fmoe_forward_t*). Pointers are device pointers (uint64). Returns ROCKE_OK
 * on a completed launch chain, or ROCKE_ERR_NOTIMPL while the launch path is a
 * TODO(port) stub. */
typedef rocke_status_t (*rocke_fmoe_forward_fn_t)(void* self,
                                                  uint64_t routing_logits,
                                                  uint64_t X,
                                                  uint64_t W_gate,
                                                  uint64_t W_up,
                                                  uint64_t W_down,
                                                  uint64_t Y,
                                                  uint64_t stream);

/* ============================================================ *
 * rocke_fmoe_forward_t   (FusedMoeForward host instance)
 * ============================================================ *
 *
 * The orchestrator instance. Holds the resolved arch + arch-swapped tile, the
 * static-vs-dynamic gate + slot size, and -- in a full HIP backend -- the
 * launcher caches + workspace pool. The cache/pool internals live in the
 * internal header (rocke_fmoe_build_ctx_t); this public type is an opaque-ish
 * handle the KernelDef threads to forward_fn as `self`.
 *
 * Field set mirrors FusedMoeForward.__init__'s eagerly-computed scalars
 * (the lazily-allocated launchers / workspaces are part of the internal ctx). */
typedef struct rocke_fmoe_forward
{
    rocke_fmoe_forward_spec_t spec; /* the (tile-policy-adjusted) spec     */
    const char* arch; /* _resolve_launch_arch(spec.arch)     */
    bool use_static_offsets; /* spec.tokens*topk*experts <= 512     */
    int static_slot_size; /* ceil(T*K / tile_m) * tile_m         */

    /* Opaque pointer to the internal build context (launcher caches, pool,
     * cached packed weights). NULL until rocke_fmoe_forward_init populates it.
     * Defined in rocke/instance_fused_moe_e2e_internal.h. */
    void* ctx;

    rocke_fmoe_forward_fn_t forward_fn; /* bound dispatch entry (static/dyn) */
} rocke_fmoe_forward_t;

/* FusedMoeForward.__init__(spec): resolve arch, run the shape-aware tile-swap
 * policy (bf16 -> bf16 tile; large dense -> large_batch tile; sparse ->
 * sparse_batch tile; gfx942 -> narrow-atom variants), compute the
 * static-offset gate (T*K*E <= 512) and static slot_size, and zero the
 * launcher caches. `arch` may be NULL (resolves to the device / "gfx950").
 * On a tile-policy / dtype ValueError returns ROCKE_ERR_VALUE. */
rocke_status_t rocke_fmoe_forward_init(rocke_fmoe_forward_t* self,
                                       const rocke_fmoe_forward_spec_t* spec,
                                       const char* arch);

/* Release any internal ctx resources (launcher caches, cached weights). Safe
 * on a zeroed / partially-initialised self. */
void rocke_fmoe_forward_destroy(rocke_fmoe_forward_t* self);

/* ============================================================ *
 * build_fused_moe_forward   (the C build entry)
 * ============================================================ *
 *
 * Construct the orchestrator from `spec` (running the tile policy + static gate)
 * and return a rocke_kernel_def_t* whose forward_fn drives the full MoE pipeline.
 * `arch` NULL => resolved per FusedMoeForward.__init__ (device / "gfx950").
 *
 * The returned KernelDef is owned by the orchestrator instance referenced by
 * its forward_fn `self`; free both with rocke_fused_moe_forward_free(). On
 * failure returns NULL (allocation / tile-policy ValueError).
 *
 * Usage (task contract):
 *     rocke_kernel_def_t* kd = rocke_build_fused_moe_forward(&spec, "gfx950");
 *     rocke_fmoe_forward_fn_t forward = kd->forward_fn;       // see note below
 *     status = forward(self, rl, X, Wg, Wu, Wd, Y, stream);
 *
 * NOTE: rocke_kernel_def_t (ir.h) carries the lowerable IR fields only. The
 * forward_fn / self pair is delivered alongside the KernelDef via the
 * out-params here so the call site does not depend on extending the shared IR
 * struct. The single-pointer convenience matching the literal task snippet is
 * provided by rocke_build_fused_moe_forward_simple(). */
rocke_kernel_def_t* rocke_build_fused_moe_forward(const rocke_fmoe_forward_spec_t* spec,
                                                  const char* arch,
                                                  rocke_fmoe_forward_t** out_self,
                                                  rocke_fmoe_forward_fn_t* out_forward_fn);

/* Convenience matching the literal task call shape. Allocates the orchestrator
 * + the representative KernelDef, binds forward_fn, and returns the KernelDef;
 * the bound `self` is retrievable via the KernelDef-adjacent registry inside
 * the instance. Caller frees with rocke_fused_moe_forward_free(kd). */
rocke_kernel_def_t* rocke_build_fused_moe_forward_simple(const rocke_fmoe_forward_spec_t* spec,
                                                         const char* arch);

/* Free a KernelDef returned by either build entry plus its bound orchestrator
 * instance and internal ctx. NULL-safe. */
void rocke_fused_moe_forward_free(rocke_kernel_def_t* kernel);

/* ============================================================ *
 * lower-to-.ll convenience   (golden-digest verification path)
 * ============================================================ *
 *
 * The orchestrator emits no single monolithic kernel; for .ll verification the
 * relevant artifacts are the five spec-selected sub-kernel builders. This
 * convenience lowers ONE named pipeline stage to AMDGPU LLVM IR text so a
 * golden test can diff each stage independently. `stage` selects which builder
 * runs (see rocke_fmoe_stage_t); the spec's tile + trait policy is applied
 * exactly as the live orchestrator would for that stage.
 *
 * `arch` NULL => resolved per the spec. On ROCKE_OK *out_ll receives a malloc'd
 * NUL-terminated string the caller frees with free(); on failure it is NULL and
 * (if err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
typedef enum rocke_fmoe_stage
{
    ROCKE_FMOE_STAGE_ROUTER = 0, /* build_topk_softmax            */
    ROCKE_FMOE_STAGE_SORT_HISTOGRAM, /* moe_sorting histogram         */
    ROCKE_FMOE_STAGE_SORT_SCAN, /* moe_sorting scan              */
    ROCKE_FMOE_STAGE_SORT_SCATTER, /* moe_sorting scatter           */
    ROCKE_FMOE_STAGE_GATHER, /* fused_moe gather              */
    ROCKE_FMOE_STAGE_GATE_UP_GEMM, /* batched / interleaved gate+up */
    ROCKE_FMOE_STAGE_SILU_MUL, /* fused_moe silu_mul(_packed)   */
    ROCKE_FMOE_STAGE_DOWN_GEMM, /* batched / down-reduce GEMM    */
    ROCKE_FMOE_STAGE_TOPK_REDUCE /* fused_moe topk_reduce         */
} rocke_fmoe_stage_t;

rocke_status_t rocke_fused_moe_forward_lower_to_llvm(const rocke_fmoe_forward_spec_t* spec,
                                                     const char* arch,
                                                     rocke_fmoe_stage_t stage,
                                                     rocke_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_FUSED_MOE_E2E_H */
