/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_moe_fused_mega.h -- PUBLIC C99 API for the single-launch
 * fused-MoE MEGA-kernel instance builder ported from
 * rocke/instances/common/moe_fused_mega.py (740 LOC).
 *
 * A SINGLE fused kernel computes, per threadgroup, the full per-expert MoE path:
 *
 *   GEMM0 (gate) + GEMM1 (up) sharing one LDS-resident X tile
 *     -> SiLU(gate) * up in registers
 *     -> staged through PERSISTENT LDS (Hidden stays in LDS, never HBM)
 *     -> GEMM2 (down) -> weighted atomic reduce into Y.
 *
 *   Python (moe_fused_mega.py)              C99 (this header)
 *   -------------------------------------   --------------------------------------
 *   class FusedMegaKernelSpec (frozen)      rocke_moe_fused_mega_kernel_spec_t
 *   FusedMegaKernelSpec defaults+__post_init__  rocke_moe_fused_mega_kernel_spec_default()
 *                                               + rocke_moe_fused_mega_kernel_spec_finalize()
 *   spec.gate_up_universal_spec()           rocke_moe_fused_mega_gate_up_universal_spec(...)
 *   spec.down_universal_spec()              rocke_moe_fused_mega_down_universal_spec(...)
 *   spec.kernel_name()                      rocke_moe_fused_mega_kernel_name(...)
 *   moe_fused_mega_grid(...)                rocke_moe_fused_mega_grid(...)
 *   moe_fused_mega_signature(spec)          rocke_moe_fused_mega_signature(...)
 *   build_moe_fused_mega_gemm(spec, arch)   rocke_build_moe_fused_mega_gemm(...)   <-- PRIMARY
 *   (+ convenience: build -> lower .ll)     rocke_moe_fused_mega_lower_to_llvm(...)
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python dataclass is a frozen value type with
 * defaults + a __post_init__ that derives block_size (warp_m*warp_n*wave_size).
 * In C the caller fills a rocke_moe_fused_mega_kernel_spec_t:
 *   rocke_moe_fused_mega_kernel_spec_default() returns every field at the Python
 *   default; the caller overrides what it cares about (at least `name`), then
 *   calls rocke_moe_fused_mega_kernel_spec_finalize() to run __post_init__.
 *
 * REUSED, ALREADY-PORTED COMPONENTS (do NOT re-port): the universal GEMM spec /
 * validity gate (instance_gemm_universal.h), the MoE GEMM fusion helpers
 * (_silu_mul_f32, _CWarpDecode, _MoeOperand, _MoeKloopPlan, _emit_cshuffle_stage,
 * _emit_down_reduce_epilogue_atomic, _emit_moe_prefetch_kloop) in
 * helper_rocke.instances.common.moe_gemm_fused.h, and the tensor_view /
 * spec / atoms helper headers.
 *
 * THIS INSTANCE ADDS the FusedMegaKernelSpec value type, the grid/signature, the
 * two STAGE-4 down-GEMM-with-LDS-A k-loop helpers, and the public build entry
 * that wires the prologue + STAGE 1..5 body in Python order. The shared build
 * context + the body/phase prototypes live in
 *   rocke/instance_moe_fused_mega_internal.h
 * and are consumed by the part-.c translation units.
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (rocke_b_*); an invalid spec yields NULL + ROCKE_ERR_VALUE
 * (ValueError "invalid fused-mega ... GEMM spec"); the convenience lower returns
 * a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_MOE_FUSED_MEGA_H
#define ROCKE_INSTANCE_MOE_FUSED_MEGA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t          */
#include "rocke/instance_gemm_universal.h" /* rocke_gemm_*_spec_t        */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ Spec *
 *
 * Mirror of Python FusedMegaKernelSpec. All tiling is static; M/N/K/H_out and
 * the strides/slot/token counts are runtime kernel args (see the signature).
 * `block_size==0` => derived by finalize() as warp_m*warp_n*wave_size.
 * `dtype` is "fp16" | "f16" | "bf16" | "fp8e4m3" (string-compared like Python).
 */
typedef struct rocke_moe_fused_mega_kernel_spec
{
    const char* name;
    int tile_m; /* default 16  (sorted tokens per m-block, pyisa sub_x) */
    int tile_n_inter; /* default 256 (inter cols this TG owns, pyisa sub_gu)   */
    int tile_k_gu; /* default 32  (K-loop tile along hidden H for gate/up)  */
    int warp_m; /* default 1   */
    int warp_n; /* default 4   */
    int warp_tile_m; /* default 16  */
    int warp_tile_n; /* default 16  */
    int warp_tile_k; /* default 32  */
    int tile_n_down; /* default 256 (down output H_out tile)                  */
    int tile_k_down; /* default 64  (down K-loop tile along inter contraction)*/
    rocke_gemm_trait_spec_t trait; /* default: epilogue="default" (rest gemm dflts) */
    int wave_size; /* default 64  */
    int block_size; /* default 0 => derived at finalize()                    */
    const char* dtype; /* default "fp16"                                        */
} rocke_moe_fused_mega_kernel_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The caller
 * must still set `name`. NOTE the Python default tile_m is 16 (NOT the 32 of the
 * baseline sample config); set tile_m=32 explicitly for the byte-identical
 * f16 verify target. */
rocke_moe_fused_mega_kernel_spec_t rocke_moe_fused_mega_kernel_spec_default(void);

/* __post_init__: when block_size==0, derive it as warp_m*warp_n*wave_size.
 * Idempotent. Call after filling the spec. */
void rocke_moe_fused_mega_kernel_spec_finalize(rocke_moe_fused_mega_kernel_spec_t* spec);

/* spec.gate_up_universal_spec(): the UniversalGemmSpec for the gate+up GEMM
 * (M x inter-slice, contract H). Writes into *out. */
void rocke_moe_fused_mega_gate_up_universal_spec(const rocke_moe_fused_mega_kernel_spec_t* spec,
                                                 rocke_gemm_universal_spec_t* out);

/* spec.down_universal_spec(): the UniversalGemmSpec for the down GEMM
 * (M x H_out-slice, contract inter). Writes into *out. */
void rocke_moe_fused_mega_down_universal_spec(const rocke_moe_fused_mega_kernel_spec_t* spec,
                                              rocke_gemm_universal_spec_t* out);

/* spec.kernel_name() = gate_up_universal_spec().kernel_name() + "_fused_mega".
 * NUL-terminated into out (capacity out_cap). Returns ROCKE_OK or ROCKE_ERR_VALUE
 * (buffer too small). */
rocke_status_t rocke_moe_fused_mega_kernel_name(const rocke_moe_fused_mega_kernel_spec_t* spec,
                                                char* out,
                                                size_t out_cap);

/* ------------------------------------------------------------------ Grid *
 *
 * moe_fused_mega_grid(num_m_blocks, inter, spec) ->
 *   grid = (ceil(inter / tile_n_inter), num_m_blocks, 1).
 * grid.x splits the down-GEMM contraction (inter dim I) across TGs (each
 * atomic-adds a partial Y); grid.y selects the sorted m-block (-> expert id +
 * weight base). Writes gx/gy/gz. */
void rocke_moe_fused_mega_grid(int num_m_blocks,
                               int inter,
                               const rocke_moe_fused_mega_kernel_spec_t* spec,
                               int* out_gx,
                               int* out_gy,
                               int* out_gz);

/* ------------------------------------------------------------- Signature *
 *
 * moe_fused_mega_signature(spec): the 8 pointer params + 11 scalar params in
 * Python order:
 *   ptr A, WGate, WUp, WDown (dt); ptr SortedTokenIds(i32), SortedWeights(f32),
 *   BlockExpertIds(i32), Y(f32); scalar M,N,K,H_out, stride_a, stride_b_gate,
 *   stride_b_up, stride_b_down, slot_size, tokens (all i32).
 *   dt = spec.dtype if in {f16,fp16,bf16} else "f16".
 *
 * Fills `out` (caller array, capacity out_cap entries) and writes the count to
 * *out_count. Returns ROCKE_OK, or ROCKE_ERR_VALUE if out_cap is too small. */
rocke_status_t rocke_moe_fused_mega_signature(const rocke_moe_fused_mega_kernel_spec_t* spec,
                                              rocke_sig_entry_t* out,
                                              size_t out_cap,
                                              size_t* out_count);

/* ------------------------------------------------------------------ *
 * PRIMARY build entry -- build_moe_fused_mega_gemm(spec, arch)
 * ------------------------------------------------------------------ *
 *
 * Builds the single-launch fused-MoE mega-kernel into the supplied (already
 * rocke_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set. `arch` NULL => "gfx950".
 *
 * CALL PATTERN (byte-faithful to Python build_moe_fused_mega_gemm):
 *   1. u_gu = spec.gate_up_universal_spec(); is_valid_gemm_spec(u_gu, arch)
 *      -> NULL + ROCKE_ERR_VALUE "invalid fused-mega gate+up GEMM spec: <why>".
 *   2. u_down = spec.down_universal_spec(); is_valid_gemm_spec(u_down, arch)
 *      -> NULL + ROCKE_ERR_VALUE "invalid fused-mega down GEMM spec: <why>".
 *   3. Set builder attrs (max_workgroup_size, optional waves_per_eu).
 *   4. rocke_moe_mega_build_ctx_init(...) -- run the WHOLE prologue into the ctx
 *      (params -> geometry -> thread decode -> per-expert B byte bases -> LDS
 *      allocs -> views -> plans/operands -> acc inits -> down setup).
 *   5. scf_if(expert_idx >= 0) { rocke_moe_mega_emit_body(ctx); }  -- the empty
 *      tail block (BlockExpertIds == -1) skips all work.
 *   6. Return b->kernel.
 *
 * The prologue locals + the STAGE-1..5 body phases live in
 * rocke/instance_moe_fused_mega_internal.h (rocke_moe_mega_build_ctx_t +
 * rocke_moe_mega_emit_* prototypes); this entry owns the ctx + the u_gu/u_down
 * scratch for the duration of the build.
 *
 * NOTE: like the Python, this expects the builder created with the spec's
 * kernel_name(); use rocke_build_moe_fused_mega_gemm_new() for the convenience. */
rocke_kernel_def_t* rocke_build_moe_fused_mega_gemm(rocke_ir_builder_t* b,
                                                    const rocke_moe_fused_mega_kernel_spec_t* spec,
                                                    const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns `b`
 * and frees it with rocke_ir_builder_free(). Returns the kernel or NULL with b's
 * sticky error set. Mirrors the *_new convenience the other instance ports
 * expose. */
rocke_kernel_def_t* rocke_build_moe_fused_mega_gemm_new(
    rocke_ir_builder_t* b, const rocke_moe_fused_mega_kernel_spec_t* spec, const char* arch);

/* ------------------------------------------------------------------ *
 * Convenience: build -> lower to LLVM .ll text
 * ------------------------------------------------------------------ *
 *
 * Given a spec, init a builder with spec.kernel_name(), build, and lower to LLVM
 * .ll text. `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd
 * NUL-terminated string the caller frees with free(); on failure it is left NULL
 * and (if err != NULL, capacity err_cap) a diagnostic is written. Internally
 * owns and frees its IRBuilder. */
rocke_status_t rocke_moe_fused_mega_lower_to_llvm(const rocke_moe_fused_mega_kernel_spec_t* spec,
                                                  const char* arch,
                                                  rocke_llvm_flavor_t flavor,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_MOE_FUSED_MEGA_H */
