/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_moe_gemm_fused.h -- C99 port of the MoE-specialized MFMA GEMM
 * fusion kernel-instance builders
 * ck_dsl/instances/common/moe_gemm_fused.py (2031 LOC).
 *
 *   Python (moe_gemm_fused.py)                 C99 (this header)
 *   ----------------------------------------   ----------------------------------
 *   @dataclass FusedGateUpSiluGemmSpec         ckc_moe_gate_up_silu_gemm_spec_t
 *   @dataclass FusedInterleavedGateUpSiluGemmSpec
 *                                              ckc_moe_interleaved_gate_up_silu_gemm_spec_t
 *   @dataclass FusedDownReduceGemmSpec         ckc_moe_down_reduce_gemm_spec_t
 *   @dataclass FusedDownSiluReduceGemmSpec     ckc_moe_down_silu_reduce_gemm_spec_t
 *                                                (the four spec value types + their
 *                                                 default/finalize/to_universal/
 *                                                 kernel_name helpers are ALREADY
 *                                                 ported in the value-type helper
 *                                                 header included below)
 *   build_moe_gate_up_silu_gemm                ckc_build_moe_gate_up_silu_gemm
 *   build_moe_interleaved_gate_up_silu_gemm    ckc_build_moe_interleaved_gate_up_silu_gemm
 *   build_moe_down_reduce_gemm                 ckc_build_moe_down_reduce_gemm
 *   build_moe_down_silu_reduce_gemm            ckc_build_moe_down_silu_reduce_gemm
 *   moe_*_gemm_signature                       ckc_moe_*_gemm_signature
 *   moe_*_gemm_grid / *_grouped_grid           ckc_moe_*_gemm_grid / *_grouped_grid
 *   (+ convenience: build -> lower .ll)        ckc_moe_*_lower_to_llvm
 *
 * SPECS AS EXPLICIT C STRUCTS. The four Python spec dataclasses are frozen
 * dataclasses with field defaults plus the to_universal_spec() / kernel_name()
 * methods. Their C value types AND the default/finalize/to_universal/kernel_name
 * helpers are already ported in
 * ckc/helper_ck_dsl.instances.common.moe_gemm_fused.h (the value-type helper),
 * which this header includes and re-exports. A caller fills a spec (tile geometry
 * is the only required field), calls *_finalize() to derive block_size, then
 * passes it to the matching build entry.
 *
 * THREE BUILD FAMILIES, ONE SHARED K-LOOP CORE. The three primary builders drive
 * the SAME software-prefetched MFMA k-loop (ckc_moe_emit_prefetch_kloop, in the
 * value-type helper header) parameterised by an operand list (1 B for
 * interleaved / down, 2 B for gate+up). They differ only in their epilogue:
 *   gate+up      : _emit_gate_up_silu_epilogue_default (cshuffle stage + wide store)
 *   interleaved  : _emit_interleaved_silu_epilogue     (LDS pair-load + silu + store)
 *   down+reduce  : _emit_down_reduce_epilogue_atomic   (atomic_add via C-warp decode)
 * The 4th builder (down+silu+reduce) is a thin wrapper that converts its spec to a
 * FusedDownReduceGemmSpec and calls build_moe_down_reduce_gemm (Python P65 MVP).
 *
 * Each builder's body is a long prologue (param decls, batched/grouped dispatch,
 * smem + view + accumulator setup) followed by a guarded compute+epilogue. That
 * private shared state + the per-phase / per-closure function contract lives in
 * the sibling PRIVATE header ckc/instance_moe_gemm_fused_internal.h; public
 * callers only ever touch this header.
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (ckc_b_*); the convenience lower returns a
 * ckc_status_t. The is_valid_spec gate (run at build prologue, via the universal
 * GEMM validator) sets the builder error and returns NULL on a rejected spec.
 */
#ifndef CKC_INSTANCE_MOE_GEMM_FUSED_H
#define CKC_INSTANCE_MOE_GEMM_FUSED_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/arena.h" /* ckc_arena_t (signature storage) */
/* The four spec value types + their default/finalize/to_universal/kernel_name
 * helpers, plus the shared leaf/closure/k-loop helpers (silu, magic-div, CWarp
 * decode, operand, kloop plan, prefetch kloop, cshuffle stage, atomic epilogue).
 * Re-exported so a public caller has the full spec surface from this one header. */
#include "ckc/helper_ck_dsl.instances.common.moe_gemm_fused.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 * Signature descriptors   (moe_*_gemm_signature, Python)
 * ============================================================ *
 *
 * Each *_signature() builds the ordered kernel-param list (SignatureBuilder) the
 * launcher/dispatcher consumes. The C port writes a NUL-terminated array of
 * ckc_sig_entry_t into `out` (capacity out_cap entries) and the realised count
 * into *out_count. The trailing optional params (BlockExpertIds / SortedTokenIds
 * + slot_size) are appended exactly as the Python `if grouped:` / `elif
 * active_tile_skip:` branches do. Returns CKC_OK or CKC_ERR_VALUE (too-small
 * buffer / NULL args). `arena` backs any owned-string storage the entries need. */

ckc_status_t ckc_moe_gate_up_silu_gemm_signature(const ckc_moe_gate_up_silu_gemm_spec_t* spec,
                                                 ckc_arena_t* arena,
                                                 ckc_sig_entry_t* out,
                                                 size_t out_cap,
                                                 size_t* out_count);

ckc_status_t ckc_moe_interleaved_gate_up_silu_gemm_signature(
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    ckc_arena_t* arena,
    ckc_sig_entry_t* out,
    size_t out_cap,
    size_t* out_count);

ckc_status_t ckc_moe_down_reduce_gemm_signature(const ckc_moe_down_reduce_gemm_spec_t* spec,
                                                ckc_arena_t* arena,
                                                ckc_sig_entry_t* out,
                                                size_t out_cap,
                                                size_t* out_count);

/* ============================================================ *
 * Launch grids   (moe_*_gemm_grid / *_grouped_grid, Python)
 * ============================================================ *
 *
 * Pure integer arithmetic over the spec tile geometry. The batched grids take
 * (batch, m, n); the grouped grids take (num_m_blocks, n). The grid is written
 * into out_grid[3] = (gx, gy, gz). The interleaved grids use GEMM N == 2*n. */

void ckc_moe_gate_up_silu_gemm_grid(
    int batch, int m, int n, const ckc_moe_gate_up_silu_gemm_spec_t* spec, int out_grid[3]);
void ckc_moe_gate_up_silu_gemm_grouped_grid(int num_m_blocks,
                                            int n,
                                            const ckc_moe_gate_up_silu_gemm_spec_t* spec,
                                            int out_grid[3]);

void ckc_moe_interleaved_gate_up_silu_gemm_grid(
    int batch,
    int m,
    int n,
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    int out_grid[3]);
void ckc_moe_interleaved_gate_up_silu_gemm_grouped_grid(
    int num_m_blocks,
    int n,
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    int out_grid[3]);

void ckc_moe_down_reduce_gemm_grid(
    int batch, int m, int n, const ckc_moe_down_reduce_gemm_spec_t* spec, int out_grid[3]);
void ckc_moe_down_reduce_gemm_grouped_grid(int num_m_blocks,
                                           int n,
                                           const ckc_moe_down_reduce_gemm_spec_t* spec,
                                           int out_grid[3]);

/* ============================================================ *
 * build_moe_gate_up_silu_gemm   (Python lines 710-909)
 * ============================================================ *
 *
 * Build the dual-B fused gate+up+silu MFMA kernel into the supplied
 * (already ckc_ir_builder_init'd) builder `b`, exactly as the Python build does,
 * and returns the kernel (b->kernel) on success or NULL with b's sticky error
 * set. `arch` NULL => "gfx950". A rejected spec (is_valid_gemm_spec) sets the
 * builder error ("invalid fused gate+up GEMM spec: ...") and returns NULL.
 *
 * Like the Python, this expects the builder to have been created with the spec's
 * kernel_name(). Use the *_new() convenience for the init-from-spec path. */
ckc_kernel_def_t* ckc_build_moe_gate_up_silu_gemm(ckc_ir_builder_t* b,
                                                  const ckc_moe_gate_up_silu_gemm_spec_t* spec,
                                                  const char* arch);
ckc_kernel_def_t* ckc_build_moe_gate_up_silu_gemm_new(ckc_ir_builder_t* b,
                                                      const ckc_moe_gate_up_silu_gemm_spec_t* spec,
                                                      const char* arch);

/* ============================================================ *
 * build_moe_interleaved_gate_up_silu_gemm   (Python lines 1144-1391)
 * ============================================================ *
 *
 * Build the single-B interleaved gate/up GEMM with fused SiLU epilogue. The
 * preshuffle_b trait selects the per-tile contiguous B load override; grouped /
 * active_tile_skip traits add the BlockExpertIds / SortedTokenIds dispatch gate.
 * Same builder-lifetime + arch + error contract as above. A rejected spec sets
 * "invalid interleaved gate/up GEMM spec: ...". */
ckc_kernel_def_t* ckc_build_moe_interleaved_gate_up_silu_gemm(
    ckc_ir_builder_t* b,
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    const char* arch);
ckc_kernel_def_t* ckc_build_moe_interleaved_gate_up_silu_gemm_new(
    ckc_ir_builder_t* b,
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    const char* arch);

/* ============================================================ *
 * build_moe_down_reduce_gemm   (Python lines 1637-1821)
 * ============================================================ *
 *
 * Build the single-B down GEMM with top-k weighted atomic-reduce epilogue
 * (Y[token, n] += weight * down_acc). Same builder-lifetime + arch + error
 * contract. A rejected spec sets "invalid fused down-reduce GEMM spec: ...". */
ckc_kernel_def_t* ckc_build_moe_down_reduce_gemm(ckc_ir_builder_t* b,
                                                 const ckc_moe_down_reduce_gemm_spec_t* spec,
                                                 const char* arch);
ckc_kernel_def_t* ckc_build_moe_down_reduce_gemm_new(ckc_ir_builder_t* b,
                                                     const ckc_moe_down_reduce_gemm_spec_t* spec,
                                                     const char* arch);

/* ============================================================ *
 * build_moe_down_silu_reduce_gemm   (Python lines 2006-2031)
 * ============================================================ *
 *
 * P65 minimum-viable wrapper: converts its FusedDownSiluReduceGemmSpec to a
 * FusedDownReduceGemmSpec (name/tile/trait/wave_size/block_size carried; grouped
 * defaults false) and calls ckc_build_moe_down_reduce_gemm. The silu fusion is a
 * documented follow-up call-site rewrite. Same builder-lifetime + arch + error
 * contract. */
ckc_kernel_def_t* ckc_build_moe_down_silu_reduce_gemm(
    ckc_ir_builder_t* b, const ckc_moe_down_silu_reduce_gemm_spec_t* spec, const char* arch);
ckc_kernel_def_t* ckc_build_moe_down_silu_reduce_gemm_new(
    ckc_ir_builder_t* b, const ckc_moe_down_silu_reduce_gemm_spec_t* spec, const char* arch);

/* ============================================================ *
 * Convenience: build -> lower to LLVM .ll text
 * ============================================================ *
 *
 * Each given a spec, inits a builder with spec.kernel_name(), builds the stock
 * body, and lowers to LLVM .ll text. `arch` NULL => "gfx950". On CKC_OK *out_ll
 * receives a malloc'd NUL-terminated string the caller frees with free(); on
 * failure it is left NULL and (if err!=NULL, capacity err_cap) a diagnostic is
 * written. Each internally owns and frees its IRBuilder. */
ckc_status_t ckc_moe_gate_up_silu_lower_to_llvm(const ckc_moe_gate_up_silu_gemm_spec_t* spec,
                                                const char* arch,
                                                ckc_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap);

ckc_status_t ckc_moe_interleaved_gate_up_silu_lower_to_llvm(
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    const char* arch,
    ckc_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap);

ckc_status_t ckc_moe_down_reduce_lower_to_llvm(const ckc_moe_down_reduce_gemm_spec_t* spec,
                                               const char* arch,
                                               ckc_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap);

ckc_status_t
ckc_moe_down_silu_reduce_lower_to_llvm(const ckc_moe_down_silu_reduce_gemm_spec_t* spec,
                                       const char* arch,
                                       ckc_llvm_flavor_t flavor,
                                       char** out_ll,
                                       char* err,
                                       size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_MOE_GEMM_FUSED_H */
