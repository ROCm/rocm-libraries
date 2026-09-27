/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_streamk_gemm.h -- C99 port of the StreamK GEMM kernel instance
 * builder rocke/instances/common/streamk_gemm.py (CK Tile 40_streamk_gemm
 * parity).
 *
 *   Python (streamk_gemm.py)              C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   class StreamKGemmSpec                 rocke_streamk_gemm_spec_t
 *   is_valid_spec(spec, arch)             rocke_streamk_gemm_is_valid_spec(...)
 *   build_streamk_gemm(spec, arch)        rocke_build_streamk_gemm(...)
 *   streamk_gemm_grid(spec)               rocke_streamk_gemm_grid(...)
 *   streamk_gemm_workspace_bytes(spec)    rocke_streamk_gemm_workspace_bytes(...)
 *   build_streamk_gemm_block_tile(...)    rocke_build_streamk_gemm_block_tile(...)
 *   (+ convenience: build -> lower .ll)   rocke_streamk_gemm_lower_to_llvm(...)
 *
 * The Python @property values (partition, grid_size, atom, block_size,
 * persistent_max_iters, kernel_name) become pure accessor helpers; their
 * derivation is byte-faithful to the Python so the emitted op stream matches.
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_STREAMK_GEMM_H
#define ROCKE_INSTANCE_STREAMK_GEMM_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom_t */
#include "rocke/helper_rocke.helpers.streamk.h" /* rocke_streamk_partition_t, strategy */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- StreamKGemmSpec *
 *
 * Mirror of the Python @dataclass(frozen=True) StreamKGemmSpec. The derived
 * @property values are NOT stored; they are recomputed by the accessor helpers
 * below (matching the Python properties, including the divisibility ValueError
 * which is surfaced as an out-param status / sentinel return). Fields are 1:1
 * with the Python declaration order with their dataclass defaults noted. */
typedef struct rocke_streamk_gemm_spec
{
    int M;
    int N;
    int K;
    int tile_m; /* default 16  */
    int tile_n; /* default 16  */
    int tile_k; /* default 16  */
    const char* dtype; /* default "f16" */
    int num_cus; /* default 304 */
    int blocks_per_cu; /* default 1   */
    rocke_streamk_reduction_strategy_t reduction; /* default Atomic */
    bool persistent; /* default false */
    const char* name; /* default "rocke_streamk_gemm" */
} rocke_streamk_gemm_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still set the required M/N/K geometry. */
rocke_streamk_gemm_spec_t rocke_streamk_gemm_spec_default(void);

/* @property partition: StreamKPartition(m_tiles=M//tile_m, n_tiles=N//tile_n,
 * k_iters=K//tile_k). On the Python ValueError (M/N/K not divisible by their
 * tile sizes) returns false and leaves *out untouched; else writes *out and
 * returns true. */
bool rocke_streamk_gemm_partition(const rocke_streamk_gemm_spec_t* spec,
                                  rocke_streamk_partition_t* out);

/* @property atom: the square f16 MFMA atom for (tile_m, tile_n):
 *   (16,16) -> f16_16x16x16 ; (32,32) -> f16_32x32x8.
 * Returns a pointer into the static MFMA catalog, or NULL on the Python
 * ValueError (unsupported tile shape). */
const rocke_mfma_atom_t* rocke_streamk_gemm_atom(const rocke_streamk_gemm_spec_t* spec);

/* @property grid_size: compute_streamk_grid_size(partition, num_cus,
 * blocks_per_cu). Returns -1 on the Python ValueError path (degenerate
 * partition / zero macro tiles). */
int rocke_streamk_gemm_grid_size(const rocke_streamk_gemm_spec_t* spec);

/* @property block_size: 64 (one wave64 warp per CTA). */
int rocke_streamk_gemm_block_size(const rocke_streamk_gemm_spec_t* spec);

/* @property persistent_max_iters: ceil(num_macro_tiles / grid_size). Returns
 * -1 on a degenerate partition (grid_size <= 0). */
int rocke_streamk_gemm_persistent_max_iters(const rocke_streamk_gemm_spec_t* spec);

/* StreamKGemmSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 * Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too small / degenerate spec). */
rocke_status_t rocke_streamk_gemm_kernel_name(const rocke_streamk_gemm_spec_t* spec,
                                              char* out,
                                              size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message; returns false. On accept returns true and writes "ok". */
bool rocke_streamk_gemm_is_valid_spec(const rocke_streamk_gemm_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap);

/* build_streamk_gemm(spec, arch). Builds the IR into the supplied (already
 * rocke_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set. `arch` NULL => "gfx950". This routine does NOT re-init
 * the builder (so the caller controls its lifetime). */
rocke_kernel_def_t* rocke_build_streamk_gemm(rocke_ir_builder_t* b,
                                             const rocke_streamk_gemm_spec_t* spec,
                                             const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_streamk_gemm_new(rocke_ir_builder_t* b,
                                                 const rocke_streamk_gemm_spec_t* spec,
                                                 const char* arch);

/* build_streamk_gemm_block_tile(spec, arch): dispatches into the build with
 * persistent forced true (dataclasses.replace(spec, persistent=True)). */
rocke_kernel_def_t* rocke_build_streamk_gemm_block_tile(rocke_ir_builder_t* b,
                                                        const rocke_streamk_gemm_spec_t* spec,
                                                        const char* arch);

/* streamk_gemm_grid(spec): launch grid. persistent=False => (num_macro_tiles,
 * 1, 1); persistent=True => (grid_size, 1, 1). Writes out[0..2]. Returns
 * ROCKE_OK, or ROCKE_ERR_VALUE on a degenerate partition. */
rocke_status_t rocke_streamk_gemm_grid(const rocke_streamk_gemm_spec_t* spec, int out[3]);

/* streamk_gemm_workspace_bytes(spec): 4 * M * N + 4. */
long rocke_streamk_gemm_workspace_bytes(const rocke_streamk_gemm_spec_t* spec);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
rocke_status_t rocke_streamk_gemm_lower_to_llvm(const rocke_streamk_gemm_spec_t* spec,
                                                const char* arch,
                                                rocke_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_STREAMK_GEMM_H */
