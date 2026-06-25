/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_streamk_gemm.h -- C99 port of the StreamK GEMM kernel instance
 * builder ck_dsl/instances/common/streamk_gemm.py (CK Tile 40_streamk_gemm
 * parity).
 *
 *   Python (streamk_gemm.py)              C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   class StreamKGemmSpec                 ckc_streamk_gemm_spec_t
 *   is_valid_spec(spec, arch)             ckc_streamk_gemm_is_valid_spec(...)
 *   build_streamk_gemm(spec, arch)        ckc_build_streamk_gemm(...)
 *   streamk_gemm_grid(spec)               ckc_streamk_gemm_grid(...)
 *   streamk_gemm_workspace_bytes(spec)    ckc_streamk_gemm_workspace_bytes(...)
 *   build_streamk_gemm_block_tile(...)    ckc_build_streamk_gemm_block_tile(...)
 *   (+ convenience: build -> lower .ll)   ckc_streamk_gemm_lower_to_llvm(...)
 *
 * The Python @property values (partition, grid_size, atom, block_size,
 * persistent_max_iters, kernel_name) become pure accessor helpers; their
 * derivation is byte-faithful to the Python so the emitted op stream matches.
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (ckc_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a ckc_status_t.
 */
#ifndef CKC_INSTANCE_STREAMK_GEMM_H
#define CKC_INSTANCE_STREAMK_GEMM_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/helper_ck_dsl.helpers.atoms.h" /* ckc_mfma_atom_t */
#include "ckc/helper_ck_dsl.helpers.streamk.h" /* ckc_streamk_partition_t, strategy */
#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

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
typedef struct ckc_streamk_gemm_spec
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
    ckc_streamk_reduction_strategy_t reduction; /* default Atomic */
    bool persistent; /* default false */
    const char* name; /* default "ck_dsl_streamk_gemm" */
} ckc_streamk_gemm_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still set the required M/N/K geometry. */
ckc_streamk_gemm_spec_t ckc_streamk_gemm_spec_default(void);

/* @property partition: StreamKPartition(m_tiles=M//tile_m, n_tiles=N//tile_n,
 * k_iters=K//tile_k). On the Python ValueError (M/N/K not divisible by their
 * tile sizes) returns false and leaves *out untouched; else writes *out and
 * returns true. */
bool ckc_streamk_gemm_partition(const ckc_streamk_gemm_spec_t* spec, ckc_streamk_partition_t* out);

/* @property atom: the square f16 MFMA atom for (tile_m, tile_n):
 *   (16,16) -> f16_16x16x16 ; (32,32) -> f16_32x32x8.
 * Returns a pointer into the static MFMA catalog, or NULL on the Python
 * ValueError (unsupported tile shape). */
const ckc_mfma_atom_t* ckc_streamk_gemm_atom(const ckc_streamk_gemm_spec_t* spec);

/* @property grid_size: compute_streamk_grid_size(partition, num_cus,
 * blocks_per_cu). Returns -1 on the Python ValueError path (degenerate
 * partition / zero macro tiles). */
int ckc_streamk_gemm_grid_size(const ckc_streamk_gemm_spec_t* spec);

/* @property block_size: 64 (one wave64 warp per CTA). */
int ckc_streamk_gemm_block_size(const ckc_streamk_gemm_spec_t* spec);

/* @property persistent_max_iters: ceil(num_macro_tiles / grid_size). Returns
 * -1 on a degenerate partition (grid_size <= 0). */
int ckc_streamk_gemm_persistent_max_iters(const ckc_streamk_gemm_spec_t* spec);

/* StreamKGemmSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 * Returns CKC_OK or CKC_ERR_VALUE (buffer too small / degenerate spec). */
ckc_status_t
    ckc_streamk_gemm_kernel_name(const ckc_streamk_gemm_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message; returns false. On accept returns true and writes "ok". */
bool ckc_streamk_gemm_is_valid_spec(const ckc_streamk_gemm_spec_t* spec,
                                    const char* arch,
                                    char* reason,
                                    size_t reason_cap);

/* build_streamk_gemm(spec, arch). Builds the IR into the supplied (already
 * ckc_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set. `arch` NULL => "gfx950". This routine does NOT re-init
 * the builder (so the caller controls its lifetime). */
ckc_kernel_def_t* ckc_build_streamk_gemm(ckc_ir_builder_t* b,
                                         const ckc_streamk_gemm_spec_t* spec,
                                         const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with ckc_ir_builder_free(). Returns the kernel or NULL. */
ckc_kernel_def_t* ckc_build_streamk_gemm_new(ckc_ir_builder_t* b,
                                             const ckc_streamk_gemm_spec_t* spec,
                                             const char* arch);

/* build_streamk_gemm_block_tile(spec, arch): dispatches into the build with
 * persistent forced true (dataclasses.replace(spec, persistent=True)). */
ckc_kernel_def_t* ckc_build_streamk_gemm_block_tile(ckc_ir_builder_t* b,
                                                    const ckc_streamk_gemm_spec_t* spec,
                                                    const char* arch);

/* streamk_gemm_grid(spec): launch grid. persistent=False => (num_macro_tiles,
 * 1, 1); persistent=True => (grid_size, 1, 1). Writes out[0..2]. Returns
 * CKC_OK, or CKC_ERR_VALUE on a degenerate partition. */
ckc_status_t ckc_streamk_gemm_grid(const ckc_streamk_gemm_spec_t* spec, int out[3]);

/* streamk_gemm_workspace_bytes(spec): 4 * M * N + 4. */
long ckc_streamk_gemm_workspace_bytes(const ckc_streamk_gemm_spec_t* spec);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On CKC_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
ckc_status_t ckc_streamk_gemm_lower_to_llvm(const ckc_streamk_gemm_spec_t* spec,
                                            const char* arch,
                                            ckc_llvm_flavor_t flavor,
                                            char** out_ll,
                                            char* err,
                                            size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_STREAMK_GEMM_H */
