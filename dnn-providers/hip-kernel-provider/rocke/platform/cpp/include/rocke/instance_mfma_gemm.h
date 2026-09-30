/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_mfma_gemm.h -- C99 port of the MFMA-tiled GEMM kernel instance
 * builder rocke/instances/common/mfma_gemm.py (the first K-packed MFMA
 * instance: one atom per CTA, no LDS staging).
 *
 *   Python (mfma_gemm.py)                 C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   class MfmaGemmSpec                    rocke_mfma_gemm_spec_t
 *   MfmaGemmSpec.atom                     rocke_mfma_gemm_atom(spec)
 *   MfmaGemmSpec.tile_k                   rocke_mfma_gemm_tile_k(spec)
 *   MfmaGemmSpec.block_size               rocke_mfma_gemm_block_size(spec) [== 64]
 *   MfmaGemmSpec.kernel_name()            rocke_mfma_gemm_kernel_name(...)
 *   is_valid_spec(spec, arch)             rocke_mfma_gemm_is_valid_spec(...)
 *   build_mfma_gemm(spec, arch)           rocke_build_mfma_gemm(...)
 *   mfma_gemm_grid(spec)                  rocke_mfma_gemm_grid(...)
 *   (+ convenience: build -> lower .ll)   rocke_mfma_gemm_lower_to_llvm(...)
 *
 * The build directly reuses the seven ported helpers from
 * rocke/helper_rocke.helpers.mfma_gemm_inner.h (decode_mfma_lanes,
 * load_a_row_major_contiguous, load_b_col_strided_scalars, mfma_atom_for_dtype,
 * mfma_k_loop, store_acc_to_global) and the MfmaAtom catalog from
 * rocke/helper_rocke.helpers.atoms.h, plus rocke/helper_rocke.core.arch.h for the
 * is_valid_spec MMA-catalog gate and rocke/helper_rocke.helpers.spec.h for
 * kernel_name_join.
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python frozen dataclass has defaults; in C
 * the caller fills a rocke_mfma_gemm_spec_t. rocke_mfma_gemm_spec_default() returns
 * a struct with every field set to the Python dataclass default; the caller
 * then sets M / N / K (required) and overrides the rest as needed.
 *
 * Error model mirrors the rest of the C port: build routes errors through the
 * sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + reason
 * string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_MFMA_GEMM_H
#define ROCKE_INSTANCE_MFMA_GEMM_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- MfmaGemmSpec *
 *
 * Mirror of Python MfmaGemmSpec (frozen dataclass):
 *
 *     M: int
 *     N: int
 *     K: int
 *     dtype: DType = "f16"        # "f16" | "bf16"
 *     tile_m: int = 16
 *     tile_n: int = 16
 *     kpack: bool = True
 *     name: str = "rocke_mfma_gemm"
 *
 * M / N / K are compile-time so the partitioner can statically derive the grid
 * and the K-loop trip count. tile_m / tile_n in {16, 32} pick the MFMA hero
 * shape; kpack picks the K-packed atom (gfx950+) vs the legacy atom. */
typedef struct rocke_mfma_gemm_spec
{
    int M;
    int N;
    int K;
    const char* dtype; /* default "f16" */
    int tile_m; /* default 16 */
    int tile_n; /* default 16 */
    bool kpack; /* default true */
    const char* name; /* default "rocke_mfma_gemm" */
} rocke_mfma_gemm_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still set M / N / K. */
rocke_mfma_gemm_spec_t rocke_mfma_gemm_spec_default(void);

/* MfmaGemmSpec.atom @property: the dispatched MfmaAtom for
 * (dtype, tile_m, tile_n, prefer_packed_k=kpack). Returns a pointer into the
 * static MFMA catalog (do NOT free/mutate), or NULL on the Python ValueError
 * path (unsupported dtype/shape). */
const rocke_mfma_atom_t* rocke_mfma_gemm_atom(const rocke_mfma_gemm_spec_t* spec);

/* MfmaGemmSpec.tile_k @property: atom.k (0 if the atom is unresolved). */
int rocke_mfma_gemm_tile_k(const rocke_mfma_gemm_spec_t* spec);

/* MfmaGemmSpec.block_size @property: one wave64 warp per CTA == 64. */
int rocke_mfma_gemm_block_size(const rocke_mfma_gemm_spec_t* spec);

/* MfmaGemmSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 *
 *     kernel_name_join(self.name, f"M{M}N{N}K{K}", self.dtype,
 *                      f"atom{m}x{n}x{k}", flags={"kpack": self.kpack})
 *
 * Returns ROCKE_OK, or ROCKE_ERR_VALUE (buffer too small / unresolved atom). */
rocke_status_t
    rocke_mfma_gemm_kernel_name(const rocke_mfma_gemm_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950".
 *
 * Gate (mirrors mfma_gemm.is_valid_spec):
 *   - dtype in ("f16", "bf16")
 *   - (tile_m, tile_n) in ((16,16), (32,32))
 *   - M % atom.m == 0, N % atom.n == 0, K % atom.k == 0
 *   - the (dtype, atom shape) is present in the target's MMA catalog
 *     (K-packed atoms exist only on gfx950+).
 *
 * On a reject, `reason` (if non-NULL, capacity reason_cap) receives the
 * structured message and false is returned. On accept returns true and writes
 * "ok". */
bool rocke_mfma_gemm_is_valid_spec(const rocke_mfma_gemm_spec_t* spec,
                                   const char* arch,
                                   char* reason,
                                   size_t reason_cap);

/* build_mfma_gemm(spec, arch). Builds the IR into the supplied (already
 * rocke_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set. `arch` NULL => "gfx950".
 *
 * Kernel signature: (A: ptr<dtype>, B: ptr<dtype>, C: ptr<dtype>,
 *                    M: i32, N: i32, K: i32).
 * Grid: (N // atom.n, M // atom.m, 1). Block: 64 threads (one wave64 warp). */
rocke_kernel_def_t* rocke_build_mfma_gemm(rocke_ir_builder_t* b,
                                          const rocke_mfma_gemm_spec_t* spec,
                                          const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_mfma_gemm_new(rocke_ir_builder_t* b,
                                              const rocke_mfma_gemm_spec_t* spec,
                                              const char* arch);

/* mfma_gemm_grid(spec) -> (N // atom.n, M // atom.m, 1). Returns ROCKE_OK and
 * writes out[0..2]; ROCKE_ERR_VALUE on an unresolved atom. */
rocke_status_t rocke_mfma_gemm_grid(const rocke_mfma_gemm_spec_t* spec, int out[3]);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err != NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
rocke_status_t rocke_mfma_gemm_lower_to_llvm(const rocke_mfma_gemm_spec_t* spec,
                                             const char* arch,
                                             rocke_llvm_flavor_t flavor,
                                             char** out_ll,
                                             char* err,
                                             size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_MFMA_GEMM_H */
