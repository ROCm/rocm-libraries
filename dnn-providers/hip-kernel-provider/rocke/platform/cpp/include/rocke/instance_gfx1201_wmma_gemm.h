/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1201_wmma_gemm.h -- C99 port of the gfx1201 (RDNA4 / Navi48)
 * WMMA GEMM kernel instance builder rocke/instances/gfx1201/wmma_gemm.py.
 *
 *   Python (gfx1201/wmma_gemm.py)         C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   class WmmaGemmSpec                    rocke_wmma_gemm_gfx1201_spec_t
 *   WmmaGemmSpec.block_size               rocke_wmma_gemm_gfx1201_block_size(spec)
 *   WmmaGemmSpec.kernel_name()            rocke_wmma_gemm_gfx1201_kernel_name(...)
 *   is_valid_spec(spec, arch)             rocke_wmma_gemm_gfx1201_is_valid_spec(...)
 *   build_wmma_gemm(spec, arch)           rocke_build_wmma_gemm_gfx1201(...)
 *   wmma_gemm_grid(M, N)                  rocke_wmma_gemm_gfx1201_grid(...)
 *   (+ convenience: build -> lower .ll)   rocke_wmma_gemm_gfx1201_lower_to_llvm(...)
 *
 * RDNA4 WMMA fragment ABI (vs RDNA3/3.5):
 *   * No cross-half operand duplication: A/B fragments are <8 x half> per lane;
 *     the 16 K-elements of one WMMA are split across the two lane-halves (lanes
 *     0-15 carry K 0..7, lanes 16-31 carry K 8..15). Per K-step (stride 16) lane
 *     l loads 8 contiguous elements starting at K = k0 + (l//16)*8.
 *   * Distinct intrinsic, selected via wmma_gfx12_f32_16x16x16_f16
 *     (rocke_b_wmma_gfx12_f32_16x16x16_f16).
 *   * Column-distributed accumulator: slot i of lane l maps to output
 *     (row = m0 + (l//16)*8 + i, col = n0 + l%16).
 *
 * Layout is RCR (C = A @ B.T, A row-major M*K, B row-major N*K), one wave (32
 * lanes) per 16x16 output tile, no LDS.
 *
 * The build reuses rocke_ir_builder_t methods (rocke_b_const_i32, rocke_b_mod,
 * rocke_b_div, rocke_b_mul, rocke_b_add, rocke_b_thread_id_x, rocke_b_block_id_x,
 * rocke_b_block_id_y, rocke_b_zero_vec_f32, rocke_b_scf_for_iter,
 * rocke_b_global_load_vN_f16, rocke_b_wmma_gfx12_f32_16x16x16_f16, rocke_b_scf_yield,
 * rocke_b_vec_extract, rocke_b_trunc_f32_to_f16, rocke_b_global_store, rocke_b_ret),
 * rocke/helper_rocke.core.arch.h for the is_valid_spec MMA-catalog gate +
 * wave_size, and rocke/helper_rocke.helpers.spec.h for kernel_name_join.
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python frozen dataclass has defaults; in C
 * the caller fills a rocke_wmma_gemm_gfx1201_spec_t.
 * rocke_wmma_gemm_gfx1201_spec_default() returns a struct with every field set to
 * the Python dataclass default.
 *
 * Error model mirrors the rest of the C port: build routes errors through the
 * sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + reason
 * string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_GFX1201_WMMA_GEMM_H
#define ROCKE_INSTANCE_GFX1201_WMMA_GEMM_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------- WmmaGemmSpec *
 *
 * Mirror of Python WmmaGemmSpec (frozen dataclass):
 *
 *     name: str = "rocke_wmma_gemm_gfx12"
 *     dtype: str = "fp16"        # fp16 only
 *
 * __post_init__ raises ValueError unless dtype == "fp16"; in C that check moves
 * into the validity gate / build (the caller may construct any struct). */
typedef struct rocke_wmma_gemm_gfx1201_spec
{
    const char* name; /* default "rocke_wmma_gemm_gfx12" */
    const char* dtype; /* default "fp16" (only) */
} rocke_wmma_gemm_gfx1201_spec_t;

/* Default-constructed spec (every field == Python dataclass default). */
rocke_wmma_gemm_gfx1201_spec_t rocke_wmma_gemm_gfx1201_spec_default(void);

/* WmmaGemmSpec.block_size @property: one wave32 == 32. */
int rocke_wmma_gemm_gfx1201_block_size(const rocke_wmma_gemm_gfx1201_spec_t* spec);

/* WmmaGemmSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 *
 *     kernel_name_join(self.name, "wmma16x16x16", self.dtype, "rcr")
 *
 * Returns ROCKE_OK, or ROCKE_ERR_VALUE (buffer too small / null args). */
rocke_status_t rocke_wmma_gemm_gfx1201_kernel_name(const rocke_wmma_gemm_gfx1201_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx1201".
 *
 * Gate (mirrors gfx1201/wmma_gemm.is_valid_spec):
 *   - ArchTarget.from_gfx(arch) must resolve (else the KeyError string).
 *   - the WMMA 16x16x16 (fp16,fp16,fp32) atom must exist in the target catalog.
 *   - target.wave_size must be 32 (wave32 kernel).
 *
 * On a reject, `reason` (if non-NULL, capacity reason_cap) receives the
 * structured message and false is returned. On accept returns true and writes
 * "ok". (dtype != "fp16" is rejected here too, mirroring __post_init__.) */
bool rocke_wmma_gemm_gfx1201_is_valid_spec(const rocke_wmma_gemm_gfx1201_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap);

/* build_wmma_gemm(spec, arch). Builds the IR into the supplied (already
 * rocke_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set. `arch` NULL => "gfx1201".
 *
 * Kernel signature: (A: ptr<f16>, B: ptr<f16>, C: ptr<f16>,
 *                    M: i32, N: i32, K: i32).
 * Grid: ((N+15)//16, (M+15)//16, 1). Block: 32 threads (one wave32). */
rocke_kernel_def_t* rocke_build_wmma_gemm_gfx1201(rocke_ir_builder_t* b,
                                                  const rocke_wmma_gemm_gfx1201_spec_t* spec,
                                                  const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_wmma_gemm_gfx1201_new(rocke_ir_builder_t* b,
                                                      const rocke_wmma_gemm_gfx1201_spec_t* spec,
                                                      const char* arch);

/* wmma_gemm_grid(M, N) -> ((N+15)//16, (M+15)//16, 1). Returns ROCKE_OK and
 * writes out[0..2]; ROCKE_ERR_VALUE on null out. */
rocke_status_t rocke_wmma_gemm_gfx1201_grid(int M, int N, int out[3]);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text
 * at arch=gfx1201 (or the supplied arch; NULL => "gfx1201"). On ROCKE_OK *out_ll
 * receives a malloc'd NUL-terminated string the caller frees with free(); on
 * failure it is left NULL and (if err != NULL, capacity err_cap) a diagnostic is
 * written. Internally owns and frees its IRBuilder. */
rocke_status_t rocke_wmma_gemm_gfx1201_lower_to_llvm(const rocke_wmma_gemm_gfx1201_spec_t* spec,
                                                     const char* arch,
                                                     rocke_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1201_WMMA_GEMM_H */
