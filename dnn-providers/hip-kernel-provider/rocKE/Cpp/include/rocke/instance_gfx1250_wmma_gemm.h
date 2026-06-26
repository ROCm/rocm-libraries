/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1250_wmma_gemm.h -- C99 port of the gfx1250 (CDNA, GFX12
 * programming model) WMMA GEMM kernel rocke/instances/gfx1250/wmma_gemm.py.
 *
 *   Python (gfx1250/wmma_gemm.py)         C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   class WmmaGemmSpec                    rocke_wmma_gemm_gfx1250_spec_t
 *   WmmaGemmSpec.block_size               rocke_wmma_gemm_gfx1250_block_size(spec)
 *   WmmaGemmSpec.kernel_name()            rocke_wmma_gemm_gfx1250_kernel_name(...)
 *   is_valid_spec(spec, arch)             rocke_wmma_gemm_gfx1250_is_valid_spec(...)
 *   build_wmma_gemm(spec, arch)           rocke_build_wmma_gemm_gfx1250(...)
 *   wmma_gemm_grid(M, N)                  rocke_wmma_gemm_gfx1250_grid(...)
 *
 * gfx1250 WMMA fragment ABI (K=32, vs gfx1201's K=16):
 *   * A/B fragments are <16 x half> per lane; the 32 K-elements of one WMMA are
 *     split across the two lane-halves (lanes 0-15 carry K 0..15, lanes 16-31
 *     carry K 16..31). Per K-step (stride 32) lane l loads 16 contiguous
 *     elements starting at K = k0 + (l//16)*16.
 *   * 8-operand intrinsic via wmma_gfx1250_f32_16x16x32_f16
 *     (rocke_b_wmma_gfx1250_f32_16x16x32_f16).
 *   * Column-distributed accumulator: slot i of lane l maps to output
 *     (row = m0 + (l//16)*8 + i, col = n0 + l%16).
 *
 * Layout is RCR (C = A @ B.T), one wave (32 lanes) per 16x16 output tile, no LDS.
 */
#ifndef ROCKE_INSTANCE_GFX1250_WMMA_GEMM_H
#define ROCKE_INSTANCE_GFX1250_WMMA_GEMM_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of Python WmmaGemmSpec (frozen dataclass):
 *     name: str = "rocke_wmma_gemm_gfx1250"
 *     dtype: str = "fp16"        # fp16 only
 * __post_init__ raises unless dtype == "fp16"; in C that moves into the validity
 * gate / build. */
typedef struct rocke_wmma_gemm_gfx1250_spec
{
    const char* name; /* default "rocke_wmma_gemm_gfx1250" */
    const char* dtype; /* default "fp16" (only) */
} rocke_wmma_gemm_gfx1250_spec_t;

/* Default-constructed spec (every field == Python dataclass default). */
rocke_wmma_gemm_gfx1250_spec_t rocke_wmma_gemm_gfx1250_spec_default(void);

/* WmmaGemmSpec.block_size @property: one wave32 == 32. */
int rocke_wmma_gemm_gfx1250_block_size(const rocke_wmma_gemm_gfx1250_spec_t* spec);

/* WmmaGemmSpec.kernel_name():
 *   kernel_name_join(self.name, "wmma16x16x32", self.dtype, "rcr"). */
rocke_status_t rocke_wmma_gemm_gfx1250_kernel_name(const rocke_wmma_gemm_gfx1250_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx1250".
 *   - ArchTarget.from_gfx(arch) must resolve.
 *   - the WMMA 16x16x32 (fp16,fp16,fp32) atom must exist in the target catalog.
 *   - target.wave_size must be 32. (dtype != "fp16" rejected too.) */
bool rocke_wmma_gemm_gfx1250_is_valid_spec(const rocke_wmma_gemm_gfx1250_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap);

/* build_wmma_gemm(spec, arch). Builds into the supplied builder `b` (already
 * rocke_ir_builder_init'd with spec.kernel_name()); returns b->kernel or NULL
 * with b's sticky error set. `arch` NULL => "gfx1250".
 *
 * Kernel signature: (A,B,C: ptr<f16>, M,N,K: i32).
 * Grid: ((N+15)//16, (M+15)//16, 1). Block: 32 threads (one wave32). */
rocke_kernel_def_t* rocke_build_wmma_gemm_gfx1250(rocke_ir_builder_t* b,
                                                  const rocke_wmma_gemm_gfx1250_spec_t* spec,
                                                  const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. Caller owns `b`. */
rocke_kernel_def_t* rocke_build_wmma_gemm_gfx1250_new(rocke_ir_builder_t* b,
                                                      const rocke_wmma_gemm_gfx1250_spec_t* spec,
                                                      const char* arch);

/* wmma_gemm_grid(M, N) -> ((N+15)//16, (M+15)//16, 1). */
rocke_status_t rocke_wmma_gemm_gfx1250_grid(int M, int N, int out[3]);

/* Convenience: build + lower to .ll at arch (NULL => "gfx1250"). */
rocke_status_t rocke_wmma_gemm_gfx1250_lower_to_llvm(const rocke_wmma_gemm_gfx1250_spec_t* spec,
                                                     const char* arch,
                                                     rocke_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1250_WMMA_GEMM_H */
