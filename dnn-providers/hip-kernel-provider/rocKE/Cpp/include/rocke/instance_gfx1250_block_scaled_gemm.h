/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1250_block_scaled_gemm.h -- C99 port of the gfx1250
 * block-scaled low-bit (FP8/BF8 K=64 WMMA) dense GEMM
 * rocke/instances/gfx1250/block_scaled_gemm.py.
 *
 *   Python (gfx1250/block_scaled_gemm.py)        C99 (this header)
 *   ------------------------------------------   ------------------------------------------
 *   class BlockScaledGemmSpec                    rocke_block_scaled_gemm_gfx1250_spec_t
 *   BlockScaledGemmSpec.block_size               rocke_block_scaled_gemm_gfx1250_block_size()
 *   BlockScaledGemmSpec.kernel_name()            rocke_block_scaled_gemm_gfx1250_kernel_name()
 *   BlockScaledGemmSpec.resolved_matrix_path()   rocke_block_scaled_gemm_gfx1250_resolved_path()
 *   is_valid_spec(spec, arch)                    rocke_block_scaled_gemm_gfx1250_is_valid_spec()
 *   build_block_scaled_gemm(spec, arch)          rocke_build_block_scaled_gemm_gfx1250()
 *   block_scaled_gemm_grid(spec)                 rocke_block_scaled_gemm_gfx1250_grid()
 *
 * RCR (C = A @ B^T), one wave (32 lanes) per 16x16 output tile, no LDS. The K
 * loop runs in block_k groups; each group accumulates block_k/64 WMMA K=64 steps
 * into a fresh <8 x f32> acc, then applies per-block scales. Fragment: lane l
 * carries 32 low-bit bytes (<8 x i32>) starting at K = k0 + (l//16)*32; acc is
 * the column-distributed gfx12 <8 x f32>.
 *
 * gfx1250-ONLY (validity gate requires arch=="gfx1250", WMMA-only, no MFMA).
 */
#ifndef ROCKE_INSTANCE_GFX1250_BLOCK_SCALED_GEMM_H
#define ROCKE_INSTANCE_GFX1250_BLOCK_SCALED_GEMM_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of Python BlockScaledGemmSpec (frozen dataclass):
 *     name: str            (required, no default)
 *     M, N, K: int         (required, no default)
 *     dtype_a: str = "fp8"   # fp8/fp8e4m3/bf8/bf8e5m2
 *     dtype_b: str = "fp8"
 *     dtype_c: str = "bf16"  # bf16/fp16/f16
 *     dtype_acc: str = "fp32"
 *     scale_dtype: str = "fp32"  # fp16/fp32
 *     block_k: int = 128
 *     layout: str = "RCR"
 *     matrix_path: str = "auto"  # auto/wmma/wmma_scaffold/mfma
 *     tile_m: int = 16
 *     tile_n: int = 16
 *     tile_k: int = 128 */
typedef struct rocke_block_scaled_gemm_gfx1250_spec
{
    const char* name;
    int M;
    int N;
    int K;
    const char* dtype_a;
    const char* dtype_b;
    const char* dtype_c;
    const char* dtype_acc;
    const char* scale_dtype;
    int block_k;
    const char* layout;
    const char* matrix_path;
    int tile_m;
    int tile_n;
    int tile_k;
} rocke_block_scaled_gemm_gfx1250_spec_t;

/* Default-constructed spec. name/M/N/K have NO Python default; caller MUST set
 * them (defaults are NULL/0, which the validity gate rejects). */
rocke_block_scaled_gemm_gfx1250_spec_t rocke_block_scaled_gemm_gfx1250_spec_default(void);

/* BlockScaledGemmSpec.block_size @property: one wave32 == 32. */
int rocke_block_scaled_gemm_gfx1250_block_size(const rocke_block_scaled_gemm_gfx1250_spec_t* spec);

/* resolved_matrix_path(): auto/wmma_scaffold -> "wmma"; else matrix_path. */
const char* rocke_block_scaled_gemm_gfx1250_resolved_path(
    const rocke_block_scaled_gemm_gfx1250_spec_t* spec);

/* BlockScaledGemmSpec.kernel_name():
 *   kernel_name_join(name, "block_scaled", f"{a}_{b}", f"M..N..K..", f"bk..",
 *                    f"t{tm}x{tn}x{tk}", flags={"wmma": resolved=="wmma"}). */
rocke_status_t rocke_block_scaled_gemm_gfx1250_kernel_name(
    const rocke_block_scaled_gemm_gfx1250_spec_t* spec, char* out, size_t out_cap);

bool rocke_block_scaled_gemm_gfx1250_is_valid_spec(
    const rocke_block_scaled_gemm_gfx1250_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap);

/* build_block_scaled_gemm(spec, arch). Signature:
 *   (A: ptr<a>, B: ptr<b>, A_scale: ptr<scale>, B_scale: ptr<scale>, C: ptr<c>,
 *    M, N, K: i32). */
rocke_kernel_def_t* rocke_build_block_scaled_gemm_gfx1250(
    rocke_ir_builder_t* b, const rocke_block_scaled_gemm_gfx1250_spec_t* spec, const char* arch);
rocke_kernel_def_t* rocke_build_block_scaled_gemm_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_block_scaled_gemm_gfx1250_spec_t* spec, const char* arch);

/* block_scaled_gemm_grid(spec) -> ceil_div_grid((N,tile_n),(M,tile_m)). */
rocke_status_t
    rocke_block_scaled_gemm_gfx1250_grid(const rocke_block_scaled_gemm_gfx1250_spec_t* spec,
                                         int out[3]);

rocke_status_t rocke_block_scaled_gemm_gfx1250_lower_to_llvm(
    const rocke_block_scaled_gemm_gfx1250_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1250_BLOCK_SCALED_GEMM_H */
