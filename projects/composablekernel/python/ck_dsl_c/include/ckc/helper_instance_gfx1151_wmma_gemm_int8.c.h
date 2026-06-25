/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_instance_gfx1151_wmma_gemm_int8.c.h -- C99 port of the gfx1151
 * (RDNA3.5 / Strix Halo) INT8-storage / f16-compute WMMA GEMM kernel instance
 * builder ck_dsl/instances/gfx1151/wmma_gemm_int8.py.
 *
 *   Python (gfx1151/wmma_gemm_int8.py)        C99 (this header)
 *   ---------------------------------------   ------------------------------------
 *   class WmmaGemmInt8Spec                    ckc_wmma_gemm_int8_spec_t
 *   WmmaGemmInt8Spec.block_size               ckc_wmma_gemm_int8_block_size(spec)
 *   WmmaGemmInt8Spec.kernel_name()            ckc_wmma_gemm_int8_kernel_name(...)
 *   is_valid_spec(spec, arch)                 ckc_wmma_gemm_int8_is_valid_spec(...)
 *   build_wmma_gemm_int8(spec, arch)          ckc_build_wmma_gemm_int8(...)
 *   wmma_gemm_int8_grid(M, N)                 ckc_wmma_gemm_int8_grid(...)
 *   (+ convenience: build -> lower .ll)       ckc_wmma_gemm_int8_lower_to_llvm(...)
 *
 * Operands are stored as symmetric per-tensor int8 (A, B). Each lane fragment is
 * sign-extended -> f32 -> f16 (lossless for |x| <= 127) and fed to the existing,
 * hardware-verified wmma_f32_16x16x16_f16 (RDNA3.5) path with f32 accumulate.
 * The combined dequant scale (scale_a * scale_b) is folded into the epilogue,
 * one multiply per output element, before truncating to f16.
 *
 * Layout matches gfx1151/wmma_gemm.py exactly (RCR: A row-major M*K, B row-major
 * N*K, C = A @ B.T), grid (ceil(M/16), ceil(N/16)) with block_id.x -> M-tile and
 * block_id.y -> N-tile, one wave (32 lanes) per 16x16 tile.
 *
 * Kernel signature:
 *   (A: ptr<i8>, B: ptr<i8>, C: ptr<f16>,
 *    M: i32, N: i32, K: i32, scale_a: f32, scale_b: f32).
 *
 * The build reuses ckc_ir_builder_t methods (ckc_b_const_i32, ckc_b_mod,
 * ckc_b_div, ckc_b_mul, ckc_b_add, ckc_b_fmul, ckc_b_thread_id_x,
 * ckc_b_block_id_x, ckc_b_block_id_y, ckc_b_zero_vec_f32, ckc_b_scf_for_iter,
 * ckc_b_global_load_vN, ckc_b_vec_pack, ckc_b_vec_extract, ckc_b_sext,
 * ckc_b_sitofp_f32, ckc_b_cast_f32_to, ckc_b_wmma_f32_16x16x16_f16,
 * ckc_b_scf_yield, ckc_b_trunc_f32_to_f16, ckc_b_global_store, ckc_b_ret),
 * ckc/helper_ck_dsl.core.arch.h for the is_valid_spec MMA-catalog gate +
 * wave_size, and ckc/helper_ck_dsl.helpers.spec.h for kernel_name_join.
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python frozen dataclass has defaults; in C
 * the caller fills a ckc_wmma_gemm_int8_spec_t.
 * ckc_wmma_gemm_int8_spec_default() returns a struct with every field set to the
 * Python dataclass default.
 *
 * Error model mirrors the rest of the C port: build routes errors through the
 * sticky-error IRBuilder (ckc_b_*); the validity gate returns a bool + reason
 * string; the convenience lower returns a ckc_status_t.
 */
#ifndef CKC_HELPER_INSTANCE_GFX1151_WMMA_GEMM_INT8_H
#define CKC_HELPER_INSTANCE_GFX1151_WMMA_GEMM_INT8_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------- WmmaGemmInt8Spec *
 *
 * Mirror of Python WmmaGemmInt8Spec (frozen dataclass):
 *
 *     name: str = "ck_dsl_wmma_gemm_int8"
 *     dtype: str = "i8"          # int8 storage only
 *
 * __post_init__ raises ValueError unless dtype == "i8"; in C that check moves
 * into the validity gate / build (the caller may construct any struct). */
typedef struct ckc_wmma_gemm_int8_spec
{
    const char* name;  /* default "ck_dsl_wmma_gemm_int8" */
    const char* dtype; /* default "i8" (only) */
} ckc_wmma_gemm_int8_spec_t;

/* Default-constructed spec (every field == Python dataclass default). */
ckc_wmma_gemm_int8_spec_t ckc_wmma_gemm_int8_spec_default(void);

/* WmmaGemmInt8Spec.block_size @property: one wave32 == 32. */
int ckc_wmma_gemm_int8_block_size(const ckc_wmma_gemm_int8_spec_t* spec);

/* WmmaGemmInt8Spec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 *
 *     kernel_name_join(self.name, "wmma16x16x16", "i8_f16", "rcr")
 *
 * Returns CKC_OK, or CKC_ERR_VALUE (buffer too small / null args). */
ckc_status_t
ckc_wmma_gemm_int8_kernel_name(const ckc_wmma_gemm_int8_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx1151".
 *
 * Gate (mirrors gfx1151/wmma_gemm_int8.is_valid_spec):
 *   - ArchTarget.from_gfx(arch) must resolve (else the KeyError string).
 *   - the WMMA 16x16x16 (fp16,fp16,fp32) *compute* atom must exist in the target
 *     catalog (operands are int8 in memory but dequantized to f16 before MMA).
 *   - target.wave_size must be 32 (wave32 kernel).
 *
 * On a reject, `reason` (if non-NULL, capacity reason_cap) receives the
 * structured message and false is returned. On accept returns true and writes
 * "ok". (dtype != "i8" is rejected here too, mirroring __post_init__.) */
bool ckc_wmma_gemm_int8_is_valid_spec(const ckc_wmma_gemm_int8_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap);

/* build_wmma_gemm_int8(spec, arch). Builds the IR into the supplied (already
 * ckc_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set. `arch` NULL => "gfx1151".
 *
 * Grid: ((M+15)//16, (N+15)//16, 1). Block: 32 threads (one wave32). */
ckc_kernel_def_t* ckc_build_wmma_gemm_int8(ckc_ir_builder_t* b,
                                           const ckc_wmma_gemm_int8_spec_t* spec,
                                           const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with ckc_ir_builder_free(). Returns the kernel or NULL. */
ckc_kernel_def_t* ckc_build_wmma_gemm_int8_new(ckc_ir_builder_t* b,
                                               const ckc_wmma_gemm_int8_spec_t* spec,
                                               const char* arch);

/* wmma_gemm_int8_grid(M, N) -> ((M+15)//16, (N+15)//16, 1). Returns CKC_OK and
 * writes out[0..2]; CKC_ERR_VALUE on null out. */
ckc_status_t ckc_wmma_gemm_int8_grid(int M, int N, int out[3]);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text
 * at arch=gfx1151 (or the supplied arch; NULL => "gfx1151"). On CKC_OK *out_ll
 * receives a malloc'd NUL-terminated string the caller frees with free(); on
 * failure it is left NULL and (if err != NULL, capacity err_cap) a diagnostic is
 * written. Internally owns and frees its IRBuilder. */
ckc_status_t ckc_wmma_gemm_int8_lower_to_llvm(const ckc_wmma_gemm_int8_spec_t* spec,
                                              const char* arch,
                                              ckc_llvm_flavor_t flavor,
                                              char** out_ll,
                                              char* err,
                                              size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_INSTANCE_GFX1151_WMMA_GEMM_INT8_H */
