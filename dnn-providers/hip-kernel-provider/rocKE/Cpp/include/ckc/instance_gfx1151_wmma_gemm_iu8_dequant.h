/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_gfx1151_wmma_gemm_iu8_dequant.h -- C99 port of the gfx1151
 * (RDNA3.5 / Strix Halo) true-INT8 WMMA GEMM with f16 dequant output kernel
 * instance builder ck_dsl/instances/gfx1151/wmma_gemm_iu8_dequant.py.
 *
 *   Python (gfx1151/wmma_gemm_iu8_dequant.py)   C99 (this header)
 *   -----------------------------------------   ----------------------------------
 *   class WmmaGemmIu8DequantSpec                ckc_wmma_gemm_iu8_dequant_spec_t
 *   .block_size                                 ckc_wmma_gemm_iu8_dequant_block_size
 *   .kernel_name()                              ckc_wmma_gemm_iu8_dequant_kernel_name
 *   is_valid_spec(spec, arch)                   ckc_wmma_gemm_iu8_dequant_is_valid_spec
 *   build_wmma_gemm_iu8_dequant(spec, arch)     ckc_build_wmma_gemm_iu8_dequant
 *   wmma_gemm_iu8_dequant_grid(M, N)            ckc_wmma_gemm_iu8_dequant_grid
 *   (+ convenience: build -> lower .ll)         ckc_wmma_gemm_iu8_dequant_lower_to_llvm
 *
 * This kernel runs the SAME hardware ``wmma_i32_16x16x16_iu8`` instruction
 * (int8 x int8 -> int32 accumulate) as the native-int sibling, but applies a
 * per-tensor symmetric dequant in the epilogue and stores f16. It is the
 * true-int8-compute counterpart to wmma_gemm_int8.py (int8 storage, f16 compute).
 *
 * Operand ABI (verbatim from the iu8 sibling):
 *   * A/B are int8 logically but PASSED PACKED AS i32 (4 int8 per i32; slot j
 *     holds K=[4j..4j+3]). A/B pointers are i32. C is f16.
 *   * Each lane's WMMA operand fragment is <4 x i32> (16 int8); the accumulator
 *     is a loop-carried <8 x i32> (slot i -> row m0 + 2*i + l/16, col n0 + l%16).
 *   * Epilogue: f16 = trunc(sitofp(acc_i32) * (scale_a * scale_b)).
 *
 * Layout is RCR (C = A @ B.T, A row-major M*K int8, B row-major N*K int8), one
 * wave (32 lanes) per 16x16 output tile, no LDS.
 *
 * The build reuses ckc_ir_builder_t methods (ckc_b_const_i32, ckc_b_mod,
 * ckc_b_div, ckc_b_mul, ckc_b_add, ckc_b_fmul, ckc_b_thread_id_x,
 * ckc_b_block_id_x, ckc_b_block_id_y, ckc_b_zero_vec, ckc_b_scf_for_iter,
 * ckc_b_global_load_vN (i32), ckc_b_mma, ckc_b_scf_yield, ckc_b_vec_extract,
 * ckc_b_sitofp_f32, ckc_b_trunc_f32_to_f16, ckc_b_global_store, ckc_b_ret),
 * ckc/helper_ck_dsl.core.arch.h for the is_valid_spec MMA-catalog gate +
 * wave_size, and ckc/helper_ck_dsl.helpers.spec.h for kernel_name_join.
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python frozen dataclass has defaults; in C
 * the caller fills a ckc_wmma_gemm_iu8_dequant_spec_t.
 * ckc_wmma_gemm_iu8_dequant_spec_default() returns a struct with every field set
 * to the Python dataclass default.
 *
 * Error model mirrors the rest of the C port: build routes errors through the
 * sticky-error IRBuilder (ckc_b_*); the validity gate returns a bool + reason
 * string; the convenience lower returns a ckc_status_t.
 */
#ifndef CKC_INSTANCE_GFX1151_WMMA_GEMM_IU8_DEQUANT_H
#define CKC_INSTANCE_GFX1151_WMMA_GEMM_IU8_DEQUANT_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------- WmmaGemmIu8DequantSpec *
 *
 * Mirror of Python WmmaGemmIu8DequantSpec (frozen dataclass):
 *
 *     name: str = "ck_dsl_wmma_gemm_iu8_dequant"
 *
 * There is no dtype field / __post_init__: int8-in / i32-acc / f16-out is fixed.
 * The runtime per-tensor symmetric scales (scale_a, scale_b) are kernel args. */
typedef struct ckc_wmma_gemm_iu8_dequant_spec
{
    const char* name; /* default "ck_dsl_wmma_gemm_iu8_dequant" */
} ckc_wmma_gemm_iu8_dequant_spec_t;

/* Default-constructed spec (every field == Python dataclass default). */
ckc_wmma_gemm_iu8_dequant_spec_t ckc_wmma_gemm_iu8_dequant_spec_default(void);

/* WmmaGemmIu8DequantSpec.block_size @property: one wave32 == 32. */
int ckc_wmma_gemm_iu8_dequant_block_size(const ckc_wmma_gemm_iu8_dequant_spec_t* spec);

/* WmmaGemmIu8DequantSpec.kernel_name() -> NUL-terminated into out (out_cap).
 *
 *     kernel_name_join(self.name, "wmma16x16x16", "iu8_f16", "rcr")
 *
 * Returns CKC_OK, or CKC_ERR_VALUE (buffer too small / null args). */
ckc_status_t ckc_wmma_gemm_iu8_dequant_kernel_name(const ckc_wmma_gemm_iu8_dequant_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx1151".
 *
 * Gate (mirrors gfx1151/wmma_gemm_iu8_dequant.is_valid_spec):
 *   - ArchTarget.from_gfx(arch) must resolve (else the KeyError string).
 *   - the wmma_i32_16x16x16_iu8 atom must exist (target.mma.by_op_id(_OP_ID)).
 *   - target.wave_size must be 32 (wave32 kernel).
 *
 * On a reject, `reason` (if non-NULL, capacity reason_cap) receives the
 * structured message and false is returned. On accept returns true and writes
 * "ok". */
bool ckc_wmma_gemm_iu8_dequant_is_valid_spec(const ckc_wmma_gemm_iu8_dequant_spec_t* spec,
                                             const char* arch,
                                             char* reason,
                                             size_t reason_cap);

/* build_wmma_gemm_iu8_dequant(spec, arch). Builds the IR into the supplied
 * (already ckc_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly
 * as the Python build does, and returns the kernel (b->kernel) on success or
 * NULL with b's sticky error set. `arch` NULL => "gfx1151".
 *
 * Kernel signature: (A: ptr<i32>, B: ptr<i32>, C: ptr<f16>,
 *                    M: i32, N: i32, K: i32, scale_a: f32, scale_b: f32).
 * Grid: ((N+15)//16, (M+15)//16, 1). Block: 32 threads (one wave32). */
ckc_kernel_def_t* ckc_build_wmma_gemm_iu8_dequant(ckc_ir_builder_t* b,
                                                  const ckc_wmma_gemm_iu8_dequant_spec_t* spec,
                                                  const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with ckc_ir_builder_free(). Returns the kernel or NULL. */
ckc_kernel_def_t* ckc_build_wmma_gemm_iu8_dequant_new(ckc_ir_builder_t* b,
                                                      const ckc_wmma_gemm_iu8_dequant_spec_t* spec,
                                                      const char* arch);

/* wmma_gemm_iu8_dequant_grid(M, N) -> ((M+15)//16, (N+15)//16, 1). Returns
 * CKC_OK and writes out[0..2]; CKC_ERR_VALUE on null out. */
ckc_status_t ckc_wmma_gemm_iu8_dequant_grid(int M, int N, int out[3]);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text
 * at arch=gfx1151 (or the supplied arch; NULL => "gfx1151"). On CKC_OK *out_ll
 * receives a malloc'd NUL-terminated string the caller frees with free(); on
 * failure it is left NULL and (if err != NULL, capacity err_cap) a diagnostic is
 * written. Internally owns and frees its IRBuilder. */
ckc_status_t ckc_wmma_gemm_iu8_dequant_lower_to_llvm(const ckc_wmma_gemm_iu8_dequant_spec_t* spec,
                                                     const char* arch,
                                                     ckc_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_GFX1151_WMMA_GEMM_IU8_DEQUANT_H */
