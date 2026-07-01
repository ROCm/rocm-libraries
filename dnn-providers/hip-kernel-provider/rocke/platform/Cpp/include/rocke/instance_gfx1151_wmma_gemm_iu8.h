/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1151_wmma_gemm_iu8.h -- C99 port of the gfx1151 (RDNA3.5)
 * native-integer WMMA GEMM kernel instance builder
 * rocke/instances/gfx1151/wmma_gemm_iu8.py.
 *
 *   Python (gfx1151/wmma_gemm_iu8.py)        C99 (this header)
 *   --------------------------------------   ------------------------------------
 *   class WmmaGemmIu8Spec                    rocke_wmma_gemm_iu8_spec_t
 *   WmmaGemmIu8Spec.block_size               rocke_wmma_gemm_iu8_block_size(spec)
 *   WmmaGemmIu8Spec.kernel_name()            rocke_wmma_gemm_iu8_kernel_name(...)
 *   is_valid_spec(spec, arch)                rocke_wmma_gemm_iu8_is_valid_spec(...)
 *   build_wmma_gemm_iu8(spec, arch)          rocke_build_wmma_gemm_iu8(...)
 *   wmma_gemm_iu8_grid(M, N)                 rocke_wmma_gemm_iu8_grid(...)
 *   (+ convenience: build -> lower .ll)      rocke_wmma_gemm_iu8_lower_to_llvm(...)
 *
 * Integer WMMA fragment ABI (iu8, RDNA3.5):
 *   * A/B are int8 logically but passed packed as i32 (4 int8 / i32). C is i32.
 *   * One wave32 computes one 16x16 output tile; grid is
 *     ((M+15)//16, (N+15)//16, 1), threads_per_block = 32.
 *   * Each lane's operand fragment is <4 x i32> = 16 int8 K-values packed
 *     4-per-i32 (slot j holds K=[4j..4j+3]).
 *   * Accumulator is a loop-carried <8 x i32>; slot i of lane l maps to output
 *     (row = m0 + 2*i + l/16, col = n0 + l%16). Output is stored as i32 (no
 *     rounding, no f32->f16 truncation).
 *   * MMA is the op_id "wmma_i32_16x16x16_iu8".
 *
 * Layout is RCR (C = A @ B.T, A row-major M*K int8, B row-major N*K int8), no
 * LDS. The build op order tracks build_wmma_gemm_iu8() top-to-bottom.
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python frozen dataclass has a single `name`
 * field with a default; in C the caller fills a rocke_wmma_gemm_iu8_spec_t.
 * rocke_wmma_gemm_iu8_spec_default() returns a struct with the field set to the
 * Python dataclass default.
 *
 * Error model mirrors the rest of the C port: build routes errors through the
 * sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + reason
 * string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_GFX1151_WMMA_GEMM_IU8_H
#define ROCKE_INSTANCE_GFX1151_WMMA_GEMM_IU8_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------- WmmaGemmIu8Spec *
 *
 * Mirror of Python WmmaGemmIu8Spec (frozen dataclass):
 *
 *     name: str = "rocke_wmma_gemm_iu8"
 *
 * Unlike WmmaGemmSpec there is NO dtype field and NO __post_init__ check. */
typedef struct rocke_wmma_gemm_iu8_spec
{
    const char* name; /* default "rocke_wmma_gemm_iu8" */
} rocke_wmma_gemm_iu8_spec_t;

/* Default-constructed spec (field == Python dataclass default). */
rocke_wmma_gemm_iu8_spec_t rocke_wmma_gemm_iu8_spec_default(void);

/* WmmaGemmIu8Spec.block_size @property: one wave32 == 32. */
int rocke_wmma_gemm_iu8_block_size(const rocke_wmma_gemm_iu8_spec_t* spec);

/* WmmaGemmIu8Spec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 *
 *     kernel_name_join(self.name, "wmma16x16x16", "iu8", "rcr")
 *
 * Returns ROCKE_OK, or ROCKE_ERR_VALUE (buffer too small / null args). */
rocke_status_t rocke_wmma_gemm_iu8_kernel_name(const rocke_wmma_gemm_iu8_spec_t* spec,
                                               char* out,
                                               size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx1151".
 *
 * Gate (mirrors gfx1151/wmma_gemm_iu8.is_valid_spec):
 *   - ArchTarget.from_gfx(arch) must resolve (else the KeyError string).
 *   - the iu8 WMMA atom (op_id "wmma_i32_16x16x16_iu8") must exist in the
 *     target catalog (target.mma.by_op_id(_OP_ID) is not None).
 *   - target.wave_size must be 32 (wave32 kernel).
 *
 * On a reject, `reason` (if non-NULL, capacity reason_cap) receives the
 * structured message and false is returned. On accept returns true and writes
 * "ok". */
bool rocke_wmma_gemm_iu8_is_valid_spec(const rocke_wmma_gemm_iu8_spec_t* spec,
                                       const char* arch,
                                       char* reason,
                                       size_t reason_cap);

/* build_wmma_gemm_iu8(spec, arch). Builds the IR into the supplied (already
 * rocke_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set. `arch` NULL => "gfx1151".
 *
 * Kernel signature: (A: ptr<i32>, B: ptr<i32>, C: ptr<i32>,
 *                    M: i32, N: i32, K: i32).
 * Grid: ((M+15)//16, (N+15)//16, 1). Block: 32 threads (one wave32). */
rocke_kernel_def_t* rocke_build_wmma_gemm_iu8(rocke_ir_builder_t* b,
                                              const rocke_wmma_gemm_iu8_spec_t* spec,
                                              const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_wmma_gemm_iu8_new(rocke_ir_builder_t* b,
                                                  const rocke_wmma_gemm_iu8_spec_t* spec,
                                                  const char* arch);

/* wmma_gemm_iu8_grid(M, N) -> ((M+15)//16, (N+15)//16, 1). Returns ROCKE_OK and
 * writes out[0..2]; ROCKE_ERR_VALUE on null out. */
rocke_status_t rocke_wmma_gemm_iu8_grid(int M, int N, int out[3]);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text
 * at arch=gfx1151 (or the supplied arch; NULL => "gfx1151"). On ROCKE_OK *out_ll
 * receives a malloc'd NUL-terminated string the caller frees with free(); on
 * failure it is left NULL and (if err != NULL, capacity err_cap) a diagnostic is
 * written. Internally owns and frees its IRBuilder. */
rocke_status_t rocke_wmma_gemm_iu8_lower_to_llvm(const rocke_wmma_gemm_iu8_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1151_WMMA_GEMM_IU8_H */
