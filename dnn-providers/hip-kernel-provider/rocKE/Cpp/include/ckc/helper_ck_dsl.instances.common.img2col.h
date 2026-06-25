/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_ck_dsl.instances.common.img2col.h -- C99 port of ONE symbol from
 * ck_dsl/instances/common/img2col.py:
 *
 *   Python                              C99 (this header)
 *   ---------------------------------   ---------------------------------------
 *   Img2ColSpec (frozen dataclass)     ckc_img2col_spec_t
 *     .block_size  (@property)          ckc_img2col_block_size()
 *     .can_vector_load  (@property)     ckc_img2col_can_vector_load()
 *     .kernel_name()                    ckc_img2col_kernel_name()
 *
 * Only Img2ColSpec is in scope here. is_valid_spec / build_img2col /
 * img2col_grid / img2col_block_tile_m_for_M / img2col_signature are NOT ported
 * by this module.
 *
 * Img2ColSpec.problem is a conv_implicit_gemm.ConvProblem. That type is not yet
 * ported to C, so this header defines the minimal subset of ConvProblem that
 * Img2ColSpec actually reads -- the seven layout dims (N, Hi, Wi, C, K, Y, X),
 * the stride/pad/dilation fields needed by short()'s siblings, and the derived
 * accessors M / K_gemm / short() -- as ckc_conv_problem_t. When the full
 * ConvProblem port lands it should subsume / replace this peer struct.
 *
 * Img2ColSpec is a pure value type: none of these entry points touch the IR
 * builder (ckc_b_*). block_size / can_vector_load are arithmetic on the fields;
 * kernel_name() reproduces the exact kernel_name_join(...) call sequence so the
 * baked-in kernel name is byte-identical to the Python.
 */
#ifndef CKC_HELPER_CK_DSL_INSTANCES_COMMON_IMG2COL_H
#define CKC_HELPER_CK_DSL_INSTANCES_COMMON_IMG2COL_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h" /* ckc_status_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * ConvProblem (minimal subset read by Img2ColSpec)
 * ------------------------------------------------------------------ *
 *
 * Python (ck_dsl/instances/common/conv_implicit_gemm.py):
 *
 *   @dataclass(frozen=True)
 *   class ConvProblem:
 *       N: int; Hi: int; Wi: int; C: int; K: int; R: int; S: int
 *       sH: int = 1; sW: int = 1
 *       pH: int = 0; pW: int = 0
 *       dH: int = 1; dW: int = 1
 *
 *       @property
 *       def Ho(self): return (Hi + 2*pH - dH*(R-1) - 1)//sH + 1
 *       @property
 *       def Wo(self): return (Wi + 2*pW - dW*(S-1) - 1)//sW + 1
 *       @property
 *       def M(self):  return N * Ho * Wo
 *       @property
 *       def K_gemm(self): return Y * X * C
 *       def short(self):
 *           return f"N{N}H{Hi}W{Wi}C{C}_K{K}Y{Y}X{X}"
 *
 * The frozen dataclass becomes a plain struct; the defaults are applied by
 * ckc_conv_problem_default(). Only the accessors Img2ColSpec needs (C field +
 * M / K_gemm / short()) are exposed here. */
typedef struct ckc_conv_problem
{
    int N;
    int Hi;
    int Wi;
    int C;
    int K;
    int Y; /* #8355: was R */
    int X; /* #8355: was S */
    int sH; /* default 1 */
    int sW; /* default 1 */
    int pH; /* default 0 */
    int pW; /* default 0 */
    int dH; /* default 1 */
    int dW; /* default 1 */
    /* 3-D-only fields (img2col is 2-D-only; kept for layout parity with the
     * conv helper header's ckc_conv_problem_t). 0/false => 2-D. */
    bool is_3d;
    int Di;
    int Z;
    int sD;
    int pD;
    int dD;
} ckc_conv_problem_t;

/* A ConvProblem with the dataclass defaults installed (sH=sW=dH=dW=1,
 * pH=pW=0) and the seven required dims zeroed. The caller fills N, Hi, Wi, C,
 * K, Y, X (and overrides stride/pad/dilation as needed). */
ckc_conv_problem_t ckc_img2col_conv_problem_default(void);

/* ConvProblem.Ho property: (Hi + 2*pH - dH*(R-1) - 1)//sH + 1.
 * Floor division matches Python // on the non-negative numerator the conv
 * shapes produce. */
int ckc_img2col_conv_problem_ho(const ckc_conv_problem_t* p);

/* ConvProblem.Wo property: (Wi + 2*pW - dW*(S-1) - 1)//sW + 1. */
int ckc_img2col_conv_problem_wo(const ckc_conv_problem_t* p);

/* ConvProblem.M property: N * Ho * Wo. */
int ckc_img2col_conv_problem_m(const ckc_conv_problem_t* p);

/* ConvProblem.K_gemm property: Y * X * C. */
int ckc_img2col_conv_problem_k_gemm(const ckc_conv_problem_t* p);

/* ConvProblem.short(): "N{N}H{Hi}W{Wi}C{C}_K{K}Y{Y}X{X}" written NUL-terminated
 * into out (capacity out_cap). Returns CKC_OK, or CKC_ERR_VALUE on NULL args /
 * a buffer too small to hold the whole string. */
ckc_status_t ckc_img2col_conv_problem_short(const ckc_conv_problem_t* p, char* out, size_t out_cap);

/* ------------------------------------------------------------------ *
 * Img2ColSpec
 * ------------------------------------------------------------------ *
 *
 * Python:
 *
 *   DType = Literal["f16"]
 *
 *   @dataclass(frozen=True)
 *   class Img2ColSpec:
 *       problem: ConvProblem
 *       dtype: DType = "f16"
 *       block_tile_m: int = 8
 *       block_tile_k: int = 128
 *       vec_k: int = 8
 *       name: str = "ck_dsl_img2col"
 *
 * The frozen dataclass becomes a value struct. `dtype` is the Literal["f16"]
 * string (default "f16"); `name` defaults to "ck_dsl_img2col". Use
 * ckc_img2col_spec_default() to get the defaulted record, then fill `problem`
 * and override any field. */
typedef struct ckc_img2col_spec
{
    ckc_conv_problem_t problem; /* convolution shape                      */
    const char* dtype; /* Literal["f16"]; default "f16"          */
    int block_tile_m; /* default 8                              */
    int block_tile_k; /* default 128                            */
    int vec_k; /* default 8                              */
    const char* name; /* default "ck_dsl_img2col"               */
} ckc_img2col_spec_t;

/* Default-constructed Img2ColSpec: dtype "f16", block_tile_m 8,
 * block_tile_k 128, vec_k 8, name "ck_dsl_img2col", problem ==
 * ckc_conv_problem_default(). The caller fills problem's seven required dims. */
ckc_img2col_spec_t ckc_img2col_spec_default(void);

/* Img2ColSpec.block_size property:
 *   (block_tile_m * block_tile_k) // vec_k
 * Integer (floor) division, matching Python //. Result is signed 32-bit. */
int ckc_img2col_block_size(const ckc_img2col_spec_t* spec);

/* Img2ColSpec.can_vector_load property:
 *   vec_k > 1 and (problem.C % vec_k) == 0
 * True iff a single buffer_load_vN_f16 covers each chunk. */
bool ckc_img2col_can_vector_load(const ckc_img2col_spec_t* spec);

/* Img2ColSpec.kernel_name():
 *   return kernel_name_join(
 *       self.name,
 *       self.problem.short(),
 *       self.dtype,
 *       f"t{self.block_tile_m}x{self.block_tile_k}",
 *       f"v{self.vec_k}",
 *   )
 *
 * Reproduces the exact kernel_name_join(prefix, *parts) call (no flags). The
 * result is written NUL-terminated into out (capacity out_cap). Returns CKC_OK,
 * or CKC_ERR_VALUE on NULL args / buffer too small. */
ckc_status_t ckc_img2col_kernel_name(const ckc_img2col_spec_t* spec, char* out, size_t out_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_CK_DSL_INSTANCES_COMMON_IMG2COL_H */
