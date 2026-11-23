/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common.img2col.h -- C99 port of ONE symbol from
 * rocke/instances/common/img2col.py:
 *
 *   Python                              C99 (this header)
 *   ---------------------------------   ---------------------------------------
 *   Img2ColSpec (frozen dataclass)     rocke_img2col_spec_t
 *     .block_size  (@property)          rocke_img2col_block_size()
 *     .can_vector_load  (@property)     rocke_img2col_can_vector_load()
 *     .kernel_name()                    rocke_img2col_kernel_name()
 *
 * Only Img2ColSpec is in scope here. is_valid_spec / build_img2col /
 * img2col_grid / img2col_block_tile_m_for_M / img2col_signature are NOT ported
 * by this module.
 *
 * Img2ColSpec.problem is a conv_implicit_gemm.ConvProblem, so the spec holds
 * rocke_conv_problem_t, the C port of ConvProblem in
 * rocke/helper_rocke.instances.common.conv_implicit_gemm.h, and reads its
 * M / K_gemm / short() through that header's accessors.
 *
 * Img2ColSpec is a pure value type: none of these entry points touch the IR
 * builder (rocke_b_*). block_size / can_vector_load are arithmetic on the fields;
 * kernel_name() reproduces the exact kernel_name_join(...) call sequence so the
 * baked-in kernel name is byte-identical to the Python.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON_IMG2COL_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON_IMG2COL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h" /* rocke_conv_problem_t */
#include "rocke/ir.h" /* rocke_status_t */

#ifdef __cplusplus
extern "C" {
#endif

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
 *       name: str = "rocke_img2col"
 *
 * The frozen dataclass becomes a value struct. `dtype` is the Literal["f16"]
 * string (default "f16"); `name` defaults to "rocke_img2col". Use
 * rocke_img2col_spec_default() to get the defaulted record, then fill `problem`
 * and override any field. */
typedef struct rocke_img2col_spec
{
    rocke_conv_problem_t problem; /* convolution shape                      */
    const char* dtype; /* Literal["f16"]; default "f16"          */
    int block_tile_m; /* default 8                              */
    int block_tile_k; /* default 128                            */
    int vec_k; /* default 8                              */
    const char* name; /* default "rocke_img2col"               */
} rocke_img2col_spec_t;

/* Default-constructed Img2ColSpec: dtype "f16", block_tile_m 8,
 * block_tile_k 128, vec_k 8, name "rocke_img2col", problem ==
 * rocke_conv_problem_default() with its seven required dims zeroed. The caller
 * fills those dims. */
rocke_img2col_spec_t rocke_img2col_spec_default(void);

/* Img2ColSpec.block_size property:
 *   (block_tile_m * block_tile_k) // vec_k
 * Integer (floor) division, matching Python //. Result is signed 32-bit. */
int rocke_img2col_block_size(const rocke_img2col_spec_t* spec);

/* Img2ColSpec.can_vector_load property:
 *   vec_k > 1 and (problem.C % vec_k) == 0
 * True iff a single buffer_load_vN_f16 covers each chunk. */
bool rocke_img2col_can_vector_load(const rocke_img2col_spec_t* spec);

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
 * result is written NUL-terminated into out (capacity out_cap). Returns ROCKE_OK,
 * or ROCKE_ERR_VALUE on NULL args / buffer too small. */
rocke_status_t
    rocke_img2col_kernel_name(const rocke_img2col_spec_t* spec, char* out, size_t out_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON_IMG2COL_H */
