/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_dw_wgrad_mfma.h -- C++ engine port of the
 * depthwise wgrad on the batched 4x4x4 MFMA in
 * library/kernels/common/conv_direct_dw_wgrad_mfma.py.
 *
 * mfma_f32_4x4x4 runs 16 independent 4x4x4 products per wave, one channel
 * each: per input row and 4-column group, A[s][k] = X[y][q0 + k + s - PAD]
 * (a Hankel window built from an aligned LDS window with v_perm_b32) times
 * B[k][r] = dY[y + PAD - r][q0 + k] (a register ring of dY rows rotated by
 * quad_perm). Rows reach LDS by DMA in NHWC order and are transposed
 * channel-major with ds_read_tr16_b64. The block stores one partial slot
 * ws[part][tap][c], folded into dW by the DwWgradFoldSpec kernel
 * (rocke/instance_conv_direct_dw_wgrad.h) with taps = KH * KW, channels = C.
 *
 *   Python (conv_direct_dw_wgrad_mfma.py)      C (this header)
 *   -------------------------------------   -----------------------------------
 *   @dataclass(frozen) DwWgradMfmaSpec       rocke_dw_wgrad_mfma_spec_t
 *     host-side geometry                     rocke_dw_wgrad_mfma_{lds_bytes,
 *                                              image_pitch,n_partitions,
 *                                              workspace_bytes,launch_geometry}
 *     .kernel_name()                         rocke_dw_wgrad_mfma_kernel_name
 *   is_valid_dw_wgrad_mfma_spec(spec, arch)  rocke_dw_wgrad_mfma_is_valid_spec
 *   build_dw_wgrad_mfma(spec, arch)          rocke_build_dw_wgrad_mfma(_new)
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_DW_WGRAD_MFMA_H
#define ROCKE_INSTANCE_CONV_DIRECT_DW_WGRAD_MFMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  DwWgradMfmaSpec
 *
 *  @dataclass(frozen=True)
 *  class DwWgradMfmaSpec:
 *      problem: DirectConvProblem          # depthwise, stride 1
 *      name: str = "dw_wgrad_mfma"
 *      q_tiles: int = 8
 *      rows_per_chunk: int = 16
 *      prefetch: int = 2
 *      target_blocks: int = 2048
 *      unfold_n: int = 1
 * ===================================================================== */
typedef struct rocke_dw_wgrad_mfma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "dw_wgrad_mfma" */
    int q_tiles; /* default 8 */
    int rows_per_chunk; /* default 16 */
    int prefetch; /* default 2 */
    int target_blocks; /* default 2048 */
    int unfold_n; /* default 1 */
} rocke_dw_wgrad_mfma_spec_t;

rocke_dw_wgrad_mfma_spec_t rocke_dw_wgrad_mfma_spec_default(void);
long rocke_dw_wgrad_mfma_lds_bytes(const rocke_dw_wgrad_mfma_spec_t* spec);
/* image_pitch() = max(Wo, Wo + KW - 1 - PAD, PAD + W). */
int rocke_dw_wgrad_mfma_image_pitch(const rocke_dw_wgrad_mfma_spec_t* spec);
int rocke_dw_wgrad_mfma_n_partitions(const rocke_dw_wgrad_mfma_spec_t* spec);
long rocke_dw_wgrad_mfma_workspace_bytes(const rocke_dw_wgrad_mfma_spec_t* spec);
void rocke_dw_wgrad_mfma_launch_geometry(const rocke_dw_wgrad_mfma_spec_t* spec,
                                         int grid[3],
                                         int block[3]);
/* kernel_name_join(name, f{KH}x{KW}p{PAD}s{stride}, qt{n}, rpc{n}, pf{n},
 * un{n}, tb{n}, flags [bf16]). */
rocke_status_t rocke_dw_wgrad_mfma_kernel_name(const rocke_dw_wgrad_mfma_spec_t* spec,
                                               char* out,
                                               size_t out_cap);
/* is_valid_dw_wgrad_mfma_spec(spec, arch) -> (ok, reason). arch NULL => "gfx950". */
bool rocke_dw_wgrad_mfma_is_valid_spec(const rocke_dw_wgrad_mfma_spec_t* spec,
                                       const char* arch,
                                       char* reason,
                                       size_t reason_cap);
rocke_kernel_def_t* rocke_build_dw_wgrad_mfma(rocke_ir_builder_t* b,
                                              const rocke_dw_wgrad_mfma_spec_t* spec,
                                              const char* arch);
rocke_kernel_def_t* rocke_build_dw_wgrad_mfma_new(rocke_ir_builder_t* b,
                                                  const rocke_dw_wgrad_mfma_spec_t* spec,
                                                  const char* arch);
rocke_status_t rocke_dw_wgrad_mfma_lower_to_llvm(const rocke_dw_wgrad_mfma_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_DW_WGRAD_MFMA_H */
