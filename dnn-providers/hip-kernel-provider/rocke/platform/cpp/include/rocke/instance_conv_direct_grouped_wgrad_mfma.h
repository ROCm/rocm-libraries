/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_grouped_wgrad_mfma.h -- C++ engine port of the
 * grouped wgrad for 4 and 8 channels per group on the batched 4x4x4 MFMA in
 * library/kernels/common/conv_direct_grouped_wgrad_mfma.py.
 *
 * A wave runs 16 groups of 4 channels as the 16 batches of mfma_f32_4x4x4:
 * per input row, tap (r, s) and 4-column group, A = X (lane = channel, four
 * columns shifted by s) and B = dY (the row of tap r, from a register ring),
 * both read straight from the NHWC rows staged in LDS by ds_read_tr16_b64. With
 * 8 channels per group a second dY read with the group's two halves swapped
 * gives the off-diagonal 4x4 blocks. The waves reduce through LDS and the block
 * stores one partial slot ws[part][tap][o] (tap = (r * KW + s) * cpg + c),
 * folded into dW by the DwWgradFoldSpec kernel
 * (rocke/instance_conv_direct_dw_wgrad.h) with taps = KH * KW * cpg and
 * channels = K.
 *
 *   Python (conv_direct_grouped_wgrad_mfma.py)       C (this header)
 *   ---------------------------------------------   ---------------------------
 *   @dataclass(frozen) GroupedWgradMfmaSpec          rocke_grouped_wgrad_mfma_spec_t
 *     host-side geometry                             rocke_grouped_wgrad_mfma_{
 *                                                      threads,lds_bytes,
 *                                                      n_partitions,fold_taps,
 *                                                      workspace_bytes,
 *                                                      launch_geometry}
 *     .kernel_name()                                 rocke_grouped_wgrad_mfma_kernel_name
 *   is_valid_grouped_wgrad_mfma_spec(spec, arch)     rocke_grouped_wgrad_mfma_is_valid_spec
 *   build_grouped_wgrad_mfma(spec, arch)             rocke_build_grouped_wgrad_mfma(_new)
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_GROUPED_WGRAD_MFMA_H
#define ROCKE_INSTANCE_CONV_DIRECT_GROUPED_WGRAD_MFMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  GroupedWgradMfmaSpec
 *
 *  @dataclass(frozen=True)
 *  class GroupedWgradMfmaSpec:
 *      problem: DirectConvProblem          # cpg = kpg in (4, 8), stride 1
 *      name: str = "grouped_wgrad_mfma"
 *      waves_q: int = 4
 *      wave_q4: int = 1
 *      rows_per_chunk: int = 56
 *      prefetch: int = 2
 *      target_blocks: int = 1024
 * ===================================================================== */
typedef struct rocke_grouped_wgrad_mfma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "grouped_wgrad_mfma" */
    int waves_q; /* default 4 */
    int wave_q4; /* default 1 */
    int rows_per_chunk; /* default 56 */
    int prefetch; /* default 2 */
    int target_blocks; /* default 1024 */
} rocke_grouped_wgrad_mfma_spec_t;

rocke_grouped_wgrad_mfma_spec_t rocke_grouped_wgrad_mfma_spec_default(void);
int rocke_grouped_wgrad_mfma_threads(const rocke_grouped_wgrad_mfma_spec_t* spec);
long rocke_grouped_wgrad_mfma_lds_bytes(const rocke_grouped_wgrad_mfma_spec_t* spec);
int rocke_grouped_wgrad_mfma_n_partitions(const rocke_grouped_wgrad_mfma_spec_t* spec);
/* taps() = KH * KW * cpg; fold_channels() = K. */
int rocke_grouped_wgrad_mfma_fold_taps(const rocke_grouped_wgrad_mfma_spec_t* spec);
long rocke_grouped_wgrad_mfma_workspace_bytes(const rocke_grouped_wgrad_mfma_spec_t* spec);
void rocke_grouped_wgrad_mfma_launch_geometry(const rocke_grouped_wgrad_mfma_spec_t* spec,
                                              int grid[3],
                                              int block[3]);
/* kernel_name_join(name, f{KH}x{KW}p{PAD}s{stride}c{cpg}, wq{n}, q4{n}, rpc{n},
 * pf{n}, tb{n}, flags [bf16]). */
rocke_status_t rocke_grouped_wgrad_mfma_kernel_name(const rocke_grouped_wgrad_mfma_spec_t* spec,
                                                    char* out,
                                                    size_t out_cap);
/* is_valid_grouped_wgrad_mfma_spec(spec, arch) -> (ok, reason). arch NULL => "gfx950". */
bool rocke_grouped_wgrad_mfma_is_valid_spec(const rocke_grouped_wgrad_mfma_spec_t* spec,
                                            const char* arch,
                                            char* reason,
                                            size_t reason_cap);
rocke_kernel_def_t* rocke_build_grouped_wgrad_mfma(rocke_ir_builder_t* b,
                                                   const rocke_grouped_wgrad_mfma_spec_t* spec,
                                                   const char* arch);
rocke_kernel_def_t* rocke_build_grouped_wgrad_mfma_new(rocke_ir_builder_t* b,
                                                       const rocke_grouped_wgrad_mfma_spec_t* spec,
                                                       const char* arch);
rocke_status_t rocke_grouped_wgrad_mfma_lower_to_llvm(const rocke_grouped_wgrad_mfma_spec_t* spec,
                                                      const char* arch,
                                                      rocke_llvm_flavor_t flavor,
                                                      char** out_ll,
                                                      char* err,
                                                      size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_GROUPED_WGRAD_MFMA_H */
