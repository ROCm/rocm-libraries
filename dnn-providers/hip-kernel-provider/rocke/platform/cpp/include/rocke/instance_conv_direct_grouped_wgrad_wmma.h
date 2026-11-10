/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_grouped_wgrad_wmma.h -- C++ engine port of the
 * gfx1250 grouped wgrad for 4 and 8 channels per group on the wave32 WMMA
 * 16x16x32 in library/kernels/common/conv_direct_grouped_wgrad_wmma.py.
 *
 * A wave owns 16 channels (four groups of 4 or two of 8) packed
 * block-diagonally into one 16x16x32 WMMA: per tap (r, s) and 32 output
 * columns, A = dY (M = 16 dY channels) and B = X (N = 16 X channels), both
 * read from [column][channel] rows staged in LDS (global_load_async_to_lds)
 * by ds_load_tr16_b128. The dY fragments of the KH - 1 previous rows stay in
 * a register ring. Each wave stores the diagonal blocks of its accumulators
 * into one partial slot ws[part][tap][o] (tap = (r * KW + s) * cpg + c),
 * folded into dW by the DwWgradFoldSpec kernel
 * (rocke/instance_conv_direct_dw_wgrad.h) with taps = KH * KW * cpg and
 * channels = K.
 *
 *   Python (conv_direct_grouped_wgrad_wmma.py)       C (this header)
 *   ---------------------------------------------   ---------------------------
 *   @dataclass(frozen) GroupedWgradWmmaSpec          rocke_grouped_wgrad_wmma_spec_t
 *     host-side geometry                             rocke_grouped_wgrad_wmma_{
 *                                                      threads,lds_bytes,
 *                                                      n_partitions,fold_taps,
 *                                                      fold_channels,
 *                                                      workspace_bytes,
 *                                                      launch_geometry}
 *     .kernel_name()                                 rocke_grouped_wgrad_wmma_kernel_name
 *   is_valid_grouped_wgrad_wmma_spec(spec, arch)     rocke_grouped_wgrad_wmma_is_valid_spec
 *   build_grouped_wgrad_wmma(spec, arch)             rocke_build_grouped_wgrad_wmma(_new)
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_GROUPED_WGRAD_WMMA_H
#define ROCKE_INSTANCE_CONV_DIRECT_GROUPED_WGRAD_WMMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  GroupedWgradWmmaSpec
 *
 *  @dataclass(frozen=True)
 *  class GroupedWgradWmmaSpec:
 *      problem: DirectConvProblem          # cpg = kpg in (4, 8), stride 1
 *      name: str = "grouped_wgrad_wmma"
 *      waves_c: int = 4
 *      q_steps: int = 2
 *      rows_per_chunk: int = 32
 *      target_blocks: int = 1024
 *      stages: int = 2
 * ===================================================================== */
typedef struct rocke_grouped_wgrad_wmma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "grouped_wgrad_wmma" */
    int waves_c; /* default 4: waves along the channels, 16 each */
    int q_steps; /* default 2: 32-column WMMA steps per block */
    int rows_per_chunk; /* default 32 */
    int target_blocks; /* default 1024 */
    int stages; /* default 2: LDS buffers (rows in flight + 1) */
} rocke_grouped_wgrad_wmma_spec_t;

rocke_grouped_wgrad_wmma_spec_t rocke_grouped_wgrad_wmma_spec_default(void);
int rocke_grouped_wgrad_wmma_threads(const rocke_grouped_wgrad_wmma_spec_t* spec);
long rocke_grouped_wgrad_wmma_lds_bytes(const rocke_grouped_wgrad_wmma_spec_t* spec);
int rocke_grouped_wgrad_wmma_n_partitions(const rocke_grouped_wgrad_wmma_spec_t* spec);
/* fold_taps() = KH * KW * cpg; fold_channels() = K. */
int rocke_grouped_wgrad_wmma_fold_taps(const rocke_grouped_wgrad_wmma_spec_t* spec);
long long rocke_grouped_wgrad_wmma_fold_channels(const rocke_grouped_wgrad_wmma_spec_t* spec);
long rocke_grouped_wgrad_wmma_workspace_bytes(const rocke_grouped_wgrad_wmma_spec_t* spec);
void rocke_grouped_wgrad_wmma_launch_geometry(const rocke_grouped_wgrad_wmma_spec_t* spec,
                                              int grid[3],
                                              int block[3]);
/* kernel_name_join(name, f{KH}x{KW}p{PAD}s{stride}c{cpg}, wc{n}, qs{n},
 * rpc{n}, tb{n}, st{n}, flags [bf16]). */
rocke_status_t rocke_grouped_wgrad_wmma_kernel_name(const rocke_grouped_wgrad_wmma_spec_t* spec,
                                                    char* out,
                                                    size_t out_cap);
/* is_valid_grouped_wgrad_wmma_spec(spec, arch) -> (ok, reason). arch NULL => "gfx1250". */
bool rocke_grouped_wgrad_wmma_is_valid_spec(const rocke_grouped_wgrad_wmma_spec_t* spec,
                                            const char* arch,
                                            char* reason,
                                            size_t reason_cap);
rocke_kernel_def_t* rocke_build_grouped_wgrad_wmma(rocke_ir_builder_t* b,
                                                   const rocke_grouped_wgrad_wmma_spec_t* spec,
                                                   const char* arch);
rocke_kernel_def_t* rocke_build_grouped_wgrad_wmma_new(rocke_ir_builder_t* b,
                                                       const rocke_grouped_wgrad_wmma_spec_t* spec,
                                                       const char* arch);
rocke_status_t rocke_grouped_wgrad_wmma_lower_to_llvm(const rocke_grouped_wgrad_wmma_spec_t* spec,
                                                      const char* arch,
                                                      rocke_llvm_flavor_t flavor,
                                                      char** out_ll,
                                                      char* err,
                                                      size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_GROUPED_WGRAD_WMMA_H */
