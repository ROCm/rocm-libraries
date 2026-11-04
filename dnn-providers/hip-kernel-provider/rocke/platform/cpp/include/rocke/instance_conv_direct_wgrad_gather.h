/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_wgrad_gather.h -- C++ engine port of the dense
 * small-C direct wgrad in library/kernels/common/conv_direct_wgrad_gather.py.
 *
 * One GEMM per filter row r: M = k, N = (s, c) (KW * C columns, padded to 16),
 * reduced over the output columns of a 32-wide strip by mfma_f32_16x16x32. The
 * block streams output rows, DMAs S X rows and one dY row per step into LDS
 * rings, transpose-reads dY, and gathers X (or transpose-reads it when
 * S * C keeps every lane 8-byte aligned). It stores one partial slot per block,
 * ws[part][(r, s, c)][k], folded into dW by the DwWgradFoldSpec kernel
 * (rocke/instance_conv_direct_dw_wgrad.h) with taps = KH * KW * C and
 * channels = K.
 *
 *   Python (conv_direct_wgrad_gather.py)       C (this header)
 *   ------------------------------------   -----------------------------------
 *   @dataclass(frozen) WgradGatherSpec      rocke_direct_wgrad_gather_spec_t
 *     host-side geometry                    rocke_direct_wgrad_gather_{threads,
 *                                             lds_bytes,n_partitions,fold_taps,
 *                                             workspace_bytes,launch_geometry}
 *     .kernel_name()                        rocke_direct_wgrad_gather_kernel_name
 *   is_valid_wgrad_gather_spec(spec, arch)  rocke_direct_wgrad_gather_is_valid_spec
 *   build_wgrad_gather(spec, arch)          rocke_build_direct_wgrad_gather(_new)
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_WGRAD_GATHER_H
#define ROCKE_INSTANCE_CONV_DIRECT_WGRAD_GATHER_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  WgradGatherSpec
 *
 *  @dataclass(frozen=True)
 *  class WgradGatherSpec:
 *      problem: DirectConvProblem          # groups must be 1
 *      name: str = "direct_wgrad_gather"
 *      wave_k16: int = 2
 *      waves_k: int = 1
 *      prefetch: int = 2
 *      rows_per_seg: int = 16
 *      target_blocks: int = 1024
 * ===================================================================== */
typedef struct rocke_direct_wgrad_gather_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "direct_wgrad_gather" */
    int wave_k16; /* default 2 */
    int waves_k; /* default 1 */
    int prefetch; /* default 2 */
    int rows_per_seg; /* default 16 */
    int target_blocks; /* default 1024 */
} rocke_direct_wgrad_gather_spec_t;

rocke_direct_wgrad_gather_spec_t rocke_direct_wgrad_gather_spec_default(void);
int rocke_direct_wgrad_gather_threads(const rocke_direct_wgrad_gather_spec_t* spec);
long rocke_direct_wgrad_gather_lds_bytes(const rocke_direct_wgrad_gather_spec_t* spec);
int rocke_direct_wgrad_gather_n_partitions(const rocke_direct_wgrad_gather_spec_t* spec);
/* fold_taps() = KH * KW * C; fold_channels() = K. */
int rocke_direct_wgrad_gather_fold_taps(const rocke_direct_wgrad_gather_spec_t* spec);
long rocke_direct_wgrad_gather_workspace_bytes(const rocke_direct_wgrad_gather_spec_t* spec);
void rocke_direct_wgrad_gather_launch_geometry(const rocke_direct_wgrad_gather_spec_t* spec,
                                               int grid[3],
                                               int block[3]);
/* kernel_name_join(name, f{KH}x{KW}p{PAD}s{stride}c{cpg}, wk{wk16}x{waves_k},
 * pf{n}, rps{n}, tb{n}, flags [bf16]). */
rocke_status_t rocke_direct_wgrad_gather_kernel_name(const rocke_direct_wgrad_gather_spec_t* spec,
                                                     char* out,
                                                     size_t out_cap);
/* is_valid_wgrad_gather_spec(spec, arch) -> (ok, reason). arch NULL => "gfx950". */
bool rocke_direct_wgrad_gather_is_valid_spec(const rocke_direct_wgrad_gather_spec_t* spec,
                                             const char* arch,
                                             char* reason,
                                             size_t reason_cap);
rocke_kernel_def_t* rocke_build_direct_wgrad_gather(rocke_ir_builder_t* b,
                                                    const rocke_direct_wgrad_gather_spec_t* spec,
                                                    const char* arch);
rocke_kernel_def_t* rocke_build_direct_wgrad_gather_new(
    rocke_ir_builder_t* b, const rocke_direct_wgrad_gather_spec_t* spec, const char* arch);
rocke_status_t rocke_direct_wgrad_gather_lower_to_llvm(const rocke_direct_wgrad_gather_spec_t* spec,
                                                       const char* arch,
                                                       rocke_llvm_flavor_t flavor,
                                                       char** out_ll,
                                                       char* err,
                                                       size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_WGRAD_GATHER_H */
