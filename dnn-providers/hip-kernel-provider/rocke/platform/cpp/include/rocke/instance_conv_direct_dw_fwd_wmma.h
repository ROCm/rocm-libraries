/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_dw_fwd_wmma.h -- C++ engine port of the Toeplitz
 * WMMA depthwise forward in library/kernels/common/conv_direct_dw_fwd_wmma.py.
 *
 * Stride-1 depthwise forward (cpg = kpg = 1) on the gfx1250 wave32
 * 16x16x32 WMMA: one-hot weights (A) times 4 window columns x 8 channels of an
 * input row (B) into 32 output columns x 8 channels (D). Input rows stream
 * through LDS (double buffered) into KH circular accumulator tiles; finished
 * output rows drain through LDS as 16-byte stores.
 *
 *   Python (conv_direct_dw_fwd_wmma.py)      C (this header)
 *   ------------------------------------     ---------------------------------------
 *   @dataclass(frozen) DwFwdWmmaSpec         rocke_direct_dw_fwd_wmma_spec_t
 *     derived @property / methods            rocke_direct_dw_fwd_wmma_{threads_per_block,
 *                                              block_ch,tile_w,grid,in_passes,
 *                                              frag_vgprs,lds_bytes,unrolled_wmmas}
 *     .kernel_name()                         rocke_direct_dw_fwd_wmma_kernel_name
 *   dw_fwd_wmma_reason / is_valid_dw_fwd_wmma_spec
 *                                            rocke_direct_dw_fwd_wmma_is_valid_spec
 *   build_dw_fwd_wmma(spec, arch)            rocke_build_direct_dw_fwd_wmma(_new)
 *   (+ convenience: build -> lower .ll)      rocke_direct_dw_fwd_wmma_lower_to_llvm
 *
 * AOT: the kernel declares the direct-conv kernarg block of
 * rocke_conv_direct_arg_names("fwd"). N, H, W, Ho, Wo, groups and the strides
 * are runtime; KH, KW and PAD are baked capabilities, the tile is the knobs.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_DW_FWD_WMMA_H
#define ROCKE_INSTANCE_CONV_DIRECT_DW_FWD_WMMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* DW_FWD_WMMA_* (conv_direct_dw_fwd_wmma.py). */
#define ROCKE_DW_FWD_WMMA_WAVE 32
#define ROCKE_DW_FWD_WMMA_MAX_WAVES 16
#define ROCKE_DW_FWD_WMMA_MAX_FRAG_VGPRS 224
#define ROCKE_DW_FWD_WMMA_WIDE_WAVES 8
#define ROCKE_DW_FWD_WMMA_WIDE_FRAG_VGPRS 112
#define ROCKE_DW_FWD_WMMA_MAX_UNROLL 1024

/* ===================================================================== *
 *  DwFwdWmmaSpec
 *
 *  @dataclass(frozen=True)
 *  class DwFwdWmmaSpec:
 *      problem: DirectConvProblem
 *      name: str = "rocke_direct_dw_fwd_wmma"
 *      block_waves: int = 4
 *      block_h: int = 8
 *      w_fold: int = 1
 *      prefetch_rows: int = 2
 * ===================================================================== */
typedef struct rocke_direct_dw_fwd_wmma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "rocke_direct_dw_fwd_wmma" */
    int block_waves; /* default 4 */
    int block_h; /* default 8 */
    int w_fold; /* default 1 */
    int prefetch_rows; /* default 2 */
} rocke_direct_dw_fwd_wmma_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_direct_dw_fwd_wmma_spec_t rocke_direct_dw_fwd_wmma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_direct_dw_fwd_wmma_threads_per_block(const rocke_direct_dw_fwd_wmma_spec_t* spec);
int rocke_direct_dw_fwd_wmma_block_ch(const rocke_direct_dw_fwd_wmma_spec_t* spec);
int rocke_direct_dw_fwd_wmma_tile_w(const rocke_direct_dw_fwd_wmma_spec_t* spec);
int rocke_direct_dw_fwd_wmma_in_passes(const rocke_direct_dw_fwd_wmma_spec_t* spec);
int rocke_direct_dw_fwd_wmma_frag_vgprs(const rocke_direct_dw_fwd_wmma_spec_t* spec);
long rocke_direct_dw_fwd_wmma_lds_bytes(const rocke_direct_dw_fwd_wmma_spec_t* spec);
int rocke_direct_dw_fwd_wmma_unrolled_wmmas(const rocke_direct_dw_fwd_wmma_spec_t* spec);
/* spec.grid() -> (ceil(Wo / tile_w), ceil(groups / block_ch),
 * ceil(N / w_fold) * ceil(Ho / block_h)). */
void rocke_direct_dw_fwd_wmma_grid(const rocke_direct_dw_fwd_wmma_spec_t* spec, int grid[3]);

/* kernel_name(): kernel_name_join(name, r{KH}s{KW}p{PAD}, wv{}, bh{}, f{}, pf{}, [bf16]). */
rocke_status_t rocke_direct_dw_fwd_wmma_kernel_name(const rocke_direct_dw_fwd_wmma_spec_t* spec,
                                                    char* out,
                                                    size_t out_cap);

/* is_valid_dw_fwd_wmma_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx1250". */
bool rocke_direct_dw_fwd_wmma_is_valid_spec(const rocke_direct_dw_fwd_wmma_spec_t* spec,
                                            const char* arch,
                                            char* reason,
                                            size_t reason_cap);

/* build_dw_fwd_wmma(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_direct_dw_fwd_wmma(rocke_ir_builder_t* b,
                                                   const rocke_direct_dw_fwd_wmma_spec_t* spec,
                                                   const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_direct_dw_fwd_wmma_new(rocke_ir_builder_t* b,
                                                       const rocke_direct_dw_fwd_wmma_spec_t* spec,
                                                       const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_direct_dw_fwd_wmma_lower_to_llvm(const rocke_direct_dw_fwd_wmma_spec_t* spec,
                                                      const char* arch,
                                                      rocke_llvm_flavor_t flavor,
                                                      char** out_ll,
                                                      char* err,
                                                      size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_DW_FWD_WMMA_H */
