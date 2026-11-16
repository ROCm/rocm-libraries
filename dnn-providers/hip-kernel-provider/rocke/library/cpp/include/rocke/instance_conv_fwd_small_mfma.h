/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_fwd_small_mfma.h -- C++ engine port of the wave64 MFMA
 * forward convolutions of tiny channel counts in
 * library/kernels/common/conv_fwd_small_mfma.py.
 *
 * Two kernels, both ungrouped and undilated on the gfx950 MFMA 16x16x32 atom
 * with buffer-load staging:
 *
 *   fwd_smallc_mfma  few input channels (C <= 4, stems): one GEMM over the
 *                    flattened (r, s, c) reduction; each block stages its
 *                    output tile's input window into LDS with 16-byte buffer
 *                    loads and gathers the im2col fragments from LDS; the
 *                    weights stay in registers across `tiles_per_block`
 *                    tiles, whose window loads `prefetch` issues one tile
 *                    ahead.
 *   fwd_smallk_mfma  few output channels (K <= 8): Z = X * W[(r, s, k), c]
 *                    over the window, fragments straight from global memory,
 *                    then an fp32 overlap-add in LDS.
 *
 *   Python (conv_fwd_small_mfma.py)          C (this header)
 *   ---------------------------------------  ---------------------------------
 *   @dataclass(frozen) FwdSmallcMfmaSpec     rocke_fwd_smallc_mfma_spec_t
 *   @dataclass(frozen) FwdSmallkMfmaSpec     rocke_fwd_smallk_mfma_spec_t
 *     derived @property geometry             rocke_fwd_small{c,k}_mfma_{block_*,
 *                                              threads_per_block,lds_bytes,grid}
 *     .kernel_name()                         rocke_fwd_small{c,k}_mfma_kernel_name
 *   fwd_small{c,k}_mfma_reason / is_valid_*  rocke_fwd_small{c,k}_mfma_is_valid_spec
 *   build_fwd_small{c,k}_mfma(spec, arch)    rocke_build_fwd_small{c,k}_mfma(_new)
 *   (+ convenience: build -> lower .ll)      rocke_fwd_small{c,k}_mfma_lower_to_llvm
 *
 * AOT: both kernels declare the forward direct-conv kernarg block of
 * rocke_conv_direct_arg_names("fwd"). Batch, extents and the other channel
 * count are runtime; the filter, stride, padding, the small channel count, the
 * dtype and the tile geometry are baked.
 */
#ifndef ROCKE_INSTANCE_CONV_FWD_SMALL_MFMA_H
#define ROCKE_INSTANCE_CONV_FWD_SMALL_MFMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  FwdSmallcMfmaSpec
 *
 *  @dataclass(frozen=True)
 *  class FwdSmallcMfmaSpec:
 *      problem: DirectConvProblem
 *      name: str = "fwd_smallc_mfma"
 *      waves_k: int = 1          # waves along output channels
 *      waves_p: int = 4          # waves along output pixels
 *      tiles_k: int = 4          # 16-channel tiles per wave
 *      tiles_p: int = 2          # 16-pixel tiles per wave
 *      out_h: int = 4            # output rows per tile
 *      out_w: int = 32           # output columns per tile
 *      tiles_per_block: int = 1  # output tiles a block walks
 *      prefetch: bool = False    # next tile's window loads in flight
 *      wave_size: int = 64
 * ===================================================================== */
typedef struct rocke_fwd_smallc_mfma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "fwd_smallc_mfma" */
    int waves_k; /* default 1 */
    int waves_p; /* default 4 */
    int tiles_k; /* default 4 */
    int tiles_p; /* default 2 */
    int out_h; /* default 4 */
    int out_w; /* default 32 */
    int tiles_per_block; /* default 1 */
    bool prefetch; /* default false */
    int wave_size; /* default 64 */
} rocke_fwd_smallc_mfma_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_fwd_smallc_mfma_spec_t rocke_fwd_smallc_mfma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_fwd_smallc_mfma_block_k(const rocke_fwd_smallc_mfma_spec_t* spec);
int rocke_fwd_smallc_mfma_block_p(const rocke_fwd_smallc_mfma_spec_t* spec);
int rocke_fwd_smallc_mfma_threads_per_block(const rocke_fwd_smallc_mfma_spec_t* spec);
long rocke_fwd_smallc_mfma_lds_bytes(const rocke_fwd_smallc_mfma_spec_t* spec);
/* spec.grid() -> (ceil(n_tiles / tiles_per_block), ceil(total_k / block_k), 1).
 */
void rocke_fwd_smallc_mfma_grid(const rocke_fwd_smallc_mfma_spec_t* spec, int grid[3]);

/* kernel_name(): kernel_name_join(name, f{KH}x{KW}, s{stride}, p{PAD}, c{C},
 * wk{}, wp{}, tk{}, tp{}, oh{}, ow{}, tb{}, [pf], [bf16]). */
rocke_status_t rocke_fwd_smallc_mfma_kernel_name(const rocke_fwd_smallc_mfma_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap);

/* is_valid_fwd_smallc_mfma_spec(spec, arch) -> (ok, reason). `arch` NULL =>
 * "gfx950". */
bool rocke_fwd_smallc_mfma_is_valid_spec(const rocke_fwd_smallc_mfma_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap);

/* build_fwd_smallc_mfma(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_fwd_smallc_mfma(rocke_ir_builder_t* b,
                                                const rocke_fwd_smallc_mfma_spec_t* spec,
                                                const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_fwd_smallc_mfma_new(rocke_ir_builder_t* b,
                                                    const rocke_fwd_smallc_mfma_spec_t* spec,
                                                    const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_fwd_smallc_mfma_lower_to_llvm(const rocke_fwd_smallc_mfma_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap);

/* ===================================================================== *
 *  FwdSmallkMfmaSpec
 *
 *  @dataclass(frozen=True)
 *  class FwdSmallkMfmaSpec:
 *      problem: DirectConvProblem
 *      name: str = "fwd_smallk_mfma"
 *      waves: int = 4       # waves along window-pixel tiles
 *      out_h: int = 16      # output rows per block
 *      out_w: int = 16      # output columns per block
 *      wave_size: int = 64
 * ===================================================================== */
typedef struct rocke_fwd_smallk_mfma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "fwd_smallk_mfma" */
    int waves; /* default 4  */
    int out_h; /* default 16 */
    int out_w; /* default 16 */
    int wave_size; /* default 64 */
} rocke_fwd_smallk_mfma_spec_t;

rocke_fwd_smallk_mfma_spec_t rocke_fwd_smallk_mfma_spec_default(void);

int rocke_fwd_smallk_mfma_threads_per_block(const rocke_fwd_smallk_mfma_spec_t* spec);
long rocke_fwd_smallk_mfma_lds_bytes(const rocke_fwd_smallk_mfma_spec_t* spec);
/* spec.grid() -> (N * ceil(Ho / out_h) * ceil(Wo / out_w), 1, 1). */
void rocke_fwd_smallk_mfma_grid(const rocke_fwd_smallk_mfma_spec_t* spec, int grid[3]);

/* kernel_name(): kernel_name_join(name, f{KH}x{KW}, s{stride}, p{PAD},
 * [pw{PAD_W}], k{K}, w{}, oh{}, ow{}, [bf16]). */
rocke_status_t rocke_fwd_smallk_mfma_kernel_name(const rocke_fwd_smallk_mfma_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap);

/* is_valid_fwd_smallk_mfma_spec(spec, arch) -> (ok, reason). `arch` NULL =>
 * "gfx950". */
bool rocke_fwd_smallk_mfma_is_valid_spec(const rocke_fwd_smallk_mfma_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap);

rocke_kernel_def_t* rocke_build_fwd_smallk_mfma(rocke_ir_builder_t* b,
                                                const rocke_fwd_smallk_mfma_spec_t* spec,
                                                const char* arch);

rocke_kernel_def_t* rocke_build_fwd_smallk_mfma_new(rocke_ir_builder_t* b,
                                                    const rocke_fwd_smallk_mfma_spec_t* spec,
                                                    const char* arch);

rocke_status_t rocke_fwd_smallk_mfma_lower_to_llvm(const rocke_fwd_smallk_mfma_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_FWD_SMALL_MFMA_H */
