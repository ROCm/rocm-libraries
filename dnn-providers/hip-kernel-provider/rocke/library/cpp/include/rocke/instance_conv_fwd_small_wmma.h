/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_fwd_small_wmma.h -- C++ engine port of the wave32 WMMA
 * forward convolutions of tiny channel counts in
 * library/kernels/common/conv_fwd_small_wmma.py.
 *
 * Two kernels, both ungrouped and undilated on the gfx1250 WMMA 16x16x32
 * atom with TDM staging:
 *
 *   fwd_smallc_wmma  few input channels (C <= 4, stems): one GEMM over the
 *                    flattened (r, s, c) reduction; each block stages its
 *                    output tile's input window with one TDM transfer and
 *                    gathers the im2col fragments from LDS; the weights stay
 *                    in registers across `tiles_per_block` tiles.
 *   fwd_smallk_wmma  few output channels (K <= 8): Z = X * W[(r, s, k), c]
 *                    over the window, input channel chunks through a TDM
 *                    ring, then an fp32 overlap-add in LDS.
 *
 *   Python (conv_fwd_small_wmma.py)          C (this header)
 *   ------------------------------------
 * ---------------------------------------
 *   @dataclass(frozen) FwdSmallcWmmaSpec     rocke_fwd_smallc_wmma_spec_t
 *   @dataclass(frozen) FwdSmallkWmmaSpec     rocke_fwd_smallk_wmma_spec_t
 *     derived @property geometry rocke_fwd_small{c,k}_wmma_{block_*,
 *                                              threads_per_block,lds_bytes,grid}
 *     .kernel_name() rocke_fwd_small{c,k}_wmma_kernel_name
 *   fwd_small{c,k}_reason / is_valid_* rocke_fwd_small{c,k}_wmma_is_valid_spec
 *   build_fwd_small{c,k}_wmma(spec, arch) rocke_build_fwd_small{c,k}_wmma(_new)
 *   (+ convenience: build -> lower .ll) rocke_fwd_small{c,k}_wmma_lower_to_llvm
 *
 * AOT: both kernels declare the forward direct-conv kernarg block of
 * rocke_conv_direct_arg_names("fwd"). Batch, extents and the other channel
 * count are runtime; the filter, stride, PAD, the small channel count, the
 * dtype and the tile geometry are baked.
 *
 * `sched` is the optional Python scheduler_strategy (NULL = backend default).
 */
#ifndef ROCKE_INSTANCE_CONV_FWD_SMALL_WMMA_H
#define ROCKE_INSTANCE_CONV_FWD_SMALL_WMMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  FwdSmallcWmmaSpec
 *
 *  @dataclass(frozen=True)
 *  class FwdSmallcWmmaSpec:
 *      problem: DirectConvProblem
 *      name: str = "fwd_smallc_wmma"
 *      waves_k: int = 1          # waves along output channels
 *      waves_p: int = 4          # waves along output pixels
 *      tiles_k: int = 2          # 16-channel tiles per wave
 *      tiles_p: int = 4          # 16-pixel tiles per wave
 *      out_h: int = 8            # output rows per tile
 *      out_w: int = 32           # output columns per tile
 *      tiles_per_block: int = 1  # output tiles a block walks
 *      tdm_store: bool = False   # stage the output tile, one TDM store
 *      sched: str | None = None
 *      wave_size: int = 32
 * ===================================================================== */
typedef struct rocke_fwd_smallc_wmma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "fwd_smallc_wmma" */
    int waves_k; /* default 1 */
    int waves_p; /* default 4 */
    int tiles_k; /* default 2 */
    int tiles_p; /* default 4 */
    int out_h; /* default 8 */
    int out_w; /* default 32 */
    int tiles_per_block; /* default 1 */
    bool tdm_store; /* default false */
    const char* sched; /* default NULL */
    int wave_size; /* default 32 */
} rocke_fwd_smallc_wmma_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_fwd_smallc_wmma_spec_t rocke_fwd_smallc_wmma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_fwd_smallc_wmma_block_k(const rocke_fwd_smallc_wmma_spec_t* spec);
int rocke_fwd_smallc_wmma_block_p(const rocke_fwd_smallc_wmma_spec_t* spec);
int rocke_fwd_smallc_wmma_threads_per_block(const rocke_fwd_smallc_wmma_spec_t* spec);
long rocke_fwd_smallc_wmma_lds_bytes(const rocke_fwd_smallc_wmma_spec_t* spec);
/* spec.grid() -> (ceil(n_tiles / tiles_per_block), ceil(total_k / block_k), 1).
 */
void rocke_fwd_smallc_wmma_grid(const rocke_fwd_smallc_wmma_spec_t* spec, int grid[3]);

/* kernel_name(): kernel_name_join(name, f{KH}x{KW}, s{stride}, p{PAD}, c{C},
 * wk{}, wp{}, tk{}, tp{}, oh{}, ow{}, tb{}, [ts], [s{sched}], [bf16]). */
rocke_status_t rocke_fwd_smallc_wmma_kernel_name(const rocke_fwd_smallc_wmma_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap);

/* is_valid_fwd_smallc_spec(spec, arch) -> (ok, reason). `arch` NULL =>
 * "gfx1250". */
bool rocke_fwd_smallc_wmma_is_valid_spec(const rocke_fwd_smallc_wmma_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap);

/* build_fwd_smallc_wmma(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_fwd_smallc_wmma(rocke_ir_builder_t* b,
                                                const rocke_fwd_smallc_wmma_spec_t* spec,
                                                const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_fwd_smallc_wmma_new(rocke_ir_builder_t* b,
                                                    const rocke_fwd_smallc_wmma_spec_t* spec,
                                                    const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_fwd_smallc_wmma_lower_to_llvm(const rocke_fwd_smallc_wmma_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap);

/* ===================================================================== *
 *  FwdSmallkWmmaSpec
 *
 *  @dataclass(frozen=True)
 *  class FwdSmallkWmmaSpec:
 *      problem: DirectConvProblem
 *      name: str = "fwd_smallk_wmma"
 *      waves_c: int = 1     # waves along GEMM columns (r, s, k)
 *      waves_p: int = 4     # waves along window pixels
 *      tiles_c: int = 3
 *      tiles_p: int = 6
 *      out_h: int = 16      # output rows per block
 *      out_w: int = 16      # output columns per block
 *      ck: int = 32         # input channels per LDS stage
 *      stages: int = 2      # LDS ring depth
 *      prefetch: bool = False
 *      sched: str | None = None
 *      wave_size: int = 32
 * ===================================================================== */
typedef struct rocke_fwd_smallk_wmma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "fwd_smallk_wmma" */
    int waves_c; /* default 1  */
    int waves_p; /* default 4  */
    int tiles_c; /* default 3  */
    int tiles_p; /* default 6  */
    int out_h; /* default 16 */
    int out_w; /* default 16 */
    int ck; /* default 32 */
    int stages; /* default 2  */
    bool prefetch; /* default false */
    const char* sched; /* default NULL */
    int wave_size; /* default 32 */
} rocke_fwd_smallk_wmma_spec_t;

rocke_fwd_smallk_wmma_spec_t rocke_fwd_smallk_wmma_spec_default(void);

int rocke_fwd_smallk_wmma_block_c(const rocke_fwd_smallk_wmma_spec_t* spec);
int rocke_fwd_smallk_wmma_block_p(const rocke_fwd_smallk_wmma_spec_t* spec);
int rocke_fwd_smallk_wmma_threads_per_block(const rocke_fwd_smallk_wmma_spec_t* spec);
long rocke_fwd_smallk_wmma_lds_bytes(const rocke_fwd_smallk_wmma_spec_t* spec);
/* spec.grid() -> (N * ceil(Ho / out_h) * ceil(Wo / out_w), 1, 1). */
void rocke_fwd_smallk_wmma_grid(const rocke_fwd_smallk_wmma_spec_t* spec, int grid[3]);

/* kernel_name(): kernel_name_join(name, f{KH}x{KW}, s{stride}, p{PAD}, k{K},
 * wc{}, wp{}, tc{}, tp{}, oh{}, ow{}, ck{}, st{}, [pf], [s{sched}], [bf16]). */
rocke_status_t rocke_fwd_smallk_wmma_kernel_name(const rocke_fwd_smallk_wmma_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap);

/* is_valid_fwd_smallk_spec(spec, arch) -> (ok, reason). `arch` NULL =>
 * "gfx1250". */
bool rocke_fwd_smallk_wmma_is_valid_spec(const rocke_fwd_smallk_wmma_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap);

rocke_kernel_def_t* rocke_build_fwd_smallk_wmma(rocke_ir_builder_t* b,
                                                const rocke_fwd_smallk_wmma_spec_t* spec,
                                                const char* arch);

rocke_kernel_def_t* rocke_build_fwd_smallk_wmma_new(rocke_ir_builder_t* b,
                                                    const rocke_fwd_smallk_wmma_spec_t* spec,
                                                    const char* arch);

rocke_status_t rocke_fwd_smallk_wmma_lower_to_llvm(const rocke_fwd_smallk_wmma_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_FWD_SMALL_WMMA_H */
