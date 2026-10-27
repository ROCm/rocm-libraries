/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_dgrad_wmma.h -- C++ engine port of the wave32 WMMA
 * direct dgrad builder in library/kernels/common/conv_direct_dgrad_wmma.py.
 *
 * Stride-1 dgrad, dense and grouped, on the gfx1250 WMMA 16x16x32 atom: per
 * dY channel chunk, one 3-D TDM transfer stages the dY halo of the block's dX
 * tile and one stages the chunk's weights for every tap, in an LDS ring of
 * `stages` slots; the weights are transposed on the LDS read
 * (ds_load_tr16_b128) and every tap reads a shifted window of the halo. The dX
 * tile is staged back through LDS and stored with one TDM transfer per image.
 *
 *   Python (conv_direct_dgrad_wmma.py)       C (this header)
 *   ------------------------------------     ---------------------------------------
 *   @dataclass(frozen) DirectDgradWmmaSpec   rocke_direct_dgrad_wmma_spec_t
 *     derived @property geometry             rocke_direct_dgrad_wmma_{block_c,block_p,
 *                                              tile_h,threads_per_block,lds_bytes,grid}
 *     .kernel_name()                         rocke_direct_dgrad_wmma_kernel_name
 *   dgrad_wmma_reason / is_valid_dgrad_wmma_spec
 *                                            rocke_direct_dgrad_wmma_is_valid_spec
 *   build_direct_dgrad_wmma(spec, arch)      rocke_build_direct_dgrad_wmma(_new)
 *   (+ convenience: build -> lower .ll)      rocke_direct_dgrad_wmma_lower_to_llvm
 *
 * AOT: the kernel declares the direct-conv kernarg block of
 * rocke_conv_direct_arg_names("dgrad"). Batch, extents, the group count and
 * both channel counts are runtime; the filter, PAD, dtype and tile geometry
 * are baked, so the emitted IR does not depend on the problem's extents,
 * groups or channel counts.
 *
 * `sched` is the optional Python scheduler_strategy (NULL = backend default).
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_DGRAD_WMMA_H
#define ROCKE_INSTANCE_CONV_DIRECT_DGRAD_WMMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  DirectDgradWmmaSpec
 *
 *  @dataclass(frozen=True)
 *  class DirectDgradWmmaSpec:
 *      problem: DirectConvProblem
 *      name: str = "direct_dgrad_wmma"
 *      waves_c: int = 2      # waves along dX channels
 *      waves_p: int = 4      # waves along dX pixels
 *      tiles_c: int = 4      # 16-channel tiles per wave
 *      tiles_p: int = 4      # 16-pixel tiles per wave
 *      tile_w: int = 16      # dX columns per block
 *      tile_n: int = 1       # images per block
 *      ck: int = 32          # dY channels per LDS stage
 *      stages: int = 2       # LDS ring depth
 *      prefetch: bool = False
 *      sched: str | None = None
 *      wave_size: int = 32
 * ===================================================================== */
typedef struct rocke_direct_dgrad_wmma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "direct_dgrad_wmma" */
    int waves_c; /* default 2  */
    int waves_p; /* default 4  */
    int tiles_c; /* default 4  */
    int tiles_p; /* default 4  */
    int tile_w; /* default 16 */
    int tile_n; /* default 1  */
    int ck; /* default 32 */
    int stages; /* default 2  */
    bool prefetch; /* default false */
    const char* sched; /* default NULL */
    int wave_size; /* default 32 */
} rocke_direct_dgrad_wmma_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_direct_dgrad_wmma_spec_t rocke_direct_dgrad_wmma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_direct_dgrad_wmma_block_c(const rocke_direct_dgrad_wmma_spec_t* spec);
int rocke_direct_dgrad_wmma_block_p(const rocke_direct_dgrad_wmma_spec_t* spec);
int rocke_direct_dgrad_wmma_tile_h(const rocke_direct_dgrad_wmma_spec_t* spec);
int rocke_direct_dgrad_wmma_threads_per_block(const rocke_direct_dgrad_wmma_spec_t* spec);
long rocke_direct_dgrad_wmma_lds_bytes(const rocke_direct_dgrad_wmma_spec_t* spec);
/* spec.grid() -> (ceil(N / tile_n) * groups * n_h_tiles * n_w_tiles * n_c_tiles, 1, 1). */
void rocke_direct_dgrad_wmma_grid(const rocke_direct_dgrad_wmma_spec_t* spec, int grid[3]);

/* kernel_name(): kernel_name_join(name, f{KH}x{KW}, p{PAD}, wc{}, wp{}, tc{}, tp{},
 * tw{}, [tn{}], ck{}, st{}, [pf], [s{sched}], [bf16]). */
rocke_status_t rocke_direct_dgrad_wmma_kernel_name(const rocke_direct_dgrad_wmma_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap);

/* is_valid_dgrad_wmma_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx1250". */
bool rocke_direct_dgrad_wmma_is_valid_spec(const rocke_direct_dgrad_wmma_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap);

/* build_direct_dgrad_wmma(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_direct_dgrad_wmma(rocke_ir_builder_t* b,
                                                  const rocke_direct_dgrad_wmma_spec_t* spec,
                                                  const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_direct_dgrad_wmma_new(rocke_ir_builder_t* b,
                                                      const rocke_direct_dgrad_wmma_spec_t* spec,
                                                      const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_direct_dgrad_wmma_lower_to_llvm(const rocke_direct_dgrad_wmma_spec_t* spec,
                                                     const char* arch,
                                                     rocke_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_DGRAD_WMMA_H */
