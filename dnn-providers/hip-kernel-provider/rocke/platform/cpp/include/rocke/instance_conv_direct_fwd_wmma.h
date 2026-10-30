/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_fwd_wmma.h -- C++ engine port of the wave32 WMMA
 * direct forward builder in library/kernels/common/conv_direct_fwd_wmma.py.
 *
 * Dense and grouped forward convolution on the gfx1250 WMMA 16x16x32 atom:
 * per input channel chunk, one 3-D TDM transfer stages the X halo of the
 * block's output tile and one stages the chunk's weights, in an LDS ring of
 * `stages` slots; every filter tap reads a shifted window of the halo. The Y
 * tile is staged back through LDS and stored with TDM.
 *
 *   Python (conv_direct_fwd_wmma.py)         C (this header)
 *   ------------------------------------     ---------------------------------------
 *   @dataclass(frozen) DirectFwdWmmaSpec     rocke_direct_fwd_wmma_spec_t
 *     derived @property geometry             rocke_direct_fwd_wmma_{block_k,block_p,
 *                                              tile_h,threads_per_block,lds_bytes,grid}
 *     .kernel_name()                         rocke_direct_fwd_wmma_kernel_name
 *   fwd_wmma_reason / is_valid_fwd_wmma_spec rocke_direct_fwd_wmma_is_valid_spec
 *   build_direct_fwd_wmma(spec, arch)        rocke_build_direct_fwd_wmma(_new)
 *   (+ convenience: build -> lower .ll)      rocke_direct_fwd_wmma_lower_to_llvm
 *
 * AOT: the kernel declares the direct-conv kernarg block of
 * rocke_conv_direct_arg_names("fwd"). Batch, extents, the group count and
 * both channel counts are runtime (pack_groups > 1 bakes cpg and kpg); the
 * filter, PAD, stride, dilation, dtype and tile geometry are baked.
 *
 * `sched` is the optional Python scheduler_strategy (NULL = backend default).
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_FWD_WMMA_H
#define ROCKE_INSTANCE_CONV_DIRECT_FWD_WMMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  DirectFwdWmmaSpec
 *
 *  @dataclass(frozen=True)
 *  class DirectFwdWmmaSpec:
 *      problem: DirectConvProblem
 *      name: str = "direct_fwd_wmma"
 *      waves_k: int = 2          # waves along output channels
 *      waves_p: int = 4          # waves along output pixels
 *      tiles_k: int = 4          # 16-channel tiles per wave
 *      tiles_p: int = 4          # 16-pixel tiles per wave
 *      tile_w: int = 16          # output columns per block
 *      tile_n: int = 1           # images per block
 *      ck: int = 32              # input channels per LDS stage
 *      stages: int = 2           # LDS ring depth
 *      prefetch: bool = False
 *      sched: str | None = None
 *      wave_size: int = 32
 *      pack_groups: int = 1      # groups per block-diagonal super-group
 *      split_barrier: bool = False
 *      wave_epilogue: bool = False
 *      phase_split: bool = False # stride 2: even / odd column sub-halos
 *      row_ring: bool = False    # weight ring per filter row
 *      tap_ring: bool = False    # weight ring per filter tap
 *      pingpong: bool = False    # two wave ranks a barrier apart
 *      pp_group: int = 1         # steps per ping-pong phase
 *      tap_reuse: bool = False   # reuse B fragments across filter rows
 *      tap_shift: bool = False   # next filter column by v_permlane16
 * ===================================================================== */
typedef struct rocke_direct_fwd_wmma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "direct_fwd_wmma" */
    int waves_k; /* default 2  */
    int waves_p; /* default 4  */
    int tiles_k; /* default 4  */
    int tiles_p; /* default 4  */
    int tile_w; /* default 16 */
    int tile_n; /* default 1  */
    int ck; /* default 32 */
    int stages; /* default 2  */
    bool prefetch; /* default false */
    const char* sched; /* default NULL */
    int wave_size; /* default 32 */
    int pack_groups; /* default 1 */
    bool split_barrier; /* default false */
    bool wave_epilogue; /* default false */
    bool phase_split; /* default false */
    bool row_ring; /* default false */
    bool tap_ring; /* default false */
    bool pingpong; /* default false */
    int pp_group; /* default 1 */
    bool tap_reuse; /* default false */
    bool tap_shift; /* default false */
} rocke_direct_fwd_wmma_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_direct_fwd_wmma_spec_t rocke_direct_fwd_wmma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_direct_fwd_wmma_block_k(const rocke_direct_fwd_wmma_spec_t* spec);
int rocke_direct_fwd_wmma_block_p(const rocke_direct_fwd_wmma_spec_t* spec);
int rocke_direct_fwd_wmma_tile_h(const rocke_direct_fwd_wmma_spec_t* spec);
int rocke_direct_fwd_wmma_threads_per_block(const rocke_direct_fwd_wmma_spec_t* spec);
long rocke_direct_fwd_wmma_lds_bytes(const rocke_direct_fwd_wmma_spec_t* spec);
/* spec.grid() -> (ceil(N / tile_n) * groups_eff * n_h_tiles * n_w_tiles * n_k_tiles, 1, 1). */
void rocke_direct_fwd_wmma_grid(const rocke_direct_fwd_wmma_spec_t* spec, int grid[3]);

/* kernel_name(): kernel_name_join(name, f{KH}x{KW}, p{PAD}, wk{}, wp{}, tk{}, tp{},
 * tw{}, [tn{}], ck{}, st{}, [pf], [s{sched}], [str{}], [dil{}x{}], [pg{}c{}k{}],
 * [sb], [we], [ps], [rr], [tr], [pp], [pg{}], [tu], [tsh], [bf16]). */
rocke_status_t rocke_direct_fwd_wmma_kernel_name(const rocke_direct_fwd_wmma_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap);

/* is_valid_fwd_wmma_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx1250". */
bool rocke_direct_fwd_wmma_is_valid_spec(const rocke_direct_fwd_wmma_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap);

/* build_direct_fwd_wmma(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_direct_fwd_wmma(rocke_ir_builder_t* b,
                                                const rocke_direct_fwd_wmma_spec_t* spec,
                                                const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_direct_fwd_wmma_new(rocke_ir_builder_t* b,
                                                    const rocke_direct_fwd_wmma_spec_t* spec,
                                                    const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_direct_fwd_wmma_lower_to_llvm(const rocke_direct_fwd_wmma_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_FWD_WMMA_H */
