/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_wgrad_wmma.h -- C++ engine port of the wave32
 * WMMA dense direct wgrad builder in
 * library/kernels/common/conv_direct_wgrad_wmma.py.
 *
 * Dense (groups == 1) stride-1 direct wgrad on the gfx1250 WMMA 16x16x32
 * atom, streamed one input row per step: the X row and the matching dY row
 * are staged in an LDS ring of `stages` x `rows` row buffers as
 * [column][channel], by per-lane global_load_async_to_lds copies or (tdm)
 * one TDM transfer per operand, and read back transposed with
 * ds_load_tr16_b128. The dY fragments of the KH - 1 previous rows stay in a
 * register ring. A block owns one block_k x block_c dW tile and one run of
 * the flat row-step list; it adds its tile into the zeroed fp32 dW with
 * cascade atomics.
 *
 *   Python (conv_direct_wgrad_wmma.py)       C (this header)
 *   ------------------------------------     ---------------------------------------
 *   @dataclass(frozen) DirectWgradWmmaSpec   rocke_direct_wgrad_wmma_spec_t
 *     derived @property / methods            rocke_direct_wgrad_wmma_{threads,block_k,
 *                                              block_c,block_q,x_cols,pitch_c,pitch_k,
 *                                              n_k_tiles,n_c_tiles,total_steps,n_split,
 *                                              vgprs,lds_bytes,launch_geometry}
 *     .kernel_name()                         rocke_direct_wgrad_wmma_kernel_name
 *   is_valid_direct_wgrad_wmma_spec          rocke_direct_wgrad_wmma_is_valid_spec
 *   build_direct_wgrad_wmma(spec, arch)      rocke_build_direct_wgrad_wmma(_new)
 *   (+ convenience: build -> lower .ll)      rocke_direct_wgrad_wmma_lower_to_llvm
 *
 * AOT: the kernel declares the direct-conv kernarg block of
 * rocke_conv_direct_arg_names("wgrad"). Batch, extents and both channel
 * counts are runtime; the filter, PAD, dtype and tile geometry are baked, so
 * the emitted IR does not depend on the problem's extents or channel counts.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_WGRAD_WMMA_H
#define ROCKE_INSTANCE_CONV_DIRECT_WGRAD_WMMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  DirectWgradWmmaSpec
 *
 *  @dataclass(frozen=True)
 *  class DirectWgradWmmaSpec:
 *      problem: DirectConvProblem
 *      name: str = "direct_wgrad_wmma"
 *      waves_k: int = 4
 *      waves_c: int = 2
 *      tiles_k: int = 2        # 16x16 dW tiles per tap along K, per wave
 *      tiles_c: int = 2        # ... along C
 *      q_steps: int = 1        # 32-column WMMA steps per column chunk
 *      stages: int = 3         # LDS row buffers (steps in flight + 1)
 *      target_blocks: int = 256
 *      prefetch: bool = False  # read step g + 1's fragments during step g
 *      pad: int = 8            # LDS row padding (channels)
 *      sched: bool = False     # fence copies and reads ahead of the WMMAs
 *      xcd: int = 0            # XCDs to group the runs by (1-D grid)
 *      tdm: bool = False       # rows move by TDM, not per-lane copies
 *      rows: int = 1           # input rows staged per barrier
 * ===================================================================== */
typedef struct rocke_direct_wgrad_wmma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "direct_wgrad_wmma" */
    int waves_k; /* default 4 */
    int waves_c; /* default 2 */
    int tiles_k; /* default 2 */
    int tiles_c; /* default 2 */
    int q_steps; /* default 1 */
    int stages; /* default 3 */
    int target_blocks; /* default 256 */
    bool prefetch; /* default false */
    int pad; /* default 8 */
    bool sched; /* default false */
    int xcd; /* default 0 */
    bool tdm; /* default false */
    int rows; /* default 1 */
} rocke_direct_wgrad_wmma_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_direct_wgrad_wmma_spec_t rocke_direct_wgrad_wmma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_direct_wgrad_wmma_threads(const rocke_direct_wgrad_wmma_spec_t* spec);
int rocke_direct_wgrad_wmma_block_k(const rocke_direct_wgrad_wmma_spec_t* spec);
int rocke_direct_wgrad_wmma_block_c(const rocke_direct_wgrad_wmma_spec_t* spec);
int rocke_direct_wgrad_wmma_block_q(const rocke_direct_wgrad_wmma_spec_t* spec);
int rocke_direct_wgrad_wmma_x_cols(const rocke_direct_wgrad_wmma_spec_t* spec);
int rocke_direct_wgrad_wmma_pitch_c(const rocke_direct_wgrad_wmma_spec_t* spec);
int rocke_direct_wgrad_wmma_pitch_k(const rocke_direct_wgrad_wmma_spec_t* spec);
long long rocke_direct_wgrad_wmma_n_k_tiles(const rocke_direct_wgrad_wmma_spec_t* spec);
long long rocke_direct_wgrad_wmma_n_c_tiles(const rocke_direct_wgrad_wmma_spec_t* spec);
/* ceil(Wo / block_q) * N * (H + KH - 1): the row steps of every unit. */
long long rocke_direct_wgrad_wmma_total_steps(const rocke_direct_wgrad_wmma_spec_t* spec);
/* max(1, min(total_steps, round(target_blocks / tiles))): the runs along the
 * step list. 0 when the spec has no tile (Python divides by zero there). */
long long rocke_direct_wgrad_wmma_n_split(const rocke_direct_wgrad_wmma_spec_t* spec);
long long rocke_direct_wgrad_wmma_vgprs(const rocke_direct_wgrad_wmma_spec_t* spec);
long long rocke_direct_wgrad_wmma_lds_bytes(const rocke_direct_wgrad_wmma_spec_t* spec);
/* launch_geometry() -> (grid, block): xcd ? (tiles * n_split, 1, 1) : (tiles,
 * 1, n_split); block (threads, 1, 1). Only meaningful for a valid spec. */
void rocke_direct_wgrad_wmma_launch_geometry(const rocke_direct_wgrad_wmma_spec_t* spec,
                                             long long grid[3],
                                             int block[3]);

/* kernel_name(): kernel_name_join(name, f{KH}x{KW}p{PAD}s{stride}, w{wk}x{wc},
 * t{tk}x{tc}, qs{}, st{}, tb{}, [pad{}], flags bf16 / pf / sched / xcd{} /
 * tdm / g{rows}). */
rocke_status_t rocke_direct_wgrad_wmma_kernel_name(const rocke_direct_wgrad_wmma_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap);

/* is_valid_direct_wgrad_wmma_spec(spec, arch) -> (ok, reason). `arch` NULL =>
 * "gfx1250". */
bool rocke_direct_wgrad_wmma_is_valid_spec(const rocke_direct_wgrad_wmma_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap);

/* build_direct_wgrad_wmma(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_direct_wgrad_wmma(rocke_ir_builder_t* b,
                                                  const rocke_direct_wgrad_wmma_spec_t* spec,
                                                  const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_direct_wgrad_wmma_new(rocke_ir_builder_t* b,
                                                      const rocke_direct_wgrad_wmma_spec_t* spec,
                                                      const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_direct_wgrad_wmma_lower_to_llvm(const rocke_direct_wgrad_wmma_spec_t* spec,
                                                     const char* arch,
                                                     rocke_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_WGRAD_WMMA_H */
