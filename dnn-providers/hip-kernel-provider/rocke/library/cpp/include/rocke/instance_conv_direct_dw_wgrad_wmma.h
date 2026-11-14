/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_dw_wgrad_wmma.h -- C++ engine port of the
 * Toeplitz WMMA depthwise wgrad in
 * library/kernels/common/conv_direct_dw_wgrad_wmma.py.
 *
 * Stride-1 depthwise wgrad (cpg = kpg = 1) on the gfx1250 wave32 16x16x32
 * WMMA: one-hot dY pixels (A) times 16-byte NHWC reads of X (B), 8 channels
 * per wave. Each work item (image, row tile) is staged into LDS by buffer
 * loads (optionally carried one item ahead in registers); the block's
 * accumulators land in a [ws_replicas][C][KH * KW] fp32 scratch with
 * cascade atomics, folded by the wgrad workspace reduce.
 *
 *   Python (conv_direct_dw_wgrad_wmma.py)    C (this header)
 *   ------------------------------------     ---------------------------------------
 *   @dataclass(frozen) DwWgradWmmaSpec       rocke_direct_dw_wgrad_wmma_spec_t
 *     derived @property / methods            rocke_direct_dw_wgrad_wmma_{threads_per_block,
 *                                              block_ch,tiles,lds_bytes,n_split,grid}
 *     .kernel_name()                         rocke_direct_dw_wgrad_wmma_kernel_name
 *   dw_wgrad_wmma_reason                     rocke_direct_dw_wgrad_wmma_reason
 *   is_valid_dw_wgrad_wmma_spec              rocke_direct_dw_wgrad_wmma_is_valid_spec
 *   build_dw_wgrad_wmma(spec, arch)          rocke_build_direct_dw_wgrad_wmma(_new)
 *   (+ convenience: build -> lower .ll)      rocke_direct_dw_wgrad_wmma_lower_to_llvm
 *
 * AOT: the kernel declares the direct-conv kernarg block of
 * rocke_conv_direct_arg_names("wgrad"). N, H, W, Ho, Wo, groups and the
 * strides are runtime; KH, KW, PAD and the column padding (pad_w) are baked
 * capabilities, the tile is the knobs.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_DW_WGRAD_WMMA_H
#define ROCKE_INSTANCE_CONV_DIRECT_DW_WGRAD_WMMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* DW_WGRAD_WMMA_CH (conv_direct_dw_wgrad_wmma.py): channels per wave. */
#define ROCKE_DW_WGRAD_WMMA_CH 8
/* Most unrolled WMMAs per work item (dw_wgrad_wmma_reason). */
#define ROCKE_DW_WGRAD_WMMA_MAX_STEPS 1024

/* ===================================================================== *
 *  DwWgradWmmaSpec
 *
 *  @dataclass(frozen=True)
 *  class DwWgradWmmaSpec:
 *      problem: DirectConvProblem
 *      name: str = "direct_depthwise_wgrad_wmma"
 *      wave_size: int = 32
 *      waves: int = 4
 *      tile_q: int = 32
 *      rows: int = 8
 *      target_blocks: int = 1024
 *      ws_replicas: int = 64
 *      prefetch: bool = True
 * ===================================================================== */
typedef struct rocke_direct_dw_wgrad_wmma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "direct_depthwise_wgrad_wmma" */
    int wave_size; /* default 32 */
    int waves; /* default 4: channel groups of 8 per block */
    int tile_q; /* default 32: dY columns per work item (multiple of 4) */
    int rows; /* default 8: dY rows per work item (even) */
    int target_blocks; /* default 1024: n_split = ceil(target / grid x) */
    int ws_replicas; /* default 64 */
    bool prefetch; /* default true: carry the next item's loads in registers */
} rocke_direct_dw_wgrad_wmma_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_direct_dw_wgrad_wmma_spec_t rocke_direct_dw_wgrad_wmma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_direct_dw_wgrad_wmma_threads_per_block(const rocke_direct_dw_wgrad_wmma_spec_t* spec);
int rocke_direct_dw_wgrad_wmma_block_ch(const rocke_direct_dw_wgrad_wmma_spec_t* spec);
/* n_rb * n_s accumulator tiles: (KH // 2 + 1) * ceil(KW / 8). */
int rocke_direct_dw_wgrad_wmma_tiles(const rocke_direct_dw_wgrad_wmma_spec_t* spec);
/* (x_rows * x_cols + rows * tile_q) * ch_pitch * 2. */
long rocke_direct_dw_wgrad_wmma_lds_bytes(const rocke_direct_dw_wgrad_wmma_spec_t* spec);
/* n_split() -> max(1, min(ceil(target_blocks / grid x), N * ceil(Ho / rows))). */
int rocke_direct_dw_wgrad_wmma_n_split(const rocke_direct_dw_wgrad_wmma_spec_t* spec);
/* grid() -> (ceil(groups / block_ch) * ceil(Wo / tile_q), n_split, 1). */
void rocke_direct_dw_wgrad_wmma_grid(const rocke_direct_dw_wgrad_wmma_spec_t* spec, int grid[3]);

/* kernel_name(): kernel_name_join(name, short, f{KH}x{KW}p{PAD}x{PAD_W},
 * wv{}, q{}, r{}, tb{}, [wsr{}], [nopf], [bf16]). */
rocke_status_t rocke_direct_dw_wgrad_wmma_kernel_name(const rocke_direct_dw_wgrad_wmma_spec_t* spec,
                                                      char* out,
                                                      size_t out_cap);

/* dw_wgrad_wmma_reason(spec): the arch-independent rules. Returns true (and
 * writes the reason) when the spec cannot be built. */
bool rocke_direct_dw_wgrad_wmma_reason(const rocke_direct_dw_wgrad_wmma_spec_t* spec,
                                       char* reason,
                                       size_t reason_cap);

/* is_valid_dw_wgrad_wmma_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx1250". */
bool rocke_direct_dw_wgrad_wmma_is_valid_spec(const rocke_direct_dw_wgrad_wmma_spec_t* spec,
                                              const char* arch,
                                              char* reason,
                                              size_t reason_cap);

/* build_dw_wgrad_wmma(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_direct_dw_wgrad_wmma(rocke_ir_builder_t* b,
                                                     const rocke_direct_dw_wgrad_wmma_spec_t* spec,
                                                     const char* arch);

/* Validate, initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_direct_dw_wgrad_wmma_new(
    rocke_ir_builder_t* b, const rocke_direct_dw_wgrad_wmma_spec_t* spec, const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t
    rocke_direct_dw_wgrad_wmma_lower_to_llvm(const rocke_direct_dw_wgrad_wmma_spec_t* spec,
                                             const char* arch,
                                             rocke_llvm_flavor_t flavor,
                                             char** out_ll,
                                             char* err,
                                             size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_DW_WGRAD_WMMA_H */
