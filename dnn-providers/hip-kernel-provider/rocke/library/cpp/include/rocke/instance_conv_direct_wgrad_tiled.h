/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_wgrad_tiled.h -- C++ engine port of the tiled
 * direct wgrad builder in library/kernels/common/conv_direct_wgrad_tiled.py.
 *
 * A stride-1 NHWC weight gradient: a wave owns wave_k16*16 x wave_c16*16
 * channels for all KH*KW taps, the block streams input rows of a 32-column
 * strip through LDS ring buffers filled by buffer_load ... lds, with an XOR
 * chunk swizzle that keeps the ds_read_b64_tr_b16 fragment reads bank-conflict
 * free, and atomically adds its fp32 accumulators into dW once at the end.
 *
 *   Python (conv_direct_wgrad_tiled.py)        C (this header)
 *   -----------------------------------   ------------------------------------
 *   @dataclass(frozen) DirectWgradTiledSpec rocke_direct_wgrad_tiled_spec_t
 *     derived @property geometry          rocke_direct_wgrad_tiled_{threads_per_block,
 *                                           unroll,acc_vgprs,lds_bytes,n_channel_tiles,
 *                                           n_segments,n_splits,launch_geometry}
 *     .kernel_name()                      rocke_direct_wgrad_tiled_kernel_name
 *   is_valid_wgrad_tiled_spec(spec, arch) rocke_direct_wgrad_tiled_is_valid_spec
 *   pick_swizzle(chunks, shifts, ...)     rocke_direct_wgrad_pick_swizzle
 *   build_direct_wgrad_tiled(spec, arch)  rocke_build_direct_wgrad_tiled(_new)
 *   (+ convenience: build -> lower .ll)   rocke_direct_wgrad_tiled_lower_to_llvm
 *
 * AOT: the kernel declares the direct-conv kernarg block of
 * rocke_conv_direct_arg_names("wgrad"); N, H, W, Ho, Wo, groups, the channel
 * totals and the strides are runtime, the filter, PAD, dtype and tile geometry
 * are baked. dW is fp32 and the caller zero-initialises it, unless the spec
 * stores it (one_split; dw16 in the activation dtype), adds packed 16-bit
 * pairs into it (pk16) or stores fp32 slabs for the slab fold below (slab).
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_WGRAD_TILED_H
#define ROCKE_INSTANCE_CONV_DIRECT_WGRAD_TILED_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  DirectWgradTiledSpec
 *
 *  @dataclass(frozen=True)
 *  class DirectWgradTiledSpec:
 *      problem: DirectConvProblem
 *      name: str = "direct_wgrad_tiled"
 *      wave_k16: int = 2
 *      wave_c16: int = 2
 *      waves_k: int = 4
 *      waves_c: int = 2
 *      rows_per_seg: int = 30
 *      prefetch: int = 2
 *      target_blocks: int = 256
 *      row_unroll: int = 0
 *      unfold_n: int = 1
 *      q_groups: int = 1
 *      pingpong: bool = False
 *      pp_tight: bool = False
 *      row_tail: bool = False
 *      one_split: bool = False
 *      dw16: bool = False
 *      pk16: bool = False
 *      gpack: bool = False
 *      slab: bool = False
 *      l64: bool = False
 *      pp_sched: bool = False
 * ===================================================================== */
typedef struct rocke_direct_wgrad_tiled_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "direct_wgrad_tiled" */
    int wave_k16; /* default 2 */
    int wave_c16; /* default 2 */
    int waves_k; /* default 4 */
    int waves_c; /* default 2 */
    int rows_per_seg; /* default 30; 0 = whole strips, one-round grid */
    int prefetch; /* default 2 */
    int target_blocks; /* default 256 */
    int row_unroll; /* default 0 (= lcm(n_buffers, KH)) */
    int unfold_n; /* default 1 */
    int q_groups; /* default 1; 1, 2 or 4 wave groups per block */
    bool pingpong; /* default false; with q_groups 1 or 4 the two wave halves */
    bool pp_tight; /* ping-pong in halves without the extra row buffer */
    bool row_tail; /* auto launch: whole unrolled blocks plus a row tail */
    bool one_split; /* one block per channel tile: dW stored, not added */
    bool dw16; /* one_split: dW stored in the activation dtype */
    bool pk16; /* fp16 packed-pair atomics into the 16-bit dW */
    bool gpack; /* small groups: a wave owns a whole group */
    bool slab; /* fp32 tile stored into slab `split` (WgradSlabFoldSpec) */
    bool l64; /* per-segment descriptors rebased (64-bit) to its first image */
    bool pp_sched; /* ping-pong: scheduling fences around the row's DMA setup */
} rocke_direct_wgrad_tiled_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_direct_wgrad_tiled_spec_t rocke_direct_wgrad_tiled_spec_default(void);

/* Derived geometry (Python @property / host-side methods). */
int rocke_direct_wgrad_tiled_threads_per_block(const rocke_direct_wgrad_tiled_spec_t* spec);
int rocke_direct_wgrad_tiled_unroll(const rocke_direct_wgrad_tiled_spec_t* spec);
int rocke_direct_wgrad_tiled_acc_vgprs(const rocke_direct_wgrad_tiled_spec_t* spec);
/* lds_bytes(): the padded dY / X rings plus the q-group reduction buffers. */
long rocke_direct_wgrad_tiled_lds_bytes(const rocke_direct_wgrad_tiled_spec_t* spec);
int rocke_direct_wgrad_tiled_n_channel_tiles(const rocke_direct_wgrad_tiled_spec_t* spec);
int rocke_direct_wgrad_tiled_n_segments(const rocke_direct_wgrad_tiled_spec_t* spec);
int rocke_direct_wgrad_tiled_n_splits(const rocke_direct_wgrad_tiled_spec_t* spec);
/* launch_geometry() -> (grid, block). */
void rocke_direct_wgrad_tiled_launch_geometry(const rocke_direct_wgrad_tiled_spec_t* spec,
                                              int grid[3],
                                              int block[3]);

/* kernel_name(): kernel_name_join(name, f{KH}x{KW}p{PAD}[x{PAD_W}], wt{wk}x{wc},
 * w{waves_k}x{waves_c}, rps{n}, pf{n}, tb{n}, flags [bf16, u{n}, un{n},
 * qg{n}, pp, ppt, rt, os, d16, pk, gp, sl, l64, pps]). */
rocke_status_t rocke_direct_wgrad_tiled_kernel_name(const rocke_direct_wgrad_tiled_spec_t* spec,
                                                    char* out,
                                                    size_t out_cap);

/* is_valid_wgrad_tiled_spec(spec, arch) -> (ok, reason). arch NULL => "gfx950". */
bool rocke_direct_wgrad_tiled_is_valid_spec(const rocke_direct_wgrad_tiled_spec_t* spec,
                                            const char* arch,
                                            char* reason,
                                            size_t reason_cap);

/* pick_swizzle(chunks, shifts, wu, seg, col_origins): the bit selection
 * `bits` of the chunk XOR 2 * sum(((row >> bits[j]) & 1) << j) that makes
 * every fragment read conflict free (or none). Writes up to 6 bits into
 * bits_out and returns how many. Shared with conv_direct_wgrad_gather. */
int rocke_direct_wgrad_pick_swizzle(int chunks,
                                    const int* shifts,
                                    int n_shifts,
                                    int wu,
                                    int seg,
                                    const int* col_origins,
                                    int n_origins,
                                    int bits_out[6]);

/* build_direct_wgrad_tiled(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_direct_wgrad_tiled(rocke_ir_builder_t* b,
                                                   const rocke_direct_wgrad_tiled_spec_t* spec,
                                                   const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_direct_wgrad_tiled_new(rocke_ir_builder_t* b,
                                                       const rocke_direct_wgrad_tiled_spec_t* spec,
                                                       const char* arch);

/* Convenience: build + lower to LLVM IR text (*out_ll malloc'd). */
rocke_status_t rocke_direct_wgrad_tiled_lower_to_llvm(const rocke_direct_wgrad_tiled_spec_t* spec,
                                                      const char* arch,
                                                      rocke_llvm_flavor_t flavor,
                                                      char** out_ll,
                                                      char* err,
                                                      size_t err_cap);

/* ===================================================================== *
 *  WgradSlabFoldSpec -- the slab epilogue's reduce:
 *  dW[e] = sum_split ws[split][e], ws fp32, dW in `out` ("fp32", "fp16" or
 *  "bf16"). Shape-generic: the slab count and the element count are kernargs
 *  (A, D, A_bytes, D_bytes, parts, n_elems).
 *
 *  @dataclass(frozen=True)
 *  class WgradSlabFoldSpec:
 *      out: str = "fp16"
 *      slices: int = 8
 *      name: str = "direct_wgrad_slab_fold"
 * ===================================================================== */
typedef struct rocke_direct_wgrad_slab_fold_spec
{
    const char* out; /* default "fp16" */
    int slices; /* default 8; divides 256, at most 64 */
    const char* name; /* default "direct_wgrad_slab_fold" */
} rocke_direct_wgrad_slab_fold_spec_t;

rocke_direct_wgrad_slab_fold_spec_t rocke_direct_wgrad_slab_fold_spec_default(void);
/* elems_per_block = 4 * (256 / slices); launch_geometry(n_elems). */
int rocke_direct_wgrad_slab_fold_elems_per_block(const rocke_direct_wgrad_slab_fold_spec_t* spec);
void rocke_direct_wgrad_slab_fold_launch_geometry(const rocke_direct_wgrad_slab_fold_spec_t* spec,
                                                  int n_elems,
                                                  int grid[3],
                                                  int block[3]);
/* kernel_name_join(name, s{slices}, out). */
rocke_status_t rocke_direct_wgrad_slab_fold_kernel_name(
    const rocke_direct_wgrad_slab_fold_spec_t* spec, char* out, size_t out_cap);
/* is_valid_slab_fold_spec(spec) -> (ok, reason). */
bool rocke_direct_wgrad_slab_fold_is_valid_spec(const rocke_direct_wgrad_slab_fold_spec_t* spec,
                                                char* reason,
                                                size_t reason_cap);
/* build_wgrad_slab_fold(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_direct_wgrad_slab_fold(
    rocke_ir_builder_t* b, const rocke_direct_wgrad_slab_fold_spec_t* spec, const char* arch);
rocke_kernel_def_t* rocke_build_direct_wgrad_slab_fold_new(
    rocke_ir_builder_t* b, const rocke_direct_wgrad_slab_fold_spec_t* spec, const char* arch);
rocke_status_t
    rocke_direct_wgrad_slab_fold_lower_to_llvm(const rocke_direct_wgrad_slab_fold_spec_t* spec,
                                               const char* arch,
                                               rocke_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_WGRAD_TILED_H */
