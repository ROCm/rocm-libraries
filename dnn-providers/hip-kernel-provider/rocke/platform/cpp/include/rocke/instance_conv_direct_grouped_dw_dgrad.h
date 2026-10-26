/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_grouped_dw_dgrad.h -- C++ engine port of the
 * depthwise dgrad kernels of library/kernels/common/conv_direct_grouped.py
 * that the original rocke/instance_conv_direct_grouped.h does not cover:
 *
 *   - DirectDepthwiseDgradStreamSpec: the ho-streaming depthwise dgrad
 *     (any stride, cpg = kpg = 1), including its channel-packed form
 *     (cpl = 2, 4 or 8 channels per lane, packed <2 x f32> FMAs);
 *   - the helpers of the rotated-filter dgrad mode of
 *     DirectDepthwiseTiledSpec (dgrad = true) and of the per-axis padding
 *     (DirectConvProblem.pad_w) it takes.
 *
 *   Python (conv_direct_grouped.py)                    C (this header)
 *   -----------------------------------------------   -------------------------------------------
 *   @dataclass(frozen) DirectDepthwiseDgradStreamSpec  rocke_direct_depthwise_dgrad_stream_spec_t
 *     .threads_per_block / .block_ch                   rocke_direct_depthwise_dgrad_stream_{threads_per_block,block_ch}
 *     .kernel_name() / .validate()                     rocke_direct_depthwise_dgrad_stream_{kernel_name,validate}
 *   is_valid_depthwise_dgrad_stream_spec(spec, arch)   rocke_direct_depthwise_dgrad_stream_is_valid_spec
 *   build_direct_depthwise_dgrad_streaming(spec, arch) rocke_build_direct_depthwise_dgrad_streaming(_new)
 *   _build_direct_depthwise_dgrad_streaming_packed     (cpl > 1 path of the builder above)
 *   DirectDepthwiseTiledSpec.validate() dgrad / pad_w  rocke_dconv_dw_tiled_padding_reason
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_GROUPED_DW_DGRAD_H
#define ROCKE_INSTANCE_CONV_DIRECT_GROUPED_DW_DGRAD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  DirectDepthwiseDgradStreamSpec  (cpg = kpg = 1 dgrad, any stride)
 *
 *  @dataclass(frozen=True)
 *  class DirectDepthwiseDgradStreamSpec:
 *      problem: DirectConvProblem
 *      name: str = "direct_depthwise_dgrad_stream"
 *      block_w: int = 8
 *      block_waves: int = 1
 *      wave_size: int = 64
 *      cpl: int = 1          # channels per lane: 1, 2, 4 or 8
 *
 *  Streams dY rows in a runtime loop, KH rows per iteration, into
 *  stride * KH circular dX accumulator slots that are flushed once complete.
 *  The KH x KW weights are preloaded. cpl > 1 gives every lane cpl adjacent
 *  channels: dY and dX move as one cpl-wide vector per lane and every tap is
 *  cpl / 2 packed <2 x f32> FMAs over channel pairs; cpl == 1 is the scalar
 *  kernel.
 *
 *  Takes the direct dgrad kernarg block; KH, KW, PAD and stride are baked.
 *  Grid: (ceil(W / block_w), ceil(groups / block_ch), N)
 *  Block: (block_waves * wave_size, 1, 1)
 * ===================================================================== */
typedef struct rocke_direct_depthwise_dgrad_stream_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "direct_depthwise_dgrad_stream" */
    int block_w; /* default 8 */
    int block_waves; /* default 1 */
    int wave_size; /* default 64 */
    int cpl; /* default 1 */
} rocke_direct_depthwise_dgrad_stream_spec_t;

/* Bound on cpl * block_w * KH for cpl > 1 (Python: _DW_DGRAD_CPL_MAX_ACC):
 * past it the packed accumulator slots spill. The packed weights are bounded
 * by the preloading depthwise tap limit (_DW_MAX_PRELOAD_TAPS). */
#define ROCKE_DCONV_DW_DGRAD_CPL_MAX_ACC 192
#define ROCKE_DCONV_DW_DGRAD_STREAM_MAX_PRELOAD_TAPS 200

rocke_direct_depthwise_dgrad_stream_spec_t rocke_direct_depthwise_dgrad_stream_spec_default(void);
/* @property threads_per_block -> block_waves * wave_size */
int rocke_direct_depthwise_dgrad_stream_threads_per_block(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec);
/* @property block_ch -> block_waves * wave_size * cpl */
int rocke_direct_depthwise_dgrad_stream_block_ch(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec);
/* kernel_name_join(name, p.short(), f"bw{block_w}", f"bw{block_waves}wv",
 *                  [f"cpl{cpl}" if cpl > 1], flags={"bf16": dtype == "bf16"}) */
rocke_status_t rocke_direct_depthwise_dgrad_stream_kernel_name(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec, char* out, size_t out_cap);
rocke_status_t rocke_direct_depthwise_dgrad_stream_validate(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec, char* reason, size_t reason_cap);
bool rocke_direct_depthwise_dgrad_stream_is_valid_spec(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap);

/* Build into an initialised builder `b`; NULL on an invalid spec. */
rocke_kernel_def_t* rocke_build_direct_depthwise_dgrad_streaming(
    rocke_ir_builder_t* b,
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec,
    const char* arch);
/* Initialises `b` with spec.kernel_name(), then builds. Caller owns `b`. */
rocke_kernel_def_t* rocke_build_direct_depthwise_dgrad_streaming_new(
    rocke_ir_builder_t* b,
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec,
    const char* arch);
/* Build + lower to LLVM IR text (*out_ll is malloc'd; caller frees). */
rocke_status_t rocke_direct_depthwise_dgrad_stream_lower_to_llvm(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap);

/* ===================================================================== *
 *  DirectDepthwiseTiledSpec dgrad mode / per-axis padding
 * ===================================================================== */

/* The first two checks of DirectDepthwiseTiledSpec.validate(): with dgrad,
 * stride 1 and padding below the filter on both axes; without it, one padding
 * for both axes. Returns true (and the reason) when the spec fails them. */
bool rocke_dconv_dw_tiled_padding_reason(const rocke_direct_depthwise_tiled_spec_t* spec,
                                         char* reason,
                                         size_t reason_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_GROUPED_DW_DGRAD_H */
