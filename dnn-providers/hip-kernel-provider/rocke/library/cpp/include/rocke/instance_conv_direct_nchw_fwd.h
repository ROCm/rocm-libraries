/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_nchw_fwd.h -- C++ engine port of the native-NCHW
 * direct forward convolution builder in
 * library/kernels/common/conv_direct_nchw_fwd.py.
 *
 * A dense (groups == 1) forward that consumes X in NCHW and produces Y in
 * NKHW directly (no host transpose of the activations), reusing the design of
 * the native dgrad prototype: a workgroup owns a tile_i x tile_j block of
 * output positions and tile_k output channels, stages the input window in LDS
 * as [row][col][c] (c contiguous, padded by lds_pad; rows padded by
 * lds_wpad), and reads the window at column wo*S + s per filter tap s. At
 * stride 2, col_split splits the LDS columns into even/odd planes so the MFMA
 * fragment lanes read adjacent positions (bank-conflict free).
 *
 *   Python (conv_direct_nchw_fwd.py)        C++ (this header)
 *   ------------------------------------    ---------------------------------
 *   @dataclass(frozen) DirectNchwFwdSpec     rocke_direct_nchw_fwd_spec_t
 *     .kernel_name()                        rocke_direct_nchw_fwd_kernel_name
 *   is_valid_direct_nchw_fwd_spec            rocke_direct_nchw_fwd_is_valid_spec
 *   build_direct_conv_nchw_fwd(spec, arch)   rocke_build_direct_nchw_fwd(_new)
 *   (+ convenience: build -> lower .ll)      rocke_direct_nchw_fwd_lower_to_llvm
 *
 * AOT: the kernel declares the forward direct-conv kernarg block
 * (rocke_dconv_emit_params, direction "fwd"). Batch, extents and both channel
 * counts are runtime; the filter, stride, PAD and every knob are baked.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_NCHW_FWD_H
#define ROCKE_INSTANCE_CONV_DIRECT_NCHW_FWD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  DirectNchwFwdSpec
 *
 *  @dataclass(frozen=True)
 *  class DirectNchwFwdSpec:
 *      problem: DirectConvProblem
 *      name: str = "conv_direct_nchw_fwd"
 *      tile_i: int = 8
 *      tile_j: int = 32
 *      tile_k: int = 64
 *      ck: int = 32
 *      waves_m: int = 2
 *      waves_n: int = 2
 *      atom: str = "32x32x16"
 *      wave_size: int = 64
 *      lds_pad: int = 8
 *      lds_wpad: int = 0
 *      stage_group: int = 4
 *      col_split: bool = False
 *      waves_per_eu: int = 0
 *      pin_prefetch: bool = False
 * ===================================================================== */
typedef struct rocke_direct_nchw_fwd_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "conv_direct_nchw_fwd" */
    int tile_i; /* default 8   */
    int tile_j; /* default 32  */
    int tile_k; /* default 64  */
    int ck; /* default 32  */
    int waves_m; /* default 2   */
    int waves_n; /* default 2   */
    const char* atom; /* default "32x32x16" */
    int wave_size; /* default 64  */
    int lds_pad; /* default 8   */
    int lds_wpad; /* default 0   */
    int stage_group; /* default 4   */
    bool col_split; /* default false */
    int waves_per_eu; /* default 0   */
    bool pin_prefetch; /* default false */
} rocke_direct_nchw_fwd_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_direct_nchw_fwd_spec_t rocke_direct_nchw_fwd_spec_default(void);

/* kernel_name(): kernel_name_join(name, f"f{KH}x{KW}s{stride}p{PAD}",
 * f"t{tile_i}x{tile_j}x{tile_k}", f"ck{ck}", f"w{waves_m}x{waves_n}",
 * f"a{atom}", f"g{stage_group}", f"lp{lds_pad}", f"wp{lds_wpad}" (if nonzero),
 * "cs" (if col_split), f"wpe{waves_per_eu}" (if nonzero), "pin" (if
 * pin_prefetch), "bf16" (if bf16)). */
rocke_status_t rocke_direct_nchw_fwd_kernel_name(const rocke_direct_nchw_fwd_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap);

/* is_valid_direct_nchw_fwd_spec(spec, arch) -> (ok, reason). `arch` NULL =>
 * "gfx950". */
bool rocke_direct_nchw_fwd_is_valid_spec(const rocke_direct_nchw_fwd_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap);

/* build_direct_conv_nchw_fwd(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_direct_nchw_fwd(rocke_ir_builder_t* b,
                                                const rocke_direct_nchw_fwd_spec_t* spec,
                                                const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_direct_nchw_fwd_new(rocke_ir_builder_t* b,
                                                    const rocke_direct_nchw_fwd_spec_t* spec,
                                                    const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_direct_nchw_fwd_lower_to_llvm(const rocke_direct_nchw_fwd_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_NCHW_FWD_H */
