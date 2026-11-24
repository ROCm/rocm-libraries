/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_nchw_dgrad.h -- C++ engine port of the
 * native-NCHW direct dgrad convolution builder in
 * library/kernels/common/conv_direct_nchw_dgrad.py.
 *
 * A dense (groups == 1) backward-data pass that consumes dY in NKHW and
 * produces dX in NCHW directly (no host transpose of the activations),
 * via the stride-phase direct dgrad algorithm: the dX grid is split into its
 * S x S stride phases, each phase a small dense stride-1 correlation of dY.
 * One workgroup owns a tile_i x tile_j block of the phase grid (all S*S
 * phases of it) and tile_c input channels; the dY window every phase reads
 * is staged once in LDS as [row][col][k] (k contiguous, padded by lds_pad).
 * The MFMA operands are swapped (the dY fragment is the A operand); the
 * epilogue interleaves the pw=0/pw=1 accumulators into one dwordx4 store at
 * stride 2 so the stride-2 scatter along w never reaches memory.
 *
 *   Python (conv_direct_nchw_dgrad.py)       C++ (this header)
 *   ------------------------------------     ---------------------------------
 *   @dataclass(frozen) DirectNchwDgradSpec   rocke_direct_nchw_dgrad_spec_t
 *     .kernel_name()                         rocke_direct_nchw_dgrad_kernel_name
 *   is_valid_direct_nchw_dgrad_spec           rocke_direct_nchw_dgrad_is_valid_spec
 *   build_direct_conv_nchw_dgrad(spec, arch)  rocke_build_direct_nchw_dgrad(_new)
 *   (+ convenience: build -> lower .ll)       rocke_direct_nchw_dgrad_lower_to_llvm
 *
 * AOT: the kernel declares the forward direct-conv kernarg block
 * (rocke_dconv_emit_params, direction "fwd" -- the Python builder calls
 * emit_direct_params with its "fwd" default too; A/D carry dY/dX, not X/Y).
 * Batch, extents and both channel counts are runtime; the filter, stride,
 * PAD and every knob are baked.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_NCHW_DGRAD_H
#define ROCKE_INSTANCE_CONV_DIRECT_NCHW_DGRAD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  DirectNchwDgradSpec
 *
 *  @dataclass(frozen=True)
 *  class DirectNchwDgradSpec:
 *      problem: DirectConvProblem   # the *forward* problem: H/W are dX's
 *                                   # extents, Ho/Wo are dY's; cpg = C, kpg = K
 *      name: str = "conv_direct_nchw_dgrad"
 *      tile_i: int = 8
 *      tile_j: int = 32
 *      tile_c: int = 64
 *      ck: int = 32
 *      waves_m: int = 2
 *      waves_n: int = 2
 *      atom: str = "32x32x16"
 *      wave_size: int = 64
 *      lds_pad: int = 8
 *      stage_group: int = 8
 * ===================================================================== */
typedef struct rocke_direct_nchw_dgrad_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "conv_direct_nchw_dgrad" */
    int tile_i; /* default 8   */
    int tile_j; /* default 32  */
    int tile_c; /* default 64  */
    int ck; /* default 32  */
    int waves_m; /* default 2   */
    int waves_n; /* default 2   */
    const char* atom; /* default "32x32x16" */
    int wave_size; /* default 64  */
    int lds_pad; /* default 8   */
    int stage_group; /* default 8   */
} rocke_direct_nchw_dgrad_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_direct_nchw_dgrad_spec_t rocke_direct_nchw_dgrad_spec_default(void);

/* kernel_name(): kernel_name_join(name, f"f{KH}x{KW}s{stride}p{PAD}",
 * f"t{tile_i}x{tile_j}x{tile_c}", f"ck{ck}", f"w{waves_m}x{waves_n}",
 * f"a{atom}", f"g{stage_group}", f"lp{lds_pad}", "bf16" (if bf16)). */
rocke_status_t rocke_direct_nchw_dgrad_kernel_name(const rocke_direct_nchw_dgrad_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap);

/* is_valid_direct_nchw_dgrad_spec(spec, arch) -> (ok, reason). `arch` NULL =>
 * "gfx950". */
bool rocke_direct_nchw_dgrad_is_valid_spec(const rocke_direct_nchw_dgrad_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap);

/* build_direct_conv_nchw_dgrad(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_direct_nchw_dgrad(rocke_ir_builder_t* b,
                                                  const rocke_direct_nchw_dgrad_spec_t* spec,
                                                  const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_direct_nchw_dgrad_new(rocke_ir_builder_t* b,
                                                      const rocke_direct_nchw_dgrad_spec_t* spec,
                                                      const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_direct_nchw_dgrad_lower_to_llvm(const rocke_direct_nchw_dgrad_spec_t* spec,
                                                     const char* arch,
                                                     rocke_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_NCHW_DGRAD_H */
