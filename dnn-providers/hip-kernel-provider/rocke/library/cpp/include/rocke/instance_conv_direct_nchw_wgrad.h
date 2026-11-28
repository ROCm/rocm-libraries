/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_nchw_wgrad.h -- C++ engine port of the
 * native-NCHW direct wgrad (backward-weight) convolution builder in
 * library/kernels/common/conv_direct_nchw_wgrad.py.
 *
 * A dense (groups == 1) backward-weight pass that consumes X in NCHW and dY
 * in NKHW and produces dW in KCYX directly (no host transpose of either
 * activation). A workgroup owns a tile_m x tile_c block of (k, c) for all the
 * filter taps of KH / y_groups filter rows and walks the spatial tiles of one
 * image (tile_h x tile_w output positions), staging the dY tile and the X
 * window in LDS once per spatial tile so every tap reads the same staged
 * window at a build-time offset. The partials meet in a caller-zeroed dW
 * through packed 16-bit atomics (or plain stores at split == 1 with
 * atomic=False).
 *
 *   Python (conv_direct_nchw_wgrad.py)        C++ (this header)
 *   ------------------------------------      ---------------------------------
 *   @dataclass(frozen) DirectNchwWgradSpec    rocke_direct_nchw_wgrad_spec_t
 *     .kernel_name()                          rocke_direct_nchw_wgrad_kernel_name
 *   is_valid_direct_nchw_wgrad_spec            rocke_direct_nchw_wgrad_is_valid_spec
 *   build_direct_conv_nchw_wgrad(spec, arch)   rocke_build_direct_nchw_wgrad(_new)
 *   (+ convenience: build -> lower .ll)        rocke_direct_nchw_wgrad_lower_to_llvm
 *
 * Kernargs are this family's own ABI (dY, X, dW pointers, their byte sizes and
 * split), not the shared direct-conv kernarg block emit_direct_params emits
 * for the fwd/dgrad native-NCHW families.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_NCHW_WGRAD_H
#define ROCKE_INSTANCE_CONV_DIRECT_NCHW_WGRAD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  DirectNchwWgradSpec
 *
 *  @dataclass(frozen=True)
 *  class DirectNchwWgradSpec:
 *      problem: DirectConvProblem
 *      name: str = "conv_direct_nchw_wgrad"
 *      tile_m: int = 64
 *      tile_c: int = 64
 *      tile_h: int = 4
 *      tile_w: int = 32
 *      waves_m: int = 2
 *      waves_c: int = 2
 *      waves_t: int = 1
 *      y_groups: int = 1
 *      stage_group: int = 4
 *      lds_pad: int = 8
 *      waves_per_eu: int = 0
 *      mfma_order: str = "row"      # "row" or "ks"
 *      prefetch: bool = True
 *      xcd_remap: bool = True
 *      epilogue: str = "lds"       # "lds" or "direct"
 *      kmap: str = "row"           # "row" or "2row"
 *      lds_wpad: int = 0
 *      lds_buffers: int = 1        # 1 or 2
 *      atomic: bool = True
 * ===================================================================== */
typedef struct rocke_direct_nchw_wgrad_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "conv_direct_nchw_wgrad" */
    int tile_m; /* default 64 */
    int tile_c; /* default 64 */
    int tile_h; /* default 4  */
    int tile_w; /* default 32 */
    int waves_m; /* default 2  */
    int waves_c; /* default 2  */
    int waves_t; /* default 1  */
    int y_groups; /* default 1  */
    int stage_group; /* default 4  */
    int lds_pad; /* default 8  */
    int waves_per_eu; /* default 0  */
    const char* mfma_order; /* default "row" */
    bool prefetch; /* default true */
    bool xcd_remap; /* default true */
    const char* epilogue; /* default "lds" */
    const char* kmap; /* default "row" */
    int lds_wpad; /* default 0 */
    int lds_buffers; /* default 1 */
    bool atomic; /* default true */
} rocke_direct_nchw_wgrad_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_direct_nchw_wgrad_spec_t rocke_direct_nchw_wgrad_spec_default(void);

/* kernel_name(): kernel_name_join(name, problem.short(), f"t{tile_m}x{tile_c}",
 * f"s{tile_h}x{tile_w}", f"w{waves_m}x{waves_c}x{waves_t}", f"g{stage_group}",
 * f"p{lds_pad}", f"yg{y_groups}" (if > 1), f"wpe{waves_per_eu}" (if set),
 * "oks" (if mfma_order == "ks"), "" / "nopf" (prefetch), "" / "noxcd"
 * (xcd_remap), "" / "edir" (epilogue), "k2r" (if kmap == "2row"),
 * f"wp{lds_wpad}" (if set), "db" (if lds_buffers == 2), "" / "st" (atomic),
 * "bf16" (if bf16)). */
rocke_status_t rocke_direct_nchw_wgrad_kernel_name(const rocke_direct_nchw_wgrad_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap);

/* is_valid_direct_nchw_wgrad_spec(spec, arch) -> (ok, reason). `arch` NULL =>
 * "gfx950". */
bool rocke_direct_nchw_wgrad_is_valid_spec(const rocke_direct_nchw_wgrad_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap);

/* build_direct_conv_nchw_wgrad(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_direct_nchw_wgrad(rocke_ir_builder_t* b,
                                                  const rocke_direct_nchw_wgrad_spec_t* spec,
                                                  const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_direct_nchw_wgrad_new(rocke_ir_builder_t* b,
                                                      const rocke_direct_nchw_wgrad_spec_t* spec,
                                                      const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_direct_nchw_wgrad_lower_to_llvm(const rocke_direct_nchw_wgrad_spec_t* spec,
                                                     const char* arch,
                                                     rocke_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_NCHW_WGRAD_H */
