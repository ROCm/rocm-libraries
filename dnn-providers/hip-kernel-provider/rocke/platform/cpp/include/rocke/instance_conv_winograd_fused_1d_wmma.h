/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_winograd_fused_1d_wmma.h -- C++ engine port of the fused
 * 1-D Winograd forward builder in
 * library/kernels/common/conv_winograd_fused_1d_wmma.py (gfx1250, wave32 WMMA).
 *
 * One kernel computes a 1 x r conv (Y = 1, H = 1, stride 1) as F(m, r) or
 * segmented S x F(m, m) Winograd from raw x to y: the transformed weights of a
 * block of Kb output channels stay in VGPRs, the workgroup loops persistently
 * over blocks of 16 q tiles, stages the raw input in LDS, transforms it (VALU
 * or WMMA), runs the per-bin WMMA GEMMs and applies the output transform.
 *
 *   Python (conv_winograd_fused_1d_wmma.py)  C (this header)
 *   ---------------------------------------  ---------------------------------------
 *   @dataclass(frozen) FusedWino1DSpec       rocke_winograd_fused_1d_spec_t
 *     (ax: WinoAxis(m, r, seg); default      ax_m / ax_r / ax_seg (default points
 *      points only)                           only; there is no pts field)
 *     .reason(arch="gfx1250")                rocke_winograd_fused_1d_is_valid_spec
 *     .grid() / .lds_bytes()                 rocke_winograd_fused_1d_{grid,lds_bytes}
 *     .kernel_name()                         rocke_winograd_fused_1d_kernel_name
 *   build_fused_wino_1d(spec, arch)          rocke_build_winograd_fused_1d(_new)
 *   (+ convenience: build -> lower .ll)      rocke_winograd_fused_1d_lower_to_llvm
 *
 * Kernargs (fused_signature): x, u, y (fp16 pointers), x_bytes, u_bytes,
 * y_bytes (i32). The problem extents and every knob are baked into the IR.
 */
#ifndef ROCKE_INSTANCE_CONV_WINOGRAD_FUSED_1D_WMMA_H
#define ROCKE_INSTANCE_CONV_WINOGRAD_FUSED_1D_WMMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h" /* rocke_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  FusedWino1DSpec
 *
 *  @dataclass(frozen=True)
 *  class FusedWino1DSpec:
 *      problem: ConvProblem
 *      ax: WinoAxis           # W axis: F(m, r) or S x F(m, m)
 *      q: int = 4             # 16-tile fragments per tile block
 *      fk: int = 2            # 16-channel K fragments per block
 *      grid_blocks: int = 256
 *      prefetch: bool = False
 *      m_f16: bool = False
 *      xf: str = "valu"       # "valu" | "wmma"
 *      xf_batch: int = 4
 *      merge: bool = False
 *      waves: int = 8
 *      waves_per_eu: int = 0
 *      name: str = "conv_wino_fused1d"
 * ===================================================================== */
typedef struct rocke_winograd_fused_1d_spec
{
    rocke_conv_problem_t problem;
    int ax_m; /* WinoAxis.m (outputs per tile)        */
    int ax_r; /* WinoAxis.r (filter taps, = X)        */
    int ax_seg; /* WinoAxis.seg (segments), default 1  */
    int q; /* default 4   */
    int fk; /* default 2   */
    int grid_blocks; /* default 256 */
    bool prefetch; /* default false */
    bool m_f16; /* default false */
    const char* xf; /* default "valu" */
    int xf_batch; /* default 4   */
    bool merge; /* default false */
    int waves; /* default 8   */
    int waves_per_eu; /* default 0   */
    const char* name; /* default "conv_wino_fused1d" */
} rocke_winograd_fused_1d_spec_t;

/* Spec with the Python dataclass defaults (problem zeroed, axis F(4, 3)). */
rocke_winograd_fused_1d_spec_t rocke_winograd_fused_1d_spec_default(void);

/* spec.grid(): persistent workgroups (a multiple of the K-block count). */
int rocke_winograd_fused_1d_grid(const rocke_winograd_fused_1d_spec_t* spec);
/* spec.lds_bytes(). */
long rocke_winograd_fused_1d_lds_bytes(const rocke_winograd_fused_1d_spec_t* spec);

/* kernel_name(): "<name>_<problem.short()>_<ax.label()>_q{q}_fk{fk}_g{grid}"
 * plus the knob tags of FusedWino1DSpec.short(). */
rocke_status_t rocke_winograd_fused_1d_kernel_name(const rocke_winograd_fused_1d_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap);

/* FusedWino1DSpec.reason() (checked for gfx1250, as the Python __post_init__
 * does). On failure `reason` holds the Python message. */
bool rocke_winograd_fused_1d_is_valid_spec(const rocke_winograd_fused_1d_spec_t* spec,
                                           char* reason,
                                           size_t reason_cap);

/* build_fused_wino_1d(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_winograd_fused_1d(rocke_ir_builder_t* b,
                                                  const rocke_winograd_fused_1d_spec_t* spec,
                                                  const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_winograd_fused_1d_new(rocke_ir_builder_t* b,
                                                      const rocke_winograd_fused_1d_spec_t* spec,
                                                      const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_winograd_fused_1d_lower_to_llvm(const rocke_winograd_fused_1d_spec_t* spec,
                                                     const char* arch,
                                                     rocke_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_WINOGRAD_FUSED_1D_WMMA_H */
