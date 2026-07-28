/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_implicit_gemm_wgrad.h -- C99 port of the implicit-GEMM
 * backward-weight (wgrad) convolution kernel instance builder
 * rocke/instances/common/conv_implicit_gemm_wgrad.py (NHWK x NHWC -> KYXC).
 *
 * GEMM orientation (wgrad vs forward):
 *
 *   Forward (conv_implicit_gemm.py):
 *     M     = N*Ho*Wo    (output spatial positions)
 *     N_fwd = K          (output channels)
 *     K_fwd = Y*X*C      (filter x input channel)
 *     A: NHWC, B: KYXC, D: NHWK
 *
 *   Wgrad (this file):
 *     M     = K          (output channels -- weight rows)
 *     N_wg  = Y*X*C      (filter spatial x input channel -- weight cols)
 *     K_wg  = N*Ho*Wo    (output spatial positions -- reduction)
 *     A: dY (NHWK), B: X (NHWC), D: dW (KYXC)
 *
 * The C99 port mirrors the Python WgradConvSpec dataclass and the
 * build_implicit_gemm_conv_wgrad() builder.
 *
 *   Python (conv_implicit_gemm_wgrad.py)   C99 (this header)
 *   -----------------------------------    -----------------------------------------
 *   @dataclass WgradConvSpec               rocke_implicit_gemm_conv_wgrad_spec_t
 *   spec.* @property / methods             rocke_wgrad_conv_spec_*(...)
 *   is_valid_wgrad_spec(spec, arch)        rocke_implicit_gemm_conv_wgrad_is_valid_spec
 *   make_dy_descriptor(p)                  rocke_wgrad_make_dy_descriptor
 *   make_dw_descriptor(p)                  rocke_wgrad_make_dw_descriptor
 *   make_x_wgrad_descriptor(p)             rocke_wgrad_make_x_descriptor
 *   build_implicit_gemm_conv_wgrad(spec)   rocke_build_implicit_gemm_conv_wgrad
 *   (+ convenience: build -> lower .ll)    rocke_conv_implicit_gemm_wgrad_lower_to_llvm
 *
 * Split-K: when split_k > 1 the kernel partitions K_wg into `split_k` equal
 * slices along the Z grid axis and atomic-adds each CTA's partial f32
 * accumulator directly into dW.  Supported output dtypes: fp32 (scalar atomic),
 * bf16/fp16 (packed <2 x dtype> atomic, gfx940+).  split_k == -1 triggers
 * automatic selection via the CK formula.  split_k == 1 disables split-K (the
 * default: normal store).
 *
 * ConvProblem is reused verbatim from the already-ported value-type helper
 * (helper_rocke.instances.common.conv_implicit_gemm.h); this header includes it.
 */
#ifndef ROCKE_INSTANCE_CONV_IMPLICIT_GEMM_WGRAD_H
#define ROCKE_INSTANCE_CONV_IMPLICIT_GEMM_WGRAD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h" /* rocke_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 * WgradConvSpec   (Python lines 256-451)
 * ============================================================ *
 *
 * One concrete implicit-GEMM backward-weight convolution configuration.
 * Field order follows the Python dataclass declaration order.
 *
 * pipeline / epilogue are compared by strcmp:
 *   pipeline : "mem" | "compv3" | "compv4"
 *   epilogue : "default" | "cshuffle"
 *
 * split_k:
 *   -1 = auto (resolved at build time via CK formula)
 *    1 = disabled (default, normal store)
 *   >1 = fixed split-K degree
 *
 * Caller contract when split_k > 1:
 *   1. Zero-initialise the dW buffer before EVERY launch:
 *        hipMemset(dW_ptr, 0, dW_bytes)
 *      The kernel only issues atomic-adds, never a direct store, so any
 *      non-zero initial content accumulates into the result, producing
 *      silently wrong gradients with no runtime error.
 *   2. Launch with grid (ceil(wg_N/tile_n), ceil(wg_M/tile_m), split_k).
 *
 * When split_k == 1 the kernel writes dW normally (no atomics, no pre-zeroing
 * required).
 *
 * dtype_a / dtype_b / dtype_d: "fp16" | "bf16" | "fp32" (default all "fp16").
 */
typedef struct rocke_implicit_gemm_conv_wgrad_spec
{
    rocke_conv_problem_t problem;
    const char* name; /* default "conv_igemm_wgrad" */

    /* dtype fields (ConvDataSpec) */
    const char* dtype_a; /* default "fp16" */
    const char* dtype_b; /* default "fp16" */
    const char* dtype_d; /* default "fp16" */
    const char* dtype_acc; /* default "fp32" */

    int tile_m; /* default 64 */
    int tile_n; /* default 64 */
    int tile_k; /* default 64 */

    int warp_m; /* default 2 */
    int warp_n; /* default 2 */

    int warp_tile_m; /* default 32 */
    int warp_tile_n; /* default 32 */
    int warp_tile_k; /* default 16 */

    int wave_size; /* default 64 */

    const char* pipeline; /* default "mem"     */
    const char* epilogue; /* default "default" */
    bool async_dma; /* default false */
    bool unroll_k; /* default false */

    bool has_lds_k_pad; /* false => Python None */
    int lds_k_pad;
    void* lds_layout; /* NULL => Python None */

    bool chiplet_swizzle; /* default false */
    int chiplet_wgm; /* default 8  */
    int chiplet_num_xcds; /* default 8  */
    int chiplet_chunk_size; /* default 64 */

    bool has_waves_per_eu; /* false => Python None */
    int waves_per_eu;

    bool has_vector_size_a;
    int vector_size_a;
    bool has_vector_size_b;
    int vector_size_b;
    bool has_vector_size_c;
    int vector_size_c;

    /* ConvAccumulatorEpilogue (bias/scale/relu/clamp) is omitted in the
     * initial port; the default identity epilogue is always used.  Add when
     * needed. */

    /* split_k: -1 = auto, 1 = off, >1 = fixed degree. */
    int split_k; /* default 1 */
} rocke_implicit_gemm_conv_wgrad_spec_t;

/* Default-constructed spec (every field == Python dataclass default). */
rocke_implicit_gemm_conv_wgrad_spec_t rocke_implicit_gemm_conv_wgrad_spec_default(void);

/* ---- WgradConvSpec @property analogues (pure int arithmetic) ---- */

/* spec.block_size: warp_m * warp_n * wave_size. */
int rocke_wgrad_conv_spec_block_size(const rocke_implicit_gemm_conv_wgrad_spec_t* s);

/* spec.k_atoms_per_tile_k: tile_k / warp_tile_k. */
int rocke_wgrad_conv_spec_k_atoms_per_tile_k(const rocke_implicit_gemm_conv_wgrad_spec_t* s);

/* spec.mfmas_per_warp_m: tile_m / (warp_m * warp_tile_m). */
int rocke_wgrad_conv_spec_mfmas_per_warp_m(const rocke_implicit_gemm_conv_wgrad_spec_t* s);

/* spec.mfmas_per_warp_n: tile_n / (warp_n * warp_tile_n). */
int rocke_wgrad_conv_spec_mfmas_per_warp_n(const rocke_implicit_gemm_conv_wgrad_spec_t* s);

/* spec.wg_M: output channels per group (p.K / p.groups). */
int rocke_wgrad_conv_spec_wg_M(const rocke_implicit_gemm_conv_wgrad_spec_t* s);

/* spec.wg_N: filter spatial x input channels per group (Z * Y * X * C/groups). */
int rocke_wgrad_conv_spec_wg_N(const rocke_implicit_gemm_conv_wgrad_spec_t* s);

/* spec.wg_K: output spatial positions (N * Ho * Wo [* Do]). */
int rocke_wgrad_conv_spec_wg_K(const rocke_implicit_gemm_conv_wgrad_spec_t* s);

/* spec.wg_K_padded(): wg_K rounded up to tile_k * split_k. */
int rocke_wgrad_conv_spec_wg_K_padded(const rocke_implicit_gemm_conv_wgrad_spec_t* s);

/* spec.kernel_name() -> NUL-terminated into out (capacity out_cap). */
rocke_status_t rocke_wgrad_conv_spec_kernel_name(const rocke_implicit_gemm_conv_wgrad_spec_t* s,
                                                 char* out,
                                                 size_t out_cap);

/* is_valid_wgrad_spec(spec, arch) -> (ok, reason).
 * arch NULL => "gfx950".  Returns false + reason string on reject. */
bool rocke_implicit_gemm_conv_wgrad_is_valid_spec(const rocke_implicit_gemm_conv_wgrad_spec_t* s,
                                                  const char* arch,
                                                  char* reason,
                                                  size_t reason_cap);

/* ============================================================ *
 * Descriptor builders   (Python lines 147-248)
 * ============================================================ *
 *
 *   make_dy_descriptor(p):       (k_wg, k_out=m_wg) -> NHWK offset.
 *   make_x_wgrad_descriptor(p):  (k_wg, n_wg) -> NHWC offset (== make_a_descriptor).
 *   make_dw_descriptor(p):       (m_wg, n_wg) -> KYXC offset.
 */
struct rocke_tensor_descriptor; /* fwd (full decl in helper transforms header) */

struct rocke_tensor_descriptor* rocke_wgrad_make_dy_descriptor(rocke_ir_builder_t* b,
                                                               const rocke_conv_problem_t* p,
                                                               const char* dtype);

struct rocke_tensor_descriptor* rocke_wgrad_make_x_descriptor(rocke_ir_builder_t* b,
                                                              const rocke_conv_problem_t* p,
                                                              const char* dtype);

struct rocke_tensor_descriptor* rocke_wgrad_make_dw_descriptor(rocke_ir_builder_t* b,
                                                               const rocke_conv_problem_t* p,
                                                               const char* dtype);

/* ============================================================ *
 * build_implicit_gemm_conv_wgrad
 * ============================================================ *
 *
 * Builds the IR for one implicit-GEMM backward-weight conv kernel.
 *
 * Convenience: rocke_build_implicit_gemm_conv_wgrad_new inits `b` from
 * spec.kernel_name() then builds. The caller owns `b` and frees it with
 * rocke_ir_builder_free().
 *
 * rocke_conv_implicit_gemm_wgrad_lower_to_llvm builds a stock-body kernel and
 * lowers it to .ll text in one shot; internally owns and frees its IRBuilder.
 */
rocke_kernel_def_t* rocke_build_implicit_gemm_conv_wgrad(
    rocke_ir_builder_t* b, const rocke_implicit_gemm_conv_wgrad_spec_t* spec, const char* arch);

rocke_kernel_def_t* rocke_build_implicit_gemm_conv_wgrad_new(
    rocke_ir_builder_t* b, const rocke_implicit_gemm_conv_wgrad_spec_t* spec, const char* arch);

rocke_status_t
    rocke_conv_implicit_gemm_wgrad_lower_to_llvm(const rocke_implicit_gemm_conv_wgrad_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_IMPLICIT_GEMM_WGRAD_H */
