/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_implicit_gemm_dgrad.h -- C99 port of the implicit-GEMM
 * backward-data (dgrad) convolution kernel instance builder
 * rocke/instances/common/conv_implicit_gemm_dgrad.py (dY x W -> dX).
 *
 * GEMM orientation (dgrad):
 *
 *   M     = N*Hi*Wi       (input spatial positions)
 *   N_dg  = C             (input channels)
 *   K_dg  = Y*X*K         (filter spatial x output channels -- reduction)
 *   A: dY (NHWK), B: W (KYXC), D: dX (NHWC)
 *
 * For stride=1 a single GEMM is used (compile-time descriptor transforms).
 * For stride>1 a tilde decomposition splits the problem into up to 64+
 * sub-GEMMs dispatched inside a single kernel via a parameter buffer +
 * binary search.
 *
 *   Python (conv_implicit_gemm_dgrad.py)    C99 (this header)
 *   -----------------------------------    ----------------------------------------
 *   @dataclass DgradConvSpec               rocke_dgrad_conv_spec_t
 *   spec.* @property / methods             rocke_dgrad_conv_spec_*(...)
 *   is_valid_dgrad_spec(spec, arch)        rocke_dgrad_conv_is_valid_spec(...)
 *   make_dgrad_dy_descriptor(p)            rocke_dgrad_make_dy_descriptor(...)
 *   make_dgrad_w_descriptor(p)             rocke_dgrad_make_w_descriptor(...)
 *   make_dgrad_dx_descriptor(p)            rocke_dgrad_make_dx_descriptor(...)
 *   TildeDecomposition                     rocke_tilde_decomposition_t
 *   SubGemmParams                          rocke_sub_gemm_params_t
 *   compute_tilde(p)                       rocke_compute_tilde(...)
 *   enumerate_sub_gemms(...)               rocke_enumerate_sub_gemms(...)
 *   pack_sub_gemm_buffer(...)              rocke_pack_sub_gemm_buffer(...)
 *   build_implicit_gemm_conv_dgrad(spec)   rocke_build_implicit_gemm_conv_dgrad(...)
 *   (+ convenience: build -> lower .ll)    rocke_dgrad_conv_implicit_gemm_lower_to_llvm
 *
 * ConvProblem is reused verbatim from the already-ported value-type helper
 * (helper_rocke.instances.common.conv_implicit_gemm.h); this header includes it.
 */
#ifndef ROCKE_INSTANCE_CONV_IMPLICIT_GEMM_DGRAD_H
#define ROCKE_INSTANCE_CONV_IMPLICIT_GEMM_DGRAD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h" /* rocke_conv_problem_t */
#include "rocke/instance_conv_implicit_gemm.h" /* rocke_conv_acc_epilogue_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 * DgradConvSpec   (Python lines 462-675)
 * ============================================================ *
 *
 * One concrete implicit-GEMM backward-data convolution configuration.
 * Field order follows the Python dataclass declaration order.
 *
 * pipeline / epilogue are compared by strcmp:
 *   pipeline : "mem" | "compv3" | "compv4" | "wavelet"
 *   epilogue : "default" | "cshuffle"
 *
 * split_k:
 *   -1 = auto (resolved at build time via CK formula)
 *    1 = disabled (default, normal store)
 *   >1 = fixed split-K degree
 *
 * dtype_a / dtype_b / dtype_d: "fp16" | "bf16" | "fp32" (default all "fp16").
 */
typedef struct rocke_dgrad_conv_spec
{
    rocke_conv_problem_t problem;
    const char* name; /* default "conv_igemm_dgrad" */

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

    rocke_conv_acc_epilogue_t acc_epilogue; /* default identity */

    /* split_k: -1 = auto, 1 = off, >1 = fixed degree. */
    int split_k; /* default 1 */

    /* Wavelet pipeline (pipeline="wavelet", WMMA/gfx1250 only).
     * num_load_waves: extra load waves appended after the math waves (default 4).
     * launch_block_size = block_size + num_load_waves * wave_size. */
    int num_load_waves; /* default 4 */
} rocke_dgrad_conv_spec_t;

/* Default-constructed spec (every field == Python dataclass default). */
rocke_dgrad_conv_spec_t rocke_dgrad_conv_spec_default(void);

/* ---- DgradConvSpec @property analogues (pure int arithmetic) ---- */

/* spec.block_size: warp_m * warp_n * wave_size. */
int rocke_dgrad_conv_spec_block_size(const rocke_dgrad_conv_spec_t* s);

/* spec.launch_block_size: block_size for non-wavelet; block_size + num_load_waves * wave_size
 * for pipeline="wavelet". */
int rocke_dgrad_conv_spec_launch_block_size(const rocke_dgrad_conv_spec_t* s);

/* spec.k_atoms_per_tile_k: tile_k / warp_tile_k. */
int rocke_dgrad_conv_spec_k_atoms_per_tile_k(const rocke_dgrad_conv_spec_t* s);

/* spec.mfmas_per_warp_m: tile_m / (warp_m * warp_tile_m). */
int rocke_dgrad_conv_spec_mfmas_per_warp_m(const rocke_dgrad_conv_spec_t* s);

/* spec.mfmas_per_warp_n: tile_n / (warp_n * warp_tile_n). */
int rocke_dgrad_conv_spec_mfmas_per_warp_n(const rocke_dgrad_conv_spec_t* s);

/* spec.dg_M: N * Hi * Wi (input spatial positions). */
int rocke_dgrad_conv_spec_dg_M(const rocke_dgrad_conv_spec_t* s);

/* spec.dg_N: C / groups (input channels per group). */
int rocke_dgrad_conv_spec_dg_N(const rocke_dgrad_conv_spec_t* s);

/* spec.dg_K: Y * X * (K / groups) (filter x output channels per group). */
int rocke_dgrad_conv_spec_dg_K(const rocke_dgrad_conv_spec_t* s);

/* spec.dg_K_padded(): K_dg rounded up to next multiple of tile_k * split_k. */
int rocke_dgrad_conv_spec_dg_K_padded(const rocke_dgrad_conv_spec_t* s);

/* spec.is_strided: true when sH != 1 or sW != 1 or dH != 1 or dW != 1. */
bool rocke_dgrad_conv_spec_is_strided(const rocke_dgrad_conv_spec_t* s);

/* spec.needs_atomic: true when split_k > 1 (atomic accumulation across CTAs).
 * Tilde sub-GEMMs with split_k=1 use direct buffer_store (disjoint writes). */
bool rocke_dgrad_conv_spec_needs_atomic(const rocke_dgrad_conv_spec_t* s);

/* spec.kernel_name() -> NUL-terminated into out (capacity out_cap). */
rocke_status_t
    rocke_dgrad_conv_spec_kernel_name(const rocke_dgrad_conv_spec_t* s, char* out, size_t out_cap);

/* is_valid_dgrad_spec(spec, arch) -> (ok, reason).
 * arch NULL => "gfx950".  Returns false + reason string on reject. */
bool rocke_dgrad_conv_is_valid_spec(const rocke_dgrad_conv_spec_t* s,
                                    const char* arch,
                                    char* reason,
                                    size_t reason_cap);

/* ============================================================ *
 * Tilde decomposition   (Python lines 128-330)
 * ============================================================ *
 *
 * For stride > 1 the backward-data convolution decomposes into
 * y_tilde * x_tilde independent sub-GEMMs. These types and helpers
 * compute the decomposition on the host side; the kernel receives
 * the packed parameter buffer at launch time.
 */

typedef struct rocke_tilde_decomposition
{
    int gcd_h;
    int gcd_w;
    int y_tilde;
    int x_tilde;
    int y_dot;
    int x_dot;
    int h_tilde;
    int w_tilde;
} rocke_tilde_decomposition_t;

/* Number of i32 fields packed per sub-GEMM record in the parameter buffer. */
#define ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS 22

typedef struct rocke_sub_gemm_params
{
    int i_ytilde;
    int i_xtilde;
    int y_dot_slice;
    int x_dot_slice;
    int h_tilde_slice_begin;
    int h_tilde_slice;
    int w_tilde_slice_begin;
    int w_tilde_slice;
    int gemm_m; /* N * HTildeSlice * WTildeSlice */
    int gemm_n; /* C per group */
    int gemm_k; /* YDotSlice * XDotSlice * K per group */
    int block_start; /* cumulative tile offset */
    int block_end;

    /* Pre-computed offset coefficients for the kernel. */
    int a_embed_h_coeff;
    int a_embed_w_coeff;
    int b_y_stride;
    int b_y_offset;
    int b_x_stride;
    int b_x_offset;
    int d_h_stride;
    int d_h_offset;
    int d_w_stride;
    int d_w_offset;
    int gemm_k_padded;
} rocke_sub_gemm_params_t;

/* Compute tilde decomposition parameters from a ConvProblem. */
rocke_tilde_decomposition_t rocke_compute_tilde(const rocke_conv_problem_t* p);

/* Enumerate all non-empty sub-GEMMs. Writes up to out_cap entries into out[].
 * Returns the number written (may be less than y_tilde * x_tilde if some
 * phases are empty). */
int rocke_enumerate_sub_gemms(const rocke_conv_problem_t* p,
                              const rocke_tilde_decomposition_t* tilde,
                              int tile_m,
                              int tile_n,
                              int tile_k,
                              int split_k,
                              rocke_sub_gemm_params_t* out,
                              int out_cap);

/* Pack sub-GEMM records into a flat i32 buffer for the kernel.
 * Writes ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS * count i32s into out_buf.
 * Returns the total number of i32s written. */
int rocke_pack_sub_gemm_buffer(const rocke_sub_gemm_params_t* sgs,
                               int count,
                               int tile_m,
                               int tile_n,
                               int* out_buf,
                               int out_cap);

/* ============================================================ *
 * Descriptor builders   (Python lines 332-450)
 * ============================================================ */
struct rocke_tensor_descriptor; /* fwd (full decl in helper transforms header) */

struct rocke_tensor_descriptor* rocke_dgrad_make_dy_descriptor(rocke_ir_builder_t* b,
                                                               const rocke_conv_problem_t* p,
                                                               const char* dtype);

struct rocke_tensor_descriptor* rocke_dgrad_make_w_descriptor(rocke_ir_builder_t* b,
                                                              const rocke_conv_problem_t* p,
                                                              const char* dtype);

struct rocke_tensor_descriptor* rocke_dgrad_make_dx_descriptor(rocke_ir_builder_t* b,
                                                               const rocke_conv_problem_t* p,
                                                               const char* dtype);

/* ============================================================ *
 * build_implicit_gemm_conv_dgrad   (Python lines 813-1295)
 * ============================================================ *
 *
 * Builds the IR for one implicit-GEMM backward-data conv kernel.
 * For stride=1 uses compile-time descriptors. For stride>1 emits a
 * single kernel with runtime sub-GEMM dispatch (binary search over
 * a parameter buffer).
 *
 * arch NULL => "gfx950". Returns the kernel or NULL with b's sticky
 * error set.
 */
rocke_kernel_def_t* rocke_build_implicit_gemm_conv_dgrad(rocke_ir_builder_t* b,
                                                         const rocke_dgrad_conv_spec_t* spec,
                                                         const char* arch);

/* Convenience: init `b` from spec.kernel_name() then build. */
rocke_kernel_def_t* rocke_build_implicit_gemm_conv_dgrad_new(rocke_ir_builder_t* b,
                                                             const rocke_dgrad_conv_spec_t* spec,
                                                             const char* arch);

/* Convenience: build + lower to LLVM .ll text in one shot. */
rocke_status_t rocke_dgrad_conv_implicit_gemm_lower_to_llvm(const rocke_dgrad_conv_spec_t* spec,
                                                            const char* arch,
                                                            rocke_llvm_flavor_t flavor,
                                                            char** out_ll,
                                                            char* err,
                                                            size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_IMPLICIT_GEMM_DGRAD_H */
