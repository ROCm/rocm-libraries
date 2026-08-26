/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_winograd.h -- C API for the Winograd convolution
 * kernel family (stride=1, dilation=1, 3×3 filter, NHWC layout).
 *
 * Three transform kernels are provided; the GEMM step is handled externally
 * (e.g. via the existing implicit-GEMM infrastructure):
 *
 *   1. Data transform    — B^T × input_patch × B,  per (n, tile_h, tile_w, c)
 *   2. Filter transform  — G × filter × G^T,        per (k, c)  [done once]
 *   3. Output transform  — A^T × gemm_result × A,  per (n, tile_h, tile_w, k)
 *
 * C ABI mirror of:
 *   platform/python/rocke/instances/common/conv_winograd.py
 *   platform/python/rocke/instances/common/_conv_winograd_common.py
 *
 * Byte-identity contract: every build function here must emit the same
 * LLVM IR as its Python counterpart for the same (spec, arch) inputs.
 * Run `python tools/check_byte_identity.py --only winograd` to verify.
 */
#ifndef ROCKE_INSTANCE_CONV_WINOGRAD_H
#define ROCKE_INSTANCE_CONV_WINOGRAD_H

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ problem */

/* 2-D Winograd convolution shape.
 *
 * Restrictions (same as Python WinogradProblem):
 *   - filter always 3×3, stride=1, dilation=1
 *   - NHWC input, KYXC filter, NHWK output
 *   - pH, pW in {0, 1}
 *
 * Derived:
 *   Ho = Hi + 2*pH - 2
 *   Wo = Wi + 2*pW - 2
 */
typedef struct rocke_winograd_problem
{
    int N;
    int Hi;
    int Wi;
    int C;
    int K;
    int pH; /* vertical padding   (0 or 1) */
    int pW; /* horizontal padding (0 or 1) */
} rocke_winograd_problem_t;

static inline rocke_winograd_problem_t
    rocke_winograd_problem_make(int N, int Hi, int Wi, int C, int K, int pH, int pW)
{
    rocke_winograd_problem_t p;
    p.N = N;
    p.Hi = Hi;
    p.Wi = Wi;
    p.C = C;
    p.K = K;
    p.pH = pH;
    p.pW = pW;
    return p;
}

/* Default: N=8 Hi=56 Wi=56 C=64 K=64 pH=1 pW=1 */
static inline rocke_winograd_problem_t
    rocke_winograd_problem_default(int N, int Hi, int Wi, int C, int K)
{
    return rocke_winograd_problem_make(N, Hi, Wi, C, K, 1, 1);
}

static inline int rocke_winograd_problem_Ho(const rocke_winograd_problem_t* p)
{
    return p->Hi + 2 * p->pH - 2;
}
static inline int rocke_winograd_problem_Wo(const rocke_winograd_problem_t* p)
{
    return p->Wi + 2 * p->pW - 2;
}

/* -------------------------------------------------------------------- spec */

/* Configuration for the three Winograd transform kernels.
 *
 * out_tile: 2 → F(2,3) — 4×4 transform domain
 *           4 → F(4,3) — 6×6 transform domain (better FLOP reduction)
 *
 * block_c   — input channels per thread block
 * block_k   — output channels per thread block
 * block_nhw — (n, tile_h, tile_w) triples per thread block
 */
typedef struct rocke_winograd_conv_spec
{
    rocke_winograd_problem_t problem;
    const char* name; /* kernel family name prefix */
    int out_tile; /* 2 or 4 */
    int block_c;
    int block_k;
    int block_nhw;
} rocke_winograd_conv_spec_t;

static inline rocke_winograd_conv_spec_t rocke_winograd_conv_spec_default(void)
{
    rocke_winograd_conv_spec_t s;
    s.problem = rocke_winograd_problem_default(8, 56, 56, 64, 64);
    s.name = "conv_winograd";
    s.out_tile = 4;
    s.block_c = 32;
    s.block_k = 32;
    s.block_nhw = 4;
    return s;
}

/* Derived geometry helpers */
static inline int rocke_winograd_spec_xform_size(const rocke_winograd_conv_spec_t* s)
{
    /* xform_size = out_tile + filter_size - 1 = out_tile + 2 */
    return s->out_tile + 2;
}
static inline int rocke_winograd_spec_tiles_h(const rocke_winograd_conv_spec_t* s)
{
    int Ho = rocke_winograd_problem_Ho(&s->problem);
    return (Ho + s->out_tile - 1) / s->out_tile;
}
static inline int rocke_winograd_spec_tiles_w(const rocke_winograd_conv_spec_t* s)
{
    int Wo = rocke_winograd_problem_Wo(&s->problem);
    return (Wo + s->out_tile - 1) / s->out_tile;
}
static inline int rocke_winograd_spec_num_tiles(const rocke_winograd_conv_spec_t* s)
{
    return rocke_winograd_spec_tiles_h(s) * rocke_winograd_spec_tiles_w(s);
}

/* Fill out (at most cap-1 chars + NUL) the kernel name for the given suffix.
 * Matches Python WinogradConvSpec.kernel_name(suffix). */
void rocke_winograd_conv_spec_kernel_name(const rocke_winograd_conv_spec_t* s,
                                          const char* suffix,
                                          char* out,
                                          int cap);

/* ---------------------------------------------------- build entry points */

/* Each _new function:
 *   1. Calls rocke_ir_builder_init(b, kernel_name) with the derived name.
 *   2. Emits all params and SSA ops that mirror the Python builder.
 *   3. Returns b->kernel on success, NULL on error (check rocke_ir_builder_error(b)).
 *
 * The caller owns the builder lifetime; call rocke_ir_builder_free(b) when done.
 */

rocke_kernel_def_t* rocke_build_winograd_data_transform_new(rocke_ir_builder_t* b,
                                                            const rocke_winograd_conv_spec_t* s,
                                                            const char* arch);

rocke_kernel_def_t* rocke_build_winograd_filter_transform_new(rocke_ir_builder_t* b,
                                                              const rocke_winograd_conv_spec_t* s,
                                                              const char* arch);

rocke_kernel_def_t* rocke_build_winograd_output_transform_new(rocke_ir_builder_t* b,
                                                              const rocke_winograd_conv_spec_t* s,
                                                              const char* arch);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_WINOGRAD_H */
