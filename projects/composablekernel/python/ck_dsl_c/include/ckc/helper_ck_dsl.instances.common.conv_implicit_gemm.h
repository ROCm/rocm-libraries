/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_ck_dsl.instances.common.conv_implicit_gemm.h -- C99 port of the
 * ConvProblem dataclass from
 * ck_dsl/instances/common/conv_implicit_gemm.py (implicit-GEMM convolution,
 * NHWC x KRSC -> NHWK).
 *
 *   Python (conv_implicit_gemm.py)        C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   @dataclass(frozen=True) ConvProblem   ckc_conv_problem_t
 *   ConvProblem(N, Hi, Wi, C, K, R, S,    ckc_conv_problem_make(...) /
 *               sH, sW, pH, pW, dH, dW)     ckc_conv_problem_default()
 *   .Ho            (property)             ckc_conv_problem_ho(...)
 *   .Wo            (property)             ckc_conv_problem_wo(...)
 *   .M             (property)             ckc_conv_problem_m(...)
 *   .N_gemm        (property)             ckc_conv_problem_n_gemm(...)
 *   .K_gemm        (property)             ckc_conv_problem_k_gemm(...)
 *   .flops         (property)             ckc_conv_problem_flops(...)
 *   .short()                              ckc_conv_problem_short(...)
 *
 * ConvProblem is a pure value type: its fields are the convolution shape
 * parameters and its members are integer-arithmetic derived quantities plus a
 * naming string. NONE of them touch the IR builder, so these are bit-for-bit
 * value producers whose results are later baked into the descriptor DAG (e.g.
 * via const_i32). A byte-identical IR sequence therefore follows from
 * byte-identical return values here.
 *
 * Only ConvProblem is ported in this file. The surrounding spec/dataclasses
 * (ConvAccumulatorEpilogue, ImplicitGemmConvSpec) and the builder entry points
 * (build_implicit_gemm_conv, descriptor builders, epilogues) are NOT ported
 * here; they are peers.
 *
 * Error model mirrors the rest of the C port: the short()-style string method
 * uses an out-buffer + ckc_status_t return (CKC_ERR_VALUE when the buffer is
 * too small / args are NULL), like the other spec helpers.
 */
#ifndef CKC_HELPER_CK_DSL_INSTANCES_COMMON_CONV_IMPLICIT_GEMM_H
#define CKC_HELPER_CK_DSL_INSTANCES_COMMON_CONV_IMPLICIT_GEMM_H

#include <stddef.h>

#include "ckc/ir.h" /* ckc_status_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * ConvProblem
 * ------------------------------------------------------------------ *
 *
 * @dataclass(frozen=True)
 * class ConvProblem:
 *     N: int; Hi: int; Wi: int; C: int
 *     K: int; R: int; S: int
 *     sH: int = 1; sW: int = 1
 *     pH: int = 0; pW: int = 0
 *     dH: int = 1; dW: int = 1
 *
 * Layouts:
 *   A: NHWC fp16, shape [N, Hi, Wi, C]
 *   B: KRSC fp16, shape [K, R, S, C]
 *   D: NHWK fp16, shape [N, Ho, Wo, K]
 *
 * Implicit-GEMM packing:
 *   M = N * Ho * Wo
 *   N_gemm = K
 *   K_gemm = R * S * C
 *
 * Fields use int (signed 32-bit), matching the Python int() arithmetic the
 * derived properties perform.
 */
typedef struct ckc_conv_problem
{
    int N;
    int Hi;
    int Wi;
    int C;
    int K;
    int R;
    int S;
    int sH; /* default 1 */
    int sW; /* default 1 */
    int pH; /* default 0 */
    int pW; /* default 0 */
    int dH; /* default 1 */
    int dW; /* default 1 */
} ckc_conv_problem_t;

/* ConvProblem(N, Hi, Wi, C, K, R, S, sH=1, sW=1, pH=0, pW=0, dH=1, dW=1):
 * construct a ConvProblem with all fields explicit. (The Python dataclass has
 * required N..S and defaulted strides/pads/dilations; the C constructor takes
 * them all so callers can be explicit; use ckc_conv_problem_default() for the
 * defaulted optional fields.) */
ckc_conv_problem_t ckc_conv_problem_make(int N,
                                         int Hi,
                                         int Wi,
                                         int C,
                                         int K,
                                         int R,
                                         int S,
                                         int sH,
                                         int sW,
                                         int pH,
                                         int pW,
                                         int dH,
                                         int dW);

/* Construct a ConvProblem from only the required fields, taking the Python
 * dataclass defaults for the optional ones (sH=sW=1, pH=pW=0, dH=dW=1). */
ckc_conv_problem_t ckc_conv_problem_default(int N, int Hi, int Wi, int C, int K, int R, int S);

/* ConvProblem.Ho property:
 *   (Hi + 2*pH - dH*(R - 1) - 1) // sH + 1
 * Floor division matches Python's `//` for the non-negative operands this
 * shape arithmetic produces. */
int ckc_conv_problem_ho(const ckc_conv_problem_t* p);

/* ConvProblem.Wo property:
 *   (Wi + 2*pW - dW*(S - 1) - 1) // sW + 1 */
int ckc_conv_problem_wo(const ckc_conv_problem_t* p);

/* ConvProblem.M property:  N * Ho * Wo */
int ckc_conv_problem_m(const ckc_conv_problem_t* p);

/* ConvProblem.N_gemm property:  K */
int ckc_conv_problem_n_gemm(const ckc_conv_problem_t* p);

/* ConvProblem.K_gemm property:  R * S * C */
int ckc_conv_problem_k_gemm(const ckc_conv_problem_t* p);

/* ConvProblem.flops property:  2 * M * N_gemm * K_gemm
 * Computed in 64-bit to avoid the int32 overflow Python's arbitrary-precision
 * int never hits. */
long long ckc_conv_problem_flops(const ckc_conv_problem_t* p);

/* ConvProblem.short() ->
 *   f"N{N}H{Hi}W{Wi}C{C}_K{K}R{R}S{S}"
 * Writes the NUL-terminated string into `out` (capacity out_cap). On success
 * returns CKC_OK and, if out_len != NULL, sets *out_len to the byte length
 * (excluding the NUL). Returns CKC_ERR_VALUE on NULL args or a too-small
 * buffer. */
ckc_status_t
ckc_conv_problem_short(const ckc_conv_problem_t* p, char* out, size_t out_cap, size_t* out_len);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_CK_DSL_INSTANCES_COMMON_CONV_IMPLICIT_GEMM_H */
