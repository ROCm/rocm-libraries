/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_ck_dsl.instances.common.conv_implicit_gemm.h -- C99 port of the
 * ConvProblem dataclass from
 * ck_dsl/instances/common/conv_implicit_gemm.py (implicit-GEMM convolution,
 * NHWC x KYXC -> NHWK).
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

#include <stdbool.h>
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
 *     K: int; Y: int; X: int
 *     sH: int = 1; sW: int = 1
 *     pH: int = 0; pW: int = 0
 *     dH: int = 1; dW: int = 1
 *     # 3-D-only (Optional[int], None => 2-D): Di, Z, sD, pD, dD
 *
 * Layouts (2-D):
 *   A: NHWC fp16, shape [N, Hi, Wi, C]
 *   B: KYXC fp16, shape [K, Y, X, C]
 *   D: NHWK fp16, shape [N, Ho, Wo, K]
 * Layouts (3-D, when Di/Z set):
 *   A: NDHWC, [N, Di, Hi, Wi, C]; B: KZYXC, [K, Z, Y, X, C];
 *   D: NDHWK, [N, Do, Ho, Wo, K]
 *
 * Implicit-GEMM packing:
 *   M = N * Ho * Wo  (* Do for 3-D)
 *   N_gemm = K
 *   K_gemm = Y * X * C  (Z * Y * X * C for 3-D)
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
    int Y;
    int X;
    int sH; /* default 1 */
    int sW; /* default 1 */
    int pH; /* default 0 */
    int pW; /* default 0 */
    int dH; /* default 1 */
    int dW; /* default 1 */
    /* 3-D-only fields; 0/false => Python None (2-D conv). */
    bool is_3d; /* true when Di/Z/sD/pD/dD are set */
    int Di;
    int Z;
    int sD;
    int pD;
    int dD;
} ckc_conv_problem_t;

/* ConvProblem(N, Hi, Wi, C, K, Y, X, sH=1, sW=1, pH=0, pW=0, dH=1, dW=1):
 * construct a 2-D ConvProblem with all fields explicit (is_3d=false; the 3-D
 * fields zeroed). Use ckc_conv_problem_default() for the defaulted optional
 * fields, or ckc_conv_problem_make_3d() for a 3-D problem. */
ckc_conv_problem_t ckc_conv_problem_make(int N,
                                         int Hi,
                                         int Wi,
                                         int C,
                                         int K,
                                         int Y,
                                         int X,
                                         int sH,
                                         int sW,
                                         int pH,
                                         int pW,
                                         int dH,
                                         int dW);

/* 3-D ConvProblem: adds the depth dims Di/Z/sD/pD/dD and sets is_3d=true. */
ckc_conv_problem_t ckc_conv_problem_make_3d(int N,
                                            int Di,
                                            int Hi,
                                            int Wi,
                                            int C,
                                            int K,
                                            int Z,
                                            int Y,
                                            int X,
                                            int sD,
                                            int sH,
                                            int sW,
                                            int pD,
                                            int pH,
                                            int pW,
                                            int dD,
                                            int dH,
                                            int dW);

/* Construct a ConvProblem from only the required fields, taking the Python
 * dataclass defaults for the optional ones (sH=sW=1, pH=pW=0, dH=dW=1). 2-D. */
ckc_conv_problem_t ckc_conv_problem_default(int N, int Hi, int Wi, int C, int K, int Y, int X);

/* ConvProblem.is_3d property: Di is not None. */
bool ckc_conv_problem_is_3d(const ckc_conv_problem_t* p);

/* ConvProblem.Do property (3-D only):
 *   (Di + 2*pD - dD*(Z - 1) - 1) // sD + 1.  Returns 1 for 2-D problems
 *   (so M = N*Ho*Wo*Do collapses to N*Ho*Wo). */
int ckc_conv_problem_do(const ckc_conv_problem_t* p);

/* ConvProblem.Ho property:
 *   (Hi + 2*pH - dH*(Y - 1) - 1) // sH + 1
 * Floor division matches Python's `//` for the non-negative operands this
 * shape arithmetic produces. */
int ckc_conv_problem_ho(const ckc_conv_problem_t* p);

/* ConvProblem.Wo property:
 *   (Wi + 2*pW - dW*(X - 1) - 1) // sW + 1 */
int ckc_conv_problem_wo(const ckc_conv_problem_t* p);

/* ConvProblem.M property:  N * Ho * Wo  (* Do for 3-D) */
int ckc_conv_problem_m(const ckc_conv_problem_t* p);

/* ConvProblem.N_gemm property:  K */
int ckc_conv_problem_n_gemm(const ckc_conv_problem_t* p);

/* ConvProblem.K_gemm property:  Y * X * C  (Z * Y * X * C for 3-D) */
int ckc_conv_problem_k_gemm(const ckc_conv_problem_t* p);

/* ConvProblem.flops property:  2 * M * N_gemm * K_gemm
 * Computed in 64-bit to avoid the int32 overflow Python's arbitrary-precision
 * int never hits. */
long long ckc_conv_problem_flops(const ckc_conv_problem_t* p);

/* ConvProblem.short() ->
 *   2-D: f"N{N}H{Hi}W{Wi}C{C}_K{K}Y{Y}X{X}"
 *   3-D: f"N{N}D{Di}H{Hi}W{Wi}C{C}_K{K}Z{Z}Y{Y}X{X}"
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
