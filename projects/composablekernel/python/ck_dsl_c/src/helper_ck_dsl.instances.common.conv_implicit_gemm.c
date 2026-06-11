/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C99 port of the ConvProblem dataclass from
 * ck_dsl/instances/common/conv_implicit_gemm.py.
 *
 * Python integer semantics note: every // in ConvProblem operates on
 * non-negative operands (valid convolution shapes), so C integer division
 * (truncation toward zero) matches Python floor division exactly. The flops
 * product is computed in 64-bit because the int32 result Python's
 * arbitrary-precision int never overflows can exceed 2^31 for large shapes.
 */
#include "ckc/helper_ck_dsl.instances.common.conv_implicit_gemm.h"

#include <stdio.h> /* snprintf */

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
                                         int dW)
{
    ckc_conv_problem_t p;
    p.N = N;
    p.Hi = Hi;
    p.Wi = Wi;
    p.C = C;
    p.K = K;
    p.R = R;
    p.S = S;
    p.sH = sH;
    p.sW = sW;
    p.pH = pH;
    p.pW = pW;
    p.dH = dH;
    p.dW = dW;
    return p;
}

ckc_conv_problem_t ckc_conv_problem_default(int N,
                                            int Hi,
                                            int Wi,
                                            int C,
                                            int K,
                                            int R,
                                            int S)
{
    /* sH=1, sW=1, pH=0, pW=0, dH=1, dW=1 (Python dataclass defaults). */
    return ckc_conv_problem_make(N, Hi, Wi, C, K, R, S, 1, 1, 0, 0, 1, 1);
}

/* (Hi + 2*pH - dH*(R - 1) - 1) // sH + 1 */
int ckc_conv_problem_ho(const ckc_conv_problem_t* p)
{
    return (p->Hi + 2 * p->pH - p->dH * (p->R - 1) - 1) / p->sH + 1;
}

/* (Wi + 2*pW - dW*(S - 1) - 1) // sW + 1 */
int ckc_conv_problem_wo(const ckc_conv_problem_t* p)
{
    return (p->Wi + 2 * p->pW - p->dW * (p->S - 1) - 1) / p->sW + 1;
}

/* N * Ho * Wo */
int ckc_conv_problem_m(const ckc_conv_problem_t* p)
{
    return p->N * ckc_conv_problem_ho(p) * ckc_conv_problem_wo(p);
}

/* K */
int ckc_conv_problem_n_gemm(const ckc_conv_problem_t* p)
{
    return p->K;
}

/* R * S * C */
int ckc_conv_problem_k_gemm(const ckc_conv_problem_t* p)
{
    return p->R * p->S * p->C;
}

/* 2 * M * N_gemm * K_gemm */
long long ckc_conv_problem_flops(const ckc_conv_problem_t* p)
{
    long long m = (long long)ckc_conv_problem_m(p);
    long long n = (long long)ckc_conv_problem_n_gemm(p);
    long long k = (long long)ckc_conv_problem_k_gemm(p);
    return 2LL * m * n * k;
}

/* f"N{N}H{Hi}W{Wi}C{C}_K{K}R{R}S{S}" */
ckc_status_t ckc_conv_problem_short(const ckc_conv_problem_t* p,
                                    char* out,
                                    size_t out_cap,
                                    size_t* out_len)
{
    int written;

    if (p == NULL || out == NULL || out_cap == 0)
    {
        return CKC_ERR_VALUE;
    }

    written = snprintf(out, out_cap, "N%dH%dW%dC%d_K%dR%dS%d", p->N, p->Hi,
                       p->Wi, p->C, p->K, p->R, p->S);
    if (written < 0 || (size_t)written >= out_cap)
    {
        /* Encoding error or truncation: the buffer is too small. */
        return CKC_ERR_VALUE;
    }
    if (out_len != NULL)
    {
        *out_len = (size_t)written;
    }
    return CKC_OK;
}
