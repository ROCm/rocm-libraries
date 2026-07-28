// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of the ConvProblem dataclass from
 * rocke/instances/common/conv_implicit_gemm.py.
 *
 * Python integer semantics note: every // in ConvProblem operates on
 * non-negative operands (valid convolution shapes), so C integer division
 * (truncation toward zero) matches Python floor division exactly. The flops
 * product is computed in 64-bit because the int32 result Python's
 * arbitrary-precision int never overflows can exceed 2^31 for large shapes.
 *
 * #8355: ConvProblem fields R->Y, S->X and optional 3-D depth dims
 * (Di/Z/sD/pD/dD). is_3d gates the depth-aware Do/M/K_gemm/short().
 */
#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h"

#include <stdio.h> /* snprintf */

rocke_conv_problem_t rocke_conv_problem_make(int N,
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
                                             int dW)
{
    rocke_conv_problem_t p;
    p.N = N;
    p.Hi = Hi;
    p.Wi = Wi;
    p.C = C;
    p.K = K;
    p.Y = Y;
    p.X = X;
    p.sH = sH;
    p.sW = sW;
    p.pH = pH;
    p.pW = pW;
    p.dH = dH;
    p.dW = dW;
    p.is_3d = false;
    p.Di = 0;
    p.Z = 0;
    p.sD = 0;
    p.pD = 0;
    p.dD = 0;
    return p;
}

rocke_conv_problem_t rocke_conv_problem_make_3d(int N,
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
                                                int dW)
{
    rocke_conv_problem_t p = rocke_conv_problem_make(N, Hi, Wi, C, K, Y, X, sH, sW, pH, pW, dH, dW);
    p.is_3d = true;
    p.Di = Di;
    p.Z = Z;
    p.sD = sD;
    p.pD = pD;
    p.dD = dD;
    return p;
}

rocke_conv_problem_t rocke_conv_problem_default(int N, int Hi, int Wi, int C, int K, int Y, int X)
{
    /* sH=1, sW=1, pH=0, pW=0, dH=1, dW=1 (Python dataclass defaults). */
    return rocke_conv_problem_make(N, Hi, Wi, C, K, Y, X, 1, 1, 0, 0, 1, 1);
}

bool rocke_conv_problem_is_3d(const rocke_conv_problem_t* p)
{
    return p->is_3d;
}

/* (Di + 2*pD - dD*(Z - 1) - 1) // sD + 1 ; 1 for 2-D. */
int rocke_conv_problem_do(const rocke_conv_problem_t* p)
{
    if(!p->is_3d)
    {
        return 1;
    }
    return (p->Di + 2 * p->pD - p->dD * (p->Z - 1) - 1) / p->sD + 1;
}

/* (Hi + 2*pH - dH*(Y - 1) - 1) // sH + 1 */
int rocke_conv_problem_ho(const rocke_conv_problem_t* p)
{
    return (p->Hi + 2 * p->pH - p->dH * (p->Y - 1) - 1) / p->sH + 1;
}

/* (Wi + 2*pW - dW*(X - 1) - 1) // sW + 1 */
int rocke_conv_problem_wo(const rocke_conv_problem_t* p)
{
    return (p->Wi + 2 * p->pW - p->dW * (p->X - 1) - 1) / p->sW + 1;
}

/* N * Ho * Wo  (* Do for 3-D) */
int rocke_conv_problem_m(const rocke_conv_problem_t* p)
{
    int base = p->N * rocke_conv_problem_ho(p) * rocke_conv_problem_wo(p);
    return p->is_3d ? base * rocke_conv_problem_do(p) : base;
}

/* K */
int rocke_conv_problem_n_gemm(const rocke_conv_problem_t* p)
{
    return p->K;
}

/* Y * X * C  (Z * Y * X * C for 3-D) */
int rocke_conv_problem_k_gemm(const rocke_conv_problem_t* p)
{
    int z = p->is_3d ? p->Z : 1;
    return z * p->Y * p->X * p->C;
}

/* 2 * M * N_gemm * K_gemm */
long long rocke_conv_problem_flops(const rocke_conv_problem_t* p)
{
    long long m = (long long)rocke_conv_problem_m(p);
    long long n = (long long)rocke_conv_problem_n_gemm(p);
    long long k = (long long)rocke_conv_problem_k_gemm(p);
    return 2LL * m * n * k;
}

/* 2-D: f"N{N}H{Hi}W{Wi}C{C}_K{K}Y{Y}X{X}"
 * 3-D: f"N{N}D{Di}H{Hi}W{Wi}C{C}_K{K}Z{Z}Y{Y}X{X}" */
rocke_status_t rocke_conv_problem_short(const rocke_conv_problem_t* p,
                                        char* out,
                                        size_t out_cap,
                                        size_t* out_len)
{
    int written;

    if(p == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }

    if(p->is_3d)
    {
        written = snprintf(out,
                           out_cap,
                           "N%dD%dH%dW%dC%d_K%dZ%dY%dX%d",
                           p->N,
                           p->Di,
                           p->Hi,
                           p->Wi,
                           p->C,
                           p->K,
                           p->Z,
                           p->Y,
                           p->X);
    }
    else
    {
        written = snprintf(
            out, out_cap, "N%dH%dW%dC%d_K%dY%dX%d", p->N, p->Hi, p->Wi, p->C, p->K, p->Y, p->X);
    }
    if(written < 0 || (size_t)written >= out_cap)
    {
        /* Encoding error or truncation: the buffer is too small. */
        return ROCKE_ERR_VALUE;
    }
    if(out_len != NULL)
    {
        *out_len = (size_t)written;
    }
    return ROCKE_OK;
}
