// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Backward data of a 1D convolution, which is also the forward of a ConvTranspose1d. Two kernels,
// selected by CT_GENERIC:
//
// - ConvDgradFma1d (CT_GENERIC 0): the kernel of MIOpen's ConvHipImplicitGemmFmaBwdData1d
//   (MIOpenConvTranspose1dFmaHip.cpp) with its entry point renamed and without its weight
//   conversion kernel: groups == 1, NCL, dilation 1, FMA only, fp32 accumulation. fp32 weights
//   are read in place (CT_WS 1), fp16 / bf16 weights as 32-bit words holding two values each
//   (CT_WS 0), so no variant needs a workspace.
//
//   With o + P = S * q + r (0 <= r < S), output phase r receives the taps t = r + S * m, each
//   reading dy[q - m]: dx[co, S * q + r - P] = sum over (ci, m) of w[ci, co, r + S * m] *
//   dy[ci, q - m]. A 256-thread block covers 8 * CT_CO_W output channels and 32 * CT_QR values
//   of q. Wave v owns CT_CO_W channels, so its weights are wave-uniform and are read with scalar
//   loads straight from w[ci][co][k]; lane l owns CT_QR consecutive q, i.e. S * CT_QR consecutive
//   outputs per channel. dy is staged in LDS CT_CI_T input channels at a time.
// - ConvDgradGeneric1d (CT_GENERIC 1): one dx element per lane, gathering over the dy channels of
//   its group and the taps that reach it. Any filter width, stride, dilation, padding and groups,
//   channels-first or channels-last.

#include <hip/hip_runtime.h>

#ifndef CT_GENERIC
#define CT_GENERIC 0
#endif

#ifndef CT_DTYPE
#define CT_DTYPE 0 // 0 fp32, 1 fp16, 2 bf16
#endif
#ifndef CT_K
#define CT_K 4
#endif
#ifndef CT_S
#define CT_S 2
#endif
#ifndef CT_QR
#define CT_QR 2
#endif
#ifndef CT_CO_W
#define CT_CO_W 8
#endif
#ifndef CT_CI_T
#define CT_CI_T 32
#endif
#ifndef CT_WS
#define CT_WS 1 // 1: fp32 weights; 0: fp16 / bf16 weights read in place as 32-bit words
#endif

#define CT_BLOCK_SIZE 256
#define CT_T ((CT_K + CT_S - 1) / CT_S) // taps per phase
#define CT_CO_T (8 * CT_CO_W) // output channels per block
#define CT_Q_T (32 * CT_QR) // q per block
#define CT_XW (CT_Q_T + CT_T - 1) // dy window per input channel
#define CT_XL ((CT_CI_T * CT_XW + CT_BLOCK_SIZE - 1) / CT_BLOCK_SIZE) // dy loads per thread

#if CT_DTYPE == 0
typedef float data_t;
__device__ static inline float to_float(data_t v)
{
    return v;
}
__device__ static inline data_t from_float(float v)
{
    return v;
}
#elif CT_DTYPE == 1
typedef _Float16 data_t;
__device__ static inline float to_float(data_t v)
{
    return static_cast<float>(v);
}
__device__ static inline data_t from_float(float v)
{
    return static_cast<data_t>(v);
}
#else
typedef unsigned short data_t;
__device__ static inline float to_float(data_t v)
{
    return __builtin_bit_cast(float, static_cast<unsigned int>(v) << 16);
}
// Round to nearest even; NaN stays NaN.
__device__ static inline data_t from_float(float v)
{
    const unsigned int u = __builtin_bit_cast(unsigned int, v);
    if((u & 0x7fffffffu) > 0x7f800000u)
        return static_cast<data_t>((u >> 16) | 0x40u);
    return static_cast<data_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
#endif

#if !CT_GENERIC

// 16-bit weights read in place.
#define CT_W16 (CT_DTYPE != 0 && !CT_WS)
#if CT_W16
typedef data_t weight_t;
#else
typedef float weight_t;
#endif

#if CT_W16
// 32-bit words covering the K weights of one output channel: a channel starts on a word boundary
// when K is even, and may start in the upper half of a word when K is odd.
#define CT_KW (CT_K % 2 == 0 ? CT_K / 2 : CT_K / 2 + 1)

__device__ static inline float half_to_float(unsigned int h)
{
#if CT_DTYPE == 1
    return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<unsigned short>(h)));
#else
    return __builtin_bit_cast(float, h << 16);
#endif
}

// The K values of a channel from its words; sh is 16 when the channel starts in the upper half of
// its first word (K odd only), else 0. Scalar (SALU) work only.
__device__ static inline void
    unpack_weights(const unsigned int (&wd)[CT_KW], unsigned int sh, float (&wv)[CT_K])
{
#pragma unroll
    for(int i = 0; i < (CT_K + 1) / 2; ++i)
    {
        unsigned int word = wd[i];
#if CT_K % 2 != 0
        if(i + 1 < CT_KW)
            word = static_cast<unsigned int>(
                ((static_cast<unsigned long long>(wd[i + 1]) << 32) | wd[i]) >> sh);
        else
            word >>= sh;
#endif
        wv[2 * i] = half_to_float(word & 0xffffu);
        if(2 * i + 1 < CT_K)
            wv[2 * i + 1] = half_to_float(word >> 16);
    }
}
#endif

extern "C" __global__ void __launch_bounds__(CT_BLOCK_SIZE)
    ConvDgradFma1d(const data_t* __restrict__ dy,
                   const weight_t* __restrict__ w,
                   data_t* __restrict__ dx,
                   int ci_n,
                   int co_n,
                   int len_dy,
                   int len_dx,
                   int pad)
{
    __shared__ float xs[CT_CI_T][CT_XW];

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int wave = __builtin_amdgcn_readfirstlane(tid / 32);
    const int qb = blockIdx.x * CT_Q_T;
    const int co0 = blockIdx.y * CT_CO_T + wave * CT_CO_W;
    dy += static_cast<size_t>(blockIdx.z) * ci_n * len_dy;
    dx += static_cast<size_t>(blockIdx.z) * co_n * len_dx;
#if CT_W16
    const size_t w_words = (static_cast<size_t>(ci_n) * co_n * CT_K + 1) / 2;
#endif

    float acc[CT_S][CT_CO_W][CT_QR];
#pragma unroll
    for(int r = 0; r < CT_S; ++r)
#pragma unroll
        for(int c = 0; c < CT_CO_W; ++c)
#pragma unroll
            for(int j = 0; j < CT_QR; ++j)
                acc[r][c][j] = 0.f;

    for(int cb = 0; cb < ci_n; cb += CT_CI_T)
    {
        float xr[CT_XL];
#pragma unroll
        for(int u = 0; u < CT_XL; ++u)
        {
            const int i = tid + CT_BLOCK_SIZE * u;
            const int ci = cb + i / CT_XW;
            const int q = qb - (CT_T - 1) + i % CT_XW;
            xr[u] = (i < CT_CI_T * CT_XW && ci < ci_n && q >= 0 && q < len_dy)
                        ? to_float(dy[static_cast<size_t>(ci) * len_dy + q])
                        : 0.f;
        }
        __syncthreads();
#pragma unroll
        for(int u = 0; u < CT_XL; ++u)
        {
            const int i = tid + CT_BLOCK_SIZE * u;
            if(i < CT_CI_T * CT_XW)
                xs[i / CT_XW][i % CT_XW] = xr[u];
        }
        __syncthreads();

        const int cn = min(CT_CI_T, ci_n - cb);
        for(int ii = 0; ii < cn; ++ii)
        {
            // dy[q - m] for the lane's q, m = 0 .. T - 1.
            float xv[CT_QR + CT_T - 1];
#pragma unroll
            for(int j = 0; j < CT_QR + CT_T - 1; ++j)
                xv[j] = xs[ii][lane * CT_QR + j];
            // First weight of each of the wave's channels; clamped so that the channels past co_n
            // in the last tile read valid memory (their results are not stored).
            size_t e[CT_CO_W];
#pragma unroll
            for(int c = 0; c < CT_CO_W; ++c)
                e[c] = (static_cast<size_t>(cb + ii) * co_n + min(co0 + c, co_n - 1)) * CT_K;
#if CT_W16
            // All the words first, so that the scalar loads are issued together.
            const unsigned int* w32 = reinterpret_cast<const unsigned int*>(w);
            unsigned int wd[CT_CO_W][CT_KW];
#pragma unroll
            for(int c = 0; c < CT_CO_W; ++c)
#pragma unroll
                for(int i = 0; i < CT_KW; ++i)
                {
                    // With K odd the last word may lie past the tensor: clamp it.
                    const size_t k = e[c] / 2 + i;
                    wd[c][i] = w32[CT_K % 2 == 0 || k < w_words ? k : w_words - 1];
                }
#endif
#pragma unroll
            for(int c = 0; c < CT_CO_W; ++c)
            {
                float wv[CT_K];
#if !CT_W16
#pragma unroll
                for(int t = 0; t < CT_K; ++t)
                    wv[t] = w[e[c] + t];
#else
                const unsigned int sh
                    = CT_K % 2 != 0 ? static_cast<unsigned int>(e[c] % 2) * 16 : 0u;
                unpack_weights(wd[c], sh, wv);
#endif
#pragma unroll
                for(int t = 0; t < CT_K; ++t)
                {
#pragma unroll
                    for(int j = 0; j < CT_QR; ++j)
                        acc[t % CT_S][c][j] += wv[t] * xv[j + CT_T - 1 - t / CT_S];
                }
            }
        }
    }

#pragma unroll
    for(int c = 0; c < CT_CO_W; ++c)
    {
        if(co0 + c >= co_n)
            break;
        data_t* yp = dx + static_cast<size_t>(co0 + c) * len_dx;
#pragma unroll
        for(int j = 0; j < CT_QR; ++j)
        {
            const int q = qb + lane * CT_QR + j;
#pragma unroll
            for(int r = 0; r < CT_S; ++r)
            {
                const int o = CT_S * q + r - pad;
                if(o >= 0 && o < len_dx)
                    yp[o] = from_float(acc[r][c][j]);
            }
        }
    }
}

#else // CT_GENERIC

#ifndef CT_CHANNELS_LAST
#define CT_CHANNELS_LAST 0
#endif

// dx (N, C, Lx), w (K, C / G, R), dy (N, K, Ly): dx channel c of group g = c / (C / G) receives dy
// channels g * K / G .. (g + 1) * K / G - 1. Dy position o reaches dx position x through tap t
// when o * s + t * d - p == x. Channels-last stores C (K, and C / G of w) innermost.
extern "C" __global__ void __launch_bounds__(CT_BLOCK_SIZE)
    ConvDgradGeneric1d(const data_t* __restrict__ dy,
                       const data_t* __restrict__ w,
                       data_t* __restrict__ dx,
                       int n_n,
                       int c_n,
                       int k_n,
                       int groups,
                       int len_dx,
                       int len_dy,
                       int r_n,
                       int s,
                       int p,
                       int d)
{
    const int cg = c_n / groups;
    const int kg = k_n / groups;
    const long long total = static_cast<long long>(n_n) * c_n * len_dx;
    const long long step = static_cast<long long>(gridDim.x) * CT_BLOCK_SIZE;

    for(long long e = static_cast<long long>(blockIdx.x) * CT_BLOCK_SIZE + threadIdx.x; e < total;
        e += step)
    {
#if CT_CHANNELS_LAST
        const int c = static_cast<int>(e % c_n);
        const int x = static_cast<int>((e / c_n) % len_dx);
        const int n = static_cast<int>(e / c_n / len_dx);
#else
        const int x = static_cast<int>(e % len_dx);
        const int c = static_cast<int>((e / len_dx) % c_n);
        const int n = static_cast<int>(e / len_dx / c_n);
#endif
        const int g = c / cg;
        const int cl = c - g * cg;
        float acc = 0.0f;
        for(int t = 0; t < r_n; ++t)
        {
            const int os = x + p - t * d;
            if(os < 0 || os % s != 0 || os / s >= len_dy)
                continue;
            const int o = os / s;
            for(int j = 0; j < kg; ++j)
            {
                const int k = g * kg + j;
#if CT_CHANNELS_LAST
                const long long dy_index = (static_cast<long long>(n) * len_dy + o) * k_n + k;
                const long long w_index = (static_cast<long long>(k) * r_n + t) * cg + cl;
#else
                const long long dy_index = (static_cast<long long>(n) * k_n + k) * len_dy + o;
                const long long w_index = (static_cast<long long>(k) * cg + cl) * r_n + t;
#endif
                acc = __builtin_fmaf(to_float(dy[dy_index]), to_float(w[w_index]), acc);
            }
        }
        dx[e] = from_float(acc);
    }
}

#endif // CT_GENERIC
