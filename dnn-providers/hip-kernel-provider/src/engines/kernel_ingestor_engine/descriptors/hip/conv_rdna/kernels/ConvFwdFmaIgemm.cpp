// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

// Forward convolution as an implicit GEMM on plain FMA (no WMMA / MFMA), so it runs on every
// RDNA target including the ones without matrix instructions. Derived from MIOpen's
// MIOpenConv3dFwdFmaHip.cpp. The kernel is 3D; 2D problems run with depth 1 and 1D problems as
// 2D with height 1.
//
// GEMM view, per group g:
//   M = N * Do * Ho * Wo (output positions), N_gemm = K / groups, K_gemm = Z * Y * X * C / groups
//   y[m][k] = sum_{z, y, x, c} x[n][di][hi][wi][c] * w[k][z][y][x][c]
//
// Each 256-thread workgroup computes a TILE_M x TILE_N output tile and every thread accumulates
// a MICRO_M x MICRO_N register tile in fp32. One K_gemm iteration covers one filter tap (z, y, x)
// and TILE_C input channels, so the spatial bounds check is done once per tap. Inputs are
// converted to fp32 when they are staged in LDS. The global loads for the next iteration are
// issued before the FMAs of the current one and land in the other half of a double-buffered LDS
// tile, so each iteration needs one barrier.
//
// Weights are read in their own layout, so no workspace is needed: channels-last weights are
// K x (Z * Y * X) x (C / groups), channels-first weights are K x (C / groups) x (Z * Y * X) and
// are read with the channels of one tap `taps` apart (CONV_WEI_KCZYX).
//
// Compile-time parameters (one set per UKD):
//   CONV_DTYPE          0 = fp32, 1 = fp16, 2 = bf16
//   CONV_CHANNELS_LAST  0 = NCDHW input / NKDHW output, 1 = NDHWC input / NDHWK output
//   CONV_WEI_KCZYX      0 = weights K x taps x (C / groups), 1 = K x (C / groups) x taps
//   TILE_N              output channels per workgroup: 128, 64 or 32 (MICRO_N = TILE_N / 16)
//   CONV_HAS_BIAS       1 = add bias[k] (one value per output channel, the dtype of the data)
//                       to every output before it is stored; the kernel then takes the bias
//                       pointer after `out`
//   CONV_HAS_ACT        1 = apply an activation to every output, after the bias, before it is
//                       stored; the kernel then takes the activation (ACT_* below), its three
//                       parameters and has_bias after `groups`. Built with CONV_HAS_BIAS; a graph
//                       without a bias passes has_bias 0, and the bias pointer is not read

#ifndef CONV_DTYPE
#define CONV_DTYPE 0
#endif
#ifndef CONV_CHANNELS_LAST
#define CONV_CHANNELS_LAST 0
#endif
#ifndef CONV_WEI_KCZYX
#define CONV_WEI_KCZYX 0
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef CONV_HAS_BIAS
#define CONV_HAS_BIAS 0
#endif
#ifndef CONV_HAS_ACT
#define CONV_HAS_ACT 0
#endif

#define TILE_M 128
#define TILE_C 16
#define MICRO_M 8
#define MICRO_N (TILE_N / 16)
#define BLOCK_SIZE 256
// Each thread stages one row of the A (input) and B (weight) tiles, LOAD_A / LOAD_B channels long.
#define LOAD_A (TILE_M * TILE_C / BLOCK_SIZE)
#define LOAD_B (TILE_N * TILE_C / BLOCK_SIZE)

#if CONV_DTYPE == 0
typedef float data_t;
__device__ static inline float to_float(data_t v)
{
    return v;
}
__device__ static inline data_t from_float(float v)
{
    return v;
}
#elif CONV_DTYPE == 1
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
__device__ static inline data_t from_float(float v)
{
    // Round to nearest even; NaN keeps a quiet mantissa bit.
    const unsigned int u = __builtin_bit_cast(unsigned int, v);
    if((u & 0x7fffffffu) > 0x7f800000u)
        return static_cast<data_t>((u >> 16) | 0x40u);
    return static_cast<data_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
#endif

#if CONV_HAS_ACT
// The activations, numbered as conv_rdna::fma_igemm::ActivationMode.
#define ACT_RELU 0
#define ACT_SIGMOID 1
#define ACT_TANH 2
#define ACT_ELU 3
#define ACT_GELU 4
#define ACT_GELU_APPROX_TANH 5
#define ACT_SWISH 6
#define ACT_SOFTPLUS 7
#define ACT_IDENTITY 8

// The forward activations of hipDNN's pointwise reference, in fp32. `act` is a kernel argument,
// so every thread takes the same branch.
//   ACT_RELU      p0 lower clip, p1 upper clip, p2 slope below the lower clip
//   ACT_ELU       p0 alpha
//   ACT_SWISH     p0 beta
//   ACT_SOFTPLUS  p0 beta
__device__ static inline float activate(float v, int act, float p0, float p1, float p2)
{
    switch(act)
    {
    case ACT_RELU:
        return v <= p0 ? p2 * (v - p0) + p0 : (v >= p1 ? p1 : v);
    case ACT_SIGMOID:
        return 1.f / (1.f + expf(-v));
    case ACT_TANH:
        return tanhf(v);
    case ACT_ELU:
        return v > 0.f ? v : p0 * expm1f(v);
    case ACT_GELU:
        return 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
    case ACT_GELU_APPROX_TANH:
        return 0.5f * v * (1.f + tanhf(0.79788456080286536f * (v + 0.044715f * v * v * v)));
    case ACT_SWISH:
        return v / (1.f + expf(-p0 * v));
    case ACT_SOFTPLUS:
    {
        // Linear above 20, where log1p(exp(s)) / beta equals v in fp32 and exp would overflow.
        const float s = p0 * v;
        return s > 20.f ? v : log1pf(expf(s)) / p0;
    }
    default:
        return v;
    }
}
#endif

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE)
    ConvFwdFmaIgemm(const data_t* __restrict__ in,
                    const data_t* __restrict__ wei,
                    data_t* __restrict__ out,
#if CONV_HAS_BIAS
                    const data_t* __restrict__ bias,
#endif
                    int N,
                    int C,
                    int Di,
                    int Hi,
                    int Wi,
                    int K,
                    int Z,
                    int Y,
                    int X,
                    int Do,
                    int Ho,
                    int Wo,
                    int sd,
                    int sh,
                    int sw,
                    int pd,
                    int ph,
                    int pw,
                    int dd,
                    int dh,
                    int dw,
                    int groups
#if CONV_HAS_ACT
                    ,
                    int act,
                    float act_p0,
                    float act_p1,
                    float act_p2,
                    int has_bias
#endif
    )
{
    __shared__ float as[2][TILE_C][TILE_M + 4];
    __shared__ float bs[2][TILE_C][TILE_N + 4];

    const int tid = threadIdx.x;
    const int Cg = C / groups;
    const int Kg = K / groups;
    const int nb_per_g = (Kg + TILE_N - 1) / TILE_N;
    const int g = blockIdx.y / nb_per_g;
    const int k0 = (blockIdx.y % nb_per_g) * TILE_N;
    // Offsets are long long: long is 32 bits in HIP device code built on Windows.
    const long long M = static_cast<long long>(N) * Do * Ho * Wo;
    const long long m0 = static_cast<long long>(blockIdx.x) * TILE_M;
    const long long sp_in = static_cast<long long>(Di) * Hi * Wi;
    const long long sp_out = static_cast<long long>(Do) * Ho * Wo;
    const int taps = Z * Y * X;

    // A row (output position) staged by this thread. NCDHW: consecutive threads take consecutive
    // positions so the loads coalesce along W; NDHWC: consecutive threads take consecutive
    // channels of one position.
#if CONV_CHANNELS_LAST
    const int a_row = tid / (TILE_C / LOAD_A);
    const int a_col = (tid % (TILE_C / LOAD_A)) * LOAD_A;
#else
    const int a_row = tid % TILE_M;
    const int a_col = (tid / TILE_M) * LOAD_A;
#endif
    const long long am = m0 + a_row;
    const bool am_ok = am < M;
    int an = 0, aod = 0, aoh = 0, aow = 0;
    if(am_ok)
    {
        long long t = am;
        aow = static_cast<int>(t % Wo);
        t /= Wo;
        aoh = static_cast<int>(t % Ho);
        t /= Ho;
        aod = static_cast<int>(t % Do);
        an = static_cast<int>(t / Do);
    }
    const int di0 = aod * sd - pd;
    const int hi0 = aoh * sh - ph;
    const int wi0 = aow * sw - pw;
#if CONV_CHANNELS_LAST
    const data_t* in_n
        = in + static_cast<long long>(an) * sp_in * C + static_cast<long long>(g) * Cg;
    const long long in_cstride = 1;
#else
    const data_t* in_n
        = in + (static_cast<long long>(an) * C + static_cast<long long>(g) * Cg) * sp_in;
    const long long in_cstride = sp_in;
#endif

    // B row (output channel) staged by this thread.
    const int b_row = tid / (TILE_C / LOAD_B);
    const int b_col = (tid % (TILE_C / LOAD_B)) * LOAD_B;
    const int bk = k0 + b_row;
    const bool bk_ok = bk < Kg;
    // Both weight layouts hold taps * Cg values per output channel; they differ only inside it.
    const data_t* wei_k
        = wei
          + (static_cast<long long>(g) * Kg + (bk_ok ? bk : 0)) * static_cast<long long>(taps) * Cg;
#if CONV_WEI_KCZYX
    const long long wei_cstride = taps;
#else
    const long long wei_cstride = 1;
#endif

    // Register tile owned by this thread, strided by TILE / MICRO so that each store instruction
    // writes consecutive addresses. Consecutive threads step along the dimension that is
    // contiguous in the output: K for NDHWK, the output position for NKDHW.
#if CONV_CHANNELS_LAST
    const int tn = tid % (TILE_N / MICRO_N);
    const int tm = tid / (TILE_N / MICRO_N);
#else
    const int tm = tid % (TILE_M / MICRO_M);
    const int tn = tid / (TILE_M / MICRO_M);
#endif

    float acc[MICRO_M][MICRO_N];
    for(int i = 0; i < MICRO_M; ++i)
        for(int j = 0; j < MICRO_N; ++j)
            acc[i][j] = 0.f;

    // Flattened K loop: iteration `it` covers filter tap it / nc and channels
    // [(it % nc) * TILE_C, + TILE_C).
    const int nc = (Cg + TILE_C - 1) / TILE_C;
    const int iters = taps * nc;
    const int yx = Y * X;
    float ra[LOAD_A], rb[LOAD_B];

    auto load = [&](int it) {
        const int tap = it / nc;
        const int c0 = (it - tap * nc) * TILE_C;
        const int z = tap / yx;
        const int r = tap - z * yx;
        const int y = r / X;
        const int x = r - y * X;
        const int di = di0 + z * dd;
        const int hi = hi0 + y * dh;
        const int wi = wi0 + x * dw;
        const bool a_ok = am_ok && di >= 0 && di < Di && hi >= 0 && hi < Hi && wi >= 0 && wi < Wi;
        // Out-of-range rows read the clamped offset 0 and are replaced by 0 with a select, so
        // the loads do not branch.
        const long long a_sp = a_ok ? (static_cast<long long>(di) * Hi + hi) * Wi + wi : 0;
#if CONV_CHANNELS_LAST
        const data_t* a_ptr = in_n + a_sp * C;
#else
        const data_t* a_ptr = in_n + a_sp;
#endif
#if CONV_WEI_KCZYX
        const data_t* b_ptr = wei_k + tap;
#else
        const data_t* b_ptr = wei_k + static_cast<long long>(tap) * Cg;
#endif
        if(c0 + TILE_C <= Cg)
        {
            for(int i = 0; i < LOAD_A; ++i)
            {
                const float v = to_float(a_ptr[(c0 + a_col + i) * in_cstride]);
                ra[i] = a_ok ? v : 0.f;
            }
            for(int i = 0; i < LOAD_B; ++i)
            {
                const float v = to_float(b_ptr[(c0 + b_col + i) * wei_cstride]);
                rb[i] = bk_ok ? v : 0.f;
            }
            return;
        }
        // Partial channel chunk (wave-uniform).
        for(int i = 0; i < LOAD_A; ++i)
        {
            const int c = c0 + a_col + i;
            ra[i] = (a_ok && c < Cg) ? to_float(a_ptr[c * in_cstride]) : 0.f;
        }
        for(int i = 0; i < LOAD_B; ++i)
        {
            const int c = c0 + b_col + i;
            rb[i] = (bk_ok && c < Cg) ? to_float(b_ptr[c * wei_cstride]) : 0.f;
        }
    };
    auto store = [&](int buf) {
        for(int i = 0; i < LOAD_A; ++i)
            as[buf][a_col + i][a_row] = ra[i];
        for(int i = 0; i < LOAD_B; ++i)
            bs[buf][b_col + i][b_row] = rb[i];
    };
    auto compute = [&](int buf) {
        for(int kk = 0; kk < TILE_C; ++kk)
        {
            float a[MICRO_M], b[MICRO_N];
            for(int i = 0; i < MICRO_M; ++i)
                a[i] = as[buf][kk][tm + i * (TILE_M / MICRO_M)];
            for(int j = 0; j < MICRO_N; ++j)
                b[j] = bs[buf][kk][tn + j * (TILE_N / MICRO_N)];
            for(int i = 0; i < MICRO_M; ++i)
                for(int j = 0; j < MICRO_N; ++j)
                    acc[i][j] = __builtin_fmaf(a[i], b[j], acc[i][j]);
        }
    };

    load(0);
    store(0);
    __syncthreads();
    for(int it = 0; it < iters; ++it)
    {
        const int buf = it & 1;
        if(it + 1 < iters)
            load(it + 1);
        compute(buf);
        if(it + 1 < iters)
            store(buf ^ 1);
        __syncthreads();
    }

#if CONV_HAS_BIAS
    // The bias of this thread's output channels, added in fp32 before the one rounding.
    float bv[MICRO_N];
    for(int j = 0; j < MICRO_N; ++j)
    {
        const int k = k0 + tn + j * (TILE_N / MICRO_N);
#if CONV_HAS_ACT
        const bool read = k < Kg && has_bias != 0;
#else
        const bool read = k < Kg;
#endif
        bv[j] = read ? to_float(bias[static_cast<long long>(g) * Kg + k]) : 0.f;
    }
#endif

    for(int i = 0; i < MICRO_M; ++i)
    {
        const long long m = m0 + tm + i * (TILE_M / MICRO_M);
        if(m >= M)
            continue;
#if CONV_CHANNELS_LAST
        data_t* out_m = out + m * K + static_cast<long long>(g) * Kg;
        const long long out_kstride = 1;
#else
        const long long n = m / sp_out;
        data_t* out_m = out + (n * K + static_cast<long long>(g) * Kg) * sp_out + (m - n * sp_out);
        const long long out_kstride = sp_out;
#endif
        for(int j = 0; j < MICRO_N; ++j)
        {
            const int k = k0 + tn + j * (TILE_N / MICRO_N);
#if CONV_HAS_BIAS
            float v = acc[i][j] + bv[j];
#else
            float v = acc[i][j];
#endif
#if CONV_HAS_ACT
            v = activate(v, act, act_p0, act_p1, act_p2);
#endif
            if(k < Kg)
                out_m[k * out_kstride] = from_float(v);
        }
    }
}
