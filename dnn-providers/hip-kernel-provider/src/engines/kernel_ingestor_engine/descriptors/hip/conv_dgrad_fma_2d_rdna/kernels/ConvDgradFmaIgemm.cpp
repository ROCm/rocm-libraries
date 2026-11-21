// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

// Backward data of a convolution (the forward of a transposed convolution) as an implicit GEMM
// on plain FMA (no WMMA / MFMA), so it runs on every RDNA target including the ones without
// matrix instructions. The tiling, LDS double buffering and register tiles follow the forward
// kernel of hip/conv_rdna/ (ConvFwdFmaIgemm.cpp). The kernel is 3D; 2D problems run with
// depth 1. hip/conv_dgrad_fma_3d_rdna/ builds its variants from this file too.
//
// Stride decomposition: dx position (di, hi, wi) is written di = qd * sd + rd and alike, and the
// taps that reach it are the z with (rd + pd - z * dd) divisible by sd (and alike for y, x),
// reading dy at od = qd + (rd + pd - z * dd) / sd. Each workgroup takes the dx positions of one
// phase (rd, rh, rw), so all its rows share one set of taps, and it walks only those taps.
//
// GEMM view, per group g and phase:
//   M = N * Dq * Hq * Wq (dx positions of the phase), N_gemm = C / groups,
//   K_gemm = (taps of the phase) * K / groups
//   dx[m][c] = sum_{tap, k} dy[n][k][od][oh][ow] * w[k][c][z][y][x]
//
// Each 256-thread workgroup computes a TILE_M x TILE_N tile and every thread accumulates a
// MICRO_M x MICRO_N register tile in fp32. One K_gemm iteration covers one tap and TILE_C dy
// channels. dy and the weights are read in place (channels-first weights K x (C / groups) x
// taps, channels-last weights K x taps x (C / groups)), so no workspace is needed. dx positions
// no tap reaches are written as 0. Offsets are long long: long is 32 bits in HIP device code
// built on Windows.
//
// Compile-time parameters (one set per UKD):
//   CONV_DTYPE          0 = fp32, 1 = fp16, 2 = bf16
//   CONV_CHANNELS_LAST  0 = NCDHW dx / NKDHW dy / KCZYX weights, 1 = NDHWC / NDHWK / KZYXC
//   TILE_N              dx channels per workgroup: 128, 64 or 32 (MICRO_N = TILE_N / 16)

#ifndef CONV_DTYPE
#define CONV_DTYPE 0
#endif
#ifndef CONV_CHANNELS_LAST
#define CONV_CHANNELS_LAST 0
#endif
#ifndef TILE_N
#define TILE_N 128
#endif

#define TILE_M 128
#define TILE_C 16
#define MICRO_M 8
#define MICRO_N (TILE_N / 16)
#define BLOCK_SIZE 256
// Each thread stages LOAD_A dy values of one A row and LOAD_B weights of one B row.
#define LOAD_A (TILE_M * TILE_C / BLOCK_SIZE)
#define LOAD_B (TILE_N * TILE_C / BLOCK_SIZE)

// reg_t holds a value between its global load and its store to LDS: the value itself for fp32,
// the 16 bits zero-extended for fp16 / bf16. Converting only at the LDS store lets the loads of
// the next iteration stay in flight across the FMAs; converting at the load makes every
// iteration wait for them.
#if CONV_DTYPE == 0
typedef float data_t;
typedef float reg_t;
__device__ static inline float to_float(data_t v)
{
    return v;
}
__device__ static inline data_t from_float(float v)
{
    return v;
}
#elif CONV_DTYPE == 1
typedef unsigned short data_t;
typedef unsigned int reg_t;
__device__ static inline float to_float(data_t v)
{
    return static_cast<float>(__builtin_bit_cast(_Float16, v));
}
__device__ static inline data_t from_float(float v)
{
    return __builtin_bit_cast(data_t, static_cast<_Float16>(v));
}
#else
typedef unsigned short data_t;
typedef unsigned int reg_t;
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

// The taps of one axis that reach phase r: t = first + j * step for j < count, where
// (r + p - t * d) is divisible by s. Wave-uniform.
struct AxisTaps
{
    int first;
    int step;
    int count;
};

__device__ static inline int gcd_int(int a, int b)
{
    while(b != 0)
    {
        const int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

__device__ static inline AxisTaps axis_taps(int r, int p, int s, int d, int taps)
{
    const int step = s / gcd_int(s, d);
    AxisTaps a{0, step, 0};
    for(int t = 0; t < step && t < taps; ++t)
    {
        // (r + p - t * d) mod s, kept non-negative.
        const int v = ((r + p - t * d) % s + s) % s;
        if(v == 0)
        {
            a.first = t;
            a.count = (taps - 1 - t) / step + 1;
            return a;
        }
    }
    return a;
}

// Phase positions along one axis: ceil((len - r) / s), 0 when r >= len.
__device__ static inline int phase_len(int len, int r, int s)
{
    return r < len ? (len - r + s - 1) / s : 0;
}

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE)
    ConvDgradFmaIgemm(const data_t* __restrict__ dy,
                      const data_t* __restrict__ wei,
                      data_t* __restrict__ dx,
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
                      int groups)
{
    __shared__ float as[2][TILE_C][TILE_M + 4];
    __shared__ float bs[2][TILE_C][TILE_N + 4];

    const int tid = threadIdx.x;
    const int Cg = C / groups;
    const int Kg = K / groups;
    const int nb_per_g = (Cg + TILE_N - 1) / TILE_N;
    const int g = blockIdx.y / nb_per_g;
    const int c0 = (blockIdx.y % nb_per_g) * TILE_N;

    // Phase of this workgroup and the dx positions it covers. Phase (0, 0, 0) has the most
    // positions; every phase gets that many blocks and masks the rest.
    const long long m_max = static_cast<long long>(N) * phase_len(Di, 0, sd) * phase_len(Hi, 0, sh)
                            * phase_len(Wi, 0, sw);
    const long long blocks_per_phase = (m_max + TILE_M - 1) / TILE_M;
    const long long phase = blockIdx.x / blocks_per_phase;
    const long long m0 = (blockIdx.x % blocks_per_phase) * TILE_M;
    const int rw = static_cast<int>(phase % sw);
    const int rh = static_cast<int>((phase / sw) % sh);
    const int rd = static_cast<int>(phase / sw / sh);
    const int Dq = phase_len(Di, rd, sd);
    const int Hq = phase_len(Hi, rh, sh);
    const int Wq = phase_len(Wi, rw, sw);
    const long long M = static_cast<long long>(N) * Dq * Hq * Wq;

    const AxisTaps tz = axis_taps(rd, pd, sd, dd, Z);
    const AxisTaps ty = axis_taps(rh, ph, sh, dh, Y);
    const AxisTaps tx = axis_taps(rw, pw, sw, dw, X);
    const int taps_v = tz.count * ty.count * tx.count;
    const int taps = Z * Y * X;
    const long long sp_dx = static_cast<long long>(Di) * Hi * Wi;
    const long long sp_dy = static_cast<long long>(Do) * Ho * Wo;

    // A row (dx position) staged by this thread. Channels-first: consecutive threads take
    // consecutive positions, which within a phase read consecutive dy positions along W.
#if CONV_CHANNELS_LAST
    const int a_row = tid / (TILE_C / LOAD_A);
    const int a_col = (tid % (TILE_C / LOAD_A)) * LOAD_A;
#else
    const int a_row = tid % TILE_M;
    const int a_col = (tid / TILE_M) * LOAD_A;
#endif
    const long long am = m0 + a_row;
    const bool am_ok = am < M;
    int an = 0, aqd = 0, aqh = 0, aqw = 0;
    if(am_ok)
    {
        long long t = am;
        aqw = static_cast<int>(t % Wq);
        t /= Wq;
        aqh = static_cast<int>(t % Hq);
        t /= Hq;
        aqd = static_cast<int>(t % Dq);
        an = static_cast<int>(t / Dq);
    }
#if CONV_CHANNELS_LAST
    const data_t* dy_n
        = dy + static_cast<long long>(an) * sp_dy * K + static_cast<long long>(g) * Kg;
    const long long dy_kstride = 1;
#else
    const data_t* dy_n
        = dy + (static_cast<long long>(an) * K + static_cast<long long>(g) * Kg) * sp_dy;
    const long long dy_kstride = sp_dy;
#endif

    // B row (dx channel) staged by this thread; consecutive threads take consecutive channels.
    const int b_row = tid % TILE_N;
    const int b_col = (tid / TILE_N) * LOAD_B;
    const int bc = c0 + b_row;
    const bool bc_ok = bc < Cg;
    const data_t* wei_g = wei + static_cast<long long>(g) * Kg * taps * Cg;

    // Register tile owned by this thread, strided by TILE / MICRO so that each store
    // instruction writes consecutive addresses where dx is contiguous.
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

    // Flattened K loop over the phase's taps: iteration `it` covers phase tap it / nk and dy
    // channels [(it % nk) * TILE_C, + TILE_C).
    const int nk = (Kg + TILE_C - 1) / TILE_C;
    const int iters = taps_v * nk;
    reg_t ra[LOAD_A], rb[LOAD_B];

    auto load = [&](int it) {
        const int j = it / nk;
        const int k0 = (it - j * nk) * TILE_C;
        const int jx = j % tx.count;
        const int jy = (j / tx.count) % ty.count;
        const int jz = j / tx.count / ty.count;
        const int z = tz.first + jz * tz.step;
        const int y = ty.first + jy * ty.step;
        const int x = tx.first + jx * tx.step;
        // Exact divisions: the phase's taps make the numerators multiples of the strides.
        const int od = aqd + (rd + pd - z * dd) / sd;
        const int oh = aqh + (rh + ph - y * dh) / sh;
        const int ow = aqw + (rw + pw - x * dw) / sw;
        const bool a_ok = am_ok && od >= 0 && od < Do && oh >= 0 && oh < Ho && ow >= 0 && ow < Wo;
        // Out-of-range rows read the clamped offset 0 and are replaced by 0 with a select, so
        // the loads do not branch.
        const long long a_sp = a_ok ? (static_cast<long long>(od) * Ho + oh) * Wo + ow : 0;
#if CONV_CHANNELS_LAST
        const data_t* a_ptr = dy_n + a_sp * K;
#else
        const data_t* a_ptr = dy_n + a_sp;
#endif
        const int tap = (z * Y + y) * X + x;
        for(int i = 0; i < LOAD_A; ++i)
        {
            const int k = k0 + a_col + i;
            ra[i] = (a_ok && k < Kg) ? static_cast<reg_t>(a_ptr[k * dy_kstride]) : reg_t(0);
        }
        for(int i = 0; i < LOAD_B; ++i)
        {
            const int k = k0 + b_col + i;
            if(bc_ok && k < Kg)
            {
#if CONV_CHANNELS_LAST
                const long long w_index = (static_cast<long long>(k) * taps + tap) * Cg + bc;
#else
                const long long w_index = (static_cast<long long>(k) * Cg + bc) * taps + tap;
#endif
                rb[i] = static_cast<reg_t>(wei_g[w_index]);
            }
            else
            {
                rb[i] = reg_t(0);
            }
        }
    };
    auto store = [&](int buf) {
        for(int i = 0; i < LOAD_A; ++i)
            as[buf][a_col + i][a_row] = to_float(static_cast<data_t>(ra[i]));
        for(int i = 0; i < LOAD_B; ++i)
            bs[buf][b_col + i][b_row] = to_float(static_cast<data_t>(rb[i]));
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

    if(iters > 0)
    {
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
    }

    for(int i = 0; i < MICRO_M; ++i)
    {
        const long long m = m0 + tm + i * (TILE_M / MICRO_M);
        if(m >= M)
            continue;
        long long t = m;
        const int qw = static_cast<int>(t % Wq);
        t /= Wq;
        const int qh = static_cast<int>(t % Hq);
        t /= Hq;
        const int qd = static_cast<int>(t % Dq);
        const long long n = t / Dq;
        const long long sp
            = (static_cast<long long>(qd * sd + rd) * Hi + (qh * sh + rh)) * Wi + (qw * sw + rw);
#if CONV_CHANNELS_LAST
        data_t* dx_m = dx + (n * sp_dx + sp) * C + static_cast<long long>(g) * Cg;
        const long long dx_cstride = 1;
#else
        data_t* dx_m = dx + (n * C + static_cast<long long>(g) * Cg) * sp_dx + sp;
        const long long dx_cstride = sp_dx;
#endif
        for(int j = 0; j < MICRO_N; ++j)
        {
            const int c = c0 + tn + j * (TILE_N / MICRO_N);
            if(c < Cg)
                dx_m[c * dx_cstride] = from_float(acc[i][j]);
        }
    }
}
