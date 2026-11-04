/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_runtime.h>
#endif

// Forward 3D convolution as an implicit GEMM on plain FMA (no WMMA / MFMA), so it runs on every
// RDNA target including the ones without matrix instructions (gfx101X, gfx103X).
//
// GEMM view, per group g:
//   M = N * Do * Ho * Wo (output positions), N_gemm = K / groups, K_gemm = Z * Y * X * C / groups
//   y[m][k] = sum_{z, y, x, c} x[n][di][hi][wi][c] * w[k][z][y][x][c]
//
// Each 256-thread workgroup computes a TILE_M x TILE_N output tile and every thread accumulates
// a 4x4 register tile in fp32. The K_gemm loop runs over filter taps (z, y, x) outside and input
// channels in TILE_C chunks inside, so the spatial bounds check is done once per tap. Inputs are
// converted to fp32 when they are staged in LDS.
//
// The weights are always read as K x (Z * Y * X) x (C / groups), so the channels of one filter tap
// are contiguous. That is the NDHWC weight layout as is. For NCDHW (K x (C / groups) x Z x Y x X)
// the solver first runs MIOpenConv3dFwdFmaHipWeiToKzyxc into the workspace; with a 1x1x1 filter
// both layouts coincide and the transpose is skipped.
//
// Compile-time parameters (set by the solver):
//   CONV3D_DTYPE  0 = fp32, 1 = fp16, 2 = bf16
//   CONV3D_NDHWC  0 = NCDHW input / NKDHW output, 1 = NDHWC input / NDHWK output

#ifndef CONV3D_DTYPE
#define CONV3D_DTYPE 0
#endif
#ifndef CONV3D_NDHWC
#define CONV3D_NDHWC 0
#endif

#define TILE_M 64
#define TILE_N 64
#define TILE_C 16
#define MICRO_M 4
#define MICRO_N 4
#define BLOCK_SIZE 256
// Each thread stages one row of the A (input) and B (weight) tiles, LOAD_C channels long.
#define LOAD_C (TILE_M * TILE_C / BLOCK_SIZE)

#if CONV3D_DTYPE == 0
typedef float data_t;
__device__ static inline float to_float(data_t v) { return v; }
__device__ static inline data_t from_float(float v) { return v; }
#elif CONV3D_DTYPE == 1
typedef _Float16 data_t;
__device__ static inline float to_float(data_t v) { return static_cast<float>(v); }
__device__ static inline data_t from_float(float v) { return static_cast<data_t>(v); }
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

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE)
    MIOpenConv3dFwdFmaHip(const data_t* __restrict__ in,
                          const data_t* __restrict__ wei,
                          data_t* __restrict__ out,
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
    __shared__ float as[TILE_C][TILE_M + 4];
    __shared__ float bs[TILE_C][TILE_N + 4];

    const int tid      = threadIdx.x;
    const int Cg       = C / groups;
    const int Kg       = K / groups;
    const int nb_per_g = (Kg + TILE_N - 1) / TILE_N;
    const int g        = blockIdx.y / nb_per_g;
    const int k0       = (blockIdx.y % nb_per_g) * TILE_N;
    const long M       = static_cast<long>(N) * Do * Ho * Wo;
    const long m0      = static_cast<long>(blockIdx.x) * TILE_M;
    const long sp_in   = static_cast<long>(Di) * Hi * Wi;
    const long sp_out  = static_cast<long>(Do) * Ho * Wo;
    const int taps     = Z * Y * X;

    // A row (output position) staged by this thread. NCDHW: consecutive threads take consecutive
    // positions so the loads coalesce along W; NDHWC: consecutive threads take consecutive
    // channels of one position.
#if CONV3D_NDHWC
    const int a_row = tid / (TILE_C / LOAD_C);
    const int a_col = (tid % (TILE_C / LOAD_C)) * LOAD_C;
#else
    const int a_row = tid % TILE_M;
    const int a_col = (tid / TILE_M) * LOAD_C;
#endif
    const long am    = m0 + a_row;
    const bool am_ok = am < M;
    int an = 0, aod = 0, aoh = 0, aow = 0;
    if(am_ok)
    {
        long t = am;
        aow    = static_cast<int>(t % Wo);
        t /= Wo;
        aoh = static_cast<int>(t % Ho);
        t /= Ho;
        aod = static_cast<int>(t % Do);
        an  = static_cast<int>(t / Do);
    }
    const int di0 = aod * sd - pd;
    const int hi0 = aoh * sh - ph;
    const int wi0 = aow * sw - pw;
#if CONV3D_NDHWC
    const data_t* in_n    = in + static_cast<long>(an) * sp_in * C + static_cast<long>(g) * Cg;
    const long in_cstride = 1;
#else
    const data_t* in_n    = in + (static_cast<long>(an) * C + static_cast<long>(g) * Cg) * sp_in;
    const long in_cstride = sp_in;
#endif

    // B row (output channel) staged by this thread.
    const int b_row  = tid / (TILE_C / LOAD_C);
    const int b_col  = (tid % (TILE_C / LOAD_C)) * LOAD_C;
    const int bk     = k0 + b_row;
    const bool bk_ok = bk < Kg;
    const data_t* wei_k =
        wei + (static_cast<long>(g) * Kg + (bk_ok ? bk : 0)) * static_cast<long>(taps) * Cg;

    // Register tile owned by this thread, strided by TILE / MICRO so that each store instruction
    // writes consecutive addresses. Consecutive threads step along the dimension that is
    // contiguous in the output: K for NDHWK, the output position for NKDHW.
#if CONV3D_NDHWC
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

    for(int z = 0; z < Z; ++z)
    {
        const int di = di0 + z * dd;
        for(int y = 0; y < Y; ++y)
        {
            const int hi = hi0 + y * dh;
            for(int x = 0; x < X; ++x)
            {
                const int wi = wi0 + x * dw;
                const bool a_ok =
                    am_ok && di >= 0 && di < Di && hi >= 0 && hi < Hi && wi >= 0 && wi < Wi;
                const long a_sp = a_ok ? (static_cast<long>(di) * Hi + hi) * Wi + wi : 0;
                const int tap   = (z * Y + y) * X + x;
#if CONV3D_NDHWC
                const data_t* a_ptr = in_n + a_sp * C;
#else
                const data_t* a_ptr = in_n + a_sp;
#endif
                const data_t* b_ptr = wei_k + static_cast<long>(tap) * Cg;
                for(int c0 = 0; c0 < Cg; c0 += TILE_C)
                {
                    for(int i = 0; i < LOAD_C; ++i)
                    {
                        const int c = c0 + a_col + i;
                        as[a_col + i][a_row] =
                            (a_ok && c < Cg) ? to_float(a_ptr[c * in_cstride]) : 0.f;
                    }
                    for(int i = 0; i < LOAD_C; ++i)
                    {
                        const int c          = c0 + b_col + i;
                        bs[b_col + i][b_row] = (bk_ok && c < Cg) ? to_float(b_ptr[c]) : 0.f;
                    }
                    __syncthreads();
                    for(int kk = 0; kk < TILE_C; ++kk)
                    {
                        float a[MICRO_M], b[MICRO_N];
                        for(int i = 0; i < MICRO_M; ++i)
                            a[i] = as[kk][tm + i * (TILE_M / MICRO_M)];
                        for(int j = 0; j < MICRO_N; ++j)
                            b[j] = bs[kk][tn + j * (TILE_N / MICRO_N)];
                        for(int i = 0; i < MICRO_M; ++i)
                            for(int j = 0; j < MICRO_N; ++j)
                                acc[i][j] = __builtin_fmaf(a[i], b[j], acc[i][j]);
                    }
                    __syncthreads();
                }
            }
        }
    }

    for(int i = 0; i < MICRO_M; ++i)
    {
        const long m = m0 + tm + i * (TILE_M / MICRO_M);
        if(m >= M)
            continue;
#if CONV3D_NDHWC
        data_t* out_m          = out + m * K + static_cast<long>(g) * Kg;
        const long out_kstride = 1;
#else
        const long n  = m / sp_out;
        data_t* out_m = out + (n * K + static_cast<long>(g) * Kg) * sp_out + (m - n * sp_out);
        const long out_kstride = sp_out;
#endif
        for(int j = 0; j < MICRO_N; ++j)
        {
            const int k = k0 + tn + j * (TILE_N / MICRO_N);
            if(k < Kg)
                out_m[k * out_kstride] = from_float(acc[i][j]);
        }
    }
}

// KCZYX -> KZYXC weight transpose for the NCDHW path. w is K x Cg x taps, wt is K x taps x Cg.
extern "C" __global__ void __launch_bounds__(BLOCK_SIZE) MIOpenConv3dFwdFmaHipWeiToKzyxc(
    const data_t* __restrict__ w, data_t* __restrict__ wt, int K, int Cg, int taps)
{
    const long total = static_cast<long>(K) * Cg * taps;
    for(long idx = static_cast<long>(blockIdx.x) * BLOCK_SIZE + threadIdx.x; idx < total;
        idx += static_cast<long>(gridDim.x) * BLOCK_SIZE)
    {
        const long t                = idx % taps;
        const long kc               = idx / taps;
        const long c                = kc % Cg;
        const long k                = kc / Cg;
        wt[(k * taps + t) * Cg + c] = w[idx];
    }
}
