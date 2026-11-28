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
// a MICRO_M x MICRO_N register tile in fp32. One K_gemm iteration covers one filter tap (z, y, x)
// and TILE_C input channels, so the spatial bounds check is done once per tap. Inputs are
// converted to fp32 when they are staged in LDS. The global loads for the next iteration are
// issued before the FMAs of the current one and land in the other half of a double-buffered LDS
// tile, so each iteration needs one barrier.
//
// By default the weights are read as K x (Z * Y * X) x (C / groups), so the channels of one filter
// tap are contiguous. That is the NDHWC weight layout as is. NCDHW weights are
// K x (C / groups) x (Z * Y * X): the 3D solver first transposes them into the workspace with
// MIOpenConv3dFwdFmaHipWeiToKzyxc (with a 1x1x1 filter both layouts coincide and the transpose is
// skipped). With CONV3D_WEI_KCZYX the kernel reads the NCDHW weights in place instead (the channels
// of one tap are `taps` apart), so no workspace is needed; the 2D solver
// (ConvHipImplicitGemmFma2dFwd) runs the kernel that way with depth 1.
//
// Compile-time parameters (set by the solver):
//   CONV3D_DTYPE      0 = fp32, 1 = fp16, 2 = bf16
//   CONV3D_NDHWC      0 = NCDHW input / NKDHW output, 1 = NDHWC input / NDHWK output
//   CONV3D_WEI_KCZYX  0 = weights K x taps x (C / groups), 1 = K x (C / groups) x taps
//   TILE_N            output channels per workgroup: 128, 64 or 32 (MICRO_N = TILE_N / 16)

#ifndef CONV3D_DTYPE
#define CONV3D_DTYPE 0
#endif
#ifndef CONV3D_NDHWC
#define CONV3D_NDHWC 0
#endif
#ifndef CONV3D_WEI_KCZYX
#define CONV3D_WEI_KCZYX 0
#endif
#ifndef TILE_N
#define TILE_N 128
#endif

#define TILE_M 128
#define TILE_C 16
#define MICRO_M 8
#define MICRO_N (TILE_N / 16)
#define BLOCK_SIZE 256
// Each thread stages one row of the A (input) and B (weight) tiles, LOAD_A / LOAD_B channels long.
#define LOAD_A (TILE_M * TILE_C / BLOCK_SIZE)
#define LOAD_B (TILE_N * TILE_C / BLOCK_SIZE)

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
    __shared__ float as[2][TILE_C][TILE_M + 4];
    __shared__ float bs[2][TILE_C][TILE_N + 4];

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
    const int a_row = tid / (TILE_C / LOAD_A);
    const int a_col = (tid % (TILE_C / LOAD_A)) * LOAD_A;
#else
    const int a_row = tid % TILE_M;
    const int a_col = (tid / TILE_M) * LOAD_A;
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
    const int b_row  = tid / (TILE_C / LOAD_B);
    const int b_col  = (tid % (TILE_C / LOAD_B)) * LOAD_B;
    const int bk     = k0 + b_row;
    const bool bk_ok = bk < Kg;
    // Both weight layouts hold taps * Cg values per output channel; they differ only inside it.
    const data_t* wei_k =
        wei + (static_cast<long>(g) * Kg + (bk_ok ? bk : 0)) * static_cast<long>(taps) * Cg;
#if CONV3D_WEI_KCZYX
    const long wei_cstride = taps;
#else
    const long wei_cstride = 1;
#endif

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

    // Flattened K loop: iteration `it` covers filter tap it / nc and channels
    // [(it % nc) * TILE_C, + TILE_C).
    const int nc    = (Cg + TILE_C - 1) / TILE_C;
    const int iters = taps * nc;
    const int yx    = Y * X;
    float ra[LOAD_A], rb[LOAD_B];

    auto load = [&](int it) {
        const int tap   = it / nc;
        const int c0    = (it - tap * nc) * TILE_C;
        const int z     = tap / yx;
        const int r     = tap - z * yx;
        const int y     = r / X;
        const int x     = r - y * X;
        const int di    = di0 + z * dd;
        const int hi    = hi0 + y * dh;
        const int wi    = wi0 + x * dw;
        const bool a_ok = am_ok && di >= 0 && di < Di && hi >= 0 && hi < Hi && wi >= 0 && wi < Wi;
        // Out-of-range rows read the clamped offset 0 and are replaced by 0 with a select, so
        // the loads do not branch.
        const long a_sp = a_ok ? (static_cast<long>(di) * Hi + hi) * Wi + wi : 0;
#if CONV3D_NDHWC
        const data_t* a_ptr = in_n + a_sp * C;
#else
        const data_t* a_ptr = in_n + a_sp;
#endif
#if CONV3D_WEI_KCZYX
        const data_t* b_ptr = wei_k + tap;
#else
        const data_t* b_ptr = wei_k + static_cast<long>(tap) * Cg;
#endif
        if(c0 + TILE_C <= Cg)
        {
            for(int i = 0; i < LOAD_A; ++i)
            {
                const float v = to_float(a_ptr[(c0 + a_col + i) * in_cstride]);
                ra[i]         = a_ok ? v : 0.f;
            }
            for(int i = 0; i < LOAD_B; ++i)
            {
                const float v = to_float(b_ptr[(c0 + b_col + i) * wei_cstride]);
                rb[i]         = bk_ok ? v : 0.f;
            }
            return;
        }
        // Partial channel chunk (wave-uniform).
        for(int i = 0; i < LOAD_A; ++i)
        {
            const int c = c0 + a_col + i;
            ra[i]       = (a_ok && c < Cg) ? to_float(a_ptr[c * in_cstride]) : 0.f;
        }
        for(int i = 0; i < LOAD_B; ++i)
        {
            const int c = c0 + b_col + i;
            rb[i]       = (bk_ok && c < Cg) ? to_float(b_ptr[c * wei_cstride]) : 0.f;
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
