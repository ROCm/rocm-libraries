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

// FP32 forward 1D convolution (NCHW with H == 1, filter height 1).
//
// y[n][g*Kg+k][wo] = sum_{c < Cg, r < KR} x[n][g*Cg+c][wo*STRIDE - pad + r*DIL] * w[g*Kg+k][c][r]
//
// Each 256-thread workgroup computes TILE_K output channels x TILE_W output positions. The input
// segment and the filter slice for TILE_C input channels are staged in LDS, and every thread
// accumulates a (TILE_K / 16) x (TILE_W / 16) register tile.
//
// Compile-time parameters (set by the solver): KR (filter width), STRIDE, DIL.
// Batch size, channels, groups, widths and padding are run-time arguments.

#ifndef KR
#define KR 3
#endif
#ifndef STRIDE
#define STRIDE 1
#endif
#ifndef DIL
#define DIL 1
#endif

#define TILE_K 64
#define TILE_W 128
#define TILE_C 8
#define MICRO_K (TILE_K / 16)
#define MICRO_W (TILE_W / 16)
#define SPAN ((TILE_W - 1) * STRIDE + (KR - 1) * DIL + 1)

extern "C" __global__ void __launch_bounds__(256) MIOpenConv1dFwdHipF32(const float* __restrict__ x,
                                                                        const float* __restrict__ w,
                                                                        float* __restrict__ y,
                                                                        int C,
                                                                        int Wi,
                                                                        int K,
                                                                        int Wo,
                                                                        int pad,
                                                                        int groups)
{
    __shared__ float xs[TILE_C * SPAN];
    __shared__ float ws[TILE_C * KR * TILE_K];

    const int Cg       = C / groups;
    const int Kg       = K / groups;
    const int kb_per_g = (Kg + TILE_K - 1) / TILE_K;
    const int n        = blockIdx.z;
    const int g        = blockIdx.y / kb_per_g;
    const int k0       = (blockIdx.y % kb_per_g) * TILE_K;
    const int w0       = blockIdx.x * TILE_W;
    const int tid      = threadIdx.x;
    const int tk       = tid / 16;
    const int tw       = tid % 16;

    const float* xg = x + (static_cast<size_t>(n) * C + static_cast<size_t>(g) * Cg) * Wi;
    const float* wg = w + static_cast<size_t>(g) * Kg * Cg * KR;
    const int in0   = w0 * STRIDE - pad;

    float acc[MICRO_K][MICRO_W];
#pragma unroll
    for(int i = 0; i < MICRO_K; ++i)
#pragma unroll
        for(int j = 0; j < MICRO_W; ++j)
            acc[i][j] = 0.f;

    for(int c0 = 0; c0 < Cg; c0 += TILE_C)
    {
        // Input segment, zero outside [0, Wi) and past the last channel.
        for(int idx = tid; idx < TILE_C * SPAN; idx += 256)
        {
            const int c  = idx / SPAN;
            const int wi = in0 + idx % SPAN;
            float v      = 0.f;
            if(c0 + c < Cg && wi >= 0 && wi < Wi)
                v = xg[static_cast<size_t>(c0 + c) * Wi + wi];
            xs[idx] = v;
        }
        // Filter slice, stored as [c * KR + r][k].
        for(int idx = tid; idx < TILE_C * KR * TILE_K; idx += 256)
        {
            const int k  = idx % TILE_K;
            const int cr = idx / TILE_K;
            const int c  = cr / KR;
            const int r  = cr % KR;
            float v      = 0.f;
            if(c0 + c < Cg && k0 + k < Kg)
                v = wg[(static_cast<size_t>(k0 + k) * Cg + (c0 + c)) * KR + r];
            ws[idx] = v;
        }
        __syncthreads();

        for(int c = 0; c < TILE_C; ++c)
        {
#pragma unroll
            for(int r = 0; r < KR; ++r)
            {
                float a[MICRO_K];
                float b[MICRO_W];
#pragma unroll
                for(int i = 0; i < MICRO_K; ++i)
                    a[i] = ws[(c * KR + r) * TILE_K + tk + 16 * i];
#pragma unroll
                for(int j = 0; j < MICRO_W; ++j)
                    b[j] = xs[c * SPAN + (tw + 16 * j) * STRIDE + r * DIL];
#pragma unroll
                for(int i = 0; i < MICRO_K; ++i)
#pragma unroll
                    for(int j = 0; j < MICRO_W; ++j)
                        acc[i][j] += a[i] * b[j];
            }
        }
        __syncthreads();
    }

    float* yg = y + (static_cast<size_t>(n) * K + static_cast<size_t>(g) * Kg) * Wo;
#pragma unroll
    for(int i = 0; i < MICRO_K; ++i)
    {
        const int k = k0 + tk + 16 * i;
        if(k >= Kg)
            continue;
#pragma unroll
        for(int j = 0; j < MICRO_W; ++j)
        {
            const int wo = w0 + tw + 16 * j;
            if(wo < Wo)
                yg[static_cast<size_t>(k) * Wo + wo] = acc[i][j];
        }
    }
}
