// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU kernel for naive 2D forward convolution (cross-correlation).
// Compiled at runtime via HIPRTC and launched by ConvFwdPlan.
// Each thread computes one output element.

#include "IndexType.hpp"

extern "C" __global__ void conv_forward_naive_kernel(const float* input,
                                                     const float* weight,
                                                     float* output,
                                                     int N,
                                                     int C,
                                                     int H,
                                                     int W,
                                                     int K,
                                                     int R,
                                                     int S,
                                                     int outH,
                                                     int outW,
                                                     int padH,
                                                     int padW,
                                                     int strideH,
                                                     int strideW,
                                                     unsigned long long spinNs,
                                                     unsigned long long clockMHz)
{
    IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType totalOutputElements = static_cast<IndexType>(N) * K * outH * outW;

    if(idx >= totalOutputElements)
    {
        return;
    }

    // Decompose linear index into (n, k, oh, ow)
    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int kk = (idx / (static_cast<IndexType>(outW) * outH)) % K;
    int nn = idx / (static_cast<IndexType>(outW) * outH * K);

    float sum = 0.0f;
    for(int cc = 0; cc < C; ++cc)
    {
        for(int rr = 0; rr < R; ++rr)
        {
            for(int ss = 0; ss < S; ++ss)
            {
                int ih = oh * strideH - padH + rr;
                int iw = ow * strideW - padW + ss;

                if(ih >= 0 && ih < H && iw >= 0 && iw < W)
                {
                    // Input: NCHW layout
                    int inputIdx = ((nn * C + cc) * H + ih) * W + iw;
                    // Weight: KCRS layout
                    int weightIdx = ((kk * C + cc) * R + rr) * S + ss;
                    sum += input[inputIdx] * weight[weightIdx];
                }
            }
        }
    }

    // Output: NKHW layout (same as NCHW with K output channels)
    output[idx] = sum;

    // Real convolution result is written; delay completion by a fixed wall-clock duration so the
    // autotune framework measures a deterministic per-engine GPU time. wall_clock64() advances at
    // the GPU shader clock, so convert the requested nanoseconds to that clock's cycles.
    unsigned long long spinCycles = spinNs * clockMHz / 1000ULL;
    unsigned long long start = wall_clock64();
    while(wall_clock64() - start < spinCycles)
    {
    }
}
