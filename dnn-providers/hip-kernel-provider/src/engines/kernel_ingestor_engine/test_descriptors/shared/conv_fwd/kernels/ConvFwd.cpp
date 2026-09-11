// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Reference kernel behind the conv-forward pack: naive direct convolution, one thread per
// output element. The matcher admits only stride 1, dilation 1, no padding, so
// p = h - r + 1 and q = width - s + 1 are computed here rather than passed in, for
// NCHW/KCRS/NKPQ layouts; the accumulator stays float regardless of HIP_PLUGIN_CONV_TYPE
// since a _Float16 accumulator loses too much precision against the CPU float reference.
// HIP_PLUGIN_CONV_BLOCK_SIZE is unused by the body but must reach the compiler for knob
// reporting.

#include <cstdint>

#include <hip/hip_runtime.h>

extern "C" __global__ void ConvFwd(const HIP_PLUGIN_CONV_TYPE* x,
                                   const HIP_PLUGIN_CONV_TYPE* w,
                                   HIP_PLUGIN_CONV_TYPE* y,
                                   int n,
                                   int c,
                                   int h,
                                   int width,
                                   int k,
                                   int r,
                                   int s)
{
    // int64_t: n*k*p*q can exceed 2^31 for shapes the matcher admits; 32-bit overflow here
    // would silently defeat the `index >= total` guard, causing an out-of-bounds access.
    const int64_t p = h - r + 1;
    const int64_t q = width - s + 1;
    const int64_t total = static_cast<int64_t>(n) * k * p * q;

    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index >= total)
    {
        return;
    }

    // Unravel the linear index over (n, k, p, q), q fastest -- the same order y is
    // stored in, so index also addresses y directly.
    int64_t remaining = index;
    const int64_t qOut = remaining % q;
    remaining /= q;
    const int64_t pOut = remaining % p;
    remaining /= p;
    const int64_t kOut = remaining % k;
    remaining /= k;
    const int64_t nOut = remaining;

    float accumulator = 0.0f;
    for(int cIn = 0; cIn < c; ++cIn)
    {
        for(int rIn = 0; rIn < r; ++rIn)
        {
            for(int sIn = 0; sIn < s; ++sIn)
            {
                const int64_t hIn = pOut + rIn;
                const int64_t wIn = qOut + sIn;
                const int64_t xIndex = ((nOut * c + cIn) * h + hIn) * width + wIn;
                const int64_t wIndex = ((kOut * c + cIn) * r + rIn) * s + sIn;
                accumulator += static_cast<float>(x[xIndex]) * static_cast<float>(w[wIndex]);
            }
        }
    }

    y[index] = static_cast<HIP_PLUGIN_CONV_TYPE>(accumulator);
}

// A second entry point that no descriptor names. It exists so this translation unit
// compiles to one code object carrying two symbols, which is the only way two symbols can
// share a toc_key -- the subject of TestKpackKernelLoader.TwoSymbolsResolveAgainstOneModule,
// which resolves both against a single key and asserts they came from one module.
// Deleting it as dead code deletes that test's only evidence.
//
// It must stay undescribed: a descriptor of its own would pack it under a second toc_key
// and into a second blob, which is the case the test exists to rule out.
extern "C" __global__ void ConvFwdSecondSymbol(const HIP_PLUGIN_CONV_TYPE* x,
                                               const HIP_PLUGIN_CONV_TYPE* w,
                                               HIP_PLUGIN_CONV_TYPE* y,
                                               int n,
                                               int c,
                                               int h,
                                               int width,
                                               int k,
                                               int r,
                                               int s)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        y[0] = static_cast<HIP_PLUGIN_CONV_TYPE>(
            static_cast<float>(x[0]) * static_cast<float>(w[0])
            * static_cast<float>(n + c + h + width + k + r + s));
    }
}
