// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Reference kernel behind the conv-forward descriptor pack: naive direct convolution,
// one thread per output element.
//
// HIP_PLUGIN_CONV_TYPE is the element type from the kernel descriptor's dtype metadata.
// HIP_PLUGIN_CONV_BLOCK_SIZE is the descriptor's block size; unused by the kernel body,
// which reads its launch geometry from blockDim/blockIdx instead, but must reach the
// compiler for ranking and knob reporting, matching PointwiseAdd.cpp's two macros.
//
// Narrow on purpose: the matcher admits only stride 1, dilation 1, no padding, so
// p = h - r + 1 and q = width - s + 1 are computed here rather than passed in. x is
// packed NCHW, w is packed KCRS, y is packed NKPQ -- the matcher rejects anything else,
// so the kernel may assume all of it. The accumulator is always float regardless of
// HIP_PLUGIN_CONV_TYPE: a _Float16 accumulator loses too much precision for a reference
// a CPU float reference is compared against.

extern "C" __global__ void ConvFwd(const HIP_PLUGIN_CONV_TYPE* x,
                                   const HIP_PLUGIN_CONV_TYPE* w,
                                   HIP_PLUGIN_CONV_TYPE* y,
                                   int n, int c, int h, int width, int k, int r, int s)
{
    const int p = h - r + 1;
    const int q = width - s + 1;
    const int total = n * k * p * q;

    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if(index >= total)
    {
        return;
    }

    // Unravel the linear index over (n, k, p, q), q fastest -- the same order y is
    // stored in, so index also addresses y directly.
    int remaining = index;
    const int qOut = remaining % q;
    remaining /= q;
    const int pOut = remaining % p;
    remaining /= p;
    const int kOut = remaining % k;
    remaining /= k;
    const int nOut = remaining;

    float accumulator = 0.0f;
    for(int cIn = 0; cIn < c; ++cIn)
    {
        for(int rIn = 0; rIn < r; ++rIn)
        {
            for(int sIn = 0; sIn < s; ++sIn)
            {
                const int hIn = pOut + rIn;
                const int wIn = qOut + sIn;
                const int xIndex = ((nOut * c + cIn) * h + hIn) * width + wIn;
                const int wIndex = ((kOut * c + cIn) * r + rIn) * s + sIn;
                accumulator += static_cast<float>(x[xIndex]) * static_cast<float>(w[wIndex]);
            }
        }
    }

    y[index] = static_cast<HIP_PLUGIN_CONV_TYPE>(accumulator);
}
