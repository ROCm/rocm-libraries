// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Two kernels that hand a value to each other through a workspace buffer, so a test that
// launches them as the two launches of one multi-launch kernel can tell from the output
// alone that both ran, in order, against the same workspace.
//
// MultiLaunchProduce is the one kernel a descriptor names: the packer can pack a single-launch
// `hip` kernel and cannot pack a `sources` map, so the packed descriptor describes this symbol
// and the test builds the two-launch form from the archive. MultiLaunchConsume is named by no
// descriptor. It shares this translation unit so that both land in one code object under one
// toc_key, which is how the launches of one rocKE kernel are packed. Deleting it as dead code
// deletes the test's second launch.
//
// Both take the same argument list, because the packer records the signature of the described
// kernel only and the test reuses that list for the other.
//
//   MultiLaunchProduce:  ws[i] = 2 * x[i] + 1
//   MultiLaunchConsume:  y[i]  = 3 * ws[i] + x[i]
//
// The result 7 * x[i] + 3 differs from what any one kernel, the pair in the other order, or
// the pair reading some buffer other than the workspace would write.
//
// The include is explicit, as in the other packed hip fixtures: the packager drives clang
// directly with `-x hip`, which adds no runtime preamble.

#include <hip/hip_runtime.h>

extern "C" __global__ void MultiLaunchProduce(const float* x, float* ws, float* y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        ws[i] = 2.0f * x[i] + 1.0f;
    }
}

extern "C" __global__ void MultiLaunchConsume(const float* x, float* ws, float* y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        y[i] = 3.0f * ws[i] + x[i];
    }
}
