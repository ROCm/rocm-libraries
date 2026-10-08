// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Same body as the unit set's PointwiseAdd, but a packed root's embedded sources must live
// inside that root under a key no other root claims, so it cannot borrow that file.

extern "C" __global__ void PointwiseModelAdd(const HIP_PLUGIN_POINTWISE_TYPE* a,
                                             const HIP_PLUGIN_POINTWISE_TYPE* b,
                                             HIP_PLUGIN_POINTWISE_TYPE* c)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        c[0] = a[0] + b[0];
    }
}
