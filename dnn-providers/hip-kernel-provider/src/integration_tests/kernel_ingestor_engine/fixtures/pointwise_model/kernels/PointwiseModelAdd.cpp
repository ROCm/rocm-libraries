// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The pointwise_model pack's kernel. Identical in body to the unit set's PointwiseAdd, and
// deliberately a separate file: a packed root is validated whole, so an embedded source has
// to be authored inside the root that names it and registered under a key no other root
// claims. Borrowing the unit set's file fails that check from both ends -- the source sits
// outside this root, and its key already belongs to another descriptor.

extern "C" __global__ void PointwiseModelAdd(const HIP_PLUGIN_POINTWISE_TYPE* a,
                                             const HIP_PLUGIN_POINTWISE_TYPE* b,
                                             HIP_PLUGIN_POINTWISE_TYPE* c)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        c[0] = a[0] + b[0];
    }
}
