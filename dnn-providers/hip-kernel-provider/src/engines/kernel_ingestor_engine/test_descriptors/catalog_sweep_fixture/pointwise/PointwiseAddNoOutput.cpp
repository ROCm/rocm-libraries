// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// TEST-ONLY deliberately broken variant of PointwiseAdd.cpp: the same entry-point signature,
// but it writes nothing to `c`. It exists so a catalog sweep has a kernel that builds, loads
// and launches cleanly yet leaves its output untouched, which the sweep must report as a
// numeric failure rather than a pass. This set ships only in the test descriptor roots, never
// in the production descriptors.
//
// HIP_PLUGIN_POINTWISE_BLOCK_SIZE is unused for the reason PointwiseAdd.cpp gives.

#include <hip/hip_runtime.h>

extern "C" __global__ void PointwiseAddNoOutput(const HIP_PLUGIN_POINTWISE_TYPE* /*a*/,
                                                const HIP_PLUGIN_POINTWISE_TYPE* /*b*/,
                                                HIP_PLUGIN_POINTWISE_TYPE* /*c*/)
{
}
