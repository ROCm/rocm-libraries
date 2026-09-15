// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
//
// BatchnormInference for a hipDNN `BatchnormInferenceAttributes` node.
//
//   y[n,c,h,w] = scale[c] * ((x[n,c,h,w] - mean[c]) * inv_variance[c]) + bias[c]
//
// The multiply/add order is the CPU reference's, verbatim
// (test_sdk/include/hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp:38-55):
// subtract the mean, multiply by inv_variance, multiply by scale, add bias. No
// epsilon: `BatchnormInferenceAttributes` has no epsilon field and the reference
// consumes inv_variance with epsilon already baked in. Applying one here would be a
// silent numeric defect.
//
// Indexing is through the graph's strides, never an assumed packed layout: layout is
// not a schema field, it exists only as the stride pattern. The per-channel operands
// (dims [1,C,1,...]) are addressed as `base + c * stride1` because that is what the
// reference's `getHostValue(0, c)` resolves to
// (data_sdk/include/hipdnn_data_sdk/utilities/Tensor.hpp:562-565).
//
// `__restrict__` on y is a promise the graph must keep, and the promise is enforced
// outside this file: packs/BatchnormInferenceNative.cpp's graph_match refuses any graph
// whose y uid equals one of the five input uids. y is the only pointer written through,
// so it is the only aliasing that is undefined behaviour here -- two const inputs naming
// one tensor is a read/read overlap, which `restrict` permits. An in-place batchnorm is
// therefore declined rather than miscompiled, and every other graph keeps the alias
// information on the x load and the y store.

#include <cstdint>

#include "BatchnormInferenceTypes.h"

extern "C" __global__ void __launch_bounds__(HIPDNN_BN_BLOCK) BatchnormInference(
    const BnIoElement* __restrict__ x,
    const BnParamElement* __restrict__ mean,
    const BnParamElement* __restrict__ invVariance,
    const BnParamElement* __restrict__ scale,
    const BnParamElement* __restrict__ bias,
    BnIoElement* __restrict__ y,
    // Logical extents of x (and therefore of y). Channel is dim 1, as the frontend
    // node hard-codes (BatchnormInferenceNode.hpp:110).
    int64_t dimN,
    int64_t dimC,
    int64_t dimH,
    int64_t dimW,
    // Element strides of x, per dimension. Arbitrary: packed NCHW, packed NHWC and
    // non-packed patterns are all just different values here.
    int64_t xStrideN,
    int64_t xStrideC,
    int64_t xStrideH,
    int64_t xStrideW,
    // Element strides of y. Independent of x's: the frontend only defaults y's
    // strides to x's when they were left unset.
    int64_t yStrideN,
    int64_t yStrideC,
    int64_t yStrideH,
    int64_t yStrideW,
    // Channel stride of each per-channel operand, i.e. its own strides[1].
    int64_t meanStrideC,
    int64_t invVarianceStrideC,
    int64_t scaleStrideC,
    int64_t biasStrideC)
{
    const int64_t total = dimN * dimC * dimH * dimW;

    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index >= total)
    {
        return;
    }

    // Unravel against the logical dims, then address through the strides. Doing it in
    // this order is what makes the kernel layout-agnostic: the iteration space is the
    // shape, the memory walk is the strides.
    int64_t remaining = index;
    const int64_t w = remaining % dimW;
    remaining /= dimW;
    const int64_t h = remaining % dimH;
    remaining /= dimH;
    const int64_t c = remaining % dimC;
    remaining /= dimC;
    const int64_t n = remaining;

    const int64_t xOffset = n * xStrideN + c * xStrideC + h * xStrideH + w * xStrideW;
    const int64_t yOffset = n * yStrideN + c * yStrideC + h * yStrideH + w * yStrideW;

    const BnCompute meanValue = static_cast<BnCompute>(mean[c * meanStrideC]);
    const BnCompute invVarValue = static_cast<BnCompute>(invVariance[c * invVarianceStrideC]);
    const BnCompute scaleValue = static_cast<BnCompute>(scale[c * scaleStrideC]);
    const BnCompute biasValue = static_cast<BnCompute>(bias[c * biasStrideC]);

    const BnCompute inputValue = static_cast<BnCompute>(x[xOffset]);
    const BnCompute normalized = (inputValue - meanValue) * invVarValue;

    y[yOffset] = static_cast<BnIoElement>(scaleValue * normalized + biasValue);
}
