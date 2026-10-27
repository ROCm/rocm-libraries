// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <numeric>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormCommon.hpp"

// Reads the alignment a graph declares for its tensors (TensorAttributes::alignment()) into the
// terms the batchnorm kernel selection works in. See BatchnormCommon.hpp for what a declared
// alignment guarantees and how the kernels' accesses are checked against it.
namespace hip_kernel_provider::batchnorm
{

/**
 * Size in bytes of one element of a data type the batchnorm kernels are instantiated with, 0 for
 * any other type.
 */
inline size_t dataTypeBytes(hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    switch(type)
    {
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT:
        return FP32_BYTES;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::HALF:
    case hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16:
        return HALF_BYTES;
    default:
        return 0;
    }
}

/**
 * The alignment, in bytes, a tensor declares for its device pointer. A non-positive declaration
 * guarantees nothing and gives 0.
 */
inline size_t
    declaredAlignmentBytes(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor)
{
    return tensor.alignment() > 0 ? static_cast<size_t>(tensor.alignment()) : size_t{0};
}

/**
 * The alignment, in bytes, that every one of the tensors is guaranteed to have: the greatest common
 * divisor of their declarations, or 0 if any of them guarantees nothing. Null tensors are skipped,
 * and DEFAULT_TENSOR_ALIGNMENT_BYTES is returned when there is none.
 */
inline size_t commonDeclaredAlignmentBytes(
    std::initializer_list<const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*> tensors)
{
    size_t common = 0;
    for(const auto* tensor : tensors)
    {
        if(tensor == nullptr)
        {
            continue;
        }

        const size_t declared = declaredAlignmentBytes(*tensor);
        if(declared == 0)
        {
            return 0;
        }
        common = std::gcd(common, declared);
    }
    return common == 0 ? DEFAULT_TENSOR_ALIGNMENT_BYTES : common;
}

/**
 * The widest vector size (MAX_VECTOR_SIZE, then halving down to 1) whose width in bytes the
 * tensor's declared alignment covers, 0 when it does not even cover one element.
 */
inline size_t
    maxVectorSizeForAlignment(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor)
{
    const size_t declared = declaredAlignmentBytes(tensor);
    const size_t elementBytes = dataTypeBytes(tensor.data_type());
    for(size_t vectorsize = MAX_VECTOR_SIZE; vectorsize > 0; vectorsize >>= 1)
    {
        if(alignmentCovers(declared, vectorsize * elementBytes))
        {
            return vectorsize;
        }
    }
    return 0;
}

/**
 * The widest vector size the batchnorm forward inference kernels can use under the declared
 * alignments, 0 when they cannot run at all.
 *
 * x and the output are loaded and stored as vectors in both layouts
 * (kernels/batchnorm/BatchNormFwdInferSpatial.cpp:58-59,87-94). The per-channel tensors (scale,
 * bias, mean, variance) are loaded as vectors in NHWC (:141-148,230-237) and one element at a time
 * in NCHW (:152-155,241-244), where they only need to cover that one element.
 */
inline size_t maxInferenceVectorSize(
    bool isLayoutNHWC,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& x,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& output,
    std::initializer_list<const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>
        channelTensors)
{
    size_t limit = std::min(maxVectorSizeForAlignment(x), maxVectorSizeForAlignment(output));
    for(const auto* tensor : channelTensors)
    {
        const size_t channelLimit = maxVectorSizeForAlignment(*tensor);
        if(isLayoutNHWC)
        {
            limit = std::min(limit, channelLimit);
        }
        else if(channelLimit == 0)
        {
            limit = 0;
        }
    }
    return limit;
}

} // namespace hip_kernel_provider::batchnorm
