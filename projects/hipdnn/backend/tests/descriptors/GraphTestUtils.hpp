// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "DescriptorTestUtils.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "hipdnn_backend.h"
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>

namespace hipdnn_backend::test_utilities
{

/// Bundles the tensor and operation descriptors for a single convolution forward operation.
/// Owns all descriptors via unique_ptr to ensure cleanup on scope exit.
struct ConvOpBundle
{
    std::unique_ptr<HipdnnBackendDescriptor> xDesc;
    std::unique_ptr<HipdnnBackendDescriptor> wDesc;
    std::unique_ptr<HipdnnBackendDescriptor> yDesc;
    std::unique_ptr<HipdnnBackendDescriptor> convOp;
};

/// Creates a ConvOpBundle with default X/W/Y tensors and a finalized convolution forward
/// operation using shared test constants from ConvFpropConstants.hpp.
inline ConvOpBundle createDefaultConvOp(hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    ConvOpBundle bundle;
    bundle.xDesc = createFinalizedTensor(hipdnn_tests::constants::K_FPROP_TENSOR_X_UID);
    bundle.wDesc = createFinalizedTensor(
        hipdnn_tests::constants::K_FPROP_TENSOR_W_UID,
        hipdnn_tests::toVec(hipdnn_tests::constants::K_FPROP_TENSOR_W_DIMS),
        hipdnn_tests::toVec(hipdnn_tests::constants::K_FPROP_TENSOR_W_STRIDES));
    bundle.yDesc = createFinalizedTensor(
        hipdnn_tests::constants::K_FPROP_TENSOR_Y_UID,
        hipdnn_tests::toVec(hipdnn_tests::constants::K_FPROP_TENSOR_Y_DIMS),
        hipdnn_tests::toVec(hipdnn_tests::constants::K_FPROP_TENSOR_Y_STRIDES));
    bundle.convOp = createFinalizedConvOp(
        bundle.xDesc.get(), bundle.wDesc.get(), bundle.yDesc.get(), computeType);
    return bundle;
}

/// Finds a tensor in a GraphT by UID, returns nullptr if not found.
inline const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT*
    findTensorByUid(const hipdnn_flatbuffers_sdk::data_objects::GraphT& graphT, int64_t uid)
{
    for(const auto& tensor : graphT.tensors)
    {
        if(tensor->uid == uid)
        {
            return tensor.get();
        }
    }
    return nullptr;
}

} // namespace hipdnn_backend::test_utilities
