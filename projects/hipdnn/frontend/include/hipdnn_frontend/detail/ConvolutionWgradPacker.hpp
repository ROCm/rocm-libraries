// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a convolutionwrw operation descriptor from ConvWgradAttributes.
// Tensor descriptors are created/deduplicated via createOrFindTensorDesc.
inline Error createConvWgradOperation(
    const graph::ConvWgradAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Ensure tensor descriptors exist for X, DY, DW
    auto [errX, xUid] = createOrFindTensorDesc(tensorDescs, attributes.get_x());
    HIPDNN_CHECK_ERROR(errX);
    auto [errDY, dyUid] = createOrFindTensorDesc(tensorDescs, attributes.get_dy());
    HIPDNN_CHECK_ERROR(errDY);
    auto [errDW, dwUid] = createOrFindTensorDesc(tensorDescs, attributes.get_dw());
    HIPDNN_CHECK_ERROR(errDW);

    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_CONVOLUTION_WRW_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create convolutionwrw operation descriptor"};
    }

    // Set tensor references
    HIPDNN_CHECK_ERROR(setDescriptorAttrTensorRef(opDesc.get(),
                                                  HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_X,
                                                  xUid,
                                                  tensorDescs,
                                                  "convolutionwrw X"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrTensorRef(opDesc.get(),
                                                  HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_DY,
                                                  dyUid,
                                                  tensorDescs,
                                                  "convolutionwrw DY"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrTensorRef(opDesc.get(),
                                                  HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_DW,
                                                  dwUid,
                                                  tensorDescs,
                                                  "convolutionwrw DW"));

    // Set convolutionwrw parameters
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_pre_padding(),
                                            "convolutionwrw pre_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_post_padding(),
                                            "convolutionwrw post_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_stride(),
                                            "convolutionwrw stride"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_dilation(),
                                            "convolutionwrw dilation"));

    // Set convolutionwrw mode and compute data type
    auto convMode
        = static_cast<int64_t>(hipdnn_frontend::toSdkType(attributes.get_convolution_mode()));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                                               HIPDNN_TYPE_INT64,
                                               convMode,
                                               "convolutionwrw mode"));

    auto computeDataType = hipdnn_frontend::toSdkType(attributes.compute_data_type);
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_CONVOLUTION_COMP_TYPE,
                                               HIPDNN_TYPE_DATA_TYPE,
                                               computeDataType,
                                               "convolutionwrw compute data type"));

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "convolutionwrw operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
