// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BackendTestHelpers.hpp"
#include "hipdnn_backend.h"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_common_generated.h>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <test_plugins/TestPluginConstants.hpp>
#include <vector>

using namespace backend_test;
using namespace hipdnn_tests::constants;
class IntegrationConvOperationDescriptorApi : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        for(auto* desc : _descriptors)
        {
            hipdnnBackendDestroyDescriptor(desc);
        }
    }

    hipdnnBackendDescriptor_t createAndTrackTensor(int64_t uid,
                                                   const char* name,
                                                   const std::vector<int64_t>& dims,
                                                   const std::vector<int64_t>& strides)
    {
        auto desc = createAndFinalizeTensorDesc(uid, name, dims, strides);
        _descriptors.push_back(desc);
        return desc;
    }

    std::vector<hipdnnBackendDescriptor_t> _descriptors;
};

TEST_F(IntegrationConvOperationDescriptorApi, CreateAndFinalizeConvOperation)
{
    auto xDesc = createAndTrackTensor(
        K_FPROP_TENSOR_X_UID, "X", toVec(K_FPROP_TENSOR_X_DIMS), toVec(K_FPROP_TENSOR_X_STRIDES));
    auto wDesc = createAndTrackTensor(
        K_FPROP_TENSOR_W_UID, "W", toVec(K_FPROP_TENSOR_W_DIMS), toVec(K_FPROP_TENSOR_W_STRIDES));
    auto yDesc = createAndTrackTensor(
        K_FPROP_TENSOR_Y_UID, "Y", toVec(K_FPROP_TENSOR_Y_DIMS), toVec(K_FPROP_TENSOR_Y_STRIDES));

    hipdnnBackendDescriptor_t opDesc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                            &opDesc),
              HIPDNN_STATUS_SUCCESS);
    _descriptors.push_back(opDesc);

    // Set tensor references
    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&xDesc)),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&wDesc)),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&yDesc)),
              HIPDNN_STATUS_SUCCESS);

    // Set convolution parameters
    auto padding = toVec(K_FPROP_CONV_PADDING);
    auto stride = toVec(K_FPROP_CONV_STRIDE);
    auto dilation = toVec(K_FPROP_CONV_DILATION);

    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(padding.size()),
                                        padding.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(padding.size()),
                                        padding.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(stride.size()),
                                        stride.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(dilation.size()),
                                        dilation.data()),
              HIPDNN_STATUS_SUCCESS);

    hipdnnConvolutionMode_t convMode = HIPDNN_CROSS_CORRELATION;
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            opDesc, HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &convMode),
        HIPDNN_STATUS_SUCCESS);

    auto computeType = HIPDNN_DATA_FLOAT;
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            opDesc, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType),
        HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(opDesc), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationConvOperationDescriptorApi, GetAfterSetVerifiesAllAttributes)
{
    auto xDesc = createAndTrackTensor(
        K_FPROP_TENSOR_X_UID, "X", toVec(K_FPROP_TENSOR_X_DIMS), toVec(K_FPROP_TENSOR_X_STRIDES));
    auto wDesc = createAndTrackTensor(
        K_FPROP_TENSOR_W_UID, "W", toVec(K_FPROP_TENSOR_W_DIMS), toVec(K_FPROP_TENSOR_W_STRIDES));
    auto yDesc = createAndTrackTensor(
        K_FPROP_TENSOR_Y_UID, "Y", toVec(K_FPROP_TENSOR_Y_DIMS), toVec(K_FPROP_TENSOR_Y_STRIDES));

    hipdnnBackendDescriptor_t opDesc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                            &opDesc),
              HIPDNN_STATUS_SUCCESS);
    _descriptors.push_back(opDesc);

    // Set tensor references
    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&xDesc)),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&wDesc)),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&yDesc)),
              HIPDNN_STATUS_SUCCESS);

    // Set convolution parameters
    auto padding = toVec(K_FPROP_CONV_PADDING);
    auto stride = toVec(K_FPROP_CONV_STRIDE);
    auto dilation = toVec(K_FPROP_CONV_DILATION);

    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(padding.size()),
                                        padding.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(padding.size()),
                                        padding.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(stride.size()),
                                        stride.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(dilation.size()),
                                        dilation.data()),
              HIPDNN_STATUS_SUCCESS);

    hipdnnConvolutionMode_t convMode = HIPDNN_CROSS_CORRELATION;
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            opDesc, HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &convMode),
        HIPDNN_STATUS_SUCCESS);

    auto computeType = HIPDNN_DATA_FLOAT;
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            opDesc, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendFinalize(opDesc), HIPDNN_STATUS_SUCCESS);

    // Verify tensor descriptor references via getAttribute
    int64_t elementCount = 0;
    hipdnnBackendDescriptor_t retrievedDesc = nullptr;

    EXPECT_EQ(hipdnnBackendGetAttribute(opDesc,
                                        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &elementCount,
                                        static_cast<void*>(&retrievedDesc)),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(elementCount, 1);
    EXPECT_NE(retrievedDesc, nullptr);
    _descriptors.push_back(retrievedDesc);

    elementCount = 0;
    retrievedDesc = nullptr;
    EXPECT_EQ(hipdnnBackendGetAttribute(opDesc,
                                        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &elementCount,
                                        static_cast<void*>(&retrievedDesc)),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(elementCount, 1);
    EXPECT_NE(retrievedDesc, nullptr);
    _descriptors.push_back(retrievedDesc);

    elementCount = 0;
    retrievedDesc = nullptr;
    EXPECT_EQ(hipdnnBackendGetAttribute(opDesc,
                                        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &elementCount,
                                        static_cast<void*>(&retrievedDesc)),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(elementCount, 1);
    EXPECT_NE(retrievedDesc, nullptr);
    _descriptors.push_back(retrievedDesc);

    // Verify convolution parameter arrays
    std::vector<int64_t> retrievedPadding(2, 0);
    elementCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(opDesc,
                                        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                        HIPDNN_TYPE_INT64,
                                        2,
                                        &elementCount,
                                        retrievedPadding.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(elementCount, 2);
    EXPECT_EQ(retrievedPadding, padding);

    std::vector<int64_t> retrievedPostPadding(2, 0);
    elementCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(opDesc,
                                        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                        HIPDNN_TYPE_INT64,
                                        2,
                                        &elementCount,
                                        retrievedPostPadding.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(elementCount, 2);
    EXPECT_EQ(retrievedPostPadding, padding);

    std::vector<int64_t> retrievedStride(2, 0);
    elementCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(opDesc,
                                        HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                        HIPDNN_TYPE_INT64,
                                        2,
                                        &elementCount,
                                        retrievedStride.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(elementCount, 2);
    EXPECT_EQ(retrievedStride, stride);

    std::vector<int64_t> retrievedDilation(2, 0);
    elementCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(opDesc,
                                        HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                                        HIPDNN_TYPE_INT64,
                                        2,
                                        &elementCount,
                                        retrievedDilation.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(elementCount, 2);
    EXPECT_EQ(retrievedDilation, dilation);

    // Verify conv mode
    hipdnnConvolutionMode_t retrievedConvMode = {};
    elementCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(opDesc,
                                        HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                                        HIPDNN_TYPE_CONVOLUTION_MODE,
                                        1,
                                        &elementCount,
                                        &retrievedConvMode),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(elementCount, 1);
    EXPECT_EQ(retrievedConvMode, HIPDNN_CROSS_CORRELATION);

    // Verify compute type
    hipdnnDataType_t retrievedCompType = {};
    elementCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(opDesc,
                                        HIPDNN_ATTR_CONVOLUTION_COMP_TYPE,
                                        HIPDNN_TYPE_DATA_TYPE,
                                        1,
                                        &elementCount,
                                        &retrievedCompType),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(elementCount, 1);
    EXPECT_EQ(retrievedCompType, HIPDNN_DATA_FLOAT);
}

TEST_F(IntegrationConvOperationDescriptorApi, ConvOperationFailsWithoutTensorRefs)
{
    hipdnnBackendDescriptor_t opDesc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                            &opDesc),
              HIPDNN_STATUS_SUCCESS);
    _descriptors.push_back(opDesc);

    // Set conv params but omit tensor references
    auto padding = toVec(K_FPROP_CONV_PADDING);
    auto stride = toVec(K_FPROP_CONV_STRIDE);
    auto dilation = toVec(K_FPROP_CONV_DILATION);

    hipdnnBackendSetAttribute(opDesc,
                              HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                              HIPDNN_TYPE_INT64,
                              static_cast<int64_t>(padding.size()),
                              padding.data());
    hipdnnBackendSetAttribute(opDesc,
                              HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                              HIPDNN_TYPE_INT64,
                              static_cast<int64_t>(padding.size()),
                              padding.data());
    hipdnnBackendSetAttribute(opDesc,
                              HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                              HIPDNN_TYPE_INT64,
                              static_cast<int64_t>(stride.size()),
                              stride.data());
    hipdnnBackendSetAttribute(opDesc,
                              HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                              HIPDNN_TYPE_INT64,
                              static_cast<int64_t>(dilation.size()),
                              dilation.data());

    hipdnnConvolutionMode_t convMode = HIPDNN_CROSS_CORRELATION;
    hipdnnBackendSetAttribute(
        opDesc, HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &convMode);

    auto computeType = HIPDNN_DATA_FLOAT;
    hipdnnBackendSetAttribute(
        opDesc, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    EXPECT_NE(hipdnnBackendFinalize(opDesc), HIPDNN_STATUS_SUCCESS);
}
