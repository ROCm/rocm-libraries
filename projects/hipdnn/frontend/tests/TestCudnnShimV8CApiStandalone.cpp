// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   samples/legacy_samples/test_list.cpp
//     Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

// cudnn_graph.h must come first: this TU proves it compiles standalone.
#include <hipdnn_compatibility/cudnn/cudnn_graph.h>

#include <gtest/gtest.h>

#include "fake_backend/MockHipdnnBackend.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace
{

class TestCudnnShimV8CApiStandalone : public ::testing::Test
{
protected:
    std::shared_ptr<::testing::NiceMock<Mock_hipdnn_backend>> _mockBackend;

    void SetUp() override
    {
        // Failure paths may fall back to the backend's last-error string.
        _mockBackend = std::make_shared<::testing::NiceMock<Mock_hipdnn_backend>>();
        hipdnn_frontend::detail::IHipdnnBackend::setInstance(_mockBackend);
    }

    void TearDown() override
    {
        hipdnn_frontend::detail::IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }
};

// Raw backend half of the legacy "Tensor creation comparison" case
// (test_list.cpp:68-86).
TEST_F(TestCudnnShimV8CApiStandalone, TensorCreationComparison)
{
    std::array<int64_t, 4> tensorDim = {4, 32, 32, 32};
    std::array<int64_t, 4> tensorStr = {32768, 1024, 32, 1}; // NCHW format
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    int64_t alignment = sizeof(float);
    int64_t id = 0xD0D0CACA;

    cudnnBackendDescriptor_t tensor;
    ASSERT_EQ(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensor),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendSetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dataType),
              CUDNN_STATUS_SUCCESS);
    // Upstream passes size_t counts; the casts keep -Wsign-conversion quiet.
    EXPECT_EQ(cudnnBackendSetAttribute(tensor,
                                       CUDNN_ATTR_TENSOR_DIMENSIONS,
                                       CUDNN_TYPE_INT64,
                                       static_cast<int64_t>(tensorDim.size()),
                                       tensorDim.data()),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendSetAttribute(tensor,
                                       CUDNN_ATTR_TENSOR_STRIDES,
                                       CUDNN_TYPE_INT64,
                                       static_cast<int64_t>(tensorStr.size()),
                                       tensorStr.data()),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(
        cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &id),
        CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendSetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendDestroyDescriptor(tensor), CUDNN_STATUS_SUCCESS);
}

} // namespace
