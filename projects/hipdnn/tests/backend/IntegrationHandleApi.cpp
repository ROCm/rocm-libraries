// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>

TEST(IntegrationHandleApi, CreateAndDestroy)
{
    hipdnnHandle_t handle = nullptr;

    auto createStatus = hipdnnCreate(&handle);
    ASSERT_EQ(createStatus, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto destroyStatus = hipdnnDestroy(handle);
    ASSERT_EQ(destroyStatus, HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationHandleApi, CreateWithNullptr)
{
    hipdnnStatus_t status = hipdnnCreate(nullptr);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(IntegrationHandleApi, SetStreamNullptrHandle)
{
    hipStream_t testStream = nullptr;
    auto setStreamStatus = hipdnnSetStream(nullptr, testStream);
    ASSERT_EQ(setStreamStatus, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(IntegrationHandleApi, GetStreamNullptrHandle)
{
    hipStream_t retrievedStream = nullptr;
    auto getStreamStatus = hipdnnGetStream(nullptr, &retrievedStream);
    ASSERT_EQ(getStreamStatus, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(IntegrationHandleApi, GetStreamNullptrStreamPointer)
{
    hipdnnHandle_t handle = nullptr;

    auto createStatus = hipdnnCreate(&handle);
    ASSERT_EQ(createStatus, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    hipStream_t testStream = nullptr;
    auto getStreamStatus = hipdnnGetStream(handle, &testStream);
    ASSERT_EQ(getStreamStatus, HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(testStream, nullptr);

    auto destroyStatus = hipdnnDestroy(handle);
    ASSERT_EQ(destroyStatus, HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationGpuHandleApi, GetStreamPointer)
{
    SKIP_IF_NO_DEVICES();

    hipdnnHandle_t handle = nullptr;

    auto createStatus = hipdnnCreate(&handle);
    ASSERT_EQ(createStatus, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess) << "Failed to create HIP stream.";

    auto setStreamStatus = hipdnnSetStream(handle, stream);
    ASSERT_EQ(setStreamStatus, HIPDNN_STATUS_SUCCESS);

    hipStream_t retrievedStream = nullptr;
    auto getStreamStatus = hipdnnGetStream(handle, &retrievedStream);
    ASSERT_EQ(getStreamStatus, HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(retrievedStream, stream);

    auto destroyStatus = hipdnnDestroy(handle);
    ASSERT_EQ(destroyStatus, HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess) << "Failed to destroy HIP stream.";
}
