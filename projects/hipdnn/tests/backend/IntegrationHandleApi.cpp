// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <test_plugins/TestPluginConstants.hpp>

class IntegrationHandleApi : public ::testing::Test
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

    void TearDown() override {}
};

class IntegrationGpuHandleApi : public ::testing::Test
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

    void TearDown() override {}
};

TEST_F(IntegrationHandleApi, CreateAndDestroy)
{
    hipdnnHandle_t handle = nullptr;

    auto createStatus = hipdnnCreate(&handle);
    ASSERT_EQ(createStatus, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto destroyStatus = hipdnnDestroy(handle);
    ASSERT_EQ(destroyStatus, HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationHandleApi, CreateWithNullptr)
{
    const hipdnnStatus_t status = hipdnnCreate(nullptr);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationHandleApi, SetStreamNullptrHandle)
{
    hipStream_t testStream = nullptr;
    auto setStreamStatus = hipdnnSetStream(nullptr, testStream);
    ASSERT_EQ(setStreamStatus, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationHandleApi, GetStreamNullptrHandle)
{
    hipStream_t retrievedStream = nullptr;
    auto getStreamStatus = hipdnnGetStream(nullptr, &retrievedStream);
    ASSERT_EQ(getStreamStatus, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationHandleApi, GetStreamNullptrStreamPointer)
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

TEST_F(IntegrationGpuHandleApi, GetStreamPointer)
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
