// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BackendTestHelpers.hpp"
#include "hipdnn_backend.h"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_common_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <test_plugins/TestPluginConstants.hpp>
#include <vector>

using namespace backend_test;
using namespace hipdnn_tests::constants;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;

class IntegrationGraphDescriptorApi : public ::testing::Test
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
};

TEST_F(IntegrationGraphDescriptorApi, CreateAndDeserializeGraphExtWithNullGraph)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(&descriptor, nullptr, 0);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(descriptor, nullptr);
}

TEST_F(IntegrationGraphDescriptorApi, OverrideShapeEnabledSetGetRoundTrip)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor),
              HIPDNN_STATUS_SUCCESS);

    bool enabled = true;
    EXPECT_EQ(hipdnnBackendSetAttribute(descriptor,
                                        HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                                        HIPDNN_TYPE_BOOLEAN,
                                        1,
                                        &enabled),
              HIPDNN_STATUS_SUCCESS);

    bool retrieved = false;
    int64_t elementCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(descriptor,
                                        HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                                        HIPDNN_TYPE_BOOLEAN,
                                        1,
                                        &elementCount,
                                        &retrieved),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(elementCount, 1);
    EXPECT_TRUE(retrieved);

    EXPECT_EQ(hipdnnBackendDestroyDescriptor(descriptor), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationGraphDescriptorApi, SetOperationGraph)
{
    SKIP_IF_NO_DEVICES();
    // Any valid graph — tests exercise the API, not a specific operation type
    auto graphBuilder = hipdnn_test_sdk::utilities::createValidReductionGraph();
    flatbuffers::DetachedBuffer serializedGraph = graphBuilder.Release();

    hipdnnBackendDescriptor_t descriptor = nullptr;

    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &descriptor, serializedGraph.data(), serializedGraph.size());

    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendSetAttribute(descriptor,
                                       HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                       HIPDNN_TYPE_HANDLE,
                                       1,
                                       static_cast<const void*>(&handle));
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendFinalize(descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDestroyDescriptor(descriptor);
    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationGraphDescriptorApi, FinalizeInvalidOperationGraph)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    auto status
        = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendFinalize(descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);

    status = hipdnnBackendDestroyDescriptor(descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationGraphDescriptorApi, GetSerializedGraphSucceedsWithoutFinalization)
{
    hipdnnBackendDescriptor_t desc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &desc),
              HIPDNN_STATUS_SUCCESS);

    size_t size = 0;
    EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(desc, 0, &size, nullptr),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_GT(size, 0u);

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationGraphDescriptorApi, GetSerializedGraphFailsWithNullParams)
{
    size_t size = 0;
    EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(nullptr, 0, &size, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationGraphDescriptorApi, GetSerializedGraphFailsWithNullSizeParam)
{
    hipdnnBackendDescriptor_t desc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &desc),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(desc, 0, nullptr, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDestroyDescriptor(desc);
}

TEST_F(IntegrationGraphDescriptorApi, GetSerializedGraphSizeQueryMatchesCopySize)
{
    auto graphBuilder = hipdnn_test_sdk::utilities::createValidReductionGraph();
    flatbuffers::DetachedBuffer serializedGraph = graphBuilder.Release();

    hipdnnBackendDescriptor_t desc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateAndDeserializeGraph_ext(
                  &desc, serializedGraph.data(), serializedGraph.size()),
              HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                        HIPDNN_TYPE_HANDLE,
                                        1,
                                        static_cast<const void*>(&handle)),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    // Query size
    size_t queriedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(desc, 0, &queriedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(queriedSize, 0u);

    // Copy with queried size
    std::vector<uint8_t> buffer(queriedSize);
    size_t copySize = 0;
    ASSERT_EQ(
        hipdnnBackendGetSerializedBinaryGraph_ext(desc, queriedSize, &copySize, buffer.data()),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(copySize, queriedSize);

    // Verify data is valid FlatBuffer
    auto graphFb = hipdnn_flatbuffers_sdk::data_objects::GetGraph(buffer.data());
    ASSERT_NE(graphFb, nullptr);

    hipdnnBackendDestroyDescriptor(desc);
    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationGraphDescriptorApi, SerializedGraphRoundTripPreservesGraphProperties)
{
    auto graphBuilder = hipdnn_test_sdk::utilities::createValidReductionGraph();
    flatbuffers::DetachedBuffer serializedGraph = graphBuilder.Release();

    // Deserialize, set handle, finalize
    hipdnnBackendDescriptor_t desc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateAndDeserializeGraph_ext(
                  &desc, serializedGraph.data(), serializedGraph.size()),
              HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                        HIPDNN_TYPE_HANDLE,
                                        1,
                                        static_cast<const void*>(&handle)),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    // Use two-call pattern to get serialized data
    size_t size = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(desc, 0, &size, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(size, 0u);

    std::vector<uint8_t> buffer(size);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(desc, size, &size, buffer.data()),
              HIPDNN_STATUS_SUCCESS);

    // Verify graph properties match what we set
    auto graphFb = hipdnn_flatbuffers_sdk::data_objects::GetGraph(buffer.data());
    ASSERT_NE(graphFb, nullptr);
    hipdnn_flatbuffers_sdk::data_objects::GraphT graphT;
    graphFb->UnPackTo(&graphT);

    EXPECT_EQ(graphT.name, "test");
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.tensors.size(), 2u);
    EXPECT_EQ(graphT.nodes.size(), 1u);

    hipdnnBackendDestroyDescriptor(desc);
    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationGraphDescriptorApi, GetSerializedGraphFailsWithInsufficientBuffer)
{
    auto graphBuilder = hipdnn_test_sdk::utilities::createValidReductionGraph();
    flatbuffers::DetachedBuffer serializedGraph = graphBuilder.Release();

    hipdnnBackendDescriptor_t desc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateAndDeserializeGraph_ext(
                  &desc, serializedGraph.data(), serializedGraph.size()),
              HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                        HIPDNN_TYPE_HANDLE,
                                        1,
                                        static_cast<const void*>(&handle)),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    // Query actual size
    size_t queriedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(desc, 0, &queriedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(queriedSize, 1u);

    // Attempt copy with undersized buffer
    std::vector<uint8_t> buffer(1);
    size_t reportedSize = 0;
    EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(desc, 1, &reportedSize, buffer.data()),
              HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT);

    hipdnnBackendDestroyDescriptor(desc);
    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationGraphDescriptorApi, GetSerializedGraphSucceedsWithOversizedBuffer)
{
    auto graphBuilder = hipdnn_test_sdk::utilities::createValidReductionGraph();
    flatbuffers::DetachedBuffer serializedGraph = graphBuilder.Release();

    hipdnnBackendDescriptor_t desc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateAndDeserializeGraph_ext(
                  &desc, serializedGraph.data(), serializedGraph.size()),
              HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                        HIPDNN_TYPE_HANDLE,
                                        1,
                                        static_cast<const void*>(&handle)),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    // Query actual size
    size_t queriedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(desc, 0, &queriedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(queriedSize, 0u);

    // Copy with oversized buffer
    auto oversizedSize = queriedSize * 2;
    std::vector<uint8_t> buffer(oversizedSize);
    size_t reportedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  desc, oversizedSize, &reportedSize, buffer.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(reportedSize, queriedSize);

    // Verify data is valid
    auto graphFb = hipdnn_flatbuffers_sdk::data_objects::GetGraph(buffer.data());
    ASSERT_NE(graphFb, nullptr);

    hipdnnBackendDestroyDescriptor(desc);
    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationGraphDescriptorApi, GetGraphNameViaCApi)
{
    SKIP_IF_NO_DEVICES();

    auto graphBuilder = hipdnn_test_sdk::utilities::createValidReductionGraph();
    flatbuffers::DetachedBuffer serializedGraph = graphBuilder.Release();

    // Deserialize into a backend descriptor
    hipdnnBackendDescriptor_t desc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateAndDeserializeGraph_ext(
                  &desc, serializedGraph.data(), serializedGraph.size()),
              HIPDNN_STATUS_SUCCESS);

    // Set handle and finalize
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                        HIPDNN_TYPE_HANDLE,
                                        1,
                                        static_cast<const void*>(&handle)),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);

    // Query name count
    int64_t count = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(
                  desc, HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(count, 0);

    // Query name value
    std::vector<char> nameBuffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(desc,
                                        HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        count,
                                        &actualCount,
                                        nameBuffer.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_STREQ(nameBuffer.data(), "test");

    hipdnnBackendDestroyDescriptor(desc);
    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
