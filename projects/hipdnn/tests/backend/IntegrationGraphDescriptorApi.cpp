// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BackendTestHelpers.hpp"
#include "hipdnn_backend.h"
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_common_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
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

// ============================================================================
// RFC-0016 §4.2 serialized-graph min-required-engine-API-version guard.
//
// deserializeGraph() must reject a serialized Graph whose
// min_required_engine_api_version exceeds what this build understands
// (the ceiling is K_MAX_SUPPORTED_API_VERSION == "1.3.0", the highest
// version constant any graph feature can currently gate on), and must accept
// the baseline "1.0.0" and the ceiling "1.3.0". An unstamped field (a graph a
// writer never populated, e.g. a hand-built fixture) is treated as "1.0.0".
// Exercised through the public C API hipdnnBackendCreateAndDeserializeGraph_ext,
// which surfaces the guard's HipdnnException as HIPDNN_STATUS_NOT_SUPPORTED.
//
// Complementary contract: a graph built via the backend API stamps
// min_required_engine_api_version to the highest feature floor it triggers --
// "1.2.0" for a runtime pass-by-value tensor, "1.3.0" for a non-default tensor
// alignment, else "1.0.0" -- see
// hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion(), the single
// shared graph -> required-version mapping used by both this guard and
// EnginePluginResourceManager's plugin applicability filter.
// ============================================================================

namespace
{
// Serialize an otherwise-valid reduction graph whose min_required_engine_api_version
// is forced to `major.minor.patch`. Round-trips through GraphT so the field is
// stamped regardless of the flatbuffer default-elision behavior.
flatbuffers::DetachedBuffer
    serializeReductionGraphWithVersion(uint32_t major, uint32_t minor, uint32_t patch)
{
    auto builder = hipdnn_test_sdk::utilities::createValidReductionGraph();
    const auto* graphFb
        = hipdnn_flatbuffers_sdk::data_objects::GetGraph(builder.GetBufferPointer());
    hipdnn_flatbuffers_sdk::data_objects::GraphT graphT;
    graphFb->UnPackTo(&graphT);
    graphT.min_required_engine_api_version
        = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::EngineApiVersion>(
            major, minor, patch);

    flatbuffers::FlatBufferBuilder rebuilder;
    rebuilder.Finish(hipdnn_flatbuffers_sdk::data_objects::Graph::Pack(rebuilder, &graphT));
    return rebuilder.Release();
}

// Serialize an otherwise-valid reduction graph with min_required_engine_api_version
// left unstamped (null), simulating a writer that never populated the field.
flatbuffers::DetachedBuffer serializeReductionGraphWithUnstampedVersion()
{
    auto builder = hipdnn_test_sdk::utilities::createValidReductionGraph();
    const auto* graphFb
        = hipdnn_flatbuffers_sdk::data_objects::GetGraph(builder.GetBufferPointer());
    hipdnn_flatbuffers_sdk::data_objects::GraphT graphT;
    graphFb->UnPackTo(&graphT);
    graphT.min_required_engine_api_version.reset();

    flatbuffers::FlatBufferBuilder rebuilder;
    rebuilder.Finish(hipdnn_flatbuffers_sdk::data_objects::Graph::Pack(rebuilder, &graphT));
    return rebuilder.Release();
}

// Read back the stamped min_required_engine_api_version from a graph built and
// serialized through the backend C API. `runtimePassByValue` toggles the
// runtime flag on the reduction input tensor; `tensorAlignment` sets the input
// tensor's byte alignment (16 is the schema default and leaves it unstamped).
hipdnn_data_sdk::utilities::Version buildAndReadStampedVersion(bool runtimePassByValue,
                                                               int64_t tensorAlignment = 16)
{
    const std::vector<int64_t> inDims = {4, 8};
    const std::vector<int64_t> inStrides = {8, 1};
    const std::vector<int64_t> outDims = {1, 8};
    const std::vector<int64_t> outStrides = {8, 1};

    std::vector<hipdnnBackendDescriptor_t> owned;
    const auto cleanup = [&owned]() {
        for(auto* d : owned)
        {
            hipdnnBackendDestroyDescriptor(d);
        }
    };

    hipdnnBackendDescriptor_t xDesc = nullptr;
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_TENSOR_DESCRIPTOR, &xDesc),
              HIPDNN_STATUS_SUCCESS);
    owned.push_back(xDesc);
    setAllTensorAttributes(xDesc, 1, "input", inDims, inStrides);
    if(::testing::Test::HasFatalFailure())
    {
        cleanup();
        return hipdnn_data_sdk::utilities::Version{0, 0, 0};
    }
    if(runtimePassByValue)
    {
        bool flag = true;
        EXPECT_EQ(hipdnnBackendSetAttribute(xDesc,
                                            HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT,
                                            HIPDNN_TYPE_BOOLEAN,
                                            1,
                                            &flag),
                  HIPDNN_STATUS_SUCCESS);
    }
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  xDesc, HIPDNN_ATTR_TENSOR_BYTE_ALIGNMENT, HIPDNN_TYPE_INT64, 1, &tensorAlignment),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendFinalize(xDesc), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t yDesc = createAndFinalizeTensorDesc(2, "output", outDims, outStrides);
    owned.push_back(yDesc);

    hipdnnBackendDescriptor_t opDesc = nullptr;
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR, &opDesc),
              HIPDNN_STATUS_SUCCESS);
    owned.push_back(opDesc);
    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_OPERATION_REDUCTION_XDESC,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&xDesc)),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_OPERATION_REDUCTION_YDESC,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&yDesc)),
              HIPDNN_STATUS_SUCCESS);
    const hipdnnReduceTensorOp_t reduceOp = HIPDNN_REDUCE_TENSOR_ADD;
    EXPECT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_REDUCTION_OPERATOR,
                                        HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                                        1,
                                        &reduceOp),
              HIPDNN_STATUS_SUCCESS);
    const hipdnnDataType_t compType = HIPDNN_DATA_FLOAT;
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  opDesc, HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &compType),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendFinalize(opDesc), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t graphDesc = nullptr;
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &graphDesc),
              HIPDNN_STATUS_SUCCESS);
    owned.push_back(graphDesc);
    const std::array<hipdnnBackendDescriptor_t, 1> ops = {opDesc};
    EXPECT_EQ(hipdnnBackendSetAttribute(graphDesc,
                                        HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(ops.data())),
              HIPDNN_STATUS_SUCCESS);

    size_t size = 0;
    EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(graphDesc, 0, &size, nullptr),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_GT(size, 0u);
    if(::testing::Test::HasFatalFailure() || size == 0)
    {
        cleanup();
        return hipdnn_data_sdk::utilities::Version{0, 0, 0};
    }

    std::vector<uint8_t> buffer(size);
    EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(graphDesc, size, &size, buffer.data()),
              HIPDNN_STATUS_SUCCESS);

    const auto* graphFb = hipdnn_flatbuffers_sdk::data_objects::GetGraph(buffer.data());
    EXPECT_NE(graphFb, nullptr);
    const auto stamped = hipdnn_plugin_sdk::fromEngineApiVersion(
        graphFb != nullptr ? graphFb->min_required_engine_api_version() : nullptr);

    cleanup();
    return stamped;
}
} // namespace

// A serialized graph demanding a newer engine plugin API version than this
// build's ceiling (K_MAX_SUPPORTED_API_VERSION == "1.3.0") must be rejected, not
// silently accepted. "1.4.0" is used as a version strictly above that ceiling.
TEST_F(IntegrationGraphDescriptorApi, DeserializeRejectsFutureApiVersion)
{
    const flatbuffers::DetachedBuffer serialized = serializeReductionGraphWithVersion(1, 4, 0);

    hipdnnBackendDescriptor_t descriptor = nullptr;
    EXPECT_EQ(hipdnnBackendCreateAndDeserializeGraph_ext(
                  &descriptor, serialized.data(), serialized.size()),
              HIPDNN_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(descriptor, nullptr);
}

// "1.3.0" sits at this build's ceiling (K_MAX_SUPPORTED_API_VERSION) and must deserialize.
TEST_F(IntegrationGraphDescriptorApi, DeserializeAcceptsCeilingApiVersion)
{
    const flatbuffers::DetachedBuffer serialized = serializeReductionGraphWithVersion(1, 3, 0);

    hipdnnBackendDescriptor_t descriptor = nullptr;
    EXPECT_EQ(hipdnnBackendCreateAndDeserializeGraph_ext(
                  &descriptor, serialized.data(), serialized.size()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_NE(descriptor, nullptr);
    hipdnnBackendDestroyDescriptor(descriptor);
}

// "1.0.0" is the legacy/default floor and must deserialize.
TEST_F(IntegrationGraphDescriptorApi, DeserializeAcceptsBaselineApiVersion)
{
    const flatbuffers::DetachedBuffer serialized = serializeReductionGraphWithVersion(1, 0, 0);

    hipdnnBackendDescriptor_t descriptor = nullptr;
    EXPECT_EQ(hipdnnBackendCreateAndDeserializeGraph_ext(
                  &descriptor, serialized.data(), serialized.size()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_NE(descriptor, nullptr);
    hipdnnBackendDestroyDescriptor(descriptor);
}

// An unstamped field (a writer that never populated it) reads as "1.0.0" and
// must deserialize, exactly like the legacy min_reader_version's implicit 0.
TEST_F(IntegrationGraphDescriptorApi, DeserializeAcceptsUnstampedApiVersion)
{
    const flatbuffers::DetachedBuffer serialized = serializeReductionGraphWithUnstampedVersion();

    hipdnnBackendDescriptor_t descriptor = nullptr;
    EXPECT_EQ(hipdnnBackendCreateAndDeserializeGraph_ext(
                  &descriptor, serialized.data(), serialized.size()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_NE(descriptor, nullptr);
    hipdnnBackendDestroyDescriptor(descriptor);
}

// A graph carrying a runtime pass-by-value tensor stamps "1.2.0" so that
// pre-1.2.0 readers/plugins refuse it (mirrors the guard rejection above).
TEST_F(IntegrationGraphDescriptorApi, StampsPassByValueVersionForRuntimePassByValue)
{
    const auto stamped = buildAndReadStampedVersion(/*runtimePassByValue=*/true);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_EQ(
        stamped,
        hipdnn_data_sdk::utilities::Version{hipdnn_plugin_sdk::K_PASS_BY_VALUE_MIN_API_VERSION});
}

// A graph with only ordinary/compile-time tensors stamps the baseline "1.0.0"
// so legacy readers/plugins still accept it.
TEST_F(IntegrationGraphDescriptorApi, StampsBaselineVersionForOrdinaryGraph)
{
    const auto stamped = buildAndReadStampedVersion(/*runtimePassByValue=*/false);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_EQ(stamped,
              hipdnn_data_sdk::utilities::Version{
                  hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE});
}

// A graph carrying a tensor with a non-default byte alignment stamps "1.3.0" so
// that plugins predating custom-alignment support refuse it.
TEST_F(IntegrationGraphDescriptorApi, StampsTensorAlignmentVersionForNonDefaultAlignment)
{
    const auto stamped
        = buildAndReadStampedVersion(/*runtimePassByValue=*/false, /*tensorAlignment=*/32);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_EQ(stamped,
              hipdnn_data_sdk::utilities::Version{
                  hipdnn_plugin_sdk::K_TENSOR_ATTRIBUTE_ALIGNMENT_MIN_VERSION});
}

// The default byte alignment (16) must not raise the floor: such a graph stays
// at the baseline "1.0.0" and is served by every plugin.
TEST_F(IntegrationGraphDescriptorApi, StampsBaselineVersionForDefaultAlignment)
{
    const auto stamped
        = buildAndReadStampedVersion(/*runtimePassByValue=*/false, /*tensorAlignment=*/16);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_EQ(stamped,
              hipdnn_data_sdk::utilities::Version{
                  hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE});
}

// End-to-end: a runtime pass-by-value tensor's classification (the
// HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT flag) must survive a full
// build -> serialize -> deserialize round trip through the public C API, not
// just the version stamp on the graph as a whole. Builds a graph with a
// runtime-pbv input tensor, serializes it, deserializes it via
// hipdnnBackendCreateAndDeserializeGraph_ext, pulls the reduction op's XDESC
// back out of the *deserialized* graph, and re-reads the flag off it.
TEST_F(IntegrationGraphDescriptorApi, DeserializedGraphRestoresTensorRuntimePassByValueFlag)
{
    const std::vector<int64_t> inDims = {4, 8};
    const std::vector<int64_t> inStrides = {8, 1};
    const std::vector<int64_t> outDims = {1, 8};
    const std::vector<int64_t> outStrides = {8, 1};

    std::vector<hipdnnBackendDescriptor_t> owned;
    const auto cleanup = [&owned]() {
        for(auto* d : owned)
        {
            hipdnnBackendDestroyDescriptor(d);
        }
    };

    hipdnnBackendDescriptor_t xDesc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_TENSOR_DESCRIPTOR, &xDesc),
              HIPDNN_STATUS_SUCCESS);
    owned.push_back(xDesc);
    setAllTensorAttributes(xDesc, 1, "input", inDims, inStrides);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    bool flag = true;
    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            xDesc, HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT, HIPDNN_TYPE_BOOLEAN, 1, &flag),
        HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendFinalize(xDesc), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t yDesc = createAndFinalizeTensorDesc(2, "output", outDims, outStrides);
    owned.push_back(yDesc);

    hipdnnBackendDescriptor_t opDesc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR, &opDesc),
              HIPDNN_STATUS_SUCCESS);
    owned.push_back(opDesc);
    ASSERT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_OPERATION_REDUCTION_XDESC,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&xDesc)),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_OPERATION_REDUCTION_YDESC,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&yDesc)),
              HIPDNN_STATUS_SUCCESS);
    const hipdnnReduceTensorOp_t reduceOp = HIPDNN_REDUCE_TENSOR_ADD;
    ASSERT_EQ(hipdnnBackendSetAttribute(opDesc,
                                        HIPDNN_ATTR_REDUCTION_OPERATOR,
                                        HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                                        1,
                                        &reduceOp),
              HIPDNN_STATUS_SUCCESS);
    const hipdnnDataType_t compType = HIPDNN_DATA_FLOAT;
    ASSERT_EQ(hipdnnBackendSetAttribute(
                  opDesc, HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &compType),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendFinalize(opDesc), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t graphDesc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &graphDesc),
              HIPDNN_STATUS_SUCCESS);
    owned.push_back(graphDesc);
    const std::array<hipdnnBackendDescriptor_t, 1> ops = {opDesc};
    ASSERT_EQ(hipdnnBackendSetAttribute(graphDesc,
                                        HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(ops.data())),
              HIPDNN_STATUS_SUCCESS);

    size_t size = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(graphDesc, 0, &size, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(size, 0u);
    std::vector<uint8_t> buffer(size);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(graphDesc, size, &size, buffer.data()),
              HIPDNN_STATUS_SUCCESS);

    // Deserialize into a brand-new descriptor -- this is the restored graph
    // under test, distinct from the one that built the buffer above.
    hipdnnBackendDescriptor_t restoredGraphDesc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateAndDeserializeGraph_ext(&restoredGraphDesc, buffer.data(), size),
              HIPDNN_STATUS_SUCCESS);
    owned.push_back(restoredGraphDesc);

    // Pull the single operation back out of the restored graph.
    int64_t opCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(restoredGraphDesc,
                                        HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &opCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(opCount, 1);
    hipdnnBackendDescriptor_t restoredOpDesc = nullptr;
    int64_t gotOpCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(restoredGraphDesc,
                                        HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &gotOpCount,
                                        static_cast<void*>(&restoredOpDesc)),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(gotOpCount, 1);
    owned.push_back(restoredOpDesc);

    // Pull the reduction op's X tensor descriptor back out.
    hipdnnBackendDescriptor_t restoredXDesc = nullptr;
    int64_t gotXCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(restoredOpDesc,
                                        HIPDNN_ATTR_OPERATION_REDUCTION_XDESC,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &gotXCount,
                                        static_cast<void*>(&restoredXDesc)),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(gotXCount, 1);
    owned.push_back(restoredXDesc);

    // The flag set on the ORIGINAL tensor before serialization must be
    // readable, unchanged, off the tensor descriptor extracted from the
    // DESERIALIZED graph -- proving the full round trip, not just the
    // graph-level version stamp checked by StampsPassByValueVersionFor*.
    bool restoredFlag = false;
    int64_t elementCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(restoredXDesc,
                                        HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT,
                                        HIPDNN_TYPE_BOOLEAN,
                                        1,
                                        &elementCount,
                                        &restoredFlag),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(elementCount, 1);
    EXPECT_TRUE(restoredFlag);

    cleanup();
}
