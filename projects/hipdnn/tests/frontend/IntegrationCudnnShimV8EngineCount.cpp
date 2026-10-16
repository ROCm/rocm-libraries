// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT against the real hipDNN backend
// and the test engine plugins. The count comes from hipDNN heuristics, which
// query the device, so these tests skip on device-less hosts.

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include "CudnnShimV8GraphBuilder.hpp"

namespace
{
using hipdnn_shim_v8_test::GraphBuilder;
using hipdnn_shim_v8_test::lastError;

class IntegrationCudnnShimV8EngineCount : public hipdnn_tests::IntegrationTestFixture
{
protected:
    // A finalized single-op graph; every test plugin claims any graph.
    cudnnBackendDescriptor_t finalizedGraph()
    {
        const std::vector<std::int64_t> dims = {2, 4, 8, 8};
        cudnnBackendDescriptor_t relu = _builder.create(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        GraphBuilder::set(
            relu, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, CUDNN_POINTWISE_RELU_FWD);
        GraphBuilder::set(
            relu, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        GraphBuilder::finalize(relu);
        cudnnBackendDescriptor_t op = _builder.finalizedOperation(
            CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
            {{CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, relu},
             {CUDNN_ATTR_OPERATION_POINTWISE_XDESC, _builder.tensor(1, dims)},
             {CUDNN_ATTR_OPERATION_POINTWISE_YDESC, _builder.tensor(2, dims)}});
        cudnnBackendDescriptor_t graph = _builder.graph(_handle, {op});
        EXPECT_EQ(cudnnBackendFinalize(graph), CUDNN_STATUS_SUCCESS) << lastError();
        return graph;
    }

    static std::int64_t engineCount(cudnnBackendDescriptor_t graph)
    {
        std::int64_t count = -1;
        std::int64_t elementCount = 0;
        EXPECT_EQ(cudnnBackendGetAttribute(graph,
                                           CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &elementCount,
                                           static_cast<void*>(&count)),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        EXPECT_EQ(elementCount, 1);
        return count;
    }

    GraphBuilder _builder;
};

TEST_F(IntegrationCudnnShimV8EngineCount, GoodPluginHasOneEngine)
{
    EXPECT_EQ(engineCount(finalizedGraph()), 1);
}

TEST_F(IntegrationCudnnShimV8EngineCount, RepeatedQueriesAgree)
{
    cudnnBackendDescriptor_t graph = finalizedGraph();
    EXPECT_EQ(engineCount(graph), 1);
    EXPECT_EQ(engineCount(graph), 1);
}

TEST_F(IntegrationCudnnShimV8EngineCount, NullArrayReportsElementCount)
{
    cudnnBackendDescriptor_t graph = finalizedGraph();
    std::int64_t elementCount = 0;
    EXPECT_EQ(cudnnBackendGetAttribute(graph,
                                       CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                                       CUDNN_TYPE_INT64,
                                       0,
                                       &elementCount,
                                       nullptr),
              CUDNN_STATUS_SUCCESS)
        << lastError();
    EXPECT_EQ(elementCount, 1);
    EXPECT_EQ(engineCount(graph), 1);
}

TEST_F(IntegrationCudnnShimV8EngineCount, ZeroRequestedLeavesArrayUntouched)
{
    cudnnBackendDescriptor_t graph = finalizedGraph();
    std::int64_t count = -1;
    std::int64_t elementCount = 0;
    EXPECT_EQ(cudnnBackendGetAttribute(graph,
                                       CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                                       CUDNN_TYPE_INT64,
                                       0,
                                       &elementCount,
                                       static_cast<void*>(&count)),
              CUDNN_STATUS_SUCCESS)
        << lastError();
    EXPECT_EQ(elementCount, 1);
    EXPECT_EQ(count, -1);
}

// The hashed-name plugin accessor exists only where the target defines its name.
#ifdef TEST_HASHED_NAME_PLUGIN_NAME
class IntegrationCudnnShimV8EngineCountThreePlugins : public IntegrationCudnnShimV8EngineCount
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testGoodPluginPath(),
                hipdnn_tests::plugin_constants::testExecuteFailsPluginPath(),
                hipdnn_tests::plugin_constants::testHashedNamePluginPath()};
    }
};

TEST_F(IntegrationCudnnShimV8EngineCountThreePlugins, EveryLoadedEngineCounted)
{
    EXPECT_EQ(engineCount(finalizedGraph()), 3);
}
#endif

class IntegrationCudnnShimV8EngineCountNoEngines : public IntegrationCudnnShimV8EngineCount
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath()};
    }
};

TEST_F(IntegrationCudnnShimV8EngineCountNoEngines, ZeroIsNotAnError)
{
    EXPECT_EQ(engineCount(finalizedGraph()), 0);
}

} // namespace
