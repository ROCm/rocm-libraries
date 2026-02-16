// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/HipblasltMatmulPlanBuilder.hpp"

using namespace hipblaslt_plugin;
using namespace hipdnn_plugin_sdk;
using namespace hipdnn_test_sdk::utilities;

class TestHipblasltMatmulPlanBuilder : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipblasLtCreate(&_handle.hipblasltHandle), HIPBLAS_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle.hipblasltHandle != nullptr)
        {
            EXPECT_EQ(hipblasLtDestroy(_handle.hipblasltHandle), HIPBLAS_STATUS_SUCCESS);
        }
    }

    HipblasltMatmulPlanBuilder _planBuilder;
    HipdnnEnginePluginHandle _handle;
};

TEST_F(TestHipblasltMatmulPlanBuilder, IsApplicable)
{
    // MultiNode Graph
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, mockGraph));
    }

    // Unsupported Graph with batchnorm
    {
        auto builder = createValidBatchnormInferenceGraph();
        GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported broadcastable batch dimensions
    {
        std::vector<int64_t> aDims = {2, 4, 8};
        std::vector<int64_t> aStrides = {32, 8, 1};
        std::vector<int64_t> bDims = {1, 8, 5};
        std::vector<int64_t> bStrides = {40, 5, 1};
        std::vector<int64_t> cDims = {2, 4, 5};
        std::vector<int64_t> cStrides = {20, 5, 1};

        auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
        GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Different tensor ranks
    {
        std::vector<int64_t> aDims = {2, 4, 8};
        std::vector<int64_t> aStrides = {32, 8, 1};
        std::vector<int64_t> bDims = {8, 5};
        std::vector<int64_t> bStrides = {5, 1};
        std::vector<int64_t> cDims = {2, 4, 5};
        std::vector<int64_t> cStrides = {20, 5, 1};

        auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
        GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    }

    // Incorrect C batch dimension
    {
        std::vector<int64_t> aDims = {2, 4, 8};
        std::vector<int64_t> aStrides = {32, 8, 1};
        std::vector<int64_t> bDims = {3, 8, 5};
        std::vector<int64_t> bStrides = {40, 5, 1};
        std::vector<int64_t> cDims = {6, 4, 5};
        std::vector<int64_t> cStrides = {20, 5, 1};

        auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
        GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    }

    // Unsupported data type
    {
        std::vector<int64_t> aDims = {2, 4, 8};
        std::vector<int64_t> aStrides = {32, 8, 1};
        std::vector<int64_t> bDims = {2, 8, 5};
        std::vector<int64_t> bStrides = {40, 5, 1};
        std::vector<int64_t> cDims = {2, 4, 5};
        std::vector<int64_t> cStrides = {20, 5, 1};

        auto builder = createValidMatmulGraph(aDims,
                                              aStrides,
                                              bDims,
                                              bStrides,
                                              cDims,
                                              cStrides,
                                              hipdnn_data_sdk::data_objects::DataType::INT32);

        GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    }

    // Unsupported compute data type
    {
        flatbuffers::FlatBufferBuilder builder = createValidMatmulGraph();

        auto mutableGraph
            = hipdnn_data_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer());
        mutableGraph->mutable_nodes()->GetMutableObject(0)->mutate_compute_data_type(
            hipdnn_data_sdk::data_objects::DataType::HALF);

        GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported graph with matmul
    {
        auto builder = createValidMatmulGraph();
        GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }
}

TEST_F(TestHipblasltMatmulPlanBuilder, GetWorkspaceSize)
{
    // MultiNode Graph
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

        EXPECT_THROW(_planBuilder.getWorkspaceSize(_handle, mockGraph), HipdnnPluginException);
    }

    // Unsupported Graph with batchnorm
    {
        auto builder = createValidBatchnormInferenceGraph();
        GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_THROW(_planBuilder.getWorkspaceSize(_handle, graph), HipdnnPluginException);
    }

    // Supported Graph
    {
        auto builder = createValidMatmulGraph();
        GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_NO_THROW(_planBuilder.getWorkspaceSize(_handle, graph));
    }
}

TEST_F(TestHipblasltMatmulPlanBuilder, BuildPlan)
{
    // MultiNode Graph
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_THROW(_planBuilder.buildPlan(_handle, mockGraph, ctx), HipdnnPluginException);
        EXPECT_FALSE(ctx.hasValidPlan());
    }

    // Unsupported Graph with batchnorm
    {
        auto builder = createValidBatchnormInferenceGraph();
        GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_THROW(_planBuilder.buildPlan(_handle, graph, ctx), HipdnnPluginException);
        EXPECT_FALSE(ctx.hasValidPlan());
    }

    // Supported Graph with matmul
    {
        auto builder = createValidMatmulGraph();
        GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }
}
