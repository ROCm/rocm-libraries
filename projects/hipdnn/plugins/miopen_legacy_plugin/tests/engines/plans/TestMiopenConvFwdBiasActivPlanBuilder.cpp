/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

#include <gtest/gtest.h>
#include <hipdnn_sdk/plugin/test_utils/MockGraph.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/StringUtil.hpp>
#include <miopen/miopen.h>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/MiopenConvFwdBiasActivPlanBuilder.hpp"
#include "tests/common/ConvolutionCommon.hpp"

using namespace miopen_legacy_plugin;
using namespace hipdnn_plugin;

class TestMiopenConvFwdBiasActivPlanBuilder : public ::testing::Test
{
protected:
    MiopenConvFwdBiasActivPlanBuilder _planBuilder;
    HipdnnEnginePluginHandle _dummyHandle;
};

class TestGpuMiopenConvFwdBiasActivPlanBuilder : public TestMiopenConvFwdBiasActivPlanBuilder
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(miopenCreate(&_handle.miopenHandle), miopenStatusSuccess);
    }

    void TearDown() override
    {
        if(_handle.miopenHandle != nullptr)
        {
            EXPECT_EQ(miopenDestroy(_handle.miopenHandle), miopenStatusSuccess);
        }
    }

    HipdnnEnginePluginHandle _handle;
};

TEST_F(TestGpuMiopenConvFwdBiasActivPlanBuilder, IsApplicableReturnsTrueForSupportedGraph)
{
    auto builder = hipdnn_sdk::test_utilities::createValidConvFwdBiasActivGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_handle, graph);
    EXPECT_TRUE(applicable);
}

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, IsApplicableReturnsFalseForUnsupportedGraph)
{
    {
        auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
        hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);
        EXPECT_FALSE(applicable);
    }
    {
        auto builder = hipdnn_sdk::test_utilities::createValidBatchnormBwdGraph();
        hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);
        EXPECT_FALSE(applicable);
    }
    {
        auto builder = hipdnn_sdk::test_utilities::createValidConvFwdGraph();
        hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);
        EXPECT_FALSE(applicable);
    }
    {
        auto builder = hipdnn_sdk::test_utilities::createValidConvBwdGraph();
        hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);
        EXPECT_FALSE(applicable);
    }
    {
        auto builder = hipdnn_sdk::test_utilities::createValidConvWrwGraph();
        hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);
        EXPECT_FALSE(applicable);
    }
}

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, IsApplicableReturnsFalseForWrongNodeCountGraph)
{
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));
        bool applicable = _planBuilder.isApplicable(_dummyHandle, mockGraph);
        EXPECT_FALSE(applicable);
    }
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));
        bool applicable = _planBuilder.isApplicable(_dummyHandle, mockGraph);
        EXPECT_FALSE(applicable);
    }
}

TEST_F(TestGpuMiopenConvFwdBiasActivPlanBuilder, IsApplicableVariousLayouts)
{
    char** s = environ;

    for(; *s != nullptr; s++)
    {
        std::cout << *s << "\n";
    }

    std::vector<std::pair<std::vector<int64_t>, bool>> layoutsAndExpectedResults
        = {{{3, 2, 1, 0}, true},
           {{3, 0, 2, 1}, true},
           {{4, 3, 2, 1, 0}, true},
           {{4, 0, 3, 2, 1}, true},
           {{0, 1, 2, 3}, false}};

    using namespace hipdnn_sdk::utilities;

    for(const auto& [layoutOrder, isApplicable] : layoutsAndExpectedResults)
    {
        std::vector<int64_t> xDims(layoutOrder.size(), 16);
        xDims[0] = 1;
        auto xStrides = hipdnn_sdk::utilities::generateStrides(xDims, layoutOrder);
        std::vector<int64_t> wDims(layoutOrder.size(), 3);
        wDims[0] = 1;
        wDims[1] = xDims[1];
        auto wStrides = hipdnn_sdk::utilities::generateStrides(wDims, layoutOrder);

        std::vector<int64_t> convPrePadding(layoutOrder.size() - 2, 0);
        std::vector<int64_t> convPostPadding(layoutOrder.size() - 2, 0);
        std::vector<int64_t> convStrides(layoutOrder.size() - 2, 1);
        std::vector<int64_t> convDilation(layoutOrder.size() - 2, 1);

        test_conv_common::ConvTestCase convTestCase(std::move(xDims),
                                                    std::move(wDims),
                                                    std::move(convPrePadding),
                                                    std::move(convPostPadding),
                                                    std::move(convStrides),
                                                    std::move(convDilation),
                                                    0);

        auto yStrides = generateStrides(convTestCase.yDims, layoutOrder);
        auto builder = hipdnn_sdk::test_utilities::createValidConvFwdBiasActivGraph(
            convTestCase.xDims,
            xStrides,
            convTestCase.wDims,
            wStrides,
            convTestCase.yDims,
            yStrides,
            convTestCase.convPrePadding,
            convTestCase.convPostPadding,
            convTestCase.convStride,
            convTestCase.convDilation);

        hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_EQ(_planBuilder.isApplicable(_handle, graph), isApplicable)
            << "Layout order " + hipdnn_sdk::utilities::vecToString(layoutOrder);
    }
}

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, GetWorkspaceSizeThrowsForWrongNodeCountGraph)
{
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));

        EXPECT_THROW(_planBuilder.getWorkspaceSize(_dummyHandle, mockGraph),
                     hipdnn_plugin::HipdnnPluginException);
    }
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));

        EXPECT_THROW(_planBuilder.getWorkspaceSize(_dummyHandle, mockGraph),
                     hipdnn_plugin::HipdnnPluginException);
    }
}

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, GetWorkspaceSizeThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(_planBuilder.getWorkspaceSize(_dummyHandle, graph),
                 hipdnn_plugin::HipdnnPluginException);
}

TEST_F(TestGpuMiopenConvFwdBiasActivPlanBuilder, GetWorkspaceSizeReturnsValueForSupportedGraph)
{
    auto builder = hipdnn_sdk::test_utilities::createValidConvFwdBiasActivGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_NO_THROW(_planBuilder.getWorkspaceSize(_handle, graph));
}

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, BuildPlanThrowsForWrongNodeCountGraph)
{
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, mockGraph, ctx),
                     hipdnn_plugin::HipdnnPluginException);
        EXPECT_FALSE(ctx.hasValidPlan());
    }
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, mockGraph, ctx),
                     hipdnn_plugin::HipdnnPluginException);
        EXPECT_FALSE(ctx.hasValidPlan());
    }
}

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, BuildPlanThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx),
                 hipdnn_plugin::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestGpuMiopenConvFwdBiasActivPlanBuilder, BuildPlanCreatesValidPlanForSupportedGraph)
{
    auto builder = hipdnn_sdk::test_utilities::createValidConvFwdBiasActivGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}
