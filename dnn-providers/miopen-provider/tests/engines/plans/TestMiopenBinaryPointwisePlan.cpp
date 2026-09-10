// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "common/PointwiseCommon.hpp"
#include "engines/plans/MiopenBinaryPointwisePlan.hpp"

using namespace miopen_plugin;
using namespace pointwise_common;

namespace
{

using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;

} // namespace

class TestGpuMiopenBinaryPointwisePlan : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }

    std::unique_ptr<HipdnnMiopenHandle> _dummyHandle;
};

TEST_F(TestGpuMiopenBinaryPointwisePlan, GetWorkspaceSizeReturnsZero)
{
    auto fbb = createPointwiseGraph(validBinarySpec());
    const GraphWrapper graph(fbb.GetBufferPointer(), fbb.GetSize());

    const auto& nodeWrapper = graph.getNodeWrapper(0);
    const auto& attrs
        = nodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

    const MiopenBinaryPointwisePlan plan(attrs, graph.getTensorMap());
    EXPECT_EQ(plan.getWorkspaceSize(*_dummyHandle), 0u);
}

TEST_F(TestGpuMiopenBinaryPointwisePlan, ConstructorDoesNotThrowForValidGraph)
{
    auto fbb = createPointwiseGraph(validBinarySpec());
    const GraphWrapper graph(fbb.GetBufferPointer(), fbb.GetSize());

    const auto& nodeWrapper = graph.getNodeWrapper(0);
    const auto& attrs
        = nodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

    EXPECT_NO_THROW(MiopenBinaryPointwisePlan(attrs, graph.getTensorMap()));
}

TEST_F(TestGpuMiopenBinaryPointwisePlan, ConstructorThrowsInternalErrorWhenIn1TensorUidMissing)
{
    // A default-constructed PointwiseGraphSpec produces a unary-shaped node (no
    // in_1_tensor_uid) -- isApplicable would reject this graph, but here the plan
    // constructor is invoked directly to exercise its own defense against isApplicable and
    // buildPlan drifting apart.
    PointwiseGraphSpec spec;
    spec.mode = PointwiseMode::ADD;
    auto fbb = createPointwiseGraph(spec);
    const GraphWrapper graph(fbb.GetBufferPointer(), fbb.GetSize());

    const auto& nodeWrapper = graph.getNodeWrapper(0);
    const auto& attrs
        = nodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

    try
    {
        const MiopenBinaryPointwisePlan plan(attrs, graph.getTensorMap());
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    }
}
