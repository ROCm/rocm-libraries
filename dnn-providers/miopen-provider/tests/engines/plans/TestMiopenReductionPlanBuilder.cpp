// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenContext.hpp"
#include "HipdnnMiopenHandle.hpp"
#include "HipdnnMiopenSettings.hpp"
#include "MiopenApi.hpp"
#include "engines/plans/MiopenReductionPlanBuilder.hpp"

using namespace miopen_plugin;

namespace
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_test_sdk::utilities::createReductionGraph;
using hipdnn_test_sdk::utilities::MockEngineConfig;
using hipdnn_test_sdk::utilities::ReductionGraphSpec;
using hipdnn_test_sdk::utilities::ReductionTensorSpec;

// Owns the flatbuffer a GraphWrapper views, so the two cannot be separated.
class ReductionGraph
{
public:
    explicit ReductionGraph(const ReductionGraphSpec& spec)
        : _builder(createReductionGraph(spec))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
    {
    }

    const GraphWrapper& graph() const
    {
        return _graph;
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    GraphWrapper _graph;
};

// The workspace size miopenGetReductionWorkspaceSize reports on `handle` for `spec` reduced with
// `op`. Built from raw MIOpen calls, so none of the provider code under test is involved. Any
// MIOpen failure is recorded as a test failure and yields 0.
size_t queryMiopenReductionWorkspaceSize(miopenHandle_t handle,
                                         miopenReduceTensorOp_t op,
                                         const ReductionGraphSpec& spec)
{
    using TensorDescriptor = std::unique_ptr<std::remove_pointer_t<miopenTensorDescriptor_t>,
                                             decltype(&miopenDestroyTensorDescriptor)>;
    using ReduceDescriptor = std::unique_ptr<std::remove_pointer_t<miopenReduceTensorDescriptor_t>,
                                             decltype(&miopenDestroyReduceTensorDescriptor)>;

    const auto toMiopenType
        = [](DataType type) { return type == DataType::HALF ? miopenHalf : miopenFloat; };

    const auto makeTensor = [](miopenDataType_t type, const ReductionTensorSpec& tensor) {
        miopenTensorDescriptor_t raw = nullptr;
        EXPECT_EQ(miopenCreateTensorDescriptor(&raw), miopenStatusSuccess);
        TensorDescriptor descriptor(raw, &miopenDestroyTensorDescriptor);
        const std::vector<int> dims(tensor.dims.begin(), tensor.dims.end());
        const std::vector<int> strides(tensor.strides->begin(), tensor.strides->end());
        EXPECT_EQ(miopenSetTensorDescriptor(
                      raw, type, static_cast<int>(dims.size()), dims.data(), strides.data()),
                  miopenStatusSuccess);
        return descriptor;
    };

    const auto x = makeTensor(toMiopenType(spec.dataType), spec.x);
    const auto y = makeTensor(toMiopenType(spec.outputDataType()), spec.y);

    miopenReduceTensorDescriptor_t rawReduce = nullptr;
    EXPECT_EQ(miopenCreateReduceTensorDescriptor(&rawReduce), miopenStatusSuccess);
    const ReduceDescriptor reduce(rawReduce, &miopenDestroyReduceTensorDescriptor);
    EXPECT_EQ(miopenSetReduceTensorDescriptor(rawReduce,
                                              op,
                                              miopenFloat,
                                              MIOPEN_PROPAGATE_NAN,
                                              MIOPEN_REDUCE_TENSOR_NO_INDICES,
                                              MIOPEN_32BIT_INDICES),
              miopenStatusSuccess);

    size_t workspaceSize = 0;
    EXPECT_EQ(miopenGetReductionWorkspaceSize(handle, rawReduce, x.get(), y.get(), &workspaceSize),
              miopenStatusSuccess);
    return workspaceSize;
}

} // namespace

class TestGpuMiopenReductionPlanBuilder : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _handle = std::make_unique<HipdnnMiopenHandle>();
    }

    // A multi-block ADD: MIOpen's workspace holds a descriptor page plus partials sized from the
    // tensor lengths, so it differs if the plan queries with the wrong tensors. A single-pass
    // shape would always get the one page.
    const ReductionGraphSpec _multiBlockSpec
        = ReductionGraphSpec::packed(ReductionMode::ADD, {2, 4096}, {2, 1});
    MiopenReductionPlanBuilder _planBuilder;
    std::unique_ptr<HipdnnMiopenHandle> _handle;
    MockEngineConfig _mockEngineConfig;
};

// isApplicable() delegates to reduction_applicability::isSupported(), whose rules
// TestMiopenReductionChecks pins one by one, so one accepted and one declined graph suffice here.
TEST_F(TestGpuMiopenReductionPlanBuilder, IsApplicableDelegatesToGate)
{
    ReductionGraphSpec mulNoZerosSpec;
    mulNoZerosSpec.mode = ReductionMode::MUL_NO_ZEROS;
    const ReductionGraph accepted{ReductionGraphSpec{}};
    const ReductionGraph declined{mulNoZerosSpec};

    EXPECT_TRUE(_planBuilder.isApplicable(*_handle, accepted.graph()));
    EXPECT_FALSE(_planBuilder.isApplicable(*_handle, declined.graph()));
}

TEST_F(TestGpuMiopenReductionPlanBuilder, GetCustomKnobsReturnsEmpty)
{
    const ReductionGraph graph{ReductionGraphSpec{}};

    EXPECT_TRUE(_planBuilder.getCustomKnobs(*_handle, graph.graph()).empty());
}

TEST_F(TestGpuMiopenReductionPlanBuilder, InitializeExecutionSettingsLeavesSettingsUnchanged)
{
    const ReductionGraph graph{ReductionGraphSpec{}};
    HipdnnMiopenSettings settings;

    _planBuilder.initializeExecutionSettings(*_handle, graph.graph(), _mockEngineConfig, settings);

    EXPECT_FALSE(settings.benchmarkingEnabled());
    EXPECT_FALSE(settings.workspaceSizeLimit().has_value());
    EXPECT_FALSE(settings.defaultWorkspaceSize().has_value());
}

TEST_F(TestGpuMiopenReductionPlanBuilder, GetMaxWorkspaceSizeMatchesMiopenQuery)
{
    const ReductionGraph graph{_multiBlockSpec};
    const HipdnnMiopenSettings settings;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(*_handle, graph.graph(), settings),
              queryMiopenReductionWorkspaceSize(
                  _handle->miopenHandle, MIOPEN_REDUCE_TENSOR_ADD, _multiBlockSpec));
}

TEST_F(TestGpuMiopenReductionPlanBuilder, BuildPlanSetsPlanWithQueriedWorkspaceSize)
{
    const ReductionGraph graph{_multiBlockSpec};
    HipdnnMiopenContext context;

    _planBuilder.buildPlan(*_handle, graph.graph(), _mockEngineConfig, context);

    ASSERT_TRUE(context.hasValidPlan());
    EXPECT_EQ(context.plan().getWorkspaceSize(*_handle),
              queryMiopenReductionWorkspaceSize(
                  _handle->miopenHandle, MIOPEN_REDUCE_TENSOR_ADD, _multiBlockSpec));
}
