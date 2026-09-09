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
#include "common/PointwiseExecutionCommon.hpp"
#include "common/PointwiseGraphCommon.hpp"
#include "engines/plans/MiopenBinaryPointwisePlan.hpp"

using namespace miopen_plugin;
using namespace test_pointwise_execution_common;
using namespace test_pointwise_graph_common;

namespace
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;

// Owns the flatbuffer + wrapper so the PointwiseAttributes and tensorMap references handed to
// the plan constructor stay alive for the duration of the call.
class BinaryGraph
{
public:
    BinaryGraph(const std::vector<int64_t>& aDims,
                const std::vector<int64_t>& aStrides,
                const std::vector<int64_t>& bDims,
                const std::vector<int64_t>& bStrides,
                PointwiseMode mode = PointwiseMode::ADD,
                DataType dtype = DataType::FLOAT)
        : _fbb(buildSpec(aDims, aStrides, bDims, bStrides, mode, dtype))
        , _graph(_fbb.GetBufferPointer(), _fbb.GetSize())
    {
    }

    // _graph points into _fbb's buffer; moving would leave it referencing a moved-from builder.
    BinaryGraph(const BinaryGraph&) = delete;
    BinaryGraph& operator=(const BinaryGraph&) = delete;
    BinaryGraph(BinaryGraph&&) = delete;
    BinaryGraph& operator=(BinaryGraph&&) = delete;

    // uids follow test_pointwise_graph_common convention: in_0 = 1, in_1 = 3, out_0 = 2.
    std::unique_ptr<MiopenBinaryPointwisePlan> makePlan() const
    {
        const auto& nodeWrapper = _graph.getNodeWrapper(0);
        const auto& attrs
            = nodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();
        return std::make_unique<MiopenBinaryPointwisePlan>(attrs, _graph.getTensorMap());
    }

private:
    static flatbuffers::FlatBufferBuilder buildSpec(const std::vector<int64_t>& aDims,
                                                    const std::vector<int64_t>& aStrides,
                                                    const std::vector<int64_t>& bDims,
                                                    const std::vector<int64_t>& bStrides,
                                                    PointwiseMode mode,
                                                    DataType dtype)
    {
        PointwiseGraphSpec spec;
        spec.mode = mode;
        spec.ioDataType = dtype;
        spec.secondInputDataType = dtype;
        spec.inputDims = aDims;
        spec.inputStrides = aStrides;
        spec.outputDims = aDims;
        spec.outputStrides = aStrides;
        spec.secondInputDims = bDims;
        spec.secondInputStrides = bStrides;
        return createPointwiseGraph(spec);
    }

    flatbuffers::FlatBufferBuilder _fbb;
    GraphWrapper _graph;
};

// A canonical rank-4 graph: in_0 uid 1 (1,3,4,4), in_1 uid 3 broadcasting on the last two axes,
// out_0 uid 2 (1,3,4,4).
BinaryGraph canonicalGraph()
{
    return BinaryGraph({1, 3, 4, 4}, {48, 16, 4, 1}, {1, 3, 1, 1}, {3, 1, 1, 1});
}

struct RankCase
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    const char* name;
};

const std::vector<RankCase>& getRankCases()
{
    static const std::vector<RankCase> s_cases = {{{4, 4, 4}, {16, 4, 1}, "Rank3"},
                                                  {{1, 3, 4, 4}, {48, 16, 4, 1}, "Rank4"},
                                                  {{1, 2, 3, 4, 4}, {96, 48, 16, 4, 1}, "Rank5"}};
    return s_cases;
}

struct ModeCase
{
    PointwiseMode mode;
    const char* name;
};

const std::vector<ModeCase>& getModeCases()
{
    static const std::vector<ModeCase> s_cases = {{PointwiseMode::ADD, "Add"},
                                                  {PointwiseMode::SUB, "Sub"},
                                                  {PointwiseMode::MUL, "Mul"},
                                                  {PointwiseMode::MAX_OP, "MaxOp"},
                                                  {PointwiseMode::MIN_OP, "MinOp"}};
    return s_cases;
}

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
    const auto graph = canonicalGraph();
    auto plan = graph.makePlan();
    EXPECT_EQ(plan->getWorkspaceSize(*_dummyHandle), 0u);
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

class TestGpuMiopenBinaryPointwisePlanRanks : public ::testing::TestWithParam<RankCase>
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }

    std::unique_ptr<HipdnnMiopenHandle> _dummyHandle;
};

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestGpuMiopenBinaryPointwisePlanRanks,
                         ::testing::ValuesIn(getRankCases()),
                         [](const ::testing::TestParamInfo<RankCase>& info) {
                             return std::string(info.param.name);
                         });

TEST_P(TestGpuMiopenBinaryPointwisePlanRanks, ConstructsSuccessfullyAtEveryAcceptedRank)
{
    // b is broadcast to all-ones so this exercises the same rank at every arity without
    // needing a per-rank literal for the broadcasting operand.
    const std::vector<int64_t> bDims(GetParam().dims.size(), 1);
    const std::vector<int64_t> bStrides(GetParam().strides.size(), 1);

    const BinaryGraph graph(GetParam().dims, GetParam().strides, bDims, bStrides);

    EXPECT_NO_THROW({ auto plan = graph.makePlan(); });
}

TEST_P(TestGpuMiopenBinaryPointwisePlanRanks, ExecutesAtEveryAcceptedRank)
{
    // b is broadcast to all-ones so this exercises the same rank at every arity without
    // needing a per-rank literal for the broadcasting operand.
    const std::vector<int64_t> bDims(GetParam().dims.size(), 1);
    const std::vector<int64_t> bStrides(GetParam().strides.size(), 1);

    const BinaryGraph graph(GetParam().dims, GetParam().strides, bDims, bStrides);
    auto plan = graph.makePlan();

    const TensorShape aShape{GetParam().dims, GetParam().strides};
    const TensorShape bShape{bDims, bStrides};
    const TensorShape cShape{GetParam().dims, GetParam().strides};

    executeAndVerify<float>(
        *plan, *_dummyHandle, PointwiseMode::ADD, DataType::FLOAT, aShape, bShape, cShape);
}

class TestGpuMiopenBinaryPointwisePlanModes : public ::testing::TestWithParam<ModeCase>
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }
    std::unique_ptr<HipdnnMiopenHandle> _dummyHandle;
};

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestGpuMiopenBinaryPointwisePlanModes,
                         ::testing::ValuesIn(getModeCases()),
                         [](const ::testing::TestParamInfo<ModeCase>& info) {
                             return std::string(info.param.name);
                         });

TEST_P(TestGpuMiopenBinaryPointwisePlanModes, ExecutesEachSupportedModeCorrectly)
{
    const BinaryGraph graph(
        {1, 3, 4, 4}, {48, 16, 4, 1}, {1, 3, 1, 1}, {3, 1, 1, 1}, GetParam().mode);
    auto plan = graph.makePlan();

    const TensorShape aShape{{1, 3, 4, 4}, {48, 16, 4, 1}};
    const TensorShape bShape{{1, 3, 1, 1}, {3, 1, 1, 1}};
    const TensorShape cShape{{1, 3, 4, 4}, {48, 16, 4, 1}};

    executeAndVerify<float>(
        *plan, *_dummyHandle, GetParam().mode, DataType::FLOAT, aShape, bShape, cShape);
}

TEST_F(TestGpuMiopenBinaryPointwisePlan, OverwritesOutputBuffer)
{
    // MiopenBinaryPointwisePlan::execute hardcodes beta = 0.0f: the output buffer's prior
    // content must never leak into the result. Pre-filling it with a large, distinctive
    // sentinel (instead of the usual zero-fill) is the only way to catch a beta = 1.0f
    // regression -- on a cold/zeroed allocator that regression is silently indistinguishable
    // from correct output.
    constexpr float SENTINEL = 12345.0f;

    const auto graph = canonicalGraph();
    auto plan = graph.makePlan();

    const TensorShape aShape{{1, 3, 4, 4}, {48, 16, 4, 1}};
    const TensorShape bShape{{1, 3, 1, 1}, {3, 1, 1, 1}};
    const TensorShape cShape{{1, 3, 4, 4}, {48, 16, 4, 1}};

    auto aTensor
        = hipdnn_test_sdk::detail::createTensor(DataType::FLOAT, aShape.dims, aShape.strides);
    auto bTensor
        = hipdnn_test_sdk::detail::createTensor(DataType::FLOAT, bShape.dims, bShape.strides);
    auto cTensor
        = hipdnn_test_sdk::detail::createTensor(DataType::FLOAT, cShape.dims, cShape.strides);

    aTensor->fillTensorWithRandomValues(-1.0f, 1.0f, 1234u);
    bTensor->fillTensorWithRandomValues(-1.0f, 1.0f, 5678u);
    cTensor->fillTensorWithValue(SENTINEL);

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers = {{1, aTensor->rawDeviceData()},
                                                             {3, bTensor->rawDeviceData()},
                                                             {2, cTensor->rawDeviceData()}};

    plan->execute(
        *_dummyHandle, deviceBuffers.data(), static_cast<uint32_t>(deviceBuffers.size()), nullptr);

    // See PointwiseExecutionCommon.hpp's executeAndVerify for why this is required: without
    // it, hostData() below returns the stale sentinel instead of MIOpen's device-side write.
    cTensor->markDeviceModified();

    int64_t count = 1;
    for(const auto d : cShape.dims)
    {
        count *= d;
    }

    for(int64_t linear = 0; linear < count; ++linear)
    {
        const auto cIndex = unflattenRowMajor(linear, cShape.dims);
        const auto bIndex = broadcastIndex(cIndex, bShape.dims);

        const double aVal = readElement<float>(*aTensor, cIndex);
        const double bVal = readElement<float>(*bTensor, bIndex);
        const double expected = referenceBinaryOp(PointwiseMode::ADD, aVal, bVal);
        const double actual = readElement<float>(*cTensor, cIndex);

        ASSERT_NEAR(expected, actual, tolerance<float>())
            << "mismatch at linear index " << linear << " -- output buffer's prior sentinel ("
            << SENTINEL << ") leaked into the result";
    }
}
