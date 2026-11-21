// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Executes MiopenResampleFwdPlan on small inputs with values worked out by hand. Each test fails
// when the plan's own code is wrong: the descriptor or tensors it builds from the configuration,
// the workspace it reports and checks, the alpha and beta it passes, or the index it copies out.
// How MIOpen pools is MIOpen's to test.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "MiopenApi.hpp"
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "engines/plans/MiopenResampleFwdChecks.hpp"
#include "engines/plans/MiopenResampleFwdPlan.hpp"

using namespace miopen_plugin;

using hipdnn_data_sdk::types::half;
using hipdnn_data_sdk::utilities::iterateAlongDimensions;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::ResampleMode;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_test_sdk::utilities::createResampleFwdGraph;
using hipdnn_test_sdk::utilities::ResampleFwdGraphSpec;

namespace
{

constexpr float Y_SENTINEL = -12345.0f;
constexpr int INDEX_SENTINEL = -7;

std::vector<float> ascending(int64_t count, float first = 0.0f)
{
    std::vector<float> values(static_cast<size_t>(count));
    std::iota(values.begin(), values.end(), first);
    return values;
}

// Y and the indices in row-major logical order over the Y dims; `index` is empty when the node
// has no index tensor.
struct PoolingResult
{
    std::vector<float> y;
    std::vector<int64_t> index;
};

void expectYNear(const std::vector<float>& actual,
                 const std::vector<float>& expected,
                 double relativeTolerance)
{
    ASSERT_EQ(actual.size(), expected.size());
    for(size_t i = 0; i < actual.size(); ++i)
    {
        const auto want = static_cast<double>(expected[i]);
        EXPECT_NEAR(
            static_cast<double>(actual[i]), want, relativeTolerance * std::max(1.0, std::abs(want)))
            << "Y element " << i;
    }
}

void expectMatches(const PoolingResult& actual,
                   const PoolingResult& expected,
                   double relativeTolerance = 1e-5)
{
    expectYNear(actual.y, expected.y, relativeTolerance);
    EXPECT_EQ(actual.index, expected.index);
}

// Host/device tensors for one execution: X filled from logical (row-major) values, Y and the
// index pre-filled with sentinels so unwritten elements show up.
template <typename T, typename IndexT>
class PoolingTensors
{
public:
    PoolingTensors(const ResampleFwdGraphSpec& spec, const std::vector<float>& xValues)
        : _spec(spec)
        , _x(spec.x.dims, spec.x.strides.value())
        , _y(spec.y.dims, spec.y.strides.value())
    {
        size_t next = 0;
        iterateAlongDimensions(_spec.x.dims, [&](const std::vector<int64_t>& index) {
            _x.setHostValue(static_cast<T>(xValues.at(next++)), index);
        });
        _x.markHostModified();

        _y.fillWithValue(static_cast<T>(Y_SENTINEL));

        if(_spec.index.has_value())
        {
            _index.emplace(_spec.index->dims, _spec.index->strides.value());
            _index->fillWithValue(static_cast<IndexT>(INDEX_SENTINEL));
        }
    }

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers(bool includeIndex = true)
    {
        std::vector<hipdnnPluginDeviceBuffer_t> buffers{{_spec.x.uid, _x.rawDeviceData()},
                                                        {_spec.y.uid, _y.rawDeviceData()}};
        if(includeIndex && _index.has_value())
        {
            buffers.push_back({_spec.index->uid, _index->rawDeviceData()});
        }
        return buffers;
    }

    // Reads Y and the index back in logical order. The device work must have completed.
    PoolingResult read()
    {
        PoolingResult result;

        _y.markDeviceModified();
        _y.rawHostData();
        iterateAlongDimensions(_spec.y.dims, [&](const std::vector<int64_t>& index) {
            result.y.push_back(static_cast<float>(_y.getHostValue(index)));
        });

        if(_index.has_value())
        {
            _index->markDeviceModified();
            _index->rawHostData();
            iterateAlongDimensions(_spec.index->dims, [&](const std::vector<int64_t>& index) {
                result.index.push_back(static_cast<int64_t>(_index->getHostValue(index)));
            });
        }
        return result;
    }

private:
    ResampleFwdGraphSpec _spec;
    Tensor<T> _x;
    Tensor<T> _y;
    std::optional<Tensor<IndexT>> _index;
};

std::unique_ptr<MiopenResampleFwdPlan> buildPlan(const ResampleFwdGraphSpec& spec)
{
    auto fbb = createResampleFwdGraph(spec);
    const GraphWrapper graph(fbb.GetBufferPointer(), fbb.GetSize());
    return std::make_unique<MiopenResampleFwdPlan>(resample_fwd_applicability::resolve(graph));
}

// 4x4 input pooled 2x2 / 2 into 2x2. Window maxima (row, col): (0,1)=5, (1,2)=8, (3,1)=11,
// (2,3)=7.
const std::vector<float>& planarMaxInput()
{
    // clang-format off
    static const std::vector<float> s_values{
        1,  5, 2, 0,
        3,  4, 8, 6,
        9,  2, 1, 7,
        0, 11, 3, 2,
    };
    // clang-format on
    return s_values;
}

const PoolingResult& planarMaxExpected()
{
    static const PoolingResult s_expected{{5, 8, 11, 7}, {1, 6, 13, 11}};
    return s_expected;
}

ResampleFwdGraphSpec planarMaxSpec(std::optional<DataType> indexType,
                                   DataType dataType = DataType::FLOAT)
{
    return ResampleFwdGraphSpec::packed(
        ResampleMode::MAXPOOL, {1, 1, 4, 4}, {2, 2}, {2, 2}, {0, 0}, dataType, indexType);
}

// `channels` slices of 3x3, 3x3 window, stride 1, padding 1: corner windows see 4 valid inputs,
// edge windows 6 and the centre window all 9.
ResampleFwdGraphSpec paddedAverageSpec(int64_t channels = 1)
{
    return ResampleFwdGraphSpec::packed(
        ResampleMode::AVGPOOL_EXCLUDE_PADDING, {1, channels, 3, 3}, {3, 3}, {1, 1}, {1, 1});
}

// paddedAverageSpec over 1..9: window sums divided by 4 (corners), 6 (edges) and 9 (centre).
const std::vector<float>& paddedAverageExpected()
{
    static const std::vector<float> s_expected{
        3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f};
    return s_expected;
}

class ResampleFwdPlanFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _handle = std::make_unique<HipdnnMiopenHandle>();
    }

    void synchronize() const
    {
        hipStream_t stream = nullptr;
        ASSERT_EQ(miopenGetStream(_handle->miopenHandle, &stream), miopenStatusSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    }

    // Builds the plan for `spec`, executes it with a workspace of the size the plan asks for and
    // returns Y (and the index) in logical order.
    template <typename T = float, typename IndexT = int32_t>
    PoolingResult run(const ResampleFwdGraphSpec& spec, const std::vector<float>& xValues)
    {
        const auto plan = buildPlan(spec);
        PoolingTensors<T, IndexT> tensors(spec, xValues);
        auto buffers = tensors.deviceBuffers();

        const hipdnn_data_sdk::utilities::Workspace workspace(plan->getWorkspaceSize(*_handle));
        plan->execute(
            *_handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get());
        synchronize();

        return tensors.read();
    }

    std::unique_ptr<HipdnnMiopenHandle> _handle;
};

} // namespace

class TestGpuMiopenResampleFwdPlan : public ResampleFwdPlanFixture
{
};

// ============================================================================
// Workspace: MIOpen's size for the index type the plan passes.
// ============================================================================

TEST_F(TestGpuMiopenResampleFwdPlan, WorkspaceHoldsInt32Indices)
{
    const auto plan = buildPlan(planarMaxSpec(DataType::INT32));
    EXPECT_EQ(plan->getWorkspaceSize(*_handle), 4 * sizeof(int32_t));
}

TEST_F(TestGpuMiopenResampleFwdPlan, WorkspaceHoldsInt8Indices)
{
    const auto plan = buildPlan(planarMaxSpec(DataType::INT8));
    EXPECT_EQ(plan->getWorkspaceSize(*_handle), 4 * sizeof(int8_t));
}

// ============================================================================
// Execution. Y and the index start as sentinels, so a wrong alpha or beta, or an index that is
// not copied out, shows in the values.
// ============================================================================

TEST_F(TestGpuMiopenResampleFwdPlan, MaxPoolWritesValuesAndInt32Indices)
{
    expectMatches(run(planarMaxSpec(DataType::INT32), planarMaxInput()), planarMaxExpected(), 0.0);
}

TEST_F(TestGpuMiopenResampleFwdPlan, MaxPoolWritesInt8Indices)
{
    expectMatches(run<float, int8_t>(planarMaxSpec(DataType::INT8), planarMaxInput()),
                  planarMaxExpected(),
                  0.0);
}

TEST_F(TestGpuMiopenResampleFwdPlan, MaxPoolWritesHalfValues)
{
    // The inputs are small integers, exact in half precision.
    expectMatches(
        run<half, int32_t>(planarMaxSpec(DataType::INT32, DataType::HALF), planarMaxInput()),
        planarMaxExpected(),
        0.0);
}

TEST_F(TestGpuMiopenResampleFwdPlan, MaxPoolWithoutIndexWritesValues)
{
    const auto result = run(planarMaxSpec(std::nullopt), planarMaxInput());
    expectYNear(result.y, planarMaxExpected().y, 0.0);
    EXPECT_TRUE(result.index.empty());
}

TEST_F(TestGpuMiopenResampleFwdPlan, MaxPool3dWritesInt32Indices)
{
    // 4x4x4 input pooled 2x2x2 / 2 into 2x2x2. The inner 2x2x2 cube (every coordinate 1 or 2)
    // holds 10..17 and the rest is 0, so each window's maximum sits at the window corner nearest
    // the centre, a different (d, h, w) offset in every window. Indices are flattened offsets into
    // the 4x4x4 input: d * 16 + h * 4 + w.
    std::vector<float> x(64, 0.0f);
    float next = 10.0f;
    for(size_t d = 1; d <= 2; ++d)
    {
        for(size_t h = 1; h <= 2; ++h)
        {
            for(size_t w = 1; w <= 2; ++w)
            {
                x.at(d * 16 + h * 4 + w) = next++;
            }
        }
    }

    const auto spec = ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL,
                                                   {1, 1, 4, 4, 4},
                                                   {2, 2, 2},
                                                   {2, 2, 2},
                                                   {0, 0, 0},
                                                   DataType::FLOAT,
                                                   DataType::INT32);
    const PoolingResult expected{{10, 11, 12, 13, 14, 15, 16, 17},
                                 {21, 22, 25, 26, 37, 38, 41, 42}};
    expectMatches(run(spec, x), expected, 0.0);
}

TEST_F(TestGpuMiopenResampleFwdPlan, AvgExcludePaddingDividesByValidCount)
{
    const auto result = run(paddedAverageSpec(), ascending(9, 1.0f));
    expectYNear(result.y, paddedAverageExpected(), 1e-5);
    EXPECT_TRUE(result.index.empty());
}

TEST_F(TestGpuMiopenResampleFwdPlan, AvgPoolingNhwcWritesValues)
{
    // Two channels: 1..9 and 10..18, whose averages are those of channel 0 plus 9. The plan hands
    // MIOpen channels-last strides, so MIOpen transposes X and Y through the workspace.
    auto spec = paddedAverageSpec(/*channels=*/2);
    spec.makeChannelsLast();
    ASSERT_GT(buildPlan(spec)->getWorkspaceSize(*_handle), 0u);

    auto expected = paddedAverageExpected();
    for(const auto value : paddedAverageExpected())
    {
        expected.push_back(value + 9.0f);
    }
    expectYNear(run(spec, ascending(18, 1.0f)).y, expected, 1e-5);
}

TEST_F(TestGpuMiopenResampleFwdPlan, PoolingChannelsLastStyleInputIntoNchwOutput)
{
    // X {2, 3, 1, 1} carries NHWC strides {3, 1, 3, 3}, which address like packed NCHW; Y is
    // packed NCHW. Taken at face value MIOpen would read X as NHWC and Y as NCHW; the plan hands
    // it canonical NCHW strides for both. Every 2x2 window holds the single valid pixel.
    auto spec = ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL,
                                             {2, 3, 1, 1},
                                             {2, 2},
                                             {1, 1},
                                             {1, 1},
                                             DataType::FLOAT,
                                             DataType::INT32);
    spec.x.strides = std::vector<int64_t>{3, 1, 3, 3};
    ASSERT_EQ(spec.y.strides.value(), (std::vector<int64_t>{12, 4, 2, 1}));

    const auto x = ascending(6, 1.0f);
    PoolingResult expected;
    for(const auto value : x)
    {
        expected.y.insert(expected.y.end(), 4, value);
        expected.index.insert(expected.index.end(), 4, 0);
    }
    expectMatches(run(spec, x), expected, 0.0);
}

// ============================================================================
// Execution errors
// ============================================================================

TEST_F(TestGpuMiopenResampleFwdPlan, ExecuteThrowsInvalidValueWithoutRequiredWorkspace)
{
    const auto spec = planarMaxSpec(DataType::INT32);
    const auto plan = buildPlan(spec);
    ASSERT_GT(plan->getWorkspaceSize(*_handle), 0u);

    PoolingTensors<float, int32_t> tensors(spec, planarMaxInput());
    auto buffers = tensors.deviceBuffers();
    try
    {
        plan->execute(*_handle, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr);
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    }
}

TEST_F(TestGpuMiopenResampleFwdPlan, ExecuteThrowsInvalidValueWithoutWorkspaceForNhwcAveragePooling)
{
    // No index, but channels-last tensors still need MIOpen's transpose buffers.
    auto spec = ResampleFwdGraphSpec::packed(
        ResampleMode::AVGPOOL_INCLUDE_PADDING, {2, 3, 5, 5}, {3, 3}, {2, 2}, {1, 1});
    spec.makeChannelsLast();
    const auto plan = buildPlan(spec);
    ASSERT_GT(plan->getWorkspaceSize(*_handle), 0u);

    PoolingTensors<float, int32_t> tensors(spec, ascending(150));
    auto buffers = tensors.deviceBuffers();
    try
    {
        plan->execute(*_handle, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr);
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    }
}

TEST_F(TestGpuMiopenResampleFwdPlan, ExecuteSucceedsWithoutWorkspaceWhenNoneIsRequired)
{
    const auto spec = paddedAverageSpec();
    const auto plan = buildPlan(spec);
    ASSERT_EQ(plan->getWorkspaceSize(*_handle), 0u);

    PoolingTensors<float, int32_t> tensors(spec, ascending(9, 1.0f));
    auto buffers = tensors.deviceBuffers();
    ASSERT_NO_THROW(
        plan->execute(*_handle, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr));
    synchronize();

    expectYNear(tensors.read().y, paddedAverageExpected(), 1e-5);
}

TEST_F(TestGpuMiopenResampleFwdPlan, ExecuteThrowsInvalidValueForMissingIndexBuffer)
{
    const auto spec = planarMaxSpec(DataType::INT32);
    const auto plan = buildPlan(spec);

    PoolingTensors<float, int32_t> tensors(spec, planarMaxInput());
    auto buffers = tensors.deviceBuffers(/*includeIndex=*/false);
    const hipdnn_data_sdk::utilities::Workspace workspace(plan->getWorkspaceSize(*_handle));
    try
    {
        plan->execute(
            *_handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get());
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    }
}

TEST_F(TestGpuMiopenResampleFwdPlan, ExecuteThrowsInvalidValueForNullIndexPointer)
{
    const auto spec = planarMaxSpec(DataType::INT32);
    const auto plan = buildPlan(spec);

    PoolingTensors<float, int32_t> tensors(spec, planarMaxInput());
    auto buffers = tensors.deviceBuffers();
    ASSERT_EQ(buffers.size(), 3u);
    ASSERT_EQ(buffers[2].uid, ResampleFwdGraphSpec::INDEX_UID);
    buffers[2].ptr = nullptr;

    const hipdnn_data_sdk::utilities::Workspace workspace(plan->getWorkspaceSize(*_handle));
    try
    {
        plan->execute(
            *_handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get());
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    }
    synchronize();

    // The plan threw before MIOpen ran: Y and the index still hold their sentinels.
    const auto result = tensors.read();
    EXPECT_EQ(result.y, std::vector<float>(4, Y_SENTINEL));
    EXPECT_EQ(result.index, std::vector<int64_t>(4, INDEX_SENTINEL));
}
