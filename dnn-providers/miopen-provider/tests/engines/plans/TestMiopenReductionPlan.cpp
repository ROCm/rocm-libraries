// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "common/ReductionCommon.hpp"
#include "engines/plans/MiopenReductionPlan.hpp"

using namespace miopen_plugin;
using namespace reduction_common;

namespace
{

using hipdnn_data_sdk::types::half;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_data_sdk::utilities::Workspace;
using hipdnn_flatbuffers_sdk::data_objects::ReductionAttributes;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;

// Builds the plan for `spec` the way MiopenReductionPlanBuilder::buildPlan does.
class ReductionPlanUnderTest
{
public:
    ReductionPlanUnderTest(const HipdnnMiopenHandle& handle, const ReductionGraphSpec& spec)
        : _builder(createReductionGraph(spec))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
        , _plan(handle,
                _graph.getNodeWrapper(0).attributesAs<ReductionAttributes>(),
                _graph.getTensorMap())
    {
    }

    const MiopenReductionPlan& plan() const
    {
        return _plan;
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    GraphWrapper _graph;
    MiopenReductionPlan _plan;
};

// Passes when fn throws a HipdnnPluginException with expectedStatus and, when given, a message
// containing expectedMessagePart.
template <typename Fn>
::testing::AssertionResult throwsPluginException(const Fn& fn,
                                                 hipdnnPluginStatus_t expectedStatus,
                                                 const std::string& expectedMessagePart = "")
{
    try
    {
        fn();
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        if(e.getStatus() == expectedStatus
           && e.getMessage().find(expectedMessagePart) != std::string::npos)
        {
            return ::testing::AssertionSuccess();
        }
        return ::testing::AssertionFailure()
               << "threw status " << e.getStatus() << " (expected " << expectedStatus
               << ", message containing \"" << expectedMessagePart << "\"): " << e.getMessage();
    }
    return ::testing::AssertionFailure() << "did not throw";
}

int64_t elementCount(const std::vector<int64_t>& dims)
{
    return std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<>());
}

// Calls fn(index) for every multi-index of `dims`, in row-major order.
template <typename Fn>
void forEachIndex(const std::vector<int64_t>& dims, const Fn& fn)
{
    std::vector<int64_t> index(dims.size(), 0);
    const auto count = elementCount(dims);
    for(int64_t n = 0; n < count; ++n)
    {
        fn(index);
        for(auto axis = dims.size(); axis > 0; --axis)
        {
            auto& coordinate = index[axis - 1];
            if(++coordinate < dims[axis - 1])
            {
                break;
            }
            coordinate = 0;
        }
    }
}

// Accumulates the reduced elements of one output element in double precision.
struct Accumulator
{
    void add(double value)
    {
        sum += value;
        sumAbs += std::abs(value);
        sumSquares += value * value;
        product *= value;
        min = std::min(min, value);
        max = std::max(max, value);
        ++count;
    }

    double result(ReductionMode mode) const
    {
        switch(mode)
        {
        case ReductionMode::ADD:
            return sum;
        case ReductionMode::MUL:
            return product;
        case ReductionMode::MIN_OP:
            return min;
        case ReductionMode::MAX_OP:
            return max;
        case ReductionMode::AMAX:
            return std::max(std::abs(min), std::abs(max));
        case ReductionMode::AVG:
            return sum / static_cast<double>(count);
        case ReductionMode::NORM1:
            return sumAbs;
        case ReductionMode::NORM2:
            return std::sqrt(sumSquares);
        case ReductionMode::NOT_SET:
        case ReductionMode::MUL_NO_ZEROS:
        default:
            ADD_FAILURE() << "no reference for this mode";
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    double sum = 0.0;
    double sumAbs = 0.0;
    double sumSquares = 0.0;
    double product = 1.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    int64_t count = 0;
};

// Reduces `xValues` (the logical elements of x, row-major over xDims) over the axes where yDims
// is 1. Returns the logical elements of y, row-major over yDims.
std::vector<double> referenceReduce(ReductionMode mode,
                                    const std::vector<int64_t>& xDims,
                                    const std::vector<int64_t>& yDims,
                                    const std::vector<double>& xValues)
{
    std::vector<Accumulator> accumulators(static_cast<size_t>(elementCount(yDims)));
    size_t xLinear = 0;
    forEachIndex(xDims, [&](const std::vector<int64_t>& xIndex) {
        int64_t yLinear = 0;
        for(size_t axis = 0; axis < xDims.size(); ++axis)
        {
            yLinear = yLinear * yDims[axis] + (yDims[axis] == 1 ? 0 : xIndex[axis]);
        }
        accumulators[static_cast<size_t>(yLinear)].add(xValues[xLinear++]);
    });

    std::vector<double> results;
    results.reserve(accumulators.size());
    for(const auto& accumulator : accumulators)
    {
        results.push_back(accumulator.result(mode));
    }
    return results;
}

struct ExecutionCase
{
    const char* name;
    // The data type is set by the fixture.
    ReductionGraphSpec spec;
    // Range of the uniformly distributed inputs. ADD and AVG keep it positive, so cancellation
    // cannot defeat the relative tolerance, and MUL keeps it around 1, so the product stays a
    // normal fp16 value.
    float low;
    float high;
};

std::ostream& operator<<(std::ostream& os, const ExecutionCase& executionCase)
{
    return os << executionCase.name;
}

// MIOpen's legacy reduce picks its method from the output element count and the reduced length n
// of each output (ReductionKernelConfigurator in projects/miopen/src/reducetensor.cpp, with the
// BlockSize of 256 from reduce_tunables.hpp). Several outputs run DirectThreadWise up to
// n = warpSize, DirectWarpWise up to 256, BlockWise up to 1024 and MultiBlock beyond, with one
// block per 1024 inputs and at most 32 blocks per output. A single output runs BlockWise up to
// n = 1024 and MultiBlock beyond, with one block per 256 inputs and no cap. MultiBlock reduces the
// partials of each output in a second pass: DirectThreadWise up to warpSize / 4 of them,
// DirectWarpWise up to 256, BlockWise beyond. The groups below run every method of both passes
// that a wave64 device can reach.
std::vector<ExecutionCase> getExecutionCases()
{
    auto norm1OverChannelsNhwc
        = ReductionGraphSpec::packed(ReductionMode::NORM1, {2, 3, 4, 5}, {2, 1, 4, 5});
    norm1OverChannelsNhwc.x.strides = nhwcStrides(norm1OverChannelsNhwc.x.dims);
    norm1OverChannelsNhwc.y.strides = nhwcStrides(norm1OverChannelsNhwc.y.dims);

    // C padded from 3 to 4 elements.
    auto addPaddedOutput
        = ReductionGraphSpec::packed(ReductionMode::ADD, {2, 3, 4, 5}, {2, 3, 1, 1});
    addPaddedOutput.y.strides = std::vector<int64_t>{4, 1, 1, 1};

    auto addSelfOverlappingInput = ReductionGraphSpec::packed(ReductionMode::ADD, {4, 8}, {4, 1});
    addSelfOverlappingInput.x.strides = std::vector<int64_t>{1, 1};

    return {
        // DirectThreadWise: several outputs, n <= 64.
        {"AddOverHwNchw",
         ReductionGraphSpec::packed(ReductionMode::ADD, {2, 3, 4, 5}, {2, 3, 1, 1}),
         0.5f,
         1.5f},
        {"MulInnermostAxis",
         ReductionGraphSpec::packed(ReductionMode::MUL, {2, 3, 4, 5}, {2, 3, 4, 1}),
         0.5f,
         1.5f},
        {"MinOpNonAdjacentAxes",
         ReductionGraphSpec::packed(ReductionMode::MIN_OP, {2, 3, 4, 5}, {1, 3, 1, 5}),
         -2.0f,
         2.0f},
        {"MaxOpInnermostAxis",
         ReductionGraphSpec::packed(ReductionMode::MAX_OP, {3, 7}, {3, 1}),
         -2.0f,
         2.0f},
        {"AmaxOutermostAxis",
         ReductionGraphSpec::packed(ReductionMode::AMAX, {4, 6}, {1, 6}),
         -2.0f,
         2.0f},
        {"Norm1OverChannelsNhwc", norm1OverChannelsNhwc, -2.0f, 2.0f},
        {"AddPaddedOutput", addPaddedOutput, 0.5f, 1.5f},
        // Stride-1 axes side by side, so x reads most of its 11 stored elements several times.
        {"AddSelfOverlappingInput", addSelfOverlappingInput, 0.5f, 1.5f},
        // DirectWarpWise: several outputs, 64 < n <= 256.
        {"Norm1OverHwWarpWise",
         ReductionGraphSpec::packed(ReductionMode::NORM1, {2, 3, 12, 12}, {2, 3, 1, 1}),
         -2.0f,
         2.0f},
        // BlockWise: a single output with n <= 1024, or several with 256 < n <= 1024.
        {"AvgFullReduction",
         ReductionGraphSpec::packed(ReductionMode::AVG, {2, 3, 4, 5}, {1, 1, 1, 1}),
         0.5f,
         1.5f},
        {"Norm2Rank1", ReductionGraphSpec::packed(ReductionMode::NORM2, {64}, {1}), -2.0f, 2.0f},
        {"Norm2OverHwBlockWise",
         ReductionGraphSpec::packed(ReductionMode::NORM2, {2, 3, 24, 24}, {2, 3, 1, 1}),
         -2.0f,
         2.0f},
        // MultiBlock over several outputs: n > 1024, with the partials kept in the workspace. 4
        // blocks per output make the second pass DirectThreadWise. n = 33792 asks for 33, which
        // are capped to 32, and makes it DirectWarpWise.
        {"AddMultiBlock",
         ReductionGraphSpec::packed(ReductionMode::ADD, {2, 4096}, {2, 1}),
         0.5f,
         1.5f},
        {"AddMultiBlockCapped",
         ReductionGraphSpec::packed(ReductionMode::ADD, {2, 33792}, {2, 1}),
         0.5f,
         1.5f},
        // MultiBlock to a single output, uncapped: 16, 32 and 384 blocks make the second pass
        // DirectThreadWise, DirectWarpWise and BlockWise. AVG keeps the largest fp16 result in
        // range.
        {"Norm1MultiBlockSingleOutput",
         ReductionGraphSpec::packed(ReductionMode::NORM1, {4096}, {1}),
         -2.0f,
         2.0f},
        {"Norm2MultiBlockSingleOutput",
         ReductionGraphSpec::packed(ReductionMode::NORM2, {8192}, {1}),
         -2.0f,
         2.0f},
        {"AvgMultiBlockSingleOutput",
         ReductionGraphSpec::packed(ReductionMode::AVG, {3, 256, 128}, {1, 1, 1}),
         0.5f,
         1.5f},
    };
}

// Runs every execution case through a real plan on the GPU and compares y with referenceReduce.
template <typename T>
class ReductionPlanExecutionFixture : public ::testing::TestWithParam<ExecutionCase>
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _handle = std::make_unique<HipdnnMiopenHandle>();
    }

    void executeAndCompareWithReference()
    {
        auto spec = GetParam().spec;
        spec.dataType = DATA_TYPE;
        const ReductionPlanUnderTest underTest(*_handle, spec);

        Tensor<T> x(spec.x.dims, spec.x.strides.value());
        Tensor<T> y(spec.y.dims, spec.y.strides.value());
        // An output element the plan fails to write stays NaN and fails the comparison.
        y.fillWithSentinelValue();

        std::mt19937 generator(hipdnn_test_sdk::utilities::getGlobalTestSeed());
        std::uniform_real_distribution<float> distribution(GetParam().low, GetParam().high);
        forEachIndex(spec.x.dims, [&](const std::vector<int64_t>& index) {
            x.setHostValue(static_cast<T>(distribution(generator)), index);
        });
        x.memory().markHostModified();
        // The reference reads back what x holds: the values after rounding to T, and, where input
        // strides overlap, the last value written to each shared element.
        std::vector<double> xValues;
        forEachIndex(spec.x.dims, [&](const std::vector<int64_t>& index) {
            xValues.push_back(static_cast<double>(static_cast<float>(x.getHostValue(index))));
        });

        const Workspace workspace(underTest.plan().getWorkspaceSize(*_handle));
        const std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers{
            {spec.x.uid, x.memory().deviceData()}, {spec.y.uid, y.memory().deviceData()}};
        underTest.plan().execute(*_handle,
                                 deviceBuffers.data(),
                                 static_cast<uint32_t>(deviceBuffers.size()),
                                 workspace.get());
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        y.memory().markDeviceModified();
        // Copies y back to the host; the const getHostValue below cannot.
        y.memory().hostData();

        const auto expected = referenceReduce(spec.mode, spec.x.dims, spec.y.dims, xValues);
        size_t yLinear = 0;
        forEachIndex(spec.y.dims, [&](const std::vector<int64_t>& index) {
            const auto actual = static_cast<double>(static_cast<float>(y.getHostValue(index)));
            const auto want = expected[yLinear];
            EXPECT_NEAR(actual, want, TOLERANCE * std::abs(want))
                << "at output element " << yLinear;
            ++yLinear;
        });
    }

    static constexpr DataType DATA_TYPE
        = std::is_same_v<T, half> ? DataType::HALF : DataType::FLOAT;
    // Relative to |expected|. The reference is exact at this scale: it works in double on the
    // values x holds after rounding to T. MIOpen computes in fp32, and on every path these cases
    // take, each input passes through at most 64 fp32 roundings on its way to y (none for MIN_OP,
    // MAX_OP and AMAX): DirectThreadWise chains up to warpSize = 64 inputs, the other methods
    // chain a few per thread into trees of depth log2(256) = 8, and AVG's scaling or NORM2's
    // square and square root add two. Summed terms all have one sign (see ExecutionCase::low),
    // so the relative error stays below about 64 * 2^-24 = 3.8e-6, for which fp32 allows 1e-5.
    // fp16 adds two roundings to T of at most 2^-11 each, of y and of the MultiBlock partials
    // MIOpen keeps in the workspace as T: 2 * 2^-11 + 3.8e-6 = 9.8e-4 fits in 1e-3.
    static constexpr double TOLERANCE = std::is_same_v<T, half> ? 1e-3 : 1e-5;

    std::unique_ptr<HipdnnMiopenHandle> _handle;
};

std::string executionCaseName(const ::testing::TestParamInfo<ExecutionCase>& info)
{
    return info.param.name;
}

} // namespace

using TestGpuMiopenReductionPlanFp32 = ReductionPlanExecutionFixture<float>;
using TestGpuMiopenReductionPlanFp16 = ReductionPlanExecutionFixture<half>;

TEST_P(TestGpuMiopenReductionPlanFp32, ExecuteMatchesHostReference)
{
    executeAndCompareWithReference();
}

TEST_P(TestGpuMiopenReductionPlanFp16, ExecuteMatchesHostReference)
{
    executeAndCompareWithReference();
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestGpuMiopenReductionPlanFp32,
                         ::testing::ValuesIn(getExecutionCases()),
                         executionCaseName);

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestGpuMiopenReductionPlanFp16,
                         ::testing::ValuesIn(getExecutionCases()),
                         executionCaseName);

// Plan construction, workspace and device-buffer handling. SetUp builds the plan for an fp32 ADD
// of x [2, 4, 4, 5] into y [2, 4, 1, 1]. Both spans are multiples of 16 bytes, so placing one
// tensor directly after the other keeps the default 16-byte tensor alignment.
class TestGpuMiopenReductionPlan : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _handle = std::make_unique<HipdnnMiopenHandle>();
        _underTest = std::make_unique<ReductionPlanUnderTest>(*_handle, _spec);
        _workspace = std::make_unique<Workspace<>>(_underTest->plan().getWorkspaceSize(*_handle));
        // One allocation that every x/y placement below fits in.
        _buffers = std::make_unique<Workspace<>>(X_SPAN_BYTES + Y_SPAN_BYTES);
        ASSERT_EQ(hipMemset(_buffers->get(), 0, X_SPAN_BYTES + Y_SPAN_BYTES), hipSuccess);
    }

    void* bufferAt(size_t offsetBytes) const
    {
        return static_cast<char*>(_buffers->get()) + offsetBytes;
    }

    void execute(const std::vector<hipdnnPluginDeviceBuffer_t>& deviceBuffers,
                 void* workspace) const
    {
        _underTest->plan().execute(
            *_handle, deviceBuffers.data(), static_cast<uint32_t>(deviceBuffers.size()), workspace);
    }

    void execute(void* xPtr, void* yPtr) const
    {
        execute({{ReductionGraphSpec::X_UID, xPtr}, {ReductionGraphSpec::Y_UID, yPtr}},
                _workspace->get());
    }

    // Bytes from the first element to one past the last, for the packed tensors of _spec.
    static constexpr size_t X_SPAN_BYTES = size_t{2} * 4 * 4 * 5 * sizeof(float);
    static constexpr size_t Y_SPAN_BYTES = size_t{2} * 4 * sizeof(float);

    const ReductionGraphSpec _spec
        = ReductionGraphSpec::packed(ReductionMode::ADD, {2, 4, 4, 5}, {2, 4, 1, 1});
    std::unique_ptr<HipdnnMiopenHandle> _handle;
    std::unique_ptr<ReductionPlanUnderTest> _underTest;
    std::unique_ptr<Workspace<>> _workspace;
    std::unique_ptr<Workspace<>> _buffers;
};

// The gate declines modes with no miopenReduceTensorOp_t, so reaching the plan with one is a
// provider bug, reported as an internal error rather than a bad parameter.
TEST_F(TestGpuMiopenReductionPlan, ConstructorThrowsInternalErrorForModeWithoutMiopenOp)
{
    // Otherwise valid, so creating the tensors succeeds and the mode lookup is what throws.
    auto spec = _spec;
    spec.mode = ReductionMode::MUL_NO_ZEROS;

    EXPECT_TRUE(
        throwsPluginException([&] { const ReductionPlanUnderTest underTest(*_handle, spec); },
                              HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                              "does not map to a miopenReduceTensorOp_t"));
}

TEST_F(TestGpuMiopenReductionPlan, GetWorkspaceSizeMatchesMiopenQuery)
{
    const auto workspaceSize = _underTest->plan().getWorkspaceSize(*_handle);

    EXPECT_GT(workspaceSize, 0u);
    EXPECT_EQ(
        workspaceSize,
        queryMiopenReductionWorkspaceSize(_handle->miopenHandle, MIOPEN_REDUCE_TENSOR_ADD, _spec));
}

TEST_F(TestGpuMiopenReductionPlan, ExecuteThrowsForNullWorkspace)
{
    EXPECT_TRUE(throwsPluginException(
        [&] {
            execute({{ReductionGraphSpec::X_UID, bufferAt(0)},
                     {ReductionGraphSpec::Y_UID, bufferAt(X_SPAN_BYTES)}},
                    nullptr);
        },
        HIPDNN_PLUGIN_STATUS_BAD_PARAM));
}

TEST_F(TestGpuMiopenReductionPlan, ExecuteThrowsForMissingInputBuffer)
{
    EXPECT_TRUE(throwsPluginException(
        [&] { execute({{ReductionGraphSpec::Y_UID, bufferAt(X_SPAN_BYTES)}}, _workspace->get()); },
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE));
}

TEST_F(TestGpuMiopenReductionPlan, ExecuteThrowsForMissingOutputBuffer)
{
    EXPECT_TRUE(throwsPluginException(
        [&] { execute({{ReductionGraphSpec::X_UID, bufferAt(0)}}, _workspace->get()); },
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE));
}

TEST_F(TestGpuMiopenReductionPlan, ExecuteThrowsWhenOutputAliasesInput)
{
    EXPECT_TRUE(throwsPluginException([&] { execute(bufferAt(0), bufferAt(0)); },
                                      HIPDNN_PLUGIN_STATUS_BAD_PARAM));
}

TEST_F(TestGpuMiopenReductionPlan, ExecuteThrowsWhenOutputStartsOnLastInputElement)
{
    EXPECT_TRUE(
        throwsPluginException([&] { execute(bufferAt(0), bufferAt(X_SPAN_BYTES - sizeof(float))); },
                              HIPDNN_PLUGIN_STATUS_BAD_PARAM));
}

TEST_F(TestGpuMiopenReductionPlan, ExecuteThrowsWhenInputStartsOnLastOutputElement)
{
    EXPECT_TRUE(
        throwsPluginException([&] { execute(bufferAt(Y_SPAN_BYTES - sizeof(float)), bufferAt(0)); },
                              HIPDNN_PLUGIN_STATUS_BAD_PARAM));
}

TEST_F(TestGpuMiopenReductionPlan, ExecuteAcceptsOutputDirectlyAfterInput)
{
    EXPECT_NO_THROW(execute(bufferAt(0), bufferAt(X_SPAN_BYTES)));
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(TestGpuMiopenReductionPlan, ExecuteAcceptsInputDirectlyAfterOutput)
{
    EXPECT_NO_THROW(execute(bufferAt(Y_SPAN_BYTES), bufferAt(0)));
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}
