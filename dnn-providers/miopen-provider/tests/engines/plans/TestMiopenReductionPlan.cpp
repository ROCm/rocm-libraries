// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "engines/plans/MiopenReductionPlan.hpp"

using namespace miopen_plugin;

namespace
{

using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_data_sdk::utilities::Workspace;
using hipdnn_flatbuffers_sdk::data_objects::ReductionAttributes;
using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_test_sdk::utilities::createReductionGraph;
using hipdnn_test_sdk::utilities::ReductionGraphSpec;
using hipdnn_test_sdk::utilities::ReductionTensorSpec;

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

} // namespace

// Plan construction, execution, workspace and device-buffer handling. SetUp builds the plan for
// an fp32 ADD of x [2, 4, 4, 5] into y [2, 4, 1, 1]. Both spans are multiples of 16 bytes, so
// placing one tensor directly after the other keeps the default 16-byte tensor alignment.
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
        ASSERT_EQ(hipMemset(_buffers->get(), FILL_BYTE, X_SPAN_BYTES + Y_SPAN_BYTES), hipSuccess);
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
    // Every float of the buffers reads about 0.747, so a reduction that runs changes the bytes it
    // writes.
    static constexpr unsigned char FILL_BYTE = 0x3F;

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

// Checks what the plan passes to MIOpen through the result, with hand-computed exact values.
// x [2, 3] has strides {1, 2}, so its storage {1, -2, -3, 4, 5, -6} holds the rows {1, -3, 5} and
// {-2, 4, -6}, whose NORM1 is {9, 12}. Reading x with packed strides gives {6, 15}, and ADD gives
// {3, -4}. y [2, 1] has stride 2, so a packed y would write its second element into the gap. y
// starts as NaN, so an element that is not written, or is written with beta != 0, stays NaN.
TEST_F(TestGpuMiopenReductionPlan, ExecuteReducesStridedInputIntoPaddedOutput)
{
    ReductionGraphSpec spec;
    spec.mode = ReductionMode::NORM1;
    spec.x = ReductionTensorSpec::dense(ReductionGraphSpec::X_UID, {2, 3}, {1, 2});
    spec.y = ReductionTensorSpec::dense(ReductionGraphSpec::Y_UID, {2, 1}, {2, 1});
    const ReductionPlanUnderTest underTest(*_handle, spec);

    Tensor<float> x(spec.x.dims, spec.x.strides.value());
    const std::vector<float> xStorage{1.0f, -2.0f, -3.0f, 4.0f, 5.0f, -6.0f};
    x.fillWithData(xStorage.data(), xStorage.size() * sizeof(float));
    Tensor<float> y(spec.y.dims, spec.y.strides.value());
    y.fillWithSentinelValue();

    const Workspace workspace(underTest.plan().getWorkspaceSize(*_handle));
    const std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers{
        {ReductionGraphSpec::X_UID, x.memory().deviceData()},
        {ReductionGraphSpec::Y_UID, y.memory().deviceData()}};
    underTest.plan().execute(*_handle,
                             deviceBuffers.data(),
                             static_cast<uint32_t>(deviceBuffers.size()),
                             workspace.get());
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    y.memory().markDeviceModified();
    // Copies y back to the host; the const getHostValue below cannot.
    y.memory().hostData();

    EXPECT_EQ(y.getHostValue(0, 0), 9.0f);
    EXPECT_EQ(y.getHostValue(1, 0), 12.0f);
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

TEST_F(TestGpuMiopenReductionPlan, ExecuteThrowsForMissingDeviceBuffer)
{
    EXPECT_TRUE(throwsPluginException(
        [&] { execute({{ReductionGraphSpec::Y_UID, bufferAt(X_SPAN_BYTES)}}, _workspace->get()); },
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE))
        << "x missing";
    EXPECT_TRUE(throwsPluginException(
        [&] { execute({{ReductionGraphSpec::X_UID, bufferAt(0)}}, _workspace->get()); },
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE))
        << "y missing";
}

// y starts on the last element of x. The guard must throw before MIOpen runs: a launch would
// overwrite that element with y[0], so x must still hold the fill pattern afterwards.
TEST_F(TestGpuMiopenReductionPlan, ExecuteThrowsWhenOutputStartsOnLastInputElement)
{
    EXPECT_TRUE(
        throwsPluginException([&] { execute(bufferAt(0), bufferAt(X_SPAN_BYTES - sizeof(float))); },
                              HIPDNN_PLUGIN_STATUS_BAD_PARAM));

    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    std::vector<unsigned char> xBytes(X_SPAN_BYTES);
    ASSERT_EQ(hipMemcpy(xBytes.data(), bufferAt(0), X_SPAN_BYTES, hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(xBytes, std::vector<unsigned char>(X_SPAN_BYTES, FILL_BYTE));
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
