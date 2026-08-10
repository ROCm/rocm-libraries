// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "ingestor_poc/NativeSymbolNames.hpp"
#include "ingestor_poc/PointwiseAddDispatchHandler.hpp"
#include "tests/ingestor_poc/PointwiseAddGraphs.hpp"

/**
 * @file TestPointwiseAddDispatchHandler.cpp
 * @brief The POC's dispatch: workspace sizing, and a real compile-and-launch.
 *
 * The launch tests run on device deliberately. A recorded no-op would exercise neither
 * the runtime compile nor the uid-to-pointer resolution, which are the two parts most
 * likely to be wrong when real kernels arrive.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::ingestor_poc;
using namespace hip_kernel_provider::ingestor_poc::testing;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;

KernelDefinition makeKernel(int64_t blockSize, const std::string& dtype)
{
    KernelDefinition kernel;
    kernel.kernelId = "test.kernel";
    kernel.packId = "test.pack";
    kernel.dispatchId = "test.dispatch";
    kernel.sourceFile = "PointwiseAdd.cpp";
    kernel.entryPoint = "PointwiseAdd";
    kernel.metadata
        = {{std::string(BLOCK_SIZE_FIELD), blockSize}, {std::string(DTYPE_FIELD), dtype}};
    return kernel;
}

hipDeviceProp_t currentDeviceProperties()
{
    hipDeviceProp_t properties{};
    int deviceId = 0;
    if(hipGetDevice(&deviceId) == hipSuccess)
    {
        static_cast<void>(hipGetDeviceProperties(&properties, deviceId));
    }
    return properties;
}

class GraphFixture
{
public:
    explicit GraphFixture(flatbuffers::FlatBufferBuilder builder)
        : _builder(std::move(builder))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
        , _properties(currentDeviceProperties())
    {
    }

    MatchContext context() const
    {
        return MatchContext{_graph, 0, _properties};
    }

    const hipDeviceProp_t& deviceProperties() const
    {
        return _properties;
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper _graph;
    hipDeviceProp_t _properties;
};

/// Device buffers for one 1-element add, freed on scope exit.
class AddBuffers
{
public:
    AddBuffers(float a, float b)
    {
        EXPECT_EQ(hipSuccess, hipMalloc(&_a, sizeof(float)));
        EXPECT_EQ(hipSuccess, hipMalloc(&_b, sizeof(float)));
        EXPECT_EQ(hipSuccess, hipMalloc(&_c, sizeof(float)));
        EXPECT_EQ(hipSuccess, hipMemcpy(_a, &a, sizeof(float), hipMemcpyHostToDevice));
        EXPECT_EQ(hipSuccess, hipMemcpy(_b, &b, sizeof(float), hipMemcpyHostToDevice));
    }

    ~AddBuffers()
    {
        static_cast<void>(hipFree(_a));
        static_cast<void>(hipFree(_b));
        static_cast<void>(hipFree(_c));
    }

    AddBuffers(const AddBuffers&) = delete;
    AddBuffers& operator=(const AddBuffers&) = delete;

    std::array<hipdnnPluginDeviceBuffer_t, 3> descriptors() const
    {
        return {hipdnnPluginDeviceBuffer_t{INPUT_A_UID, _a},
                hipdnnPluginDeviceBuffer_t{INPUT_B_UID, _b},
                hipdnnPluginDeviceBuffer_t{OUTPUT_UID, _c}};
    }

    float readResult() const
    {
        float result = 0.0f;
        EXPECT_EQ(hipSuccess, hipMemcpy(&result, _c, sizeof(float), hipMemcpyDeviceToHost));
        return result;
    }

private:
    void* _a = nullptr;
    void* _b = nullptr;
    void* _c = nullptr;
};

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

TEST(TestPointwiseAddDispatch, ReportsWorkspaceFromKernelMetadata)
{
    const HipMlopsKernelCompiler compiler;
    const PointwiseAddDispatchHandler handler(compiler, hipDeviceProp_t{});

    // The two surviving kernels report different requirements, so the engine's
    // "maximum across survivors" has something to actually maximize over.
    EXPECT_EQ(handler.workspaceBytes(makeKernel(64, "FLOAT")), 0U);
    EXPECT_EQ(handler.workspaceBytes(makeKernel(256, "FLOAT")), 1024U);
}

TEST(TestPointwiseAddDispatch, WorkspaceDoesNotDependOnTheGraph)
{
    const HipMlopsKernelCompiler compiler;
    const PointwiseAddDispatchHandler handler(compiler, hipDeviceProp_t{});

    // Workspace is asked before a kernel is chosen and before any plan exists, so it
    // must be answerable from the kernel alone.
    EXPECT_EQ(handler.workspaceBytes(makeKernel(256, "FLOAT")),
              handler.workspaceBytes(makeKernel(256, "HALF")));
}

// ---------------------------------------------------------------------------
// Prepare and launch
// ---------------------------------------------------------------------------

TEST(TestPointwiseAddDispatch, LaunchesARealAddOnDevice)
{
    SKIP_IF_NO_DEVICES();

    const GraphFixture fixture(buildPointwiseGraph());
    const HipMlopsKernelCompiler compiler;
    const PointwiseAddDispatchHandler handler(compiler, fixture.deviceProperties());

    const auto prepared = handler.prepare(fixture.context(), makeKernel(64, "FLOAT"));
    ASSERT_NE(prepared, nullptr);

    const AddBuffers buffers(3.0f, 4.0f);
    const auto descriptors = buffers.descriptors();
    const Handle handle;

    handler.launch(handle, *prepared, descriptors.data(), descriptors.size(), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

    EXPECT_FLOAT_EQ(buffers.readResult(), 7.0f);
}

TEST(TestPointwiseAddDispatch, LaunchesTheSameResultForEitherBlockSize)
{
    SKIP_IF_NO_DEVICES();

    const GraphFixture fixture(buildPointwiseGraph());
    const HipMlopsKernelCompiler compiler;
    const PointwiseAddDispatchHandler handler(compiler, fixture.deviceProperties());

    // block_size reaches the compiler and the launch geometry, so both kernels are
    // genuinely different builds; a one-element add must still agree.
    for(const int64_t blockSize : {64, 256})
    {
        const auto prepared = handler.prepare(fixture.context(), makeKernel(blockSize, "FLOAT"));
        const AddBuffers buffers(1.5f, 2.25f);
        const auto descriptors = buffers.descriptors();
        const Handle handle;

        handler.launch(handle, *prepared, descriptors.data(), descriptors.size(), nullptr);
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

        EXPECT_FLOAT_EQ(buffers.readResult(), 3.75f) << "block size " << blockSize;
    }
}

TEST(TestPointwiseAddDispatch, PreparedLaunchIsReusableAcrossExecutions)
{
    SKIP_IF_NO_DEVICES();

    const GraphFixture fixture(buildPointwiseGraph());
    const HipMlopsKernelCompiler compiler;
    const PointwiseAddDispatchHandler handler(compiler, fixture.deviceProperties());

    // A plan is built once and may execute many times with different buffers, so
    // preparation must hold nothing tied to one execution.
    const auto prepared = handler.prepare(fixture.context(), makeKernel(64, "FLOAT"));
    const Handle handle;

    for(const auto& [a, b, expected] : std::vector<std::array<float, 3>>{
            {1.0f, 2.0f, 3.0f}, {10.0f, -4.0f, 6.0f}, {0.5f, 0.25f, 0.75f}})
    {
        const AddBuffers buffers(a, b);
        const auto descriptors = buffers.descriptors();

        handler.launch(handle, *prepared, descriptors.data(), descriptors.size(), nullptr);
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

        EXPECT_FLOAT_EQ(buffers.readResult(), expected);
    }
}

TEST(TestPointwiseAddDispatch, RefusesToPrepareAGraphTheMatcherRejects)
{
    SKIP_IF_NO_DEVICES();

    const GraphFixture fixture(buildTwoNodePointwiseGraph());
    const HipMlopsKernelCompiler compiler;
    const PointwiseAddDispatchHandler handler(compiler, fixture.deviceProperties());

    // Preparation reads the operand binding the matcher established, so a graph that
    // never matched cannot be prepared.
    EXPECT_THROW(handler.prepare(fixture.context(), makeKernel(64, "FLOAT")),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
