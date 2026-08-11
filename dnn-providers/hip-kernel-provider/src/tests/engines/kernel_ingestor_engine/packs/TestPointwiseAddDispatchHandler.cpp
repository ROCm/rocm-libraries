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

#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "core/Container.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddDispatchHandler.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddMatchers.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddSymbols.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseAddTestGraphs.hpp"

/**
 * @file TestPointwiseAddDispatchHandler.cpp
 * @brief The pack's dispatch: workspace sizing, and a real compile-and-launch.
 *
 * The launch tests run on device deliberately. A recorded no-op would exercise neither
 * the runtime compile nor the uid-to-pointer resolution, which are the two parts most
 * likely to be wrong when real kernels arrive.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_flatbuffers_sdk::utilities::parseUuid;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;

/// The bindings a real plan build would hand the handler, produced the way the state
/// manager produces them: by running the graph matcher. Building them by hand instead
/// would let these tests pass against a matcher that binds the wrong uids.
BoundTokens bindingsFor(const MatchContext& context)
{
    BoundTokens bound;
    if(!pointwiseAddGraphMatches(context, bound))
    {
        throw std::logic_error("test graph does not match the pack it is dispatched against");
    }
    return bound;
}

KernelDefinition makeKernel(int64_t blockSize, const std::string& dtype)
{
    KernelDefinition kernel;
    kernel.kernelId = parseUuid("00000000-0000-4000-8000-000000000001");
    kernel.packId = parseUuid("00000000-0000-4000-8000-000000000002");
    kernel.dispatchId = parseUuid("00000000-0000-4000-8000-000000000003");
    kernel.source.sourceFile = "PointwiseAdd.cpp";
    kernel.source.entryPoint = "PointwiseAdd";
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
    const GraphFixture fixture(buildPointwiseGraph());
    const HipMlopsKernelCompiler compiler;
    const PointwiseAddDispatchHandler handler(compiler);

    // The two surviving kernels report different requirements, so the engine's
    // "maximum across survivors" has something to actually maximize over.
    EXPECT_EQ(handler.workspaceBytes(
                  fixture.context(), bindingsFor(fixture.context()), makeKernel(64, "FLOAT")),
              0U);
    EXPECT_EQ(handler.workspaceBytes(
                  fixture.context(), bindingsFor(fixture.context()), makeKernel(256, "FLOAT")),
              1024U);
}

TEST(TestPointwiseAddDispatch, ReportsWorkspaceWithoutSeeingTheRestOfTheCatalog)
{
    const GraphFixture fixture(buildPointwiseGraph());
    const HipMlopsKernelCompiler compiler;
    const PointwiseAddDispatchHandler handler(compiler);

    // The query is answered per kernel, before selection and before any plan exists, so
    // the answer must not depend on which other kernels are in the catalog. Asking twice
    // for the same kernel, either side of a different one, gives the same number.
    const auto first = handler.workspaceBytes(
        fixture.context(), bindingsFor(fixture.context()), makeKernel(256, "FLOAT"));
    static_cast<void>(handler.workspaceBytes(
        fixture.context(), bindingsFor(fixture.context()), makeKernel(64, "FLOAT")));
    const auto second = handler.workspaceBytes(
        fixture.context(), bindingsFor(fixture.context()), makeKernel(256, "FLOAT"));

    EXPECT_EQ(first, second);
}

// ---------------------------------------------------------------------------
// Prepare and launch
// ---------------------------------------------------------------------------

TEST(TestPointwiseAddDispatch, LaunchesARealAddOnDevice)
{
    SKIP_IF_NO_DEVICES();

    const GraphFixture fixture(buildPointwiseGraph());
    const HipMlopsKernelCompiler compiler;
    const PointwiseAddDispatchHandler handler(compiler);

    const auto prepared = handler.prepare(
        fixture.context(), bindingsFor(fixture.context()), makeKernel(64, "FLOAT"));
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
    const PointwiseAddDispatchHandler handler(compiler);

    // block_size reaches the compiler and the launch geometry, so both kernels are
    // genuinely different builds; a one-element add must still agree.
    for(const int64_t blockSize : {64, 256})
    {
        const auto prepared = handler.prepare(
            fixture.context(), bindingsFor(fixture.context()), makeKernel(blockSize, "FLOAT"));
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
    const PointwiseAddDispatchHandler handler(compiler);

    // A plan is built once and may execute many times with different buffers, so
    // preparation must hold nothing tied to one execution.
    const auto prepared = handler.prepare(
        fixture.context(), bindingsFor(fixture.context()), makeKernel(64, "FLOAT"));
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

TEST(TestPointwiseAddDispatch, RefusesToPrepareWithoutTheMatcherSBindings)
{
    SKIP_IF_NO_DEVICES();

    const GraphFixture fixture(buildPointwiseGraph());
    const HipMlopsKernelCompiler compiler;
    const PointwiseAddDispatchHandler handler(compiler);

    // Preparation reads the operand uids the matcher bound rather than re-deriving them,
    // so bindings that never came from this pack's matcher are a wiring error. It must
    // fail loudly instead of guessing which tensor is which operand.
    EXPECT_THROW(handler.prepare(fixture.context(), BoundTokens{}, makeKernel(64, "FLOAT")),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestPointwiseAddDispatch, DispatchStaysResolvableAcrossContainerLifetimes)
{
    SKIP_IF_NO_DEVICES();

    // A container is destroyed when its last handle closes, and a process that opens
    // handles again builds a new one. This pack's implementations are registered once in
    // a process-wide registry, so anything registered there must outlive every container
    // -- a handler owned by a container's engine would be freed while the registration
    // still pointed at it.
    //
    // This asserts the invariant holds: after a container is destroyed and rebuilt, the
    // registered dispatch still resolves and still runs. It does NOT by itself prove the
    // absence of a use-after-free, because a freed handler carrying no per-instance state
    // usually keeps answering. What catches that is the integration binary's exit status:
    // owning the handler on the engine makes hip_kernel_provider_integration_tests abort
    // at process teardown while every test still reports as passing.
    {
        const core::Container first;
    }

    const core::Container second;

    const auto* handler = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
        std::string(DISPATCH_SYMBOL));
    ASSERT_NE(handler, nullptr);

    const GraphFixture fixture(buildPointwiseGraph());
    const auto prepared = handler->prepare(
        fixture.context(), bindingsFor(fixture.context()), makeKernel(64, "FLOAT"));
    ASSERT_NE(prepared, nullptr);

    const AddBuffers buffers(2.0f, 5.0f);
    const auto descriptors = buffers.descriptors();
    const Handle handle;

    handler->launch(handle, *prepared, descriptors.data(), descriptors.size(), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

    EXPECT_FLOAT_EQ(buffers.readResult(), 7.0f);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
