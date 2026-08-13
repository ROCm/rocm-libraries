// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Container.hpp"
#include "core/Handle.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestPointwiseAddDispatchHandler.cpp
 * @brief The pack's dispatch: workspace sizing, prepare's unhappy paths, and a real
 *        compile-and-launch.
 *
 * The handler is reached through DispatchRegistry, the way a plan build reaches it:
 * it is internal to PointwiseAddNative.cpp, and its process-lifetime instance is the
 * one a plan actually runs.
 *
 * The launch tests run on device to exercise the runtime compile and uid-to-pointer
 * resolution. Everything that fails before those is asserted CPU-only.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::MatchContext;

/// The bindings a real plan build would hand the handler, produced by running the graph
/// matcher rather than built by hand.
BoundTokens bindingsFor(const MatchContext& context)
{
    BoundTokens bound;
    if(!matchesGraph(POINTWISE_ADD, context, bound))
    {
        throw std::logic_error("test graph does not match the pack it is dispatched against");
    }
    return bound;
}

/// Device buffers for one 1-element add, freed on scope exit. Templated on the element
/// type to serve every dtype this pack's dispatch handler compiles for.
template <typename T>
class AddBuffers
{
public:
    AddBuffers(T a, T b)
    {
        EXPECT_EQ(hipSuccess, hipMalloc(&_a, sizeof(T)));
        EXPECT_EQ(hipSuccess, hipMalloc(&_b, sizeof(T)));
        EXPECT_EQ(hipSuccess, hipMalloc(&_c, sizeof(T)));
        EXPECT_EQ(hipSuccess, hipMemcpy(_a, &a, sizeof(T), hipMemcpyHostToDevice));
        EXPECT_EQ(hipSuccess, hipMemcpy(_b, &b, sizeof(T), hipMemcpyHostToDevice));
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

    T readResult() const
    {
        T result{};
        EXPECT_EQ(hipSuccess, hipMemcpy(&result, _c, sizeof(T), hipMemcpyDeviceToHost));
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

struct WorkspaceCase
{
    int64_t blockSize;
    size_t expectedBytes;
};

class TestPointwiseAddDispatchWorkspace : public ::testing::TestWithParam<WorkspaceCase>
{
};

TEST_P(TestPointwiseAddDispatchWorkspace, ReportsWorkspaceFromKernelMetadata)
{
    const GraphFixture fixture(buildPointwiseGraph(), currentDeviceProperties());
    const auto& handler = dispatchHandler(POINTWISE_ADD);

    EXPECT_EQ(handler.workspaceBytes(fixture.context(),
                                     bindingsFor(fixture.context()),
                                     makeKernel(GetParam().blockSize, "FLOAT")),
              GetParam().expectedBytes);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestPointwiseAddDispatchWorkspace,
                         ::testing::Values(WorkspaceCase{64, 0U}, WorkspaceCase{256, 1024U}),
                         [](const ::testing::TestParamInfo<WorkspaceCase>& info) {
                             return "BlockSize" + std::to_string(info.param.blockSize);
                         });

TEST(TestPointwiseAddDispatch, ReportsWorkspaceWithoutSeeingTheRestOfTheCatalog)
{
    const GraphFixture fixture(buildPointwiseGraph(), currentDeviceProperties());
    const auto& handler = dispatchHandler(POINTWISE_ADD);

    // The query is answered per kernel, before selection and before any plan exists, so
    // the answer must not depend on which other kernels are in the catalog.
    const auto first = handler.workspaceBytes(
        fixture.context(), bindingsFor(fixture.context()), makeKernel(256, "FLOAT"));
    static_cast<void>(handler.workspaceBytes(
        fixture.context(), bindingsFor(fixture.context()), makeKernel(64, "FLOAT")));
    const auto second = handler.workspaceBytes(
        fixture.context(), bindingsFor(fixture.context()), makeKernel(256, "FLOAT"));

    EXPECT_EQ(first, second);
}

// ---------------------------------------------------------------------------
// Prepare: unhappy paths, CPU-only, since each fails before reaching a compiler.
// ---------------------------------------------------------------------------
//
// prepare()'s compile options are not asserted against a mock compiler: the handler is
// internal to PointwiseAddNative.cpp, reachable only as the process-lifetime instance
// the registry holds, so there is no seam to inject one through. The HALF launch below
// fails if the dtype option is wrong, and the E2E suite fails if the kernel does not
// compile. Both need a device.

TEST(TestPointwiseAddDispatch, PrepareRejectsAKernelDeclaringAnUnsupportedDtype)
{
    // elementTypeFor's dtype switch is unreachable via matching, which admits only the
    // dtypes this pack declares (FLOAT, HALF). Never reaches the compiler --
    // elementTypeFor throws before prepare() calls it.
    const GraphFixture fixture(buildPointwiseGraph());
    const auto& handler = dispatchHandler(POINTWISE_ADD);

    EXPECT_THROW(handler.prepare(
                     fixture.context(), bindingsFor(fixture.context()), makeKernel(64, "BFLOAT16")),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestPointwiseAddDispatch, RefusesToPrepareWithoutTheMatcherSBindings)
{
    // Binding lookup throws on a missing token before prepare() touches HIP.
    const GraphFixture fixture(buildPointwiseGraph());
    const auto& handler = dispatchHandler(POINTWISE_ADD);

    // Preparation reads the operand uids the matcher bound rather than re-deriving them;
    // bindings not produced by this pack's matcher are a wiring error.
    EXPECT_THROW(handler.prepare(fixture.context(), BoundTokens{}, makeKernel(64, "FLOAT")),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ---------------------------------------------------------------------------
// Prepare and launch on device
// ---------------------------------------------------------------------------

// Runs over every dtype the pack ships a kernel for.
struct RealLaunchCase
{
    std::string name;
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType;
    std::string kernelDtype;
};

class TestPointwiseAddDispatchRealLaunch : public ::testing::TestWithParam<RealLaunchCase>
{
};

TEST_P(TestPointwiseAddDispatchRealLaunch, LaunchesARealAddOnDevice)
{
    SKIP_IF_NO_DEVICES();

    const auto& param = GetParam();
    const GraphFixture fixture(
        buildPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD,
                            param.dataType),
        currentDeviceProperties());
    const auto& handler = dispatchHandler(POINTWISE_ADD);

    const auto prepared = handler.prepare(
        fixture.context(), bindingsFor(fixture.context()), makeKernel(64, param.kernelDtype));
    ASSERT_NE(prepared, nullptr);

    const Handle handle;
    if(param.dataType == hipdnn_flatbuffers_sdk::data_objects::DataType::HALF)
    {
        const AddBuffers<hipdnn_data_sdk::types::half> buffers(hipdnn_data_sdk::types::half(3.0f),
                                                               hipdnn_data_sdk::types::half(4.0f));
        const auto descriptors = buffers.descriptors();

        handler.launch(handle, *prepared, descriptors.data(), descriptors.size(), nullptr);
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
        EXPECT_NEAR(static_cast<float>(buffers.readResult()), 7.0f, 1e-2f);
    }
    else
    {
        const AddBuffers<float> buffers(3.0f, 4.0f);
        const auto descriptors = buffers.descriptors();

        handler.launch(handle, *prepared, descriptors.data(), descriptors.size(), nullptr);
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
        EXPECT_FLOAT_EQ(buffers.readResult(), 7.0f);
    }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestPointwiseAddDispatchRealLaunch,
    ::testing::Values(
        RealLaunchCase{"Float", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, "FLOAT"},
        RealLaunchCase{"Half", hipdnn_flatbuffers_sdk::data_objects::DataType::HALF, "HALF"}),
    [](const ::testing::TestParamInfo<RealLaunchCase>& info) { return info.param.name; });

TEST(TestPointwiseAddDispatch, LaunchesTheSameResultForEitherBlockSize)
{
    SKIP_IF_NO_DEVICES();

    const GraphFixture fixture(buildPointwiseGraph(), currentDeviceProperties());
    const auto& handler = dispatchHandler(POINTWISE_ADD);

    // block_size reaches the compiler and the launch geometry; a one-element add must
    // still agree across both kernels.
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

    const GraphFixture fixture(buildPointwiseGraph(), currentDeviceProperties());
    const auto& handler = dispatchHandler(POINTWISE_ADD);

    // A plan is built once and executes many times, so preparation must hold nothing
    // tied to one execution.
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

TEST(TestPointwiseAddDispatch, DispatchStaysResolvableAcrossContainerLifetimes)
{
    SKIP_IF_NO_DEVICES();

    // Registered dispatch implementations live in a process-wide registry and must
    // outlive every container. This asserts a rebuilt container still resolves and runs
    // the registration; a use-after-free is instead caught by the integration binary's
    // exit status at process teardown.
    {
        const core::Container first;
    }

    const core::Container second;

    const auto* handler = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
        std::string(POINTWISE_ADD.dispatch));
    ASSERT_NE(handler, nullptr);

    const GraphFixture fixture(buildPointwiseGraph(), currentDeviceProperties());
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
