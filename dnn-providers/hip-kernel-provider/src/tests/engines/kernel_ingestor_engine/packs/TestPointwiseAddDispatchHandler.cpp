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
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddDispatchHandler.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddMatchers.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddSymbols.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseAddTestGraphs.hpp"

/**
 * @file TestPointwiseAddDispatchHandler.cpp
 * @brief The pack's dispatch: workspace sizing, prepare's compile options, and a real
 *        compile-and-launch.
 *
 * The launch tests run on device deliberately. A recorded no-op would exercise neither
 * the runtime compile nor the uid-to-pointer resolution, which are the two parts most
 * likely to be wrong when real kernels arrive. Everything that fails before either of
 * those -- an unbound dispatch, an unsupported dtype, the exact options handed to the
 * compiler -- is asserted CPU-only, so a decline is cheap to verify and does not need a
 * device to run in CI.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
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

/// Device buffers for one 1-element add, freed on scope exit. Templated on the element
/// type so the same buffer plumbing serves every dtype this pack's dispatch handler
/// compiles for, rather than one copy per type that could quietly drift apart.
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
//
// Collapses two block-size cases -- the 256-block kernel's non-zero requirement is what
// makes the engine's "maximum across survivors" answer observably a maximum rather than
// a constant zero -- into one TEST_P over the (block size, expected bytes) pairs.
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
    const HipMlopsKernelCompiler compiler;
    const PointwiseAddDispatchHandler handler(compiler);

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
// Prepare: compile options and unhappy paths -- all CPU-only, since everything here
// either never reaches the compiler or replaces it with a mock.
// ---------------------------------------------------------------------------

TEST(TestPointwiseAddDispatch, PreparePassesTheKernelsTypeAndBlockSizeToTheCompiler)
{
    // Mirrors TestRMSnormBwdPlan.cpp's captured-options pattern: a mock compiler in
    // place of a real one proves exactly what prepare() sends it, without depending on
    // hiprtc or a device.
    const GraphFixture fixture(buildPointwiseGraph());
    const MockKernelCompiler compiler;
    std::vector<std::string> capturedOptions;

    EXPECT_CALL(compiler, compile("PointwiseAdd.cpp", ::testing::_))
        .WillOnce([&](const std::string&, const std::vector<std::string>& options) {
            capturedOptions = options;

            auto kernel = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            EXPECT_CALL(*kernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);

            auto program = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*program, getKernel("PointwiseAdd"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernel))));
            return program;
        });

    const PointwiseAddDispatchHandler handler(compiler);
    const auto prepared = handler.prepare(
        fixture.context(), bindingsFor(fixture.context()), makeKernel(256, "FLOAT"));
    ASSERT_NE(prepared, nullptr);

    const auto hasOption = [&capturedOptions](const std::string& option) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), option)
               != capturedOptions.end();
    };
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_POINTWISE_ADD_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_POINTWISE_ADD_BLOCK_SIZE=256"));
}

TEST(TestPointwiseAddDispatch, PrepareRejectsAKernelDeclaringAnUnsupportedDtype)
{
    // elementTypeFor's dtype switch is unreachable via matching, which admits only the
    // dtypes this pack declares (FLOAT, HALF); reached directly here to prove the
    // fallback reports rather than silently compiling the wrong kernel. Never reaches
    // the compiler -- elementTypeFor throws before prepare() calls it -- so this needs
    // no device and no mock beyond a real, unused compiler.
    const GraphFixture fixture(buildPointwiseGraph());
    const HipMlopsKernelCompiler compiler;
    const PointwiseAddDispatchHandler handler(compiler);

    EXPECT_THROW(handler.prepare(
                     fixture.context(), bindingsFor(fixture.context()), makeKernel(64, "BFLOAT16")),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestPointwiseAddDispatch, RefusesToPrepareWithoutTheMatcherSBindings)
{
    // Not gated: pointwiseAddBinding() throws on a missing token before prepare() reads
    // context.deviceProperties or does anything HIP-touching, so this needs no device.
    const GraphFixture fixture(buildPointwiseGraph());
    const HipMlopsKernelCompiler compiler;
    const PointwiseAddDispatchHandler handler(compiler);

    // Preparation reads the operand uids the matcher bound rather than re-deriving them,
    // so bindings that never came from this pack's matcher are a wiring error. It must
    // fail loudly instead of guessing which tensor is which operand.
    EXPECT_THROW(handler.prepare(fixture.context(), BoundTokens{}, makeKernel(64, "FLOAT")),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ---------------------------------------------------------------------------
// Prepare and launch on device
// ---------------------------------------------------------------------------

// Runs over every dtype the pack ships a kernel for, so a broken HALF compile path
// fails a real launch rather than only the matcher tests, which never reach hiprtc.
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
    const HipMlopsKernelCompiler compiler;
    const PointwiseAddDispatchHandler handler(compiler);

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

    const GraphFixture fixture(buildPointwiseGraph(), currentDeviceProperties());
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
