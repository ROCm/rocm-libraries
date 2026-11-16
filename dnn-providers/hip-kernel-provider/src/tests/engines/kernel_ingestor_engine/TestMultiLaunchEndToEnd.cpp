// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "TestDescriptorRoot.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "engines/kernel_ingestor_engine/MultiLaunchTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestMultiLaunchEndToEnd.cpp
 * @brief A multi-launch kernel (a UKD whose `sources` map names two launches) carried through
 *        a real engine: loaded from a descriptor tree, matched, sized, planned and executed on
 *        device.
 *
 * The packed set under test_descriptors/multi_launch/ is authored as a hip `sources` kernel,
 * two launches of one translation unit, and hkp_pack packs it as it is. These cases copy that
 * packed shard into a scratch tree and load it unchanged, so the loader and a native dispatch
 * handler see what the packer wrote. That tree is the only content of the engine under test,
 * and nothing a Container loads names it.
 *
 * The device case runs the two launches through a handler of the cases' own, registered under a
 * symbol no production code registers. The first launch writes the workspace from the input and
 * the second reads the workspace to write the output, so only the right kernels, in the right
 * order, on the workspace the plan was given, produce the output the case asserts. The host
 * cases need no device: they show the state manager keeping a kernel whose launch set its
 * handler declares and dropping one whose launch set it does not.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::multi_launch;
using namespace hipdnn_plugin_sdk::ingestor;

// ---------------------------------------------------------------------------
// Host-only: which kernels the state manager keeps
// ---------------------------------------------------------------------------

TEST(TestMultiLaunchLaunchSet, KeepsAKernelWhoseLaunchesItsHandlerLaunches)
{
    PackedShard shard;
    ASSERT_NO_FATAL_FAILURE(concreteShardOrSkip(shard));
    if(shard.arch.empty())
    {
        return;
    }

    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);

    TwoStageDispatchHandler handler({PRODUCE_LAUNCH, CONSUME_LAUNCH});
    const MultiLaunchHarness harness(handler);

    // The control for the case below: the same tree, the same graph and the same device, with
    // a handler that declares what the kernel carries.
    std::vector<ValidatedSet<Handle>> validated;
    const KernelDefinition* kept = nullptr;
    ASSERT_NO_FATAL_FAILURE(
        loadWithKeptKernel(harness, shard, devicePropertiesFor(shard.arch), validated, kept));
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "carries the launches"))
        << recorder.getRecordedLogsAsString();
}

TEST(TestMultiLaunchLaunchSet, DropsAKernelWhoseLaunchesItsHandlerDoesNotLaunch)
{
    PackedShard shard;
    ASSERT_NO_FATAL_FAILURE(concreteShardOrSkip(shard));
    if(shard.arch.empty())
    {
        return;
    }

    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);

    // The handler launches one of the kernel's two launches.
    TwoStageDispatchHandler handler({PRODUCE_LAUNCH});
    const MultiLaunchHarness harness(handler);
    ASSERT_NO_FATAL_FAILURE(harness.copyShard(shard));

    const auto validated = harness.load();

    // The engine still loads: a kernel it cannot launch costs only itself.
    ASSERT_EQ(validated.size(), 1U);
    EXPECT_EQ(validated[0].set.engine.name, ENGINE_NAME);

    // The tree holds nothing else, so the engine offers no kernel for a graph its matcher
    // accepts and a device its kernel was packed for.
    EXPECT_EQ(keptDefinitions(validated[0], devicePropertiesFor(shard.arch)).size(), 0U);

    // And the drop says why, naming both launch sets and the kernel.
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "two_stage.f32"))
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(
        recorder.hasLogContaining(HIPDNN_SEV_ERROR, "carries the launches {consume, produce}"))
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "launches {produce};"))
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "dropping it"))
        << recorder.getRecordedLogsAsString();
}

// ---------------------------------------------------------------------------
// On device: both launches through a real engine
// ---------------------------------------------------------------------------

TEST(TestMultiLaunchEndToEnd, RunsBothLaunchesInOrderOverOneWorkspace)
{
    SKIP_IF_NO_DEVICES();

    PackedShard shard;
    ASSERT_NO_FATAL_FAILURE(deviceShardOrSkip(shard));
    if(shard.arch.empty())
    {
        return;
    }

    TwoStageDispatchHandler handler({PRODUCE_LAUNCH, CONSUME_LAUNCH});
    const MultiLaunchHarness harness(handler);

    // The one kernel the tree holds survived the handler's launch-set check, and it is the
    // two-launch one the packer wrote.
    std::vector<ValidatedSet<Handle>> validated;
    const KernelDefinition* kept = nullptr;
    ASSERT_NO_FATAL_FAILURE(
        loadWithKeptKernel(harness, shard, currentDeviceProperties(), validated, kept));

    Handle handle;
    std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> engine;
    Context context;
    ASSERT_NO_FATAL_FAILURE(makeTwoStageContext(validated[0], handle, engine, context));

    const DeviceFloats x(inputValues());
    const DeviceFloats y(std::vector<float>(ELEMENTS, POISON));
    // Poisoned so that the second launch reads poison, and not an accident of fresh memory, if
    // the first does not run, or runs after it.
    const DeviceFloats workspace(std::vector<float>(ELEMENTS, POISON));

    const std::array<hipdnnPluginDeviceBuffer_t, 2> buffers{
        hipdnnPluginDeviceBuffer_t{INPUT_A_UID, x.get()},
        hipdnnPluginDeviceBuffer_t{OUTPUT_UID, y.get()}};
    ASSERT_NO_THROW(context.plan().execute(
        handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(handle.getStream()));

    // The first launch wrote 2x + 1 into the workspace the plan was given, and the second read
    // it back to write 3 * (2x + 1) + x.
    EXPECT_EQ(workspace.read(), expectedWorkspace());
    EXPECT_EQ(y.read(), expectedOutput());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
