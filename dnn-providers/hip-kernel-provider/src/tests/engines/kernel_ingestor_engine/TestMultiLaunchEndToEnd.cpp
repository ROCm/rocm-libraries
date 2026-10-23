// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "TestDescriptorRoot.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/LaunchKernelDefinition.hpp"
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
using namespace hipdnn_plugin_sdk::ingestor;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::MockEngineConfig;
using hipdnn_test_sdk::utilities::ScopedDirectory;

constexpr const char* SCRATCH_LABEL = "multilaunch";

/// The names the test-only descriptors in test_descriptors/multi_launch/two_stage/ carry.
constexpr const char* ENGINE_NAME = "hipkernel:multi_launch_probe";
constexpr const char* GRAPH_MATCH_SYMBOL = "hipkernel.test.multi_launch.graph_match";
constexpr const char* DISPATCH_SYMBOL = "hipkernel.test.multi_launch.dispatch";

/// The two entry points the fixture's translation unit exports and its descriptor names, which
/// must match test_descriptors/multi_launch/two_stage/.
constexpr const char* PRODUCE_SYMBOL = "MultiLaunchProduce";
constexpr const char* CONSUME_SYMBOL = "MultiLaunchConsume";

/// The launch names the test gives them. Sorted the other way round from launch order, so a
/// handler or a loader that walked the `sources` map in key order would run them backwards.
constexpr const char* PRODUCE_LAUNCH = "produce";
constexpr const char* CONSUME_LAUNCH = "consume";

/// One element per thread, in whole blocks.
constexpr size_t ELEMENTS = 1024;
constexpr unsigned int BLOCK_SIZE = 256;
static_assert(ELEMENTS % BLOCK_SIZE == 0, "the launch geometry covers whole blocks");

/// The intermediate the first launch writes and the second reads: one float per element.
constexpr size_t WORKSPACE_BYTES = ELEMENTS * sizeof(float);

/// Stands in for whatever the device buffers held before the kernels ran. Neither kernel
/// can produce it, so a buffer that still holds it after execution was never written, and a
/// launch that read it instead of its input produces a value the case rules out.
constexpr float POISON = -1.0e6F;

/// The argument list both kernels take and the handler marshals: the input, the workspace
/// and the output as device pointers, then the element count.
const std::vector<KernelArgument>& twoStageSignature()
{
    static const KernelArgument s_buffer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_extent{"by_value", static_cast<uint32_t>(sizeof(int)), 0, ""};
    static const std::vector<KernelArgument> s_signature{s_buffer, s_buffer, s_buffer, s_extent};
    return s_signature;
}

// ---------------------------------------------------------------------------
// The dispatch handler under test
// ---------------------------------------------------------------------------

/// Both launches, ready to enqueue: the code resolved at prepare() time, with its geometry.
class PreparedTwoStage : public PreparedDispatch
{
public:
    PreparedTwoStage(IngestorKernelCode produce, IngestorKernelCode consume)
        : _produce(std::move(produce))
        , _consume(std::move(consume))
    {
    }

    const IngestorKernelCode& produce() const
    {
        return _produce;
    }

    const IngestorKernelCode& consume() const
    {
        return _consume;
    }

private:
    IngestorKernelCode _produce;
    IngestorKernelCode _consume;
};

/// Launches `produce` and then `consume` on the handle's stream, handing the first's output to
/// the second through the workspace. Declares the launch names it is given, which is how one
/// handler can stand for either a handler that matches the kernel or one that does not.
class TwoStageDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit TwoStageDispatchHandler(std::set<std::string> launches)
        : _launches(std::move(launches))
    {
    }

    std::set<std::string> launchNames() const override
    {
        return _launches;
    }

    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return WORKSPACE_BYTES;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        auto produce = buildIngestorKernelCode(
            _loader, context, launchKernelDefinition(kernel, PRODUCE_LAUNCH), twoStageSignature());
        auto consume = buildIngestorKernelCode(
            _loader, context, launchKernelDefinition(kernel, CONSUME_LAUNCH), twoStageSignature());
        for(auto* code : {&produce, &consume})
        {
            code->setBlockSize(BLOCK_SIZE);
            code->setGridSize(static_cast<unsigned int>(ELEMENTS / BLOCK_SIZE));
        }
        return std::make_unique<PreparedTwoStage>(std::move(produce), std::move(consume));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        // The plan refuses a missing workspace itself. This one refuses it too, so that a plan
        // that stopped passing the workspace on fails here by name and not as a device fault.
        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "the two-stage handler was launched without its workspace");
        }

        const auto& stages = dynamic_cast<const PreparedTwoStage&>(prepared);
        const auto input
            = hipdnn_plugin_sdk::findDeviceBuffer(INPUT_A_UID, deviceBuffers, numDeviceBuffers);
        const auto output
            = hipdnn_plugin_sdk::findDeviceBuffer(OUTPUT_UID, deviceBuffers, numDeviceBuffers);

        hipStream_t stream = handle.getStream();
        const void* x = input.ptr;
        void* ws = workspace;
        void* y = output.ptr;
        const auto count = static_cast<int>(ELEMENTS);

        stages.produce().kernelForStream(stream).launch(stream, x, ws, y, count);
        stages.consume().kernelForStream(stream).launch(stream, x, ws, y, count);
    }

private:
    std::set<std::string> _launches;
    // This handler's own, so that the launches of one kernel resolving out of one module is
    // not something another test's earlier load can answer for.
    hip_kernel_provider::compilation::KpackModuleCache _cache;
    hip_kernel_provider::compilation::KpackKernelLoader _loader{_cache};
};

std::optional<BoundTokens> acceptAnyGraph(const MatchContext& /*context*/)
{
    return BoundTokens{};
}

// ---------------------------------------------------------------------------
// The descriptor tree
// ---------------------------------------------------------------------------

/// A packed shard of the multi-launch set and the architecture it is for.
struct PackedShard
{
    std::filesystem::path directory;
    std::string arch;
};

/// The shard packed for @p arch, or an empty directory when the build packed nothing for it.
PackedShard shardFor(const std::string& arch)
{
    const std::filesystem::path candidate
        = hip_kernel_provider::testing::descriptorSetRoot(HIPKERNELPROVIDER_MULTI_LAUNCH_RELDIR)
          / arch;
    std::error_code failed;
    return std::filesystem::is_directory(candidate, failed) ? PackedShard{candidate, arch}
                                                            : PackedShard{};
}

/// Any shard this build packed for a concrete architecture, which is all a case that never
/// touches a device needs. A generic target names a family and matches no single device
/// string, so it is passed over. Empty when the build packed none.
PackedShard anyConcreteShard()
{
    const std::filesystem::path root
        = hip_kernel_provider::testing::descriptorSetRoot(HIPKERNELPROVIDER_MULTI_LAUNCH_RELDIR);
    std::vector<std::string> arches;
    std::error_code failed;
    for(const auto& entry : std::filesystem::directory_iterator(root, failed))
    {
        const std::string name = entry.path().filename().string();
        if(entry.is_directory(failed) && name.find("generic") == std::string::npos)
        {
            arches.push_back(name);
        }
    }
    if(arches.empty())
    {
        return {};
    }
    return shardFor(*std::min_element(arches.begin(), arches.end()));
}

DeviceProperties devicePropertiesFor(const std::string& arch)
{
    DeviceProperties properties;
    properties.gcnArchName = arch;
    properties.warpSize = 64;
    return properties;
}

/// The engine's own descriptors, a handler for them, and the symbols that connect the two.
///
/// The symbols are registered for as long as this object lives and unregistered with it, so no
/// other case sees them. The tree holds the one multi-launch kernel and nothing else, which is
/// what lets a case that sees a kernel dropped know it is that kernel.
class MultiLaunchHarness
{
public:
    explicit MultiLaunchHarness(std::set<std::string> launches)
        : _handler(std::move(launches))
        , _tree(claimScratchDirectory(SCRATCH_LABEL))
    {
        _scope.add(GRAPH_MATCH_SYMBOL, &acceptAnyGraph);
        _scope.add(DISPATCH_SYMBOL, &_handler);
    }

    /// Copies @p shard into the tree as the packer wrote it. Fatal assertions: call through
    /// ASSERT_NO_FATAL_FAILURE.
    void copyShard(const PackedShard& shard) const
    {
        std::error_code failed;
        std::filesystem::copy(shard.directory,
                              _tree.path() / shard.arch,
                              std::filesystem::copy_options::recursive,
                              failed);
        ASSERT_FALSE(failed) << "could not copy " << shard.directory << ": " << failed.message();
    }

    std::vector<ValidatedSet<Handle>> load() const
    {
        return loadValidatedDescriptorSets<Handle>(_tree.path());
    }

private:
    // Declared in the order they are released in reverse: the symbols come out before the
    // handler they point at, and the tree outlasts both.
    TwoStageDispatchHandler _handler;
    SymbolScope<Handle> _scope;
    ScopedDirectory _tree;
};

/// The kernels @p validated offers a pointwise graph on a device of architecture @p arch,
/// pointing into its state manager.
std::vector<const KernelDefinition*> keptDefinitions(const ValidatedSet<Handle>& validated,
                                                     const std::string& arch)
{
    const GraphFixture fixture(buildPointwiseGraph(), devicePropertiesFor(arch));
    return validated.stateManager->unsortedDefinitions(fixture.context());
}

/// What the packer wrote for the fixture's two launches: one source each, with its own symbol,
/// and both in one code object of one archive. The symbols differ and the key does not, so a
/// packer that gave both launches one kernel's symbol, or packed the one translation unit
/// twice, fails here. Fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
void expectLaunchesShareOneCodeObject(const KernelDefinition& kernel)
{
    ASSERT_EQ(kernel.sources.size(), 2U);
    const auto produce = kernel.sources.find(PRODUCE_LAUNCH);
    const auto consume = kernel.sources.find(CONSUME_LAUNCH);
    ASSERT_NE(produce, kernel.sources.end());
    ASSERT_NE(consume, kernel.sources.end());

    EXPECT_EQ(produce->second.kind, KernelSourceKind::KPACK);
    EXPECT_EQ(consume->second.kind, KernelSourceKind::KPACK);
    EXPECT_EQ(produce->second.symbol, PRODUCE_SYMBOL);
    EXPECT_EQ(consume->second.symbol, CONSUME_SYMBOL);
    EXPECT_FALSE(produce->second.tocKey.empty());
    EXPECT_EQ(produce->second.tocKey, consume->second.tocKey);
    EXPECT_FALSE(produce->second.library.empty());
    EXPECT_EQ(produce->second.library, consume->second.library);
    EXPECT_EQ(produce->second.sha256, consume->second.sha256);
}

// ---------------------------------------------------------------------------
// Host-only: which kernels the state manager keeps
// ---------------------------------------------------------------------------

TEST(TestMultiLaunchLaunchSet, KeepsAKernelWhoseLaunchesItsHandlerLaunches)
{
    const PackedShard shard = anyConcreteShard();
    if(shard.arch.empty())
    {
        GTEST_SKIP() << "nothing was packed for a concrete architecture: "
                     << hip_kernel_provider::testing::descriptorSetRoot(
                            HIPKERNELPROVIDER_MULTI_LAUNCH_RELDIR)
                     << ". Environmental -- the build packs per GPU_TARGETS entry.";
    }

    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);

    const MultiLaunchHarness harness({PRODUCE_LAUNCH, CONSUME_LAUNCH});
    ASSERT_NO_FATAL_FAILURE(harness.copyShard(shard));

    const auto validated = harness.load();
    ASSERT_EQ(validated.size(), 1U);
    EXPECT_EQ(validated[0].set.engine.name, ENGINE_NAME);

    // The control for the case below: the same tree, the same graph and the same device, with
    // a handler that declares what the kernel carries.
    const auto kept = keptDefinitions(validated[0], shard.arch);
    ASSERT_EQ(kept.size(), 1U);
    ASSERT_NO_FATAL_FAILURE(expectLaunchesShareOneCodeObject(*kept[0]));
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "carries the launches"))
        << recorder.getRecordedLogsAsString();
}

TEST(TestMultiLaunchLaunchSet, DropsAKernelWhoseLaunchesItsHandlerDoesNotLaunch)
{
    const PackedShard shard = anyConcreteShard();
    if(shard.arch.empty())
    {
        GTEST_SKIP() << "nothing was packed for a concrete architecture: "
                     << hip_kernel_provider::testing::descriptorSetRoot(
                            HIPKERNELPROVIDER_MULTI_LAUNCH_RELDIR)
                     << ". Environmental -- the build packs per GPU_TARGETS entry.";
    }

    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);

    // The handler launches one of the kernel's two launches.
    const MultiLaunchHarness harness({PRODUCE_LAUNCH});
    ASSERT_NO_FATAL_FAILURE(harness.copyShard(shard));

    const auto validated = harness.load();

    // The engine still loads: a kernel it cannot launch costs only itself.
    ASSERT_EQ(validated.size(), 1U);
    EXPECT_EQ(validated[0].set.engine.name, ENGINE_NAME);

    // The tree holds nothing else, so the engine offers no kernel for a graph its matcher
    // accepts and a device its kernel was packed for.
    EXPECT_EQ(keptDefinitions(validated[0], shard.arch).size(), 0U);

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

/// A device buffer of ELEMENTS floats holding @p initial, freed on scope exit.
class DeviceFloats
{
public:
    explicit DeviceFloats(const std::vector<float>& initial)
    {
        EXPECT_EQ(hipSuccess, hipMalloc(&_buffer, WORKSPACE_BYTES));
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(_buffer, initial.data(), WORKSPACE_BYTES, hipMemcpyHostToDevice));
    }

    ~DeviceFloats()
    {
        static_cast<void>(hipFree(_buffer));
    }

    DeviceFloats(const DeviceFloats&) = delete;
    DeviceFloats& operator=(const DeviceFloats&) = delete;

    void* get() const
    {
        return _buffer;
    }

    std::vector<float> read() const
    {
        std::vector<float> values(ELEMENTS);
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(values.data(), _buffer, WORKSPACE_BYTES, hipMemcpyDeviceToHost));
        return values;
    }

private:
    void* _buffer = nullptr;
};

TEST(TestMultiLaunchEndToEnd, RunsBothLaunchesInOrderOverOneWorkspace)
{
    SKIP_IF_NO_DEVICES();

    hipDeviceProp_t properties{};
    ASSERT_EQ(hipGetDeviceProperties(&properties, 0), hipSuccess);
    const std::string arch(hipdnn_plugin_sdk::stripArchFeatures(properties.gcnArchName));
    const PackedShard shard = shardFor(arch);
    if(shard.arch.empty())
    {
        GTEST_SKIP() << "nothing was packed for this device (" << arch << ") under "
                     << hip_kernel_provider::testing::descriptorSetRoot(
                            HIPKERNELPROVIDER_MULTI_LAUNCH_RELDIR)
                     << ". Environmental -- the build packs per arch and this device is outside "
                        "GPU_TARGETS.";
    }

    const MultiLaunchHarness harness({PRODUCE_LAUNCH, CONSUME_LAUNCH});
    ASSERT_NO_FATAL_FAILURE(harness.copyShard(shard));

    auto validated = harness.load();
    ASSERT_EQ(validated.size(), 1U);
    ASSERT_EQ(validated[0].set.engine.name, ENGINE_NAME);

    // The one kernel the tree holds survived the handler's launch-set check, and it is the
    // two-launch one the packer wrote.
    const GraphFixture fixture(buildPointwiseGraph(), currentDeviceProperties());
    const auto kept = validated[0].stateManager->unsortedDefinitions(fixture.context());
    ASSERT_EQ(kept.size(), 1U);
    ASSERT_NO_FATAL_FAILURE(expectLaunchesShareOneCodeObject(*kept[0]));

    const auto engine = makeEngine<Handle, Settings, Context>(
        validated[0].set, std::move(validated[0].stateManager), deviceResolver());

    const auto graphBuilder
        = buildPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              {static_cast<int64_t>(ELEMENTS), 1, 1, 1});
    const GraphWrapper graph(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    Handle handle;
    ASSERT_TRUE(engine->isApplicable(handle, graph));

    const MockEngineConfig engineConfig;
    EXPECT_CALL(engineConfig, isValid()).WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(engineConfig, engineId())
        .WillRepeatedly(::testing::Return(hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME)));

    // The workspace the handler sizes is what the engine reports, and what the plan carries.
    EXPECT_EQ(engine->getMaxWorkspaceSize(handle, graph, engineConfig), WORKSPACE_BYTES);

    Context context;
    ASSERT_NO_THROW(engine->initializeExecutionContext(handle, graph, engineConfig, context));
    ASSERT_TRUE(context.hasValidPlan());
    ASSERT_EQ(context.plan().getWorkspaceSize(handle), WORKSPACE_BYTES);

    std::vector<float> input(ELEMENTS);
    std::iota(input.begin(), input.end(), 0.0F);
    const DeviceFloats x(input);
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
    // it back to write 3 * (2x + 1) + x. Whole-float exact: every value is a small integer.
    // Each of these differs if a launch is skipped, if the two run in the other order, or if
    // the second reads its input in place of the workspace.
    std::vector<float> expectedWorkspace(ELEMENTS);
    std::vector<float> expectedOutput(ELEMENTS);
    for(size_t i = 0; i < ELEMENTS; ++i)
    {
        expectedWorkspace[i] = (2.0F * input[i]) + 1.0F;
        expectedOutput[i] = (3.0F * expectedWorkspace[i]) + input[i];
    }
    EXPECT_EQ(workspace.read(), expectedWorkspace);
    EXPECT_EQ(y.read(), expectedOutput);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
