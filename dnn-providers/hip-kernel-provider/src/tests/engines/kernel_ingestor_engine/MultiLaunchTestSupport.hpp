// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

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
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "TestDescriptorRoot.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/LaunchKernelDefinition.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/// @file MultiLaunchTestSupport.hpp
/// The fixture of a multi-launch kernel (a UKD whose `sources` map names two launches): the
/// packed shard under test_descriptors/multi_launch/, a scratch descriptor tree, a harness that
/// registers the caller's dispatch handler beside an accept-any-graph matcher, and a
/// two-stage handler whose launches the caller's own handler may build on. Shared by the
/// end-to-end cases and the save and load cases.
namespace hip_kernel_provider::kernel_ingestor_engine::testing::multi_launch
{

using namespace hipdnn_plugin_sdk::ingestor;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;

inline constexpr const char* SCRATCH_LABEL = "multilaunch";
inline constexpr const char* GRAPH_MATCH_SYMBOL = "hipkernel.test.multi_launch.graph_match";
inline constexpr const char* DISPATCH_SYMBOL = "hipkernel.test.multi_launch.dispatch";
/// The versioned name a handler that opts in to saving registers beside DISPATCH_SYMBOL, and
/// saves its launch inputs under.
inline constexpr const char* DISPATCH_SYMBOL_V1 = "hipkernel.test.multi_launch.dispatch.v1";

/// The name of the engine the packed shard's descriptors declare.
inline constexpr const char* ENGINE_NAME = "hipkernel:multi_launch_probe";

/// The two entry points the shard's translation unit exports and its descriptor names.
inline constexpr const char* PRODUCE_SYMBOL = "MultiLaunchProduce";
inline constexpr const char* CONSUME_SYMBOL = "MultiLaunchConsume";

/// Stands in for whatever the device buffers held before the kernels ran. Neither kernel
/// can produce it, so a buffer that still holds it after execution was never written, and a
/// launch that read it instead of its input produces a value the case rules out.
inline constexpr float POISON = -1.0e6F;

/// The launch names the test gives them. Sorted the other way round from launch order, so a
/// handler or a loader that walked the `sources` map in key order would run them backwards.
inline constexpr const char* PRODUCE_LAUNCH = "produce";
inline constexpr const char* CONSUME_LAUNCH = "consume";

/// One element per thread, in whole blocks.
inline constexpr size_t ELEMENTS = 1024;
inline constexpr unsigned int BLOCK_SIZE = 256;
static_assert(ELEMENTS % BLOCK_SIZE == 0, "the launch geometry covers whole blocks");

/// The intermediate the first launch writes and the second reads: one float per element.
inline constexpr size_t WORKSPACE_BYTES = ELEMENTS * sizeof(float);

/// The argument list both kernels take and the handler marshals: the input, the workspace
/// and the output as device pointers, then the element count.
inline const std::vector<KernelArgument>& twoStageSignature()
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
class PreparedTwoStage : public IngestorPreparedDispatch
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

    const IngestorKernelCode& kernelCode() const override
    {
        return _produce;
    }

    std::vector<IngestorLaunchedKernel> launchedKernels() const override
    {
        return {IngestorLaunchedKernel{
                    PRODUCE_LAUNCH, &_produce, KernelSourceKind::KPACK, twoStageSignature()},
                IngestorLaunchedKernel{
                    CONSUME_LAUNCH, &_consume, KernelSourceKind::KPACK, twoStageSignature()}};
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

protected:
    // This handler's own, so that the launches of one kernel resolving out of one module is
    // not something another test's earlier load can answer for.
    hip_kernel_provider::compilation::KpackModuleCache _cache;
    hip_kernel_provider::compilation::KpackKernelLoader _loader{_cache};

private:
    std::set<std::string> _launches;
};

inline std::optional<BoundTokens> acceptAnyGraph(const MatchContext& /*context*/)
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
inline PackedShard shardFor(const std::string& arch)
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
inline PackedShard anyConcreteShard()
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

/// Whether the multi-launch root holds a shard at all. Called when anyConcreteShard() found
/// none, so each shard it holds is for a generic target: the build selected no concrete
/// architecture, and a case that needs a concrete shard has nothing to run on by design.
inline bool packsOnlyGenericShards()
{
    const std::filesystem::path root
        = hip_kernel_provider::testing::descriptorSetRoot(HIPKERNELPROVIDER_MULTI_LAUNCH_RELDIR);
    std::error_code failed;
    for(const auto& entry : std::filesystem::directory_iterator(root, failed))
    {
        if(entry.is_directory(failed))
        {
            return true;
        }
    }
    return false;
}

/// Sets @p shard to a shard this build packed for a concrete architecture, for a case that
/// needs no device. Skips only when the build packs generic targets alone, by design. A root
/// that is missing or empty is a packing failure, and fails: the build packs a shard per
/// concrete GPU_TARGETS entry, so a case that skipped there would report success over nothing.
/// Test only @p shard.arch after the call: an empty one means the case was skipped or failed.
/// Fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void concreteShardOrSkip(PackedShard& shard)
{
    shard = anyConcreteShard();
    if(!shard.arch.empty())
    {
        return;
    }
    const auto root
        = hip_kernel_provider::testing::descriptorSetRoot(HIPKERNELPROVIDER_MULTI_LAUNCH_RELDIR);
    if(!packsOnlyGenericShards())
    {
        FAIL() << "the build packed no shard under " << root
               << ", and it packs one per concrete GPU_TARGETS entry";
    }
    GTEST_SKIP() << "the build selects generic targets only, so nothing was packed for a "
                    "concrete architecture under "
                 << root;
}

/// Sets @p shard to the shard packed for device 0, and skips when the build packed nothing for
/// it: the build packs per GPU_TARGETS entry, and the device may be outside them.
/// Fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void deviceShardOrSkip(PackedShard& shard)
{
    hipDeviceProp_t properties{};
    ASSERT_EQ(hipGetDeviceProperties(&properties, 0), hipSuccess);
    const std::string arch(hipdnn_plugin_sdk::stripArchFeatures(properties.gcnArchName));
    shard = shardFor(arch);
    if(shard.arch.empty())
    {
        GTEST_SKIP() << "nothing was packed for this device (" << arch << ") under "
                     << hip_kernel_provider::testing::descriptorSetRoot(
                            HIPKERNELPROVIDER_MULTI_LAUNCH_RELDIR)
                     << ". Environmental -- the build packs per arch and this device is outside "
                        "GPU_TARGETS.";
    }
}

inline DeviceProperties devicePropertiesFor(const std::string& arch)
{
    DeviceProperties properties;
    properties.gcnArchName = arch;
    properties.warpSize = 64;
    return properties;
}

/// The engine's own descriptors, and the symbols that connect them to a handler of the caller's.
/// The handler must outlive the harness.
///
/// The symbols are registered for as long as this object lives and unregistered with it, so no
/// other case sees them. The tree holds the one multi-launch kernel and nothing else, which is
/// what lets a case that sees a kernel dropped know it is that kernel.
class MultiLaunchHarness
{
public:
    /// @p handler is registered under DISPATCH_SYMBOL, and also under DISPATCH_SYMBOL_V1 when
    /// @p alsoVersioned is set: the name a saved plan of the handler's names.
    explicit MultiLaunchHarness(IKernelDispatchHandler<Handle>& handler, bool alsoVersioned = false)
        : _tree(claimScratchDirectory(SCRATCH_LABEL))
    {
        _scope.add(GRAPH_MATCH_SYMBOL, &acceptAnyGraph);
        _scope.add(DISPATCH_SYMBOL, &handler);
        if(alsoVersioned)
        {
            _scope.add(DISPATCH_SYMBOL_V1, &handler);
        }
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
    // Declared in the order they are released in reverse: the symbols come out first, and the
    // tree outlasts them.
    ScopedDirectory _tree;
    SymbolScope<Handle> _scope;
};

/// The kernels @p validated offers a pointwise graph on a device of architecture @p device,
/// pointing into its state manager.
inline std::vector<const KernelDefinition*> keptDefinitions(const ValidatedSet<Handle>& validated,
                                                            const DeviceProperties& device)
{
    const GraphFixture fixture(buildPointwiseGraph(), device);
    return validated.stateManager->unsortedDefinitions(fixture.context());
}

/// What the packer wrote for the fixture's two launches: one source each, with its own symbol,
/// and both in one code object of one archive. The symbols differ and the key does not, so a
/// packer that gave both launches one kernel's symbol, or packed the one translation unit
/// twice, fails here. Fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void expectLaunchesShareOneCodeObject(const KernelDefinition& kernel)
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

/// Copies @p shard into the tree of @p harness, loads it, and checks that the tree holds the
/// one engine of the packed set and that its handler kept the one two-launch kernel the packer
/// wrote, for a pointwise graph on @p device. @p validated holds the loaded set, and @p kept
/// points into it. Fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void loadWithKeptKernel(const MultiLaunchHarness& harness,
                               const PackedShard& shard,
                               const DeviceProperties& device,
                               std::vector<ValidatedSet<Handle>>& validated,
                               const KernelDefinition*& kept)
{
    ASSERT_NO_FATAL_FAILURE(harness.copyShard(shard));
    validated = harness.load();
    ASSERT_EQ(validated.size(), 1U);
    ASSERT_EQ(validated[0].set.engine.name, ENGINE_NAME);

    const auto definitions = keptDefinitions(validated[0], device);
    ASSERT_EQ(definitions.size(), 1U);
    kept = definitions[0];
    ASSERT_NO_FATAL_FAILURE(expectLaunchesShareOneCodeObject(*kept));
}

/// A device buffer of ELEMENTS floats holding the values it was given, freed on scope exit.
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

    std::vector<uint8_t> readBytes() const
    {
        std::vector<uint8_t> bytes(WORKSPACE_BYTES);
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(bytes.data(), _buffer, WORKSPACE_BYTES, hipMemcpyDeviceToHost));
        return bytes;
    }

private:
    void* _buffer = nullptr;
};

/// The input the cases run the two launches over: 0, 1, 2, ...
inline std::vector<float> inputValues()
{
    std::vector<float> input(ELEMENTS);
    std::iota(input.begin(), input.end(), 0.0F);
    return input;
}

/// What the first launch writes into the workspace for inputValues(): 2x + 1. Whole-float
/// exact, since every value is a small integer.
inline std::vector<float> expectedWorkspace()
{
    auto values = inputValues();
    for(auto& value : values)
    {
        value = (2.0F * value) + 1.0F;
    }
    return values;
}

/// What the second launch writes to the output for inputValues(): it reads the workspace the
/// first launch wrote, and the input, to write 3 * (2x + 1) + x. Differs if a launch is
/// skipped, if the two run in the other order, or if the second reads its input in place of
/// the workspace.
inline std::vector<float> expectedOutput()
{
    const auto input = inputValues();
    auto values = expectedWorkspace();
    for(size_t i = 0; i < ELEMENTS; ++i)
    {
        values[i] = (3.0F * values[i]) + input[i];
    }
    return values;
}

/// Makes the engine of @p validated, the one set of a harness's tree, and fills @p context
/// with the plan that engine builds for the pointwise graph of ELEMENTS floats. The caller
/// keeps @p engine alive for as long as it uses the plan. Checks that the engine applies to the
/// graph and that the engine and the plan report the workspace the handler sizes.
///
/// Moves the state manager out of @p validated. Uses fatal assertions: call through
/// ASSERT_NO_FATAL_FAILURE.
inline void makeTwoStageContext(
    ValidatedSet<Handle>& validated,
    Handle& handle,
    std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>>& engine,
    Context& context)
{
    engine = makeEngine<Handle, Settings, Context>(
        validated.set, std::move(validated.stateManager), deviceResolver());

    const auto graphBuilder
        = buildPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              {static_cast<int64_t>(ELEMENTS), 1, 1, 1});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    ASSERT_TRUE(engine->isApplicable(handle, graph));

    const hipdnn_test_sdk::utilities::MockEngineConfig engineConfig;
    EXPECT_CALL(engineConfig, isValid()).WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(engineConfig, engineId())
        .WillRepeatedly(::testing::Return(hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME)));

    // The workspace the handler sizes is what the engine reports, and what the plan carries.
    EXPECT_EQ(engine->getMaxWorkspaceSize(handle, graph, engineConfig), WORKSPACE_BYTES);
    ASSERT_NO_THROW(engine->initializeExecutionContext(handle, graph, engineConfig, context));
    ASSERT_TRUE(context.hasValidPlan());
    ASSERT_EQ(context.plan().getWorkspaceSize(handle), WORKSPACE_BYTES);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::multi_launch

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
