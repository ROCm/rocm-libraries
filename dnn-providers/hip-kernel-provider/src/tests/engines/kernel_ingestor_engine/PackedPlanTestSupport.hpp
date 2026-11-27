// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "PackedKernelSource.hpp"
#include "TestDescriptorRoot.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Container.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorLaunchRestore.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/MultiLaunchTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanHeader.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanPayload.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanTestUtilities.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockRunnableKernel.hpp"
#include "utilities/Digest.hpp"

// Plans built from the packed kpack sets that this build stages for the local device.
// Read the packed sets only through a copy in a scratch directory. Other suites corrupt
// or delete the staged archives. Each pack describes its packed set with a PackedPlanCase
// in its own header under packs/.
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

using hip_kernel_provider::MockCompiledProgram;
using hip_kernel_provider::MockRunnableKernel;

// The packed Pointwise descriptor with block size 256 (workspace 1024).
inline constexpr const char* POINTWISE_B256_DESCRIPTOR = "packed_pointwise_add_b256.ukd.json";
// The packed Pointwise pack, whose inline kernel has block size 64 (workspace 0).
inline constexpr const char* POINTWISE_B64_DESCRIPTOR = "packed_pointwise_add.kdp.json";
// The packed ConvFwd descriptor with block size 64.
inline constexpr const char* CONV_FWD_DESCRIPTOR = "conv_fwd_f16_block64.ukd.json";

// The text of the save refusal for a benchmarking plan without a chosen kernel.
inline constexpr const char* NO_WINNER_PHRASE = "has not chosen a kernel yet";
// The text of the save refusal for a plan that no ingestor engine built.
inline constexpr const char* NOT_AN_INGESTOR_ENGINE_PHRASE
    = "does not support saving execution plans";

// One packed set: where it is staged, the plan the tests build from it, and the exact
// result that plan computes.
struct PackedPlanCase
{
    // The test instance name.
    std::string_view name;
    // The descriptor of the kernel the tests build a plan for.
    const char* descriptor;
    const PackSymbols* pack;
    int64_t outputUid;
    // The root that holds one packed directory per arch.
    const std::filesystem::path& (*root)();
    // The graph that the capture tests run.
    flatbuffers::FlatBufferBuilder (*captureGraph)();
    // The graph that the round-trip tests run.
    flatbuffers::FlatBufferBuilder (*roundTripGraph)();
    // Writes a fixed pattern to the inputs of roundTripGraph.
    void (*fillInputs)(hipdnn_test_sdk::utilities::GraphTensorBundle& tensors);
    // Checks `output` against the exact result of roundTripGraph for the fillInputs values.
    void (*expectExactOutput)(const std::vector<uint8_t>& output);
};

// The elements of `bytes` read as values of type T.
template <typename T>
std::vector<T> valuesOf(const std::vector<uint8_t>& bytes)
{
    std::vector<T> values(bytes.size() / sizeof(T));
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(T));
    return values;
}

// Copies the packed directory of device 0 under `root` into `scratch`. `copy` stays empty
// when the build packed nothing for the device. `deviceProperties` receives the properties
// of device 0, with the decorated arch name the device reports.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void copyPackedArchForDevice(const std::filesystem::path& root,
                                    const std::filesystem::path& scratch,
                                    hipdnn_plugin_sdk::ingestor::DeviceProperties& deviceProperties,
                                    std::filesystem::path& copy)
{
    hipDeviceProp_t properties{};
    std::string arch;
    std::filesystem::path packed;
    ASSERT_NO_FATAL_FAILURE(
        hip_kernel_provider::testing::findPackedArchDirectoryUnder(root, properties, arch, packed));

    deviceProperties.gcnArchName = properties.gcnArchName;
    deviceProperties.warpSize = properties.warpSize;
    deviceProperties.multiProcessorCount = properties.multiProcessorCount;

    if(!packed.empty())
    {
        copy = hip_kernel_provider::testing::copyPackedArchTree(packed, scratch);
    }
}

// A GenericPlan over the registered handler of `packedCase` for `kernel`, on `graph`.
// `tensors` receives the tensors of `graph` with every value zero and the output of
// `packedCase` marked.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void makePackedPlan(const PackedPlanCase& packedCase,
                           flatbuffers::FlatBufferBuilder graph,
                           const hipdnn_plugin_sdk::ingestor::DeviceProperties& deviceProperties,
                           const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                           std::unique_ptr<hipdnn_plugin_sdk::ingestor::GenericPlan<Handle>>& plan,
                           hipdnn_test_sdk::utilities::GraphTensorBundle& tensors)
{
    const PackSymbols& pack = *packedCase.pack;
    const GraphFixture fixture(std::move(graph), deviceProperties);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the test graph does not match its pack";
    plan = std::make_unique<hipdnn_plugin_sdk::ingestor::GenericPlan<Handle>>(
        hipdnn_plugin_sdk::ingestor::KernelDispatcher<Handle>{kernel, &dispatchHandler(pack)},
        fixture.context(),
        *bound);

    tensors = hipdnn_test_sdk::utilities::GraphTensorBundle(fixture.context().graph.getTensorMap());
    for(auto& entry : tensors.tensors)
    {
        entry.second->fillTensorWithValue(0.0F);
    }
    tensors.outputTensorIds = {packedCase.outputUid};
}

// A plan built from a copy of a packed set, the facts restore needs, and the tensors of
// its test graph.
struct PackedPlan
{
    std::filesystem::path copy;
    hipdnn_plugin_sdk::ingestor::DeviceProperties deviceProperties;
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    std::unique_ptr<hipdnn_plugin_sdk::ingestor::GenericPlan<Handle>> plan;
    hipdnn_test_sdk::utilities::GraphTensorBundle tensors;

    std::filesystem::path archive() const
    {
        return std::filesystem::weakly_canonical(kernel.originDirectory / kernel.source.library);
    }
};

// Copies the packed set of `packedCase` into `scratch` and builds a plan for the kernel of
// its descriptor on `graph`. `built.plan` stays null when nothing was packed for device 0.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void buildPackedPlan(const PackedPlanCase& packedCase,
                            flatbuffers::FlatBufferBuilder graph,
                            const std::filesystem::path& scratch,
                            PackedPlan& built)
{
    ASSERT_NO_FATAL_FAILURE(
        copyPackedArchForDevice(packedCase.root(), scratch, built.deviceProperties, built.copy));
    if(built.copy.empty())
    {
        return;
    }

    ASSERT_NO_FATAL_FAILURE(hip_kernel_provider::testing::readPackedKernelDefinition(
        built.copy, packedCase.descriptor, built.kernel));
    ASSERT_NO_FATAL_FAILURE(makePackedPlan(packedCase,
                                           std::move(graph),
                                           built.deviceProperties,
                                           built.kernel,
                                           built.plan,
                                           built.tensors));
}

// The device buffers of `tensors`, in the form that IPlan::execute takes.
inline std::vector<hipdnnPluginDeviceBuffer_t>
    deviceBuffersOf(hipdnn_test_sdk::utilities::GraphTensorBundle& tensors)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers;
    for(const auto& [uid, pointer] : tensors.toDeviceVariantPack())
    {
        buffers.push_back(hipdnnPluginDeviceBuffer_t{uid, pointer});
    }
    return buffers;
}

// Executes `plan` once on `handle` over `tensors` and writes the output bytes to `output`.
// The output holds the sentinel before the launch, so an element that the launch does not
// write cannot match.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void runPlan(const hipdnn_plugin_sdk::IPlan<Handle>& plan,
                    const Handle& handle,
                    hipdnn_test_sdk::utilities::GraphTensorBundle& tensors,
                    std::vector<uint8_t>& output)
{
    ASSERT_EQ(tensors.outputTensorIds.size(), 1U);
    tensors.sentinelFillOutputTensors();

    const auto buffers = deviceBuffersOf(tensors);
    const hipdnn_data_sdk::utilities::Workspace<> workspace(plan.getWorkspaceSize(handle));
    plan.execute(handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get());
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess) << "the launch failed";

    auto& result = tensors.getTensor(*tensors.outputTensorIds.begin());
    result.markDeviceModified();
    const auto* bytes = static_cast<const uint8_t*>(result.rawHostData());
    output.assign(bytes, bytes + (result.elementSpace() * result.elementSize()));
}

// A kernel definition of `kind` that names nothing on disk. A stub dispatch handler never
// reads it.
inline hipdnn_plugin_sdk::ingestor::KernelDefinition
    stubKernel(uint8_t seed, hipdnn_plugin_sdk::ingestor::KernelSourceKind kind)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.kernelId.fill(seed);
    kernel.name = "stub_kernel";
    kernel.source.kind = kind;
    return kernel;
}

// A GenericPlan over `handler` for `kernel`, on the Pointwise test graph. Builds no code
// object and needs no device.
inline std::unique_ptr<hipdnn_plugin_sdk::ingestor::GenericPlan<Handle>>
    makeStubPlan(const hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>& handler,
                 hipdnn_plugin_sdk::ingestor::KernelDefinition kernel)
{
    const GraphFixture fixture(buildPointwiseGraph());
    const hipdnn_plugin_sdk::ingestor::BoundTokens bound;
    return std::make_unique<hipdnn_plugin_sdk::ingestor::GenericPlan<Handle>>(
        hipdnn_plugin_sdk::ingestor::KernelDispatcher<Handle>{std::move(kernel), &handler},
        fixture.context(),
        bound);
}

// makeStubPlan() for the stub kernel of `seed` and `kind`.
inline std::unique_ptr<hipdnn_plugin_sdk::ingestor::GenericPlan<Handle>>
    makeStubPlan(const hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>& handler,
                 uint8_t seed,
                 hipdnn_plugin_sdk::ingestor::KernelSourceKind kind
                 = hipdnn_plugin_sdk::ingestor::KernelSourceKind::KPACK)
{
    return makeStubPlan(handler, stubKernel(seed, kind));
}

// A plan of an engine other than an ingestor engine. It has no saveable plan, and it does not
// execute.
class NonIngestorPlan final : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    size_t getWorkspaceSize(const Handle& /*handle*/) const override
    {
        return 0;
    }

    void execute(const Handle& /*handle*/,
                 const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                 uint32_t /*numDeviceBuffers*/,
                 void* /*workspace*/) const override
    {
    }
};

// ---------------------------------------------------------------------------
// A multi-kernel stub: the dispatch handler and the real code objects of the packed
// multi_launch shard, for the cases that need no device
// ---------------------------------------------------------------------------

// A copy of the packed multi_launch shard in a scratch directory. Its kernel declares the
// launches PACKED_PRODUCE_LAUNCH and PACKED_CONSUME_LAUNCH in `sources`, both kpack and both
// in one code object of one archive, and leaves `source` default. Reading a code object
// loads no module, so no device is needed. `packed` stays false when the build packs generic
// targets only.
inline constexpr const char* PACKED_PRODUCE_LAUNCH = "produce";
inline constexpr const char* PACKED_CONSUME_LAUNCH = "consume";

struct PackedLaunchSource
{
    hipdnn_test_sdk::utilities::ScopedDirectory scratch
        = hipdnn_test_sdk::utilities::claimScratchDirectory("packedlaunchsource");
    bool packed = false;
    std::string arch;
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;

    // The kernel with its one launch `launch` as the single `source`, as a descriptor with
    // one launch declares it.
    hipdnn_plugin_sdk::ingestor::KernelDefinition singleLaunchKernel(const std::string& launch
                                                                     = PACKED_PRODUCE_LAUNCH) const
    {
        auto single = kernel;
        single.source = kernel.sources.at(launch);
        single.sources.clear();
        return single;
    }
};

// Skips when the build packs generic targets only, by design (`packed` stays false), and fails
// when it packed no shard at all (multi_launch::concreteShardOrSkip). Uses fatal assertions:
// call through ASSERT_NO_FATAL_FAILURE, and test IsSkipped() before the next step.
inline void loadPackedLaunchSource(PackedLaunchSource& out)
{
    multi_launch::PackedShard shard;
    ASSERT_NO_FATAL_FAILURE(multi_launch::concreteShardOrSkip(shard));
    if(shard.arch.empty())
    {
        return;
    }
    out.arch = shard.arch;

    const auto copy
        = hip_kernel_provider::testing::copyPackedArchTree(shard.directory, out.scratch.path());
    std::filesystem::path descriptor;
    hipdnn_plugin_sdk::ingestor::KernelDescriptor parsed;
    ASSERT_NO_FATAL_FAILURE(hip_kernel_provider::testing::detail::readPackedKernelDescriptor(
        copy, "two_stage.ukd.json", descriptor, parsed));
    ASSERT_EQ(parsed.sources.size(), 2U);

    out.kernel.kernelId = parsed.id;
    out.kernel.name = parsed.name;
    out.kernel.sources = parsed.sources;
    out.kernel.metadata = parsed.metadata;
    out.kernel.originDirectory = descriptor.parent_path();
    out.kernel.treeRoot = copy;
    out.packed = true;
}

// A code of `source` over the archive of `kernel`, whose bytes readCodeObject() reads back.
// The program and the kernel are mocks, because nothing launches.
inline IngestorKernelCode
    packedKernelCode(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                     const hipdnn_plugin_sdk::ingestor::KernelSource& source,
                     const std::string& arch,
                     const compilation::KpackKernelLoader& loader)
{
    KpackSource coordinates;
    coordinates.archive
        = std::filesystem::weakly_canonical(kernel.originDirectory / source.library);
    coordinates.tocKey = source.tocKey;
    coordinates.symbol = source.symbol;
    coordinates.sha256 = source.sha256;
    coordinates.label = "stub kernel '" + kernel.name + "'";
    coordinates.strippedArch = arch;
    coordinates.deviceArch = arch;
    return {loader,
            std::move(coordinates),
            0,
            std::make_unique<MockCompiledProgram>(),
            std::make_unique<MockRunnableKernel>()};
}

// What a stub dispatch handler prepares and restores.
class StubKernelPrepared final : public IngestorPreparedDispatch
{
public:
    struct Launch
    {
        std::string name;
        hipdnn_plugin_sdk::ingestor::KernelSourceKind sourceKind
            = hipdnn_plugin_sdk::ingestor::KernelSourceKind::KPACK;
        std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument> signature;
    };

    // @param reportsLaunches false is a dispatch over a kernel with a single `source`: it
    //        answers launchedKernels() with the empty list.
    StubKernelPrepared(std::vector<IngestorKernelCode> codes,
                       std::vector<Launch> launches,
                       bool reportsLaunches,
                       std::vector<int64_t> uids,
                       std::string tag)
        : _codes(std::move(codes))
        , _launches(std::move(launches))
        , _reportsLaunches(reportsLaunches)
        , _uids(std::move(uids))
        , _tag(std::move(tag))
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _codes.front();
    }

    std::vector<IngestorLaunchedKernel> launchedKernels() const override
    {
        std::vector<IngestorLaunchedKernel> launched;
        if(!_reportsLaunches)
        {
            return launched;
        }
        for(size_t index = 0; index < _codes.size(); ++index)
        {
            launched.push_back(IngestorLaunchedKernel{_launches[index].name,
                                                      &_codes[index],
                                                      _launches[index].sourceKind,
                                                      _launches[index].signature});
        }
        return launched;
    }

    std::vector<int64_t> userSuppliedPassByValueUids() const override
    {
        return _uids;
    }

    const std::string& tag() const
    {
        return _tag;
    }

private:
    std::vector<IngestorKernelCode> _codes;
    std::vector<Launch> _launches;
    bool _reportsLaunches;
    std::vector<int64_t> _uids;
    std::string _tag;
};

// A restored dispatch of a handler that restores to a prepared dispatch no ingestor pack
// would build: it exposes no kernel code and reports no scalar.
class StubForeignPrepared final : public hipdnn_plugin_sdk::ingestor::PreparedDispatch
{
public:
    explicit StubForeignPrepared(std::string tag)
        : _tag(std::move(tag))
    {
    }

    const std::string& tag() const
    {
        return _tag;
    }

private:
    std::string _tag;
};

// A dispatch handler that saves and restores through the multi-kernel contract. At plan
// build it prepares real, readable code objects of a PackedLaunchSource kernel. At restore it
// refuses what its Config tells it to refuse and builds a prepared dispatch over mock
// code. It records every launch and every restore call.
class MultiKernelStubHandler final
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    // One kernel the handler launches: the launch name it reports (empty for a stage that
    // launches one kernel) and the key of the kernel's `sources` it reads (empty for the
    // kernel's `source`).
    struct Launch
    {
        std::string name;
        std::string sourceKey;
    };

    struct Config
    {
        // The name the handler saves its launch inputs under. Register it with a SymbolScope.
        std::string alias;
        // Saved as the launch value `tag`, and read back from it at restore.
        std::string tag;
        // The kernels the handler launches. Empty is one launch of `source` that the
        // prepared dispatch does not report.
        std::vector<Launch> launches;
        // saveLaunchInputs() answers nullopt when false. supportsSaving() needs it and `restores`.
        bool saves = true;
        // supportsSavingAllLaunches().
        bool optsIn = false;
        // The user-supplied scalar UIDs that prepared and restored dispatches report.
        std::vector<int64_t> uids;
        size_t workspaceBytes = 0;

        // Restore: false is a handler that restores nothing (the SDK default, nullptr).
        bool restores = true;
        // Restore refuses the launch values.
        bool rejectsLaunchValues = false;
        // Restore refuses a launch set other than this one.
        std::optional<std::set<std::string>> launchNames;
        // Restore refuses a code whose recorded signature differs, by launch name.
        std::map<std::string, std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>> signatures;
        // The restored prepared dispatch is a StubForeignPrepared.
        bool restoresForeignPrepared = false;
    };

    struct Received
    {
        bool several = false;
        hipdnn_plugin_sdk::ingestor::SavedLaunchInputs inputs;
        std::vector<hipdnn_plugin_sdk::ingestor::SavedKernelCode> codes;
        int deviceOrdinal = -1;
    };

    struct Launched
    {
        std::string tag;
        void* workspace = nullptr;
    };

    MultiKernelStubHandler(Config config, std::string arch)
        : _config(std::move(config))
        , _arch(std::move(arch))
    {
    }

    const Config& config() const
    {
        return _config;
    }

    const std::vector<Received>& received() const
    {
        return _received;
    }

    const std::vector<Launched>& launched() const
    {
        return _launched;
    }

    size_t workspaceBytes(
        const hipdnn_plugin_sdk::ingestor::MatchContext& /*context*/,
        const hipdnn_plugin_sdk::ingestor::BoundTokens& /*bound*/,
        const hipdnn_plugin_sdk::ingestor::KernelDefinition& /*kernel*/) const override
    {
        return _config.workspaceBytes;
    }

    std::unique_ptr<hipdnn_plugin_sdk::ingestor::PreparedDispatch>
        prepare(const hipdnn_plugin_sdk::ingestor::MatchContext& /*context*/,
                const hipdnn_plugin_sdk::ingestor::BoundTokens& /*bound*/,
                const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel) const override
    {
        std::vector<IngestorKernelCode> codes;
        std::vector<StubKernelPrepared::Launch> launches;
        const auto add = [&](const std::string& name, const std::string& sourceKey) {
            const auto& source = sourceKey.empty() ? kernel.source : kernel.sources.at(sourceKey);
            codes.push_back(packedKernelCode(kernel, source, _arch, _loader));
            launches.push_back({name, source.kind, source.signature});
        };
        if(_config.launches.empty())
        {
            add("", "");
        }
        for(const auto& launch : _config.launches)
        {
            add(launch.name, launch.sourceKey);
        }
        return std::make_unique<StubKernelPrepared>(std::move(codes),
                                                    std::move(launches),
                                                    !_config.launches.empty(),
                                                    _config.uids,
                                                    _config.tag);
    }

    void launch(const Handle& /*handle*/,
                const hipdnn_plugin_sdk::ingestor::PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                uint32_t /*numDeviceBuffers*/,
                void* workspace) const override
    {
        _launched.push_back({tagOf(prepared), workspace});
    }

    std::optional<hipdnn_plugin_sdk::ingestor::SavedLaunchInputs> saveLaunchInputs(
        const hipdnn_plugin_sdk::ingestor::PreparedDispatch& prepared) const override
    {
        if(!_config.saves)
        {
            return std::nullopt;
        }
        return hipdnn_plugin_sdk::ingestor::SavedLaunchInputs{_config.alias,
                                                              {{"tag", tagOf(prepared)}}};
    }

    std::unique_ptr<hipdnn_plugin_sdk::ingestor::PreparedDispatch>
        restoreLaunch(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                      hipdnn_plugin_sdk::ingestor::SavedKernelCode code,
                      int deviceOrdinal) const override
    {
        std::vector<hipdnn_plugin_sdk::ingestor::SavedKernelCode> codes;
        codes.push_back(std::move(code));
        return restore(false, inputs, codes, deviceOrdinal);
    }

    bool supportsSaving() const override
    {
        return _config.saves && _config.restores;
    }

    bool supportsSavingAllLaunches() const override
    {
        return _config.optsIn;
    }

    std::unique_ptr<hipdnn_plugin_sdk::ingestor::PreparedDispatch>
        restoreLaunches(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                        std::vector<hipdnn_plugin_sdk::ingestor::SavedKernelCode> codes,
                        int deviceOrdinal) const override
    {
        return restore(true, inputs, codes, deviceOrdinal);
    }

private:
    static std::string tagOf(const hipdnn_plugin_sdk::ingestor::PreparedDispatch& prepared)
    {
        if(const auto* built = dynamic_cast<const StubKernelPrepared*>(&prepared))
        {
            return built->tag();
        }
        if(const auto* foreign = dynamic_cast<const StubForeignPrepared*>(&prepared))
        {
            return foreign->tag();
        }
        return {};
    }

    std::unique_ptr<hipdnn_plugin_sdk::ingestor::PreparedDispatch>
        restore(bool several,
                const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                const std::vector<hipdnn_plugin_sdk::ingestor::SavedKernelCode>& codes,
                int deviceOrdinal) const
    {
        _received.push_back({several, inputs, codes, deviceOrdinal});
        if(!_config.restores)
        {
            return nullptr;
        }
        if(_config.rejectsLaunchValues)
        {
            serialization::refuseIngestorPlan(serialization::IngestorPlanRefusal::INCOMPATIBLE,
                                              "the stub handler refuses its launch values");
        }
        std::set<std::string> names;
        for(const auto& code : codes)
        {
            const auto expected = _config.signatures.find(code.launchName);
            if(expected != _config.signatures.end())
            {
                // A drifted signature is a plan this provider cannot restore, as
                // restoreIngestorLaunch() refuses it.
                requireRestoredSignature(inputs, code, expected->second);
            }
            names.insert(code.launchName);
        }
        if(_config.launchNames.has_value() && names != *_config.launchNames)
        {
            serialization::refuseIngestorPlan(serialization::IngestorPlanRefusal::INCOMPATIBLE,
                                              "the stub handler does not launch this set of "
                                              "launches");
        }

        std::string tag;
        const auto saved = inputs.values.find("tag");
        if(saved != inputs.values.end())
        {
            tag = std::get<std::string>(saved->second);
        }
        if(_config.restoresForeignPrepared)
        {
            return std::make_unique<StubForeignPrepared>(std::move(tag));
        }
        std::vector<IngestorKernelCode> restored;
        std::vector<StubKernelPrepared::Launch> launches;
        for(const auto& code : codes)
        {
            restored.emplace_back(std::make_unique<MockCompiledProgram>(),
                                  std::make_unique<MockRunnableKernel>());
            launches.push_back({code.launchName, code.sourceKind, code.recordedSignature});
        }
        return std::make_unique<StubKernelPrepared>(
            std::move(restored), std::move(launches), several, _config.uids, std::move(tag));
    }

    Config _config;
    std::string _arch;
    compilation::KpackModuleCache _cache;
    compilation::KpackKernelLoader _loader{_cache};
    mutable std::vector<Received> _received;
    mutable std::vector<Launched> _launched;
};

// A MultiKernelStubHandler registered under its alias for as long as this object lives.
class RegisteredStubHandler
{
public:
    explicit RegisteredStubHandler(MultiKernelStubHandler::Config config,
                                   const std::string& arch = {})
        : handler(std::move(config), arch)
    {
        // Not committed: the alias is unregistered again when this object dies.
        _scope.add(handler.config().alias, &handler);
    }

    MultiKernelStubHandler handler;

private:
    hipdnn_plugin_sdk::ingestor::SymbolScope<Handle> _scope;
};

// The number of entries in the kpack module caches of the packs every test build compiles:
// Pointwise, ConvFwd, gfx950 dense and unified 2D/3D, and gfx942 dense. The FlyDSL caches
// (optional) and the gfx1151 caches (file-local) are not counted.
inline size_t packModuleCacheEntries()
{
    return pointwiseKpackModuleCache().size() + convFwdKpackModuleCache().size()
           + gfx950AttentionDenseKpackModuleCache().size()
           + gfx950AttentionUnified2DKpackModuleCache().size()
           + gfx950AttentionUnified3DKpackModuleCache().size()
           + gfx942AttentionDenseKpackModuleCache().size();
}

// ---------------------------------------------------------------------------
// The plan serialization hooks, called the way hipDNN calls them
// ---------------------------------------------------------------------------

namespace plan_hooks
{

inline std::string lastError()
{
    const char* message = nullptr;
    hipdnnPluginGetLastErrorString(&message);
    return message == nullptr ? std::string() : std::string(message);
}

// A handle whose container loads the engines of the unit descriptor root, which is what the
// hooks get from hipDNN.
inline std::unique_ptr<Handle> makeHandle()
{
    auto handle = std::make_unique<Handle>();
    handle->container = std::make_shared<core::Container>();
    return handle;
}

// Checks that a hook returned `expected`, and that the last plugin error starts with `prefix`
// and holds `phrase`. Returns the last plugin error.
inline std::string expectRefusal(hipdnnPluginStatus_t status,
                                 hipdnnPluginStatus_t expected,
                                 std::string_view prefix,
                                 const std::string& phrase)
{
    const std::string message = lastError();
    EXPECT_EQ(status, expected) << message;
    EXPECT_EQ(message.rfind(prefix, 0), 0U) << message;
    EXPECT_NE(message.find(phrase), std::string::npos) << message;
    return message;
}

// Saves the plan of `context` through the plugin save hook. `bytes` receives a copy of the
// saved bytes, and the hook's own buffer goes back through the destroy hook.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void save(Handle& handle, Context& context, std::vector<uint8_t>& bytes)
{
    hipdnnPluginConstData_t serialized{};
    ASSERT_EQ(hipdnnEnginePluginSerializeExecutionContext(&handle, &context, &serialized),
              HIPDNN_PLUGIN_STATUS_SUCCESS)
        << lastError();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0U);
    const auto* begin = static_cast<const uint8_t*>(serialized.ptr);
    bytes.assign(begin, begin + serialized.size);
    ASSERT_EQ(hipdnnEnginePluginDestroySerializedExecutionContext(&handle, &serialized),
              HIPDNN_PLUGIN_STATUS_SUCCESS)
        << lastError();
}

// An execution context that the load hook created. Destroys it through the plugin API.
class LoadedPlan
{
public:
    explicit LoadedPlan(Handle& handle)
        : _handle(&handle)
    {
    }

    ~LoadedPlan()
    {
        if(_context != nullptr)
        {
            static_cast<void>(hipdnnEnginePluginDestroyExecutionContext(_handle, _context));
        }
    }

    LoadedPlan(const LoadedPlan&) = delete;
    LoadedPlan& operator=(const LoadedPlan&) = delete;
    LoadedPlan(LoadedPlan&&) = delete;
    LoadedPlan& operator=(LoadedPlan&&) = delete;

    // Loads `bytes` through the plugin load hook, on the handle this object was made for.
    //
    // Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
    void load(const std::vector<uint8_t>& bytes)
    {
        const hipdnnPluginConstData_t serialized{bytes.data(), bytes.size()};
        ASSERT_EQ(
            hipdnnEnginePluginCreateExecutionContextFromSerialized(_handle, &serialized, &_context),
            HIPDNN_PLUGIN_STATUS_SUCCESS)
            << lastError();
        ASSERT_NE(_context, nullptr);
    }

    const hipdnn_plugin_sdk::IPlan<Handle>& plan() const
    {
        return static_cast<const Context*>(_context)->plan();
    }

private:
    Handle* _handle;
    hipdnnEnginePluginExecutionContext_t _context = nullptr;
};

// Loads `bytes` through the plugin load hook and expects the refusal of `kind`: the status and
// the message prefix of the category, a message that holds `phrase`, and no execution context.
// Returns the message.
inline std::string expectLoadRefusal(Handle& handle,
                                     const std::vector<uint8_t>& bytes,
                                     serialization::IngestorPlanRefusal kind,
                                     const std::string& phrase)
{
    Context sentinel;
    hipdnnEnginePluginExecutionContext_t created = &sentinel;
    const hipdnnPluginConstData_t serialized{bytes.data(), bytes.size()};
    const auto status
        = hipdnnEnginePluginCreateExecutionContextFromSerialized(&handle, &serialized, &created);
    const bool incompatible = kind == serialization::IngestorPlanRefusal::INCOMPATIBLE;
    const auto message = expectRefusal(
        status,
        incompatible ? HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE : HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
        incompatible ? serialization::INGESTOR_PLAN_INCOMPATIBLE_PREFIX
                     : serialization::INGESTOR_PLAN_DAMAGED_PREFIX,
        phrase);
    EXPECT_EQ(created, &sentinel) << "a refused load created a context";
    return message;
}

// The header of the saved bytes.
inline serialization::IngestorPlanHeader headerOf(const std::vector<uint8_t>& bytes)
{
    return serialization::decodeIngestorPlanHeader(bytes.data(), bytes.size());
}

// Reads kind 2 `bytes` into `payload`.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void decodeMultiKernel(const std::vector<uint8_t>& bytes,
                              serialization::IngestorMultiKernelPayload& payload)
{
    auto saved = serialization::decodeIngestorSavedPlan(bytes.data(), bytes.size());
    auto* multi = std::get_if<serialization::IngestorMultiKernelPayload>(&saved);
    ASSERT_NE(multi, nullptr) << "the saved plan is not a kind 2 payload";
    payload = std::move(*multi);
}

// Replaces one byte inside the stored bytes of `object` in the encoded `payload` and writes the
// body digest again. Only the SHA-256 of the code object can notice the damage.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void damageStoredCodeObject(std::vector<uint8_t>& payload,
                                   const std::vector<uint8_t>& object)
{
    ASSERT_FALSE(object.empty());
    const auto body
        = payload.begin() + static_cast<std::ptrdiff_t>(serialization::INGESTOR_PLAN_HEADER_SIZE);
    const auto at = std::search(body, payload.end(), object.begin(), object.end());
    ASSERT_NE(at, payload.end()) << "the code object is not stored in the payload";
    *(at + static_cast<std::ptrdiff_t>(object.size() / 2)) ^= 0xFF;
    serialization::resealIngestorPlan(payload);
}

// A copy of `object` with one byte changed, and the SHA-256 of the changed bytes.
inline serialization::IngestorCodeObjectPayload
    damagedCopyOf(const serialization::IngestorCodeObjectPayload& object)
{
    auto copy = object;
    copy.codeObject.at(copy.codeObject.size() / 2) ^= 0xFF;
    copy.sha256 = utilities::sha256Hex(copy.codeObject.data(), copy.codeObject.size());
    return copy;
}

} // namespace plan_hooks

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
