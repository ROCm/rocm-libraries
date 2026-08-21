// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlanBuilder.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestPointwiseKpackDispatch.cpp
 * @brief The pointwise pack seen from the dispatch side once a kernel's code comes from
 *        a kpack archive rather than embedded source.
 *
 * Three questions, only one of which needs a device:
 *  - the workspace query is unchanged by the new source kind (AC #5);
 *  - a kpack whose archive is absent costs only itself, and a sibling still serves
 *    (AC #8) -- this is the GPU-less case the plan's AC #8 note requires, and the
 *    failure it turns on is raised at archive-open, before any HIP call;
 *  - two dispatches over one (archive, toc_key, arch) load one module (AC #6), which
 *    cannot be observed without a device and is marked [GPU] accordingly.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using hip_kernel_provider::kernel_ingestor_engine::testing::BLOCK_SIZE_FIELD;
using hip_kernel_provider::kernel_ingestor_engine::testing::buildPointwiseGraph;
using hip_kernel_provider::kernel_ingestor_engine::testing::dispatchHandler;
using hip_kernel_provider::kernel_ingestor_engine::testing::DTYPE_FIELD;
using hip_kernel_provider::kernel_ingestor_engine::testing::GraphFixture;
using hip_kernel_provider::kernel_ingestor_engine::testing::matchesGraph;
using hip_kernel_provider::kernel_ingestor_engine::testing::POINTWISE_ADD;
using hip_kernel_provider::kernel_ingestor_engine::testing::testDeviceProperties;

/// Where this build stages the descriptors it packed, one subdirectory per arch. Same
/// value main.cpp points the binary at, so the [GPU] case below dispatches against the
/// archive the packaging rule produced for the machine running the tests.
constexpr const char* PACKED_DESCRIPTOR_ROOT = HIPDNN_TEST_DESCRIPTOR_DIR;

/// The packaged descriptor the [GPU] case takes its archive and toc_key from. Read out of
/// the built file rather than written here: a toc_key tracks the build defines and the
/// packaging layout, both the packer's business, and a copy here would silently decouple
/// this test from the artifact it exists to read.
constexpr const char* PACKED_UKD_DESCRIPTOR = "packed_pointwise_add_b256.ukd.json";

/// The entry point that descriptor names.
constexpr const char* PACKED_SYMBOL = "PointwiseAdd";

/// The kpack coordinates a built descriptor declares. `library` is kept in its authored,
/// relative form because a KernelDefinition carries it that way and resolves it against
/// originDirectory.
struct PackagedKernelSource
{
    std::string library;
    std::string tocKey;
};

/// The bare arch of device 0 and the directory this build packed for it. `directory` is
/// left empty when nothing was packed for that arch -- environmental, since the build
/// packs per arch and the device may sit outside GPU_TARGETS.
///
/// hipGetDeviceProperties reports feature flags on some configurations ("gfx1152:xnack-")
/// while the packager names its subdirectory and the archive's arch entry with the bare
/// name, so everything past here uses the stripped form.
///
/// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
void findPackagedDirectory(hipDeviceProp_t& properties,
                           std::string& arch,
                           std::filesystem::path& directory)
{
    ASSERT_EQ(hipGetDeviceProperties(&properties, 0), hipSuccess);

    const std::string reported = properties.gcnArchName;
    arch = reported.substr(0, reported.find(':'));

    const std::filesystem::path candidate = std::filesystem::path(PACKED_DESCRIPTOR_ROOT) / arch;
    directory = std::filesystem::is_directory(candidate) ? candidate : std::filesystem::path{};
}

/// Reads `kernel_source` out of a built .ukd.json. Parsed directly rather than through
/// DescriptorLoader: this case exercises the dispatch handler's loader path, and routing
/// through the loader would couple it to a contract the integration tier already covers.
///
/// Everything below is asserted, never skipped: the per-arch directory exists by the time
/// this is called, so anything missing inside it is a broken build.
///
/// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
void readPackagedKernelSource(const std::filesystem::path& directory,
                              const std::string& descriptorFile,
                              PackagedKernelSource& out)
{
    const std::filesystem::path descriptor = directory / descriptorFile;
    ASSERT_TRUE(std::filesystem::exists(descriptor))
        << "the packaged descriptor is missing: " << descriptor;

    std::ifstream in(descriptor);
    ASSERT_TRUE(in.good()) << "could not open " << descriptor;

    nlohmann::json document;
    ASSERT_NO_THROW(document = nlohmann::json::parse(in)) << descriptor;
    ASSERT_TRUE(document.contains("kernel_source")) << descriptor;

    const nlohmann::json& source = document["kernel_source"];
    ASSERT_TRUE(source.contains("toc_key")) << descriptor;
    ASSERT_TRUE(source.contains("library")) << descriptor;

    out.tocKey = source["toc_key"].get<std::string>();
    out.library = source["library"].get<std::string>();
    ASSERT_TRUE(std::filesystem::exists(directory / out.library))
        << descriptor << " names an archive that is not on disk: " << directory / out.library;
}

DescriptorId id(uint8_t seed)
{
    DescriptorId value{};
    value.fill(seed);
    return value;
}

/// A KernelDefinition whose code comes from a kpack archive at
/// `originDirectory / library`. Metadata carries exactly what the pointwise handler
/// reads, so the only thing that differs from the embedded-source path is the source.
KernelDefinition makeKpackKernel(const std::filesystem::path& originDirectory,
                                 const std::string& library,
                                 const std::string& tocKey,
                                 const std::string& symbol,
                                 int64_t blockSize,
                                 uint8_t seed)
{
    KernelDefinition kernel;
    kernel.kernelId = id(seed);
    kernel.packId = id(static_cast<uint8_t>(seed + 1));
    kernel.dispatchId = id(static_cast<uint8_t>(seed + 2));
    kernel.name = "pointwise_add_f32_kpack";
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.source.library = library;
    kernel.source.tocKey = tocKey;
    kernel.source.symbol = symbol;
    kernel.originDirectory = originDirectory;
    kernel.metadata = {{std::string(BLOCK_SIZE_FIELD), blockSize},
                       {std::string(DTYPE_FIELD), std::string("FLOAT")}};
    return kernel;
}

// ---------------------------------------------------------------------------
// AC #5 -- the workspace seam, unchanged
// ---------------------------------------------------------------------------

/// `workspaceBytes` reads metadata only, so it never reaches a loader and needs no
/// device. That is the claim AC #5 makes -- "satisfied by construction" -- and this is
/// what makes it observable rather than merely argued: the same handler, asked about a
/// KPACK kernel, answers from the same metadata it always did.
TEST(TestPointwiseKpackDispatch, QueriesWorkspaceForAKpackKernel)
{
    const GraphFixture fixture(buildPointwiseGraph(), testDeviceProperties());
    const auto bound = matchesGraph(POINTWISE_ADD, fixture.context());
    ASSERT_TRUE(bound.has_value());

    const auto& handler = dispatchHandler(POINTWISE_ADD);

    // 256 is the pack's large-block kernel, the one that reports scratch; 64 is not.
    // Not named `small`: Windows' rpcndr.h defines that as a macro.
    const auto largeBlock
        = makeKpackKernel("/nonexistent", "pack.kpack", "toc#0", "PointwiseAdd", 256, 0x40);
    const auto smallBlock
        = makeKpackKernel("/nonexistent", "pack.kpack", "toc#0", "PointwiseAdd", 64, 0x50);

    EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, largeBlock), 1024U);
    EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, smallBlock), 0U);
}

// ---------------------------------------------------------------------------
// AC #8 -- a kpack that cannot load costs only itself
// ---------------------------------------------------------------------------

/// Prepares without touching HIP at all.
///
/// The sibling that must keep serving cannot be an EMBEDDED_SOURCE pointwise kernel:
/// that path runs HIPRTC and then hipModuleLoadData, so it needs a device, and the case
/// below would then be a [GPU] test -- exactly what the plan's AC #8 note forbids. It
/// also cannot be a second kpack kernel, for the same reason. So the sibling is a
/// test-local handler: the *failing* half is the real pointwise handler on the real
/// loader against a real absent path, which is where the evidence lives, and the
/// serving half only has to be something that succeeds.
class NoHipDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& /*context*/,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& /*kernel*/) const override
    {
        return std::make_unique<PreparedDispatch>();
    }

    void launch(const Handle& /*handle*/,
                const PreparedDispatch& /*prepared*/,
                const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                uint32_t /*numDeviceBuffers*/,
                void* /*workspace*/) const override
    {
    }
};

/// A fixed, device-less machine identity. The arch is testDeviceProperties()' gfx000,
/// which no kpack archive claims -- irrelevant here, because the archive this test names
/// does not exist and the failure is raised before any arch is consulted.
class FixedDeviceResolver : public IDeviceResolver<Handle>
{
public:
    DeviceId deviceId(const Handle& /*handle*/) const override
    {
        return 0;
    }

    const DeviceProperties& deviceProperties(DeviceId /*deviceId*/) const override
    {
        return _properties;
    }

private:
    DeviceProperties _properties = testDeviceProperties();
};

struct KpackSettings
{
    IngestorSettings ingestorSettings;
};

/// The minimum a GenericPlanBuilder asks of its context. The provider's real Context is
/// not used because it needs a live Container and handle; nothing below reads anything
/// the real one would have added.
class KpackContext
{
public:
    const KpackSettings& executionSettings() const
    {
        return _settings;
    }

    void setPlan(std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>> plan)
    {
        _plan = std::move(plan);
    }

    bool hasPlan() const
    {
        return _plan != nullptr;
    }

private:
    KpackSettings _settings;
    std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>> _plan;
};

using KpackPlanBuilder = GenericPlanBuilder<Handle, KpackSettings, KpackContext>;

constexpr const char* SIBLING_DISPATCH_SYMBOL = "hipkernel.pointwise.kpack_fallback_probe";

/// Two packs, not one. A KernelDefinition takes its dispatchId from its pack, so two
/// kernels reached through two different handlers cannot share a pack -- the plan's
/// "two-kernel pack" phrasing describes the intent, and two single-kernel packs over one
/// graph is the shape that expresses it. The engine declares no UMD, so both packs reach
/// the ranking; the real graph_match still runs, because the pointwise handler reads the
/// tokens it binds and would otherwise fail for the wrong reason.
DescriptorSet makeTwoPackSet(const std::filesystem::path& emptyDirectory)
{
    DescriptorSet set;
    set.engine.id = id(0x10);
    set.engine.name = "hipkernel:PointwiseKpackProbe";
    set.engine.metadataSchemaId = id(0x11);
    set.engine.graphMatchNativeSymbol = std::string(POINTWISE_ADD.graphMatcher);

    set.schema.id = id(0x11);
    set.schema.name = "pointwise kpack probe";
    set.schema.fields
        = {MetadataField{std::string(BLOCK_SIZE_FIELD), MetadataType::INT, std::nullopt},
           MetadataField{std::string(DTYPE_FIELD), MetadataType::STRING, std::nullopt}};

    DispatchDescriptor kpackDispatch;
    kpackDispatch.id = id(0x20);
    kpackDispatch.name = "the real pointwise dispatch";
    kpackDispatch.dispatchSymbol = std::string(POINTWISE_ADD.dispatch);

    DispatchDescriptor siblingDispatch;
    siblingDispatch.id = id(0x21);
    siblingDispatch.name = "a dispatch that needs no device";
    siblingDispatch.dispatchSymbol = SIBLING_DISPATCH_SYMBOL;

    set.dispatches = {kpackDispatch, siblingDispatch};

    KernelDescriptor failing;
    failing.id = id(0x30);
    failing.name = "pointwise_add_f32_kpack";
    failing.source.kind = KernelSourceKind::KPACK;
    // Relative, resolved against originDirectory, and naming nothing: the directory
    // exists so the failure is the archive's absence and not a missing parent.
    failing.source.library = "there-is-no-archive-here.kpack";
    failing.source.tocKey = "lib/libhip.so#0";
    failing.source.symbol = "PointwiseAdd";
    failing.originDirectory = emptyDirectory;
    failing.metadata = {{std::string(BLOCK_SIZE_FIELD), int64_t{256}},
                        {std::string(DTYPE_FIELD), std::string("FLOAT")}};
    // Ranked first: the heuristic is absent, so rank() falls through to priority
    // descending, and the whole point is that the *front* candidate is the broken one.
    failing.priority = 100;

    KernelDescriptor sibling;
    sibling.id = id(0x31);
    sibling.name = "the kernel that still serves";
    sibling.source.kind = KernelSourceKind::EMBEDDED_SOURCE;
    sibling.source.sourceFile = "PointwiseAdd.cpp";
    sibling.source.entryPoint = "PointwiseAdd";
    sibling.metadata = {{std::string(BLOCK_SIZE_FIELD), int64_t{64}},
                        {std::string(DTYPE_FIELD), std::string("FLOAT")}};
    sibling.priority = 1;

    KernelDescriptorPack kpackPack;
    kpackPack.id = id(0x40);
    kpackPack.name = "the kpack pack";
    kpackPack.engineId = set.engine.id;
    kpackPack.dispatchId = kpackDispatch.id;
    kpackPack.kernels = {failing};

    KernelDescriptorPack siblingPack;
    siblingPack.id = id(0x41);
    siblingPack.name = "the pack that serves";
    siblingPack.engineId = set.engine.id;
    siblingPack.dispatchId = siblingDispatch.id;
    siblingPack.kernels = {sibling};

    set.packs = {kpackPack, siblingPack};
    return set;
}

/// AC #8's only GPU-less evidence. The front-ranked candidate names a kpack archive that
/// is not there; the loader reports it at archive-open, before HIP is involved, so the
/// whole path runs on a machine with no device. The graph is still served, and the
/// failure is named rather than swallowed.
TEST(TestPointwiseKpackDispatch, SurvivesAKpackWhoseArchiveIsAbsent)
{
    registerNativeIngestorSymbols();

    SymbolScope<Handle> scope;
    const NoHipDispatchHandler siblingHandler;
    scope.add(SIBLING_DISPATCH_SYMBOL, &siblingHandler);

    const hipdnn_test_sdk::utilities::ScopedDirectory emptyDirectory(
        std::filesystem::temp_directory_path() / "hipdnn-kpack-absent-archive");

    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);

    const auto set = makeTwoPackSet(emptyDirectory.path());
    const KernelIngestorStateManager<Handle> stateManager(set.schema,
                                                          set.matchers,
                                                          set.dispatches,
                                                          set.packs,
                                                          makeKernelHeuristic(set.heuristic),
                                                          set.engine.graphMatchNativeSymbol);

    const FixedDeviceResolver resolver;
    const KpackPlanBuilder builder(set.engine, stateManager, resolver);

    const GraphFixture fixture(buildPointwiseGraph(), testDeviceProperties());
    const auto bound = matchesGraph(POINTWISE_ADD, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the probe set relies on the real graph match binding tokens";

    const Handle handle;
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper noConfig(nullptr, 0);
    KpackContext context;

    builder.buildPlan(handle, fixture.context().graph, noConfig, context);

    // The sibling serves...
    EXPECT_TRUE(context.hasPlan());

    // ...and the kernel that could not is named, with the reason, rather than dropped
    // silently. Both halves matter: a plan built from the sibling with no diagnostic
    // would be indistinguishable from the broken kernel never having been a candidate.
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, toString(id(0x30))));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "does not exist"));
}

// ---------------------------------------------------------------------------
// AC #6 -- one module across two dispatches
// ---------------------------------------------------------------------------

/// Two kernels differing only by block size, both naming one (archive, toc_key, arch):
/// the cache must grow by exactly one. Measured as a delta because the cache is
/// process-lifetime and another case may have populated it first.
TEST(TestPointwiseKpackDispatch, LoadsTheModuleOnceAcrossTwoDispatches)
{
    SKIP_IF_NO_DEVICES();

    hipDeviceProp_t properties{};
    std::string arch;
    std::filesystem::path packaged;
    ASSERT_NO_FATAL_FAILURE(findPackagedDirectory(properties, arch, packaged));
    if(packaged.empty())
    {
        GTEST_SKIP() << "nothing was packaged for this device (" << arch
                     << "): " << std::filesystem::path(PACKED_DESCRIPTOR_ROOT) / arch
                     << " does not exist. Environmental -- the build packs per arch and this "
                        "device is outside GPU_TARGETS.";
    }

    PackagedKernelSource packed;
    ASSERT_NO_FATAL_FAILURE(readPackagedKernelSource(packaged, PACKED_UKD_DESCRIPTOR, packed));

    DeviceProperties deviceProperties;
    deviceProperties.gcnArchName = arch;
    deviceProperties.warpSize = properties.warpSize;

    const GraphFixture fixture(buildPointwiseGraph(), deviceProperties);
    const auto bound = matchesGraph(POINTWISE_ADD, fixture.context());
    ASSERT_TRUE(bound.has_value());

    const auto first
        = makeKpackKernel(packaged, packed.library, packed.tocKey, PACKED_SYMBOL, 256, 0x60);
    const auto second
        = makeKpackKernel(packaged, packed.library, packed.tocKey, PACKED_SYMBOL, 64, 0x70);

    const auto& handler = dispatchHandler(POINTWISE_ADD);
    const size_t before = pointwiseKpackModuleCache().size();

    const auto preparedFirst = handler.prepare(fixture.context(), *bound, first);
    const auto preparedSecond = handler.prepare(fixture.context(), *bound, second);
    ASSERT_NE(preparedFirst, nullptr);
    ASSERT_NE(preparedSecond, nullptr);

    EXPECT_EQ(pointwiseKpackModuleCache().size(), before + 1);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
