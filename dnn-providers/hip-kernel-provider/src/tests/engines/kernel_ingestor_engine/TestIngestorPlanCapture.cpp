// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "PackedKernelSource.hpp"
#include "TestDescriptorRoot.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPlanCapture.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"
#include "utilities/Digest.hpp"
#include "version.h"

/**
 * @file TestIngestorPlanCapture.cpp
 * @brief captureIngestorPlan: which plan a save stores, the refusals for a plan it cannot
 *        store, and the payload it collects from a real kpack plan.
 *
 * Every case that reads a packed archive reads a copy of the packed set in a scratch
 * directory. Other suites corrupt or delete the staged archives.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hipdnn_plugin_sdk::HipdnnPluginException;
using hipdnn_plugin_sdk::ingestor::BenchmarkPlan;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::GenericPlan;
using hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::KernelDispatcher;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_plugin_sdk::ingestor::MatchContext;
using hipdnn_plugin_sdk::ingestor::MetadataValues;
using hipdnn_plugin_sdk::ingestor::PreparedDispatch;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;
using serialization::IngestorPlanPayload;

namespace fixtures = hip_kernel_provider::testing;
namespace packs = hip_kernel_provider::kernel_ingestor_engine::testing;

constexpr const char* SCRATCH_LABEL = "ingestorplancapture";
constexpr const char* ENGINE_NAME = "hipkernel:capture_test_engine";

int64_t testEngineId()
{
    return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
}

// The text of the refusal for a benchmarking plan without a chosen kernel.
constexpr const char* NO_WINNER_PHRASE = "has not chosen a kernel yet";

void expectSaveRefusal(const hipdnn_plugin_sdk::IPlan<Handle>& plan,
                       hipdnnPluginStatus_t status,
                       const std::string& phrase)
{
    const Handle handle;
    try
    {
        static_cast<void>(captureIngestorPlan(plan, handle, testEngineId(), ENGINE_NAME));
        ADD_FAILURE() << "expected a save refusal that contains '" << phrase << "'";
    }
    catch(const HipdnnPluginException& error)
    {
        const std::string message = error.getMessage();
        EXPECT_EQ(error.getStatus(), status) << message;
        const std::string prefix(status == HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE
                                     ? serialization::INGESTOR_PLAN_SAVE_INCOMPATIBLE_PREFIX
                                     : serialization::INGESTOR_PLAN_SAVE_DAMAGED_PREFIX);
        EXPECT_EQ(message.rfind(prefix, 0), 0U) << message;
        EXPECT_NE(message.find(phrase), std::string::npos) << message;
    }
}

void expectInternalError(const hipdnn_plugin_sdk::IPlan<Handle>& plan, const std::string& phrase)
{
    const Handle handle;
    try
    {
        static_cast<void>(captureIngestorPlan(plan, handle, testEngineId(), ENGINE_NAME));
        ADD_FAILURE() << "expected an internal error that contains '" << phrase << "'";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR) << error.getMessage();
        EXPECT_NE(error.getMessage().find(phrase), std::string::npos) << error.getMessage();
    }
}

// ---------------------------------------------------------------------------
// Refusals that need no device
// ---------------------------------------------------------------------------

// A handler with none of the save overrides. It prepares an empty launch and launches
// nothing.
class SavesNothingHandler : public IKernelDispatchHandler<Handle>
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

// Saves its launch inputs under a name no handler is registered under.
class UnregisteredAliasHandler : public SavesNothingHandler
{
public:
    std::optional<SavedLaunchInputs>
        saveLaunchInputs(const PreparedDispatch& /*prepared*/) const override
    {
        return SavedLaunchInputs{"hipkernel.capture_test.unregistered.dispatch.v1", {}};
    }
};

// A kpack kernel definition that names nothing on disk. The stub handlers never read it.
KernelDefinition stubKpackKernel(uint8_t seed)
{
    KernelDefinition kernel;
    kernel.kernelId.fill(seed);
    kernel.name = "capture_stub";
    kernel.source.kind = KernelSourceKind::KPACK;
    return kernel;
}

std::unique_ptr<GenericPlan<Handle>> makeStubPlan(const IKernelDispatchHandler<Handle>& handler,
                                                  uint8_t seed)
{
    const packs::GraphFixture fixture(packs::buildPointwiseGraph());
    const BoundTokens bound;
    return std::make_unique<GenericPlan<Handle>>(
        KernelDispatcher<Handle>{stubKpackKernel(seed), &handler}, fixture.context(), bound);
}

TEST(TestIngestorPlanCapture, RefusesABenchmarkPlanWithoutAWinner)
{
    // Real GenericPlans: a benchmarking plan that answered before a winner exists would
    // hand capture a plan to save, and the refusal would come from a later step.
    const SavesNothingHandler handler;
    std::vector<BenchmarkPlan<Handle>::Candidate> candidates;
    candidates.push_back({{}, makeStubPlan(handler, 0x01)});
    candidates.push_back({{}, makeStubPlan(handler, 0x02)});

    const Handle handle;
    const BenchmarkPlan<Handle> plan(std::move(candidates), handle);

    expectSaveRefusal(plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, NO_WINNER_PHRASE);
    expectSaveRefusal(plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "execute it once");
    expectSaveRefusal(plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "benchmarking off");
}

TEST(TestIngestorPlanCapture, RefusesAHandlerThatSavesNothing)
{
    const SavesNothingHandler handler;
    const auto plan = makeStubPlan(handler, 0x03);

    expectSaveRefusal(*plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "does not support saving");
}

TEST(TestIngestorPlanCapture, RefusesAnAliasThatIsNotRegisteredToTheHandler)
{
    const UnregisteredAliasHandler handler;
    const auto plan = makeStubPlan(handler, 0x04);

    expectInternalError(*plan, "is not registered to this handler");
}

TEST(TestIngestorPlanCapture, RefusesAnEngineNameWithoutTheEngineId)
{
    const SavesNothingHandler handler;
    const auto plan = makeStubPlan(handler, 0x05);
    const Handle handle;

    try
    {
        static_cast<void>(
            captureIngestorPlan(*plan, handle, testEngineId(), "hipkernel:another_engine"));
        ADD_FAILURE() << "expected an engine name that does not give the engine id to be refused";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR) << error.getMessage();
        EXPECT_NE(error.getMessage().find("does not give the id"), std::string::npos)
            << error.getMessage();
        EXPECT_NE(error.getMessage().find("hipkernel:another_engine"), std::string::npos)
            << error.getMessage();
        EXPECT_NE(
            error.getMessage().find(hipdnn_data_sdk::utilities::formatEngineIdHex(testEngineId())),
            std::string::npos)
            << error.getMessage();
    }
}

// ---------------------------------------------------------------------------
// Real kpack plans
// ---------------------------------------------------------------------------

// A device buffer of `bytes`, zero-filled.
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t bytes)
    {
        if(hipMalloc(&_ptr, bytes) != hipSuccess)
        {
            _ptr = nullptr;
            return;
        }
        if(hipMemset(_ptr, 0, bytes) != hipSuccess)
        {
            static_cast<void>(hipFree(_ptr));
            _ptr = nullptr;
        }
    }

    ~DeviceBuffer()
    {
        if(_ptr != nullptr)
        {
            static_cast<void>(hipFree(_ptr));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    void* get() const
    {
        return _ptr;
    }

private:
    void* _ptr = nullptr;
};

// Enough bytes for every tensor of the default pointwise and conv test graphs.
constexpr size_t TENSOR_BYTES = 256;

// One buffer per operand uid 1, 2 and 3: the uids both test graphs use.
class OperandBuffers
{
public:
    OperandBuffers()
        : _descriptors{{1, _first.get()}, {2, _second.get()}, {3, _third.get()}}
    {
    }

    bool allocated() const
    {
        return _first.get() != nullptr && _second.get() != nullptr && _third.get() != nullptr;
    }

    const hipdnnPluginDeviceBuffer_t* data() const
    {
        return _descriptors.data();
    }

    uint32_t count() const
    {
        return static_cast<uint32_t>(_descriptors.size());
    }

private:
    DeviceBuffer _first{TENSOR_BYTES};
    DeviceBuffer _second{TENSOR_BYTES};
    DeviceBuffer _third{TENSOR_BYTES};
    std::vector<hipdnnPluginDeviceBuffer_t> _descriptors;
};

enum class PackedSet
{
    POINTWISE,
    CONV_FWD,
};

// Copies the packed set of `set` for the local device into `scratch`. `copy` stays
// empty when the build packed nothing for the device. `deviceProperties` receives the
// properties of device 0, with the decorated arch name the device reports.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
void copyPackedSet(PackedSet set,
                   const ScopedDirectory& scratch,
                   DeviceProperties& deviceProperties,
                   std::filesystem::path& copy)
{
    const std::filesystem::path& root
        = set == PackedSet::POINTWISE ? fixtures::archiveFixtureRoot() : fixtures::unitKpackRoot();

    hipDeviceProp_t properties{};
    std::string arch;
    std::filesystem::path packed;
    ASSERT_NO_FATAL_FAILURE(fixtures::findPackedArchDirectoryUnder(root, properties, arch, packed));

    deviceProperties.gcnArchName = properties.gcnArchName;
    deviceProperties.warpSize = properties.warpSize;
    deviceProperties.multiProcessorCount = properties.multiProcessorCount;

    if(!packed.empty())
    {
        copy = fixtures::copyPackedArchTree(packed, scratch.path());
    }
}

// A GenericPlan over the registered handler of `set`, for `kernel`.
std::unique_ptr<GenericPlan<Handle>> makePackedPlan(PackedSet set,
                                                    const DeviceProperties& deviceProperties,
                                                    const KernelDefinition& kernel)
{
    const auto& pack = set == PackedSet::POINTWISE ? packs::POINTWISE_ADD : packs::CONV_FWD;
    const packs::GraphFixture fixture(
        set == PackedSet::POINTWISE
            ? packs::buildPointwiseGraph()
            : packs::buildConvFwdGraph(hipdnn_flatbuffers_sdk::data_objects::DataType::HALF),
        deviceProperties);

    const auto bound = packs::matchesGraph(pack, fixture.context());
    if(!bound.has_value())
    {
        ADD_FAILURE() << "the test graph does not match its pack";
        return nullptr;
    }
    return std::make_unique<GenericPlan<Handle>>(
        KernelDispatcher<Handle>{kernel, &packs::dispatchHandler(pack)}, fixture.context(), *bound);
}

// The packed Pointwise descriptor with block size 256 (workspace 1024).
constexpr const char* POINTWISE_B256_DESCRIPTOR = "packed_pointwise_add_b256.ukd.json";
// The packed Pointwise pack, whose inline kernel has block size 64 (workspace 0).
constexpr const char* POINTWISE_B64_DESCRIPTOR = "packed_pointwise_add.kdp.json";
// The packed ConvFwd descriptor with block size 64.
constexpr const char* CONV_FWD_DESCRIPTOR = "conv_fwd_f16_block64.ukd.json";

struct CaptureCase
{
    std::string name;
    PackedSet set;
    const char* descriptor;
    std::string alias;
    MetadataValues launchValues;
    uint64_t workspaceBytes;
};

class TestIngestorPlanCaptureKpack : public ::testing::TestWithParam<CaptureCase>
{
};

TEST_P(TestIngestorPlanCaptureKpack, CapturesAKpackPlan)
{
    SKIP_IF_NO_DEVICES();
    const auto& param = GetParam();

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    DeviceProperties deviceProperties;
    std::filesystem::path packed;
    ASSERT_NO_FATAL_FAILURE(copyPackedSet(param.set, scratch, deviceProperties, packed));
    if(packed.empty())
    {
        GTEST_SKIP() << "nothing was packed for this device (" << deviceProperties.gcnArchName
                     << ")";
    }

    KernelDefinition kernel;
    ASSERT_NO_FATAL_FAILURE(fixtures::readPackedKernelDefinition(packed, param.descriptor, kernel));
    const auto plan = makePackedPlan(param.set, deviceProperties, kernel);
    ASSERT_NE(plan, nullptr);

    const Handle handle;
    const IngestorPlanPayload payload
        = captureIngestorPlan(*plan, handle, testEngineId(), ENGINE_NAME);

    EXPECT_EQ(payload.engineName, ENGINE_NAME);
    EXPECT_EQ(payload.kernelDescriptorId, kernel.kernelId);
    EXPECT_EQ(payload.workspaceBytes, param.workspaceBytes);
    EXPECT_EQ(payload.workspaceBytes, plan->getWorkspaceSize(handle));
    EXPECT_TRUE(payload.runtimePassByValueUids.empty());
    EXPECT_EQ(payload.dispatchSymbol, param.alias);
    EXPECT_EQ(payload.launchValues, param.launchValues);
    EXPECT_EQ(payload.sourceKind, KernelSourceKind::KPACK);
    EXPECT_EQ(payload.symbol, kernel.source.symbol);
    EXPECT_EQ(payload.sha256, kernel.source.sha256);
    EXPECT_EQ(utilities::sha256Hex(payload.codeObject.data(), payload.codeObject.size()),
              kernel.source.sha256);
    EXPECT_TRUE(hipdnn_plugin_sdk::archMatches(
        deviceProperties.gcnArchName, payload.target, hipdnn_plugin_sdk::ArchMatchMode::PREFIX))
        << payload.target << " does not serve " << deviceProperties.gcnArchName;
    EXPECT_TRUE(serialization::detail::sameKernelSignature(payload.recordedSignature,
                                                           kernel.source.signature));
    EXPECT_EQ(payload.providerVersion, HIP_KERNEL_PROVIDER_VERSION_STRING);

    const auto encoded = serialization::encodeIngestorPlan(payload);
    EXPECT_TRUE(serialization::decodeIngestorPlan(encoded.data(), encoded.size()) == payload);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestIngestorPlanCaptureKpack,
    ::testing::Values(CaptureCase{"Pointwise",
                                  PackedSet::POINTWISE,
                                  POINTWISE_B256_DESCRIPTOR,
                                  "hipkernel.pointwise.dispatch.v1",
                                  {{"input_a.uid", int64_t{packs::INPUT_A_UID}},
                                   {"input_b.uid", int64_t{packs::INPUT_B_UID}},
                                   {"output.uid", int64_t{packs::OUTPUT_UID}},
                                   {"block_size", int64_t{256}}},
                                  1024},
                      CaptureCase{"ConvFwd",
                                  PackedSet::CONV_FWD,
                                  CONV_FWD_DESCRIPTOR,
                                  "hipkernel.conv_fwd.dispatch.v1",
                                  {{"x.uid", int64_t{packs::CONV_X_UID}},
                                   {"w.uid", int64_t{packs::CONV_W_UID}},
                                   {"y.uid", int64_t{packs::CONV_Y_UID}},
                                   {"n", int64_t{1}},
                                   {"c", int64_t{1}},
                                   {"h", int64_t{3}},
                                   {"width", int64_t{3}},
                                   {"k", int64_t{1}},
                                   {"r", int64_t{2}},
                                   {"s", int64_t{2}},
                                   {"block_size", int64_t{64}}},
                                  0}),
    [](const ::testing::TestParamInfo<CaptureCase>& info) { return info.param.name; });

// Saving accepts only kpack kernels: embedded_source kernels have no recorded argument
// signature, and their code bytes are not exposed. When either fact changes, change this
// test together with the gate.
class TestIngestorPlanCaptureGate : public ::testing::TestWithParam<KernelSourceKind>
{
};

TEST_P(TestIngestorPlanCaptureGate, AppliesTheSourceKindGate)
{
    SKIP_IF_NO_DEVICES();
    const Handle handle;

    if(GetParam() == KernelSourceKind::KPACK)
    {
        const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
        DeviceProperties deviceProperties;
        std::filesystem::path packed;
        ASSERT_NO_FATAL_FAILURE(
            copyPackedSet(PackedSet::CONV_FWD, scratch, deviceProperties, packed));
        if(packed.empty())
        {
            GTEST_SKIP() << "nothing was packed for this device (" << deviceProperties.gcnArchName
                         << ")";
        }

        KernelDefinition kernel;
        ASSERT_NO_FATAL_FAILURE(
            fixtures::readPackedKernelDefinition(packed, CONV_FWD_DESCRIPTOR, kernel));
        const auto plan = makePackedPlan(PackedSet::CONV_FWD, deviceProperties, kernel);
        ASSERT_NE(plan, nullptr);

        const auto payload = captureIngestorPlan(*plan, handle, testEngineId(), ENGINE_NAME);
        EXPECT_EQ(payload.sourceKind, KernelSourceKind::KPACK);
        return;
    }

    const packs::GraphFixture fixture(packs::buildPointwiseGraph(),
                                      packs::currentDeviceProperties());
    const auto bound = packs::matchesGraph(packs::POINTWISE_ADD, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto kernel = packs::makeKernel(64, "FLOAT");
    ASSERT_EQ(kernel.source.kind, KernelSourceKind::EMBEDDED_SOURCE);
    const GenericPlan<Handle> plan(
        KernelDispatcher<Handle>{kernel, &packs::dispatchHandler(packs::POINTWISE_ADD)},
        fixture.context(),
        *bound);

    expectSaveRefusal(plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "embedded_source");
}

INSTANTIATE_TEST_SUITE_P(,
                         TestIngestorPlanCaptureGate,
                         ::testing::Values(KernelSourceKind::KPACK,
                                           KernelSourceKind::EMBEDDED_SOURCE),
                         [](const ::testing::TestParamInfo<KernelSourceKind>& info) {
                             return info.param == KernelSourceKind::KPACK
                                        ? std::string("KPACK")
                                        : std::string("EMBEDDED_SOURCE");
                         });

TEST(TestIngestorPlanCapture, RefusesWhenTheArchiveIsGone)
{
    SKIP_IF_NO_DEVICES();

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    DeviceProperties deviceProperties;
    std::filesystem::path packed;
    ASSERT_NO_FATAL_FAILURE(copyPackedSet(PackedSet::CONV_FWD, scratch, deviceProperties, packed));
    if(packed.empty())
    {
        GTEST_SKIP() << "nothing was packed for this device (" << deviceProperties.gcnArchName
                     << ")";
    }

    KernelDefinition kernel;
    ASSERT_NO_FATAL_FAILURE(
        fixtures::readPackedKernelDefinition(packed, CONV_FWD_DESCRIPTOR, kernel));
    const auto plan = makePackedPlan(PackedSet::CONV_FWD, deviceProperties, kernel);
    ASSERT_NE(plan, nullptr);

    const Handle handle;
    const OperandBuffers buffers;
    ASSERT_TRUE(buffers.allocated());
    plan->execute(handle, buffers.data(), buffers.count());
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    const std::filesystem::path archive
        = std::filesystem::weakly_canonical(kernel.originDirectory / kernel.source.library);
    ASSERT_TRUE(std::filesystem::remove(archive)) << archive;

    expectSaveRefusal(*plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "OPEN_ARCHIVE");
    expectSaveRefusal(*plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, archive.string());

    // The plan keeps its loaded module, so it still executes.
    plan->execute(handle, buffers.data(), buffers.count());
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST(TestIngestorPlanCapture, CapturesTheBenchmarkWinnersOwnWorkspace)
{
    SKIP_IF_NO_DEVICES();

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    DeviceProperties deviceProperties;
    std::filesystem::path packed;
    ASSERT_NO_FATAL_FAILURE(copyPackedSet(PackedSet::POINTWISE, scratch, deviceProperties, packed));
    if(packed.empty())
    {
        GTEST_SKIP() << "nothing was packed for this device (" << deviceProperties.gcnArchName
                     << ")";
    }

    KernelDefinition smallKernel;
    KernelDefinition largeKernel;
    ASSERT_NO_FATAL_FAILURE(
        fixtures::readPackedKernelDefinition(packed, POINTWISE_B64_DESCRIPTOR, smallKernel));
    ASSERT_NO_FATAL_FAILURE(
        fixtures::readPackedKernelDefinition(packed, POINTWISE_B256_DESCRIPTOR, largeKernel));
    ASSERT_NE(smallKernel.kernelId, largeKernel.kernelId);

    auto smallPlan = makePackedPlan(PackedSet::POINTWISE, deviceProperties, smallKernel);
    auto largePlan = makePackedPlan(PackedSet::POINTWISE, deviceProperties, largeKernel);
    ASSERT_NE(smallPlan, nullptr);
    ASSERT_NE(largePlan, nullptr);
    const Handle handle;
    ASSERT_EQ(smallPlan->getWorkspaceSize(handle), 0U);
    ASSERT_EQ(largePlan->getWorkspaceSize(handle), 1024U);
    const hipdnn_plugin_sdk::IPlan<Handle>* const winner = smallPlan.get();

    // The winner is the candidate with the smaller workspace, so its own size differs from
    // the largest size the benchmarking plan reports.
    std::vector<BenchmarkPlan<Handle>::Candidate> candidates;
    candidates.push_back({largeKernel.kernelId, std::move(largePlan)});
    candidates.push_back({smallKernel.kernelId, std::move(smallPlan)});
    const BenchmarkPlan<Handle> plan(std::move(candidates),
                                     handle,
                                     [winner](const hipdnn_plugin_sdk::IPlan<Handle>& candidate,
                                              const Handle& /*handle*/,
                                              const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                                              uint32_t /*numDeviceBuffers*/,
                                              void* /*workspace*/) -> std::optional<double> {
                                         return &candidate == winner ? 1.0 : 5.0;
                                     });
    ASSERT_EQ(plan.getWorkspaceSize(handle), 1024U);

    const OperandBuffers buffers;
    const DeviceBuffer workspace(plan.getWorkspaceSize(handle));
    ASSERT_TRUE(buffers.allocated());
    ASSERT_NE(workspace.get(), nullptr);
    plan.execute(handle, buffers.data(), buffers.count(), workspace.get());
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    const auto payload = captureIngestorPlan(plan, handle, testEngineId(), ENGINE_NAME);
    EXPECT_EQ(payload.kernelDescriptorId, smallKernel.kernelId);
    EXPECT_EQ(payload.workspaceBytes, 0U);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
