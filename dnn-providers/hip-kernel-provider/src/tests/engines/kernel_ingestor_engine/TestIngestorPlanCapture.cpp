// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

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
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
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
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "PackedKernelSource.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPlanCapture.hpp"
#include "engines/kernel_ingestor_engine/PackedPlanTestSupport.hpp"
#include "engines/kernel_ingestor_engine/RestoredIngestorPlan.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdPackedCase.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwisePackedCase.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanTestUtilities.hpp"
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

using hipdnn_data_sdk::utilities::Workspace;
using hipdnn_plugin_sdk::ingestor::BenchmarkPlan;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::GenericPlan;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::KernelDispatcher;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_plugin_sdk::ingestor::MetadataValues;
using hipdnn_plugin_sdk::ingestor::PreparedDispatch;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::GraphTensorBundle;
using hipdnn_test_sdk::utilities::ScopedDirectory;
using serialization::IngestorPlanPayload;

namespace fixtures = hip_kernel_provider::testing;
namespace packs = hip_kernel_provider::kernel_ingestor_engine::testing;

using packs::makeStubPlan;
using packs::PackedPlan;
using packs::PackedPlanCase;

constexpr const char* SCRATCH_LABEL = "ingestorplancapture";
constexpr const char* ENGINE_NAME = "hipkernel:capture_test_engine";

int64_t testEngineId()
{
    return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
}

// The text of the refusal for a benchmarking plan without a chosen kernel.
constexpr const char* NO_WINNER_PHRASE = "has not chosen a kernel yet";

// Saves `plan` under the test engine id and the engine name `name`.
auto captureOf(const hipdnn_plugin_sdk::IPlan<Handle>& plan, std::string name = ENGINE_NAME)
{
    return [&plan, engineName = std::move(name)]() {
        const Handle handle;
        static_cast<void>(captureIngestorPlan(plan, handle, testEngineId(), engineName));
    };
}

std::string expectSaveRefusal(const hipdnn_plugin_sdk::IPlan<Handle>& plan,
                              hipdnnPluginStatus_t status,
                              const std::string& phrase)
{
    return serialization::expectIngestorPlanSaveRefusal(captureOf(plan), status, phrase);
}

// ---------------------------------------------------------------------------
// Refusals that need no device
// ---------------------------------------------------------------------------

// Saves its launch inputs under a name no handler is registered under.
class UnregisteredAliasHandler : public packs::StubDispatchHandler
{
public:
    std::optional<SavedLaunchInputs>
        saveLaunchInputs(const PreparedDispatch& /*prepared*/) const override
    {
        return SavedLaunchInputs{"hipkernel.capture_test.unregistered.dispatch.v1", {}};
    }
};

TEST(TestIngestorPlanCapture, RefusesABenchmarkPlanWithoutAWinner)
{
    // Real GenericPlans: a benchmarking plan that answered before a winner exists would
    // hand capture a plan to save, and the refusal would come from a later step.
    const packs::StubDispatchHandler handler;
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
    const packs::StubDispatchHandler handler;
    const auto plan = makeStubPlan(handler, 0x03);

    expectSaveRefusal(*plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "does not support saving");
}

TEST(TestIngestorPlanCapture, RefusesAnAliasThatIsNotRegisteredToTheHandler)
{
    const UnregisteredAliasHandler handler;
    const auto plan = makeStubPlan(handler, 0x04);

    serialization::expectPluginInternalError(captureOf(*plan), "is not registered to this handler");
}

TEST(TestIngestorPlanCapture, RefusesAnEngineNameWithoutTheEngineId)
{
    const packs::StubDispatchHandler handler;
    const auto plan = makeStubPlan(handler, 0x05);

    const std::string message = serialization::expectPluginInternalError(
        captureOf(*plan, "hipkernel:another_engine"), "does not give the id");
    EXPECT_NE(message.find("hipkernel:another_engine"), std::string::npos) << message;
    EXPECT_NE(message.find(hipdnn_data_sdk::utilities::formatEngineIdHex(testEngineId())),
              std::string::npos)
        << message;
}

TEST(TestIngestorPlanCapture, RefusesARestoredPlan)
{
    const packs::StubDispatchHandler handler;
    const RestoredIngestorPlan<Handle> plan(
        handler, std::make_unique<PreparedDispatch>(), 0, "restored test plan");

    expectSaveRefusal(plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "loaded from a saved payload");
    const std::string message
        = expectSaveRefusal(plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "keep the original bytes");

    // The benchmarking advice does not apply to a restored plan.
    EXPECT_EQ(message.find(NO_WINNER_PHRASE), std::string::npos) << message;
}

// ---------------------------------------------------------------------------
// Real kpack plans
// ---------------------------------------------------------------------------

struct CaptureCase
{
    const PackedPlanCase* packedCase;
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
    PackedPlan built;
    ASSERT_NO_FATAL_FAILURE(packs::buildPackedPlan(
        *param.packedCase, param.packedCase->captureGraph(), scratch.path(), built));
    if(built.plan == nullptr)
    {
        GTEST_SKIP() << "nothing was packed for this device (" << built.deviceProperties.gcnArchName
                     << ")";
    }
    const KernelDefinition& kernel = built.kernel;
    const auto& plan = built.plan;

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
    EXPECT_TRUE(hipdnn_plugin_sdk::archMatches(built.deviceProperties.gcnArchName,
                                               payload.target,
                                               hipdnn_plugin_sdk::ArchMatchMode::PREFIX))
        << payload.target << " does not serve " << built.deviceProperties.gcnArchName;
    EXPECT_TRUE(serialization::detail::sameKernelSignature(payload.recordedSignature,
                                                           kernel.source.signature));
    EXPECT_EQ(payload.providerVersion, HIP_KERNEL_PROVIDER_VERSION_STRING);

    const auto encoded = serialization::encodeIngestorPlan(payload);
    EXPECT_TRUE(serialization::decodeIngestorPlan(encoded.data(), encoded.size()) == payload);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestIngestorPlanCaptureKpack,
    ::testing::Values(CaptureCase{&packs::POINTWISE_PACKED_PLAN_CASE,
                                  "hipkernel.pointwise.dispatch.v1",
                                  {{"input_a.uid", int64_t{packs::INPUT_A_UID}},
                                   {"input_b.uid", int64_t{packs::INPUT_B_UID}},
                                   {"output.uid", int64_t{packs::OUTPUT_UID}},
                                   {"block_size", int64_t{256}}},
                                  1024},
                      CaptureCase{&packs::CONV_FWD_PACKED_PLAN_CASE,
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
    [](const ::testing::TestParamInfo<CaptureCase>& info) {
        return std::string(info.param.packedCase->name);
    });

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
        PackedPlan built;
        ASSERT_NO_FATAL_FAILURE(
            packs::buildPackedPlan(packs::CONV_FWD_PACKED_PLAN_CASE,
                                   packs::CONV_FWD_PACKED_PLAN_CASE.captureGraph(),
                                   scratch.path(),
                                   built));
        if(built.plan == nullptr)
        {
            GTEST_SKIP() << "nothing was packed for this device ("
                         << built.deviceProperties.gcnArchName << ")";
        }

        const auto payload = captureIngestorPlan(*built.plan, handle, testEngineId(), ENGINE_NAME);
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
    PackedPlan built;
    ASSERT_NO_FATAL_FAILURE(packs::buildPackedPlan(packs::CONV_FWD_PACKED_PLAN_CASE,
                                                   packs::CONV_FWD_PACKED_PLAN_CASE.captureGraph(),
                                                   scratch.path(),
                                                   built));
    if(built.plan == nullptr)
    {
        GTEST_SKIP() << "nothing was packed for this device (" << built.deviceProperties.gcnArchName
                     << ")";
    }

    const Handle handle;
    const auto buffers = packs::deviceBuffersOf(built.tensors);
    built.plan->execute(handle, buffers.data(), static_cast<uint32_t>(buffers.size()));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    const std::filesystem::path archive = built.archive();
    ASSERT_TRUE(std::filesystem::remove(archive)) << archive;

    expectSaveRefusal(*built.plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "OPEN_ARCHIVE");
    expectSaveRefusal(*built.plan, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, archive.string());

    // The plan keeps its loaded module, so it still executes.
    built.plan->execute(handle, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST(TestIngestorPlanCapture, CapturesTheBenchmarkWinnersOwnWorkspace)
{
    SKIP_IF_NO_DEVICES();

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    DeviceProperties deviceProperties;
    std::filesystem::path packed;
    ASSERT_NO_FATAL_FAILURE(packs::copyPackedArchForDevice(
        packs::POINTWISE_PACKED_PLAN_CASE.root(), scratch.path(), deviceProperties, packed));
    if(packed.empty())
    {
        GTEST_SKIP() << "nothing was packed for this device (" << deviceProperties.gcnArchName
                     << ")";
    }

    KernelDefinition smallKernel;
    KernelDefinition largeKernel;
    ASSERT_NO_FATAL_FAILURE(
        fixtures::readPackedKernelDefinition(packed, packs::POINTWISE_B64_DESCRIPTOR, smallKernel));
    ASSERT_NO_FATAL_FAILURE(fixtures::readPackedKernelDefinition(
        packed, packs::POINTWISE_B256_DESCRIPTOR, largeKernel));
    ASSERT_NE(smallKernel.kernelId, largeKernel.kernelId);

    std::unique_ptr<GenericPlan<Handle>> smallPlan;
    std::unique_ptr<GenericPlan<Handle>> largePlan;
    GraphTensorBundle tensors;
    ASSERT_NO_FATAL_FAILURE(packs::makePackedPlan(packs::POINTWISE_PACKED_PLAN_CASE,
                                                  packs::POINTWISE_PACKED_PLAN_CASE.captureGraph(),
                                                  deviceProperties,
                                                  smallKernel,
                                                  smallPlan,
                                                  tensors));
    ASSERT_NO_FATAL_FAILURE(packs::makePackedPlan(packs::POINTWISE_PACKED_PLAN_CASE,
                                                  packs::POINTWISE_PACKED_PLAN_CASE.captureGraph(),
                                                  deviceProperties,
                                                  largeKernel,
                                                  largePlan,
                                                  tensors));
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

    const auto buffers = packs::deviceBuffersOf(tensors);
    const Workspace<> workspace(plan.getWorkspaceSize(handle));
    ASSERT_NE(workspace.get(), nullptr);
    plan.execute(handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get());
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    const auto payload = captureIngestorPlan(plan, handle, testEngineId(), ENGINE_NAME);
    EXPECT_EQ(payload.kernelDescriptorId, smallKernel.kernelId);
    EXPECT_EQ(payload.workspaceBytes, 0U);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
