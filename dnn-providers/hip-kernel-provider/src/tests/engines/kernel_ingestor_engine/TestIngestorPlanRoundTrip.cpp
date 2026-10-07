// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/IngestorPlanCapture.hpp"
#include "engines/kernel_ingestor_engine/IngestorPlanRestore.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/PackedPlanTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdPackedCase.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwisePackedCase.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanTestUtilities.hpp"

/**
 * @file TestIngestorPlanRoundTrip.cpp
 * @brief A kpack plan is captured, encoded and restored. The original plan computes the
 *        exact expected output. The restored plan computes the same bytes on a new handle
 *        after the original plan is destroyed. Restore needs no archive, loads the code
 *        object on the handle's device, and refuses bytes or a symbol the driver does not
 *        accept.
 *
 * Every case reads a copy of the packed set in a scratch directory. Other suites corrupt
 * or delete the staged archives.
 *
 * The unit binary does not load the engine of the packed Pointwise set, so restore runs
 * with an engine lookup that knows only the engine of the plan under test.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hipdnn_data_sdk::utilities::Workspace;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::GraphTensorBundle;
using hipdnn_test_sdk::utilities::ScopedDirectory;
using serialization::IngestorPlanPayload;

namespace packs = hip_kernel_provider::kernel_ingestor_engine::testing;

using packs::PackedPlan;
using packs::PackedPlanCase;

constexpr const char* SCRATCH_LABEL = "ingestorplanroundtrip";

int64_t engineIdOf(const PackedPlanCase& packedCase)
{
    return hipdnn_data_sdk::utilities::engineNameToId(std::string(packedCase.engineName));
}

// Executes `plan` once on `handle` over `tensors` and writes the output bytes to `output`.
// The output holds the sentinel before the launch, so an element that the launch does not
// write cannot match.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
void runPlan(const hipdnn_plugin_sdk::IPlan<Handle>& plan,
             const Handle& handle,
             GraphTensorBundle& tensors,
             std::vector<uint8_t>& output)
{
    ASSERT_EQ(tensors.outputTensorIds.size(), 1U);
    tensors.sentinelFillOutputTensors();

    const auto buffers = packs::deviceBuffersOf(tensors);
    const Workspace<> workspace(plan.getWorkspaceSize(handle));
    plan.execute(handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get());
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess) << "the launch failed";

    auto& result = tensors.getTensor(*tensors.outputTensorIds.begin());
    result.markDeviceModified();
    const auto* bytes = static_cast<const uint8_t*>(result.rawHostData());
    output.assign(bytes, bytes + (result.elementSpace() * result.elementSize()));
}

// Builds a plan over the registered handler of `packedCase` from a copy of its packed set
// in `scratch`, on its round-trip graph, with the inputs filled. `built.plan` stays null
// when nothing was packed for device 0.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
void buildFilledPlan(const PackedPlanCase& packedCase,
                     const ScopedDirectory& scratch,
                     PackedPlan& built)
{
    ASSERT_NO_FATAL_FAILURE(
        packs::buildPackedPlan(packedCase, packedCase.roundTripGraph(), scratch.path(), built));
    if(built.plan != nullptr)
    {
        packedCase.fillInputs(built.tensors);
    }
}

// Restores `encoded` with this provider's device resolver and an engine lookup that knows
// only the engine of `packedCase`.
std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>> restore(const std::vector<uint8_t>& encoded,
                                                          const Handle& handle,
                                                          const PackedPlanCase& packedCase)
{
    IngestorPlanRestoreEnvironment environment;
    environment.loadedIngestorEngineName
        = [&packedCase](int64_t candidate) -> std::optional<std::string> {
        if(candidate == engineIdOf(packedCase))
        {
            return std::string(packedCase.engineName);
        }
        return std::nullopt;
    };
    environment.deviceResolver = &deviceResolver();
    return restoreIngestorPlan(encoded.data(), encoded.size(), handle, environment);
}

// ---------------------------------------------------------------------------
// Round trips, for each packed set
// ---------------------------------------------------------------------------

class TestIngestorPlanRoundTripKpack : public ::testing::TestWithParam<const PackedPlanCase*>
{
};

TEST_P(TestIngestorPlanRoundTripKpack, RestoredPlanComputesWhatTheOriginalComputes)
{
    SKIP_IF_NO_DEVICES();
    const PackedPlanCase& packedCase = *GetParam();

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    PackedPlan built;
    ASSERT_NO_FATAL_FAILURE(buildFilledPlan(packedCase, scratch, built));
    if(built.plan == nullptr)
    {
        GTEST_SKIP() << "nothing was packed for this device (" << built.deviceProperties.gcnArchName
                     << ")";
    }

    const Handle originalHandle;
    std::vector<uint8_t> original;
    ASSERT_NO_FATAL_FAILURE(runPlan(*built.plan, originalHandle, built.tensors, original));
    packedCase.expectExactOutput(original);

    const size_t originalWorkspaceBytes = built.plan->getWorkspaceSize(originalHandle);
    const auto encoded = serialization::encodeIngestorPlan(captureIngestorPlan(
        *built.plan, originalHandle, engineIdOf(packedCase), std::string(packedCase.engineName)));

    // Destroy the original plan and drop the cached modules, so that no code object of the
    // original plan stays loaded.
    built.plan.reset();
    resetIngestorModuleCachesForTesting();

    const Handle restoreHandle;
    const auto restored = restore(encoded, restoreHandle, packedCase);
    ASSERT_NE(restored, nullptr);

    EXPECT_EQ(restored->getWorkspaceSize(restoreHandle), originalWorkspaceBytes);
    for(int run = 0; run < 2; ++run)
    {
        std::vector<uint8_t> output;
        ASSERT_NO_FATAL_FAILURE(runPlan(*restored, restoreHandle, built.tensors, output));
        EXPECT_EQ(output, original) << "run " << run;
    }
}

TEST_P(TestIngestorPlanRoundTripKpack, RestoresWithoutTheArchive)
{
    SKIP_IF_NO_DEVICES();
    const PackedPlanCase& packedCase = *GetParam();

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    PackedPlan built;
    ASSERT_NO_FATAL_FAILURE(buildFilledPlan(packedCase, scratch, built));
    if(built.plan == nullptr)
    {
        GTEST_SKIP() << "nothing was packed for this device (" << built.deviceProperties.gcnArchName
                     << ")";
    }

    const Handle handle;
    std::vector<uint8_t> original;
    ASSERT_NO_FATAL_FAILURE(runPlan(*built.plan, handle, built.tensors, original));
    packedCase.expectExactOutput(original);

    const auto encoded = serialization::encodeIngestorPlan(captureIngestorPlan(
        *built.plan, handle, engineIdOf(packedCase), std::string(packedCase.engineName)));

    const auto archive = built.archive();
    ASSERT_TRUE(std::filesystem::remove(archive)) << archive;
    resetIngestorModuleCachesForTesting();
    const size_t cachedBefore = packs::packModuleCacheEntries();

    const auto restored = restore(encoded, handle, packedCase);
    ASSERT_NE(restored, nullptr);
    std::vector<uint8_t> output;
    ASSERT_NO_FATAL_FAILURE(runPlan(*restored, handle, built.tensors, output));
    EXPECT_EQ(output, original);
    EXPECT_EQ(packs::packModuleCacheEntries(), cachedBefore);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestIngestorPlanRoundTripKpack,
                         ::testing::Values(&packs::POINTWISE_PACKED_PLAN_CASE,
                                           &packs::CONV_FWD_PACKED_PLAN_CASE),
                         [](const ::testing::TestParamInfo<const PackedPlanCase*>& info) {
                             return std::string(info.param->name);
                         });

// ---------------------------------------------------------------------------
// Refusals at the module load
// ---------------------------------------------------------------------------

// A captured ConvFwd payload, built from a copy of the packed set. `payload` is left
// default when nothing was packed for the device; `built.plan` is then null.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
void captureConvFwd(const ScopedDirectory& scratch, PackedPlan& built, IngestorPlanPayload& payload)
{
    const PackedPlanCase& packedCase = packs::CONV_FWD_PACKED_PLAN_CASE;
    ASSERT_NO_FATAL_FAILURE(buildFilledPlan(packedCase, scratch, built));
    if(built.plan == nullptr)
    {
        return;
    }
    const Handle handle;
    payload = captureIngestorPlan(
        *built.plan, handle, engineIdOf(packedCase), std::string(packedCase.engineName));
}

// Reads and clears both HIP error values after a refused restore. A value that HIP sets
// must be the error that the refusal names.
void consumeRefusedHipError(const std::string& refusal)
{
    const hipError_t last = hipGetLastError();
    const hipError_t command = hipExtGetLastError();
    EXPECT_TRUE(last == hipSuccess || refusal.find(hipGetErrorString(last)) != std::string::npos)
        << "hipGetLastError gave " << last << ", the refusal is: " << refusal;
    EXPECT_TRUE(command == hipSuccess || command == last
                || refusal.find(hipGetErrorString(command)) != std::string::npos)
        << "hipExtGetLastError gave " << command << ", the refusal is: " << refusal;
}

TEST(TestIngestorPlanRoundTrip, RefusesBytesTheDriverRejectsAtRestore)
{
    SKIP_IF_NO_DEVICES();

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    PackedPlan built;
    IngestorPlanPayload payload;
    ASSERT_NO_FATAL_FAILURE(captureConvFwd(scratch, built, payload));
    if(built.plan == nullptr)
    {
        GTEST_SKIP() << "nothing was packed for this device (" << built.deviceProperties.gcnArchName
                     << ")";
    }

    payload.codeObject = std::vector<uint8_t>(256, 0xAB);
    const auto encoded = serialization::encodeIngestorPlan(payload);
    const Handle handle;

    const std::string refusal = serialization::expectIngestorPlanRefusal(
        [&]() { static_cast<void>(restore(encoded, handle, packs::CONV_FWD_PACKED_PLAN_CASE)); },
        HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
        "hipModuleLoadData");
    // HIP records the refused module load as the last error, and the restore leaves it there.
    EXPECT_NE(hipPeekAtLastError(), hipSuccess);
    consumeRefusedHipError(refusal);
}

TEST(TestIngestorPlanRoundTrip, RefusesASymbolTheCodeObjectLacks)
{
    SKIP_IF_NO_DEVICES();

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    PackedPlan built;
    IngestorPlanPayload payload;
    ASSERT_NO_FATAL_FAILURE(captureConvFwd(scratch, built, payload));
    if(built.plan == nullptr)
    {
        GTEST_SKIP() << "nothing was packed for this device (" << built.deviceProperties.gcnArchName
                     << ")";
    }

    // The signature check passes: it does not read the symbol.
    payload.symbol = "NoSuchKernel";
    const auto encoded = serialization::encodeIngestorPlan(payload);
    const Handle handle;

    serialization::expectIngestorPlanRefusal(
        [&]() { static_cast<void>(restore(encoded, handle, packs::CONV_FWD_PACKED_PLAN_CASE)); },
        HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
        "hipModuleGetFunction");
    const std::string refusal = serialization::expectIngestorPlanRefusal(
        [&]() { static_cast<void>(restore(encoded, handle, packs::CONV_FWD_PACKED_PLAN_CASE)); },
        HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
        "NoSuchKernel");
    consumeRefusedHipError(refusal);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
