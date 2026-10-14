// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPlanCapture.hpp"
#include "engines/kernel_ingestor_engine/IngestorPlanRestore.hpp"
#include "engines/kernel_ingestor_engine/PackedPlanTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdPackedCase.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanTestUtilities.hpp"

/**
 * @file TestIngestorPlanRoundTrip.cpp
 * @brief Restore refuses code bytes or a symbol that the driver does not accept.
 *
 * Every case reads a copy of the packed set in a scratch directory. Other suites corrupt
 * or delete the staged archives.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;
using serialization::IngestorPlanPayload;

namespace packs = hip_kernel_provider::kernel_ingestor_engine::testing;

using packs::PackedPlan;
using packs::PackedPlanCase;

constexpr const char* SCRATCH_LABEL = "ingestorplanroundtrip";

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

// Restores `encoded` with this provider's device resolver.
std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>> restore(const std::vector<uint8_t>& encoded,
                                                          const Handle& handle)
{
    return restoreIngestorPlan(encoded.data(), encoded.size(), handle);
}

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
    payload = captureIngestorPlan(*built.plan, handle);
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
        [&]() { static_cast<void>(restore(encoded, handle)); },
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

    serialization::expectIngestorPlanRefusal([&]() { static_cast<void>(restore(encoded, handle)); },
                                             HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                                             "hipModuleGetFunction");
    const std::string refusal = serialization::expectIngestorPlanRefusal(
        [&]() { static_cast<void>(restore(encoded, handle)); },
        HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
        "NoSuchKernel");
    consumeRefusedHipError(refusal);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
