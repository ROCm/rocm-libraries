// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "integration_tests/kernel_ingestor_engine/RankingTraceSupport.hpp"

/**
 * @file TestRankingTraceSupport.cpp
 * @brief The cold-case ranker check the GPU knobs suites apply, driven on the CPU with
 *        synthetic `uhd trace` lines in the plugin's own format: a declared-order fallback
 *        fails the check, a model decision and an un-pulled ranker's fallback pass it, and
 *        traces of the native score or of another engine are not read as the engine's.
 */
namespace
{

using hip_kernel_provider::kernel_ingestor_engine::integration::detail::expectRankerDecided;
using hip_kernel_provider::kernel_ingestor_engine::integration::detail::rankingDecidedBy;
using hipdnn_test_sdk::utilities::RecordedLog;

constexpr const char* ENGINE = "hipkernel:Gfx942AttentionDense";

std::vector<RecordedLog> logs(const std::vector<std::string>& messages)
{
    std::vector<RecordedLog> recorded;
    recorded.reserve(messages.size());
    for(const auto& message : messages)
    {
        recorded.push_back({HIPDNN_SEV_INFO, message});
    }
    return recorded;
}

const std::string FALLBACK
    = "uhd trace: engine 'hipkernel:Gfx942AttentionDense' (a74a5afb-30c0-41c2-8006-501d2961f808)"
      " decided_by=declared_order reason=ranking_failed metric=tflops candidates=10";
const std::string MODEL
    = "uhd trace: engine 'hipkernel:Gfx942AttentionDense' (a74a5afb-30c0-41c2-8006-501d2961f808)"
      " decided_by=model metric=tflops winner=2ce2d1b4-6770-4c94-8d3a-d6947f469725";

TEST(TestRankingTraceSupport, ADeclaredOrderFallbackFailsTheCheck)
{
    const auto decidedBy = rankingDecidedBy(ENGINE, logs({FALLBACK}));
    ASSERT_EQ(decidedBy, std::optional<std::string>("declared_order"));
    EXPECT_NONFATAL_FAILURE(expectRankerDecided(decidedBy, false, ""),
                            "fell back to declared kernel order");
}

TEST(TestRankingTraceSupport, AModelDecisionPassesTheCheck)
{
    const auto decidedBy = rankingDecidedBy(ENGINE, logs({MODEL}));
    ASSERT_EQ(decidedBy, std::optional<std::string>("model"));
    expectRankerDecided(decidedBy, false, "");
}

TEST(TestRankingTraceSupport, AnUnpulledRankersFallbackPassesTheCheck)
{
    const auto decidedBy = rankingDecidedBy(ENGINE, logs({FALLBACK}));
    expectRankerDecided(decidedBy, true, "");
}

TEST(TestRankingTraceSupport, NativeAndOtherEngineTracesAreNotTheEngines)
{
    EXPECT_EQ(
        rankingDecidedBy(ENGINE,
                         logs({"uhd trace: decided_by=native metric=tflops candidates=10",
                               "uhd trace: engine 'hipkernel:Gfx942AttentionUnified3d' "
                               "(4208dcf6) decided_by=declared_order reason=ranking_failed"})),
        std::nullopt);
}

} // namespace
