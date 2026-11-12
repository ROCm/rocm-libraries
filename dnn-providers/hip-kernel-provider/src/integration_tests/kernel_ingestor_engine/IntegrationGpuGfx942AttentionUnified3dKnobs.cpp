// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_backend.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "IngestorIntegrationSupport.hpp"
#include "ScopedPluginLogCapture.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hip_kernel_provider::test_utilities;

/**
 * @file IntegrationGpuGfx942AttentionUnified3dKnobs.cpp
 * @brief hipkernel:Gfx942AttentionUnified3d end to end through the frontend on a gfx942
 *        device: forcing block_size and num_segments selects exactly the expected shipped
 *        kernel, its segment and reduce launches compute what the CPU reference computes,
 *        and a second execute of the same plan does too.
 *
 * The catalog ships one kernel per GQA ratio (KMD num_queries_per_kv) and the kernels read
 * the head counts at runtime, so forcing (block_size, num_segments) leaves exactly one
 * kernel for a graph's (dtype, head_size, Hq / Hkv) cell. One case uses a head pair no
 * catalog cell lists (Hq 24, Hkv 6) to prove the runtime heads end to end. Every kernel
 * loads K/V with the "stream" cache policy (pinned in the catalog). The expected kernel id
 * is read from the production descriptors, never written here.
 *
 * The cases cover BHSD, BSHD, padded-row and mixed K/V layouts. Poisoned cases fill the
 * workspace with 0xFF before every execute: the workspace is not zeroed, so a segment that
 * leaves a split-KV slot or the seqused_k / cu_seqlens_q tail unwritten shows up as NaN or
 * a wrong row. The second execute also starts from a sentinel-filled O, which only the
 * reduce writes.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

constexpr ServedEngine ENGINE{"hipkernel:Gfx942AttentionUnified3d", "gfx942"};

constexpr const char* BLOCK_SIZE_KNOB = "block_size";
constexpr const char* NUM_SEGMENTS_KNOB = "num_segments";

struct DecodeCase
{
    const char* name;
    DataType dataType;
    int64_t batch;
    int64_t queryHeads;
    int64_t kvHeads;
    int64_t seqKv;
    int64_t headSize;
    SdpaLayout kLayout;
    SdpaLayout vLayout;
    int64_t blockSize;
    int64_t numSegments;
    bool poisonWorkspace;
};

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
void PrintTo(const DecodeCase& decodeCase, std::ostream* os)
{
    *os << decodeCase.name;
}

std::vector<DecodeCase> decodeCases()
{
    constexpr auto FP16 = DataType::HALF;
    constexpr auto BF16 = DataType::BFLOAT16;
    return {
        // 8 MiB of K/V.
        {"Fp16D128Gqa4Bhsd",
         FP16,
         4,
         32,
         8,
         512,
         128,
         SdpaLayout::BHSD,
         SdpaLayout::BHSD,
         16,
         32,
         false},
        // 7.8 MiB; K token-major, V head-major.
        {"Bf16D128MixedKBshdVBhsdPoisonedWorkspace",
         BF16,
         2,
         32,
         8,
         1000,
         128,
         SdpaLayout::BSHD,
         SdpaLayout::BHSD,
         16,
         38,
         true},
        // 288 MiB of K/V, past the 256 MiB Infinity Cache: a large K/V read from HBM.
        {"Bf16D128MixedKBshdVBhsdPoisonedWorkspaceLargeKv",
         BF16,
         9,
         32,
         8,
         8192,
         128,
         SdpaLayout::BSHD,
         SdpaLayout::BHSD,
         16,
         38,
         true},
        // Hq 24 / Hkv 6: a head pair no catalog cell lists. The ratio-4 kernel reads both
        // head counts at runtime, so it serves this pair too.
        {"Bf16D128Hq24Hkv6RuntimeHeadsBhsd",
         BF16,
         3,
         24,
         6,
         777,
         128,
         SdpaLayout::BHSD,
         SdpaLayout::BHSD,
         16,
         38,
         true},
        // MQA, K/V rows padded to D + 8.
        {"Fp16D64MqaPaddedRowPoisonedWorkspace",
         FP16,
         3,
         8,
         1,
         33,
         64,
         SdpaLayout::PADDED_BHSD,
         SdpaLayout::PADDED_BHSD,
         16,
         7,
         true},
        // D256 at block 32, whose KV tile rocKE resolves to 16.
        {"Fp16D256Block32Bshd",
         FP16,
         1,
         8,
         8,
         4096,
         256,
         SdpaLayout::BSHD,
         SdpaLayout::BSHD,
         32,
         16,
         false},
    };
}

/// One single-token SDPA-forward graph: Q/O [B, Hq, 1, D] (BSHD, contiguous at Sq = 1), K/V
/// in the case's layouts, scale 1/sqrt(D) and a bottom-right causal band (which at Sq = 1
/// hides no key).
IngestorSdpaGraphSpec decodeGraphSpec(const DecodeCase& c)
{
    IngestorSdpaGraphSpec spec{c.dataType,
                               c.headSize,
                               c.queryHeads,
                               c.kvHeads,
                               SdpaMask::BOUNDS_BOTTOM_RIGHT,
                               c.batch,
                               /*seqQ=*/1,
                               c.seqKv};
    spec.kLayout = c.kLayout;
    spec.vLayout = c.vLayout;
    return spec;
}

std::vector<KnobSetting> knobSettingsFor(const DecodeCase& c)
{
    std::vector<KnobSetting> settings;
    settings.emplace_back(BLOCK_SIZE_KNOB, c.blockSize);
    settings.emplace_back(NUM_SEGMENTS_KNOB, c.numSegments);
    return settings;
}

/// The id of the one kernel a forced (block_size, num_segments) leaves for @p c's graph cell
/// (dtype, head_size, num_queries_per_kv = Hq / Hkv), or nullopt (with a recorded failure)
/// when that is not exactly one.
std::optional<std::string> expectedKernelId(const DecodeCase& c)
{
    const detail::MetadataKey key{
        {"dtype", detail::catalogDtypeName(c.dataType)},
        {"head_size", c.headSize},
        {"num_queries_per_kv", c.queryHeads / c.kvHeads},
        {"sliding_window", int64_t{0}},
        {"block_size", c.blockSize},
        {"num_segments", c.numSegments},
    };
    return detail::onlyServedKernelCarrying(ENGINE, key);
}

} // namespace

class IntegrationGpuGfx942AttentionUnified3dKnobs : public ServedEngineHarness<DecodeCase>
{
protected:
    IntegrationGpuGfx942AttentionUnified3dKnobs()
        : ServedEngineHarness<DecodeCase>(ENGINE, "gfx942-unified-3d-knobs-case")
    {
    }

    /// 0xFF bytes are NaN in the fp32 workspace regions, so an unwritten slot poisons the row.
    void prepareWorkspace(void* workspace, size_t bytes) override
    {
        if(GetParam().poisonWorkspace && bytes > 0)
        {
            ASSERT_EQ(hipMemset(workspace, 0xFF, bytes), hipSuccess);
        }
    }
};

/// The selected kernel is read from the plugin's own selection line, so a knob dropped
/// anywhere between the frontend and the plugin's filter, or a wrong ratio match, reads as
/// another kernel; the output is then checked against the CPU reference twice through the
/// same plan.
TEST_P(IntegrationGpuGfx942AttentionUnified3dKnobs,
       SelectsTheExpectedKernelAndMatchesTheCpuReference)
{
    const auto& testCase = GetParam();

    const ScopedPluginLogCapture capture(this);
    const detail::SelectedKernelReport report(ENGINE, testCase.name, capture);
    const auto& recorder = capture.recorder();

    auto sdpa = detail::buildSdpaGraph(testCase.name, decodeGraphSpec(testCase));
    ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));

    auto result = sdpa.graph->create_execution_plan_ext(engineId(), knobSettingsFor(testCase));
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = sdpa.graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = sdpa.graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg << "\nCaptured logs:\n"
                                          << recorder.getRecordedLogsAsString();

    const auto selected = detail::selectedKernelId(ENGINE, recorder);
    ASSERT_TRUE(selected.has_value())
        << "no selection line from " << ENGINE.name << ". Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    const auto expectedId = expectedKernelId(testCase);
    ASSERT_TRUE(expectedId.has_value());
    EXPECT_EQ(*selected, *expectedId) << "Captured logs:\n" << recorder.getRecordedLogsAsString();

    // Rank 0 of the forced pair's one candidate: the scorer's own pick, not a fallback past
    // a kernel that failed to load.
    const auto selectionLine
        = std::string(detail::SELECTED_KERNEL_LOG) + *expectedId + " at rank 0 from 1 candidate(s)";
    EXPECT_TRUE(recorder.hasLogContaining(selectionLine))
        << "expected '" << selectionLine << "'. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    GraphVerificationContext context(*sdpa.graph);
    const float tolerance = detail::sdpaForwardTolerance(testCase.dataType);
    registerValidator(context, sdpa.output, tolerance, tolerance);
    VerificationRun run;
    ASSERT_NO_FATAL_FAILURE(prepareVerification(context, /*seed=*/0, run));
    // Twice through the same plan: the second execute starts from a sentinel-filled O and,
    // in poisoned cases, a re-poisoned workspace.
    ASSERT_NO_FATAL_FAILURE(executeAndValidate(run, *sdpa.graph, _handle));
    ASSERT_NO_FATAL_FAILURE(executeAndValidate(run, *sdpa.graph, _handle));
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx942AttentionUnified3dKnobs,
                         ::testing::ValuesIn(decodeCases()),
                         [](const ::testing::TestParamInfo<DecodeCase>& info) {
                             return std::string(info.param.name);
                         });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)
