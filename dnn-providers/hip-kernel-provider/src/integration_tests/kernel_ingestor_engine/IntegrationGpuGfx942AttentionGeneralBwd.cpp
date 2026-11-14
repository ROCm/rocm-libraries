// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <cstdint>
#include <string_view>

#include <gtest/gtest.h>

#include "AttentionGeneralBwdGpuSuite.hpp"

/**
 * @file IntegrationGpuGfx942AttentionGeneralBwd.cpp
 * @brief hipkernel:AttentionGeneralBwd end to end on a gfx942 device
 *        (AttentionGeneralBwdGpuSuite.hpp): one graph per catalog row (every class, every tuned
 *        key, stage_vec 1 and 8, h_k != h_v, batched and THD), named feature graphs (THD GQA and
 *        MQA with per-side multipliers, both ragged stats forms, windows, scales, BSHD), the
 *        dense-stats THD decline, HIP-graph capture and replay with changed lengths and data,
 *        and a misaligned pointer. The quick tier runs a covering subset of the rows; the full
 *        tier the rest.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration::general_bwd_gpu
{
namespace
{

struct Gfx942
{
    static constexpr std::string_view ARCH = "gfx942";
    static constexpr int64_t FULL_CUS = FULL_CUS_GFX942;
};

using IntegrationGpuGfx942AttentionGeneralBwd = AttentionGeneralBwdGpuCases<Gfx942>;
using IntegrationGpuGfx942AttentionGeneralBwdPlans = AttentionGeneralBwdGpu<Gfx942>;

TEST_P(IntegrationGpuGfx942AttentionGeneralBwd, MatchesTheReference)
{
    runCase(GetParam());
}

TEST_F(IntegrationGpuGfx942AttentionGeneralBwdPlans,
       HipGraphCaptureReplaysWithChangedLengthsAndData)
{
    const auto cases = casesTagged(CASES_GFX942, "capture");
    ASSERT_FALSE(cases.empty());
    for(const auto& c : cases)
    {
        captureAndReplay(c);
        if(HasFatalFailure())
        {
            return;
        }
    }
}

TEST_F(IntegrationGpuGfx942AttentionGeneralBwdPlans, MisalignedPointerFailsCleanly)
{
    const auto cases = casesTagged(CASES_GFX942, "misaligned");
    ASSERT_FALSE(cases.empty());
    for(const auto& c : cases)
    {
        misalignedPointerFails(c);
    }
}

TEST_F(IntegrationGpuGfx942AttentionGeneralBwdPlans, DeviceReferenceMatchesHostReference)
{
    expectGpuReferenceMatchesCpu();
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx942AttentionGeneralBwd,
                         ::testing::ValuesIn(casesOfTier(CASES_GFX942, true)),
                         caseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuGfx942AttentionGeneralBwd,
                         ::testing::ValuesIn(casesOfTier(CASES_GFX942, false)),
                         caseName);

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::integration::general_bwd_gpu

#endif // defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)
