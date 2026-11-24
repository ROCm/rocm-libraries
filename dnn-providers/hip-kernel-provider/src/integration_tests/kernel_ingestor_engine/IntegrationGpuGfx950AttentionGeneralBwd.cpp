// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <cstdint>
#include <string_view>

#include <gtest/gtest.h>

#include "AttentionGeneralBwdGpuSuite.hpp"

/**
 * @file IntegrationGpuGfx950AttentionGeneralBwd.cpp
 * @brief hipkernel:AttentionGeneralBwd end to end on a gfx950 device
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

struct Gfx950
{
    static constexpr std::string_view ARCH = "gfx950";
    static constexpr int64_t FULL_CUS = FULL_CUS_GFX950;
};

using IntegrationGpuGfx950AttentionGeneralBwd = AttentionGeneralBwdGpuCases<Gfx950>;
using IntegrationGpuGfx950AttentionGeneralBwdPlans = AttentionGeneralBwdGpu<Gfx950>;

TEST_P(IntegrationGpuGfx950AttentionGeneralBwd, MatchesTheReference)
{
    runCase(GetParam());
}

TEST_F(IntegrationGpuGfx950AttentionGeneralBwdPlans,
       HipGraphCaptureReplaysWithChangedLengthsAndData)
{
    const auto cases = casesTagged(CASES_GFX950, "capture");
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

TEST_F(IntegrationGpuGfx950AttentionGeneralBwdPlans, MisalignedPointerFailsCleanly)
{
    const auto cases = casesTagged(CASES_GFX950, "misaligned");
    ASSERT_FALSE(cases.empty());
    for(const auto& c : cases)
    {
        misalignedPointerFails(c);
    }
}

TEST_F(IntegrationGpuGfx950AttentionGeneralBwdPlans, DeviceReferenceMatchesHostReference)
{
    expectGpuReferenceMatchesCpu();
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx950AttentionGeneralBwd,
                         ::testing::ValuesIn(casesOfTier(CASES_GFX950, true)),
                         caseName);

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuGfx950AttentionGeneralBwd,
                         ::testing::ValuesIn(casesOfTier(CASES_GFX950, false)),
                         caseName);

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::integration::general_bwd_gpu

#endif // defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)
