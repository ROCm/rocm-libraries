// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <vector>

#include <HardwareMonitor.hpp>

using namespace TensileLite::Client;

// ===========================================================================
// getValidatedFrequency -- guards amdsmi_frequencies_t::frequency[current]
// against an out-of-range `current` (see GitHub issue #10716: a power-gated
// clock domain can report current == (uint32_t)-1 with AMDSMI_STATUS_SUCCESS).
// ===========================================================================

TEST(GetValidatedFrequencyTest, ValidIndexReturnsFrequency)
{
    amdsmi_frequencies_t freq{};
    freq.num_supported = 3;
    freq.current       = 1;
    freq.frequency[0]  = 100;
    freq.frequency[1]  = 200;
    freq.frequency[2]  = 300;

    EXPECT_EQ(getValidatedFrequency(freq), 200u);
}

TEST(GetValidatedFrequencyTest, PowerGatedSentinelReturnsMax)
{
    amdsmi_frequencies_t freq{};
    freq.num_supported = 8;
    freq.current       = static_cast<uint32_t>(-1); // power-gated / no reading
    freq.frequency[0]  = 100;

    EXPECT_EQ(getValidatedFrequency(freq), std::numeric_limits<uint64_t>::max());
}

TEST(GetValidatedFrequencyTest, CurrentAtOrPastNumSupportedReturnsMax)
{
    amdsmi_frequencies_t freq{};
    freq.num_supported = 3;
    freq.current       = 3; // one past the last valid index
    freq.frequency[0]  = 100;
    freq.frequency[1]  = 200;
    freq.frequency[2]  = 300;

    EXPECT_EQ(getValidatedFrequency(freq), std::numeric_limits<uint64_t>::max());
}

TEST(GetValidatedFrequencyTest, CurrentPastArrayBoundReturnsMax)
{
    amdsmi_frequencies_t freq{};
    freq.num_supported = AMDSMI_MAX_NUM_FREQUENCIES + 5; // malformed, but shouldn't matter
    freq.current       = AMDSMI_MAX_NUM_FREQUENCIES;     // one past the backing array

    EXPECT_EQ(getValidatedFrequency(freq), std::numeric_limits<uint64_t>::max());
}

TEST(GetValidatedFrequencyTest, ZeroInitializedStructReturnsMax)
{
    // amdsmi_frequencies_t freq{}; num_supported == 0, current == 0 -> no valid index.
    amdsmi_frequencies_t freq{};

    EXPECT_EQ(getValidatedFrequency(freq), std::numeric_limits<uint64_t>::max());
}

#if AMDSMI_LIB_VERSION_MAJOR >= 25

// ===========================================================================
// getValidatedGfxClocks -- guards gpu_metrics' per-XCD current_gfxclks[]
// against the UINT16_MAX "not populated" sentinel (see GitHub issue #10716:
// single-XCD parts don't fill it, and using it verbatim reports 65535 MHz).
// ===========================================================================

namespace
{
    constexpr uint16_t kNotPopulated = std::numeric_limits<uint16_t>::max();

    amdsmi_gpu_metrics_t makeMetrics(std::initializer_list<uint16_t> gfxclksMhz)
    {
        amdsmi_gpu_metrics_t metrics{};
        std::fill(
            std::begin(metrics.current_gfxclks), std::end(metrics.current_gfxclks), kNotPopulated);

        uint16_t xcd = 0;
        for(uint16_t mhz : gfxclksMhz)
        {
            metrics.current_gfxclks[xcd++] = mhz;
        }
        return metrics;
    }
} // namespace

TEST(GetValidatedGfxClocksTest, SingleXCDPopulated)
{
    auto                  metrics = makeMetrics({1500});
    std::vector<uint64_t> perXCDHz;

    EXPECT_TRUE(getValidatedGfxClocks(metrics, 1, perXCDHz));
    EXPECT_EQ(perXCDHz, (std::vector<uint64_t>{1500000000}));
}

TEST(GetValidatedGfxClocksTest, MultiXCDPopulated)
{
    auto                  metrics = makeMetrics({1000, 1100, 1200, 1300});
    std::vector<uint64_t> perXCDHz;

    EXPECT_TRUE(getValidatedGfxClocks(metrics, 4, perXCDHz));
    EXPECT_EQ(perXCDHz, (std::vector<uint64_t>{1000000000, 1100000000, 1200000000, 1300000000}));
}

TEST(GetValidatedGfxClocksTest, NotPopulatedReportsUnavailable)
{
    // The gfx1151 case: gpu_metrics reports success but every entry is UINT16_MAX.
    amdsmi_gpu_metrics_t  metrics = makeMetrics({});
    std::vector<uint64_t> perXCDHz;

    EXPECT_FALSE(getValidatedGfxClocks(metrics, 1, perXCDHz));
    EXPECT_TRUE(perXCDHz.empty());
}

TEST(GetValidatedGfxClocksTest, PartiallyPopulatedReportsUnavailable)
{
    // A partial reading would skew the per-XCD average, so reject it outright.
    auto                  metrics = makeMetrics({1000});
    std::vector<uint64_t> perXCDHz;

    EXPECT_FALSE(getValidatedGfxClocks(metrics, 2, perXCDHz));
    EXPECT_TRUE(perXCDHz.empty());
}

TEST(GetValidatedGfxClocksTest, EntriesPastXcdCountAreIgnored)
{
    auto                  metrics = makeMetrics({1000, 1100});
    std::vector<uint64_t> perXCDHz;

    EXPECT_TRUE(getValidatedGfxClocks(metrics, 2, perXCDHz));
    EXPECT_EQ(perXCDHz, (std::vector<uint64_t>{1000000000, 1100000000}));
}

TEST(GetValidatedGfxClocksTest, ZeroXcdCountReportsUnavailable)
{
    auto                  metrics = makeMetrics({1000});
    std::vector<uint64_t> perXCDHz;

    EXPECT_FALSE(getValidatedGfxClocks(metrics, 0, perXCDHz));
}

TEST(GetValidatedGfxClocksTest, XcdCountPastArrayBoundReportsUnavailable)
{
    auto                  metrics = makeMetrics({1000});
    std::vector<uint64_t> perXCDHz;

    EXPECT_FALSE(getValidatedGfxClocks(metrics, AMDSMI_MAX_NUM_GFX_CLKS + 1, perXCDHz));
}

#endif
