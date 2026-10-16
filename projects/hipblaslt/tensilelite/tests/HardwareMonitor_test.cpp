// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <limits>

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
