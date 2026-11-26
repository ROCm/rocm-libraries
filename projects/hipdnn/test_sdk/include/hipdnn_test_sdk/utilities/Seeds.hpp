// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <random>
#include <string>
#include <system_error>

namespace hipdnn_test_sdk::utilities
{

inline unsigned int getGlobalTestSeed()
{
    // Fixing generator seeds to a default value of 1 unless overridden by env var.
    // This ensures consistent test runs unless randomness is explicitly desired.
    // A value that is not RANDOM and not a plain decimal in [0, 4294967295] (sign,
    // whitespace, trailing characters, out of range) also selects the default.
    constexpr unsigned int DEFAULT_SEED = 1;

    const auto seedStr = hipdnn_data_sdk::utilities::getEnv("HIPDNN_GLOBAL_TEST_SEED");

    if(seedStr.empty())
    {
        return DEFAULT_SEED;
    }

    // Convert to uppercase for case-insensitive comparison
    std::string seedStrUpper = seedStr;
    std::transform(seedStrUpper.begin(),
                   seedStrUpper.end(),
                   seedStrUpper.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    // Allow random mode if desired.
    if(seedStrUpper == "RANDOM")
    {
        return std::random_device{}();
    }

    unsigned int seed = 0;
    const char* const first = seedStr.data();
    const char* const last = first + seedStr.size();
    const auto [end, error] = std::from_chars(first, last, seed);
    if(error != std::errc{} || end != last)
    {
        return DEFAULT_SEED;
    }
    return seed;
}

} // namespace hipdnn_test_sdk::utilities
