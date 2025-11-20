// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdlib>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <random>
#include <string>

namespace hipdnn_sdk::test_utilities
{

inline unsigned int getGlobalTestSeed()
{
    // Fixing generator seeds to a default value of 1 unless overridden by env var.
    // This ensures consistent test runs unless randomness is explicitly desired.
    constexpr unsigned int DEFAULT_SEED = 1;

    auto seedStr = hipdnn_sdk::utilities::getEnv("HIPDNN_GLOBAL_TEST_SEED");

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

    try
    {
        auto seed = std::stoul(seedStr);
        return static_cast<unsigned int>(seed);
    }
    catch(...)
    {
        return DEFAULT_SEED;
    }
}

} // namespace hipdnn_sdk::test_utilities
