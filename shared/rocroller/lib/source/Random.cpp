// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Utilities/Random.hpp>

#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    RandomGenerator::RandomGenerator(int seedNumber)
    {
        std::mt19937::result_type seedValue = static_cast<std::mt19937::result_type>(seedNumber);

        // if set
        int settingsSeed = Settings::getInstance()->get(Settings::RandomSeed);
        if(settingsSeed != Settings::RandomSeed.defaultValue)
        {
            seedValue = static_cast<std::mt19937::result_type>(settingsSeed);
        }
        m_gen.seed(seedValue);
    }

    void RandomGenerator::seed(int seedNumber)
    {
        std::mt19937::result_type seedValue = static_cast<std::mt19937::result_type>(seedNumber);
        m_gen.seed(seedValue);
    }
}
