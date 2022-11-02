
#include <rocRoller/Utilities/Random.hpp>

namespace rocRoller
{
    RandomGenerator::RandomGenerator(int seedNumber)
    {
        std::mt19937::result_type seed = static_cast<std::mt19937::result_type>(seedNumber);

        // if set
        int settingsSeed = Settings::getInstance()->get(Settings::RandomSeed);
        if(settingsSeed != Settings::RandomSeed.defaultValue)
        {
            seed = static_cast<std::mt19937::result_type>(settingsSeed);
        }
        m_gen.seed(seed);
    }

    void RandomGenerator::seed(int seedNumber)
    {
        std::mt19937::result_type seed = static_cast<std::mt19937::result_type>(seedNumber);
        m_gen.seed(seed);
    }
}
