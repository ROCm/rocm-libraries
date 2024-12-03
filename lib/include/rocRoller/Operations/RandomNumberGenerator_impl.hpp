#pragma once

#include "RandomNumberGenerator.hpp"

namespace rocRoller
{
    namespace Operations
    {
        inline RandomNumberGenerator::RandomNumberGenerator(OperationTag seed)
            : BaseOperation()
            , seed(seed)
            , mode(SeedMode::Default)
        {
        }

        inline RandomNumberGenerator::RandomNumberGenerator(OperationTag seed, SeedMode mode)
            : BaseOperation()
            , seed(seed)
            , mode(mode)
        {
        }

        inline RandomNumberGenerator::SeedMode RandomNumberGenerator::getSeedMode() const
        {
            return mode;
        }

        inline std::unordered_set<OperationTag> RandomNumberGenerator::getInputs() const
        {
            return {seed};
        }

        inline std::string RandomNumberGenerator::toString() const
        {
            return "RandomNumberGenerator";
        }

        inline std::ostream& operator<<(std::ostream& stream, RandomNumberGenerator::SeedMode mode)
        {
            switch(mode)
            {
            case RandomNumberGenerator::SeedMode::Default:
                return stream << "Default";
            case RandomNumberGenerator::SeedMode::PerThread:
                return stream << "PerThread";
            default:
                Throw<rocRoller::FatalError>("Bad value!");
            }
        }
    }
}
