/**
 * random number generator
 */

#pragma once

#include "Operation.hpp"

#include <memory>
#include <unordered_set>

namespace rocRoller
{
    namespace Operations
    {
        class RandomNumberGenerator : public BaseOperation
        {
        public:
            enum class SeedMode
            {
                Default, //< Use original value
                PerThread, //< Thread (workitem) ID will be added to seed
                Count
            };

            RandomNumberGenerator() = delete;
            explicit RandomNumberGenerator(OperationTag seed);
            RandomNumberGenerator(OperationTag seed, SeedMode mode);

            std::unordered_set<OperationTag> getInputs() const;
            std::string                      toString() const;
            SeedMode                         getSeedMode() const;

            OperationTag seed;
            SeedMode     mode;

            bool operator==(RandomNumberGenerator const&) const;
        };

        std::string   toString(RandomNumberGenerator::SeedMode const&);
        std::ostream& operator<<(std::ostream& stream, RandomNumberGenerator::SeedMode mode);
    }
}
