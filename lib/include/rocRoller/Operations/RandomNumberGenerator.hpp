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
            };

            RandomNumberGenerator() = delete;
            RandomNumberGenerator(OperationTag seed);
            RandomNumberGenerator(OperationTag seed, SeedMode mode);

            std::unordered_set<OperationTag> getInputs() const;
            std::string                      toString() const;
            SeedMode                         getSeedMode() const;

            OperationTag seed;
            SeedMode     mode;
        };
    }
}

#include "RandomNumberGenerator_impl.hpp"
