/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/Operations/RandomNumberGenerator.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    namespace Operations
    {
        RandomNumberGenerator::RandomNumberGenerator(OperationTag seed)
            : BaseOperation()
            , seed(seed)
            , mode(SeedMode::Default)
        {
        }

        RandomNumberGenerator::RandomNumberGenerator(OperationTag seed, SeedMode mode)
            : BaseOperation()
            , seed(seed)
            , mode(mode)
        {
        }

        RandomNumberGenerator::SeedMode RandomNumberGenerator::getSeedMode() const
        {
            return mode;
        }

        std::unordered_set<OperationTag> RandomNumberGenerator::getInputs() const
        {
            return {seed};
        }

        std::string RandomNumberGenerator::toString() const
        {
            return "RandomNumberGenerator";
        }

        std::ostream& operator<<(std::ostream& stream, RandomNumberGenerator::SeedMode mode)
        {
            return stream << toString(mode);
        }

        std::string toString(RandomNumberGenerator::SeedMode const& mode)
        {
            switch(mode)
            {
            case RandomNumberGenerator::SeedMode::Default:
                return "Default";
            case RandomNumberGenerator::SeedMode::PerThread:
                return "PerThread";
            default:
                Throw<rocRoller::FatalError>("Bad value!");
            }
        }

        bool RandomNumberGenerator::operator==(RandomNumberGenerator const& other) const
        {
            return other.m_tag == m_tag && other.seed == seed && other.mode == mode;
        }
    }
}
