
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
