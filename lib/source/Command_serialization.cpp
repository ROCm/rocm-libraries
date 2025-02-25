
#include <rocRoller/Operations/Command.hpp>

#include <rocRoller/Serialization/Command.hpp>
#include <rocRoller/Serialization/YAML.hpp>

namespace rocRoller
{
    std::string Command::toYAML(Command const& x)
    {
        return Serialization::toYAML(x);
    }

    Command Command::fromYAML(std::string const& str)
    {
        return Serialization::fromYAML<Command>(str);
    }
}
