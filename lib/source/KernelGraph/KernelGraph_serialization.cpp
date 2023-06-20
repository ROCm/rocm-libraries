

#include <rocRoller/KernelGraph/KernelGraph.hpp>

#include <rocRoller/Serialization/KernelGraph.hpp>
#include <rocRoller/Serialization/YAML.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        std::string toYAML(KernelGraph const& g)
        {
            return Serialization::toYAML(g);
        }

        KernelGraph fromYAML(std::string const& str)
        {
            return Serialization::fromYAML<KernelGraph>(str);
        }
    }
}
