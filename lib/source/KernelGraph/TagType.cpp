#include <string>

#include <rocRoller/KernelGraph/TagType.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        std::string TagType::toString() const
        {
            return "{" + std::to_string(ctag) + ", " + std::to_string(dim) + ", "
                   + (output ? "o" : "i") + "}";
        }

        std::ostream& operator<<(std::ostream& stream, TagType const& x)
        {
            return stream << x.toString();
        }
    }
}
