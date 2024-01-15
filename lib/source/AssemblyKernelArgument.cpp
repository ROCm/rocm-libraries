/**
 */

#include <string>

#include <rocRoller/AssemblyKernelArgument.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    bool AssemblyKernelArgument::operator==(AssemblyKernelArgument const& rhs) const
    {
        return name == rhs.name //
               && variableType == rhs.variableType //
               && dataDirection == rhs.dataDirection //
               && equivalent(expression, rhs.expression) //
               && offset == rhs.offset //
               && size == rhs.size;
    }

    std::string AssemblyKernelArgument::toString() const
    {
        auto rv = concatenate("KernelArg{", name, ", ", variableType);

        if(dataDirection != DataDirection::ReadOnly)
            rv += concatenate(", ", dataDirection);

        rv += concatenate(", ", expression);

        if(offset != -1)
            rv += concatenate(", o:", offset);

        if(size != -1)
            rv += concatenate(", s:", size);

        return rv + "}";
    }

    std::ostream& operator<<(std::ostream& stream, AssemblyKernelArgument const& arg)
    {
        return stream << arg.toString();
    }
}
