
#pragma once

#include <rocRoller/AssemblyKernel.hpp>

#include "Base.hpp"

namespace rocRoller
{
    namespace Serialization
    {
        template <typename IO>
        struct MappingTraits<AssemblyKernelArgument, IO, EmptyContext>;

        template <typename IO>
        struct MappingTraits<AssemblyKernel, IO, EmptyContext>;
    }
}
