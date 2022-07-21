
#pragma once

#include <rocRoller/GPUArchitecture/GPUArchitecture_fwd.hpp>

#include "Base.hpp"

namespace rocRoller
{
    namespace Serialization
    {
        template <typename IO>
        struct MappingTraits<GPUArchitecture, IO, EmptyContext>;
    }
}
