#pragma once

#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Serialization/Base.hpp>
#include <rocRoller/Serialization/Containers.hpp>

namespace rocRoller
{
    namespace Serialization
    {

        template <typename IO>
        struct CustomMappingTraits<std::map<int, int>, IO>
            : public DefaultCustomMappingTraits<std::map<int, int>, IO, false, true>
        {
        };

        template <typename IO>
        struct MappingTraits<KernelGraph::UnrollColouring, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::UnrollColouring& colouring)
            {
                iot::mapRequired(io, "separators", colouring.separators);
                // iot::mapRequired(io, "coordinates", kern.coordinates);
                // iot::mapRequired(io, "mapper", kern.mapper);
            }

            static void mapping(IO& io, KernelGraph::KernelGraph& kern, EmptyContext& ctx)
            {
                mapping(io, kern);
            }
        };
    }
}
