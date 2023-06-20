#pragma once

#ifdef ROCROLLER_USE_LLVM
#include <llvm/ObjectYAML/YAML.h>
#endif

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

#include "Base.hpp"
#include "Containers.hpp"

#include "ControlGraph.hpp"
#include "ControlToCoordinateMapper.hpp"
#include "CoordinateGraph.hpp"

namespace rocRoller
{
    namespace Serialization
    {

        template <typename IO>
        struct MappingTraits<KernelGraph::KernelGraph, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::KernelGraph& kern)
            {
                iot::mapRequired(io, "control", kern.control);
                iot::mapRequired(io, "coordinates", kern.coordinates);
                iot::mapRequired(io, "mapper", kern.mapper);
            }

            static void mapping(IO& io, KernelGraph::KernelGraph& kern, EmptyContext& ctx)
            {
                mapping(io, kern);
            }
        };

    }
}
