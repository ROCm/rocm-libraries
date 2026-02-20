// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/Serialization/Base.hpp>
#include <rocRoller/Serialization/Containers.hpp>
#include <rocRoller/Serialization/Enum.hpp>
#include <rocRoller/Serialization/HasTraits.hpp>
#include <rocRoller/Serialization/Variant.hpp>

namespace rocRoller
{
    namespace Serialization
    {
        template <typename IO, typename Context>
        struct MappingTraits<std::monostate, IO, Context>
            : public EmptyMappingTraits<std::monostate, IO, Context>
        {
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::Connections::JustNaryArgument, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::Connections::JustNaryArgument& x, Context& ctx)
            {
                iot::mapRequired(io, "argument", x.argument);
            }

            static void mapping(IO& io, KernelGraph::Connections::JustNaryArgument& x)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, x, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::Connections::TypeAndSubDimension, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void
                mapping(IO& io, KernelGraph::Connections::TypeAndSubDimension& x, Context& ctx)
            {
                iot::mapRequired(io, "id", x.id);
                iot::mapRequired(io, "subdimension", x.subdimension);
            }

            static void mapping(IO& io, KernelGraph::Connections::TypeAndSubDimension& x)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, x, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::Connections::UnrollStride, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::Connections::UnrollStride& x, Context& ctx)
            {
                iot::mapRequired(io, "unrollStride", x.unrollStride);
                iot::mapRequired(io, "unrollDimension", x.unrollDimension);
            }

            static void mapping(IO& io, KernelGraph::Connections::UnrollStride& x)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, x, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::Connections::BaseOffset, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::Connections::BaseOffset& x, Context& ctx)
            {
                iot::mapRequired(io, "base", x.base);
                iot::mapRequired(io, "subdimension", x.subdimension);
            }

            static void mapping(IO& io, KernelGraph::Connections::BaseOffset& x)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, x, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::Connections::TypeAndNaryArgument, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void
                mapping(IO& io, KernelGraph::Connections::TypeAndNaryArgument& x, Context& ctx)
            {
                iot::mapRequired(io, "id", x.id);
                iot::mapRequired(io, "argument", x.argument);
            }

            static void mapping(IO& io, KernelGraph::Connections::TypeAndNaryArgument& x)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, x, ctx);
            }
        };

        static_assert(CNamedVariant<KernelGraph::Connections::ConnectionSpec>);

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::Connections::ConnectionSpec, IO, Context>
            : public DefaultVariantMappingTraits<KernelGraph::Connections::ConnectionSpec,
                                                 IO,
                                                 Context>
        {
            static const bool flow = true;
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlToCoordinateMapper::Connection, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void
                mapping(IO& io, KernelGraph::ControlToCoordinateMapper::Connection& x, Context& ctx)
            {
                iot::mapRequired(io, "control", x.control);
                iot::mapRequired(io, "coordinate", x.coordinate);
                iot::mapRequired(io, "connection", x.connection);
            }

            static void mapping(IO& io, KernelGraph::ControlToCoordinateMapper::Connection& x)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, x, ctx);
            }
        };

        ROCROLLER_SERIALIZE_VECTOR(false, KernelGraph::ControlToCoordinateMapper::Connection);

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlToCoordinateMapper, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::ControlToCoordinateMapper& x, Context& ctx)
            {
                std::vector<KernelGraph::ControlToCoordinateMapper::Connection> connections;

                if(iot::outputting(io))
                    connections = x.getConnections();

                iot::mapRequired(io, "connections", connections);

                if(!iot::outputting(io))
                {
                    for(auto const& c : connections)
                        x.connect(c);
                }
            }

            static void mapping(IO& io, KernelGraph::ControlToCoordinateMapper& x)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, x, ctx);
            }
        };
    }
}
