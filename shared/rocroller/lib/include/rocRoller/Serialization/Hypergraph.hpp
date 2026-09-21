// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/Serialization/Base.hpp>
#include <rocRoller/Serialization/Containers.hpp>
#include <rocRoller/Serialization/Expression.hpp>
#include <rocRoller/Serialization/HasTraits.hpp>
#include <rocRoller/Serialization/Variant.hpp>

namespace rocRoller
{
    namespace Serialization
    {
        template <typename IO>
        struct MappingTraits<Graph::HypergraphIncidenceContainer, IO, EmptyContext>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, Graph::HypergraphIncidenceContainer& incidence)
            {
                iot::mapRequired(io, "incidenceBySrc", incidence.m_incidenceBySrc);
                iot::mapRequired(io, "incidenceByDst", incidence.m_incidenceByDst);
            }

            static void
                mapping(IO& io, Graph::HypergraphIncidenceContainer& incidence, EmptyContext&)
            {
                mapping(io, incidence);
            }
        };

        template <CNamedVariant Var>
        struct ElementEntry
        {
            int id;
            Var value;
        };

        template <CNamedVariant Var, typename IO, typename Context>
        struct MappingTraits<ElementEntry<Var>, IO, Context>
        {
            using iot   = IOTraits<IO>;
            using Entry = ElementEntry<Var>;

            static void mapping(IO& io, Entry& entry, Context ctx)
            {
                iot::mapRequired(io, "id", entry.id);

                MappingTraits<Var, IO, Context>::mapping(io, entry.value, ctx);
            }

            static void mapping(IO& io, Entry& entry)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, entry, ctx);
            }
        };

        template <CNamedVariant Var, typename IO>
        struct SequenceTraits<std::vector<ElementEntry<Var>>, IO>
            : public DefaultSequenceTraits<std::vector<ElementEntry<Var>>, IO, false>
        {
        };

        // template <CNamedVariant Value, typename IO>
        // struct CustomMappingTraits<std::map<int, Value>, IO>
        //     : public DefaultCustomMappingTraits<std::map<int, Value>, IO, false, false>
        // {
        // };

        template <typename Node, typename Edge, bool Hyper, typename IO>
        struct MappingTraits<Graph::Hypergraph<Node, Edge, Hyper>, IO, EmptyContext>
        {
            using iot     = IOTraits<IO>;
            using HG      = Graph::Hypergraph<Node, Edge, Hyper>;
            using Element = typename HG::Element;

            static void mapping(IO& io, HG& graph)
            {
                std::vector<ElementEntry<Element>> elements;
                if(iot::outputting(io))
                {
                    elements.reserve(graph.m_elements.size());
                    for(auto const& pair : graph.m_elements)
                        elements.push_back({pair.first, pair.second});
                }

                iot::mapRequired(io, "elements", elements);

                if(!iot::outputting(io))
                {
                    for(auto const& entry : elements)
                        graph.m_elements[entry.id] = entry.value;
                }

                iot::mapRequired(io, "incidence", graph.m_incidence);
            }

            static void mapping(IO& io, HG& graph, EmptyContext&)
            {
                mapping(io, graph);
            }
        };
    }
}
