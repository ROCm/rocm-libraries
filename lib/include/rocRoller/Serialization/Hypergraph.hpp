/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/Serialization/AssemblyKernel.hpp>
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
        struct MappingTraits<Graph::HypergraphIncident, IO, EmptyContext>
        {
            using iot              = IOTraits<IO>;
            static const bool flow = true;

            static void mapping(IO& io, typename Graph::HypergraphIncident& inc)
            {
                iot::mapRequired(io, "src", inc.src);
                iot::mapRequired(io, "dst", inc.dst);
                iot::mapRequired(io, "edgeOrder", inc.edgeOrder);
            }

            static void mapping(IO& io, typename Graph::HypergraphIncident& inc, EmptyContext&)
            {
                mapping(io, inc);
            }
        };

        ROCROLLER_SERIALIZE_VECTOR(false, Graph::HypergraphIncident);

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
                iot::mapRequired(io, "nextIndex", graph.m_nextIndex);

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

                std::vector<typename HG::Incident> incidence;

                if(iot::outputting(io))
                {
                    auto const& container = graph.m_incidence.template get<typename HG::BySrc>();

                    incidence.reserve(container.size());
                    std::copy(
                        container.begin(), container.end(), std::back_insert_iterator(incidence));
                }

                iot::mapRequired(io, "incidence", incidence);

                if(!iot::outputting(io))
                {
                    for(auto& i : incidence)
                        graph.m_incidence.insert(std::move(i));
                }
            }

            static void mapping(IO& io, HG& graph, EmptyContext&)
            {
                mapping(io, graph);
            }
        };
    }
}
