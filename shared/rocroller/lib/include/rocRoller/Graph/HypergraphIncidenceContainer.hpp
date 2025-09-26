/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <map>
#include <vector>

#include <rocRoller/Serialization/Base_fwd.hpp>
#include <rocRoller/Utilities/Concepts.hpp>

namespace rocRoller
{
    namespace Graph
    {
        class HypergraphIncidenceContainer
        {
        public:
            /**
             * @brief Add incident connections to container. These represent connection between
             * elements of a graph.
             *
             * @tparam T_Inputs Range of graph tags
             * @tparam T_Outputs Range of graph tags
             * @param tag Graph tag of element to be connected
             * @param inputs Graph tags of input elements to be connected upstream of `tag`
             * @param outputs Graph indices of output elements to be connected downstream of `tag`
             */
            template <CForwardRangeOf<int> T_Inputs, CForwardRangeOf<int> T_Outputs>
            void addIncidentConnections(int tag, T_Inputs const& inputs, T_Outputs const& outputs);

            /**
             * @brief Gets the number of incident connections in the container
             *
             * @return size_t Number of unique connections
             */
            size_t size() const;

            /**
             * @brief Purge incident connections that refer to this tag
             *
             * @param tag Graph tag of element to be deleted
             */
            void deleteTag(int tag);

            /**
             * @brief Get all connected sources
             *
             * @param tag Graph tag of element to be queried
             * @return std::vector<int> a series of graph indices for incoming elements
             */
            std::vector<int> getSrcs(int tag) const;

            /**
             * @brief Get all connected destinations
             *
             * @param tag Graph tag of element to be queried
             * @return std::vector<int> a series of graph indices for outgoing elements
             */
            std::vector<int> getDsts(int tag) const;

            /**
             * @brief Get a count of all connected sources to element at tag
             *
             * @param tag Graph tag of element to be queried
             * @return size_t Count of all connected sources
             */
            size_t getSrcCount(int tag) const;

            /**
             * @brief Get a count of all connected destinations to element at tag
             *
             * @param tag Graph tag of element to be queried
             * @return size_t Count of all connected destinations
             */
            size_t getDstCount(int tag) const;

            std::string toDOTSection(std::string const& prefix = "") const;

        private:
            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;

            /**
             * @brief The incidence structure of the Hypergraph, sorted by sources, then by
             * connection order.
             *
             * NOTE: This structure does not preserve the order of sources connected to a
             * destination element. This storage structure is duplicated to maintain that ordering
             * and for quick lookup.
             */
            std::map<int, std::vector<int>> m_incidenceBySrc;

            /**
             * @brief The incidence structure of the Hypergraph, sorted by destinations, then by
             * connection order.
             *
             * NOTE: This structure does not preserve the order of destinations connected to a
             * source element. This storage structure is duplicated to maintain that ordering and
             * for quick lookup.
             */
            std::map<int, std::vector<int>> m_incidenceByDst;
        };
    }
}

#include <rocRoller/Graph/HypergraphIncidenceContainer_impl.hpp>
