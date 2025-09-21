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

#include <rocRoller/Utilities/Concepts.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    namespace Graph
    {
        struct HypergraphIncidence
        {
            int src;
            int dst;
        };

        class HypergraphIncidenceContainer
        {
        public:
            /**
             * @brief Add incidences to container. Incidences indicate a connection between elements of a graph.
             * 
             * @tparam T_Inputs Range
             * @tparam T_Outputs Range
             * @param index Graph index of element to be connected by incidences
             * @param inputs Graph indices of input elements to be connected
             * @param outputs Graph indices of output elements to be connected
             */
            template <CForwardRangeOf<int> T_Inputs, CForwardRangeOf<int> T_Outputs>
            void addIncidences(int index, T_Inputs const& inputs, T_Outputs const& outputs);

            /**
             * @brief Gets the number of incidences in the container
             * 
             * @return size_t Number of unique incidences
             */
            size_t size() const;

            /**
             * @brief Purge incidences that refer to this index
             * 
             * @param index Graph index of element to be deleted
             */
            void deleteIndex(int index);

            /**
             * @brief Get all connected sources
             * 
             * @param index Graph index of element to be queried
             * @return Generator<int> Yields a series of graph indices for incoming elements
             */
            Generator<int> getSrcs(int index) const;

            /**
             * @brief Get all connected destinations
             * 
             * @param index Graph index of element to be queried
             * @return Generator<int> Yields a series of graph indices for outgoing elements
             */
            Generator<int> getDsts(int index) const;

            /**
             * @brief Get a count of all connected sources to element at index
             * 
             * @param index Graph index of element to be queried
             * @return size_t Count of all connected sources
             */
            size_t getSrcCount(int index) const;

            /**
             * @brief Get a count of all connected destinations to element at index
             * 
             * @param index Graph index of element to be queried
             * @return size_t Count of all connected destinations
             */
            size_t getDstCount(int index) const;

            /**
             * @brief Get all incidences in container, sorted by sources
             * 
             * @return Generator<HypergraphIncidence> Yields a series of HypergraphIncidence
             */
            Generator<HypergraphIncidence> getAllIncidences() const;

            std::string toDOTSection(std::string const& prefix = "") const;

        private:
            /**
             * @brief A check to ensure internal maps are able to be queried using std::map::at
             * 
             * @param index Graph index that might be accessed by internal maps
             */
            void accountForIndex(int index);

            /**
             * @brief All incidences, sorted by sources, then by edge order.
             * Accessing the map by a graph index will give a map showing all connected destinations, ordered by incidence order.
             * The structure can be read as map<srcIndex, std::map<incidenceOrder, dstIndex>>.
             */
            std::map<int, std::map<int, int>> m_incidencesBySrc;

            /**
             * @brief All incidences, sorted by destinations, then by edge order.
             * Accessing the map by a graph index will give a map showing all connected sources, ordered by incidence order.
             * The structure can be read as map<dstIndex, std::map<incidenceOrder, srcIndex>>.
             */
            std::map<int, std::map<int, int>> m_incidencesByDst;
        };
    }
}

#include <rocRoller/Graph/HypergraphIncidenceContainer_impl.hpp>
