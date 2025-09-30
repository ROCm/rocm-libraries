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

#include <iostream>
#include <numeric>
#include <ranges>

#include <rocRoller/Graph/HypergraphIncidenceContainer.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::Graph
{
    size_t HypergraphIncidenceContainer::size() const
    {
        return std::accumulate(
            m_incidenceBySrc.begin(), m_incidenceBySrc.end(), 0, [](size_t sum, auto const& value) {
                return sum + value.second.size();
            });
    }

    void HypergraphIncidenceContainer::deleteTag(int tag)
    {
        m_incidenceBySrc.erase(tag);
        m_incidenceByDst.erase(tag);

        for(auto& connections : m_incidenceBySrc)
            std::erase(connections.second, tag);

        for(auto& connections : m_incidenceByDst)
            std::erase(connections.second, tag);
    }

    std::vector<int> HypergraphIncidenceContainer::getSrcs(int tag) const
    {
        auto it = m_incidenceByDst.find(tag);
        if(it != m_incidenceByDst.end())
            return it->second;
        return {};
    }

    std::vector<int> HypergraphIncidenceContainer::getDsts(int tag) const
    {
        auto it = m_incidenceBySrc.find(tag);
        if(it != m_incidenceBySrc.end())
            return it->second;
        return {};
    }

    size_t HypergraphIncidenceContainer::getSrcCount(int tag) const
    {
        auto it = m_incidenceByDst.find(tag);
        if(it != m_incidenceByDst.end())
            return it->second.size();
        return 0;
    }

    size_t HypergraphIncidenceContainer::getDstCount(int tag) const
    {
        auto it = m_incidenceBySrc.find(tag);
        if(it != m_incidenceBySrc.end())
            return it->second.size();
        return 0;
    }

    std::string HypergraphIncidenceContainer::toDOTSection(std::string const& prefix) const
    {
        std::ostringstream s;

        for(auto const& connections : m_incidenceBySrc)
        {
            int src = connections.first;
            for(int dst : connections.second)
            {
                s << '"' << prefix << src << "\" -> \"" << prefix << dst << '"' << std::endl;
            }
        }

        return s.str();
    }
}
