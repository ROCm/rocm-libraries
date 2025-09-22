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

#include <rocRoller/Graph/HypergraphIncidenceContainer.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::Graph
{
    size_t HypergraphIncidenceContainer::size() const
    {
        return std::accumulate(
            m_incidencesBySrc.begin(),
            m_incidencesBySrc.end(),
            0,
            [](size_t sum, auto const& value) { return sum + value.second.size(); });
    }

    void HypergraphIncidenceContainer::deleteId(int id)
    {
        m_incidencesBySrc.erase(id);
        m_incidencesByDst.erase(id);

        for(auto& connections : m_incidencesBySrc)
            std::erase(connections.second, id);

        for(auto& connections : m_incidencesByDst)
            std::erase(connections.second, id);
    }

    std::vector<int> HypergraphIncidenceContainer::getSrcs(int id) const
    {
        if(!m_incidencesByDst.contains(id))
            return {};
        return m_incidencesByDst.at(id);
    }

    std::vector<int> HypergraphIncidenceContainer::getDsts(int id) const
    {
        if(!m_incidencesBySrc.contains(id))
            return {};
        return m_incidencesBySrc.at(id);
    }

    size_t HypergraphIncidenceContainer::getSrcCount(int id) const
    {
        if(m_incidencesByDst.contains(id))
            return m_incidencesByDst.at(id).size();
        return 0;
    }

    size_t HypergraphIncidenceContainer::getDstCount(int id) const
    {
        if(m_incidencesBySrc.contains(id))
            return m_incidencesBySrc.at(id).size();
        return 0;
    }

    Generator<HypergraphIncidence> HypergraphIncidenceContainer::getAllIncidences() const
    {
        for(auto const& connections : m_incidencesBySrc)
        {
            for(int dst : connections.second)
            {
                HypergraphIncidence incidence{.src = connections.first, .dst = dst};
                co_yield incidence;
            }
        }
    }

    void HypergraphIncidenceContainer::accountForId(int id)
    {
        if(!m_incidencesBySrc.contains(id))
            m_incidencesBySrc.emplace(id, std::vector<int>{});
        if(!m_incidencesByDst.contains(id))
            m_incidencesByDst.emplace(id, std::vector<int>{});
    }

    std::string HypergraphIncidenceContainer::toDOTSection(std::string const& prefix) const
    {
        std::ostringstream s;
        for(auto const& incidence : getAllIncidences())
        {
            s << '"' << prefix << incidence.src << "\" -> \"" << prefix << incidence.dst << '"'
              << std::endl;
        }
        return s.str();
    }
}
